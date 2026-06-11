#!/usr/bin/env python3
"""
统一模型训练脚本 — 双层架构
=============================
日模型 + 分钟模型 分开训练，分开保存。

日模型 (model_daily.pkl):
  - 数据: 日线 K 线
  - 特征: 日级别技术指标 + MarketFeatureEngineer (北向、大盘、资金流)
  - 目标: 次日收益率
  - 用途: 每天盘前选出当日值得关注的股票池
  - 保存: models/lgb_hs300/model_daily.pkl

分钟模型 (model.pkl):
  - 数据: 30分钟 K 线
  - 特征: 纯30分钟技术指标（无日级别数据泄漏）
  - 目标: 未来90分钟收益率
  - 用途: 盘中每30分钟对全市场排序，选出Top N买入
  - 保存: models/lgb_hs300/model.pkl

版本管理:
  - model.pkl / model_daily.pkl = 最新 → agent 始终引用
  - model_YYYYMMDD_HHMMSS.pkl = 时间戳备份
  - model_meta.json = 可读元数据
  - versions.yaml = 所有历史版本对比

用法:
  # 分钟模型 (常用)
  python strategy/train_model.py --start-date 2020-01-01 --trials 50

  # 日模型
  python strategy/train_model.py --model-type daily --trials 30

  # 快速测试
  python strategy/train_model.py --start-date 2025-01-01 --trials 10

  # 低内存服务器
  python strategy/train_model.py --start-date 2024-01-01 --trials 15 --memmap
"""

import os
import sys
import pickle
import json
import gc
import yaml
import numpy as np
import pandas as pd
import sqlite3
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel
from typing import Dict, List, Tuple, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lgb_enhanced import EnhancedFeatureEngineer

# ====== 常量 ======
DB_PATH = os.path.join(os.path.dirname(__file__), '../data/stock_data.db')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models/lgb_hs300')
VERSIONS_FILE = os.path.join(MODEL_DIR, 'versions.yaml')

DEFAULT_PARAMS = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
    'n_estimators': 2000,
}


# ============================================================
#  数据加载
# ============================================================

def load_30min_data(db_path: str) -> Dict[str, pd.DataFrame]:
    """加载所有股票30分钟K线"""
    conn = sqlite3.connect(db_path)
    symbols = [r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM kline_30m ORDER BY symbol").fetchall()]
    print(f"30分钟数据: {len(symbols)} 只股票")

    all_data = {}
    for i, sym in enumerate(symbols):
        df = pd.read_sql_query(
            'SELECT date, open, high, low, close, volume '
            'FROM kline_30m WHERE symbol=? ORDER BY date',
            conn, params=(sym,))
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            all_data[sym] = df
        if (i + 1) % 50 == 0:
            print(f"  已加载 {i + 1}/{len(symbols)}")
    conn.close()
    print(f"成功加载 {len(all_data)} 只\n")
    return all_data


def load_daily_data(db_path: str) -> pd.DataFrame:
    """加载日线数据 (优先 kline_daily 表, 回退到30分钟聚合)"""
    conn = sqlite3.connect(db_path)

    # 检查表
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    if 'kline_daily' in tables:
        # kline_daily 列名是 date (不是 trade_date)
        df = pd.read_sql_query(
            "SELECT symbol, date as trade_date, open, high, low, close, volume "
            "FROM kline_daily ORDER BY symbol, date", conn)
        print(f"日线(kline_daily): {len(df):,} 行, {df['symbol'].nunique()} 只")
    else:
        # 从30分钟聚合
        print("无 kline_daily 表, 从30分钟K线聚合日线...")
        df = pd.read_sql_query("""
            SELECT symbol, date(date) as trade_date,
                   FIRST_VALUE(open) as open, MAX(high) as high,
                   MIN(low) as low, LAST_VALUE(close) as close,
                   SUM(volume) as volume
            FROM kline_30m GROUP BY symbol, date(date)
            ORDER BY symbol, trade_date
        """, conn)
        print(f"日线(聚合): {len(df):,} 行, {df['symbol'].nunique() if 'symbol' in df.columns else '?'} 只")

    conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df


# ============================================================
#  分钟模型: 特征 + 训练
# ============================================================

class MarketFeatureEngineerDaily:
    """日级别特征工程: 北向资金 + 大盘 + 资金流"""

    @staticmethod
    def calculate_features(df: pd.DataFrame, db_path: str) -> pd.DataFrame:
        """计算日级别特征"""
        features = pd.DataFrame(index=df.index)
        features['trade_date'] = df['trade_date']

        # 基础技术指标
        close = df['close'].values
        features['daily_return'] = pd.Series(close).pct_change().values
        features['daily_volatility_5'] = pd.Series(close).pct_change().rolling(5).std().values
        features['daily_volatility_20'] = pd.Series(close).pct_change().rolling(20).std().values
        features['ma5_ratio'] = close / pd.Series(close).rolling(5).mean().values
        features['ma10_ratio'] = close / pd.Series(close).rolling(10).mean().values
        features['ma20_ratio'] = close / pd.Series(close).rolling(20).mean().values
        features['ma60_ratio'] = close / pd.Series(close).rolling(60).mean().values
        features['volume_ratio_5'] = df['volume'] / df['volume'].rolling(5).mean()

        # 振幅
        features['amplitude'] = (df['high'] - df['low']) / df['close']

        # 大盘数据
        try:
            conn = sqlite3.connect(db_path)
            hs300 = pd.read_sql_query(
                "SELECT trade_date, close, pct_chg, volume "
                "FROM hs300_daily ORDER BY trade_date", conn)

            if not hs300.empty:
                hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
                hs300 = hs300.set_index('trade_date')

                features['market_return'] = features['trade_date'].map(
                    hs300['pct_chg'].to_dict()) / 100.0
                features['market_ma5'] = features['trade_date'].map(
                    hs300['close'].rolling(5).mean().to_dict())
                features['market_ma20'] = features['trade_date'].map(
                    hs300['close'].rolling(20).mean().to_dict())
                features['market_trend'] = np.where(
                    features['market_ma5'].values > features['market_ma20'].values, 1, -1)

            conn.close()
        except Exception:
            pass

        # 北向资金
        try:
            conn = sqlite3.connect(db_path)
            north = pd.read_sql_query(
                "SELECT trade_date, net_flow, cumulative_flow "
                "FROM north_flow ORDER BY trade_date", conn)

            if not north.empty:
                north['trade_date'] = pd.to_datetime(north['trade_date'])
                features['north_flow'] = features['trade_date'].map(
                    north.set_index('trade_date')['net_flow'].to_dict())
                features['north_flow_5d'] = features['north_flow'].rolling(5).mean()
                features['north_cumulative'] = features['trade_date'].map(
                    north.set_index('trade_date')['cumulative_flow'].to_dict())

            conn.close()
        except Exception:
            pass

        return features.fillna(0)


def prepare_30min_features(all_data, horizon=3, start_date='2020-01-01', use_memmap=False):
    """准备30分钟模型训练数据"""
    if use_memmap:
        return _prepare_30min_memmap(all_data, horizon, start_date)
    else:
        return _prepare_30min_inmem(all_data, horizon, start_date)


def _prepare_30min_inmem(all_data, horizon, start_date):
    """内存充足时的标准模式"""
    all_features, all_targets, all_symbols = [], [], []
    feature_names = None
    success, failed = 0, 0

    print("计算30分钟特征...")
    for i, (symbol, df) in enumerate(all_data.items()):
        try:
            df = df[df['date'] >= pd.to_datetime(start_date)].reset_index(drop=True)
            if len(df) < 150:
                failed += 1; continue

            features = EnhancedFeatureEngineer.calculate_features(df)
            if feature_names is None:
                feature_names = features.columns.tolist()
                print(f"  特征数: {len(feature_names)}")

            close = df['close'].values
            target = np.zeros(len(close))
            n = len(close) - horizon
            target[:n] = (close[horizon:] - close[:n]) / close[:n]
            target[-horizon:] = np.nan

            features = features.ffill().fillna(0)
            valid = ~np.isnan(target)
            fv = features[valid].iloc[120:]
            tv = target[valid][120:]

            if len(fv) > 50:
                all_features.append(fv.values.astype(np.float32))
                all_targets.append(tv.astype(np.float32))
                all_symbols.extend([symbol] * len(fv))
                success += 1
            else:
                failed += 1
        except Exception:
            failed += 1

        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/{len(all_data)} (成功{success}, 失败{failed})")

    if not all_features:
        return None, None, None, None

    X = np.vstack(all_features)
    y = np.concatenate(all_targets)
    return _filter_extremes(X, y, all_symbols, feature_names)


def _prepare_30min_memmap(all_data, horizon, start_date):
    """低内存模式"""
    feature_names = None
    success, failed = 0, 0
    total_rows = 0
    stock_rows = {}

    print("Pass 1: 统计样本...")
    for i, (symbol, df) in enumerate(all_data.items()):
        df = df[df['date'] >= pd.to_datetime(start_date)]
        if len(df) < 150: continue
        try:
            features = EnhancedFeatureEngineer.calculate_features(df)
            if feature_names is None: feature_names = features.columns.tolist()
            close = df['close'].values
            target = np.zeros(len(close))
            n = len(close) - horizon
            target[:n] = (close[horizon:] - close[:n]) / close[:n]
            features = features.ffill().fillna(0)
            valid = ~np.isnan(target)
            fv = features[valid].iloc[120:]
            if len(fv) > 50:
                stock_rows[symbol] = len(fv)
                total_rows += len(fv)
                success += 1
        except Exception:
            failed += 1
        if (i + 1) % 100 == 0:
            print(f"  扫描 {i + 1}/{len(all_data)} ({total_rows:,} 行)")

    if total_rows == 0: return None, None, None, None

    n_feat = len(feature_names)
    fp_x, fp_y = '/tmp/train_X.dat', '/tmp/train_y.dat'
    print(f"Pass 2: 写入 memmap ({total_rows:,} × {n_feat})")

    X_mmap = np.memmap(fp_x, dtype=np.float32, mode='w+', shape=(total_rows, n_feat))
    y_mmap = np.memmap(fp_y, dtype=np.float32, mode='w+', shape=(total_rows,))
    symbols_list = []
    offset = 0

    for i, (symbol, df) in enumerate(all_data.items()):
        if symbol not in stock_rows: continue
        n = stock_rows[symbol]
        try:
            df = df[df['date'] >= pd.to_datetime(start_date)]
            features = EnhancedFeatureEngineer.calculate_features(df)
            close = df['close'].values
            target = np.zeros(len(close))
            m = len(close) - horizon
            target[:m] = (close[horizon:] - close[:m]) / close[:m]
            features = features.ffill().fillna(0)
            valid = ~np.isnan(target)
            fv = features[valid].iloc[120:]
            tv = target[valid][120:]
            X_mmap[offset:offset + n] = fv.values.astype(np.float32)
            y_mmap[offset:offset + n] = tv.astype(np.float32)
            symbols_list.extend([symbol] * n)
            offset += n
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            print(f"  写入 {i + 1}/{len(all_data)} ({offset/total_rows*100:.0f}%)")

    X_mmap.flush(); y_mmap.flush()
    del X_mmap, y_mmap; gc.collect()

    X = np.array(np.memmap(fp_x, dtype=np.float32, mode='r', shape=(total_rows, n_feat)))
    y = np.array(np.memmap(fp_y, dtype=np.float32, mode='r', shape=(total_rows,)))
    for f in [fp_x, fp_y]:
        if os.path.exists(f): os.remove(f)
    gc.collect()
    return _filter_extremes(X, y, symbols_list, feature_names)


# ============================================================
#  日模型: 特征 + 训练
# ============================================================

def prepare_daily_features(df: pd.DataFrame, start_date='2020-01-01'):
    """准备日线模型训练数据"""
    print("计算日级别特征...")

    # 过滤日期
    df = df[df['trade_date'] >= pd.to_datetime(start_date)].copy()

    all_features, all_targets, all_symbols = [], [], []
    feature_names = None
    success, failed = 0, 0

    symbols = df['symbol'].unique()
    for i, sym in enumerate(symbols):
        try:
            sym_df = df[df['symbol'] == sym].sort_values('trade_date').reset_index(drop=True)
            if len(sym_df) < 200:
                failed += 1; continue

            features = MarketFeatureEngineerDaily.calculate_features(sym_df, DB_PATH)
            if feature_names is None:
                # 提取纯数字特征列
                numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [c for c in numeric_cols if c not in ('trade_date',)]
                feature_names = numeric_cols
                print(f"  特征数: {len(feature_names)}")

            # 目标: 次日收益率
            close = sym_df['close'].values
            target = np.zeros(len(close))
            target[:-1] = (close[1:] - close[:-1]) / close[:-1]
            target[-1] = np.nan

            fv = features[feature_names].fillna(0).values
            valid = ~np.isnan(target)
            fv = fv[valid][60:]  # 跳过前60天（MA60不完整）
            tv = target[valid][60:]

            if len(fv) > 50:
                all_features.append(fv.astype(np.float32))
                all_targets.append(tv.astype(np.float32))
                all_symbols.extend([sym] * len(fv))
                success += 1
            else:
                failed += 1
        except Exception:
            failed += 1

        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/{len(symbols)} (成功{success}, 失败{failed})")

    if not all_features:
        return None, None, None, None

    X = np.vstack(all_features).astype(np.float32)
    y = np.concatenate(all_targets).astype(np.float32)

    # 过滤极端值（日线允许更大波动）
    valid = np.abs(y) < 0.15
    X, y = X[valid], y[valid]

    print(f"\n日模型数据: {len(X):,} 样本 × {X.shape[1]} 特征")
    print(f"目标: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")
    return X, y, feature_names, feature_names  # (X, y, _, feature_names)


# ============================================================
#  通用: 特征选择 + 训练 + 验证 + 保存
# ============================================================

def _filter_extremes(X, y, symbols, feature_names):
    """过滤极端收益率"""
    valid = np.abs(y) < 0.1
    X = X[valid].astype(np.float32)
    y = y[valid].astype(np.float32)
    symbols = [s for s, v in zip(symbols, valid) if v]
    print(f"\n数据: {len(X):,} 样本 × {X.shape[1]} 特征")
    print(f"目标: mean={y.mean():.4f}, std={y.std():.4f}, "
          f"min={y.min():.4f}, max={y.max():.4f}")
    return X, y, symbols, feature_names


def select_features(X, y, feature_names):
    """特征选择: 相关性 + SelectFromModel"""
    n_orig = len(feature_names)
    corr_threshold = 0.95
    corr = pd.DataFrame(X, columns=feature_names).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        high_corr = upper.index[upper[col] > corr_threshold].tolist()
        for hc in high_corr:
            ci, hi = feature_names.index(col), feature_names.index(hc)
            to_drop.add(hc if X[:, ci].var() > X[:, hi].var() else col)

    if to_drop:
        keep = [i for i, f in enumerate(feature_names) if f not in to_drop]
        X, feature_names = X[:, keep], [feature_names[i] for i in keep]
        print(f"  相关性过滤: {n_orig} → {len(feature_names)} (移除{len(to_drop)})")
    else:
        print(f"  相关性过滤: 无需移除")

    selector = SelectFromModel(
        lgb.LGBMRegressor(n_estimators=100, objective='regression_l1', verbose=-1,
                          random_state=42, n_jobs=-1),
        threshold='median'
    )
    selector.fit(X, y)
    mask = selector.get_support()
    X = X[:, mask]
    feature_names = [feature_names[i] for i, m in enumerate(mask) if m]
    print(f"  SelectFromModel: → {len(feature_names)} 特征")
    return X, feature_names


def search_hyperparams(X, y, n_trials=50):
    """Optuna 超参数搜索"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    split = int(len(X) * 0.8)

    def objective(trial):
        params = dict(DEFAULT_PARAMS, **{
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 5.0),
        })

        scores = []
        for tr, te in TimeSeriesSplit(n_splits=3, gap=1).split(X[:split]):
            m = lgb.LGBMRegressor(**params)
            m.fit(X[:split][tr], y[:split][tr],
                  eval_set=[(X[:split][te], y[:split][te])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            pred = m.predict(X[:split][te])
            if len(pred) > 2:
                c, _ = spearmanr(pred, y[:split][te])
                scores.append(c if not np.isnan(c) else 0)
        return np.mean(scores) if scores else 0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\n✅ 最优 Spearman: {study.best_value:.4f}")
    return study.best_params, study.best_value


def cross_validate(X, y, params, feature_names, model_type='30min', n_splits=5):
    """Purged 时间序列交叉验证"""
    gap = 3 if model_type == '30min' else 1
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    scores, maes, fi = [], [], {}

    for fold, (tr, te) in enumerate(tscv.split(X)):
        model = lgb.LGBMRegressor(**params)
        model.fit(X[tr], y[tr],
                  eval_set=[(X[te], y[te])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        pred = model.predict(X[te])
        c, _ = spearmanr(pred, y[te])
        mae = np.mean(np.abs(pred - y[te]))
        scores.append(c if not np.isnan(c) else 0)
        maes.append(mae)

        for f, imp in zip(feature_names, model.feature_importances_):
            fi[f] = fi.get(f, 0) + imp

        print(f"  Fold {fold + 1}: Spearman={c:.4f}, MAE={mae:.4f}")

    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    return np.mean(scores), np.std(scores), np.mean(maes), fi_sorted


def save_model(model_data, model_dir, model_name='model'):
    """保存模型 + 版本管理"""
    os.makedirs(model_dir, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_path = os.path.join(model_dir, f'{model_name}.pkl')
    backup_path = os.path.join(model_dir, f'{model_name}_{ts}.pkl')
    meta_path = os.path.join(model_dir, f'{model_name}_meta.json')

    for path in [main_path, backup_path]:
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    # 元数据
    meta = {}
    for k, v in model_data.items():
        if k == 'model':
            meta['model_type'] = type(v).__name__
        elif k == 'feature_importance':
            meta['feature_importance_top20'] = dict(list(v.items())[:20])
        elif isinstance(v, (str, int, float, bool, type(None))):
            meta[k] = v
        elif isinstance(v, list) and len(str(v)) < 200:
            meta[k] = v
    meta['saved_at'] = ts
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    # 版本历史
    versions = []
    if os.path.exists(VERSIONS_FILE):
        with open(VERSIONS_FILE) as f:
            versions = yaml.safe_load(f) or []

    versions.append({
        'version': ts,
        'model': model_name,
        'spearman': float(meta.get('cv_spearman', 0)),
        'n_features': meta.get('n_features_selected', 0),
        'n_samples': meta.get('n_samples', 0),
        'start_date': meta.get('start_date', ''),
        'trained_at': meta.get('trained_at', ''),
    })
    versions = sorted(versions, key=lambda x: x['version'])[-50:]
    with open(VERSIONS_FILE, 'w') as f:
        yaml.dump(versions, f, default_flow_style=False, allow_unicode=True)

    print(f"\n模型已保存:")
    print(f"  最新: {main_path}")
    print(f"  备份: {backup_path}")
    print(f"  元数据: {meta_path}")
    print(f"  版本: {VERSIONS_FILE} ({len(versions)} 条)")


# ============================================================
#  主入口
# ============================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description='统一模型训练 (日模型 + 分钟模型)')
    p.add_argument('--model-type', default='30min', choices=['30min', 'daily'],
                   help='模型类型: 30min=盘中择时, daily=日级别选股')
    p.add_argument('--start-date', default=None,
                   help='数据起始日期 (默认: 30min=2020-01-01, daily=2020-01-01)')
    p.add_argument('--trials', type=int, default=50, help='Optuna 搜索轮数')
    p.add_argument('--memmap', action='store_true', help='低内存模式 (<2GB RAM)')
    p.add_argument('--no-tune', action='store_true', help='跳过超参数搜索')
    args = p.parse_args()

    model_type = args.model_type

    # 默认起始日期
    start_date = args.start_date or '2020-01-01'

    model_name = 'model_daily' if model_type == 'daily' else 'model'

    print("=" * 60)
    print(f"Stock Quant 模型训练 — {'日模型 (日线选股)' if model_type == 'daily' else '分钟模型 (盘中择时)'}")
    print("=" * 60)
    print(f"数据起始: {start_date}")
    print(f"搜索轮数: {args.trials}")
    print(f"低内存: {'是' if args.memmap else '否'}")
    print(f"保存为: {model_name}.pkl")
    print("=" * 60)

    if model_type == 'daily':
        # ====== 日模型训练 ======
        df = load_daily_data(DB_PATH)
        if df is None or df.empty:
            print("无日线数据"); return

        X, y, _, feature_names = prepare_daily_features(df, start_date)
        del df; gc.collect()

        if X is None or len(X) < 500:
            print(f"训练数据不足"); return

        print("\n特征选择...")
        X, feature_names = select_features(X, y, feature_names)

        if args.no_tune:
            best_params = DEFAULT_PARAMS
            best_tune = 0
        else:
            print(f"\n🔍 Optuna ({args.trials}轮)...")
            best_params, best_tune = search_hyperparams(X, y, args.trials)

        print("\n5折 Purged CV...")
        cv_score, cv_std, cv_mae, importance = cross_validate(
            X, y, best_params, feature_names, model_type='daily')

        final = lgb.LGBMRegressor(**best_params)
        final.fit(X, y)

        model_data = {
            'model': final,
            'feature_names': feature_names,
            'feature_importance': importance,
            'cv_spearman': cv_score, 'cv_spearman_std': cv_std, 'cv_mae': cv_mae,
            'tune_spearman': best_tune, 'best_params': best_params,
            'n_samples': len(X), 'n_features_selected': len(feature_names),
            'prediction_horizon': 1,  # 日模型预测1天
            'model_type': 'daily',
            'start_date': start_date,
            'trained_at': datetime.now().isoformat(),
        }
        save_model(model_data, MODEL_DIR, model_name='model_daily')

    else:
        # ====== 分钟模型训练 ======
        all_data = load_30min_data(DB_PATH)
        if not all_data:
            print("无数据"); return

        X, y, _, feature_names = prepare_30min_features(
            all_data, start_date=start_date, use_memmap=args.memmap)
        del all_data; gc.collect()

        if X is None or len(X) < 500:
            print(f"训练数据不足"); return

        print("\n特征选择...")
        X, feature_names = select_features(X, y, feature_names)

        if args.no_tune:
            best_params = DEFAULT_PARAMS
            best_tune = 0
        else:
            print(f"\n🔍 Optuna ({args.trials}轮)...")
            best_params, best_tune = search_hyperparams(X, y, args.trials)

        print("\n5折 Purged CV...")
        cv_score, cv_std, cv_mae, importance = cross_validate(
            X, y, best_params, feature_names, model_type='30min')

        final = lgb.LGBMRegressor(**best_params)
        final.fit(X, y)

        model_data = {
            'model': final,
            'feature_names': feature_names,
            'feature_importance': importance,
            'cv_spearman': cv_score, 'cv_spearman_std': cv_std, 'cv_mae': cv_mae,
            'tune_spearman': best_tune, 'best_params': best_params,
            'n_samples': len(X), 'n_features_selected': len(feature_names),
            'prediction_horizon': 3,  # 分钟模型预测3根K线(90分钟)
            'model_type': '30min_regression',
            'start_date': start_date,
            'trained_at': datetime.now().isoformat(),
        }
        save_model(model_data, MODEL_DIR, model_name='model')

    print("\n" + "=" * 60)
    print(f"✅ 训练完成!")
    print(f"模型: {model_name}.pkl")
    print(f"Spearman: {model_data['cv_spearman']:.4f} ± {model_data['cv_spearman_std']:.4f}")
    print(f"MAE: {model_data['cv_mae']:.4f}")
    print(f"特征: {model_data['n_features_selected']} 个")
    print(f"样本: {model_data['n_samples']:,} 条")
    print(f"Spearman 0.05-0.15=可盈利, 0.15+=优秀")
    print("=" * 60)


if __name__ == '__main__':
    main()