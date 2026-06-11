#!/usr/bin/env python3
"""
统一模型训练脚本
=================
不再区分 v7/v8/v9。版本管理通过:
- 时间戳备份: model_YYYYMMDD_HHMMSS.pkl
- 版本记录: versions.yaml (记录每次训练的配置和指标)
- 最新模型: model.pkl (agent 始终引用这个)

用法:
  python strategy/train_model.py --start-date 2024-01-01 --trials 50
  python strategy/train_model.py --start-date 2025-01-01 --trials 20  # 快速测试

Mac 上完整训练 (24GB):
  python strategy/train_model.py --start-date 2024-01-01 --trials 50

低内存服务器:
  python strategy/train_model.py --start-date 2025-07-01 --trials 20 --memmap
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

# 默认超参数
DEFAULT_PARAMS = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
    'n_estimators': 2000,
}


def load_data(db_path: str) -> Dict[str, pd.DataFrame]:
    """加载所有股票30分钟K线"""
    conn = sqlite3.connect(db_path)
    symbols = [r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM kline_30m ORDER BY symbol").fetchall()]
    print(f"数据库中共有 {len(symbols)} 只股票")

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
            print(f"  已加载 {i + 1}/{len(symbols)} 只股票")
    conn.close()
    print(f"成功加载 {len(all_data)} 只股票\n")
    return all_data


def prepare_features(
    all_data: Dict[str, pd.DataFrame],
    horizon: int = 3,
    start_date: str = '2024-01-01',
    use_memmap: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]], Optional[List[str]]]:
    """准备训练特征和回归目标
    
    Args:
        use_memmap: 低内存模式, 通过磁盘文件避免 list+vstack 双倍内存
    """
    feature_names = None
    success = 0
    failed = 0

    if use_memmap:
        return _prepare_features_memmap(all_data, horizon, start_date)
    else:
        return _prepare_features_inmem(all_data, horizon, start_date)


def _prepare_features_inmem(all_data, horizon, start_date):
    """内存充足时的标准模式"""
    all_features, all_targets, all_symbols = [], [], []
    feature_names = None
    success, failed = 0, 0

    print("计算特征...")
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
            target[:len(close) - horizon] = (
                close[horizon:] - close[:len(close) - horizon]
            ) / close[:len(close) - horizon]
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


def _prepare_features_memmap(all_data, horizon, start_date):
    """低内存模式: 2-pass memmap"""
    feature_names = None
    success, failed = 0, 0

    # Pass 1: count rows
    print("Pass 1: 统计样本数...")
    total_rows = 0
    stock_rows = {}
    for i, (symbol, df) in enumerate(all_data.items()):
        df = df[df['date'] >= pd.to_datetime(start_date)]
        if len(df) < 150: continue
        try:
            features = EnhancedFeatureEngineer.calculate_features(df)
            if feature_names is None:
                feature_names = features.columns.tolist()
            close = df['close'].values
            target = np.zeros(len(close))
            target[:len(close) - horizon] = (
                close[horizon:] - close[:len(close) - horizon]
            ) / close[:len(close) - horizon]
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
            print(f"  扫描 {i + 1}/{len(all_data)} (已统计 {total_rows:,} 行)")

    if total_rows == 0:
        return None, None, None, None

    n_feat = len(feature_names)
    fp_x, fp_y = '/tmp/train_X.dat', '/tmp/train_y.dat'
    print(f"Pass 2: 写入 memmap ({total_rows:,} 行 × {n_feat} 特征)")

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
            target[:len(close) - horizon] = (
                close[horizon:] - close[:len(close) - horizon]
            ) / close[:len(close) - horizon]
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


def _filter_extremes(X, y, symbols, feature_names):
    """过滤极端值"""
    valid = np.abs(y) < 0.1
    X = X[valid].astype(np.float32)
    y = y[valid].astype(np.float32)
    symbols = [s for s, v in zip(symbols, valid) if v]
    print(f"\n数据准备: {len(X):,} 样本 × {X.shape[1]} 特征")
    print(f"目标: mean={y.mean():.4f}, std={y.std():.4f}, "
          f"min={y.min():.4f}, max={y.max():.4f}")
    return X, y, symbols, feature_names


def select_features(X, y, feature_names):
    """特征选择: 相关性过滤 + SelectFromModel"""
    n_orig = len(feature_names)

    # 1. 高相关特征去重
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

    # 2. SelectFromModel
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
        for tr, te in TimeSeriesSplit(n_splits=3, gap=3).split(X[:split]):
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


def cross_validate(X, y, params, feature_names, n_splits=5):
    """Purged 时间序列交叉验证"""
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=3)
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

    return np.mean(scores), np.std(scores), np.mean(maes), dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))


def save_model(model_data, model_dir):
    """保存模型 + 版本管理"""
    os.makedirs(model_dir, exist_ok=True)

    # 1. 保存模型文件
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(model_dir, 'model.pkl')
    backup_path = os.path.join(model_dir, f'model_{ts}.pkl')
    meta_path = os.path.join(model_dir, 'model_meta.json')

    for path in [model_path, backup_path]:
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    # 2. 保存可读的元数据
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
        elif isinstance(v, dict):
            meta[k] = {kk: vv for kk, vv in list(v.items())[:5]}
    meta['saved_at'] = ts
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    # 3. 追加版本记录
    versions = []
    if os.path.exists(VERSIONS_FILE):
        with open(VERSIONS_FILE) as f:
            versions = yaml.safe_load(f) or []

    versions.append({
        'version': ts,
        'spearman': float(meta.get('cv_spearman', 0)),
        'n_features': meta.get('n_features_selected', 0),
        'n_samples': meta.get('n_samples', 0),
        'start_date': meta.get('start_date', ''),
        'trained_at': meta.get('trained_at', ''),
    })

    # 只保留最近50个版本
    versions = sorted(versions, key=lambda x: x['version'])[-50:]
    with open(VERSIONS_FILE, 'w') as f:
        yaml.dump(versions, f, default_flow_style=False, allow_unicode=True)

    print(f"\n模型已保存:")
    print(f"  最新: {model_path}")
    print(f"  备份: {backup_path}")
    print(f"  元数据: {meta_path}")
    print(f"  版本记录: {VERSIONS_FILE} ({len(versions)} 个版本)")


def main():
    import argparse
    p = argparse.ArgumentParser(description='统一模型训练')
    p.add_argument('--start-date', default='2024-01-01', help='数据起始日期')
    p.add_argument('--trials', type=int, default=50, help='Optuna 搜索轮数')
    p.add_argument('--memmap', action='store_true', help='低内存模式 (<2GB RAM)')
    p.add_argument('--no-tune', action='store_true', help='跳过超参数搜索(用默认参数)')
    args = p.parse_args()

    print("=" * 60)
    print("Stock Quant 模型训练")
    print("=" * 60)
    print(f"数据起始: {args.start_date}")
    print(f"搜索轮数: {args.trials}")
    print(f"低内存: {'是' if args.memmap else '否 (全内存)'}")
    print("=" * 60)

    # 1. 加载数据
    all_data = load_data(DB_PATH)
    if not all_data:
        print("无数据"); return

    # 2. 准备特征
    X, y, _, feature_names = prepare_features(
        all_data, start_date=args.start_date, use_memmap=args.memmap
    )
    del all_data; gc.collect()

    if X is None or len(X) < 500:
        print(f"训练数据不足 ({len(X) if X is not None else 0} 条)"); return

    # 3. 特征选择
    print("\n特征选择...")
    X, feature_names = select_features(X, y, feature_names)

    # 4. 超参数搜索
    if args.no_tune:
        print("\n跳过超参数搜索, 使用默认参数")
        best_params = DEFAULT_PARAMS
        best_tune_score = 0
    else:
        print(f"\n🔍 Optuna 超参数搜索 ({args.trials} 轮)...")
        best_params, best_tune_score = search_hyperparams(X, y, args.trials)

    # 5. 交叉验证
    print(f"\n{n}-折 Purged 交叉验证...")
    cv_score, cv_std, cv_mae, importance = cross_validate(X, y, best_params, feature_names, n_splits=5)

    # 6. 最终模型
    print("\n训练最终模型...")
    final = lgb.LGBMRegressor(**best_params)
    final.fit(X, y)

    # 7. 保存
    model_data = {
        'model': final,
        'feature_names': feature_names,
        'feature_importance': importance,
        'cv_spearman': cv_score,
        'cv_spearman_std': cv_std,
        'cv_mae': cv_mae,
        'tune_spearman': best_tune_score,
        'best_params': best_params,
        'n_samples': len(X),
        'n_features_selected': len(feature_names),
        'prediction_horizon': 3,
        'start_date': args.start_date,
        'trained_at': datetime.now().isoformat(),
        'guideline': 'Spearman 0.05-0.15=可盈利, 0.15+=优秀',
    }
    save_model(model_data, MODEL_DIR)

    # 8. 总结
    print("\n" + "=" * 60)
    print(f"✅ 训练完成!")
    print(f"Spearman: {cv_score:.4f} ± {cv_std:.4f}")
    print(f"MAE: {cv_mae:.4f}")
    print(f"特征: {len(feature_names)} 个")
    print(f"样本: {len(X):,} 条")
    print(f"{model_data['guideline']}")
    print("=" * 60)


if __name__ == '__main__':
    main()