#!/usr/bin/env python3
"""
v9 回归模型训练 — 截面排序 (Regression + Cross-Sectional Ranking)

核心改进:
1. 分类 → 回归: 预测未来收益率（连续值），不是涨跌方向
2. 截面排序: 对所有股票打分排序，买 Top N
3. 双层架构: 日线选股 + 30分钟择时
4. 情绪因子: 涨跌停/异常量/异常收益 + 龙虎榜
5. 评估: Rank IC (Spearman 排序相关性)

业界对标: WorldQuant/Citadel 的截面排序双层架构

用法:
  python train_v9.py --model daily
  python train_v9.py --model 30m
  python train_v9.py --model daily --quick
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import sqlite3
import pickle
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.features import (
    EnhancedFeatureEngineer,
    AdvancedFeatureEngineer,
    MarketFeatureEngineer,
)

warnings.filterwarnings('ignore')

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/stock_data.db')


# ============ 配置 ============
CONFIG_30M = {
    'db_table': 'kline_30m',
    'model_dir': 'models/lgb_30m',
    'label': '30分钟',
    'horizon': 3,           # 预测未来3根K线(90分钟)
    'min_history': 150,
    'min_samples': 200,
    'n_estimators': 2000,
    'early_stopping': 50,
    'search_sample': 0.25,  # 搜索时采样25%
    'search_estimators': 500,
    'optuna_trials': 100,
    'features': 'enhanced+advanced',  # 纯30分钟特征，不含Market
    'purged_gap': 3,
}

CONFIG_DAILY = {
    'db_table': 'kline_daily',
    'model_dir': 'models/lgb_daily',
    'label': '日线',
    'horizon': 5,           # 预测未来5个交易日
    'min_history': 120,
    'min_samples': 200,
    'n_estimators': 1500,
    'early_stopping': 50,
    'search_sample': 0.5,
    'search_estimators': 500,
    'optuna_trials': 100,
    'features': 'enhanced+advanced+market',  # 日线含北向资金
    'purged_gap': 1,
}


# ============ 特征计算 ============
def compute_features(df: pd.DataFrame, symbol: str, cfg: dict) -> pd.DataFrame:
    """根据配置计算特征"""
    base = EnhancedFeatureEngineer.calculate_features(df)
    adv = AdvancedFeatureEngineer.calculate_advanced_features(df)

    features = pd.concat([base, adv], axis=1)

    if 'market' in cfg['features']:
        market = MarketFeatureEngineer.calculate_market_features(df, symbol=symbol)
        features = pd.concat([features, market], axis=1)

    # 去掉时间特征
    time_cols = ['day_of_week', 'day_of_month', 'is_month_end', 'is_month_start',
                 'hour', 'minute', 'is_morning', 'is_afternoon']
    drop = [c for c in time_cols if c in features.columns]
    return features.drop(columns=drop, errors='ignore')


def load_sentiment_features(conn) -> pd.DataFrame:
    """加载情绪因子"""
    try:
        df = pd.read_sql(
            "SELECT symbol, trade_date as date, lhb_flag, lhb_net_buy, lhb_net_buy_ratio, "
            "lhb_ret_5d, is_limit_up, is_limit_down, vol_ratio_20, "
            "abnormal_ret, consecutive_limit_up FROM sentiment_daily",
            conn
        )
        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'], format='mixed').dt.strftime('%Y-%m-%d')
            return df
    except Exception:
        pass
    return pd.DataFrame()


# ============ 数据加载 ============
def load_data(db_path: str, table: str) -> Dict[str, pd.DataFrame]:
    """从数据库加载K线数据"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT DISTINCT symbol FROM {table}")
    symbols = [r[0] for r in cursor.fetchall()]

    all_data = {}
    for sym in symbols:
        try:
            df = pd.read_sql(
                f"SELECT * FROM {table} WHERE symbol=? ORDER BY date",
                conn, params=(sym,)
            )
            if len(df) >= 120:
                df['date'] = pd.to_datetime(df['date'], format='mixed')
                df = df.sort_values('date').reset_index(drop=True)
                all_data[sym] = df
        except Exception:
            continue

    conn.close()
    print(f"加载了 {len(all_data)} 只股票 (表: {table})")
    return all_data


def prepare_training_data(
    all_data: Dict[str, pd.DataFrame],
    cfg: dict,
    conn
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    准备回归训练数据
    Returns: X, y (连续收益率), feature_names, symbols
    """
    all_features = []
    all_targets = []
    all_symbols = []
    feature_names = None

    # 加载情绪特征
    sentiment_df = load_sentiment_features(conn)
    has_sentiment = len(sentiment_df) > 0
    if has_sentiment:
        print(f" 含情绪特征: {list(sentiment_df.columns)}")

    success = 0
    horizon = cfg['horizon']

    for i, (symbol, df) in enumerate(all_data.items()):
        try:
            features = compute_features(df, symbol, cfg)

            if feature_names is None:
                feature_names = list(features.columns)

            # 回归目标: 未来N根K线的收益率
            close = df['close'].values.astype(float)
            target = np.full(len(close), np.nan)
            for j in range(len(close) - horizon):
                target[j] = (close[j + horizon] - close[j]) / close[j]

            # 合并情绪特征
            if has_sentiment:
                dates = df['date'].dt.strftime('%Y-%m-%d')
                sent = sentiment_df[sentiment_df['symbol'] == symbol].copy()
                sent = sent.set_index('date')
                # 按日期对齐
                sent_cols = [c for c in sent.columns if c not in ['symbol', 'date']]
                for col in sent_cols:
                    features[f'sent_{col}'] = dates.map(
                        lambda d: sent.loc[d, col] if d in sent.index else 0
                    ).fillna(0).values

            # NaN处理
            features = features.fillna(method='ffill').fillna(0)
            feature_names = list(features.columns)

            valid_mask = ~np.isnan(target)
            features_valid = features[valid_mask]
            target_valid = target[valid_mask]

            # 过滤前120行
            if len(features_valid) > cfg['min_history']:
                features_valid = features_valid.iloc[cfg['min_history']:]
                target_valid = target_valid[cfg['min_history']:]

            if len(features_valid) > 50:
                all_features.append(features_valid.values)
                all_targets.append(target_valid)
                all_symbols.extend([symbol] * len(features_valid))
                success += 1

        except Exception as e:
            continue

        if (i + 1) % 100 == 0:
            print(f"  处理 {i+1}/{len(all_data)} 只股票 (成功{success})")

    if not all_features:
        return None, None, None, None

    X = np.vstack(all_features)
    y = np.concatenate(all_targets)

    # 去除极端值 (>15% 的收益率)
    valid = (np.abs(y) < 0.15)
    X, y = X[valid], y[valid]
    all_symbols = [s for s, v in zip(all_symbols, valid) if v]

    print(f"\n训练数据: {len(X):,} 条, 特征: {len(feature_names)}")
    print(f"目标分布: mean={y.mean():.4f}, std={y.std():.4f}, "
          f"min={y.min():.4f}, max={y.max():.4f}")

    return X, y, feature_names, all_symbols


# ============ 模型训练 ============
def train_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cfg: dict,
    quick: bool = False
) -> Dict:
    """训练回归模型，用 Spearman 排序相关性评估"""

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_trials = 20 if quick else cfg['optuna_trials']
    n_params = 10

    # Purged K-Fold
    tscv = TimeSeriesSplit(n_splits=5, gap=cfg['purged_gap'])

    print(f"\nOptuna 超参搜索 ({n_trials}次, {n_params}个参数, "
          f"5折PurgedCV, gap={cfg['purged_gap']}, 目标=Spearman)...")

    # 搜索时用80%数据
    split_idx = int(len(X) * 0.8)
    X_search, y_search = X[:split_idx], y[:split_idx]

    def objective(trial):
        params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': 42,
            'n_estimators': cfg['search_estimators'],

            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.95),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        }

        # 采样加速
        if cfg['search_sample'] < 1.0 and len(X_search) > 100000:
            n_sample = int(len(X_search) * cfg['search_sample'])
            idx = np.random.RandomState(42 + trial.number).choice(
                len(X_search), n_sample, replace=False)
            X_s, y_s = X_search[idx], y_search[idx]
        else:
            X_s, y_s = X_search, y_search

        mini_tscv = TimeSeriesSplit(n_splits=3, gap=cfg['purged_gap'])
        scores = []
        for train_idx, test_idx in mini_tscv.split(X_s):
            X_tr, X_te = X_s[train_idx], X_s[test_idx]
            y_tr, y_te = y_s[train_idx], y_s[test_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_te, y_te)],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
            y_pred = model.predict(X_te)
            if len(y_pred) > 2:
                corr, _ = spearmanr(y_pred, y_te)
                scores.append(corr if not np.isnan(corr) else 0)
            else:
                scores.append(0)

        return np.mean(scores) if scores else 0

    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_value = study.best_value

    print(f"\n最优Spearman: {best_value:.4f}")
    print(f"最优参数:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # ====== 用最优参数 + 全量数据 + 5折CV ======
    final_params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': 42,
        'n_estimators': cfg['n_estimators'],
    }
    final_params.update(best_params)

    print(f"\n5折Purged交叉验证 (全量数据)...")
    cv_spearman = []
    cv_rmse = []
    cv_mae = []
    models = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgb.LGBMRegressor(**final_params)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  callbacks=[lgb.early_stopping(cfg['early_stopping'], verbose=False)])

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        corr, _ = spearmanr(y_pred, y_test)
        if np.isnan(corr):
            corr = 0

        cv_spearman.append(corr)
        cv_rmse.append(rmse)
        cv_mae.append(mae)
        models.append(model)

        print(f"  Fold {fold+1}: RMSE={rmse:.4f}, MAE={mae:.4f}, Spearman={corr:.4f}")

    avg_spearman = np.mean(cv_spearman)
    avg_rmse = np.mean(cv_rmse)
    avg_mae = np.mean(cv_mae)
    print(f"\n平均: Spearman={avg_spearman:.4f}, RMSE={avg_rmse:.4f}, MAE={avg_mae:.4f}")

    # ====== 特征选择 ======
    if len(feature_names) > 20:
        # 相关度过滤
        valid_start = int(len(X) * 0.8)
        X_train_part = X[:valid_start]
        corr_matrix = np.corrcoef(X_train_part.T)
        to_remove = set()
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                if abs(corr_matrix[i, j]) > 0.95:
                    if i not in to_remove and j not in to_remove:
                        to_remove.add(j)

        if to_remove:
            keep = np.ones(len(feature_names), dtype=bool)
            keep[list(to_remove)] = False
            X = X[:, keep]
            feature_names = [fn for fn, m in zip(feature_names, keep) if m]
            print(f"\n特征去冗余: 保留 {sum(keep)}/{len(feature_names)+len(to_remove)}")

        # SelectFromModel
        selector_model = lgb.LGBMRegressor(**final_params)
        selector_model.fit(X_train_part[:-50], y[:valid_start][:-50],
                          eval_set=[(X_train_part[-50:], y[:valid_start][-50:])],
                          callbacks=[lgb.early_stopping(30, verbose=False)])

        selector = SelectFromModel(selector_model, threshold='median', prefit=True)
        X = selector.transform(X)
        selected_mask = selector.get_support()
        feature_names = [fn for fn, m in zip(feature_names, selected_mask) if m]
        print(f"特征选择: 保留 {len(feature_names)} 个特征")

    # ====== 最终模型 ======
    final_model = models[-1]

    # 特征重要性
    importance = final_model.feature_importances_
    if len(feature_names) == len(importance):
        top_idx = np.argsort(importance)[-20:][::-1]
        print(f"\nTop 20 特征:")
        for idx in top_idx:
            print(f"  {feature_names[idx]}: {importance[idx]:.0f}")

    return {
        'model': final_model,
        'feature_names': feature_names,
        'best_params': best_params,
        'cv_spearman': round(avg_spearman, 4),
        'cv_rmse': round(avg_rmse, 4),
        'cv_mae': round(avg_mae, 4),
        'horizon': cfg['horizon'],
        'n_features': len(feature_names),
        'n_samples': len(X),
    }


def save_model(model_data: Dict, cfg: dict, model_type: str):
    """保存模型"""
    model_dir = cfg['model_dir']
    os.makedirs(model_dir, exist_ok=True)

    # 保存模型文件
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    size_mb = os.path.getsize(model_path) / 1024 / 1024

    # 保存元数据
    meta = {
        'model_type': model_type,
        'label': cfg['label'],
        'horizon': model_data['horizon'],
        'n_features': model_data['n_features'],
        'n_samples': model_data['n_samples'],
        'cv_spearman': model_data['cv_spearman'],
        'cv_rmse': model_data['cv_rmse'],
        'cv_mae': model_data['cv_mae'],
        'best_params': model_data['best_params'],
        'feature_names': model_data['feature_names'][:50],
        'trained_at': datetime.now().isoformat(),
        'target': f"回归 — 未来{model_data['horizon']}根K线收益率",
        'role': 'α选股层' if model_type == 'daily' else 'γ择时层',
    }

    meta_path = os.path.join(model_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 模型已保存到 {model_dir}")
    print(f"  model.pkl: {size_mb:.1f} MB")
    print(f"  meta.json: {json.dumps(meta, indent=2, ensure_ascii=False)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['daily', '30m'], default='30m')
    parser.add_argument('--quick', action='store_true', help='快速模式(20次Optuna)')
    args = parser.parse_args()

    cfg = CONFIG_DAILY if args.model == 'daily' else CONFIG_30M
    model_type = args.model

    print("=" * 60)
    print(f" LGBM {cfg['label']}模型训练 (回归 + 截面排序)")
    print(f" 数据: {cfg['db_table']} | 预测: 未来{cfg['horizon']}根K线")
    print(f" 评估: Spearman 排序相关性 (Rank IC)")
    print("=" * 60)

    # 1. 加载数据
    print(f"\n数据库: {DB_PATH}")
    all_data = load_data(DB_PATH, cfg['db_table'])

    # 2. 准备特征和目标
    conn = sqlite3.connect(DB_PATH)
    X, y, feature_names, symbols = prepare_training_data(all_data, cfg, conn)
    conn.close()

    if X is None:
        print("❌ 数据准备失败")
        return

    # 3. 训练
    model_data = train_model(X, y, feature_names, cfg, quick=args.quick)

    # 4. 保存
    save_model(model_data, cfg, model_type)

    print("\n" + "=" * 60)
    print(" 🎉 训练完成!")
    print(f"  Rank IC (Spearman): {model_data['cv_spearman']:.4f}")
    print(f"  特征数: {model_data['n_features']}")
    print(f"  样本数: {model_data['n_samples']:,}")
    print("=" * 60)


if __name__ == '__main__':
    main()