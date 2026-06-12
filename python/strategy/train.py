#!/usr/bin/env python3
"""
LGBM 统一训练脚本 (Mac 本地训练)

用法:
  python train.py --model 30m                    # 30分钟模型 (默认)
  python train.py --model daily                  # 日线模型
  python train.py --model 30m --quick            # 快速模式 (20次Optuna)
  python train.py --model daily --trials 200     # 指定Optuna搜索次数
  python train.py --model 30m --start 2025-01-01 --end 2026-05-31  # 日期过滤

输出:
  30m  → models/lgb_hs300/model.pkl   (双层架构第二层)
  daily → models/lgb_daily/model.pkl  (双层架构第一层)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
import pickle
import json
import sqlite3
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter

warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("⚠ optuna 未安装，跳过超参搜索。安装: pip install optuna")

# ============ 30分钟模型配置 ============
# 数据: kline_30m, ~410万条, ~110个特征 (Enhanced+Advanced, 无Market)
# 数据量大，模型可以更深更复杂，Optuna搜索范围宽
CONFIG_30M = {
    'db_table': 'kline_30m',
    'model_dir': 'models/lgb_hs300',
    'horizon': 3,
    'threshold': 0.010,
    'n_bagging': 3,
    'min_history': 150,
    'min_samples': 200,
    'n_estimators': 2000,             # 大树数，靠 early_stopping 截断
    'early_stopping_rounds': 100,     # 连续100轮不提升则停止
    'time_features': ['day_of_week', 'day_of_month', 'hour', 'minute',
                      'is_morning', 'is_afternoon', 'is_first_hour', 'is_last_hour'],
    'zero_imp_features': [
        'price_above_ma5', 'price_above_ma10', 'price_above_ma20',
        'price_above_ma30', 'price_above_ma60', 'price_above_ma80',
        'price_above_ma100', 'price_above_ma120',
        'ma5_cross_ma10', 'ma10_cross_ma20', 'ma20_cross_ma60', 'ma60_cross_ma120',
        'macd_cross', 'kdj_cross_signal', 'inside_bar', 'breakout_20', 'trend_direction',
    ],
    # Optuna 搜索参数 — 13个参数 (11模型超参 + horizon + threshold), 默认130次搜索
    'optuna_params': {
        # 预测目标
        'horizon':          (1, 10),         # 预测未来N根K线 (30分钟~5小时)
        'threshold':        (0.005, 0.03),   # 涨跌幅阈值 (0.5%~3%)
        # 树结构
        'num_leaves':       (31, 255),       # 大数据集允许更深
        'max_depth':        (5, 15),         # 深度限制
        'min_child_samples': (20, 200),      # 叶子最小样本
        'min_split_gain':   (0.0, 0.5),      # 分裂最小增益
        # 采样
        'subsample':        (0.5, 0.95),     # 行采样 (bagging)
        'subsample_freq':   (1, 10),         # bagging 频率
        'colsample_bytree': (0.5, 0.95),     # 列采样
        'max_bin':          (127, 511),      # 特征分箱数
        # 正则化
        'learning_rate':    (0.005, 0.1),    # 学习率 (log scale)
        'reg_alpha':        (0.0, 2.0),      # L1 正则
        'reg_lambda':       (0.0, 2.0),      # L2 正则
    },
    # 无Optuna时的默认参数 (基于历史训练最优值)
    'default_params': {
        'num_leaves': 63, 'max_depth': 9,
        'min_child_samples': 60, 'min_split_gain': 0.01,
        'subsample': 0.85, 'subsample_freq': 5,
        'colsample_bytree': 0.67, 'max_bin': 255,
        'learning_rate': 0.02, 'reg_alpha': 0.6, 'reg_lambda': 0.8,
    },
}

# ============ 日线模型配置 ============
# 数据: kline_daily, ~97万条, ~130个特征 (Enhanced+Advanced+Market, 含北向资金)
# 数据量小、特征多，偏保守防过拟合，正则化更强
CONFIG_DAILY = {
    'db_table': 'kline_daily',
    'model_dir': 'models/lgb_daily',
    'horizon': 5,
    'threshold': 0.02,
    'n_bagging': 3,
    'min_history': 120,
    'min_samples': 200,
    'n_estimators': 1000,              # 日线数据少，不需要太多树
    'early_stopping_rounds': 100,
    'time_features': ['day_of_week', 'day_of_month', 'is_month_end', 'is_month_start'],
    'zero_imp_features': [],
    # Optuna 搜索参数 — 13个参数 (11模型超参 + horizon + threshold), 默认130次搜索
    'optuna_params': {
        # 预测目标 — 日线周期更长
        'horizon':          (3, 20),         # 预测未来N个交易日 (3天~1个月)
        'threshold':        (0.01, 0.05),    # 涨跌幅阈值 (1%~5%)
        # 树结构 — 比30m保守
        'num_leaves':       (15, 127),       # 上限更低
        'max_depth':        (3, 10),         # 更浅，防过拟合
        'min_child_samples': (30, 300),      # 叶子样本更多
        'min_split_gain':   (0.0, 0.5),
        # 采样
        'subsample':        (0.5, 0.9),      # 采样范围偏保守
        'subsample_freq':   (1, 10),
        'colsample_bytree': (0.5, 0.9),
        'max_bin':          (63, 255),       # 数据少，bin不需要太多
        # 正则化 — 更强
        'learning_rate':    (0.01, 0.15),    # 数据少时学习率可稍高
        'reg_alpha':        (0.0, 3.0),      # L1 范围更大
        'reg_lambda':       (0.0, 3.0),      # L2 范围更大
    },
    # 无Optuna时的默认参数
    'default_params': {
        'num_leaves': 31, 'max_depth': 6,
        'min_child_samples': 100, 'min_split_gain': 0.01,
        'subsample': 0.8, 'subsample_freq': 3,
        'colsample_bytree': 0.7, 'max_bin': 127,
        'learning_rate': 0.03, 'reg_alpha': 0.5, 'reg_lambda': 1.0,
    },
}


# ============ 特征工程 (复用) ============
from strategy.features import EnhancedFeatureEngineer, AdvancedFeatureEngineer, MarketFeatureEngineer

# 日线独有: 北向资金+大盘特征 (命名与基础特征不冲突，直接合并)
# 30分钟线: 不含 Market (北向资金是日级别的，分到每根30分钟K线无意义)


# ============ 数据加载 ============
def load_data(db_path: str, table: str, start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
    """从数据库加载数据，支持日期过滤"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT DISTINCT symbol FROM {table}")
    symbols = [r[0] for r in cursor.fetchall()]
    conn.close()

    # 构建 SQL
    sql = f'SELECT date, open, high, low, close, volume FROM {table} WHERE symbol=?'
    params = []
    if start_date:
        sql += ' AND date >= ?'
        params.append(start_date)
    if end_date:
        sql += ' AND date <= ?'
        params.append(end_date)
    sql += ' ORDER BY date'

    all_data = {}
    for symbol in symbols:
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(sql, conn, params=([symbol] + params))
            conn.close()
            if len(df) > 200:
                df['date'] = pd.to_datetime(df['date'])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                all_data[symbol] = df
        except Exception:
            pass

    print(f"加载了 {len(all_data)} 只股票 (表: {table})")
    if start_date or end_date:
        print(f"  日期范围: {start_date or '不限'} ~ {end_date or '不限'}")
    return all_data


# ============ 特征计算 ============
def compute_features_30m(df: pd.DataFrame) -> pd.DataFrame:
    """30分钟特征: Enhanced + Advanced (无Market,北向资金是日级别不适配)"""
    base = EnhancedFeatureEngineer.calculate_features(df)
    adv = AdvancedFeatureEngineer.calculate_advanced_features(df)
    all_f = pd.concat([base, adv], axis=1)
    drop = CONFIG_30M['time_features'] + CONFIG_30M['zero_imp_features']
    keep = [c for c in all_f.columns if c not in drop]
    return all_f[keep]


def compute_features_daily(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """日线特征: Enhanced + Advanced + Market (北向资金+大盘可对齐日线)"""
    base = EnhancedFeatureEngineer.calculate_features(df)
    adv = AdvancedFeatureEngineer.calculate_advanced_features(df)
    market = MarketFeatureEngineer.calculate_market_features(df, symbol=symbol)
    all_f = pd.concat([base, adv, market], axis=1)
    drop = CONFIG_DAILY['time_features'] + CONFIG_DAILY['zero_imp_features']
    keep = [c for c in all_f.columns if c not in drop]
    return all_f[keep]


# ============ 目标计算 ============
def calculate_target_30m(df: pd.DataFrame, horizon: int = None, threshold: float = None) -> np.ndarray:
    """30分钟自适应阈值目标 (horizon/threshold 可被Optuna搜索)"""
    if horizon is None: horizon = CONFIG_30M['horizon']
    if threshold is None: threshold = CONFIG_30M['threshold']
    close = df['close'].values.astype(float)
    target = np.full(len(close), -1)
    returns = pd.Series(close).pct_change()
    vol = returns.rolling(20).std().values
    median_vol = np.nanmedian(vol)
    for i in range(len(close) - horizon - 1):
        if i < 20 or np.isnan(vol[i]):
            continue
        adj_threshold = threshold * (vol[i] / median_vol) if median_vol > 0 else threshold
        future_ret = (close[i + horizon] - close[i]) / close[i]
        target[i] = 1 if future_ret > adj_threshold else 0
    return target


def calculate_target_daily(df: pd.DataFrame, horizon: int = None, threshold: float = None) -> np.ndarray:
    """日线固定阈值目标 (horizon/threshold 可被Optuna搜索)"""
    if horizon is None: horizon = CONFIG_DAILY['horizon']
    if threshold is None: threshold = CONFIG_DAILY['threshold']
    close = df['close'].values.astype(float)
    target = np.full(len(close), -1)
    for i in range(len(close) - horizon):
        future_ret = (close[i + horizon] - close[i]) / close[i]
        target[i] = 1 if future_ret > threshold else 0
    return target


# ============ 数据准备 ============
def prepare_data(all_data: Dict[str, pd.DataFrame], model_type: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[np.ndarray]]:
    """准备训练数据, 返回 (X, y, feature_names, raw_closes)
    raw_closes 用于 Optuna 搜索 horizon/threshold 时重新计算目标
    """
    cfg = CONFIG_30M if model_type == '30m' else CONFIG_DAILY
    target_fn = calculate_target_30m if model_type == '30m' else calculate_target_daily

    sample_df = list(all_data.values())[0]
    first_symbol = list(all_data.keys())[0]
    if model_type == 'daily':
        sample_features = compute_features_daily(sample_df, symbol=first_symbol)
    else:
        sample_features = compute_features_30m(sample_df)
    feature_names = list(sample_features.columns)
    print(f"特征数: {len(feature_names)}")
    if model_type == 'daily':
        market_feats = [c for c in feature_names if c.startswith('north_') or c.startswith('market_') or c.startswith('sector_')]
        if market_feats:
            print(f"  含北向/市场特征: {market_feats}")

    all_X, all_y, all_closes = [], [], []
    for i, (symbol, df) in enumerate(all_data.items()):
        try:
            if model_type == 'daily':
                features = compute_features_daily(df, symbol=symbol)
            else:
                features = compute_features_30m(df)
            target = target_fn(df)
            mask_features = features.isna().any(axis=1).values
            mask_target = (target >= 0)
            valid_mask = ~mask_features & mask_target
            features_valid = features.iloc[valid_mask].iloc[cfg['min_history']:]
            target_valid = target[valid_mask][cfg['min_history']:]
            closes_valid = df['close'].values.astype(float)[valid_mask][cfg['min_history']:]
            features_valid = features_valid.fillna(0)
            if len(features_valid) > 30:
                all_X.append(features_valid.values)
                all_y.append(target_valid)
                all_closes.append(closes_valid)
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            print(f"  处理 {i+1}/{len(all_data)} 只股票...")

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    print(f"\n训练数据: {len(X)} 条, 正样本率: {y.mean():.1%}")
    return X, y, feature_names, all_closes


# ============ Optuna 超参搜索 ============
def optimize_hyperparams(X: np.ndarray, all_closes: List[np.ndarray],
                         model_type: str, feature_names: List[str],
                         n_trials: int = 130) -> Dict:
    """Optuna 超参数搜索 (13个参数: 11模型 + horizon + threshold)
    
    X: 预计算的特征矩阵 (不变)
    all_closes: 各股票的原始收盘价列表 (用于按 trial 的 horizon/threshold 重算目标)
    返回 best_params, 同时返回 best_y (供后续训练用)
    """
    if not HAS_OPTUNA:
        print("⚠ optuna 未安装，使用默认参数")
        cfg = CONFIG_30M if model_type == '30m' else CONFIG_DAILY
        return {**cfg['default_params'], 'horizon': cfg['horizon'], 'threshold': cfg['threshold'],
                'n_estimators': cfg['n_estimators'],
                'objective': 'binary', 'metric': 'binary_logloss',
                'boosting_type': 'gbdt', 'verbosity': -1, 'n_jobs': -1}, None

    cfg = CONFIG_30M if model_type == '30m' else CONFIG_DAILY
    ps = cfg['optuna_params']
    tscv = TimeSeriesSplit(n_splits=3)
    n_params = len(ps)
    target_fn = calculate_target_30m if model_type == '30m' else calculate_target_daily

    def objective(trial):
        # 1. 获取 trial 参数
        horizon = trial.suggest_int('horizon', *ps['horizon'])
        threshold = trial.suggest_float('threshold', *ps['threshold'])

        # 2. 用 trial 的 horizon/threshold 重新计算目标
        y_trial = _rebuild_targets(all_closes, horizon, threshold, target_fn)
        X_trial = X[:len(y_trial)]  # 对齐 X 和 y

        # 3. 模型参数
        params = {
            'num_leaves': trial.suggest_int('num_leaves', *ps['num_leaves']),
            'max_depth': trial.suggest_int('max_depth', *ps['max_depth']),
            'min_child_samples': trial.suggest_int('min_child_samples', *ps['min_child_samples']),
            'min_split_gain': trial.suggest_float('min_split_gain', *ps['min_split_gain']),
            'subsample': trial.suggest_float('subsample', *ps['subsample']),
            'subsample_freq': trial.suggest_int('subsample_freq', *ps['subsample_freq']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *ps['colsample_bytree']),
            'max_bin': trial.suggest_int('max_bin', *ps['max_bin']),
            'learning_rate': trial.suggest_float('learning_rate', *ps['learning_rate'], log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', *ps['reg_alpha']),
            'reg_lambda': trial.suggest_float('reg_lambda', *ps['reg_lambda']),
            'n_estimators': cfg['n_estimators'],
            'objective': 'binary', 'metric': 'binary_logloss',
            'boosting_type': 'gbdt', 'verbosity': -1, 'n_jobs': -1, 'random_state': 42,
        }

        # 4. 交叉验证
        scores = []
        for train_idx, test_idx in tscv.split(X_trial):
            X_train, X_test = X_trial[train_idx], X_trial[test_idx]
            y_train, y_test = y_trial[train_idx], y_trial[test_idx]
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                      callbacks=[lgb.early_stopping(cfg['early_stopping_rounds'], verbose=False),
                                 lgb.log_evaluation(period=0)])
            scores.append(accuracy_score(y_test, model.predict(X_test)))
        return np.mean(scores)

    print(f"\nOptuna 超参搜索 ({n_trials}次, {n_params}个参数: 11模型 + horizon + threshold)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = dict(study.best_params)
    best['n_estimators'] = cfg['n_estimators']
    best['objective'] = 'binary'
    best['metric'] = 'binary_logloss'
    best['boosting_type'] = 'gbdt'
    best['verbosity'] = -1
    best['n_jobs'] = -1

    # 用最优 horizon/threshold 重建最终 y
    best_y = _rebuild_targets(all_closes, best['horizon'], best['threshold'], target_fn)

    print(f"\n最优参数 ({model_type}):")
    for k in sorted(ps.keys()):
        print(f"  {k}: {best[k]}")
    print(f"最优CV准确率: {study.best_value:.4f}")
    print(f"最终训练样本: {len(best_y)} 条, 正样本率: {best_y.mean():.1%}")
    return best, best_y


def _rebuild_targets(all_closes: List[np.ndarray], horizon: int, threshold: float,
                     target_fn) -> np.ndarray:
    """用新的 horizon/threshold 重建目标变量"""
    all_y = []
    for closes in all_closes:
        # 构造临时 DataFrame 用于 target_fn
        df = pd.DataFrame({'close': closes})
        target = target_fn(df, horizon=horizon, threshold=threshold)
        # 过滤无效标签
        valid = target >= 0
        all_y.append(target[valid])
    return np.concatenate(all_y)


# ============ Bagging 集成训练 ============
def train_ensemble(X: np.ndarray, y: np.ndarray, params: Dict, feature_names: List[str],
                   model_type: str) -> Dict:
    """训练 Bagging 集成"""
    cfg = CONFIG_30M if model_type == '30m' else CONFIG_DAILY
    n_models = cfg['n_bagging']
    print(f"\n=== 训练 Bagging 集成 ({n_models} 个子模型) ===")

    tscv = TimeSeriesSplit(n_splits=5)
    models = []
    best_iterations = []

    for m_idx in range(n_models):
        print(f"\n子模型 {m_idx + 1}/{n_models}:")
        model_params = params.copy()
        model_params['random_state'] = 42 + m_idx * 7
        # 子模型微调列采样率增加多样性
        model_params['colsample_bytree'] = min(0.9, params.get('colsample_bytree', 0.8) + (m_idx % 3) * 0.05)

        cv_scores, fold_iters = [], []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model = lgb.LGBMClassifier(**model_params)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                      callbacks=[lgb.early_stopping(cfg['early_stopping_rounds'], verbose=False),
                                 lgb.log_evaluation(period=0)])
            cv_scores.append(accuracy_score(y_test, model.predict(X_test)))
            fold_iters.append(model.best_iteration_)
            print(f"  Fold {fold+1}: Acc={cv_scores[-1]:.4f}, BestIter={model.best_iteration_}")

        avg_acc = np.mean(cv_scores)
        avg_iter = int(np.mean(fold_iters))
        print(f"  平均: Acc={avg_acc:.4f}, BestIter={avg_iter}")

        final_params = model_params.copy()
        final_params['n_estimators'] = avg_iter
        final_model = lgb.LGBMClassifier(**final_params)
        final_model.fit(X, y)
        models.append(final_model)
        best_iterations.append(avg_iter)

    # 集成评估
    print(f"\n=== 集成评估 ===")
    all_preds = np.array([m.predict(X) for m in models]).T
    ensemble_pred = np.apply_along_axis(
        lambda x: Counter(x).most_common(1)[0][0], axis=1, arr=all_preds)
    probs = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    ensemble_acc = accuracy_score(y, ensemble_pred)
    ensemble_auc = roc_auc_score(y, probs)

    print(f"  单模型Acc范围: {min(accuracy_score(y, p) for p in all_preds):.2%} ~ {max(accuracy_score(y, p) for p in all_preds):.2%}")
    print(f"  集成投票Acc: {ensemble_acc:.2%}, AUC: {ensemble_auc:.4f}")
    print(classification_report(y, ensemble_pred, target_names=['下跌', '上涨']))

    avg_importance = np.mean([m.feature_importances_ for m in models], axis=0)
    top_idx = np.argsort(avg_importance)[::-1][:20]
    print(f"\nTop 20 特征:")
    for i in top_idx:
        print(f"  {feature_names[i]}: {avg_importance[i]:.0f}")

    keep_features = [feature_names[i] for i in range(len(feature_names)) if avg_importance[i] >= 1]
    print(f"建议保留特征: {len(keep_features)}/{len(feature_names)}")

    # 全部存到 model_data 里
    return {
        'models': models,
        'cv_accuracy': round(np.mean([np.mean(accuracy_score(y, p)) for p in all_preds]), 4),
        'ensemble_accuracy': round(ensemble_acc, 4),
        'ensemble_auc': round(ensemble_auc, 4),
        'best_iterations': best_iterations,
        'feature_names': feature_names,
        'keep_features': keep_features,
        'n_models': n_models,
        'horizon': params.get('horizon', cfg['horizon']),
        'threshold': params.get('threshold', cfg['threshold']),
        'train_samples': len(X),
        'train_date': datetime.now().strftime('%Y-%m-%d'),
        'params': params,
        'model_type': model_type,
    }


# ============ 模型保存 ============
def save_model(model_data: Dict, model_dir: str, model_type: str):
    """保存模型"""
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model_data, f)

    labels = {'30m': '30分钟K线', 'daily': '日线K线'}
    roles = {'30m': '双层架构第二层 — 精确进出场信号', 'daily': '双层架构第一层 — 趋势方向判断'}

    metadata = {
        "model_name": f"lgb_{model_type}_v1",
        "version": "1.0",
        "train_date": model_data['train_date'],
        "architecture": f"LGBM Bagging ({model_data['n_models']}个子模型投票)",
        "data": labels.get(model_type, model_type),
        "target": f"未来{model_data['horizon']}根K线涨>{model_data['threshold']*100}%",
        "role": roles.get(model_type, ''),
        "performance": {
            "cv_accuracy": model_data['cv_accuracy'],
            "ensemble_accuracy": model_data['ensemble_accuracy'],
            "ensemble_auc": model_data['ensemble_auc'],
        },
        "n_features": len(model_data['feature_names']),
        "n_samples": model_data['train_samples'],
    }

    with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    model_size_mb = os.path.getsize(os.path.join(model_dir, 'model.pkl')) / 1024 / 1024
    print(f"\n✅ 模型已保存到 {model_dir}")
    print(f"   model.pkl: {model_size_mb:.1f} MB")


# ============ 主流程 ============
def main():
    parser = argparse.ArgumentParser(description='LGBM 统一训练脚本')
    parser.add_argument('--model', type=str, default='30m', choices=['30m', 'daily'],
                        help='模型类型: 30m=30分钟, daily=日线 (默认: 30m)')
    parser.add_argument('--start', type=str, default=None,
                        help='数据起始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                        help='数据截止日期 (YYYY-MM-DD)')
    parser.add_argument('--trials', type=int, default=130,
                        help='Optuna 搜索次数 (13个参数, 默认: 130)')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式 (20次 Optuna, 快速验证用)')
    parser.add_argument('--no-optuna', action='store_true',
                        help='跳过 Optuna 超参搜索')
    args = parser.parse_args()

    cfg = CONFIG_30M if args.model == '30m' else CONFIG_DAILY
    model_dir = os.path.join(os.path.dirname(__file__), '..', cfg['model_dir'])
    model_dir = os.path.abspath(model_dir)
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data/stock_data.db')
    db_path = os.path.abspath(db_path)

    label = '30分钟' if args.model == '30m' else '日线'
    print("=" * 60)
    print(f"  LGBM {label}模型训练")
    print(f"  数据: {cfg['db_table']} | 预测: 未来{cfg['horizon']}根K线")
    print(f"  阈值: {cfg['threshold']*100}% | 集成: {cfg['n_bagging']}子模型")
    print("=" * 60)

    # 1. 加载数据
    print(f"\n数据库: {db_path}")
    all_data = load_data(db_path, cfg['db_table'], args.start, args.end)
    if not all_data:
        print("❌ 未加载到数据")
        return

    # 2. 准备特征 (X不变, closes 供 Optuna 重算目标)
    X, y, feature_names, all_closes = prepare_data(all_data, args.model)
    if len(X) < cfg['min_samples']:
        print(f"❌ 数据不足: {len(X)} 条 (需要≥{cfg['min_samples']})")
        return

    # 3. Optuna 超参搜索 (含 horizon/threshold)
    if args.no_optuna:
        print("\n跳过 Optuna 搜索，使用默认参数")
        params = {**cfg['default_params'], 'horizon': cfg['horizon'], 'threshold': cfg['threshold'],
                  'n_estimators': cfg['n_estimators'],
                  'objective': 'binary', 'metric': 'binary_logloss',
                  'boosting_type': 'gbdt', 'verbosity': -1, 'n_jobs': -1}
        # y 保持 prepare_data 的默认值
    else:
        n_trials = 20 if args.quick else args.trials
        print(f"Optuna 搜索: {n_trials}次 {'(快速)' if args.quick else ''}")
        params, best_y = optimize_hyperparams(X, all_closes, args.model, feature_names, n_trials=n_trials)
        if best_y is not None:
            y = best_y
            X = X[:len(y)]  # 对齐 X 和 y (horizon 变化导致有效样本数变化)

    # 4. 训练
    model_data = train_ensemble(X, y, params, feature_names, args.model)

    # 5. 保存
    save_model(model_data, model_dir, args.model)

    print("\n" + "=" * 60)
    print("  🎉 训练完成!")
    print("=" * 60)
    if args.model == '30m':
        print(f"\nscp {model_dir}/model.pkl root@47.242.158.242:/root/github/stock-quant/python/models/lgb_hs300/")
    else:
        print(f"\nscp {model_dir}/model.pkl root@47.242.158.242:/root/github/stock-quant/python/models/lgb_daily/")
        print(f"scp strategy/train.py root@47.242.158.242:/root/github/stock-quant/python/strategy/")
    print(f"\n回测: python lgbm_backtest.py")


if __name__ == '__main__':
    main()