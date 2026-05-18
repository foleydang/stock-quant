#!/usr/bin/env python3
"""
LGBM v4 - Mac 24G训练版

训练环境: Mac M1/M2/M3 24G (或其他8C16G+机器)
推理环境: 2C2G Linux服务器

产出: model_v4.pkl (推理内存<50MB, 2G完全可运行)

优化点 vs v3:
1. Optuna 200次搜索 (vs 10次)
2. LGBM 5个子模型 (vs 3个)
3. XGBoost 2个子模型 (新增)
4. CatBoost 1个子模型 (新增, 可选)
5. 特征筛选 (去掉重要性<10)
6. 5折交叉验证
7. Stacking LR元模型 (轻量, 几KB)

总共8模型, 推理峰值<50MB, pkl<15MB

使用:
  python train_lgb_v4_mac.py            # 全量 (~1小时)
  python train_lgb_v4_mac.py --quick    # 快速 (~10分钟)
  python train_lgb_v4_mac.py --no-catboost  # 跳过CatBoost
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
import sqlite3
import lightgbm as lgb
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from collections import Counter
import time

# 检测可选依赖
try:
    import xgboost as xgb
    HAS_XGB = True
    print(f"✓ XGBoost {xgb.__version__}")
except ImportError:
    HAS_XGB = False
    print("⚠ XGBoost未安装 → pip install xgboost")

try:
    from catboost import CatBoostClassifier
    HAS_CB = True
    print(f"✓ CatBoost")
except ImportError:
    HAS_CB = False
    print("⚠ CatBoost未安装 → pip install catboost")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
    print(f"✓ Optuna {optuna.__version__}")
except ImportError:
    HAS_OPTUNA = False
    print("⚠ Optuna未安装 → pip install optuna")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy.train_lgb_enhanced import EnhancedFeatureEngineer
from strategy.train_lgb_v3 import (
    AdvancedFeatureEngineer, calculate_target_adaptive,
    TIME_FEATURES, ZERO_IMP_FEATURES
)

# ============ 配置 ============
HORIZON = 3
BASE_THRESHOLD = 0.018
N_LGBM = 5    # 5个LGBM子模型
N_XGB = 2     # 2个XGBoost子模型
N_CB = 1      # 1个CatBoost子模型

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/stock_data.db')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/lgb_hs300')

# 特征筛选阈值 (重要性低于此值的特征被剔除)
FEATURE_IMPORTANCE_THRESHOLD = 10


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算所有特征"""
    base = EnhancedFeatureEngineer.calculate_features(df)
    adv = AdvancedFeatureEngineer.calculate_advanced_features(df)
    all_features = pd.concat([base, adv], axis=1)
    drop_cols = TIME_FEATURES + ZERO_IMP_FEATURES
    keep_cols = [c for c in all_features.columns if c not in drop_cols]
    return all_features[keep_cols]


def load_all_data(db_path: str) -> Dict[str, pd.DataFrame]:
    """从数据库加载所有股票数据"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM kline_30m")
    symbols = [r[0] for r in cursor.fetchall()]
    conn.close()

    all_data = {}
    for symbol in symbols:
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(
                'SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol=? ORDER BY date',
                conn, params=(symbol,)
            )
            conn.close()
            if len(df) > 200:
                df['date'] = pd.to_datetime(df['date'])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                all_data[symbol] = df
        except Exception:
            pass

    print(f"加载了 {len(all_data)} 只股票")
    return all_data


def prepare_data(all_data: Dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """准备训练数据"""
    sample = list(all_data.values())[0]
    sample_features = compute_all_features(sample)
    feature_names = list(sample_features.columns)
    print(f"初始特征数: {len(feature_names)}")

    all_X, all_y = [], []
    min_history = 150

    for i, (symbol, df) in enumerate(all_data.items()):
        try:
            features = compute_all_features(df)
            target = calculate_target_adaptive(df, horizon=HORIZON, base_threshold=BASE_THRESHOLD)
            valid_mask = ~(features.isna().any(axis=1)) & (target >= 0)
            features_valid = features[valid_mask].iloc[min_history:].fillna(0)
            target_valid = target[valid_mask][min_history:]
            if len(features_valid) > 30:
                all_X.append(features_valid.values)
                all_y.append(target_valid.astype(int))
        except Exception as e:
            if i < 3:
                print(f"  {symbol}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  已处理 {i + 1}/{len(all_data)}")

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    print(f"总样本: {len(X)}")
    print(f"  上涨: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
    print(f"  下跌: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    return X, y, feature_names


def feature_selection(X, y, feature_names):
    """特征选择: 去掉重要性太低的特征"""
    print("\n=== 特征选择 ===")
    base = lgb.LGBMClassifier(
        objective='binary', num_leaves=63, max_depth=9,
        n_estimators=300, verbose=-1, random_state=42, n_jobs=-1
    )
    base.fit(X, y)
    importances = base.feature_importances_

    keep_idx = [i for i in range(len(feature_names)) if importances[i] >= FEATURE_IMPORTANCE_THRESHOLD]
    selected = [feature_names[i] for i in keep_idx]
    removed = len(feature_names) - len(selected)
    print(f"  {len(feature_names)} → {len(selected)} (去除{removed}个, 阈值<{FEATURE_IMPORTANCE_THRESHOLD})")

    removed_names = [feature_names[i] for i in range(len(feature_names)) if importances[i] < FEATURE_IMPORTANCE_THRESHOLD]
    print(f"  去除: {removed_names}")

    return X[:, keep_idx], selected


# ============ Optuna搜索 ============

def search_lgbm(X, y, n_trials=200):
    """LGBM超参搜索"""
    print(f"\n=== Optuna LGBM ({n_trials}次) ===")
    if not HAS_OPTUNA:
        return _default_lgbm_params()

    tscv = TimeSeriesSplit(n_splits=4)

    def objective(trial):
        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'n_estimators': 500,
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.9),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 0.9),
            'bagging_freq': 5,
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
            'verbose': -1, 'random_state': 42, 'n_jobs': -1,
        }

        scores = []
        for train_idx, test_idx in tscv.split(X):
            m = lgb.LGBMClassifier(**params)
            m.fit(X[train_idx], y[train_idx],
                  eval_set=[(X[test_idx], y[test_idx])],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
            scores.append(accuracy_score(y[test_idx], m.predict(X[test_idx])))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best.update({
        'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
        'n_estimators': 500, 'bagging_freq': 5,
        'verbose': -1, 'random_state': 42, 'n_jobs': -1,
    })
    print(f"最优CV: {study.best_value:.4f}")
    for k, v in best.items():
        print(f"  {k}: {v}")
    return best


def search_xgboost(X, y, n_trials=100):
    """XGBoost超参搜索"""
    if not HAS_XGB:
        return None
    print(f"\n=== Optuna XGBoost ({n_trials}次) ===")
    if not HAS_OPTUNA:
        return _default_xgb_params()

    tscv = TimeSeriesSplit(n_splits=4)

    def objective(trial):
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'logloss',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'n_estimators': 500,
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'random_state': 42, 'n_jobs': -1, 'verbosity': 0,
        }

        scores = []
        for train_idx, test_idx in tscv.split(X):
            m = xgb.XGBClassifier(**params)
            m.fit(X[train_idx], y[train_idx],
                  eval_set=[(X[test_idx], y[test_idx])], verbose=False)
            scores.append(accuracy_score(y[test_idx], m.predict(X[test_idx])))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best.update({
        'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'n_estimators': 500, 'random_state': 42, 'n_jobs': -1, 'verbosity': 0,
    })
    print(f"最优CV: {study.best_value:.4f}")
    return best


def search_catboost(X, y, n_trials=50):
    """CatBoost超参搜索"""
    if not HAS_CB:
        return None
    print(f"\n=== Optuna CatBoost ({n_trials}次) ===")
    if not HAS_OPTUNA:
        return _default_cb_params()

    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            'loss_function': 'Logloss',
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
            'iterations': 500,
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.5, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_seed': 42, 'verbose': False,
        }

        scores = []
        for train_idx, test_idx in tscv.split(X):
            m = CatBoostClassifier(**params)
            m.fit(X[train_idx], y[train_idx],
                  eval_set=(X[test_idx], y[test_idx]),
                  early_stopping_rounds=30, verbose=False)
            scores.append(accuracy_score(y[test_idx], m.predict(X[test_idx])))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best.update({
        'loss_function': 'Logloss', 'iterations': 500,
        'random_seed': 42, 'verbose': False,
    })
    print(f"最优CV: {study.best_value:.4f}")
    return best


def _default_lgbm_params():
    return {
        'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
        'num_leaves': 63, 'max_depth': 9, 'learning_rate': 0.02,
        'n_estimators': 500, 'min_child_samples': 50,
        'feature_fraction': 0.7, 'bagging_fraction': 0.7,
        'bagging_freq': 5, 'reg_alpha': 0.5, 'reg_lambda': 0.5,
        'verbose': -1, 'random_state': 42, 'n_jobs': -1,
    }

def _default_xgb_params():
    return {
        'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'max_depth': 6, 'learning_rate': 0.02, 'n_estimators': 500,
        'min_child_weight': 10, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 0.5, 'reg_lambda': 0.5,
        'random_state': 42, 'n_jobs': -1, 'verbosity': 0,
    }

def _default_cb_params():
    return {
        'loss_function': 'Logloss', 'depth': 6, 'learning_rate': 0.02,
        'iterations': 500, 'l2_leaf_reg': 3,
        'random_seed': 42, 'verbose': False,
    }


# ============ 训练 ============

def train_lgbm_bagging(X, y, params, n_models=5):
    """训练LGBM Bagging (不同种子+微调)"""
    print(f"\n=== LGBM Bagging ({n_models}个子模型) ===")
    tscv = TimeSeriesSplit(n_splits=5)
    models = []

    for i in range(n_models):
        model_params = params.copy()
        model_params['random_state'] = 42 + i * 7
        model_params['feature_fraction'] = min(0.9, params['feature_fraction'] + (i % 3) * 0.05)
        model_params['bagging_fraction'] = min(0.9, params['bagging_fraction'] + (i % 2) * 0.03)

        print(f"\n  LGBM {i+1}/{n_models}:")
        cv_scores = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            m = lgb.LGBMClassifier(**model_params)
            m.fit(X[train_idx], y[train_idx],
                  eval_set=[(X[test_idx], y[test_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)])
            cv_scores.append(accuracy_score(y[test_idx], m.predict(X[test_idx])))
            print(f"    Fold {fold+1}: {cv_scores[-1]:.4f}")
        print(f"    平均: {np.mean(cv_scores):.4f}")

        final = lgb.LGBMClassifier(**model_params)
        final.fit(X, y)
        models.append(final)

    return models


def train_xgboost_bagging(X, y, params, n_models=2):
    """训练XGBoost Bagging"""
    if not HAS_XGB:
        return []

    print(f"\n=== XGBoost ({n_models}个子模型) ===")
    tscv = TimeSeriesSplit(n_splits=5)
    models = []

    for i in range(n_models):
        model_params = params.copy()
        model_params['random_state'] = 42 + i * 13
        model_params['subsample'] = min(0.9, params['subsample'] + i * 0.05)
        model_params['colsample_bytree'] = min(0.9, params['colsample_bytree'] + i * 0.03)

        print(f"\n  XGB {i+1}/{n_models}:")
        cv_scores = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            m = xgb.XGBClassifier(**model_params)
            m.fit(X[train_idx], y[train_idx],
                  eval_set=[(X[test_idx], y[test_idx])], verbose=False)
            cv_scores.append(accuracy_score(y[test_idx], m.predict(X[test_idx])))
            print(f"    Fold {fold+1}: {cv_scores[-1]:.4f}")
        print(f"    平均: {np.mean(cv_scores):.4f}")

        final = xgb.XGBClassifier(**model_params)
        final.fit(X, y, verbose=False)
        models.append(final)

    return models


def train_catboost(X, y, params):
    """训练单个CatBoost"""
    if not HAS_CB:
        return []

    print(f"\n=== CatBoost (1个子模型) ===")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        m = CatBoostClassifier(**params)
        m.fit(X[train_idx], y[train_idx],
              eval_set=(X[test_idx], y[test_idx]),
              early_stopping_rounds=50, verbose=False)
        cv_scores.append(accuracy_score(y[test_idx], m.predict(X[test_idx])))
        print(f"  Fold {fold+1}: {cv_scores[-1]:.4f}")
    print(f"  平均: {np.mean(cv_scores):.4f}")

    final = CatBoostClassifier(**params)
    final.fit(X, y, verbose=False)
    return [final]


# ============ Stacking元模型 ============

def train_lr_meta(X, y, all_models, model_types):
    """训练LR Stacking元模型"""
    print("\n=== LR Stacking元模型 ===")

    # 用所有子模型的概率作为特征
    all_probs = []
    for model, mtype in zip(all_models, model_types):
        if mtype == 'lgbm':
            all_probs.append(model.predict_proba(X)[:, 1])
        elif mtype == 'xgb':
            all_probs.append(model.predict_proba(X)[:, 1])
        elif mtype == 'catboost':
            all_probs.append(model.predict_proba(X)[:, 1])

    # 构建stacking特征
    n_models = len(all_probs)
    avg_prob = np.mean(all_probs, axis=0)
    std_prob = np.std(all_probs, axis=0)

    stacking_features = np.column_stack([
        *all_probs,              # 每个子模型的概率
        avg_prob,                # 平均概率
        std_prob,                # 模型分歧
        avg_prob - 0.5,          # 信号强度
        (avg_prob > 0.6).astype(float),  # 强信号标记
        (avg_prob < 0.4).astype(float),   # 强看跌标记
    ])

    # 时序交叉验证评估LR
    tscv = TimeSeriesSplit(n_splits=5)
    lr_scores = []
    avg_scores = []

    for train_idx, test_idx in tscv.split(stacking_features):
        lr = LogisticRegression(C=1.0, max_iter=500)
        lr.fit(stacking_features[train_idx], y[train_idx])
        lr_pred = lr.predict(stacking_features[test_idx])
        lr_scores.append(accuracy_score(y[test_idx], lr_pred))

        avg_pred = (avg_prob[test_idx] > 0.5).astype(int)
        avg_scores.append(accuracy_score(y[test_idx], avg_pred))

    print(f"  LR CV准确率: {np.mean(lr_scores):.4f}")
    print(f"  简单平均CV: {np.mean(avg_scores):.4f}")
    print(f"  LR vs 平均: +{(np.mean(lr_scores) - np.mean(avg_scores))*100:.2f}%")

    # 训练最终LR
    lr = LogisticRegression(C=1.0, max_iter=500)
    lr.fit(stacking_features, y)

    print(f"  LR系数: {lr.coef_[0]}")
    print(f"  LR intercept: {lr.intercept_[0]}")

    return lr


# ============ 评估 ============

def evaluate_ensemble(X, y, all_models, model_types, lr_meta=None):
    """评估混合ensemble"""
    print(f"\n=== 最终ensemble评估 ===")

    # 获取所有模型概率
    all_probs = []
    for model, mtype in zip(all_models, model_types):
        if mtype == 'lgbm':
            all_probs.append(model.predict_proba(X)[:, 1])
        elif mtype == 'xgb':
            all_probs.append(model.predict_proba(X)[:, 1])
        elif mtype == 'catboost':
            all_probs.append(model.predict_proba(X)[:, 1])

    avg_prob = np.mean(all_probs, axis=0)
    avg_pred = (avg_prob > 0.5).astype(int)
    avg_acc = accuracy_score(y, avg_pred)
    print(f"  概率平均准确率: {avg_acc:.4f}")

    # LR Stacking
    if lr_meta:
        std_prob = np.std(all_probs, axis=0)
        stacking_input = np.column_stack([
            *all_probs, avg_prob, std_prob,
            avg_prob - 0.5,
            (avg_prob > 0.6).astype(float),
            (avg_prob < 0.4).astype(float),
        ])
        lr_pred = lr_meta.predict(stacking_input)
        lr_prob = lr_meta.predict_proba(stacking_input)[:, 1]
        lr_acc = accuracy_score(y, lr_pred)
        print(f"  LR Stacking准确率: {lr_acc:.4f}")
        print(classification_report(y, lr_pred, target_names=['下跌', '上涨']))

        # 使用LR概率作为最终预测
        final_prob = lr_prob
    else:
        final_prob = avg_prob

    # 各模型单独准确率
    for i, (prob, mtype) in enumerate(zip(all_probs, model_types)):
        pred = (prob > 0.5).astype(int)
        acc = accuracy_score(y, pred)
        print(f"  {mtype}_{i}: {acc:.4f}")

    return avg_prob, final_prob


# ============ 保存 ============

def save_model(all_models, model_types, feature_names, lr_meta,
               lgb_params, xgb_params, cb_params, avg_acc, avg_imp,
               X, y, model_dir):
    """保存模型 (单个pkl, 推理轻量)"""
    os.makedirs(model_dir, exist_ok=True)

    model_data = {
        'models': all_models,
        'model_types': model_types,
        'lr_meta': lr_meta,
        'ensemble_accuracy': avg_acc,
        'feature_names': feature_names,
        'avg_importance': avg_imp if avg_imp is not None else np.zeros(len(feature_names)),
        'params': {'lgbm': lgb_params, 'xgb': xgb_params, 'catboost': cb_params},
        'n_models': len(all_models),
        'horizon': HORIZON,
        'threshold': BASE_THRESHOLD,
        'train_samples': len(X),
        'train_date': datetime.now().strftime('%Y-%m-%d'),
        'model_version': 'v4-mixed',
        # 推理时需要的stacking特征构造信息
        'stacking_info': {
            'n_models': len(all_models),
            'feature_cols': list(range(len(all_models))) +  # 各模型概率
                           ['avg', 'std', 'signal_strength', 'strong_up', 'strong_down'],
        }
    }

    output_path = os.path.join(model_dir, 'model_v4.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)

    fsize = os.path.getsize(output_path)

    # 推理内存估算
    mem_estimate = len(all_models) * 5 + 10  # 粗估: 每模型5MB + LR10MB
    print(f"\n✅ 模型已保存: {output_path}")
    print(f"   pkl大小: {fsize/1024:.1f}KB ({fsize/1024/1024:.1f}MB)")
    print(f"   推理内存预估: ~{mem_estimate}MB")
    print(f"   模型构成: {dict(Counter(model_types))}")
    print(f"   特征数: {len(feature_names)}")
    print(f"   2G服务器: {'✅ 可运行' if mem_estimate < 200 else '❌ 可能OOM'}")
    print()
    print("📋 同步到生产服务器:")
    print(f"   scp {output_path} root@47.242.158.242:/root/github/stock-quant/stock-quant/python/models/lgb_hs300/model_v4.pkl")


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(description='LGBM v4 混合模型训练 (Mac 24G)')
    parser.add_argument('--quick', action='store_true', help='快速模式 (减少搜索)')
    parser.add_argument('--no-catboost', action='store_true', help='跳过CatBoost')
    parser.add_argument('--no-xgboost', action='store_true', help='跳过XGBoost')
    parser.add_argument('--no-stacking', action='store_true', help='跳过LR Stacking')
    parser.add_argument('--db', type=str, default=DB_PATH, help='数据库路径')
    args = parser.parse_args()

    start_time = time.time()

    n_trials_lgb = 30 if args.quick else 200
    n_trials_xgb = 15 if args.quick else 100
    n_trials_cb = 10 if args.quick else 50
    use_cb = HAS_CB and not args.no_catboost
    use_xgb = HAS_XGB and not args.no_xgb
    use_stacking = not args.no_stacking

    print("=" * 60)
    print("LGBM v4 - Mac 24G 混合模型训练")
    print(f"  模式: {'快速' if args.quick else '全量'}")
    print(f"  LGBM搜索: {n_trials_lgb}次 → {N_LGBM}个子模型")
    print(f"  XGB搜索: {n_trials_xgb if use_xgb else '跳过'}次 → {N_XGB if use_xgb else 0}个子模型")
    print(f"  CB搜索: {n_trials_cb if use_cb else '跳过'}次 → {N_CB if use_cb else 0}个子模型")
    print(f"  Stacking LR: {'开启' if use_stacking else '跳过'}")
    print("=" * 60)

    # Step 1: 加载+准备数据
    print("\n[1/7] 加载数据")
    data = load_all_data(args.db)
    X, y, feature_names_orig = prepare_data(data)

    # Step 2: 特征选择
    print("\n[2/7] 特征选择")
    X, feature_names = feature_selection(X, y, feature_names_orig)
    print(f"  {len(feature_names_orig)} → {len(feature_names)} 特征")

    # Step 3: Optuna搜索
    print("\n[3/7] 超参搜索")
    lgb_params = search_lgbm(X, y, n_trials=n_trials_lgb)
    xgb_params = search_xgboost(X, y, n_trials=n_trials_xgb) if use_xgb else None
    cb_params = search_catboost(X, y, n_trials=n_trials_cb) if use_cb else None

    # Step 4: 训练各模型
    print("\n[4/7] 训练子模型")
    lgb_models = train_lgbm_bagging(X, y, lgb_params, n_models=N_LGBM)
    xgb_models = train_xgboost_bagging(X, y, xgb_params, n_models=N_XGB) if use_xgb else []
    cb_models = train_catboost(X, y, cb_params) if use_cb else []

    all_models = lgb_models + xgb_models + cb_models
    model_types = ['lgbm'] * len(lgb_models) + ['xgb'] * len(xgb_models) + ['catboost'] * len(cb_models)

    # Step 5: LR Stacking
    print("\n[5/7] Stacking元模型")
    lr_meta = train_lr_meta(X, y, all_models, model_types) if use_stacking else None

    # Step 6: 评估
    print("\n[6/7] 评估")
    avg_prob, final_prob = evaluate_ensemble(X, y, all_models, model_types, lr_meta)
    avg_acc = accuracy_score(y, (avg_prob > 0.5).astype(int))

    # LGBM平均重要性
    lgb_importances = [m.feature_importances_ for m in lgb_models]
    avg_imp = np.mean(lgb_importances, axis=0)

    # Step 7: 保存
    print("\n[7/7] 保存")
    save_model(all_models, model_types, feature_names, lr_meta,
               lgb_params, xgb_params, cb_params, avg_acc, avg_imp,
               X, y, MODEL_DIR)

    elapsed = time.time() - start_time
    print(f"\n⏱ 总耗时: {elapsed/60:.1f}分钟")


if __name__ == '__main__':
    main()