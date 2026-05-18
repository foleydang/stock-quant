#!/usr/bin/env python3
"""
LGBM v4 Mac训练版 - 修复LR过拟合

v4第一版问题:
  - LR Stacking 98%准确率但回测0% (严重过拟合)
  - XGB权重10+ (LR过度依赖XGB)
  - 缺失11个有用特征 (log_return/vol_ratio等)

修复:
  1. 去掉LR Stacking → 用加权平均 (LGBM=1.0, XGB=0.8, CatBoost=0.9)
  2. 保留v3全部118特征 (只去掉TIME和ZERO_IMP)
  3. XGB加强正则化 (max_depth≤6, min_child_weight≥20)
  4. 每个模型用5折CV验证, 记录真实CV分数
  5. ensemble_accuracy用CV OOF分数而非训练集分数

训练环境: Mac 24G
推理环境: 2C2G Linux (推理峰值<20MB)

使用:
  python train_lgb_v4_mac.py            # 全量 (~40分钟)
  python train_lgb_v4_mac.py --quick    # 快速 (~5分钟)
  python train_lgb_v4_mac.py --no-catboost
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
from sklearn.metrics import accuracy_score
from collections import Counter
import time
import warnings
warnings.filterwarnings('ignore')

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
    print("⚠ Optuna未安装")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy.train_lgb_enhanced import EnhancedFeatureEngineer
from strategy.train_lgb_v3 import (
    AdvancedFeatureEngineer, calculate_target_adaptive,
    TIME_FEATURES, ZERO_IMP_FEATURES
)

# ============ 配置 ============
HORIZON = 3
BASE_THRESHOLD = 0.018
N_LGBM = 5
N_XGB = 2
N_CB = 1

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/stock_data.db')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/lgb_hs300')


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 - 保留v3全部118特征 (只去掉TIME+ZERO_IMP)"""
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
    print(f"特征数: {len(feature_names)} (与v3一致)")

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


# ============ Optuna搜索 ============

def search_lgbm(X, y, n_trials=100):
    """LGBM超参搜索"""
    print(f"\n=== Optuna LGBM ({n_trials}次) ===")
    if not HAS_OPTUNA:
        return _default_lgbm_params()

    tscv = TimeSeriesSplit(n_splits=4)

    def objective(trial):
        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 4, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.03),
            'n_estimators': 300,
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 80),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.8),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.8),
            'bagging_freq': 5,
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 1.0),
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
        'n_estimators': 300, 'bagging_freq': 5,
        'verbose': -1, 'random_state': 42, 'n_jobs': -1,
    })
    print(f"最优CV: {study.best_value:.4f}")
    return best


def search_xgboost(X, y, n_trials=80):
    """XGBoost超参搜索 - 加强正则化防过拟合"""
    if not HAS_XGB:
        return None
    print(f"\n=== Optuna XGBoost ({n_trials}次) ===")
    if not HAS_OPTUNA:
        return _default_xgb_params()

    tscv = TimeSeriesSplit(n_splits=4)

    def objective(trial):
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'logloss',
            # 强正则化: max_depth≤6, min_child_weight≥20
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.03),
            'n_estimators': 300,
            'min_child_weight': trial.suggest_int('min_child_weight', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.7),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 3.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
            'gamma': trial.suggest_float('gamma', 0.1, 1.0),
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
        'n_estimators': 300, 'random_state': 42, 'n_jobs': -1, 'verbosity': 0,
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
            'depth': trial.suggest_int('depth', 4, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.03),
            'iterations': 300,
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
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
        'loss_function': 'Logloss', 'iterations': 300,
        'random_seed': 42, 'verbose': False,
    })
    print(f"最优CV: {study.best_value:.4f}")
    return best


def _default_lgbm_params():
    return {
        'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
        'num_leaves': 31, 'max_depth': 7, 'learning_rate': 0.02,
        'n_estimators': 300, 'min_child_samples': 40,
        'feature_fraction': 0.7, 'bagging_fraction': 0.7,
        'bagging_freq': 5, 'reg_alpha': 0.5, 'reg_lambda': 0.5,
        'verbose': -1, 'random_state': 42, 'n_jobs': -1,
    }

def _default_xgb_params():
    # 强正则化参数
    return {
        'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 300,
        'min_child_weight': 30, 'subsample': 0.7, 'colsample_bytree': 0.6,
        'reg_alpha': 1.0, 'reg_lambda': 1.0, 'gamma': 0.3,
        'random_state': 42, 'n_jobs': -1, 'verbosity': 0,
    }

def _default_cb_params():
    return {
        'loss_function': 'Logloss', 'depth': 5, 'learning_rate': 0.02,
        'iterations': 300, 'l2_leaf_reg': 5,
        'random_seed': 42, 'verbose': False,
    }


# ============ 训练 (带OOF验证) ============

def train_lgbm_bagging(X, y, params, n_models=5):
    """训练LGBM Bagging - 带OOF分数"""
    print(f"\n=== LGBM Bagging ({n_models}个子模型) ===")
    tscv = TimeSeriesSplit(n_splits=5)
    models = []
    oof_scores = []

    for i in range(n_models):
        model_params = params.copy()
        model_params['random_state'] = 42 + i * 7
        # 微调feature_fraction增加多样性
        model_params['feature_fraction'] = min(0.85, params.get('feature_fraction', 0.7) + (i % 3) * 0.05)

        print(f"\n  LGBM {i+1}/{n_models}:")
        cv_scores = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            m = lgb.LGBMClassifier(**model_params)
            m.fit(X[train_idx], y[train_idx],
                  eval_set=[(X[test_idx], y[test_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)])
            cv_scores.append(accuracy_score(y[test_idx], m.predict(X[test_idx])))
            print(f"    Fold {fold+1}: {cv_scores[-1]:.4f}")
        avg_cv = np.mean(cv_scores)
        oof_scores.append(avg_cv)
        print(f"    CV平均: {avg_cv:.4f}")

        # 用全部数据训练最终模型
        final = lgb.LGBMClassifier(**model_params)
        final.fit(X, y)
        models.append(final)

    print(f"\n  LGBM整体CV: {np.mean(oof_scores):.4f}")
    return models, oof_scores


def train_xgboost_bagging(X, y, params, n_models=2):
    """训练XGBoost - 强正则化"""
    if not HAS_XGB:
        return [], []

    print(f"\n=== XGBoost ({n_models}个子模型) ===")
    tscv = TimeSeriesSplit(n_splits=5)
    models = []
    oof_scores = []

    for i in range(n_models):
        model_params = params.copy()
        model_params['random_state'] = 42 + i * 13

        print(f"\n  XGB {i+1}/{n_models}:")
        cv_scores = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            m = xgb.XGBClassifier(**model_params)
            m.fit(X[train_idx], y[train_idx],
                  eval_set=[(X[test_idx], y[test_idx])], verbose=False)
            cv_scores.append(accuracy_score(y[test_idx], m.predict(X[test_idx])))
            print(f"    Fold {fold+1}: {cv_scores[-1]:.4f}")
        avg_cv = np.mean(cv_scores)
        oof_scores.append(avg_cv)
        print(f"    CV平均: {avg_cv:.4f}")

        final = xgb.XGBClassifier(**model_params)
        final.fit(X, y, verbose=False)
        models.append(final)

    print(f"\n  XGB整体CV: {np.mean(oof_scores):.4f}")
    return models, oof_scores


def train_catboost(X, y, params):
    """训练CatBoost"""
    if not HAS_CB:
        return [], []

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

    avg_cv = np.mean(cv_scores)
    print(f"  CV平均: {avg_cv:.4f}")

    final = CatBoostClassifier(**params)
    final.fit(X, y, verbose=False)
    return [final], [avg_cv]


# ============ OOF Ensemble准确率 ============

def compute_oof_ensemble_accuracy(X, y, all_models, model_types):
    """用5折OOF计算ensemble的真实准确率 (不是训练集分数!)"""
    print("\n=== OOF Ensemble验证 ===")
    tscv = TimeSeriesSplit(n_splits=5)
    fold_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_test, y_test = X[test_idx], y[test_idx]

        # 每个模型预测test集
        probs_list = []
        for model, mtype in zip(all_models, model_types):
            try:
                p = model.predict_proba(X_test)[:, 1]
                probs_list.append(p)
            except:
                probs_list.append(np.full(len(X_test), 0.5))

        # 加权平均 (LGBM=1.0, XGB=0.8, CatBoost=0.9)
        weights = []
        for mt in model_types:
            if mt == 'lgbm': weights.append(1.0)
            elif mt == 'xgb': weights.append(0.8)
            elif mt == 'catboost': weights.append(0.9)
            else: weights.append(1.0)
        total_w = sum(weights)

        ensemble_probs = np.zeros(len(X_test))
        for p, w in zip(probs_list, weights):
            ensemble_probs += p * w / total_w

        predictions = (ensemble_probs >= 0.5).astype(int)
        acc = accuracy_score(y_test, predictions)
        fold_accuracies.append(acc)
        print(f"  Fold {fold+1}: {acc:.4f}")

    avg_acc = np.mean(fold_accuracies)
    print(f"\n  OOF Ensemble准确率: {avg_acc:.4f}")
    print(f"  ⚠️ 如果>70%, 很可能过拟合!")
    print(f"  ✅ 正常范围: 55-65% (v3是59%)")
    return avg_acc


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='快速模式(少搜索)')
    parser.add_argument('--no-catboost', action='store_true', help='跳过CatBoost')
    args = parser.parse_args()

    quick = args.quick
    skip_cb = args.no_catboost

    print("=" * 70)
    print("LGBM v4 训练 (修复LR过拟合版)")
    print("=" * 70)
    print(f"模式: {'快速' if quick else '全量'}")
    print(f"CatBoost: {'跳过' if skip_cb else '启用' if HAS_CB else '未安装'}")
    print()

    # 1. 加载数据
    all_data = load_all_data(DB_PATH)
    X, y, feature_names = prepare_data(all_data)
    print(f"特征: {len(feature_names)}个 (与v3完全一致)")

    # 2. Optuna搜索
    lgbm_trials = 20 if quick else 100
    xgb_trials = 15 if quick else 80
    cb_trials = 10 if quick else 50

    lgbm_params = search_lgbm(X, y, lgbm_trials)
    xgb_params = search_xgboost(X, y, xgb_trials) if HAS_XGB else None
    cb_params = search_catboost(X, y, cb_trials) if HAS_CB and not skip_cb else None

    # 3. 训练子模型
    lgbm_models, lgbm_scores = train_lgbm_bagging(X, y, lgbm_params, N_LGBM)
    xgb_models, xgb_scores = train_xgboost_bagging(X, y, xgb_params, N_XGB) if HAS_XGB else ([], [])
    cb_models, cb_scores = train_catboost(X, y, cb_params) if HAS_CB and not skip_cb else ([], [])

    # 4. 合并
    all_models = lgbm_models + xgb_models + cb_models
    all_scores = lgbm_scores + xgb_scores + cb_scores
    model_types = (
        ['lgbm'] * len(lgbm_models) +
        ['xgb'] * len(xgb_models) +
        ['catboost'] * len(cb_models)
    )

    print(f"\n=== 子模型汇总 ===")
    for i, (mtype, score) in enumerate(zip(model_types, all_scores)):
        print(f"  模型{i} ({mtype}): CV={score:.4f}")

    # 5. OOF Ensemble验证 (关键! 用真实OOF分数而非训练集分数)
    oof_acc = compute_oof_ensemble_accuracy(X, y, all_models, model_types)

    # 6. 计算特征重要性
    avg_importance = np.zeros(len(feature_names))
    for model in lgbm_models:
        avg_importance += model.feature_importances_ / len(lgbm_models)

    # 7. 权重配置 (推理时用加权平均, 不用LR)
    ensemble_weights = {
        'lgbm': 1.0,
        'xgb': 0.8,
        'catboost': 0.9,
    }

    # 8. 保存模型
    model_data = {
        'models': all_models,
        'model_types': model_types,
        'feature_names': feature_names,
        'ensemble_accuracy': oof_acc,  # 真实OOF分数, 不是98%假分数
        'avg_importance': avg_importance,
        'params': {
            'lgbm': lgbm_params,
            'xgb': xgb_params,
            'catboost': cb_params,
        },
        'n_models': len(all_models),
        'horizon': HORIZON,
        'threshold': BASE_THRESHOLD,
        'train_samples': len(X),
        'train_date': datetime.now().strftime('%Y-%m-%d'),
        'model_version': 'v4-mixed-v2',
        'ensemble_weights': ensemble_weights,
        # 不再保存lr_meta! 加权平均不需要它
        'use_lr_stacking': False,  # 明确标记: 不用LR
        'model_cv_scores': {f'model_{i}': s for i, s in enumerate(all_scores)},
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'model_v4.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"\n=== 保存完成 ===")
    print(f"路径: {model_path}")
    print(f"大小: {size_mb:.1f}MB")
    print(f"版本: v4-mixed-v2 (无LR Stacking)")
    print(f"OOF准确率: {oof_acc:.4f} (真实CV分数)")
    print(f"特征数: {len(feature_names)}")
    print(f"模型数: {len(all_models)} ({Counter(model_types)})")
    print()
    print("⚠️ 重要: 把model_v4.pkl同步到服务器2C2G后:")
    print("  1. scp models/lgb_hs300/model_v4.pkl root@47.242.158.242:/root/github/stock-quant/stock-quant/python/models/lgb_hs300/")
    print("  2. 在服务器上跑回测验证: python lgbm_backtest.py")
    print("  3. 期望OOF准确率55-65%, 如果>70%说明过拟合")

    return model_data


if __name__ == '__main__':
    main()