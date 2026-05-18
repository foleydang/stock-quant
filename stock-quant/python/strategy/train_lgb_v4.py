#!/usr/bin/env python3
"""
LGBM v4 - 混合模型训练 (需要在强机器上运行)

训练要求: 8C16G以上, Python 3.10+
推理要求: 2C2G即可运行

优化点:
1. LGBM + XGBoost + CatBoost 混合 ensemble (5模型)
2. Optuna 100次超参搜索
3. 5折时序交叉验证
4. 特征筛选 (去掉重要性<20的特征)
5. 自适应阈值
6. 输出单个pkl, 推理内存<50MB

使用方法:
  python train_lgb_v4.py          # 全量训练
  python train_lgb_v4.py --quick  # 快速模式(减少搜索)
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
from collections import Counter
from sklearn.linear_model import LogisticRegression

# 尝试导入XGBoost和CatBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠ XGBoost未安装. 安装: pip install xgboost")

try:
    from catboost import CatBoostClassifier
    HAS_CB = True
except ImportError:
    HAS_CB = False
    print("⚠ CatBoost未安装. 安装: pip install catboost")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("⚠ Optuna未安装. 安装: pip install optuna")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy.train_lgb_enhanced import EnhancedFeatureEngineer
from strategy.train_lgb_v3 import (
    AdvancedFeatureEngineer, calculate_target_adaptive,
    TIME_FEATURES, ZERO_IMP_FEATURES
)

# ============ 配置 ============
HORIZON = 3
BASE_THRESHOLD = 0.018
N_ENSEMBLE = 5  # LGBM3 + XGB1 + CB1

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/stock_data.db')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/lgb_hs300')


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


def prepare_data(all_data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """准备训练数据"""
    sample = list(all_data.values())[0]
    sample_features = compute_all_features(sample)
    feature_names = list(sample_features.columns)
    print(f"总特征数: {len(feature_names)}")

    all_X = []
    all_y = []
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
            if i < 5:
                print(f"  {symbol}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  已处理 {i + 1}/{len(all_data)}")

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    print(f"总样本: {len(X)}")
    print(f"  上涨: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
    print(f"  下跌: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    return X, y, feature_names


def feature_selection(X: np.ndarray, y: np.ndarray, feature_names: List[str], threshold: int = 20) -> Tuple[np.ndarray, List[str]]:
    """特征选择: 基于初始LGBM的重要性"""
    print("\n=== 特征选择 ===")
    base_model = lgb.LGBMClassifier(
        objective='binary', metric='binary_logloss',
        num_leaves=31, max_depth=6, n_estimators=200,
        min_child_samples=50, feature_fraction=0.7,
        verbose=-1, random_state=42, n_jobs=-1
    )
    base_model.fit(X, y)
    importances = base_model.feature_importances_

    keep_indices = [i for i in range(len(feature_names)) if importances[i] >= threshold]
    selected_features = [feature_names[i] for i in keep_indices]
    X_selected = X[:, keep_indices]

    removed = len(feature_names) - len(selected_features)
    print(f"  原始: {len(feature_names)}特征")
    print(f"  保留: {len(selected_features)}特征 (重要性>= {threshold})")
    print(f"  去除: {removed}特征")

    # 显示被去除的特征
    removed_names = [feature_names[i] for i in range(len(feature_names)) if importances[i] < threshold]
    if removed_names:
        print(f"  去除列表: {removed_names[:10]}{'...' if len(removed_names) > 10 else ''}")

    return X_selected, selected_features


# ============ Optuna搜索 ============
def optimize_lgbm(X, y, n_trials=100):
    """Optuna搜索LGBM参数"""
    if not HAS_OPTUNA:
        return {
            'objective': 'binary', 'metric': 'binary_logloss',
            'num_leaves': 31, 'max_depth': 7, 'learning_rate': 0.02,
            'n_estimators': 500, 'min_child_samples': 50,
            'feature_fraction': 0.7, 'bagging_fraction': 0.7,
            'bagging_freq': 5, 'reg_alpha': 0.5, 'reg_lambda': 0.5,
            'verbose': -1, 'random_state': 42, 'n_jobs': -1,
        }

    print(f"\n=== Optuna LGBM ({n_trials}次) ===")
    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
            'n_estimators': 500,
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
            'bagging_freq': 5,
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'verbose': -1, 'random_state': 42, 'n_jobs': -1,
        }

        scores = []
        for train_idx, test_idx in tscv.split(X):
            model = lgb.LGBMClassifier(**params)
            model.fit(X[train_idx], y[train_idx],
                      eval_set=[(X[test_idx], y[test_idx])],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
            scores.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best.update({
        'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
        'n_estimators': 500, 'bagging_freq': 5,
        'verbose': -1, 'random_state': 42, 'n_jobs': -1,
    })
    print(f"最优LGBM CV: {study.best_value:.4f}")
    return best


def optimize_xgboost(X, y, n_trials=50):
    """Optuna搜索XGBoost参数"""
    if not HAS_XGB:
        return None

    print(f"\n=== Optuna XGBoost ({n_trials}次) ===")
    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'logloss',
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
            'n_estimators': 500,
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'random_state': 42, 'n_jobs': -1, 'verbosity': 0,
        }

        scores = []
        for train_idx, test_idx in tscv.split(X):
            model = xgb.XGBClassifier(**params)
            model.fit(X[train_idx], y[train_idx],
                      eval_set=[(X[test_idx], y[test_idx])],
                      verbose=False)
            scores.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best.update({
        'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'n_estimators': 500, 'random_state': 42, 'n_jobs': -1, 'verbosity': 0,
    })
    print(f"最优XGB CV: {study.best_value:.4f}")
    return best


def optimize_catboost(X, y, n_trials=30):
    """Optuna搜索CatBoost参数"""
    if not HAS_CB:
        return None

    print(f"\n=== Optuna CatBoost ({n_trials}次) ===")
    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            'loss_function': 'Logloss',
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
            'iterations': 500,
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_seed': 42,
            'verbose': False,
        }

        scores = []
        for train_idx, test_idx in tscv.split(X):
            model = CatBoostClassifier(**params)
            model.fit(X[train_idx], y[train_idx],
                      eval_set=(X[test_idx], y[test_idx]),
                      early_stopping_rounds=30, verbose=False)
            scores.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best.update({
        'loss_function': 'Logloss', 'iterations': 500,
        'random_seed': 42, 'verbose': False,
    })
    print(f"最优CatBoost CV: {study.best_value:.4f}")
    return best


# ============ 训练ensemble ============
def train_mixed_ensemble(X, y, feature_names, lgb_params, xgb_params, cb_params):
    """训练混合ensemble: 3 LGBM + 1 XGB + 1 CatBoost"""
    print(f"\n=== 训练混合ensemble ===")

    tscv = TimeSeriesSplit(n_splits=5)
    all_models = []
    model_types = []

    # 3个LGBM (不同种子+微调参数)
    for i in range(3):
        params = lgb_params.copy()
        params['random_state'] = 42 + i * 7
        params['feature_fraction'] = min(0.9, lgb_params['feature_fraction'] + (i % 3) * 0.05)

        print(f"\nLGBM {i+1}/3:")
        cv_scores = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            model = lgb.LGBMClassifier(**params)
            model.fit(X[train_idx], y[train_idx],
                      eval_set=[(X[test_idx], y[test_idx])],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)])
            cv_scores.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))
            print(f"  Fold {fold+1}: {cv_scores[-1]:.4f}")

        print(f"  平均: {np.mean(cv_scores):.4f}")

        # 用最佳迭代训练最终模型
        best_iter = int(np.mean([m.best_iteration_ for m in [lgb.LGBMClassifier(**params).fit(
            X[tscv.split(X)[-1][0]], y[tscv.split(X)[-1][0]])]]))
        final_params = params.copy()
        final_params['n_estimators'] = 500  # 保留完整迭代, 推理时用全部
        final_model = lgb.LGBMClassifier(**final_params)
        final_model.fit(X, y)

        all_models.append(final_model)
        model_types.append('lgbm')

    # 1个XGBoost
    if HAS_XGB and xgb_params:
        print(f"\nXGBoost:")
        cv_scores = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X[train_idx], y[train_idx],
                      eval_set=[(X[test_idx], y[test_idx])],
                      verbose=False)
            cv_scores.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))
            print(f"  Fold {fold+1}: {cv_scores[-1]:.4f}")

        print(f"  平均: {np.mean(cv_scores):.4f}")

        final_model = xgb.XGBClassifier(**xgb_params)
        final_model.fit(X, y)
        all_models.append(final_model)
        model_types.append('xgb')

    # 1个CatBoost
    if HAS_CB and cb_params:
        print(f"\nCatBoost:")
        cv_scores = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            model = CatBoostClassifier(**cb_params)
            model.fit(X[train_idx], y[train_idx],
                      eval_set=(X[test_idx], y[test_idx]),
                      early_stopping_rounds=50, verbose=False)
            cv_scores.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))
            print(f"  Fold {fold+1}: {cv_scores[-1]:.4f}")

        print(f"  平均: {np.mean(cv_scores):.4f}")

        final_model = CatBoostClassifier(**cb_params)
        final_model.fit(X, y, verbose=False)
        all_models.append(final_model)
        model_types.append('catboost')

    # ============ Ensemble评估 ============
    print(f"\n=== Ensemble评估 ({len(all_models)}模型) ===")

    # 投票预测
    all_preds = []
    for model in all_models:
        all_preds.append(model.predict(X))

    vote_preds = np.array(all_preds).T
    ensemble_pred = np.apply_along_axis(
        lambda x: Counter(x).most_common(1)[0][0], axis=1, arr=vote_preds
    )
    ensemble_acc = accuracy_score(y, ensemble_pred)
    print(f"  集成投票准确率: {ensemble_acc:.4f}")

    # 概率平均
    all_probs = []
    for model in all_models:
        all_probs.append(model.predict_proba(X)[:, 1])
    avg_probs = np.mean(all_probs, axis=0)
    avg_pred = (avg_probs > 0.5).astype(int)
    avg_acc = accuracy_score(y, avg_pred)
    print(f"  概率平均准确率: {avg_acc:.4f}")

    print(classification_report(y, ensemble_pred, target_names=['下跌', '上涨']))

    # 特征重要性 (只统计LGBM的)
    lgb_models = [m for m, t in zip(all_models, model_types) if t == 'lgbm']
    avg_imp = np.mean([m.feature_importances_ for m in lgb_models], axis=0)
    top_idx = np.argsort(avg_imp)[::-1][:15]
    print(f"\nTop 15 重要特征:")
    for i in top_idx:
        print(f"  {feature_names[i]}: {avg_imp[i]:.0f}")

    return all_models, model_types, ensemble_acc, avg_imp


def save_model(all_models, model_types, feature_names, ensemble_acc, avg_imp,
               lgb_params, xgb_params, cb_params, model_dir):
    """保存模型 - 单个pkl文件"""
    os.makedirs(model_dir, exist_ok=True)

    model_data = {
        'models': all_models,
        'model_types': model_types,
        'ensemble_accuracy': ensemble_acc,
        'feature_names': feature_names,
        'avg_importance': avg_imp,
        'params': {
            'lgbm': lgb_params,
            'xgb': xgb_params,
            'catboost': cb_params,
        },
        'n_models': len(all_models),
        'horizon': HORIZON,
        'threshold': BASE_THRESHOLD,
        'train_samples': 0,  # will be filled
        'train_date': datetime.now().strftime('%Y-%m-%d'),
        'model_version': 'v4-mixed',
    }

    with open(os.path.join(model_dir, 'model_v4.pkl'), 'wb') as f:
        pickle.dump(model_data, f)

    fsize = os.path.getsize(os.path.join(model_dir, 'model_v4.pkl'))
    print(f"\n✅ 模型已保存到 {model_dir}")
    print(f"   model_v4.pkl 大小: {fsize/1024:.1f} KB")
    print(f"   推理内存预估: ~{(len(all_models) * 10 + 5)}MB (2G服务器可运行)")
    print(f"   模型构成: {Counter(model_types)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='快速模式(减少Optuna搜索)')
    parser.add_argument('--db', type=str, default=DB_PATH, help='数据库路径')
    args = parser.parse_args()

    n_trials_lgb = 20 if args.quick else 100
    n_trials_xgb = 10 if args.quick else 50
    n_trials_cb = 5 if args.quick else 30

    print("=" * 60)
    print("LGBM v4 - 混合模型训练")
    print(f"模式: {'快速' if args.quick else '全量'}")
    print(f"LGBM搜索: {n_trials_lgb}次, XGB搜索: {n_trials_xgb}次, CB搜索: {n_trials_cb}次")
    print("=" * 60)

    # 加载数据
    data = load_all_data(args.db)
    X, y, feature_names = prepare_data(data)
    model_data_samples = len(X)

    # 特征选择
    X_selected, selected_features = feature_selection(X, y, feature_names, threshold=20)

    # Optuna搜索
    lgb_params = optimize_lgbm(X_selected, y, n_trials=n_trials_lgb)
    xgb_params = optimize_xgboost(X_selected, y, n_trials=n_trials_xgb) if HAS_XGB else None
    cb_params = optimize_catboost(X_selected, y, n_trials=n_trials_cb) if HAS_CB else None

    # 训练混合ensemble
    all_models, model_types, ensemble_acc, avg_imp = train_mixed_ensemble(
        X_selected, y, selected_features, lgb_params, xgb_params, cb_params
    )

    # 保存
    # avg_imp是LGBM的, 需要扩展到selected_features长度
    if len(avg_imp) != len(selected_features):
        avg_imp = np.zeros(len(selected_features))

    model_data = {
        'models': all_models,
        'model_types': model_types,
        'ensemble_accuracy': ensemble_acc,
        'feature_names': selected_features,
        'avg_importance': avg_imp,
        'params': {
            'lgbm': lgb_params,
            'xgb': xgb_params,
            'catboost': cb_params,
        },
        'n_models': len(all_models),
        'horizon': HORIZON,
        'threshold': BASE_THRESHOLD,
        'train_samples': model_data_samples,
        'train_date': datetime.now().strftime('%Y-%m-%d'),
        'model_version': 'v4-mixed',
    }

    model_dir = MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'model_v4.pkl'), 'wb') as f:
        pickle.dump(model_data, f)

    fsize = os.path.getsize(os.path.join(model_dir, 'model_v4.pkl'))
    print(f"\n✅ 模型已保存到 {model_dir}/model_v4.pkl")
    print(f"   大小: {fsize/1024:.1f} KB")
    print(f"   模型: {Counter(model_types)}")
    print(f"   特征: {len(selected_features)}")
    print(f"   阈值: {BASE_THRESHOLD}")
    print(f"   Horizon: {HORIZON}")
    print()
    print("📋 同步到生产服务器:")
    print("   scp model_v4.pkl root@47.242.158.242:/root/github/stock-quant/stock-quant/python/models/lgb_hs300/model_v4.pkl")
    print("   然后在生产服务器上修改API/回测代码加载model_v4.pkl")


if __name__ == '__main__':
    main()