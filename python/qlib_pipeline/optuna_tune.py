#!/usr/bin/env python3
"""
Optuna 超参调优 — 搜 LightGBM 最优参数
用法: python qlib_pipeline/optuna_tune.py [--trials 100]
"""
import os, sys, json, pickle, argparse, warnings
from datetime import datetime
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config_loader import get_db_path

DB_PATH = get_db_path()
CACHE_FILE = os.path.join(os.path.dirname(DB_PATH), 'features_cache_v3.parquet')
OUTPUT_DIR = os.path.join(ROOT, 'models', 'lgb_daily')


def load_dataset():
    df = pd.read_parquet(CACHE_FILE)
    y = df.pop('__label__').values
    X = df.values.astype(np.float32)
    n = len(X)
    train_end = int(n * 0.7)
    return X[:train_end], y[:train_end], X[train_end:], y[train_end:]


def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }

    model = lgb.LGBMRegressor(**params)

    # StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model.fit(
        X_train_s, y_train,
        eval_set=[(X_val_s, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    y_pred = model.predict(X_val_s)
    rank_ic, _ = spearmanr(y_pred, y_val)
    return rank_ic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--output', default=OUTPUT_DIR)
    args = parser.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("📡 加载数据...")
    X_train, y_train, X_val, y_val = load_dataset()
    print(f"  训练: {len(X_train):,} | 验证: {len(X_val):,} | 特征: {X_train.shape[1]}")

    print(f"🔍 Optuna 搜参 ({args.trials} trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    print(f"\n🏆 最佳: RankIC={study.best_value:.4f}")
    print(f"   参数: {json.dumps(study.best_params, indent=2)}")

    # 用最佳参数训练最终模型
    best = study.best_params
    print("\n🤖 训练最终模型...")
    model = lgb.LGBMRegressor(
        objective='regression', metric='rmse', boosting_type='gbdt',
        random_state=42, n_jobs=-1, verbose=-1, **best,
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model.fit(
        X_train_s, y_train,
        eval_set=[(X_val_s, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )

    y_pred = model.predict(X_val_s)
    rank_ic, _ = spearmanr(y_pred, y_val)
    ic = np.corrcoef(y_pred, y_val)[0, 1]

    print(f"\n  IC={ic:.4f}, RankIC={rank_ic:.4f}")

    # 保存模型 (使用 sklearn Pipeline 兼容推理)
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([('scaler', scaler), ('model', model)])

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, 'model.pkl'), 'wb') as f:
        pickle.dump(pipeline, f)
    model.booster_.save_model(os.path.join(args.output, 'model.txt'))

    with open(os.path.join(args.output, 'feature_names.json')) as f:
        feature_names = json.load(f)

    meta = {
        'model': 'LightGBM-Optuna',
        'horizon': feature_names.get('horizon', 5),
        'label': 'cs_rank_5d',
        'features': feature_names.get('features', X_train.shape[1]),
        'IC': round(float(ic), 4),
        'RankIC': round(float(rank_ic), 4),
        'best_params': best,
        'optuna_trials': args.trials,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'timestamp': datetime.now().isoformat(),
    }
    with open(os.path.join(args.output, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ 模型已导出: {args.output}")
    print(f"   RankIC: {meta['RankIC']} (vs 之前 0.11)")


if __name__ == '__main__':
    main()