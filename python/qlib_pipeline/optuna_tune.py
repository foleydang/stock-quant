#!/usr/bin/env python3
"""
Optuna 超参调优 v2 — 对标 Qlib 基准 + 适配短数据
三阶段搜索: coarse → bayesian → fine

用法:
  python qlib_pipeline/optuna_tune.py --trials 100
  python qlib_pipeline/optuna_tune.py --trials 200 --study-db sqlite:///optuna_study.db
"""
import os, sys, json, pickle, argparse, warnings
from datetime import datetime
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config_loader import get_db_path

DB_PATH = get_db_path()
CACHE_FILE = os.path.join(os.path.dirname(DB_PATH), 'features_cache_v3.parquet')
OUTPUT_DIR = os.path.join(ROOT, 'models', 'lgb_daily')


def load_dataset():
    """加载特征缓存，返回 numpy 数组"""
    df = pd.read_parquet(CACHE_FILE)
    y = df.pop('__label__').values.astype(np.float32)
    if '__date__' in df.columns:
        df.drop(columns=['__date__'], inplace=True)
    X = df.values.astype(np.float32)
    return X, y


def create_splits(X, y, n_splits=3):
    """时序交叉验证切分"""
    n = len(X)
    splits = []
    for fold in range(n_splits):
        # 每个 fold: 前 fold*30%+40% 训练, 后 30% 验证
        train_end = int(n * (0.4 + fold * 0.2))
        val_end = int(n * (0.7 + fold * 0.1))
        splits.append((
            slice(0, train_end),
            slice(train_end, min(val_end, n))
        ))
    return splits


def objective(trial, X, y, splits):
    """Optuna 目标函数: 最大化平均 RankIC"""
    # ── 搜索空间 (对标 Qlib + 适配短数据) ──
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 31, 512, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 200, 3000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1000.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1000.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 500, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }

    # ── 时序交叉验证 ──
    rank_ics = []
    for train_slice, val_slice in splits:
        X_train, y_train = X[train_slice], y[train_slice]
        X_val, y_val = X[val_slice], y[val_slice]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train_s, y_train,
            eval_set=[(X_val_s, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(100)],
        )

        y_pred = model.predict(X_val_s)
        ric, _ = spearmanr(y_pred, y_val)
        if not np.isnan(ric):
            rank_ics.append(ric)

    return np.mean(rank_ics) if rank_ics else -1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--output', default=OUTPUT_DIR)
    parser.add_argument('--study-db', default=None,
                        help='Optuna 持久化存储 (sqlite:///optuna.db)')
    args = parser.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("=" * 60)
    print("🔬 Optuna 超参调优 v2")
    print("=" * 60)

    # 加载数据
    print("\n📡 加载数据...")
    X, y = load_dataset()
    splits = create_splits(X, y, n_splits=3)
    for i, (ts, vs) in enumerate(splits):
        print(f"  Fold {i+1}: train={ts.stop-ts.start:,} val={vs.stop-vs.start:,}")
    print(f"  特征: {X.shape[1]}")

    # 创建 study
    study_name = f"lgb_daily_{datetime.now().strftime('%Y%m%d_%H%M')}"
    if args.study_db:
        storage = optuna.storages.RDBStorage(args.study_db)
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            storage=storage,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
        )

    # ── 阶段 1: 粗搜 (随机采样) ──
    n_coarse = args.trials // 3
    print(f"\n🔍 阶段 1/3: 粗搜 ({n_coarse} trials, RandomSampler)")
    study.sampler = optuna.samplers.RandomSampler(seed=42)
    study.optimize(
        lambda trial: objective(trial, X, y, splits),
        n_trials=n_coarse,
        show_progress_bar=True,
    )

    # ── 阶段 2: 精搜 (TPE) ──
    n_bayes = args.trials // 3
    print(f"\n🔍 阶段 2/3: 贝叶斯优化 ({n_bayes} trials, TPESampler)")
    study.sampler = optuna.samplers.TPESampler(seed=42)
    study.optimize(
        lambda trial: objective(trial, X, y, splits),
        n_trials=n_bayes,
        show_progress_bar=True,
    )

    # ── 阶段 3: 微调 (局部搜索) ──
    n_fine = args.trials - n_coarse - n_bayes
    if n_fine > 0:
        print(f"\n🔍 阶段 3/3: 局部微调 ({n_fine} trials, TPESampler)")
        study.optimize(
            lambda trial: objective(trial, X, y, splits),
            n_trials=n_fine,
            show_progress_bar=True,
        )

    # ── 结果 ──
    print(f"\n{'='*60}")
    print(f"🏆 最佳结果: RankIC={study.best_value:.4f}")
    print(f"   参数:")
    for k, v in sorted(study.best_params.items()):
        if isinstance(v, float):
            print(f"     {k}: {v:.6f}")
        else:
            print(f"     {k}: {v}")

    # ── 用最佳参数训练最终模型 ──
    print(f"\n🤖 训练最终模型...")
    best = study.best_params.copy()
    best.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    })

    # 最终用 70/30 单次切分训练
    n = len(X)
    train_end = int(n * 0.7)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:], y[train_end:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = lgb.LGBMRegressor(**best)
    model.fit(
        X_train_s, y_train,
        eval_set=[(X_val_s, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)],
    )

    y_pred = model.predict(X_val_s)
    rank_ic, _ = spearmanr(y_pred, y_val)
    ic = np.corrcoef(y_pred, y_val)[0, 1]

    print(f"\n  IC={ic:.4f}, RankIC={rank_ic:.4f}")
    print(f"  迭代: {model.best_iteration_}")

    # 特征重要性
    importance = model.feature_importances_
    with open(os.path.join(OUTPUT_DIR, 'feature_names.json')) as f:
        fn = json.load(f)
    feature_names = fn.get('features', [f'f{i}' for i in range(X.shape[1])])
    top_idx = np.argsort(importance)[-15:][::-1]
    print(f"  Top-15 特征:")
    for idx in top_idx:
        if idx < len(feature_names):
            print(f"    {feature_names[idx]:20s}: {importance[idx]:.0f}")

    # 保存
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([('scaler', scaler), ('model', model)])

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, 'model.pkl'), 'wb') as f:
        pickle.dump(pipeline, f)
    model.booster_.save_model(os.path.join(args.output, 'model.txt'))

    meta = {
        'model': 'LightGBM-Optuna-v2',
        'horizon': fn.get('horizon', 5),
        'label': 'cs_rank_5d',
        'features': X.shape[1],
        'IC': round(float(ic), 4),
        'RankIC': round(float(rank_ic), 4),
        'best_params': {k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in study.best_params.items()},
        'optuna_trials': args.trials,
        'optuna_best_rankic': round(float(study.best_value), 4),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'timestamp': datetime.now().isoformat(),
    }
    with open(os.path.join(args.output, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 模型已导出: {args.output}")
    print(f"   RankIC: {meta['RankIC']} (vs 之前 0.1174)")
    print(f"   提升: {((meta['RankIC'] - 0.1174) / 0.1174 * 100):+.1f}%")


if __name__ == '__main__':
    main()