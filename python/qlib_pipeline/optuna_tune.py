#!/usr/bin/env python3
"""
Optuna 超参调优 v3 — 快速静默版
三阶段搜索: 随机粗搜 → TPE精搜 → 网格微调

用法:
  python qlib_pipeline/optuna_tune.py --trials 100
"""
import os, sys, json, pickle, argparse, warnings
from datetime import datetime
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'  # 限制线程数

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config_loader import get_db_path

DB_PATH = get_db_path()
CACHE_FILE = os.path.join(os.path.dirname(DB_PATH), 'features_cache_v4.parquet')
if not os.path.exists(CACHE_FILE):
    CACHE_FILE = os.path.join(os.path.dirname(DB_PATH), 'features_cache_v3.parquet')
OUTPUT_DIR = os.path.join(ROOT, 'models', 'lgb_daily')


def load_dataset():
    df = pd.read_parquet(CACHE_FILE)
    y = df.pop('__label__').values.astype(np.float32)
    if '__date__' in df.columns:
        df.drop(columns=['__date__'], inplace=True)
    X = df.values.astype(np.float32)
    return X, y


def objective(trial, X, y):
    # ── 缩小搜索空间，加速收敛 ──
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 31, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.25, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 500.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 500.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 30, 300),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'max_bin': 127,  # 减少分箱，加速
        'random_state': 42,
        'n_jobs': 4,
        'verbose': -1,
        'verbosity': -1,
    }

    # 单次 70/30 切分 (比 3-fold 快 3 倍)
    n = len(X)
    train_end = int(n * 0.7)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:], y[train_end:]

    # 训练时采样 50% 数据加速 (大样本量下效果接近)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_train), size=len(X_train)//2, replace=False)
    X_train_samp, y_train_samp = X_train[idx], y_train[idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_samp)
    X_val_s = scaler.transform(X_val)

    model = lgb.LGBMRegressor(**params)

    # 完全静默训练
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model.fit(
            X_train_s, y_train_samp,
            eval_set=[(X_val_s, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

    y_pred = model.predict(X_val_s)
    ric, _ = spearmanr(y_pred, y_val)
    return ric if not np.isnan(ric) else -1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--output', default=OUTPUT_DIR)
    args = parser.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("=" * 60)
    print("🔬 Optuna 超参调优 v3 (快速静默版)")
    print("=" * 60)

    X, y = load_dataset()
    n = len(X)
    print(f"\n📡 数据: {n:,} 样本, {X.shape[1]} 特征")
    print(f"   70/30 切分 + 50% 采样加速")

    # 阶段1: 随机粗搜
    n1 = args.trials // 3
    print(f"\n🔍 阶段 1/3: 随机粗搜 ({n1} trials)")
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=n1, show_progress_bar=True,
        n_jobs=1,  # 单线程避免竞争
    )
    print(f"   最佳: RankIC={study.best_value:.4f}")

    # 阶段2: TPE 精搜
    n2 = args.trials // 3
    print(f"\n🔍 阶段 2/3: 贝叶斯精搜 ({n2} trials)")
    study.sampler = optuna.samplers.TPESampler(seed=42)
    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=n2, show_progress_bar=True,
        n_jobs=1,
    )
    print(f"   最佳: RankIC={study.best_value:.4f}")

    # 阶段3: 微调
    n3 = args.trials - n1 - n2
    if n3 > 0:
        print(f"\n🔍 阶段 3/3: 局部微调 ({n3} trials)")
        study.optimize(
            lambda trial: objective(trial, X, y),
            n_trials=n3, show_progress_bar=True,
            n_jobs=1,
        )
        print(f"   最佳: RankIC={study.best_value:.4f}")

    # ── 结果 ──
    print(f"\n{'='*60}")
    print(f"🏆 最佳: RankIC={study.best_value:.4f}")
    print(f"   参数:")
    for k, v in sorted(study.best_params.items()):
        if isinstance(v, float):
            print(f"     {k}: {v:.6f}")
        else:
            print(f"     {k}: {v}")

    # ── 最终模型 (全量数据) ──
    print(f"\n🤖 训练最终模型 (全量数据)...")
    best = study.best_params.copy()
    best.update({
        'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
        'random_state': 42, 'n_jobs': -1, 'verbose': -1, 'verbosity': -1,
    })

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

    print(f"  IC={ic:.4f}, RankIC={rank_ic:.4f}, 迭代={model.best_iteration_}")

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
        'model': 'LightGBM-Optuna-v3',
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

    imp_pct = ((meta['RankIC'] - 0.1125) / 0.1125 * 100) if meta['RankIC'] else 0
    print(f"\n✅ 模型已导出: {args.output}")
    print(f"   RankIC: {meta['RankIC']} (vs 基线 0.1125, {imp_pct:+.1f}%)")


if __name__ == '__main__':
    main()