#!/usr/bin/env python3
"""
分钟级择时模型训练脚本 v1 — LGBM + 时序CV

架构:
  分钟级模型 → 预测未来N根K线的收益率 (γ择时层)
  仅在日线模型选出的股票池内运行

关键设计:
  1. 目标: 未来N根K线收益率 (回归) + 涨跌方向 (分类评估)
  2. 时序CV: 严格按时间切分, 不shuffle
  3. 下采样: 每隔skip_bars根取一个样本, 减少自相关
  4. 样本过滤: 排除涨跌停、停牌、极端收益率
  5. 评估: Hit Ratio + MSE + 分组回测

用法:
  python strategy/intraday_train.py                          # 默认配置
  python strategy/intraday_train.py --horizon 3 --skip 5    # 自定义参数
  python strategy/intraday_train.py --quick                 # 快速验证
  python strategy/intraday_train.py --pool-size 50          # 股票池大小
"""

import sys, os, argparse, pickle, json, sqlite3, warnings, time

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, accuracy_score
from joblib import Parallel, delayed

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kw): return iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategy.intraday_features import IntradayFeaturePipeline

# ============ 路径 ============
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data/stock_data.db')
MODEL_DIR = os.path.join(ROOT, 'models/lgb_intraday')
DAILY_MODEL_PATH = os.path.join(ROOT, 'models/lgb_daily/model.pkl')

# ============ 配置 ============
DB_TABLE = 'kline_30m'
MIN_HISTORY_BARS = 100

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

HORIZON = 3
SKIP_BARS = 5
RETURN_CLIP = 0.10
MIN_VOLUME = 1000
MIN_POOL_SIZE = 20

# 训练参数 v3 (分钟级: 更多特征+更低lr → 大树量模型 ~100MB)
# 核心逻辑: 信噪比低 → 极低lr让模型慢慢学, 弱正则保留弱特征信号
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'l2',
    'boosting_type': 'gbdt',
    'num_leaves': 127,              # 63→127, 更多叶子捕获非线性
    'max_depth': 9,                 # 7→9, 允许更深树
    'learning_rate': 0.001,         # 0.005→0.001, 极低lr训练5000+棵树
    'n_estimators': 20000,
    'early_stopping_rounds': 500,   # 150→500, 给弱信号更多耐心
    'subsample': 0.6,
    'subsample_freq': 1,
    'colsample_bytree': 0.5,
    'feature_fraction_bynode': 0.6,
    'reg_alpha': 0.1,               # 0.5→0.1, 降低L1正则
    'reg_lambda': 0.5,              # 2.0→0.5, 降低L2正则
    'min_child_samples': 50,        # 100→50, 允许更细分叉
    'min_child_weight': 0.0001,     # 0.001→0.0001, 降低权重门槛
    'min_split_gain': 0.0001,       # 0.001→0.0001, 允许弱特征分裂
    'path_smooth': 10,
    'verbosity': -1,
    'random_state': None,
    'n_jobs': 3,
    'force_row_wise': True,
}

QUICK_PARAMS = {
    'objective': 'regression',
    'metric': 'l2',
    'boosting_type': 'gbdt',
    'num_leaves': 31, 'max_depth': 5, 'learning_rate': 0.05,
    'n_estimators': 500, 'early_stopping_rounds': 30,
    'subsample': 0.6, 'colsample_bytree': 0.5,
    'reg_alpha': 1.0, 'reg_lambda': 5.0,
    'min_child_samples': 200, 'verbosity': -1,
    'n_jobs': 3, 'force_row_wise': True,
}

CORR_THRESHOLD = 0.95  # 0.90→0.95, 放宽去冗余, 保留更多特征

# Bagging Ensemble
N_MODELS = 5
N_JOBS_PARALLEL = 5
SEEDS = [42, 123, 456, 789, 1024]


# ============ 数据加载 ============
def load_intraday_data(db_path: str, table: str, min_bars: int = MIN_HISTORY_BARS) -> Dict[str, pd.DataFrame]:
    conn = sqlite3.connect(db_path)
    symbols = [r[0] for r in conn.execute(f"SELECT DISTINCT symbol FROM {table}")]
    data = {}
    for sym in tqdm(symbols, desc='   加载股票', unit='stock'):
        try:
            df = pd.read_sql(
                f"SELECT * FROM {table} WHERE symbol=? ORDER BY date", conn, params=(sym,))
            if len(df) >= min_bars:
                df['date'] = pd.to_datetime(df['date'], format='mixed')
                df = df.sort_values('date').reset_index(drop=True)
                df = df.drop_duplicates(subset=['date']).reset_index(drop=True)
                if len(df) >= min_bars:
                    data[sym] = df
        except Exception:
            continue
    conn.close()
    print(f"  加载 {len(data)} 只股票, 共 {sum(len(d) for d in data.values()):,} 行 ({table})")
    return data


def get_all_timestamps(data: Dict) -> np.ndarray:
    ts = set()
    for df in data.values():
        ts.update(df['date'].values)
    return np.array(sorted(ts))


def detect_suspended(df: pd.DataFrame) -> np.ndarray:
    close = df['close'].values
    same = np.zeros(len(close), dtype=bool)
    for i in range(3, len(close)):
        if len(set(close[i - 3:i + 1])) == 1:
            same[i] = True
    return same


def detect_limit_prices(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    close = df['close'].values.astype(float)
    ret = pd.Series(close).pct_change().values
    return ret > 0.095, ret < -0.095


# ============ 目标构造 ============
def build_targets(data: Dict[str, pd.DataFrame], horizon: int,
                  return_clip: float = RETURN_CLIP) -> Dict[str, np.ndarray]:
    targets = {}
    n_filtered = {'limit': 0, 'suspended': 0, 'extreme': 0, 'total': 0}

    for sym, df in tqdm(data.items(), desc='   构建目标', unit='stock'):
        close = df['close'].values.astype(float)
        n = len(close)
        ret = np.full(n, np.nan)
        suspended = detect_suspended(df)
        limit_up, limit_down = detect_limit_prices(df)

        for i in range(n - horizon):
            if limit_up[i] or limit_down[i]:
                n_filtered['limit'] += 1
                continue
            if suspended[i]:
                n_filtered['suspended'] += 1
                continue
            future_close = close[i + horizon]
            if future_close <= 0 or close[i] <= 0:
                continue
            r = (future_close - close[i]) / close[i]
            if abs(r) > return_clip:
                n_filtered['extreme'] += 1
                r = np.sign(r) * return_clip
            ret[i] = r
            n_filtered['total'] += 1
        targets[sym] = ret

    print(f"  目标构造: 有效={n_filtered['total']:,} "
          f"涨跌停过滤={n_filtered['limit']:,} "
          f"停牌过滤={n_filtered['suspended']:,} "
          f"极端值截断={n_filtered['extreme']:,}")
    return targets


# ============ 样本准备 ============
def prepare_samples(data: Dict[str, pd.DataFrame],
                    targets: Dict[str, np.ndarray],
                    all_timestamps: np.ndarray,
                    pipeline: IntradayFeaturePipeline,
                    skip_bars: int = SKIP_BARS,
                    pool_size: int = 0) -> Optional[Tuple]:
    n_dates = len(all_timestamps)
    train_cutoff = all_timestamps[int(n_dates * TRAIN_RATIO)]
    val_cutoff = all_timestamps[int(n_dates * (TRAIN_RATIO + VAL_RATIO))]

    print(f"  时序切分: train={str(train_cutoff)[:16]}  "
          f"val={str(val_cutoff)[:16]}  test={str(all_timestamps[-1])[:16]}")

    symbols = list(data.keys())
    if pool_size > 0 and pool_size < len(symbols):
        vol_ranks = {}
        for sym in symbols:
            vol_ranks[sym] = data[sym]['volume'].sum()
        symbols = sorted(vol_ranks, key=vol_ranks.get, reverse=True)[:pool_size]
        print(f"  股票池: {pool_size}/{len(data)} (按成交量筛选)")

    # 第一遍: 计算所有股票特征
    print("  计算分钟级特征...")
    t0 = time.time()
    all_features = {}
    success = 0

    for sym in tqdm(symbols, desc='   个股特征', unit='stock'):
        try:
            df = data[sym]
            feats = pipeline.compute_stock(df, sym)
            feats = feats.fillna(method='ffill').fillna(0)
            feats.index = df['date'].values
            all_features[sym] = feats
            success += 1
        except Exception:
            continue

    print(f"  特征计算: {success}/{len(symbols)} 只股票, 耗时 {time.time()-t0:.0f}s")

    # 第二遍: 截面特征
    print("  计算截面特征...")
    t0 = time.time()
    cs_features = pipeline.compute_cross_section(all_features, all_timestamps)
    print(f"  截面特征: 耗时 {time.time()-t0:.0f}s")

    for sym in all_features:
        if sym in cs_features:
            all_features[sym] = pd.concat([all_features[sym], cs_features[sym]], axis=1)

    # 统一特征列
    all_cols = set()
    for feats in all_features.values():
        all_cols.update(feats.columns)
    feature_names = sorted(all_cols)

    # 第三遍: 构建样本 (下采样)
    X_tr, y_tr, X_va, y_va, X_te, y_te = [], [], [], [], [], []

    for sym in tqdm(symbols, desc='   合并样本', unit='stock'):
        if sym not in all_features or sym not in targets:
            continue

        feats = all_features[sym]
        feats = feats.reindex(columns=feature_names, fill_value=0)
        target = targets[sym]
        timestamps = feats.index.values

        valid_mask = ~np.isnan(target)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) < MIN_HISTORY_BARS:
            continue

        sampled_indices = valid_indices[::skip_bars]

        for idx in sampled_indices:
            ts = timestamps[idx]
            if ts <= train_cutoff:
                X_tr.append(feats.iloc[idx].values)
                y_tr.append(target[idx])
            elif ts <= val_cutoff:
                X_va.append(feats.iloc[idx].values)
                y_va.append(target[idx])
            else:
                X_te.append(feats.iloc[idx].values)
                y_te.append(target[idx])

    if not X_tr:
        print("  ❌ 无有效训练样本")
        return None

    X_train = np.vstack(X_tr).astype(np.float32)
    y_train = np.array(y_tr).astype(np.float32)
    X_val = np.vstack(X_va).astype(np.float32) if X_va else np.array([])
    y_val = np.array(y_va).astype(np.float32) if y_va else np.array([])
    X_test = np.vstack(X_te).astype(np.float32) if X_te else np.array([])
    y_test = np.array(y_te).astype(np.float32) if y_te else np.array([])

    # 极端值裁剪: 防止OBV等累积特征溢出破坏模型
    for X in [X_train, X_val, X_test]:
        if len(X) == 0:
            continue
        X = np.nan_to_num(X, nan=0, posinf=1e6, neginf=-1e6)
        np.clip(X, -1e6, 1e6, out=X)

    print(f"  样本: train={len(X_train):,} | val={len(X_val):,} | test={len(X_test):,}")
    print(f"  特征: {len(feature_names)} | 下采样: 每{skip_bars}根K线取1个")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names


# ============ 特征去冗余 + 预筛选 ============
def prefilter_features(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, X_test: np.ndarray,
                       feature_names: List[str],
                       min_abs_corr: float = 0.0005) -> Tuple:
    """
    去除与目标几乎零相关的噪声特征 (v3: 放宽到0.0005)。
    LGBM能捕捉非线性关系, 线性弱相关不代表无用。
    """
    if len(X_train) < 1000 or len(feature_names) <= 80:
        return X_train, X_val, X_test, feature_names

    # 计算每个特征与target的相关性
    corrs = np.array([abs(np.corrcoef(X_train[:, i], y_train)[0, 1])
                      if np.std(X_train[:, i]) > 1e-10 else 0
                      for i in range(len(feature_names))])
    keep = corrs >= min_abs_corr
    n_removed = sum(~keep)

    if n_removed > 0:
        print(f"  预筛选: 移除 {n_removed} 个弱相关特征 (|corr|<{min_abs_corr}), "
              f"保留 {sum(keep)}/{len(feature_names)}")
        X_train = X_train[:, keep]
        X_val = X_val[:, keep] if len(X_val) > 0 else X_val
        X_test = X_test[:, keep] if len(X_test) > 0 else X_test
        feature_names = [f for f, k in zip(feature_names, keep) if k]

    return X_train, X_val, X_test, feature_names


def remove_redundant(X: np.ndarray, feature_names: List[str]) -> Tuple:
    if X.shape[0] < 1000:
        return X, feature_names, np.ones(len(feature_names), dtype=bool)

    cm = np.corrcoef(X.T)
    rm = set()
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if abs(cm[i, j]) > CORR_THRESHOLD and i not in rm and j not in rm:
                rm.add(j)
    if not rm:
        return X, feature_names, np.ones(len(feature_names), dtype=bool)

    keep = np.ones(len(feature_names), dtype=bool)
    keep[list(rm)] = False
    new_names = [fn for fn, m in zip(feature_names, keep) if m]
    print(f"  去冗余: {sum(keep)}/{len(feature_names)} (移除 {len(rm)} 个)")
    return X[:, keep], new_names, keep


# ============ 训练 ============
def train_one(seed: int, X_train, y_train, X_val, y_val,
              feature_names: List[str], params: dict) -> Dict:
    """训练单个 LGBM 回归器"""
    t0 = time.time()

    p = {**params, 'random_state': seed}
    n_est = p.pop('n_estimators')
    es_rounds = p.pop('early_stopping_rounds')
    p.pop('n_jobs', None)

    model = lgb.LGBMRegressor(n_estimators=n_est, **p)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)] if len(X_val) > 0 else None,
              eval_metric='l2',
              callbacks=[lgb.log_evaluation(50), lgb.early_stopping(es_rounds)])

    n_trees = model.n_estimators_  # 实际训练树数
    elapsed = time.time() - t0

    if len(X_val) > 0:
        val_pred = model.predict(X_val)
        val_ic = spearmanr(y_val, val_pred)[0]
        val_mse = mean_squared_error(y_val, val_pred)
        val_hit = accuracy_score(np.sign(y_val), np.sign(val_pred))
        print(f"  [seed={seed}] {n_trees}棵 | val IC={val_ic:.4f} MSE={val_mse:.6f} Hit={val_hit:.3f} | {elapsed:.0f}s")
    else:
        val_ic, val_mse, val_hit = 0, 0, 0
        print(f"  [seed={seed}] {n_trees}棵 | {elapsed:.0f}s")

    imp = model.feature_importances_
    top_idx = np.argsort(imp)[-5:][::-1]
    top_feats = [(feature_names[i], int(imp[i])) for i in top_idx]

    return {
        'model': model, 'seed': seed, 'n_trees': n_trees,
        'val_ic': round(val_ic, 4), 'val_mse': round(val_mse, 6), 'val_hit': round(val_hit, 4),
        'train_time_s': round(elapsed, 1), 'top_features': top_feats,
    }


def train_ensemble(X_train, y_train, X_val, y_val, feature_names, params,
                   n_models=N_MODELS, seeds=SEEDS, n_jobs=N_JOBS_PARALLEL) -> List[Dict]:
    """Bagging Ensemble: 并行训练多个模型"""
    print(f"\n🏋️ 并行训练 {n_models} 个模型...")
    t0 = time.time()

    results = Parallel(n_jobs=n_jobs)(
        delayed(train_one)(seeds[i], X_train, y_train, X_val, y_val, feature_names, params)
        for i in range(n_models)
    )

    avg_trees = sum(r['n_trees'] for r in results) / len(results)
    avg_ic = sum(r['val_ic'] for r in results) / len(results)
    avg_hit = sum(r['val_hit'] for r in results) / len(results)

    print(f"\n  ✅ {n_models}模型训练完成: 总耗时 {time.time()-t0:.0f}s, "
          f"平均 {avg_trees:.0f}棵/模型, 平均 val IC={avg_ic:.4f}, Hit={avg_hit:.3f}")

    return results


# ============ 评估 ============
def evaluate_ensemble(models_info: List[Dict], X_test, y_test, feature_names):
    """测试集评估: 单模型 + Ensemble"""
    n = len(models_info)
    print(f"\n{'='*60}")
    print(f" 🧪 测试集评估 ({len(X_test):,}条, {n}模型)")
    print(f"{'='*60}")
    print(f"  目标: mean={y_test.mean():.4f} std={y_test.std():.4f} "
          f"min={y_test.min():.4f} max={y_test.max():.4f}")

    all_preds = []
    for info in models_info:
        model = info['model']
        pred = model.predict(X_test)
        all_preds.append(pred)
        ic = spearmanr(y_test, pred)[0]
        mse = mean_squared_error(y_test, pred)
        hit = accuracy_score(np.sign(y_test), np.sign(pred))
        print(f"  [seed={info['seed']}] IC={ic:.4f} | MSE={mse:.6f} | Hit={hit:.3f} | {info['n_trees']}棵")

    # Ensemble
    if n > 1:
        ensemble_pred = np.mean(all_preds, axis=0)
        ic = spearmanr(y_test, ensemble_pred)[0]
        mse = mean_squared_error(y_test, ensemble_pred)
        hit = accuracy_score(np.sign(y_test), np.sign(ensemble_pred))
        print(f"  {'─'*50}")
        print(f"  🏆 Ensemble({n}) → IC={ic:.4f} | MSE={mse:.6f} | Hit={hit:.3f}")
    else:
        ensemble_pred = all_preds[0]
        ic = spearmanr(y_test, ensemble_pred)[0]
        mse = mean_squared_error(y_test, ensemble_pred)
        hit = accuracy_score(np.sign(y_test), np.sign(ensemble_pred))

    # 分组回测
    if n > 1:
        print(f"\n  📊 分组回测 (按预测值分5组):")
        sort_idx = np.argsort(ensemble_pred)
        n_per_group = len(sort_idx) // 5
        for g in range(5):
            start = g * n_per_group
            end = start + n_per_group if g < 4 else len(sort_idx)
            group_actual = y_test[sort_idx[start:end]].mean()
            group_pred = ensemble_pred[sort_idx[start:end]].mean()
            group_hit = accuracy_score(
                np.sign(y_test[sort_idx[start:end]]),
                np.sign(ensemble_pred[sort_idx[start:end]])
            )
            print(f"    G{g+1}: 预测={group_pred:+.4f} | 实际={group_actual:+.4%} | Hit={group_hit:.3f}")

        long_ret = y_test[sort_idx[-n_per_group:]].mean()
        short_ret = y_test[sort_idx[:n_per_group]].mean()
        print(f"    多空收益差: {long_ret - short_ret:+.4%} (买入G5, 卖出G1)")

    # 特征重要性 (取平均)
    print(f"\n  Top 20 特征:")
    avg_imp = np.zeros(len(feature_names))
    for info in models_info:
        avg_imp += info['model'].feature_importances_
    avg_imp /= n
    top = np.argsort(avg_imp)[-20:][::-1]
    for idx in top:
        print(f"    {feature_names[idx]}: {int(avg_imp[idx])}")

    return ic, mse, hit


# ============ 主入口 ============
def main():
    parser = argparse.ArgumentParser(description='分钟级择时模型训练')
    parser.add_argument('--horizon', type=int, default=HORIZON, help='预测未来N根K线')
    parser.add_argument('--skip', type=int, default=SKIP_BARS, help='下采样间隔')
    parser.add_argument('--pool-size', type=int, default=0, help='股票池大小 (0=全部)')
    parser.add_argument('--quick', action='store_true', help='快速验证')
    parser.add_argument('--db', type=str, default=DB_PATH)
    parser.add_argument('--daily-model', type=str, default=DAILY_MODEL_PATH)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    params = QUICK_PARAMS if args.quick else LGBM_PARAMS
    is_quick = args.quick

    print("=" * 60)
    print(f" 分钟级择时模型训练 v1 — LGBM + 时序CV")
    print(f" 目标: 未来{args.horizon}根K线收益率 | 下采样: 每{args.skip}根")
    print(f" 时序: train({TRAIN_RATIO:.0%}) → val({VAL_RATIO:.0%}) → test({TEST_RATIO:.0%})")
    if is_quick:
        print(f" ⚠️ 快速模式: lr={params['learning_rate']} leaves={params['num_leaves']}")
    else:
        print(f" 🚀 生产模式: {N_MODELS}模型 Bagging Ensemble | lr={params['learning_rate']} leaves={params['num_leaves']}")
    print("=" * 60)

    print(f"\n📊 加载数据 ({DB_TABLE})...")
    t0 = time.time()
    data = load_intraday_data(args.db, DB_TABLE)
    all_ts = get_all_timestamps(data)
    print(f"  加载耗时: {time.time()-t0:.0f}s | {len(all_ts)} 个时间戳")

    print(f"\n🎯 构建目标 (horizon={args.horizon})...")
    targets = build_targets(data, args.horizon)

    print(f"\n🔧 特征工程 + 样本准备...")
    t0 = time.time()
    pipeline = IntradayFeaturePipeline(daily_model_path=args.daily_model)
    result = prepare_samples(data, targets, all_ts, pipeline,
                             skip_bars=args.skip, pool_size=args.pool_size)

    if result is None:
        print("❌ 样本准备失败")
        return

    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names = result
    print(f"  准备耗时: {time.time()-t0:.0f}s")

    # 特征预筛选: 去掉与目标零相关的噪声特征
    print(f"\n🔧 特征预筛选...")
    X_train, X_val, X_test, feature_names = prefilter_features(
        X_train, y_train, X_val, X_test, feature_names)

    print(f"\n🔧 特征去冗余...")
    X_train, feature_names, corr_mask = remove_redundant(X_train, feature_names)
    if len(X_val) > 0:
        X_val = X_val[:, corr_mask]
    if len(X_test) > 0:
        X_test = X_test[:, corr_mask]
    print(f"  最终特征: {len(feature_names)}")

    # 训练
    if is_quick:
        models_info = [train_one(args.seed, X_train, y_train, X_val, y_val, feature_names, params)]
    else:
        models_info = train_ensemble(X_train, y_train, X_val, y_val, feature_names, params)

    # 测试集评估
    if len(X_test) > 0:
        test_ic, test_mse, test_hit = evaluate_ensemble(models_info, X_test, y_test, feature_names)
    else:
        test_ic, test_mse, test_hit = 0, 0, 0

    # 最终模型 (全量数据)
    print(f"\n🏋️ 训练最终模型 (train+val 全量, 复用最佳树数)...")
    if len(X_val) > 0:
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
    else:
        X_full = X_train
        y_full = y_train

    final_models = []
    for info in models_info:
        m = lgb.LGBMRegressor(
            **{k: v for k, v in params.items()
               if k not in ('n_estimators', 'early_stopping_rounds', 'n_jobs', 'random_state')},
            n_estimators=info['n_trees'],
            random_state=info['seed']
        )
        m.fit(X_full, y_full)
        final_models.append(m)

    print(f"\n💾 保存模型...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_pkg = {
        'models': final_models,
        'feature_names': feature_names,
        'horizon': args.horizon,
        'skip_bars': args.skip,
        'model_type': 'intraday_timing',
        'n_models': len(models_info),
        'train_date': datetime.now().strftime('%Y-%m-%d'),
        'train_samples': len(X_full),
        'n_trees': [m['n_trees'] for m in models_info],
        'avg_trees': sum(m['n_trees'] for m in models_info) / len(models_info),
        'val_ic': [m['val_ic'] for m in models_info],
        'val_hit': [m['val_hit'] for m in models_info],
        'test_ic': round(test_ic, 4),
        'test_mse': round(test_mse, 6),
        'test_hit': round(test_hit, 4),
        'params': {k: v for k, v in params.items()
                   if k not in ('verbosity', 'random_state', 'force_row_wise',
                                'n_jobs', 'objective', 'metric',
                                'n_estimators', 'early_stopping_rounds')},
    }

    model_path = os.path.join(MODEL_DIR, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_pkg, f)

    size_mb = os.path.getsize(model_path) / 1024 / 1024

    meta = {
        'model_type': 'intraday_timing',
        'horizon': args.horizon,
        'skip_bars': args.skip,
        'n_features': len(feature_names),
        'n_models': len(models_info),
        'n_train': len(X_full),
        'n_test': len(X_test),
        'n_trees': [m['n_trees'] for m in models_info],
        'avg_trees': sum(m['n_trees'] for m in models_info) / len(models_info),
        'test_ic': round(test_ic, 4),
        'test_mse': round(test_mse, 6),
        'test_hit': round(test_hit, 4),
        'trained_at': datetime.now().isoformat(),
        'size_mb': round(size_mb, 1),
    }
    with open(os.path.join(MODEL_DIR, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f" ✅ 模型已保存: {model_path} ({size_mb:.1f} MB)")
    print(f"    特征: {len(feature_names)} | 模型数: {len(models_info)}")
    print(f"    树数: {[m['n_trees'] for m in models_info]} (avg={sum(m['n_trees'] for m in models_info)/len(models_info):.0f})")
    print(f"    测试 IC: {test_ic:.4f} | MSE: {test_mse:.6f} | Hit: {test_hit:.3f}")

    if test_hit > 0.55:
        print(f" ✅ Hit Ratio > 55%, 有效")
    elif test_hit > 0.52:
        print(f" ⚠️ Hit Ratio > 52%, 弱有效")
    else:
        print(f" ❌ Hit Ratio <= 52%, 需优化")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()