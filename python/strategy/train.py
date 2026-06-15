#!/usr/bin/env python3
"""
LGBM 生产级训练脚本 v10 — 绝对收益预测 + 截面排名 (双模式)

架构:
  日线模型 → 预测截面排名分位 (α选股层)
  30m模型  → 预测截面排名分位 (γ择时层)

关键设计:
  1. 截面排名回归: 每日对所有股票按收益率排名, 预测排名分位值
     消除市场beta干扰, 直接对齐IC评估指标
  2. Bagging Ensemble: 5个独立LGBM并行训练
  3. 时序分离: train(80%) → val(10%) → test(10%)
  4. 生产级参数: num_leaves=255, lr=0.01, 20000棵树
  5. 评估指标: IC (Spearman) + 分组回测

用法:
  python strategy/train.py --model daily           # 日线训练
  python strategy/train.py --model 30m             # 30m训练
  python strategy/train.py --model daily --quick   # 快速验证(2模型, 1000树)
  python strategy/train.py --model daily --db /path/to/stock_data.db
"""

import sys, os, argparse, pickle, json, sqlite3, warnings, time

# 抑制 sklearn 特征名警告 (numpy array predict 无列名)
warnings.filterwarnings('ignore', message='X does not have valid feature names')
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kw): return iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategy.features import FeaturePipeline

warnings.filterwarnings('ignore')

# ============ 路径 ============
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data/stock_data.db')

# ============ 模型配置 ============
CONFIGS = {
    'daily': {
        'label': '日线', 'horizon': 5, 'db_table': 'kline_daily',
        'min_history': 120, 'purged_gap': 5, 'north_shift_days': 1,
        'model_dir': os.path.join(ROOT, 'models/lgb_daily'),
        'role': 'α选股层',
    },
    '30m': {
        'label': '30分钟', 'horizon': 3, 'db_table': 'kline_30m',
        'min_history': 150, 'purged_gap': 3, 'north_shift_days': 0,
        'model_dir': os.path.join(ROOT, 'models/lgb_30m'),
        'role': 'γ择时层',
    },
}

# ============ 训练参数 ============
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# LGBM 生产级固定参数 (回归) — v7 全面防过拟合
# 策略: 小树 + 强正则 + 三重随机采样 + 路径平滑
# 预期: 训练IC 0.05-0.07, 测试IC 0.05-0.06, gap < 30%
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'l2',
    'boosting_type': 'gbdt',
    # --- 树结构: 金融数据信噪比低, 小树够用 ---
    'num_leaves': 63,              # 127→63, 减半叶子数
    'max_depth': 7,                # 10→7, 树深不超过7层
    'learning_rate': 0.005,        # 0.008→0.005, 更小步长, 更多树但每棵影响小
    'n_estimators': 20000,
    'early_stopping_rounds': 50,   # 80→50, 50轮不提升就停
    # --- 三重随机采样: 打破特征垄断, 增加模型多样性 ---
    'subsample': 0.5,              # 0.6→0.5, 每轮只用50%样本
    'subsample_freq': 1,           # 3→1, 每轮重新采样 (Dropout效果)
    'colsample_bytree': 0.35,      # 0.5→0.35, 每棵树只用35%特征
    'feature_fraction_bynode': 0.6, # 新增: 节点级再随机选60%特征
    # --- 强正则化: 压权重, 防噪声 ---
    'reg_alpha': 1.0,              # 0.5→1.0, L1 ×2
    'reg_lambda': 5.0,             # 1.0→5.0, L2 ×5, 强力压权重
    'min_child_samples': 300,      # 100→300, 叶子至少300样本
    'min_child_weight': 0.01,      # 0.001→0.01, Hessian约束 ×10
    'min_split_gain': 0.05,        # 0.01→0.05, 分裂收益不够5%不分
    # --- 路径平滑: 预测值做移动平均, 消除单棵树噪声 ---
    'path_smooth': 15,             # 新增: 平滑最近15棵树的梯度
    'verbosity': -1,
    'random_state': None,
    'n_jobs': 3,
    'force_row_wise': True,
}

# 快速验证参数 (回归)
QUICK_PARAMS = {
    'objective': 'regression',
    'metric': 'l2',
    'boosting_type': 'gbdt',
    'num_leaves': 127, 'max_depth': 8, 'learning_rate': 0.05,
    'n_estimators': 1000, 'early_stopping_rounds': 50,
    'subsample': 0.8, 'colsample_bytree': 0.7,
    'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'min_child_samples': 50, 'verbosity': -1,
    'n_jobs': 3, 'force_row_wise': True,
}

# 并行配置 (M4 Pro 16核)
N_MODELS = 5
N_JOBS_PARALLEL = 5   # 5个模型并行
SEEDS = [42, 123, 456, 789, 1024]
QUICK_MODELS = 2
QUICK_SEEDS = [42, 123]

# 去相关阈值 (v7: 0.95→0.88, 更激进去重, 预计移除 35-50 个特征)
CORR_THRESHOLD = 0.88

# 收益率异常值过滤 (绝对值超过此阈值的样本丢弃)
RETURN_CLIP = 0.20

# 目标类型: 'rank'=截面排名分位, 'return'=绝对5日收益率
TARGET_TYPE = 'rank'  # 回退到排名模式 (绝对收益预测无效, IC=-0.016)


# ============ 数据加载 ============
def load_data(db_path: str, table: str) -> Dict[str, pd.DataFrame]:
    """从 SQLite 加载所有股票数据"""
    conn = sqlite3.connect(db_path)
    symbols = [r[0] for r in conn.execute(f"SELECT DISTINCT symbol FROM {table}")]
    data = {}
    for sym in symbols:
        try:
            df = pd.read_sql(
                f"SELECT * FROM {table} WHERE symbol=? ORDER BY date", conn, params=(sym,))
            if len(df) >= 100:
                df['date'] = pd.to_datetime(df['date'], format='mixed')
                df = df.sort_values('date').reset_index(drop=True)
                data[sym] = df
        except Exception:
            continue
    conn.close()
    print(f"  加载 {len(data)} 只股票, 共 {sum(len(d) for d in data.values()):,} 行 ({table})")
    return data


def load_sentiment(conn) -> pd.DataFrame:
    try:
        df = pd.read_sql(
            "SELECT symbol, trade_date as date, lhb_flag, lhb_net_buy, "
            "lhb_net_buy_ratio, lhb_ret_5d, is_limit_up, is_limit_down, "
            "vol_ratio_20, abnormal_ret, consecutive_limit_up FROM sentiment_daily", conn)
        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'], format='mixed').dt.strftime('%Y-%m-%d')
            return df
    except Exception:
        pass
    return pd.DataFrame()


def get_all_dates(data: Dict) -> np.ndarray:
    dates = set()
    for df in data.values():
        dates.update(df['date'].values)
    return np.array(sorted(dates))


# ============ 数据准备 ============
def prepare_data(data: Dict, conn, cfg: dict,
                 train_cutoff, val_cutoff,
                 pipeline: FeaturePipeline) -> Optional[Tuple]:
    """准备训练数据: 特征 + 截面排名回归目标 + 时序切分

    目标: 每日对股票按未来收益率排名, 预测排名分位值 (0~1)
    这直接对齐 IC 评估指标, 消除市场 beta 干扰
    """
    sent_df = load_sentiment(conn)
    has_sent = len(sent_df) > 0
    horizon = cfg['horizon']

    # 第一遍: 计算所有股票特征和收益率
    print("  计算个股特征...")
    all_features = {}
    stock_data = {}
    stock_returns = {}  # {symbol: {date_str: return_val}}
    success = 0

    for sym, df in tqdm(data.items(), desc='   计算个股特征', unit='stock'):
        try:
            feats = pipeline.compute_stock(df, sym)
            if has_sent:
                feats = pipeline.merge_sentiment(feats, df, sym, sent_df)
            feats = feats.fillna(method='ffill').fillna(0)
            feats.index = df['date'].values
            all_features[sym] = feats
            stock_data[sym] = df

            # 预计算收益率 (后续用于截面排名)
            close = df['close'].values.astype(float)
            ret = np.full(len(close), np.nan)
            for j in range(len(close) - horizon):
                ret[j] = (close[j + horizon] - close[j]) / close[j]

            date_strs = [str(d)[:10] for d in df['date'].values]
            ret_map = {}
            for j, d in enumerate(date_strs):
                if not np.isnan(ret[j]) and abs(ret[j]) < RETURN_CLIP:
                    ret_map[d] = ret[j]
            stock_returns[sym] = ret_map

            success += 1
        except Exception:
            continue

    # 第二遍: 截面排名特征
    print("  计算截面排名特征...")
    cs_features = pipeline.compute_cross_section(all_features, get_all_dates(data))

    # 第三遍: 按日期截面排名, 构建目标 (仅 rank 模式)
    if TARGET_TYPE == 'rank':
        print("  计算截面排名目标...")
        # 收集每个日期的所有股票收益率
        date_returns: Dict[str, Dict[str, float]] = {}  # {date: {symbol: return}}
        for sym, ret_map in stock_returns.items():
            for d, r in ret_map.items():
                if d not in date_returns:
                    date_returns[d] = {}
                date_returns[d][sym] = r

        # 对每个日期, 截面排名 → 分位值 0~1
        stock_rank_target: Dict[str, Dict[str, float]] = {}  # {symbol: {date: rank_pct}}
        for sym in stock_returns:
            stock_rank_target[sym] = {}

        n_dates_ranked = 0
        for d, sym_rets in date_returns.items():
            if len(sym_rets) < 10:  # 至少10只股票才有排名意义
                continue
            rets = np.array(list(sym_rets.values()))
            ranks = (pd.Series(rets).rank(pct=True) - 0.5).values  # 中心化到 -0.5 ~ 0.5
            for i, sym in enumerate(sym_rets.keys()):
                stock_rank_target[sym][d] = float(ranks[i])
            n_dates_ranked += 1
        print(f"  截面排名完成: {n_dates_ranked} 个交易日, {len(stock_returns)} 只股票")
    else:
        print("  目标: 绝对5日收益率 (跳过截面排名)")
        stock_rank_target = stock_returns  # 直接用收益率

    # 第四遍: 合并特征 + 排名目标 + 实际收益率 + 切分
    feature_names = None
    X_tr, y_tr, X_va, y_va, X_te, y_te = [], [], [], [], [], []
    y_te_raw = []  # 测试集实际收益率 (用于分组回测展示)

    for sym, df in tqdm(list(stock_data.items()), desc='   合并特征+目标', unit='stock'):
        try:
            feats = all_features[sym]
            if sym in cs_features:
                feats = pd.concat([feats, cs_features[sym]], axis=1)

            if feature_names is None:
                feature_names = list(feats.columns)

            # 截面排名目标 + 实际收益率
            rank_target = stock_rank_target.get(sym, {})
            ret_map = stock_returns.get(sym, {})
            date_strs = [str(d)[:10] for d in df['date'].values]
            target = np.array([rank_target.get(d, np.nan) for d in date_strs], dtype=np.float32)
            raw_ret = np.array([ret_map.get(d, np.nan) for d in date_strs], dtype=np.float32)
            valid = ~np.isnan(target)

            if valid.sum() < cfg['min_history']:
                continue

            feats_v = feats[valid].values
            target_v = target[valid]
            raw_ret_v = raw_ret[valid]
            dates_v = feats.index[valid]

            # 时序切分
            train_mask = dates_v <= train_cutoff
            val_mask = (dates_v > train_cutoff) & (dates_v <= val_cutoff)
            test_mask = dates_v > val_cutoff

            if train_mask.sum() >= 50:
                X_tr.append(feats_v[train_mask])
                y_tr.append(target_v[train_mask])
            if val_mask.sum() >= 10:
                X_va.append(feats_v[val_mask])
                y_va.append(target_v[val_mask])
            if test_mask.sum() >= 10:
                X_te.append(feats_v[test_mask])
                y_te.append(target_v[test_mask])
                y_te_raw.append(raw_ret_v[test_mask])
        except Exception:
            continue

    if not X_tr:
        return None

    X_train = np.vstack(X_tr).astype(np.float32)
    y_train = np.concatenate(y_tr).astype(np.float32)
    X_val = np.vstack(X_va).astype(np.float32)
    y_val = np.concatenate(y_va).astype(np.float32)
    X_test = np.vstack(X_te).astype(np.float32)
    y_test = np.concatenate(y_te).astype(np.float32)
    y_test_raw = np.concatenate(y_te_raw).astype(np.float32) if y_te_raw else None

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names, y_test_raw


# ============ 特征去冗余 ============
def remove_redundant(X: np.ndarray, feature_names: List[str]) -> Tuple:
    """去掉高度相关的特征 (corr > threshold)"""
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
    print(f"  去冗余: {sum(keep)}/{len(feature_names)} (移除 {len(rm)} 个高相关)")
    return X[:, keep], new_names, keep


# ============ 单模型训练 ============
def train_one(seed: int, X_train, y_train, X_val, y_val,
              feature_names: List[str], params: dict) -> Dict:
    """训练单个 LGBM 回归器"""
    t0 = time.time()

    p = {**params, 'random_state': seed}
    model = lgb.LGBMRegressor(**p)

    n_est = p.pop('n_estimators')
    es_rounds = p.pop('early_stopping_rounds')
    p.pop('n_jobs', None)

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                         lgb.log_evaluation(0)])  # 安静模式

    n_trees = model.best_iteration_ or n_est
    elapsed = time.time() - t0

    # 验证集评估 (回归指标)
    pred = model.predict(X_val)
    ic = spearmanr(y_val, pred)[0]  # Rank IC
    mse = mean_squared_error(y_val, pred)

    # 特征重要性
    imp = model.feature_importances_
    top_idx = np.argsort(imp)[-10:][::-1]
    top_feats = [(feature_names[i], int(imp[i])) for i in top_idx]

    print(f"  [seed={seed}] {n_trees}棵 | val IC={ic:.4f} MSE={mse:.6f} | {elapsed:.0f}s "
          f"| top3: {', '.join(f[0].split('_')[-2] if '_' in f[0] else f[0] for f in top_feats[:3])}")

    return {
        'model': model, 'seed': seed, 'n_trees': n_trees,
        'val_ic': round(ic, 4), 'val_mse': round(mse, 6),
        'train_time_s': round(elapsed, 1),
        'top_features': top_feats,
    }


# ============ 测试集评估 ============
def evaluate_ensemble(models_info: List[Dict], X_test, y_test,
                      feature_names: List[str], y_test_raw=None):
    """测试集评估: 单模型 + Ensemble (IC + MSE)

    Args:
        y_test: 排名分位目标 (用于计算 IC)
        y_test_raw: 实际收益率 (用于分组回测展示)
    """
    n = len(models_info)
    print(f"\n{'='*70}")
    print(f" 🧪 测试集评估 ({len(X_test):,}条, {n}模型)")
    print(f"{'='*70}")

    # 目标统计
    target_label = '绝对收益' if TARGET_TYPE == 'return' else '排名目标'
    print(f"  {target_label}: mean={y_test.mean():.4f} std={y_test.std():.4f} "
          f"min={y_test.min():.4f} max={y_test.max():.4f}")
    if y_test_raw is not None and TARGET_TYPE != 'return':
        print(f"  实际收益: mean={y_test_raw.mean():.4f} std={y_test_raw.std():.4f} "
              f"min={y_test_raw.min():.4f} max={y_test_raw.max():.4f}")

    # 单模型评估
    all_preds = []
    for info in models_info:
        model = info['model']
        pred = model.predict(X_test)
        all_preds.append(pred)

        ic = spearmanr(y_test, pred)[0]
        mse = mean_squared_error(y_test, pred)
        print(f"  [seed={info['seed']}] IC={ic:.4f} | MSE={mse:.6f} | {info['n_trees']}棵")

    # Ensemble (预测均值)
    if n > 1:
        ensemble_pred = np.mean(all_preds, axis=0)
        ic = spearmanr(y_test, ensemble_pred)[0]
        mse = mean_squared_error(y_test, ensemble_pred)
        print(f"  {'─'*50}")
        print(f"  🏆 Ensemble({n}) → IC={ic:.4f} | MSE={mse:.6f}")

    # 分组回测 (按预测值分5组, 用实际收益率评估)
    if n > 1:
        print(f"\n  📊 分组回测 (按预测排名分5组, 展示实际收益率):")
        sort_idx = np.argsort(ensemble_pred)
        n_per_group = len(sort_idx) // 5
        for g in range(5):
            start = g * n_per_group
            end = start + n_per_group if g < 4 else len(sort_idx)
            group_rank = y_test[sort_idx[start:end]].mean()
            group_pred = ensemble_pred[sort_idx[start:end]].mean()
            if y_test_raw is not None:
                group_ret = y_test_raw[sort_idx[start:end]].mean()
                print(f"    G{g+1}: 排名={group_rank:+.4f} | 预测={group_pred:+.4f} | 实际5日收益={group_ret:+.4%}")
            else:
                print(f"    G{g+1}: 排名={group_rank:+.4f} | 预测={group_pred:+.4f}")
        # 多空收益
        if y_test_raw is not None:
            long_ret = y_test_raw[sort_idx[-n_per_group:]].mean()
            short_ret = y_test_raw[sort_idx[:n_per_group]].mean()
            print(f"    多空收益差(5日): {long_ret - short_ret:+.4%}  (买入G5 卖出G1)")
        else:
            long_rank = y_test[sort_idx[-n_per_group:]].mean()
            short_rank = y_test[sort_idx[:n_per_group]].mean()
            print(f"    多空排名差: {long_rank - short_rank:.4f}")

    # 特征重要性
    if models_info:
        imp = models_info[0]['model'].feature_importances_
        top = np.argsort(imp)[-20:][::-1]
        print(f"\n  Top 20 特征:")
        for idx in top:
            print(f"    {feature_names[idx]}: {int(imp[idx])}")

    return ic, mse


# ============ 主入口 ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['daily', '30m'], default='daily')
    parser.add_argument('--quick', action='store_true', help='快速验证 (2模型, 1000树)')
    parser.add_argument('--db', type=str, default=DB_PATH, help='SQLite数据库路径')
    args = parser.parse_args()

    cfg = CONFIGS[args.model]
    cfg['model_key'] = args.model

    n_models = QUICK_MODELS if args.quick else N_MODELS
    seeds = QUICK_SEEDS[:n_models] if args.quick else SEEDS[:n_models]
    params = QUICK_PARAMS if args.quick else LGBM_PARAMS

    print("=" * 70)
    target_desc = '绝对5日收益率' if TARGET_TYPE == 'return' else '截面排名分位 (0~1)'
    print(f"  LGBM {cfg['label']}模型训练 v10 — {n_models}模型 Bagging Ensemble + 宏观 + LSTM")
    print(f"  目标: {target_desc} | 预测周期: {cfg['horizon']}根K线")
    print(f"  时序: train({TRAIN_RATIO:.0%}) → val({VAL_RATIO:.0%}) → test({TEST_RATIO:.0%})")
    print(f"  并行: {n_models}模型并行 (joblib n_jobs={N_JOBS_PARALLEL})")
    print(f"  {'⚠️ 快速模式' if args.quick else '🚀 生产模式'}: "
          f"lr={params['learning_rate']} leaves={params['num_leaves']} "
          f"max_trees={params['n_estimators']}")
    print("=" * 70)

    # ---- 1. 加载数据 ----
    print(f"\n📊 加载数据 ({cfg['db_table']})...")
    t0 = time.time()
    data = load_data(args.db, cfg['db_table'])
    print(f"  加载耗时: {time.time()-t0:.0f}s")

    all_dates = get_all_dates(data)
    n_dates = len(all_dates)
    train_cutoff = all_dates[int(n_dates * TRAIN_RATIO)]
    val_cutoff = all_dates[int(n_dates * (TRAIN_RATIO + VAL_RATIO))]

    print(f"  {n_dates} 个交易日, {len(data)} 只股票")
    print(f"  train: ~{str(train_cutoff)[:10]}  "
          f"val: ~{str(val_cutoff)[:10]}  "
          f"test: ~{str(all_dates[-1])[:10]}")

    # ---- 2. 特征工程 ----
    print(f"\n🔧 特征工程...")
    t0 = time.time()
    pipeline = FeaturePipeline(cfg)
    conn = sqlite3.connect(args.db)
    result = prepare_data(data, conn, cfg, train_cutoff, val_cutoff, pipeline)
    conn.close()

    if result is None:
        print("❌ 数据准备失败"); return

    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names, y_test_raw = result
    print(f"  特征计算耗时: {time.time()-t0:.0f}s")
    print(f"  训练: {len(X_train):,}条 | 验证: {len(X_val):,}条 | 测试: {len(X_test):,}条")
    print(f"  特征: {len(feature_names)}")

    # ---- 3. 特征去冗余 ----
    print(f"\n🔧 特征去冗余...")
    X_train, feature_names, corr_mask = remove_redundant(X_train, feature_names)
    X_val = X_val[:, corr_mask]
    X_test = X_test[:, corr_mask]
    print(f"  最终特征: {len(feature_names)}")

    # ---- 4. 并行训练 Bagging Ensemble ----
    print(f"\n🏋️ 并行训练 {n_models} 个模型...")
    t0 = time.time()

    models_info = Parallel(n_jobs=N_JOBS_PARALLEL, verbose=10)(
        delayed(train_one)(seed, X_train, y_train, X_val, y_val, feature_names, params)
        for seed in seeds
    )

    train_time = time.time() - t0
    avg_trees = int(np.mean([m['n_trees'] for m in models_info]))
    avg_ic = np.mean([m['val_ic'] for m in models_info])
    print(f"\n  ✅ {n_models}模型训练完成: 总耗时 {train_time/60:.1f}min, "
          f"平均 {avg_trees}棵/模型, 平均 val IC={avg_ic:.4f}")

    # ---- 5. 测试集评估 ----
    test_ic, test_mse = evaluate_ensemble(models_info, X_test, y_test, feature_names, y_test_raw)

    # ---- 6. 最终模型 (train+val 全量) ----
    print(f"\n🏋️ 训练最终模型 (train+val 全量, 复用最佳树数)...")
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    final_models = []
    final_n_trees = []
    for info in models_info:
        p = {**params, 'random_state': info['seed']}
        n_est = p.pop('n_estimators')
        es_rounds = p.pop('early_stopping_rounds')
        p.pop('n_jobs', None)

        m = lgb.LGBMRegressor(**p, n_estimators=info['n_trees'])
        m.fit(X_full, y_full)
        final_models.append(m)
        final_n_trees.append(info['n_trees'])

    # ---- 7. 保存 ----
    print(f"\n💾 保存模型...")
    model_dir = cfg['model_dir']
    os.makedirs(model_dir, exist_ok=True)

    ensemble = {
        'models': final_models,
        'feature_names': feature_names,
        'keep_features': feature_names,
        'n_models': n_models,
        'horizon': cfg['horizon'],
        'model_type': 'return_regression' if TARGET_TYPE == 'return' else 'rank_regression',
        'train_date': datetime.now().strftime('%Y-%m-%d'),
        'train_samples': len(X_full),
        'seeds': seeds,
        'n_trees_per_model': final_n_trees,
        'avg_n_trees': int(np.mean(final_n_trees)),
        'val_ic_list': [m['val_ic'] for m in models_info],
        'val_mse_list': [m['val_mse'] for m in models_info],
        'test_ic': round(test_ic, 4),
        'test_mse': round(test_mse, 6),
        'params': {k: v for k, v in params.items()
                   if k not in ('verbosity', 'random_state',
                                'force_row_wise', 'n_jobs',
                                'objective', 'metric',
                                'n_estimators', 'early_stopping_rounds')},
    }

    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(ensemble, f)

    size_mb = os.path.getsize(model_path) / 1024 / 1024

    # metadata
    meta = {
        'model_type': args.model, 'label': cfg['label'],
        'horizon': cfg['horizon'], 'n_features': len(feature_names),
        'n_models': n_models, 'n_trees_per_model': final_n_trees,
        'avg_n_trees': int(np.mean(final_n_trees)),
        'n_train': len(X_full), 'n_test': len(X_test),
        'test_ic': round(test_ic, 4), 'test_mse': round(test_mse, 6),
        'val_ic': round(np.mean([m['val_ic'] for m in models_info]), 4),
        'params': ensemble['params'],
        'role': cfg['role'], 'trained_at': datetime.now().isoformat(),
        'size_mb': round(size_mb, 1),
    }
    with open(os.path.join(model_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f" ✅ 模型已保存: {model_path} ({size_mb:.1f} MB)")
    print(f"    特征: {len(feature_names)} | 模型数: {n_models}")
    print(f"    树数: {final_n_trees} (avg={int(np.mean(final_n_trees))})")
    print(f"    测试 IC: {test_ic:.4f} | MSE: {test_mse:.6f}")
    print(f"    总训练耗时: {train_time/60:.1f}min")

    if TARGET_TYPE == 'return':
        # 绝对收益模式: IC 更难, 阈值降低
        if test_ic > 0.04:
            print(f" ✅ 有效 (收益预测 IC > 0.04)")
        elif test_ic > 0.02:
            print(f" ⚠️ 弱有效 (IC > 0.02)，可优化")
        else:
            print(f" ❌ 无效 (IC <= 0.02)")
    else:
        if test_ic > 0.05:
            print(f" ✅ 样本外有效 (IC > 0.05)")
        elif test_ic > 0.03:
            print(f" ⚠️ 弱有效 (IC > 0.03)，可优化")
        else:
            print(f" ❌ 样本外不足 (IC <= 0.03)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()