#!/usr/bin/env python3
"""
LGBM 生产级训练脚本 v4 — 3分类 Bagging Ensemble

架构:
  日线模型 → 预测5日涨跌方向 (α选股层)
  30m模型  → 预测3根K线涨跌方向 (γ择时层)

关键设计:
  1. 3分类目标: 涨(0) / 震荡(1) / 跌(2)，分位数自适应切分
  2. Bagging Ensemble: 5个独立LGBM并行训练
  3. 时序分离: train(80%) → val(10%) → test(10%)
  4. 生产级参数: num_leaves=255, lr=0.02, 15000棵树
  5. 特征保留全部 (不砍半)，截面排名特征 + 冗余去相关

用法:
  python strategy/train.py --model daily           # 日线训练
  python strategy/train.py --model 30m             # 30m训练
  python strategy/train.py --model daily --quick   # 快速验证(2模型, 1000树)
  python strategy/train.py --model daily --db /path/to/stock_data.db
"""

import sys, os, argparse, pickle, json, sqlite3, warnings, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, accuracy_score, log_loss
from collections import Counter

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

# 分类阈值: 分位数自适应 (涨30% / 震荡40% / 跌30%)
UP_Q = 0.70
DOWN_Q = 0.30

# LGBM 生产级固定参数
LGBM_PARAMS = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 255,
    'max_depth': 16,
    'learning_rate': 0.01,
    'n_estimators': 20000,
    'early_stopping_rounds': 200,
    'subsample': 0.75,
    'colsample_bytree': 0.65,
    'subsample_freq': 5,
    'reg_alpha': 0.05,
    'reg_lambda': 0.05,
    'min_child_samples': 30,
    'min_split_gain': 0.005,
    'verbosity': -1,
    'random_state': None,
    'n_jobs': 3,
    'force_row_wise': True,
}

# 快速验证参数
QUICK_PARAMS = {
    'objective': 'multiclass', 'num_class': 3,
    'metric': 'multi_logloss', 'boosting_type': 'gbdt',
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

# 去相关阈值
CORR_THRESHOLD = 0.95

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


# ============ 3分类目标 ============
def make_classification_target(returns: np.ndarray) -> np.ndarray:
    """涨(0) / 震荡(1) / 跌(2)，分位数自适应切分"""
    valid = returns[~np.isnan(returns) & (np.abs(returns) < 0.15)]
    up_thresh = np.percentile(valid, UP_Q * 100)
    down_thresh = np.percentile(valid, DOWN_Q * 100)

    labels = np.full_like(returns, np.nan, dtype=float)
    labels[returns > up_thresh] = 0      # 涨
    labels[(returns >= down_thresh) & (returns <= up_thresh)] = 1  # 震荡
    labels[returns < down_thresh] = 2     # 跌

    return labels


# ============ 数据准备 ============
def prepare_data(data: Dict, conn, cfg: dict,
                 train_cutoff, val_cutoff,
                 pipeline: FeaturePipeline) -> Optional[Tuple]:
    """准备训练数据: 特征 + 3分类目标 + 时序切分"""
    sent_df = load_sentiment(conn)
    has_sent = len(sent_df) > 0
    horizon = cfg['horizon']

    # 第一遍: 计算所有股票特征
    print("  计算个股特征...")
    all_features = {}
    stock_data = {}
    success = 0

    for sym, df in data.items():
        try:
            feats = pipeline.compute_stock(df, sym)
            if has_sent:
                feats = pipeline.merge_sentiment(feats, df, sym, sent_df)
            feats = feats.fillna(method='ffill').fillna(0)
            feats.index = df['date'].values
            all_features[sym] = feats
            stock_data[sym] = df
            success += 1
        except Exception:
            continue
        if success % 100 == 0:
            print(f"    {success}/{len(data)}")

    # 第二遍: 截面排名特征
    print("  计算截面排名特征...")
    cs_features = pipeline.compute_cross_section(all_features, get_all_dates(data))

    # 第三遍: 合并 + 目标 + 切分
    feature_names = None
    X_tr, y_tr, X_va, y_va, X_te, y_te = [], [], [], [], [], []

    for sym, df in list(stock_data.items()):
        try:
            feats = all_features[sym]
            if sym in cs_features:
                feats = pd.concat([feats, cs_features[sym]], axis=1)

            if feature_names is None:
                feature_names = list(feats.columns)

            # 3分类目标
            close = df['close'].values.astype(float)
            ret = np.full(len(close), np.nan)
            for j in range(len(close) - horizon):
                ret[j] = (close[j + horizon] - close[j]) / close[j]

            target = make_classification_target(ret)
            valid = ~np.isnan(target)

            if valid.sum() < cfg['min_history']:
                continue

            feats_v = feats[valid].values
            target_v = target[valid]
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
        except Exception:
            continue

    if not X_tr:
        return None

    X_train = np.vstack(X_tr).astype(np.float32)
    y_train = np.concatenate(y_tr).astype(np.int8)
    X_val = np.vstack(X_va).astype(np.float32)
    y_val = np.concatenate(y_va).astype(np.int8)
    X_test = np.vstack(X_te).astype(np.float32)
    y_test = np.concatenate(y_te).astype(np.int8)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names


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
    """训练单个 LGBM 分类器"""
    t0 = time.time()

    p = {**params, 'random_state': seed}
    model = lgb.LGBMClassifier(**p)

    n_est = p.pop('n_estimators')
    es_rounds = p.pop('early_stopping_rounds')
    p.pop('n_jobs', None)

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                         lgb.log_evaluation(0)])  # 安静模式

    n_trees = model.best_iteration_ or n_est
    elapsed = time.time() - t0

    # 验证集评估
    pred = model.predict(X_val)
    f1 = f1_score(y_val, pred, average='macro')
    acc = accuracy_score(y_val, pred)

    # 特征重要性
    imp = model.feature_importances_
    top_idx = np.argsort(imp)[-10:][::-1]
    top_feats = [(feature_names[i], int(imp[i])) for i in top_idx]

    print(f"  [seed={seed}] {n_trees}棵 | val F1={f1:.4f} acc={acc:.4f} | {elapsed:.0f}s "
          f"| top3: {', '.join(f[0].split('_')[-2] if '_' in f[0] else f[0] for f in top_feats[:3])}")

    return {
        'model': model, 'seed': seed, 'n_trees': n_trees,
        'val_f1': round(f1, 4), 'val_acc': round(acc, 4),
        'train_time_s': round(elapsed, 1),
        'top_features': top_feats,
    }


# ============ 测试集评估 ============
def evaluate_ensemble(models_info: List[Dict], X_test, y_test,
                      feature_names: List[str]):
    """测试集评估: 单模型 + 投票集成"""
    n = len(models_info)
    print(f"\n{'='*70}")
    print(f" 🧪 测试集评估 ({len(X_test):,}条, {n}模型)")
    print(f"{'='*70}")

    # 类别分布
    dist = Counter(y_test)
    print(f"  类别分布: 涨={dist.get(0,0)} 震荡={dist.get(1,0)} 跌={dist.get(2,0)}")

    # 单模型评估
    all_preds = []
    for info in models_info:
        model = info['model']
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        all_preds.append(proba)

        f1 = f1_score(y_test, pred, average='macro')
        acc = accuracy_score(y_test, pred)
        ll = log_loss(y_test, proba)
        print(f"  [seed={info['seed']}] F1={f1:.4f} | Acc={acc:.4f} | LogLoss={ll:.4f} | {info['n_trees']}棵")

    # 投票集成 (soft voting)
    if n > 1:
        avg_proba = np.mean(all_preds, axis=0)
        ensemble_pred = np.argmax(avg_proba, axis=1)
        f1 = f1_score(y_test, ensemble_pred, average='macro')
        acc = accuracy_score(y_test, ensemble_pred)
        ll = log_loss(y_test, avg_proba)
        print(f"  {'─'*50}")
        print(f"  🏆 Ensemble({n}投票) → F1={f1:.4f} | Acc={acc:.4f} | LogLoss={ll:.4f}")

    # 特征重要性
    if models_info:
        imp = models_info[0]['model'].feature_importances_
        top = np.argsort(imp)[-20:][::-1]
        print(f"\n  Top 20 特征:")
        for idx in top:
            print(f"    {feature_names[idx]}: {int(imp[idx])}")

    return f1, acc


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
    print(f"  LGBM {cfg['label']}模型训练 v4 — {n_models}模型 Bagging Ensemble")
    print(f"  目标: 3分类(涨/震荡/跌) | 预测周期: {cfg['horizon']}根K线")
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

    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names = result
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
    print(f"\n  ✅ {n_models}模型训练完成: 总耗时 {train_time/60:.1f}min, 平均 {avg_trees}棵/模型")

    # ---- 5. 测试集评估 ----
    test_f1, test_acc = evaluate_ensemble(models_info, X_test, y_test, feature_names)

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

        m = lgb.LGBMClassifier(**p, n_estimators=info['n_trees'])
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
        'keep_features': feature_names,  # 全部保留
        'n_models': n_models,
        'horizon': cfg['horizon'],
        'model_type': 'multiclass_3',
        'train_date': datetime.now().strftime('%Y-%m-%d'),
        'train_samples': len(X_full),
        'seeds': seeds,
        'n_trees_per_model': final_n_trees,
        'avg_n_trees': int(np.mean(final_n_trees)),
        'val_f1_list': [m['val_f1'] for m in models_info],
        'val_acc_list': [m['val_acc'] for m in models_info],
        'test_f1': round(test_f1, 4),
        'test_acc': round(test_acc, 4),
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
        'test_f1': round(test_f1, 4), 'test_acc': round(test_acc, 4),
        'val_f1': round(np.mean([m['val_f1'] for m in models_info]), 4),
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
    print(f"    测试 F1: {test_f1:.4f} | Acc: {test_acc:.4f}")
    print(f"    总训练耗时: {train_time/60:.1f}min")

    if test_f1 > 0.4:
        print(f" ✅ 样本外有效 (F1 > 0.4)")
    elif test_f1 > 0.35:
        print(f" ⚠️ 弱有效 (F1 > 0.35)，可优化")
    else:
        print(f" ❌ 样本外不足 (F1 <= 0.35)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()