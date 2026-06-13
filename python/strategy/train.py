#!/usr/bin/env python3
"""
LGBM 统一训练脚本 v3 — 回归 + 截面排序

架构:
  日线模型 → 预测5日收益率 → 截面排序选股 (α层)
  30m模型  → 预测3根K线收益率 → 截面排序择时 (γ层)

关键设计:
  1. 时序分离: train(80%) → val(10%) → test(10%)
  2. Early stopping 自动定树数 (max 10000, patience 100)
  3. 特征选择仅基于训练集
  4. 截面排名特征 (CrossSection)
  5. 北向资金滞后 (日线模型 shift 1天)

用法:
  python strategy/train.py --model daily           # 日线完整训练
  python strategy/train.py --model 30m             # 30m完整训练
  python strategy/train.py --model daily --quick   # 快速验证
  python strategy/train.py --model daily --tune    # Optuna 超参搜索
"""

import sys, os, argparse, pickle, json, sqlite3, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.feature_selection import SelectFromModel
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategy.features import FeaturePipeline, compute_features as _compute_features

warnings.filterwarnings('ignore')

# ============ 路径 ============
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data/stock_data.db')

# ============ 模型配置 ============
CONFIGS = {
    'daily': {
        'label': '日线', 'horizon': 5, 'db_table': 'kline_daily',
        'min_history': 120, 'min_samples': 200,
        'features': 'enhanced+advanced+market',
        'purged_gap': 5, 'north_shift_days': 1,
        'model_dir': os.path.join(ROOT, 'models/lgb_daily'),
        'role': 'α选股层',
        'search_estimators': 500, 'optuna_trials': 100,
    },
    '30m': {
        'label': '30分钟', 'horizon': 3, 'db_table': 'kline_30m',
        'min_history': 150, 'min_samples': 200,
        'features': 'enhanced+advanced',
        'purged_gap': 3, 'north_shift_days': 0,
        'model_dir': os.path.join(ROOT, 'models/lgb_30m'),
        'role': 'γ择时层',
        'search_estimators': 500, 'optuna_trials': 100,
    },
}

# 训练参数
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MAX_TREES = 10000
EARLY_STOPPING = 100
QUICK_TREES = 500
QUICK_PATIENCE = 30
CORR_THRESHOLD = 0.95

# LGBM 固定参数
LGBM_FIXED = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'random_state': 42,
    'force_row_wise': True,
    'n_jobs': -1,
}


# ============ 数据加载 ============
def load_data(db_path: str, table: str) -> Dict[str, pd.DataFrame]:
    conn = sqlite3.connect(db_path)
    symbols = [r[0] for r in conn.execute(f"SELECT DISTINCT symbol FROM {table}")]
    data = {}
    for sym in symbols:
        try:
            df = pd.read_sql(
                f"SELECT * FROM {table} WHERE symbol=? ORDER BY date", conn, params=(sym,))
            if len(df) >= 120:
                df['date'] = pd.to_datetime(df['date'], format='mixed')
                df = df.sort_values('date').reset_index(drop=True)
                data[sym] = df
        except Exception:
            continue
    conn.close()
    print(f"加载了 {len(data)} 只股票 (表: {table})")
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
def prepare_split_data(data: Dict, conn, cfg: dict,
                       train_cutoff, val_cutoff,
                       feature_pipeline: FeaturePipeline) -> Optional[Tuple]:
    """准备时序分离的数据，包含截面排名特征"""
    sent_df = load_sentiment(conn)
    has_sent = len(sent_df) > 0
    horizon = cfg['horizon']

    # 第一遍: 计算所有股票的独立特征
    print("  计算个股特征...")
    all_features = {}
    stock_data = {}
    success = 0

    for sym, df in data.items():
        try:
            feats = feature_pipeline.compute_stock(df, sym)
            if has_sent:
                feats = feature_pipeline.merge_sentiment(feats, df, sym, sent_df)
            feats = feats.fillna(method='ffill').fillna(0)
            feats.index = df['date'].values
            all_features[sym] = feats
            stock_data[sym] = df
            success += 1
        except Exception:
            continue
        if (success) % 100 == 0:
            print(f"    {success}/{len(data)}")

    # 第二遍: 计算截面排名特征
    print("  计算截面排名特征...")
    cs_features = feature_pipeline.compute_cross_section(all_features, get_all_dates(data))

    # 第三遍: 合并截面特征 + 构建目标
    feature_names = None
    X_tr, y_tr, X_va, y_va, X_te, y_te = [], [], [], [], [], []

    for sym, df in stock_data.items():
        try:
            feats = all_features[sym]

            # 合并截面特征
            if sym in cs_features:
                cs = cs_features[sym]
                feats = pd.concat([feats, cs], axis=1)

            if feature_names is None:
                feature_names = list(feats.columns)

            # 目标
            close = df['close'].values.astype(float)
            target = np.full(len(close), np.nan)
            for j in range(len(close) - horizon):
                target[j] = (close[j + horizon] - close[j]) / close[j]

            valid = ~np.isnan(target)
            feats_v = feats[valid].values
            target_v = target[valid]
            dates_v = feats.index[valid]

            if len(feats_v) > cfg['min_history']:
                feats_v = feats_v[cfg['min_history']:]
                target_v = target_v[cfg['min_history']:]
                dates_v = dates_v[cfg['min_history']:]

            if len(feats_v) < 50:
                continue

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

    X_train = np.vstack(X_tr); y_train = np.concatenate(y_tr)
    X_val = np.vstack(X_va); y_val = np.concatenate(y_va)
    X_test = np.vstack(X_te); y_test = np.concatenate(y_te)

    # 过滤极端收益率
    for arr_x, arr_y in [(X_train, y_train), (X_val, y_val), (X_test, y_test)]:
        mask = np.abs(arr_y) < 0.15
        if hasattr(arr_x, '__getitem__'):
            filtered_x = arr_x[mask]
            filtered_y = arr_y[mask]
            arr_x = filtered_x
            arr_y = filtered_y

    # 重新赋值
    X_train, y_train = X_train[np.abs(y_train) < 0.15], y_train[np.abs(y_train) < 0.15]
    X_val, y_val = X_val[np.abs(y_val) < 0.15], y_val[np.abs(y_val) < 0.15]
    X_test, y_test = X_test[np.abs(y_test) < 0.15], y_test[np.abs(y_test) < 0.15]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names


# ============ 特征选择 ============
def remove_redundant(X: np.ndarray, feature_names: List[str]) -> Tuple:
    """去冗余 (corr > threshold)"""
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


def select_from_model(X_train, y_train, X_val, y_val, feature_names, params):
    """SelectFromModel (仅基于训练集)"""
    sel_params = {**params, 'n_estimators': 500}
    sel = lgb.LGBMRegressor(**sel_params)
    sel.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    sf = SelectFromModel(sel, threshold='median', prefit=True)
    mask = sf.get_support()
    n_sel = mask.sum()
    print(f"  SelectFromModel: {n_sel}/{len(feature_names)} (阈值=median, "
          f"{sel.best_iteration_ or 500}棵)")
    new_names = [fn for fn, m in zip(feature_names, mask) if m]
    return mask, new_names


# ============ 训练 ============
def train_model(X_train, y_train, X_val, y_val, params, max_trees, patience):
    """训练 LGBM，early stopping"""
    model = lgb.LGBMRegressor(**params, n_estimators=max_trees)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(patience, verbose=True),
                         lgb.log_evaluation(max(10, max_trees // 20))])
    n_trees = model.best_iteration_ or max_trees
    return model, n_trees


# ============ 评估 ============
def evaluate(model, X_test, y_test, feature_names):
    """测试集评估"""
    pred = model.predict(X_test)
    ic, _ = spearmanr(pred, y_test)
    if np.isnan(ic): ic = 0
    rmse = np.sqrt(np.mean((pred - y_test) ** 2))
    mae = np.mean(np.abs(pred - y_test))

    print(f"\n  📊 测试集 ({len(X_test):,}条):")
    print(f"    Rank IC: {ic:.4f}  |  RMSE: {rmse:.4f}  |  MAE: {mae:.4f}")

    imp = model.feature_importances_
    top = np.argsort(imp)[-20:][::-1]
    print(f"\n  Top 20 特征:")
    for idx in top:
        print(f"    {feature_names[idx]}: {imp[idx]:.0f}")

    return ic, rmse, mae


# ============ Optuna 超参搜索 ============
def optuna_search(X, y, cfg, quick=False):
    """Optuna 超参搜索 (保留旧逻辑)"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_trials = 20 if quick else cfg['optuna_trials']
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5, gap=cfg['purged_gap'])

    split = int(len(X) * 0.8)
    X_s, y_s = X[:split], y[:split]

    def objective(trial):
        p = {
            'objective': 'regression_l1', 'metric': 'mae',
            'boosting_type': 'gbdt', 'verbosity': -1, 'n_jobs': -1,
            'random_state': 42, 'n_estimators': cfg['search_estimators'],
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.95),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        }
        if cfg.get('search_sample', 1.0) < 1.0 and len(X_s) > 100000:
            n = int(len(X_s) * cfg['search_sample'])
            idx = np.random.RandomState(42 + trial.number).choice(len(X_s), n, replace=False)
            Xt, yt = X_s[idx], y_s[idx]
        else:
            Xt, yt = X_s, y_s

        scores = []
        for tr, te in TimeSeriesSplit(n_splits=3, gap=cfg['purged_gap']).split(Xt):
            m = lgb.LGBMRegressor(**p)
            m.fit(Xt[tr], yt[tr], eval_set=[(Xt[te], yt[te])],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
            pred = m.predict(Xt[te])
            corr, _ = spearmanr(pred, yt[te])
            if not np.isnan(corr): scores.append(corr)
        return np.mean(scores) if scores else 0

    print(f"\nOptuna超参搜索 ({n_trials}次, 5折PurgedCV(gap={cfg['purged_gap']}), 目标=Spearman)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest trial: {study.best_trial.number}. Best value: {study.best_value:.6f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # 保存最优参数
    params_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_params.json')
    existing = {}
    if os.path.exists(params_file):
        with open(params_file) as f:
            existing = json.load(f)
    existing[cfg.get('model_key', 'daily')] = study.best_params
    with open(params_file, 'w') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"  参数已保存至 {params_file}")

    return study.best_params


# ============ 主入口 ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['daily', '30m'], default='daily')
    parser.add_argument('--quick', action='store_true', help='快速验证 (跳过Optuna, 500树)')
    parser.add_argument('--tune-only', action='store_true', help='仅Optuna搜索 (不训练)')
    parser.add_argument('--no-cross-section', action='store_true', help='跳过截面特征')
    parser.add_argument('--no-tune', action='store_true', help='跳过Optuna搜索')
    args = parser.parse_args()

    cfg = CONFIGS[args.model]
    cfg['model_key'] = args.model

    print("=" * 60)
    print(f" LGBM {cfg['label']}模型训练 v3")
    print(f" 时序分离: train({TRAIN_RATIO:.0%}) → val({VAL_RATIO:.0%}) → test({TEST_RATIO:.0%})")
    print(f" Optuna: {'❌' if (args.quick or args.no_tune) else '✅'}  |  "
          f"截面特征: {'✅' if not args.no_cross_section else '❌'}  |  "
          f"purged_gap={cfg['purged_gap']}  |  north_shift={cfg['north_shift_days']}天")
    print("=" * 60)

    # ---- 1. 加载数据 + 时序切分 ----
    print(f"\n📊 加载数据 ({cfg['db_table']})...")
    data = load_data(DB_PATH, cfg['db_table'])

    all_dates = get_all_dates(data)
    n_dates = len(all_dates)
    train_cutoff = all_dates[int(n_dates * TRAIN_RATIO)]
    val_cutoff = all_dates[int(n_dates * (TRAIN_RATIO + VAL_RATIO))]

    print(f"  {n_dates} 个交易日, {len(data)} 只股票")
    print(f"  train: ~{str(train_cutoff)[:10]}  val: ~{str(val_cutoff)[:10]}  test: ~{str(all_dates[-1])[:10]}")

    pipeline = FeaturePipeline(cfg)

    conn = sqlite3.connect(DB_PATH)
    result = prepare_split_data(data, conn, cfg, train_cutoff, val_cutoff, pipeline)
    conn.close()

    if result is None:
        print("❌ 数据准备失败"); return

    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names = result

    print(f"\n  训练: {len(X_train):,}条  |  验证: {len(X_val):,}条  |  测试: {len(X_test):,}条")
    print(f"  特征: {len(feature_names)}  |  目标: mean={y_train.mean():.4f} std={y_train.std():.4f}")

    # ---- 2. Optuna 超参搜索 (默认执行) ----
    params_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_params.json')

    if args.tune_only:
        optuna_search(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]),
                      cfg, quick=args.quick)
        return

    if not args.quick and not args.no_tune:
        params = optuna_search(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]),
                               cfg, quick=False)
    else:
        params = {}
        if os.path.exists(params_file):
            with open(params_file) as f:
                params = json.load(f).get(args.model, {})
        if not params:
            print("⚠️ 未找到 best_params.json，使用默认参数")

    for k, v in LGBM_FIXED.items():
        params.setdefault(k, v)

    # ---- 3. 特征选择 (仅基于训练集) ----
    print("\n🔧 特征选择...")
    X_train, feature_names, corr_mask = remove_redundant(X_train, feature_names)
    X_val = X_val[:, corr_mask]
    X_test = X_test[:, corr_mask]

    sel_mask, feature_names = select_from_model(
        X_train, y_train, X_val, y_val, feature_names, params)
    X_train = X_train[:, sel_mask]
    X_val = X_val[:, sel_mask]
    X_test = X_test[:, sel_mask]

    print(f"  最终特征: {len(feature_names)}")

    # ---- 4. 训练 ----
    max_trees = QUICK_TREES if args.quick else MAX_TREES
    patience = QUICK_PATIENCE if args.quick else EARLY_STOPPING

    print(f"\n🏋️ 训练 (max {max_trees} 棵树, patience {patience})...")
    model, n_trees = train_model(X_train, y_train, X_val, y_val, params, max_trees, patience)
    print(f"\n  ✅ 训练完成: {n_trees} 棵树")

    # ---- 5. 测试集评估 ----
    print("\n" + "=" * 60)
    print(" 🧪 测试集评估 (真正样本外)")
    print("=" * 60)
    test_ic, test_rmse, test_mae = evaluate(model, X_test, y_test, feature_names)

    # ---- 6. 最终模型 (train+val) ----
    print(f"\n🏋️ 最终模型 (train+val, {n_trees} 棵树)...")
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    final_model = lgb.LGBMRegressor(**params, n_estimators=n_trees)
    final_model.fit(X_full, y_full)

    # ---- 7. 保存 ----
    print("\n💾 保存模型...")
    model_dir = cfg['model_dir']
    os.makedirs(model_dir, exist_ok=True)

    core_params = {k: v for k, v in params.items()
                   if k not in ('verbosity', 'random_state', 'force_row_wise',
                                'n_jobs', 'objective', 'metric', 'boosting_type')}

    model_data = {
        'model': final_model, 'feature_names': feature_names,
        'best_params': core_params,
        'test_ic': round(test_ic, 4), 'test_rmse': round(test_rmse, 4),
        'test_mae': round(test_mae, 4), 'n_trees': n_trees,
        'horizon': cfg['horizon'], 'n_features': len(feature_names),
        'n_train': len(X_full), 'n_test': len(X_test),
        'train_cutoff': str(train_cutoff)[:10], 'val_cutoff': str(val_cutoff)[:10],
    }

    with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model_data, f)
    size_mb = os.path.getsize(os.path.join(model_dir, 'model.pkl')) / 1024 / 1024

    meta = {
        'model_type': args.model, 'label': cfg['label'],
        'horizon': cfg['horizon'], 'n_features': len(feature_names),
        'n_trees': n_trees, 'n_train': len(X_full), 'n_test': len(X_test),
        'test_ic': round(test_ic, 4), 'test_rmse': round(test_rmse, 4),
        'test_mae': round(test_mae, 4), 'best_params': core_params,
        'feature_names': feature_names[:50], 'role': cfg['role'],
        'trained_at': datetime.now().isoformat(),
    }
    with open(os.path.join(model_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 模型已保存: {model_dir}/model.pkl ({size_mb:.1f} MB)")
    print(f"  特征: {len(feature_names)}  |  树数: {n_trees}  |  测试 IC: {test_ic:.4f}")

    # ---- 结论 ----
    print(f"\n{'='*60}")
    if test_ic > 0.05:
        print(f" ✅ 样本外有效: Rank IC={test_ic:.4f}")
    elif test_ic > 0.025:
        print(f" ⚠️ 弱有效: Rank IC={test_ic:.4f}，需优化")
    else:
        print(f" ❌ 样本外失效: Rank IC={test_ic:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()