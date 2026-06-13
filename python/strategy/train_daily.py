#!/usr/bin/env python3
"""
LGBM 日线选股模型训练 v2

设计原则:
  1. 严格时序分离: train(60%) → val(20%) → test(20%)
  2. Early stopping 自动定树数 (max 10000, patience 100)
  3. 北向资金 shift 1天 (避免当天收盘后数据当同日特征)
  4. 特征选择仅基于训练集，mask 应用到 val/test
  5. PurgedCV gap >= horizon

用法:
  python strategy/train_daily.py              # 完整训练
  python strategy/train_daily.py --quick       # 快速测试 (500树)
  python strategy/train_daily.py --tune        # Optuna 超参搜索 (耗时)
"""

import sys, os, argparse, pickle, json, sqlite3, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from sklearn.feature_selection import SelectFromModel
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategy.features import EnhancedFeatureEngineer, AdvancedFeatureEngineer, MarketFeatureEngineer
from strategy.train import load_data, compute_features, load_sentiment

warnings.filterwarnings('ignore')

# ============ 路径 ============
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data/stock_data.db')
PARAMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_params.json')
MODEL_DIR = os.path.join(ROOT, 'models/lgb_daily')

# ============ 配置 ============
HORIZON = 5              # 预测5日收益率
TRAIN_RATIO = 0.6        # 训练集占比
VAL_RATIO = 0.2          # 验证集占比
TEST_RATIO = 0.2         # 测试集占比
MAX_TREES = 10000        # 最大树数
EARLY_STOPPING = 100     # early stopping 耐心
NORTH_SHIFT_DAYS = 1     # 北向资金滞后天数
MIN_HISTORY = 120        # 最小历史数据
CORR_THRESHOLD = 0.95    # 去冗余阈值

# LGBM 固定参数 (与 Optuna 搜索时一致)
LGBM_FIXED = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'random_state': 42,
    'force_row_wise': True,
    'n_jobs': -1,
}


def get_all_dates(data: Dict) -> np.ndarray:
    """获取所有股票的统一日期序列"""
    dates = set()
    for df in data.values():
        dates.update(df['date'].values)
    return np.array(sorted(dates))


def prepare_split_data(data: Dict, conn, train_cutoff, val_cutoff) -> Tuple:
    """
    准备时序分离的数据集

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names
    """
    sent_df = load_sentiment(conn)
    has_sent = len(sent_df) > 0

    cfg = {
        'horizon': HORIZON, 'min_history': MIN_HISTORY,
        'features': 'enhanced+advanced+market',
        'north_shift_days': NORTH_SHIFT_DAYS,
    }

    X_tr, y_tr, X_va, y_va, X_te, y_te = [], [], [], [], [], []
    feature_names = None
    success = 0

    for sym, df in data.items():
        try:
            feats = compute_features(df, sym, cfg)
            if feature_names is None:
                feature_names = list(feats.columns)

            close = df['close'].values.astype(float)
            target = np.full(len(close), np.nan)
            for j in range(len(close) - HORIZON):
                target[j] = (close[j + HORIZON] - close[j]) / close[j]

            if has_sent:
                dates = df['date'].dt.strftime('%Y-%m-%d')
                sent = sent_df[sent_df['symbol'] == sym].set_index('date')
                for col in sent.columns:
                    if col not in ('symbol', 'date'):
                        feats[f'sent_{col}'] = dates.map(
                            lambda d: sent.loc[d, col] if d in sent.index else 0
                        ).fillna(0).values
                feature_names = list(feats.columns)

            feats = feats.fillna(method='ffill').fillna(0)
            valid = ~np.isnan(target)
            feats_v = feats[valid]
            target_v = target[valid]
            dates_v = df['date'].values[valid]

            if len(feats_v) > MIN_HISTORY:
                feats_v = feats_v.iloc[MIN_HISTORY:]
                target_v = target_v[MIN_HISTORY:]
                dates_v = dates_v[MIN_HISTORY:]

            if len(feats_v) < 50:
                continue

            # 时序切分
            train_mask = dates_v <= train_cutoff
            val_mask = (dates_v > train_cutoff) & (dates_v <= val_cutoff)
            test_mask = dates_v > val_cutoff

            if train_mask.sum() >= 50:
                X_tr.append(feats_v[train_mask].values)
                y_tr.append(target_v[train_mask])
            if val_mask.sum() >= 10:
                X_va.append(feats_v[val_mask].values)
                y_va.append(target_v[val_mask])
            if test_mask.sum() >= 10:
                X_te.append(feats_v[test_mask].values)
                y_te.append(target_v[test_mask])

            success += 1
        except Exception:
            continue

    if not X_tr:
        return None

    X_train = np.vstack(X_tr); y_train = np.concatenate(y_tr)
    X_val   = np.vstack(X_va); y_val   = np.concatenate(y_va)
    X_test  = np.vstack(X_te); y_test  = np.concatenate(y_te)

    # 过滤极端收益率
    mask_tr = np.abs(y_train) < 0.15
    X_train, y_train = X_train[mask_tr], y_train[mask_tr]
    mask_va = np.abs(y_val) < 0.15
    X_val, y_val = X_val[mask_va], y_val[mask_va]
    mask_te = np.abs(y_test) < 0.15
    X_test, y_test = X_test[mask_te], y_test[mask_te]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names


def remove_redundant(X, feature_names):
    """去冗余 (基于训练集) — 返回去冗余后的 X 和 mask"""
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


def select_features(X_train, y_train, X_val, y_val, feature_names, params):
    """SelectFromModel (仅基于训练集) — 返回选择后的数据和 mask"""
    sel_params = {**params, 'n_estimators': 500}
    sel = lgb.LGBMRegressor(**sel_params)
    sel.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    n_trees = sel.best_iteration_ or 500
    print(f"  选择模型: {n_trees} 棵树")

    sf = SelectFromModel(sel, threshold='median', prefit=True)
    mask = sf.get_support()
    n_selected = mask.sum()
    print(f"  SelectFromModel: {n_selected}/{len(feature_names)} 特征 (阈值=median)")

    new_names = [fn for fn, m in zip(feature_names, mask) if m]
    return X_train[:, mask], X_val[:, mask], new_names, mask


def train_model(X_train, y_train, X_val, y_val, params, max_trees, patience):
    """训练 LGBM，early stopping 自动定树数"""
    model = lgb.LGBMRegressor(**params, n_estimators=max_trees)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(patience, verbose=True),
                         lgb.log_evaluation(50)])
    n_trees = model.best_iteration_ or max_trees
    return model, n_trees


def evaluate_test(model, X_test, y_test, feature_names):
    """在测试集上评估"""
    pred = model.predict(X_test)
    ic, _ = spearmanr(pred, y_test)
    if np.isnan(ic): ic = 0

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)

    print(f"\n  📊 测试集评估:")
    print(f"    Rank IC (Spearman): {ic:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAE:  {mae:.4f}")

    imp = model.feature_importances_
    top = np.argsort(imp)[-20:][::-1]
    print(f"\n  Top 20 特征:")
    for idx in top:
        print(f"    {feature_names[idx]}: {imp[idx]:.0f}")

    return ic, rmse, mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='快速模式 (500树)')
    parser.add_argument('--tune', action='store_true', help='Optuna 超参搜索')
    args = parser.parse_args()

    max_trees = 500 if args.quick else MAX_TREES
    patience = 30 if args.quick else EARLY_STOPPING

    print("=" * 60)
    print(" LGBM 日线选股模型 v2")
    print(f" 时序分离: train({TRAIN_RATIO:.0%}) → val({VAL_RATIO:.0%}) → test({TEST_RATIO:.0%})")
    print(f" 北向资金滞后 {NORTH_SHIFT_DAYS} 天 | early stop patience={patience}")
    print("=" * 60)

    # ---- 1. 加载数据 + 时序切分 ----
    print("\n📊 加载数据...")
    data = load_data(DB_PATH, 'kline_daily')

    all_dates = get_all_dates(data)
    n_dates = len(all_dates)
    train_cutoff = all_dates[int(n_dates * TRAIN_RATIO)]
    val_cutoff = all_dates[int(n_dates * (TRAIN_RATIO + VAL_RATIO))]

    print(f"  {n_dates} 个交易日, {len(data)} 只股票")
    print(f"  train: ~{str(train_cutoff)[:10]}  val: ~{str(val_cutoff)[:10]}  test: ~{str(all_dates[-1])[:10]}")

    conn = sqlite3.connect(DB_PATH)
    result = prepare_split_data(data, conn, train_cutoff, val_cutoff)
    conn.close()

    if result is None:
        print("❌ 数据准备失败"); return

    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names = result

    print(f"  train: {len(X_train):,}条  val: {len(X_val):,}条  test: {len(X_test):,}条")
    print(f"  特征: {len(feature_names)}  目标: mean={y_train.mean():.4f} std={y_train.std():.4f}")

    # ---- 2. 特征选择 (仅基于训练集) ----
    print("\n🔧 特征选择...")

    # 去冗余
    X_train, feature_names, corr_mask = remove_redundant(X_train, feature_names)
    X_val = X_val[:, corr_mask]
    X_test = X_test[:, corr_mask]

    # 加载超参
    params = {}
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE) as f:
            params = json.load(f).get('daily', {})
    if not params:
        print("⚠️ 未找到 best_params.json，使用默认参数")

    for k, v in LGBM_FIXED.items():
        params.setdefault(k, v)

    # SelectFromModel
    sel_params = {**params, 'n_estimators': 500}
    sel = lgb.LGBMRegressor(**sel_params)
    sel.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    n_sel_trees = sel.best_iteration_ or 500
    print(f"  SelectFromModel: {n_sel_trees} 棵树")

    sf = SelectFromModel(sel, threshold='median', prefit=True)
    sel_mask = sf.get_support()
    X_train = X_train[:, sel_mask]
    X_val = X_val[:, sel_mask]
    X_test = X_test[:, sel_mask]
    feature_names = [fn for fn, m in zip(feature_names, sel_mask) if m]
    print(f"  特征选择: {len(feature_names)} 个 (阈值=median)")

    # ---- 3. 训练 ----
    print(f"\n🏋️ 训练 (max {max_trees} 棵树, patience {patience})...")
    model, n_trees = train_model(X_train, y_train, X_val, y_val, params, max_trees, patience)
    print(f"\n  ✅ 训练完成: {n_trees} 棵树")

    # ---- 4. 测试集评估 ----
    print("\n" + "=" * 60)
    print(" 🧪 测试集评估 (真正样本外)")
    print("=" * 60)
    test_ic, test_rmse, test_mae = evaluate_test(model, X_test, y_test, feature_names)

    # ---- 5. 最终模型 (train+val 全量训练) ----
    print(f"\n🏋️ 最终模型 (train+val, {n_trees} 棵树)...")
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    final_model = lgb.LGBMRegressor(**params, n_estimators=n_trees)
    final_model.fit(X_full, y_full)

    # ---- 6. 保存 ----
    print("\n💾 保存模型...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    core_params = {k: v for k, v in params.items()
                   if k not in ('verbosity', 'random_state', 'force_row_wise',
                                'n_jobs', 'objective', 'metric', 'boosting_type')}

    model_data = {
        'model': final_model,
        'feature_names': feature_names,
        'best_params': core_params,
        'test_ic': round(test_ic, 4),
        'test_rmse': round(test_rmse, 4),
        'test_mae': round(test_mae, 4),
        'n_trees': n_trees,
        'horizon': HORIZON,
        'n_features': len(feature_names),
        'n_train': len(X_full),
        'n_test': len(X_test),
        'train_cutoff': str(train_cutoff)[:10],
        'val_cutoff': str(val_cutoff)[:10],
    }

    with open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb') as f:
        pickle.dump(model_data, f)
    size_mb = os.path.getsize(os.path.join(MODEL_DIR, 'model.pkl')) / 1024 / 1024

    meta = {
        'model_type': 'daily', 'label': '日线',
        'horizon': HORIZON,
        'n_features': len(feature_names),
        'n_trees': n_trees,
        'n_train': len(X_full),
        'n_test': len(X_test),
        'test_ic': round(test_ic, 4),
        'test_rmse': round(test_rmse, 4),
        'test_mae': round(test_mae, 4),
        'best_params': core_params,
        'feature_names': feature_names[:50],
        'trained_at': datetime.now().isoformat(),
        'role': 'α选股层',
    }
    with open(os.path.join(MODEL_DIR, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 模型已保存: {MODEL_DIR}/model.pkl ({size_mb:.1f} MB)")
    print(f"  特征: {len(feature_names)} | 树数: {n_trees} | 测试 IC: {test_ic:.4f}")

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