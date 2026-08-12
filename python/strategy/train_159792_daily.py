#!/usr/bin/env python3
"""
ETF 日线级别择时模型
- 用日线数据 (1164条, 覆盖更长时间)
- 分类目标: 预测未来1-5天涨跌
- 特征: 日线级别技术指标 + 资金流
"""

import sys, os, argparse, pickle, json, sqlite3, warnings, time

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data/stock_data.db')
MODEL_DIR = os.path.join(ROOT, 'models/lgb_etf')

SYMBOL = '159792.SZ'

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

HORIZON = 3           # 预测未来3天
SKIP_BARS = 1         # 日线不用下采样
RETURN_CLIP = 0.15

LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'max_depth': 5,
    'learning_rate': 0.01,
    'n_estimators': 2000,
    'early_stopping_rounds': 100,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
    'reg_alpha': 1.0,
    'reg_lambda': 2.0,
    'min_child_samples': 20,
    'min_split_gain': 0.01,
    'verbosity': -1,
    'random_state': None,
    'n_jobs': 3,
    'force_row_wise': True,
    'is_unbalance': True,
}

N_MODELS = 5
SEEDS = [42, 123, 456, 789, 1024]


def compute_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """日线级别特征"""
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    open_ = df['open'].values.astype(float)
    vol = df['volume'].values.astype(float)
    
    feats = {}
    close = np.maximum(close, 1e-10)
    
    # ---- 收益率 (多周期) ----
    for w in [1, 3, 5, 10, 20, 60]:
        feats[f'ret_{w}'] = pd.Series(close).pct_change(w).values
    
    # 波动率
    for w in [5, 10, 20, 60]:
        feats[f'volatility_{w}'] = pd.Series(feats['ret_1']).rolling(w).std().values
    
    # 均线偏离
    for w in [5, 10, 20, 60, 120]:
        ma = pd.Series(close).rolling(w).mean().values
        feats[f'ma_dev_{w}'] = (close - ma) / (ma + 1e-10)
        feats[f'ma_slope_{w}'] = pd.Series(ma).pct_change(3).values
    
    # 价格位置
    for w in [20, 60, 120]:
        hh = pd.Series(high).rolling(w).max().values
        ll = pd.Series(low).rolling(w).min().values
        feats[f'price_pos_{w}'] = (close - ll) / (hh - ll + 1e-10)
    
    # ---- 成交量 ----
    for w in [5, 20, 60]:
        vma = pd.Series(vol).rolling(w).mean().values
        feats[f'vol_ratio_{w}'] = vol / (vma + 1e-10)
    
    feats['vol_change_1'] = pd.Series(vol).pct_change(1).values
    feats['vol_change_5'] = pd.Series(vol).pct_change(5).values
    feats['vol_abnormal'] = vol / (pd.Series(vol).rolling(20).mean().values + 1e-10) - 1.0
    
    # ---- MACD ----
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
    macd_l = ema12 - ema26
    sig = pd.Series(macd_l).ewm(span=9, adjust=False).mean().values
    feats['macd_hist'] = (macd_l - sig) / (close + 1e-10)
    feats['macd_line'] = macd_l / (close + 1e-10)
    feats['macd_trend'] = pd.Series(macd_l).diff(5).values / (close + 1e-10)
    
    # ---- RSI ----
    for w in [6, 14, 28]:
        delta = pd.Series(close).diff()
        gain = delta.clip(lower=0).ewm(span=w, adjust=False).mean().values
        loss = (-delta).clip(lower=0).ewm(span=w, adjust=False).mean().values
        feats[f'rsi_{w}'] = 100 - 100 / (1 + gain / (loss + 1e-10))
    
    # ---- KDJ ----
    for w in [9, 14, 28]:
        l_n = pd.Series(low).rolling(w).min().values
        h_n = pd.Series(high).rolling(w).max().values
        rsv = (close - l_n) / (h_n - l_n + 1e-10) * 100
        k = pd.Series(rsv).ewm(com=2, adjust=False).mean().values
        d = pd.Series(k).ewm(com=2, adjust=False).mean().values
        feats[f'k_{w}'] = k
        feats[f'd_{w}'] = d
        feats[f'j_{w}'] = 3 * k - 2 * d
    
    # ---- 布林带 ----
    for w in [20, 60]:
        ma = pd.Series(close).rolling(w).mean().values
        std = pd.Series(close).rolling(w).std().values
        feats[f'bb_pos_{w}'] = (close - ma) / (2 * std + 1e-10)
        feats[f'bb_width_{w}'] = (4 * std) / (ma + 1e-10)
    
    # ---- ATR ----
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr = np.maximum(tr, np.abs(low - np.roll(close, 1)))
    atr_14 = pd.Series(tr).rolling(14).mean().values
    feats['atr_ratio'] = atr_14 / (close + 1e-10)
    
    # ---- 趋势 ----
    for w in [5, 10, 20, 60]:
        feats[f'trend_{w}'] = pd.Series(close).diff(w).values / (close + 1e-10)
    
    # ---- OBV ----
    obv = np.zeros(len(close))
    for i in range(1, len(close)):
        obv[i] = obv[i-1] + vol[i] * np.sign(close[i] - close[i-1])
    for w in [5, 20]:
        feats[f'obv_roc_{w}'] = pd.Series(obv).pct_change(w).values
    
    # ---- 创新高/新低 ----
    for w in [20, 60]:
        feats[f'is_n_high_{w}'] = (high == pd.Series(high).rolling(w).max()).astype(float)
        feats[f'is_n_low_{w}'] = (low == pd.Series(low).rolling(w).min()).astype(float)
    
    # ---- 连续涨跌 ----
    for w in [3, 5, 10]:
        feats[f'up_days_{w}'] = pd.Series((close > np.roll(close, 1)).astype(int)).rolling(w).sum().values
        feats[f'down_days_{w}'] = pd.Series((close < np.roll(close, 1)).astype(int)).rolling(w).sum().values
    
    # ---- K线形态 ----
    body = np.abs(close - open_)
    total_range = high - low + 1e-10
    feats['body_ratio'] = body / total_range
    feats['is_up'] = (close > open_).astype(float)
    feats['gap'] = ((open_ - np.roll(close, 1)) / (np.roll(close, 1) + 1e-10)).clip(-0.1, 0.1)
    
    # ---- 收益分布 ----
    for w in [10, 20, 60]:
        r_roll = pd.Series(feats['ret_1']).rolling(w)
        feats[f'ret_skew_{w}'] = r_roll.skew().values
        feats[f'ret_kurt_{w}'] = r_roll.kurt().values
        feats[f'ret_max_{w}'] = r_roll.max().values
        feats[f'ret_min_{w}'] = r_roll.min().values
    
    # ---- 时序 ----
    if 'date' in df.columns:
        dates = pd.to_datetime(df['date'], format='mixed')
        feats['month'] = dates.dt.month.values / 12.0
        feats['weekday'] = dates.dt.weekday.values / 7.0
        feats['day_of_month'] = dates.dt.day.values / 31.0
    
    # ---- 量价关系 ----
    feats['vol_price_corr_20'] = pd.Series(vol).rolling(20).corr(pd.Series(close)).values
    
    # ---- 回撤 ----
    for w in [20, 60]:
        peak = pd.Series(close).rolling(w).max().values
        feats[f'drawdown_{w}'] = (close - peak) / (peak + 1e-10)
    
    result = pd.DataFrame(feats)
    result = result.ffill().fillna(0)
    result = result.clip(-1e6, 1e6)
    return result


def load_daily_data(db_path, symbol):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM kline_daily WHERE symbol=? ORDER BY date", conn, params=(symbol,))
    conn.close()
    if len(df) == 0:
        raise ValueError(f"无数据: {symbol}")
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    df = df.sort_values('date').reset_index(drop=True)
    df = df.drop_duplicates(subset=['date']).reset_index(drop=True)
    print(f"  加载 {symbol}: {len(df)} 条日线 ({df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['date'].iloc[-1].strftime('%Y-%m-%d')})")
    return df


def build_targets(df, horizon, min_ret=0.003):
    close = df['close'].values.astype(float)
    n = len(close)
    target = np.full(n, np.nan)
    actual_ret = np.full(n, np.nan)
    
    for i in range(n - horizon):
        if close[i] <= 0 or close[i + horizon] <= 0:
            continue
        r = (close[i + horizon] - close[i]) / close[i]
        actual_ret[i] = r
        if abs(r) < min_ret:
            target[i] = np.nan
        else:
            target[i] = 1.0 if r > 0 else 0.0
    
    valid = (~np.isnan(target)).sum()
    up = (target == 1).sum()
    down = (target == 0).sum()
    print(f"  目标(h={horizon}天): 有效={valid} 涨={up}({up/valid*100:.1f}%) 跌={down}({down/valid*100:.1f}%)")
    print(f"  收益分布: mean={np.nanmean(actual_ret):+.4f} std={np.nanstd(actual_ret):.4f} "
          f"min={np.nanmin(actual_ret):+.4f} max={np.nanmax(actual_ret):+.4f}")
    return target, actual_ret


def prepare_samples(df, target, skip_bars, train_ratio, val_ratio):
    n = len(df)
    train_cut = int(n * train_ratio)
    val_cut = int(n * (train_ratio + val_ratio))
    
    print(f"  时序切分: train=0:{train_cut} val={train_cut}:{val_cut} test={val_cut}:{n}")
    
    print("  计算日线特征...")
    t0 = time.time()
    features = compute_daily_features(df)
    feature_names = list(features.columns)
    print(f"  特征: {len(feature_names)} 个, 耗时 {time.time()-t0:.0f}s")
    
    X_all = features.values.astype(np.float32)
    
    valid_mask = ~np.isnan(target)
    valid_indices = np.where(valid_mask)[0]
    sampled = valid_indices[::skip_bars]
    
    train_idx = sampled[sampled < train_cut]
    val_idx = sampled[(sampled >= train_cut) & (sampled < val_cut)]
    test_idx = sampled[sampled >= val_cut]
    
    X_train, y_train = X_all[train_idx], target[train_idx].astype(int)
    X_val, y_val = X_all[val_idx], target[val_idx].astype(int)
    X_test, y_test = X_all[test_idx], target[test_idx].astype(int)
    
    print(f"  样本: train={len(X_train)}(↑{sum(y_train)}) val={len(X_val)}(↑{sum(y_val)}) test={len(X_test)}(↑{sum(y_test)})")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names


def train_one(seed, X_train, y_train, X_val, y_val, feature_names, params):
    t0 = time.time()
    p = {**params, 'random_state': seed}
    n_est = p.pop('n_estimators')
    es = p.pop('early_stopping_rounds')
    p.pop('n_jobs', None)
    
    model = lgb.LGBMClassifier(n_estimators=n_est, **p)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)] if len(X_val) > 0 else None,
              eval_metric='binary_logloss',
              callbacks=[lgb.log_evaluation(100), lgb.early_stopping(es)])
    
    n_trees = model.n_estimators_
    elapsed = time.time() - t0
    
    if len(X_val) > 0:
        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, zero_division=0)
        print(f"  [seed={seed}] {n_trees}棵 | val Acc={val_acc:.3f} F1={val_f1:.3f} | {elapsed:.0f}s")
    else:
        val_acc = val_f1 = 0
        print(f"  [seed={seed}] {n_trees}棵 | {elapsed:.0f}s")
    
    return {
        'model': model, 'seed': seed, 'n_trees': n_trees,
        'val_acc': round(val_acc, 4), 'val_f1': round(val_f1, 4),
        'train_time_s': round(elapsed, 1)
    }


def train_ensemble(X_train, y_train, X_val, y_val, feature_names, params,
                   n_models=N_MODELS, seeds=SEEDS):
    print(f"\n🏋️ 并行训练 {n_models} 个模型...")
    t0 = time.time()
    
    results = Parallel(n_jobs=n_models)(
        delayed(train_one)(seeds[i], X_train, y_train, X_val, y_val, feature_names, params)
        for i in range(n_models)
    )
    
    avg_trees = sum(r['n_trees'] for r in results) / len(results)
    avg_acc = sum(r['val_acc'] for r in results) / len(results)
    avg_f1 = sum(r['val_f1'] for r in results) / len(results)
    
    print(f"\n  ✅ {n_models}模型完成: {time.time()-t0:.0f}s, "
          f"avg {avg_trees:.0f}棵, val Acc={avg_acc:.3f} F1={avg_f1:.3f}")
    
    return results


def evaluate_ensemble(models_info, X_test, y_test, feature_names):
    n = len(models_info)
    print(f"\n{'='*60}")
    print(f" 🧪 测试集评估 ({len(X_test)}条, {n}模型)")
    print(f"{'='*60}")
    print(f"  分布: ↑{sum(y_test)}({sum(y_test)/len(y_test)*100:.1f}%) ↓{len(y_test)-sum(y_test)}")
    
    all_proba = []
    for info in models_info:
        proba = info['model'].predict_proba(X_test)[:, 1]
        all_proba.append(proba)
        pred = info['model'].predict(X_test)
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        print(f"  [seed={info['seed']}] Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")
    
    if n > 1:
        ensemble_proba = np.mean(all_proba, axis=0)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        acc = accuracy_score(y_test, ensemble_pred)
        prec = precision_score(y_test, ensemble_pred, zero_division=0)
        rec = recall_score(y_test, ensemble_pred, zero_division=0)
        f1 = f1_score(y_test, ensemble_pred, zero_division=0)
        print(f"  {'─'*50}")
        print(f"  🏆 Ensemble({n}) → Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")
    else:
        ensemble_proba = all_proba[0]
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        acc = accuracy_score(y_test, ensemble_pred)
    
    if n > 1:
        print(f"\n  📊 分组回测 (按预测概率分5组):")
        sort_idx = np.argsort(ensemble_proba)
        n_per = len(sort_idx) // 5
        for g in range(5):
            s, e = g * n_per, (g + 1) * n_per if g < 4 else len(sort_idx)
            preds = ensemble_pred[sort_idx[s:e]]
            actual = y_test[sort_idx[s:e]]
            ga = actual.mean()
            gh = accuracy_score(actual, preds)
            print(f"    G{g+1}: prob={ensemble_proba[sort_idx[s:e]].mean():.3f} | actual↑={ga:.1%} | Hit={gh:.3f}")
    
    avg_imp = np.zeros(len(feature_names))
    for info in models_info:
        avg_imp += info['model'].feature_importances_
    avg_imp /= n
    top_idx = np.argsort(avg_imp)[-15:][::-1]
    print(f"\n  Top 15 特征:")
    for idx in top_idx:
        print(f"    {feature_names[idx]}: {int(avg_imp[idx])}")
    
    return acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default=SYMBOL)
    parser.add_argument('--horizon', type=int, default=HORIZON)
    parser.add_argument('--skip', type=int, default=SKIP_BARS)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--db', type=str, default=DB_PATH)
    args = parser.parse_args()
    
    params = LGBM_PARAMS
    if args.quick:
        params = {**LGBM_PARAMS, 'n_estimators': 300, 'learning_rate': 0.05}
    
    print("=" * 60)
    print(f" ETF日线级别择时模型")
    print(f" 标的: {args.symbol} | horizon={args.horizon}天")
    print(f" 时序: train({TRAIN_RATIO:.0%}) → val({VAL_RATIO:.0%}) → test({TEST_RATIO:.0%})")
    print(f" Ens: {N_MODELS}模型 | lr={params['learning_rate']}")
    print("=" * 60)
    
    print(f"\n📊 加载日线数据...")
    df = load_daily_data(args.db, args.symbol)
    
    print(f"\n🎯 构建分类目标 (horizon={args.horizon}天)...")
    target, actual_ret = build_targets(df, args.horizon)
    
    print(f"\n🔧 特征工程 + 样本准备...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names = \
        prepare_samples(df, target, args.skip, TRAIN_RATIO, VAL_RATIO)
    
    models_info = train_ensemble(X_train, y_train, X_val, y_val, feature_names, params)
    
    if len(X_test) > 0:
        test_acc, test_f1 = evaluate_ensemble(models_info, X_test, y_test, feature_names)
    else:
        test_acc = test_f1 = 0
    
    print(f"\n🏋️ 训练最终模型 (train+val 全量)...")
    if len(X_val) > 0:
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
    else:
        X_full, y_full = X_train, y_train
    
    final_models = []
    for info in models_info:
        m = lgb.LGBMClassifier(
            **{k: v for k, v in params.items()
               if k not in ('n_estimators', 'early_stopping_rounds', 'n_jobs', 'random_state')},
            n_estimators=info['n_trees'],
            random_state=info['seed']
        )
        m.fit(X_full, y_full)
        final_models.append(m)
    
    print(f"\n💾 保存模型...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    safe_name = args.symbol.replace('.', '_')
    model_pkg = {
        'models': final_models,
        'feature_names': feature_names,
        'symbol': args.symbol,
        'horizon': args.horizon,
        'model_type': 'etf_daily_cls',
        'n_models': len(models_info),
        'data_type': 'daily',
        'train_date': datetime.now().strftime('%Y-%m-%d'),
        'train_samples': len(X_full),
        'test_acc': round(test_acc, 4),
        'test_f1': round(test_f1, 4),
    }
    
    model_path = os.path.join(MODEL_DIR, f'{safe_name}_daily.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_pkg, f)
    
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    
    meta = {
        'symbol': args.symbol, 'model_type': 'etf_daily_cls',
        'horizon': args.horizon, 'n_features': len(feature_names),
        'n_models': len(models_info), 'n_train': len(X_full),
        'n_test': len(X_test), 'test_acc': round(test_acc, 4),
        'test_f1': round(test_f1, 4),
        'trained_at': datetime.now().isoformat(), 'size_mb': round(size_mb, 1),
    }
    with open(os.path.join(MODEL_DIR, f'{safe_name}_daily_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f" ✅ 日线模型已保存: {model_path} ({size_mb:.1f} MB)")
    print(f"    特征: {len(feature_names)} | 模型: {len(models_info)}")
    print(f"    测试 Acc: {test_acc:.3f} | F1: {test_f1:.3f}")
    
    if test_acc > 0.55:
        print(f" ✅ Acc > 55%, 有效")
    elif test_acc > 0.52:
        print(f" ⚠️ Acc > 52%, 弱有效")
    else:
        print(f" ❌ Acc <= 52%, 信号弱")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()