#!/usr/bin/env python3
"""
V9 模型单股回测 — 带校准的价格模拟

对指定股票:
1. 用前60天做校准窗口 (学习 rank→return 映射)
2. 最近1个月预测 vs 实际, 用校准后的预测做价格模拟
"""

import os, sys, pickle, sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import spearmanr, pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'python'))

DB_PATH = os.path.join(ROOT, 'python', 'data', 'stock_data.db')
MODEL_PATH = os.path.join(ROOT, 'python', 'models', 'lgb_hs300_enhanced', 'model.pkl')

from strategy.train_enhanced import calculate_features, detect_market_regime, load_kline_data


def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def predict_one(model_data, feats_row, regime):
    """用 regime 对应的 ensemble 预测"""
    regime_models = model_data['regime_models']
    regime_weights = model_data['regime_weights']

    if regime not in regime_models:
        regime = 'sideways'

    models = regime_models[regime]
    weights = regime_weights[regime]

    X = feats_row.values.reshape(1, -1)
    pred = 0.0
    for name, model in models.items():
        pred += model.predict(X)[0] * weights.get(name, 0)
    return pred


def calibrate_expanding(preds, actuals, warmup=15):
    """
    扩展窗口校准: 去均值 + 波动率匹配

    模型输出是 rank percentile (-0.5~+0.5), 不是 return.
    校准逻辑:
      1. 去预测均值 (消除系统性偏空/偏多)
      2. 乘以 actual_std / pred_std (尺度匹配)
      3. 加上 actual 均值 (保留趋势)
    """
    calibrated = np.copy(preds)
    n = len(preds)

    for i in range(n):
        if i < warmup:
            continue

        past_p = preds[:i]
        past_a = actuals[:i]
        valid = ~(np.isnan(past_p) | np.isnan(past_a))
        if valid.sum() < 10:
            continue

        pp = past_p[valid]
        aa = past_a[valid]

        p_mean = pp.mean()
        p_std = pp.std()
        a_mean = aa.mean()
        a_std = aa.std()

        if p_std < 1e-8:
            continue

        # z-score in pred space, then rescale to actual space
        calibrated[i] = (preds[i] - p_mean) / p_std * a_std + a_mean

    return calibrated


def run_backtest(symbol='300124.SZ', n_days=30, calib_days=60):
    print("=" * 70)
    print(f"  V9 模型单股回测(带校准) — {symbol}")
    print("=" * 70)

    model_data = load_model()
    feature_names = model_data['feature_names']
    horizon = model_data.get('horizon', 3)
    print(f"  模型: {model_data.get('model_version', '?')}, horizon={horizon}")
    print(f"  特征: {len(feature_names)}")

    conn = sqlite3.connect(DB_PATH)
    df_30m = pd.read_sql(f"""
        SELECT t.symbol, t.trade_date,
               first_bar.open as open, t.high, t.low,
               last_bar.close as close, t.volume
        FROM (
            SELECT symbol, substr(date,1,10) as trade_date,
                   MAX(high) as high, MIN(low) as low,
                   SUM(volume) as volume,
                   MIN(date) as first_date, MAX(date) as last_date
            FROM kline_30m WHERE symbol='{symbol}'
            GROUP BY symbol, substr(date,1,10)
        ) t
        JOIN kline_30m first_bar ON first_bar.symbol = t.symbol AND first_bar.date = t.first_date
        JOIN kline_30m last_bar ON last_bar.symbol = t.symbol AND last_bar.date = t.last_date
        ORDER BY t.trade_date
    """, conn)
    conn.close()

    if df_30m.empty:
        print(f"❌ 无数据: {symbol}")
        return

    df_30m['trade_date'] = pd.to_datetime(df_30m['trade_date'])
    print(f"  数据: {len(df_30m)} 天, {df_30m['trade_date'].min().date()} → {df_30m['trade_date'].max().date()}")

    conn = sqlite3.connect(DB_PATH)
    results = calculate_features(df_30m, conn)
    conn.close()

    if symbol not in results:
        print(f"❌ 特征计算失败: {symbol}")
        return

    stock_df, feats = results[symbol]
    feats = feats.fillna(0).replace([np.inf, -np.inf], 0)

    for c in feature_names:
        if c not in feats.columns:
            feats[c] = 0
    feats = feats[feature_names]

    regime = detect_market_regime(stock_df)

    close = stock_df['close'].values
    actual_ret = np.full(len(close), np.nan)
    for i in range(len(close) - horizon):
        actual_ret[i] = (close[i + horizon] - close[i]) / close[i]

    # 取 calib_days + n_days 的窗口
    max_date = stock_df['trade_date'].max()
    cutoff_calib = max_date - timedelta(days=n_days + calib_days)
    cutoff_bt = max_date - timedelta(days=n_days)

    mask_all = (stock_df['trade_date'] >= cutoff_calib) & (~np.isnan(actual_ret))
    mask_bt = (stock_df['trade_date'] >= cutoff_bt) & (~np.isnan(actual_ret))

    dates_all = stock_df['trade_date'].values[mask_all]
    actuals_all = actual_ret[mask_all]
    regimes_all = regime[mask_all]
    feats_all = feats.iloc[mask_all.values]

    print(f"  校准窗口: {calib_days}天, 回测窗口: {n_days}天")
    print(f"  总计 {mask_all.sum()} 天可用\n")

    # 逐日预测 (全窗口)
    preds_raw = []
    for i in range(len(dates_all)):
        r = regimes_all.iloc[i] if hasattr(regimes_all, 'iloc') else regimes_all[i]
        pred = predict_one(model_data, feats_all.iloc[i], r)
        preds_raw.append(pred)

    preds_raw = np.array(preds_raw)

    # 校准
    preds_cal = calibrate_expanding(preds_raw, actuals_all, warmup=15)

    # 找回测起点
    bt_start_idx = 0
    for i, d in enumerate(dates_all):
        if pd.Timestamp(d) >= cutoff_bt:
            bt_start_idx = i
            break

    dates_bt = dates_all[bt_start_idx:]
    actuals_bt = actuals_all[bt_start_idx:]
    preds_raw_bt = preds_raw[bt_start_idx:]
    preds_cal_bt = preds_cal[bt_start_idx:]
    regimes_bt = regimes_all[bt_start_idx:] if isinstance(regimes_all, np.ndarray) else regimes_all.values[bt_start_idx:]

    n_bt = len(dates_bt)
    if n_bt == 0:
        print("❌ 回测天数为0")
        return

    # ===== 结果对比 =====
    print("=" * 70)
    print("  📊 回测结果对比 (原始 vs 校准)")
    print("=" * 70)

    ic_raw = pearsonr(preds_raw_bt, actuals_bt)[0]
    ic_cal = pearsonr(preds_cal_bt, actuals_bt)[0]
    sp_raw = spearmanr(preds_raw_bt, actuals_bt)[0]
    sp_cal = spearmanr(preds_cal_bt, actuals_bt)[0]

    dir_raw = (np.sign(preds_raw_bt) == np.sign(actuals_bt)).mean()
    dir_cal = (np.sign(preds_cal_bt) == np.sign(actuals_bt)).mean()

    mae_raw = np.mean(np.abs(preds_raw_bt - actuals_bt))
    mae_cal = np.mean(np.abs(preds_cal_bt - actuals_bt))

    print(f"                {'原始':>10} {'校准':>10} {'改善':>10}")
    print(f"  Pearson IC:   {ic_raw:>+10.4f} {ic_cal:>+10.4f}")
    print(f"  Spearman IC:  {sp_raw:>+10.4f} {sp_cal:>+10.4f}")
    print(f"  方向准确率:   {dir_raw:>10.1%} {dir_cal:>10.1%} {dir_cal-dir_raw:>+10.1%}")
    print(f"  MAE:          {mae_raw:>10.4f} {mae_cal:>10.4f} {(mae_raw-mae_cal)/mae_raw:>+10.1%}")
    print(f"  预测均值:     {preds_raw_bt.mean():>+10.4f} {preds_cal_bt.mean():>+10.4f} (实际: {actuals_bt.mean():>+.4f})")
    print(f"  预测std:      {preds_raw_bt.std():>10.4f} {preds_cal_bt.std():>10.4f} (实际: {actuals_bt.std():>.4f})")

    # ===== 价格模拟 =====
    print(f"\n{'='*70}")
    print(f"  💰 价格模拟 (基准=100)")
    print(f"{'='*70}")

    # 用每日收益累乘
    base = 100.0
    price_actual = [base]
    price_raw = [base]
    price_cal = [base]

    daily_actual = []
    daily_raw = []
    daily_cal = []

    close_bt_start = close[mask_all][bt_start_idx]

    for i in range(n_bt):
        # horizon=3 的收益分摊到每天 (年化)
        # 简单做法: 直接用 horizon 收益, 每 horizon 天更新一次价格
        pass

    # 更直接: 用 close 价格算实际走势, 用校准预测累乘做模拟
    close_vals = close[mask_all][bt_start_idx:]
    actual_prices = close_vals / close_vals[0] * 100

    # 预测价格: 每天用当日预测的 horizon 日收益, 累乘推进
    pred_price_raw = np.ones(n_bt) * 100
    pred_price_cal = np.ones(n_bt) * 100
    for i in range(1, n_bt):
        # 用 i-1 天的预测收益来推进 (预测的是 horizon 天后的收益)
        # 简化: 把 horizon 天收益平摊到 1 天
        daily_r_raw = preds_raw_bt[i-1] / horizon
        daily_r_cal = preds_cal_bt[i-1] / horizon
        pred_price_raw[i] = pred_price_raw[i-1] * (1 + daily_r_raw)
        pred_price_cal[i] = pred_price_cal[i-1] * (1 + daily_r_cal)

    print(f"\n    {'日期':<12} {'实际价格':>8} {'原始预测':>8} {'校准预测':>8} {'原始误差':>8} {'校准误差':>8}")
    print(f"    {'-'*56}")

    step = max(1, n_bt // 10)
    for i in range(0, n_bt, step):
        d = pd.Timestamp(dates_bt[i]).strftime('%m-%d')
        ap = actual_prices[i] if i < len(actual_prices) else float('nan')
        rp = pred_price_raw[i]
        cp = pred_price_cal[i]
        print(f"    {d:<12} {ap:>8.1f} {rp:>8.1f} {cp:>8.1f} {rp-ap:>+8.1f} {cp-ap:>+8.1f}")

    # 最后一天
    if n_bt > 1:
        i = n_bt - 1
        d = pd.Timestamp(dates_bt[i]).strftime('%m-%d')
        ap = actual_prices[i] if i < len(actual_prices) else float('nan')
        rp = pred_price_raw[i]
        cp = pred_price_cal[i]
        print(f"    {d:<12} {ap:>8.1f} {rp:>8.1f} {cp:>8.1f} {rp-ap:>+8.1f} {cp-ap:>+8.1f}")

    # 汇总
    final_actual = actual_prices[-1] if len(actual_prices) > 0 else 100
    final_raw = pred_price_raw[-1]
    final_cal = pred_price_cal[-1]
    print(f"\n    最终价格: 实际={final_actual:.1f}  原始预测={final_raw:.1f}  校准预测={final_cal:.1f}")
    print(f"    原始误差: {final_raw - final_actual:+.1f} ({(final_raw/final_actual - 1):+.1%})")
    print(f"    校准误差: {final_cal - final_actual:+.1f} ({(final_cal/final_actual - 1):+.1%})")

    # ===== 逐日明细 =====
    print(f"\n{'='*70}")
    print(f"  📅 逐日预测明细")
    print(f"{'='*70}")
    print(f"    {'日期':<12} {'regime':<8} {'原始':>8} {'校准':>8} {'实际':>8} {'方向':>4}")
    for i in range(n_bt):
        d = pd.Timestamp(dates_bt[i]).strftime('%Y-%m-%d')
        r = regimes_bt[i]
        p_raw = preds_raw_bt[i]
        p_cal = preds_cal_bt[i]
        a = actuals_bt[i]
        dir_ok = '✓' if np.sign(p_cal) == np.sign(a) else '✗'
        print(f"    {d:<12} {r:<8} {p_raw:>+8.4f} {p_cal:>+8.2%} {a:>+8.2%} {dir_ok:>4}")


if __name__ == '__main__':
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else '300124.SZ'
    run_backtest(symbol)
