#!/usr/bin/env python3
"""
V9 模型全量回测 — 最近1个月, 369只股票, 每日截面预测

流程:
1. 加载 V9 模型 (regime_models 格式)
2. 计算所有股票特征 (复用 train_enhanced 的特征逻辑)
3. 对每个交易日: 截面预测 → 对比实际 horizon 天收益
4. 输出: IC, ICIR, Top/Bottom 收益, 多空胜率
"""

import os, sys, pickle, sqlite3, time
import numpy as np
import pandas as pd
from datetime import timedelta
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


def predict_ensemble(model_data, X, regime):
    """用 regime 对应的 ensemble 预测"""
    regime_models = model_data['regime_models']
    regime_weights = model_data['regime_weights']
    if regime not in regime_models:
        regime = 'sideways'
    models = regime_models[regime]
    weights = regime_weights[regime]
    pred = np.zeros(len(X))
    for name, model in models.items():
        pred += model.predict(X) * weights.get(name, 0)
    return pred


def run_backtest(n_days=30):
    print("=" * 70)
    print("  V9 模型全量回测 — 最近1个月, 全部股票")
    print("=" * 70)

    model_data = load_model()
    feature_names = model_data['feature_names']
    horizon = model_data.get('horizon', 3)
    print(f"  模型: {model_data.get('model_version', '?')}")
    print(f"  特征: {len(feature_names)}, horizon={horizon}")

    conn = sqlite3.connect(DB_PATH)
    t0 = time.time()
    df = load_kline_data(conn, max_stocks=0)
    print(f"  加载 {len(df['symbol'].unique())} 只股票, {len(df)} 条日线 ({time.time()-t0:.0f}s)")

    t0 = time.time()
    results = calculate_features(df, conn)
    conn.close()
    print(f"  特征计算完成: {len(results)} 只股票 ({time.time()-t0:.0f}s)")

    # 收集所有股票的特征 + 目标 + regime
    print("\n📦 准备回测数据...")
    stock_data = {}  # {sym: DataFrame(date, feats, actual, regime)}
    for sym, (stock_df, feats) in results.items():
        feats = feats.fillna(0).replace([np.inf, -np.inf], 0)
        for c in feature_names:
            if c not in feats.columns:
                feats[c] = 0
        feats = feats[feature_names]

        close = stock_df['close'].values
        target = np.full(len(close), np.nan)
        for i in range(len(close) - horizon):
            target[i] = (close[i + horizon] - close[i]) / close[i]

        regime = detect_market_regime(stock_df)
        dates = stock_df['trade_date'].values

        valid = ~np.isnan(target)
        if valid.sum() == 0:
            continue

        stock_data[sym] = pd.DataFrame({
            'date': dates[valid],
            'feats': list(feats.values[valid]),
            'actual': target[valid],
            'regime': regime[valid],
        })

    # 收集所有回测日
    all_dates = set()
    for sdf in stock_data.values():
        all_dates.update(sdf['date'].values)
    all_dates = sorted(all_dates)

    # 取最近 n_days 天
    max_date = pd.Timestamp(all_dates[-1])
    cutoff = max_date - timedelta(days=n_days)
    bt_dates = [d for d in all_dates if pd.Timestamp(d) >= cutoff]
    print(f"  回测区间: {cutoff.date()} → {max_date.date()} ({len(bt_dates)} 天)\n")

    # 逐日截面预测
    all_ics = []
    all_top_rets = []
    all_bot_rets = []
    all_long_short = []

    for date in bt_dates:
        # 收集该日所有股票的数据
        day_preds = []  # [(pred, actual, regime)]
        for sym, sdf in stock_data.items():
            mask = sdf['date'].values == date
            if not mask.any():
                continue
            idx = mask.argmax()
            row = sdf.iloc[idx]
            X = row['feats'].reshape(1, -1)
            regime = row['regime']
            pred = predict_ensemble(model_data, X, regime)[0]
            day_preds.append((pred, row['actual'], regime))

        if len(day_preds) < 10:
            continue

        preds = np.array([x[0] for x in day_preds])
        actuals = np.array([x[1] for x in day_preds])
        regimes = np.array([x[2] for x in day_preds])

        ic_sp = spearmanr(preds, actuals)[0]
        ic_pe = pearsonr(preds, actuals)[0]
        all_ics.append(ic_sp)

        sort_idx = np.argsort(preds)
        n = len(day_preds)
        top_n = max(5, n // 10)
        top_ret = actuals[sort_idx[-top_n:]].mean()
        bot_ret = actuals[sort_idx[:top_n]].mean()
        all_top_rets.append(top_ret)
        all_bot_rets.append(bot_ret)
        all_long_short.append(top_ret - bot_ret)

        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        regime_str = '/'.join(f'{r}:{(regimes==r).sum()}' for r in np.unique(regimes))
        print(f"  {date_str} [{regime_str}] IC_sp={ic_sp:+.4f} IC_pe={ic_pe:+.4f} | "
              f"Top{top_n}={top_ret:+.2%} Bot{top_n}={bot_ret:+.2%} "
              f"多空={top_ret-bot_ret:+.2%} ({n}只)")

    print("\n" + "=" * 70)
    print("  📊 V9 全量回测汇总")
    print("=" * 70)
    if all_ics:
        print(f"  平均 Spearman IC: {np.mean(all_ics):+.4f} "
              f"(正向={sum(1 for x in all_ics if x>0)}/{len(all_ics)})")
        icir = np.mean(all_ics) / np.std(all_ics) if np.std(all_ics) > 0 else 0
        print(f"  ICIR (稳定性):    {icir:.3f}")
        print(f"  平均 Top收益:     {np.mean(all_top_rets):+.2%}")
        print(f"  平均 Bot收益:     {np.mean(all_bot_rets):+.2%}")
        print(f"  平均多空差:       {np.mean(all_long_short):+.2%}")
        win = sum(1 for x in all_long_short if x > 0)
        print(f"  多空胜率:         {win}/{len(all_long_short)} ({win/len(all_long_short):.0%})")


if __name__ == '__main__':
    run_backtest()
