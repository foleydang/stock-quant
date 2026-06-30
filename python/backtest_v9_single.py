#!/usr/bin/env python3
"""
V9 模型单股回测 — 验证是否过拟合

对指定股票, 用最近1个月每日预测 vs 实际收益
"""

import os, sys, pickle, sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'python'))

DB_PATH = os.path.join(ROOT, 'python', 'data', 'stock_data.db')
MODEL_PATH = os.path.join(ROOT, 'python', 'models', 'lgb_hs300_enhanced', 'model.pkl')

# 复用 train_enhanced 的特征计算
from strategy.train_enhanced import calculate_features, detect_market_regime, load_kline_data


def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def predict_one(model_data, feats_row, regime):
    """用 regime 对应的 ensemble 预测"""
    regime_models = model_data['regime_models']
    regime_weights = model_data['regime_weights']

    # 如果该 regime 没模型, 用 sideways
    if regime not in regime_models:
        regime = 'sideways'

    models = regime_models[regime]
    weights = regime_weights[regime]

    X = feats_row.values.reshape(1, -1)
    pred = 0.0
    for name, model in models.items():
        pred += model.predict(X)[0] * weights.get(name, 0)
    return pred


def run_backtest(symbol='300124.SZ', n_days=30):
    print("=" * 70)
    print(f"  V9 模型单股回测 — {symbol} 最近{n_days}天")
    print("=" * 70)

    model_data = load_model()
    feature_names = model_data['feature_names']
    print(f"  模型: {model_data.get('model_version', '?')}")
    print(f"  特征: {len(feature_names)}")

    conn = sqlite3.connect(DB_PATH)

    # 加载该股票足够长的日线数据 (需要120+60天历史算特征)
    # 从 kline_30m 聚合到日线
    df_30m = pd.read_sql(
        f"SELECT symbol, substr(date,1,10) as trade_date, "
        f"MIN(open) as open, MAX(high) as high, MIN(low) as low, "
        f"MAX(close) as close, SUM(volume) as volume "
        f"FROM kline_30m WHERE symbol='{symbol}' "
        f"GROUP BY symbol, substr(date,1,10) ORDER BY trade_date", conn)
    conn.close()

    if df_30m.empty:
        print(f"❌ 无数据: {symbol}")
        return

    df_30m['trade_date'] = pd.to_datetime(df_30m['trade_date'])
    print(f"  数据: {len(df_30m)} 天, {df_30m['trade_date'].min().date()} → {df_30m['trade_date'].max().date()}")

    # 计算特征
    conn = sqlite3.connect(DB_PATH)
    results = calculate_features(df_30m, conn)
    conn.close()

    if symbol not in results:
        print(f"❌ 特征计算失败: {symbol}")
        return

    stock_df, feats = results[symbol]
    feats = feats.fillna(0).replace([np.inf, -np.inf], 0)

    # 对齐特征列
    for c in feature_names:
        if c not in feats.columns:
            feats[c] = 0
    feats = feats[feature_names]

    # 检测 regime
    regime = detect_market_regime(stock_df)

    # 目标: horizon=3 天后的收益
    horizon = model_data.get('horizon', 3)
    close = stock_df['close'].values
    actual_ret = np.full(len(close), np.nan)
    for i in range(len(close) - horizon):
        actual_ret[i] = (close[i + horizon] - close[i]) / close[i]

    # 取最近 n_days 天做回测
    max_date = stock_df['trade_date'].max()
    cutoff = max_date - timedelta(days=n_days)
    mask = (stock_df['trade_date'] >= cutoff) & (~np.isnan(actual_ret))

    dates = stock_df['trade_date'].values[mask]
    actuals = actual_ret[mask]
    regimes = regime[mask]
    X_test = feats.values[mask]

    print(f"  回测 {mask.sum()} 天, horizon={horizon}")
    print(f"  日期: {pd.Timestamp(dates[0]).date()} → {pd.Timestamp(dates[-1]).date()}\n")

    # 逐日预测
    preds = []
    for i in range(len(dates)):
        r = regimes[i]
        pred = predict_one(model_data, feats.iloc[mask.values].iloc[i], r)
        preds.append(pred)

    preds = np.array(preds)

    # 结果
    from scipy.stats import spearmanr, pearsonr
    ic_pearson = pearsonr(preds, actuals)[0]
    ic_spearman = spearmanr(preds, actuals)[0]

    print(f"  📊 回测结果:")
    print(f"    Pearson IC:  {ic_pearson:+.4f}")
    print(f"    Spearman IC: {ic_spearman:+.4f}")
    print(f"    预测均值: {preds.mean():+.4f}  实际均值: {actuals.mean():+.4f}")
    print(f"    预测std:  {preds.std():.4f}   实际std:  {actuals.std():.4f}")

    # 方向准确率
    pred_dir = np.sign(preds)
    actual_dir = np.sign(actuals)
    acc = (pred_dir == actual_dir).mean()
    print(f"    方向准确率: {acc:.1%}")

    # 逐日明细
    print(f"\n  📅 逐日预测:")
    print(f"    {'日期':<12} {'regime':<8} {'预测':>8} {'实际':>8} {'方向':>4}")
    for i in range(len(dates)):
        d = pd.Timestamp(dates[i]).strftime('%Y-%m-%d')
        r = regimes[i]
        p = preds[i]
        a = actuals[i]
        dir_ok = '✓' if np.sign(p) == np.sign(a) else '✗'
        print(f"    {d:<12} {r:<8} {p:>+8.4f} {a:>+8.2%} {dir_ok:>4}")


if __name__ == '__main__':
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else '300124.SZ'
    run_backtest(symbol)
