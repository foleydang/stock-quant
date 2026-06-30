#!/usr/bin/env python3
"""
V9 模型全量回测 — 最近1个月, 369只股票, 每日截面预测 + 大盘过滤

流程:
1. 加载 V9 模型 (regime_models 格式)
2. 计算所有股票特征 (复用 train_enhanced 的特征逻辑)
3. 对每个交易日: 截面预测 → 大盘过滤 → 对比实际 horizon 天收益
4. 输出: IC, ICIR, Top/Bottom 收益, 多空胜率, 买卖信号
"""

import os, sys, pickle, sqlite3, time, argparse
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
from strategy.market_filter import MarketFilter


def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def predict_ensemble(model_data, X, regime):
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


def run_backtest(n_days=30, with_filter=True):
    print("=" * 70)
    print(f"  V9 全量回测 — 最近{n_days}天, 大盘过滤={'开' if with_filter else '关'}")
    print("=" * 70)

    model_data = load_model()
    feature_names = model_data['feature_names']
    horizon = model_data.get('horizon', 3)
    print(f"  模型: {model_data.get('model_version', '?')}, horizon={horizon}")

    mf = MarketFilter(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    t0 = time.time()
    df = load_kline_data(conn, max_stocks=0)
    n_stocks = len(df['symbol'].unique())
    print(f"  加载 {n_stocks} 只股票, {len(df)} 条日线 ({time.time()-t0:.0f}s)")

    t0 = time.time()
    results = calculate_features(df, conn)
    conn.close()
    print(f"  特征计算完成: {len(results)} 只股票 ({time.time()-t0:.0f}s)")

    print("\n📦 准备回测数据...")
    stock_data = {}
    sym_names = {}
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

        regime = detect_market_regime(stock_df, db_path=DB_PATH)
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

    all_dates = set()
    for sdf in stock_data.values():
        all_dates.update(sdf['date'].values)
    all_dates = sorted(all_dates)

    max_date = pd.Timestamp(all_dates[-1])
    cutoff = max_date - timedelta(days=n_days)
    bt_dates = [d for d in all_dates if pd.Timestamp(d) >= cutoff]
    print(f"  回测区间: {cutoff.date()} → {max_date.date()} ({len(bt_dates)} 天)\n")

    # 两组指标: 无过滤 vs 有过滤
    all_ics = []
    filtered_top_rets = []
    unfiltered_top_rets = []
    all_bot_rets = []
    all_long_short = []
    filtered_long_short = []
    n_filtered_days = 0
    n_buy_days = 0

    for date in bt_dates:
        day_data = []  # [(sym, pred, actual, regime)]
        for sym, sdf in stock_data.items():
            mask = sdf['date'].values == date
            if not mask.any():
                continue
            idx = mask.argmax()
            row = sdf.iloc[idx]
            X = row['feats'].reshape(1, -1)
            regime = row['regime']
            pred = predict_ensemble(model_data, X, regime)[0]
            day_data.append((sym, pred, row['actual'], regime))

        if len(day_data) < 10:
            continue

        syms = [x[0] for x in day_data]
        preds = np.array([x[1] for x in day_data])
        actuals = np.array([x[2] for x in day_data])
        regimes = np.array([x[3] for x in day_data])

        ic_sp = spearmanr(preds, actuals)[0]
        ic_pe = pearsonr(preds, actuals)[0]
        all_ics.append(ic_sp)

        sort_idx = np.argsort(preds)
        n = len(day_data)
        top_n = max(5, n // 10)
        top_ret = actuals[sort_idx[-top_n:]].mean()
        bot_ret = actuals[sort_idx[:top_n]].mean()
        unfiltered_top_rets.append(top_ret)
        all_bot_rets.append(bot_ret)
        all_long_short.append(top_ret - bot_ret)

        # 大盘过滤
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        allow_buy = mf.should_allow_buy(date_str)
        mkt_state = mf.get_state(date_str)
        regime_label = mkt_state['regime_label'] if mkt_state else '?'
        pos_ratio = mf.get_position_ratio(date_str)

        if allow_buy:
            filtered_top_rets.append(top_ret)
            filtered_long_short.append(top_ret - bot_ret)
            n_buy_days += 1
            filter_flag = '📈'
        else:
            filtered_top_rets.append(0.0)  # 不买 → 持现金
            filtered_long_short.append(0.0 - bot_ret)  # 只做空 bottom
            n_filtered_days += 1
            filter_flag = '🚫'

        regime_str = '/'.join(f'{r}:{(regimes==r).sum()}' for r in np.unique(regimes))
        print(f"  {date_str} {filter_flag} [{regime_label}|仓{pos_ratio:.0%}] "
              f"IC={ic_sp:+.4f} | Top{top_n}={top_ret:+.2%} Bot{top_n}={bot_ret:+.2%} "
              f"多空={top_ret-bot_ret:+.2%} ({n}只)")

    # === 汇总 ===
    print("\n" + "=" * 70)
    print("  📊 回测汇总对比")
    print("=" * 70)
    if all_ics:
        ic_mean = np.mean(all_ics)
        icir = ic_mean / np.std(all_ics) if np.std(all_ics) > 0 else 0
        print(f"  Spearman IC:  {ic_mean:+.4f} (正向={sum(1 for x in all_ics if x>0)}/{len(all_ics)})")
        print(f"  ICIR:         {icir:.3f}")

        print(f"\n  {'':20s} {'无过滤':>10s} {'有过滤':>10s}")
        print(f"  {'平均Top组收益':20s} {np.mean(unfiltered_top_rets):>+10.2%} {np.mean(filtered_top_rets):>+10.2%}")
        print(f"  {'平均Bot组收益':20s} {np.mean(all_bot_rets):>+10.2%} {np.mean(all_bot_rets):>+10.2%}")
        uf_ls = np.mean(all_long_short)
        f_ls = np.mean(filtered_long_short)
        print(f"  {'平均多空差':20s} {uf_ls:>+10.2%} {f_ls:>+10.2%}")
        uf_win = sum(1 for x in all_long_short if x > 0)
        f_win = sum(1 for x in filtered_long_short if x > 0)
        print(f"  {'多空胜率':20s} {uf_win}/{len(all_long_short):>7d} {f_win}/{len(filtered_long_short):>7d}")

        # 累计 Top 组收益
        cum_uf = np.prod([1 + r for r in unfiltered_top_rets]) - 1
        cum_f = np.prod([1 + r for r in filtered_top_rets]) - 1
        print(f"\n  {'累计Top组收益':20s} {cum_uf:>+10.2%} {cum_f:>+10.2%}")
        print(f"  买入天数: {n_buy_days}/{len(bt_dates)}, 过滤天数: {n_filtered_days}")

    # === 最后一天的买卖信号 ===
    if bt_dates:
        last_date = bt_dates[-1]
        date_str = pd.Timestamp(last_date).strftime('%Y-%m-%d')
        allow = mf.should_allow_buy(date_str)
        state = mf.get_state(date_str)

        print(f"\n{'='*70}")
        print(f"  📋 最新买卖信号 ({date_str})")
        print(f"  大盘: HS300={'↑' if allow else '↓'} MA20, regime={state['regime_label'] if state else '?'}, "
              f"建议仓位={mf.get_position_ratio(date_str):.0%}")
        print(f"{'='*70}")

        last_data = []
        for sym, sdf in stock_data.items():
            mask = sdf['date'].values == last_date
            if not mask.any():
                continue
            idx = mask.argmax()
            row = sdf.iloc[idx]
            X = row['feats'].reshape(1, -1)
            regime = row['regime']
            pred = predict_ensemble(model_data, X, regime)[0]
            last_data.append((sym, pred, row['actual']))

        if last_data:
            last_data.sort(key=lambda x: -x[1])
            n = len(last_data)
            top_n = max(5, n // 10)
            print(f"\n  🟢 Top-{top_n} 买入候选 {'(大盘过滤: 暂不买入!)' if not allow else ''}:")
            for i, (sym, pred, actual) in enumerate(last_data[:top_n]):
                signal = 'BUY' if allow else 'WAIT'
                print(f"    {i+1:2d}. {sym:12s} pred={pred:+.4f} actual={actual:+.2%} → {signal}")

            print(f"\n  🔴 Bottom-{top_n} 卖出候选:")
            for i, (sym, pred, actual) in enumerate(last_data[-top_n:]):
                print(f"    {n-top_n+i+1:2d}. {sym:12s} pred={pred:+.4f} actual={actual:+.2%} → SELL")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=30, help='回测天数')
    parser.add_argument('--no-filter', action='store_true', help='关闭大盘过滤')
    args = parser.parse_args()
    run_backtest(n_days=args.days, with_filter=not args.no_filter)
