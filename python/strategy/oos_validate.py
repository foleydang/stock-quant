#!/usr/bin/env python3
"""
日线模型样本外验证 — 独立于训练数据，验证真实预测力

策略:
  - 最后 N 个交易日作为样本外测试
  - 每天预测所有股票未来 horizon 天收益率
  - 选 Top K 等权持仓
  - 对比等权全市场基准

用法:
  python strategy/oos_validate.py
"""

import sys, os, pickle, sqlite3, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategy.train import load_data, compute_features, load_sentiment

warnings.filterwarnings('ignore')

# ============ 配置 ============
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/stock_data.db')
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models/lgb_daily/model.pkl')
HORIZON = 5         # 预测天数
TEST_DAYS = 60      # 样本外测试天数
TOP_K = 10          # 每天选多少只
CAPITAL = 100000    # 初始资金

BASE_CFG = {
    'horizon': HORIZON, 'min_history': 120, 'min_samples': 200,
    'features': 'enhanced+advanced+market',
}


def oos_validate():
    print("=" * 60)
    print(" 日线模型 样本外验证")
    print(f" 预测{HORIZON}日收益率 | 选Top{TOP_K} | 测试最后{TEST_DAYS}交易日")
    print("=" * 60)

    # 1. 加载模型
    print("\n📦 加载模型...")
    with open(MODEL_PATH, 'rb') as f:
        md = pickle.load(f)
    model = md['model']
    feature_names = md['feature_names']
    print(f"  模型特征数: {len(feature_names)}")
    print(f"  CV Rank IC: {md['cv_spearman']:.4f}")
    print(f"  CV RMSE: {md['cv_rmse']:.4f}")

    # 2. 加载数据 + 计算特征
    print("\n📊 加载数据 + 计算特征...")
    conn = sqlite3.connect(DB_PATH)
    data = load_data(DB_PATH, 'kline_daily')
    sent_df = load_sentiment(conn)
    has_sent = len(sent_df) > 0

    all_dates = None
    stock_features = {}   # symbol -> DataFrame (date index, feature columns)
    stock_close = {}      # symbol -> Series (date index, close price)
    success = 0

    for sym, df in data.items():
        try:
            feats = compute_features(df, sym, BASE_CFG)

            if has_sent:
                dates = df['date'].dt.strftime('%Y-%m-%d')
                sent = sent_df[sent_df['symbol'] == sym].set_index('date')
                for col in sent.columns:
                    if col not in ('symbol', 'date'):
                        feats[f'sent_{col}'] = dates.map(
                            lambda d: sent.loc[d, col] if d in sent.index else 0
                        ).fillna(0).values

            feats = feats.fillna(method='ffill').fillna(0)
            feats.index = df['date'].values

            # 对齐特征名
            for fn in feature_names:
                if fn not in feats.columns:
                    feats[fn] = 0
            feats = feats[feature_names]

            stock_features[sym] = feats
            stock_close[sym] = df.set_index('date')['close'].astype(float)

            if all_dates is None:
                all_dates = sorted(df['date'].unique())
            success += 1
        except Exception:
            continue

    conn.close()
    assert all_dates is not None, "无有效数据"

    # 全市场统一日期
    all_dates = sorted(all_dates)
    print(f"  {len(all_dates)} 个交易日, {success} 只股票")

    # 3. 确定样本外区间
    test_start_idx = len(all_dates) - TEST_DAYS - HORIZON
    test_dates_raw = all_dates[test_start_idx:test_start_idx + TEST_DAYS]

    # 过滤：只有当天的预测 + horizon 天后能看到结果的日期
    test_dates = []
    for d in test_dates_raw:
        d_idx = all_dates.index(d)
        if d_idx + HORIZON < len(all_dates):
            test_dates.append(d)
    print(f"  有效测试日: {len(test_dates)} (需要未来{HORIZON}天数据)")

    # 4. 逐日预测 + 评估
    print("\n🔮 逐日预测...")
    daily_rank_ic = []
    portfolio_value = CAPITAL
    portfolio_values = [CAPITAL]
    benchmark_value = CAPITAL
    benchmark_values = [CAPITAL]
    trade_log = []

    for day_idx, current_date in enumerate(test_dates):
        date_idx = all_dates.index(current_date)
        future_date = all_dates[date_idx + HORIZON]

        predictions = []
        actuals = []
        symbols_with_pred = []

        for sym in list(stock_features.keys()):
            feats = stock_features[sym]
            close = stock_close[sym]

            if current_date not in feats.index:
                continue
            if future_date not in close.index:
                continue

            # 预测
            row = feats.loc[current_date:current_date].values
            if row.shape[0] == 0 or np.isnan(row).any():
                continue
            pred = float(model.predict(row)[0])
            actual = (close.loc[future_date] - close.loc[current_date]) / close.loc[current_date]

            predictions.append(pred)
            actuals.append(actual)
            symbols_with_pred.append(sym)

        if len(predictions) < TOP_K:
            continue

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Rank IC
        valid = ~np.isnan(predictions) & ~np.isnan(actuals) & ~np.isinf(predictions) & ~np.isinf(actuals)
        if valid.sum() >= 10:
            from scipy.stats import spearmanr
            ic, _ = spearmanr(predictions[valid], actuals[valid])
            if not np.isnan(ic):
                daily_rank_ic.append(ic)

        # 选股: Top K
        top_idx = np.argsort(predictions)[-TOP_K:][::-1]
        top_returns = actuals[top_idx]
        strategy_return = np.mean(top_returns)
        benchmark_return = np.mean(actuals)

        # 更新持仓收益
        trade_amount = portfolio_value * 0.95  # 留5%现金
        per_stock = trade_amount / TOP_K
        new_portfolio = sum(per_stock * (1 + r) for r in top_returns) + portfolio_value * 0.05
        new_benchmark = benchmark_value * (1 + benchmark_return)

        trade_log.append({
            'date': str(current_date)[:10],
            'top_stocks': [symbols_with_pred[i] for i in top_idx[:5]],
            'strat_ret': round(strategy_return * 100, 2),
            'bm_ret': round(benchmark_return * 100, 2),
            'rank_ic': round(daily_rank_ic[-1], 4) if daily_rank_ic else None,
            'n_stocks': len(predictions),
        })

        portfolio_values.append(new_portfolio)
        portfolio_value = new_portfolio
        benchmark_values.append(new_benchmark)
        benchmark_value = new_benchmark

    # 5. 结果汇总
    print("\n" + "=" * 60)
    print(" 📈 样本外验证结果")
    print("=" * 60)

    if not daily_rank_ic:
        print("❌ 无有效预测结果，可能数据不足")
        return

    ic_mean = np.mean(daily_rank_ic)
    ic_std = np.std(daily_rank_ic)
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0
    ic_pos = sum(1 for x in daily_rank_ic if x > 0) / len(daily_rank_ic)

    strat_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    bm_returns = np.diff(benchmark_values) / benchmark_values[:-1]

    def sharpe(returns):
        if len(returns) < 2:
            return 0
        return np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

    def max_drawdown(values):
        peak = np.maximum.accumulate(values)
        dd = (values - peak) / peak
        return np.min(dd)

    def win_rate(returns):
        return float((np.array(returns) > 0).mean()) if len(returns) else 0

    print(f"\n  Rank IC:")
    print(f"    均值: {ic_mean:.4f}")
    print(f"    标准差: {ic_std:.4f}")
    print(f"    IR (IC/σ): {ic_ir:.2f}")
    print(f"    正值率: {ic_pos:.1%}")
    print(f"    最小: {min(daily_rank_ic):.4f}  最大: {max(daily_rank_ic):.4f}")

    print(f"\n  策略 vs 基准:")
    print(f"    策略累计收益: {(portfolio_value/CAPITAL - 1)*100:.2f}%")
    print(f"    基准累计收益: {(benchmark_value/CAPITAL - 1)*100:.2f}%")
    print(f"    超额收益:     {(portfolio_value/CAPITAL - benchmark_value/CAPITAL)*100:.2f}%")
    print(f"    策略 Sharpe:  {sharpe(strat_returns):.3f}")
    print(f"    基准 Sharpe:  {sharpe(bm_returns):.3f}")
    print(f"    策略 最大回撤: {max_drawdown(portfolio_values)*100:.2f}%")
    print(f"    基准 最大回撤: {max_drawdown(benchmark_values)*100:.2f}%")
    print(f"    策略 日胜率:   {win_rate(strat_returns):.1%}")
    print(f"    基准 日胜率:   {win_rate(bm_returns):.1%}")

    # 最近10天明细
    print(f"\n  最近交易记录:")
    print(f"  {'日期':<12} {'策略%':>7} {'基准%':>7} {'IC':>7} {'Top股票':>30}")
    for t in trade_log[-10:]:
        stocks_str = ','.join(t['top_stocks'][:3])
        print(f"  {t['date']:<12} {t['strat_ret']:>+7.2f} {t['bm_ret']:>+7.2f} "
              f"{t['rank_ic'] or 0:>+7.4f} {stocks_str:>30}")

    # 判断
    print(f"\n  📋 结论:")
    if ic_mean > 0.05 and portfolio_value > benchmark_value:
        print(f"  ✅ 模型样本外有效: Rank IC={ic_mean:.4f}({ic_pos:.0%}正), 超额{(portfolio_value/CAPITAL - benchmark_value/CAPITAL)*100:.2f}%")
    elif ic_mean > 0.025:
        print(f"  ⚠️ 模型弱有效: Rank IC={ic_mean:.4f}, 需要进一步优化")
    else:
        print(f"  ❌ 模型过拟合: Rank IC={ic_mean:.4f}, 样本外失效")


if __name__ == '__main__':
    oos_validate()