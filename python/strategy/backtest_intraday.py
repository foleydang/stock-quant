#!/usr/bin/env python3
"""
回测: 日线预测 + 30分钟线确认信号 → 当日收益

模拟逻辑:
  1. 读取某天的日线预测结果
  2. 对下一个交易日，逐根30分钟K线模拟盘中信号
  3. 一旦触发买入信号，以当前价格买入，持有到收盘
  4. 计算每只股票和整体的收益

用法:
  python strategy/backtest_intraday.py                          # 回测最近有数据的日期
  python strategy/backtest_intraday.py --pred-date 2026-06-24  # 用6/24预测, 6/25盘中
"""

import os, sys, sqlite3, argparse
import numpy as np
import pandas as pd
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'python'))

DB_PATH = os.path.join(ROOT, 'data', 'stock_data.db')
MODEL_DIR = os.path.join(ROOT, '..', 'models', 'lgb_hs300_enhanced')

# 导入30分钟线信号计算
from strategy.intraday_signal import (
    load_intraday_data, compute_intraday_signals, evaluate_buy_signal
)


def load_prediction(pred_date):
    """加载日线预测"""
    path = os.path.join(MODEL_DIR, f'prediction_{pred_date.replace("-", "")}.csv')
    df = pd.read_csv(path)
    df['symbol'] = df['symbol'].astype(str)
    return df.set_index('symbol')


def get_trading_dates(conn, after_date):
    """获取 after_date 之后有30分钟线数据的交易日"""
    dates = pd.read_sql(f"""
        SELECT DISTINCT substr(date,1,10) as trade_date
        FROM kline_30m
        WHERE substr(date,1,10) > '{after_date}'
        ORDER BY trade_date
    """, conn)
    return dates['trade_date'].tolist()


def simulate_intraday(conn, symbol, trade_date, daily_signal):
    """
    模拟盘中信号: 逐根30分钟K线检查
    返回: (entry_time, entry_price, exit_price, signal_triggered, pnl_pct)
    """
    # 加载该股票该交易日的全部30分钟K线
    df = pd.read_sql(f"""
        SELECT date, open, high, low, close, volume
        FROM kline_30m
        WHERE symbol = '{symbol}'
          AND substr(date,1,10) = '{trade_date}'
        ORDER BY date
    """, conn)

    if len(df) < 5:
        return None

    # 需要加载前一天的数据来计算技术指标（至少30根K线）
    # 加载前几天的数据
    hist_df = pd.read_sql(f"""
        SELECT date, open, high, low, close, volume
        FROM kline_30m
        WHERE symbol = '{symbol}'
          AND date < '{trade_date} 09:30:00'
        ORDER BY date DESC
        LIMIT 50
    """, conn)

    if len(hist_df) < 20:
        return None

    # 合并历史数据
    hist_df = hist_df.sort_values('date').reset_index(drop=True)

    # 逐根K线模拟 (只到10:30, 开盘前2根K线信号最可靠)
    for i in range(len(df)):
        bar_time = str(df.iloc[i]['date'])
        if '11:00' in bar_time or '11:30' in bar_time or '13:' in bar_time or '14:' in bar_time or '15:' in bar_time:
            continue
        # 合并历史 + 当前及之前的K线
        combined = pd.concat([hist_df, df.iloc[:i+1]], ignore_index=True)
        combined = combined.sort_values('date').reset_index(drop=True)

        # 取最近60根K线计算指标
        window = combined.tail(60).copy()

        intraday = compute_intraday_signals(window)
        action, confidence, reason = evaluate_buy_signal(
            daily_signal, intraday, bar_time=str(df.iloc[i]['date'])
        )

        if confidence in ('high', 'medium') and '买入' in action:
            # 触发买入信号!
            entry_price = df.iloc[i]['close']
            entry_time = df.iloc[i]['date']
            exit_price = df.iloc[-1]['close']  # 收盘价
            pnl_pct = (exit_price / entry_price - 1) * 100

            # 计算买入后到收盘的最高价和最低价
            remaining = df.iloc[i+1:]
            max_price = remaining['high'].max() if len(remaining) > 0 else entry_price
            min_price = remaining['low'].min() if len(remaining) > 0 else entry_price

            return {
                'symbol': symbol,
                'entry_time': entry_time,
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'pnl_pct': round(pnl_pct, 2),
                'max_pnl': round((max_price / entry_price - 1) * 100, 2),
                'max_loss': round((min_price / entry_price - 1) * 100, 2),
                'action': action,
                'confidence': confidence,
                'reason': reason,
                'bars_held': len(df) - i - 1,
                'total_bars': len(df),
            }

    # 没有触发信号
    return None


def run_backtest(pred_date, top_n=20):
    """运行回测"""
    conn = sqlite3.connect(DB_PATH)

    # 加载预测
    print(f"📊 加载日线预测: {pred_date}")
    pred = load_prediction(pred_date)
    print(f"   共 {len(pred)} 只股票")

    # 获取下一个交易日
    trade_dates = get_trading_dates(conn, pred_date)
    if not trade_dates:
        print("❌ 没有找到后续交易日")
        conn.close()
        return

    trade_date = trade_dates[0]
    print(f"   回测交易日: {trade_date}")

    # 只看强烈买入信号
    buy_stocks = pred[pred['signal_code'].isin(['strong_buy'])]
    buy_stocks = buy_stocks.sort_values('rank')
    print(f"   买入信号: {len(buy_stocks)} 只")

    # 逐只模拟
    results = []
    no_signal = []

    for sym in buy_stocks.index[:top_n * 2]:  # 检查 Top-N*2 只, 确保有足够的信号
        row = buy_stocks.loc[sym]
        daily_signal = {
            'signal': row.get('signal', ''),
            'signal_code': row['signal_code'],
            'score': row['score'],
            'name': row.get('name', ''),
            'rank': row['rank'],
        }

        try:
            result = simulate_intraday(conn, sym, trade_date, daily_signal)
            if result:
                result['name'] = daily_signal.get('name', '')
                result['daily_score'] = daily_signal.get('score', 0)
                results.append(result)
            else:
                no_signal.append(sym)
        except Exception as e:
            no_signal.append(sym)

    conn.close()

    # === 输出结果 ===
    print(f"\n{'='*80}")
    print(f"📈 回测结果 | 预测日 {pred_date} → 交易 {trade_date}")
    print(f"{'='*80}")
    print(f"买入信号股票: {len(buy_stocks)} 只")
    print(f"模拟检查: {len(results) + len(no_signal)} 只")
    print(f"触发信号: {len(results)} 只")
    print(f"未触发: {len(no_signal)} 只")

    if not results:
        print("\n⚠️ 没有触发买入信号")
        return

    print(f"\n{'─'*80}")
    print(f"触发买入的股票:")
    print(f"{'─'*80}")
    print(f"{'代码':>10} {'名称':>12} {'入场时间':>20} {'入场价':>8} {'收盘价':>8} "
          f"{'收益%':>7} {'最高%':>7} {'最低%':>7} {'信号'}")
    print(f"{'─'*80}")

    total_pnl = 0
    for r in sorted(results, key=lambda x: x['pnl_pct'], reverse=True):
        print(f"{r['symbol']:>10} {r.get('name',''):>12} "
              f"{str(r['entry_time']):>20} {r['entry_price']:>8.2f} {r['exit_price']:>8.2f} "
              f"{r['pnl_pct']:>+6.2f}% {r['max_pnl']:>+6.2f}% {r['max_loss']:>+6.2f}% "
              f"{r['action']}")
        total_pnl += r['pnl_pct']

    avg_pnl = total_pnl / len(results)
    win_count = sum(1 for r in results if r['pnl_pct'] > 0)
    win_rate = win_count / len(results) * 100

    print(f"{'─'*80}")
    print(f"汇总: {len(results)} 笔交易")
    print(f"  平均收益: {avg_pnl:+.2f}%")
    print(f"  胜率: {win_count}/{len(results)} = {win_rate:.1f}%")
    print(f"  总收益(等权): {total_pnl:+.2f}%")

    # 如果每只等额买入
    print(f"\n💡 假设每只等额买入:")
    print(f"  总投资回报: {avg_pnl:+.2f}%")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-date', type=str, default='2026-06-24',
                        help='预测日期 YYYY-MM-DD')
    parser.add_argument('--top', type=int, default=20,
                        help='检查Top-N买入信号')
    args = parser.parse_args()

    run_backtest(args.pred_date, top_n=args.top)