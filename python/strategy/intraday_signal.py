#!/usr/bin/env python3
"""
30分钟线买卖时机确认 — 日线定方向 + 30分钟线找时机

用法:
  python strategy/intraday_signal.py                    # 检查最新一根30分钟线
  python strategy/intraday_signal.py --watch            # 持续监控(每30分钟)
  python strategy/intraday_signal.py --symbol 000807.SZ # 单只股票
  python strategy/intraday_signal.py --top 10           # 只看Top-10买入信号
"""

import os, sys, sqlite3, argparse, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'python'))

DB_PATH = os.path.join(ROOT, 'data', 'stock_data.db')
PREDICTION_FILE = os.path.join(ROOT, '..', 'models', 'lgb_hs300_enhanced',
                               f'prediction_{datetime.now().strftime("%Y%m%d")}.csv')


def load_daily_signals(conn, top_n=None):
    """加载日线预测结果，返回 {symbol: {signal, score, name}}"""
    try:
        df = pd.read_csv(PREDICTION_FILE)
    except FileNotFoundError:
        print(f"❌ 日线预测文件不存在: {PREDICTION_FILE}")
        print("   请先运行: python strategy/predict_today_batched.py")
        return {}

    signals = {}
    for _, row in df.iterrows():
        signals[row['symbol']] = {
            'signal': row.get('signal', ''),
            'signal_code': row.get('signal_code', ''),
            'score': row.get('score', 0),
            'name': row.get('name', ''),
            'rank': row.get('rank', 0),
        }

    # 过滤只看买入信号
    if top_n:
        buy_stocks = {k: v for k, v in signals.items()
                      if v['signal_code'] in ('strong_buy', 'buy')}
        sorted_buy = sorted(buy_stocks.items(), key=lambda x: x[1]['rank'])[:top_n]
        return dict(sorted_buy)

    return signals


def load_intraday_data(conn, symbol, bars=60):
    """加载最近N根30分钟K线"""
    df = pd.read_sql(f"""
        SELECT date, open, high, low, close, volume
        FROM kline_30m
        WHERE symbol = '{symbol}'
        ORDER BY date DESC
        LIMIT {bars}
    """, conn)
    df = df.sort_values('date').reset_index(drop=True)
    return df


def compute_intraday_signals(df):
    """
    计算30分钟线技术信号，返回确认信号列表
    信号规则:
      - MACD 金叉/死叉
      - 放量突破 (成交量 > 20周期均量 × 1.5)
      - 价格突破布林带上轨/下轨
      - RSI 超买/超卖
      - 价格站上/跌破MA20
    """
    if len(df) < 30:
        return {}

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    n = len(close)
    signals = {}

    # === MACD ===
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
    macd_hist = macd_line - signal_line

    # 金叉: 上一根 hist < 0, 当前 hist > 0 (或 macd 上穿 signal)
    if n >= 2:
        prev_cross = macd_line[-2] - signal_line[-2]
        curr_cross = macd_line[-1] - signal_line[-1]
        if prev_cross < 0 and curr_cross > 0:
            signals['macd_golden_cross'] = True
        elif prev_cross > 0 and curr_cross < 0:
            signals['macd_death_cross'] = True

    # === 成交量 ===
    vol_ma20 = np.mean(volume[-21:-1]) if n > 21 else np.mean(volume[:-1])
    vol_ratio = volume[-1] / (vol_ma20 + 1e-9)
    signals['vol_ratio'] = round(vol_ratio, 2)
    if vol_ratio > 2.0:
        signals['volume_surge'] = 'high'
    elif vol_ratio > 1.5:
        signals['volume_surge'] = 'medium'
    elif vol_ratio < 0.5:
        signals['volume_surge'] = 'low'

    # === 布林带 ===
    bb_period = 20
    bb_ma = np.mean(close[-bb_period:])
    bb_std = np.std(close[-bb_period:])
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    bb_pos = (close[-1] - bb_lower) / (bb_upper - bb_lower + 1e-9)
    signals['bb_position'] = round(bb_pos, 2)  # 0=下轨, 1=上轨

    if close[-1] > bb_upper * 1.01:
        signals['bb_breakout_up'] = True
    elif close[-1] < bb_lower * 0.99:
        signals['bb_breakout_down'] = True

    # === RSI ===
    delta = np.diff(close)
    gain = np.sum(delta[delta > 0][-14:]) if len(delta[delta > 0]) > 0 else 0
    loss = abs(np.sum(delta[delta < 0][-14:])) if len(delta[delta < 0]) > 0 else 0
    rsi = 100 - 100 / (1 + gain / (loss + 1e-9))
    signals['rsi'] = round(rsi, 1)

    if rsi > 70:
        signals['rsi_overbought'] = True
    elif rsi < 30:
        signals['rsi_oversold'] = True

    # === 均线 ===
    ma5 = np.mean(close[-5:])
    ma20 = np.mean(close[-20:])
    signals['ma5_position'] = 'above' if close[-1] > ma5 else 'below'
    signals['ma20_position'] = 'above' if close[-1] > ma20 else 'below'

    # 金叉(MA5上穿MA20)
    if n >= 2:
        ma5_prev = np.mean(close[-6:-1])
        ma20_prev = np.mean(close[-21:-1])
        if ma5_prev <= ma20_prev and ma5 > ma20:
            signals['ma_golden_cross'] = True
        elif ma5_prev >= ma20_prev and ma5 < ma20:
            signals['ma_death_cross'] = True

    # === 价格动量 ===
    ret_1 = (close[-1] / close[-2] - 1) if n >= 2 else 0
    ret_5 = (close[-1] / close[-6] - 1) if n >= 6 else 0
    signals['ret_1bar'] = round(ret_1 * 100, 2)  # 百分比
    signals['ret_5bar'] = round(ret_5 * 100, 2)

    # === 振幅 ===
    amplitude = (high[-1] - low[-1]) / (close[-1] + 1e-9) * 100
    signals['amplitude'] = round(amplitude, 2)

    return signals


def evaluate_buy_signal(daily_signal, intraday, bar_time=None):
    """
    综合日线信号和30分钟线信号，给出最终买卖建议
    
    优化后规则 (v2):
      - 买入必须放量 + 收阳 + 站稳MA20
      - 有任何卖出信号则不买入
      - 只在上午 10:00-11:30 触发高置信度
      - 下午/尾盘信号降级为"观望"
    
    返回: (action, confidence, reason)
    """
    daily_code = daily_signal.get('signal_code', '')
    
    # 判断是否在黄金时段 (10:00-10:30, 开盘后第一根K线完成)
    is_golden = False
    is_morning = False
    if bar_time:
        try:
            t = pd.Timestamp(bar_time)
            is_morning = t.hour < 11 or (t.hour == 11 and t.minute <= 30)
            is_golden = t.hour == 10 and t.minute <= 30
        except:
            pass

    # 买入方向的确认
    if daily_code in ('strong_buy', 'buy'):
        buy_points = 0
        reasons = []
        warnings = []

        # === 必要条件 (不满足直接否决) ===
        
        # 1. 必须放量 (量比 > 1.2)
        vol_ratio = intraday.get('vol_ratio', 0)
        has_strong_volume = vol_ratio > 2.0
        if vol_ratio > 2.0:
            buy_points += 3
            reasons.append(f'大幅放量({vol_ratio:.1f}x)')
        elif vol_ratio > 1.5:
            buy_points += 2
            reasons.append(f'放量({vol_ratio:.1f}x)')
        elif vol_ratio > 1.2:
            buy_points += 1
            reasons.append(f'温和放量({vol_ratio:.1f}x)')
        else:
            warnings.append(f'量能不足({vol_ratio:.1f}x)')

        # 2. 必须收阳 (当前K线涨)
        ret_1 = intraday.get('ret_1bar', 0)
        if ret_1 > 0.5:
            buy_points += 2
            reasons.append(f'收阳(+{ret_1:.1f}%)')
        elif ret_1 > 0:
            buy_points += 1
            reasons.append(f'微涨(+{ret_1:.1f}%)')
        else:
            warnings.append(f'收阴({ret_1:+.1f}%)')

        # 3. 必须站稳MA20
        if intraday.get('ma20_position') == 'above':
            buy_points += 1
            reasons.append('站稳MA20')
        else:
            warnings.append('跌破MA20')

        # === 加分项 ===
        
        # MACD 金叉 (强信号)
        if intraday.get('macd_golden_cross'):
            buy_points += 3
            reasons.append('MACD金叉')

        # 布林带突破
        if intraday.get('bb_breakout_up'):
            buy_points += 2
            reasons.append('突破布林上轨')

        # 均线金叉
        if intraday.get('ma_golden_cross'):
            buy_points += 2
            reasons.append('MA5上穿MA20')

        # RSI 在合理区间 (40-65, 不能超买)
        rsi = intraday.get('rsi', 50)
        if 45 < rsi < 65:
            buy_points += 1
            reasons.append(f'RSI适中({rsi:.0f})')
        elif 40 <= rsi <= 70:
            pass  # 中性
        else:
            warnings.append(f'RSI异常({rsi:.0f})')

        # === 否决项 (有任何一条则降级) ===
        
        if intraday.get('macd_death_cross'):
            warnings.append('⚠️ MACD死叉')
            buy_points -= 3
        if intraday.get('bb_breakout_down'):
            warnings.append('⚠️ 跌破布林下轨')
            buy_points -= 2
        if intraday.get('rsi_overbought'):
            warnings.append('⚠️ RSI超买')
            buy_points -= 2
        if intraday.get('ma_death_cross'):
            warnings.append('⚠️ MA5下穿MA20')
            buy_points -= 2

        # === 时段惩罚 ===
        time_penalty = ''
        if not is_morning:
            buy_points -= 2
            time_penalty = ' [下午时段]'
        elif not is_golden:
            buy_points -= 1  # 11:00-11:30 轻度惩罚

        # === 决策 (v3: 必须有强信号才触发, 仅10:00-10:30) ===
        has_strong_signal = intraday.get('macd_golden_cross') or has_strong_volume
        
        if buy_points >= 4 and not warnings and is_golden and has_strong_signal:
            return '🔥 立即买入', 'high', ' + '.join(reasons)
        elif buy_points >= 3 and len(warnings) <= 1 and is_morning and has_strong_signal:
            return '📈 准备买入', 'medium', ' + '.join(reasons + [f'({w})' for w in warnings])
        elif buy_points >= 2:
            return '👀 关注', 'low', ' + '.join(reasons + [f'({w})' for w in warnings])
        else:
            all_reasons = reasons + warnings
            return '⏳ 等待', 'none', '信号不足' + (': ' + ', '.join(all_reasons) if all_reasons else '')

    # 卖出方向的确认
    elif daily_code in ('strong_sell', 'sell'):
        sell_points = 0
        reasons = []
        warnings = []

        # 必要条件
        if intraday.get('macd_death_cross'):
            sell_points += 3
            reasons.append('MACD死叉')
        if intraday.get('volume_surge') == 'low':
            sell_points += 2
            reasons.append('缩量')
        if intraday.get('bb_breakout_down'):
            sell_points += 2
            reasons.append('跌破布林下轨')
        if intraday.get('ma_death_cross'):
            sell_points += 2
            reasons.append('MA5下穿MA20')
        if intraday.get('rsi', 50) < 40:
            sell_points += 1
            reasons.append('RSI偏空')
        if intraday.get('ma20_position') == 'below':
            sell_points += 1
            reasons.append('跌破MA20')
        if intraday.get('ret_1bar', 0) < -0.5:
            sell_points += 1
            reasons.append(f'加速下跌({intraday["ret_1bar"]:+.1f}%)')

        # 否决
        if intraday.get('macd_golden_cross'):
            sell_points -= 3
            warnings.append('⚠️ MACD金叉')
        if intraday.get('volume_surge') in ('high', 'medium'):
            sell_points -= 2
            warnings.append('⚠️ 放量反弹')

        if sell_points >= 6:
            return '🔥 立即卖出', 'high', ' + '.join(reasons)
        elif sell_points >= 4:
            return '📉 准备卖出', 'medium', ' + '.join(reasons)
        elif sell_points >= 2:
            return '👀 注意风险', 'low', ' + '.join(reasons)
        else:
            return '⏳ 等待', 'none', '信号不足'

    else:
        return '⏸️ 持有观望', 'none', '日线评级中性'


def check_stock(conn, symbol, daily_signal):
    """检查单只股票的30分钟线信号"""
    df = load_intraday_data(conn, symbol)
    if len(df) < 30:
        return None

    intraday = compute_intraday_signals(df)
    action, confidence, reason = evaluate_buy_signal(daily_signal, intraday, bar_time=str(latest['date']))

    latest = df.iloc[-1]
    return {
        'symbol': symbol,
        'name': daily_signal.get('name', ''),
        'daily_signal': daily_signal.get('signal', ''),
        'daily_score': daily_signal.get('score', 0),
        'daily_rank': daily_signal.get('rank', 0),
        'latest_bar': str(latest['date']),
        'price': latest['close'],
        'vol_ratio': intraday.get('vol_ratio', 0),
        'rsi': intraday.get('rsi', 50),
        'ret_1bar': intraday.get('ret_1bar', 0),
        'amplitude': intraday.get('amplitude', 0),
        'action': action,
        'confidence': confidence,
        'reason': reason,
        'details': intraday,
    }


def run_check(conn, top_n=None, symbols=None):
    """运行检查"""
    # 加载日线信号
    daily_signals = load_daily_signals(conn, top_n=top_n)

    if symbols:
        # 指定股票
        daily_signals = {s: daily_signals.get(s, {}) for s in symbols}

    if not daily_signals:
        print("无信号数据")
        return []

    # 只看买入和卖出信号
    target_stocks = {k: v for k, v in daily_signals.items()
                     if v.get('signal_code') in ('strong_buy', 'buy', 'strong_sell', 'sell')}

    if not target_stocks:
        print("当前无买卖信号")
        return []

    print(f"检查 {len(target_stocks)} 只股票的30分钟线信号...")

    results = []
    for sym, sig in target_stocks.items():
        try:
            result = check_stock(conn, sym, sig)
            if result:
                results.append(result)
        except Exception as e:
            continue

    return results


def print_results(results):
    """打印结果"""
    if not results:
        print("\n无有效信号")
        return

    # 按置信度排序
    confidence_order = {'high': 0, 'medium': 1, 'low': 2, 'none': 3}
    results.sort(key=lambda x: confidence_order.get(x['confidence'], 99))

    # 分组
    buy_signals = [r for r in results if '买入' in r['action']]
    sell_signals = [r for r in results if '卖出' in r['action']]
    watch_signals = [r for r in results if '关注' in r['action'] or '等待' in r['action']]
    hold_signals = [r for r in results if '持有' in r['action']]

    latest_time = results[0]['latest_bar'] if results else ''

    print(f"\n{'='*80}")
    print(f"🕐 30分钟线确认信号 | 最新K线: {latest_time}")
    print(f"{'='*80}")

    if buy_signals:
        print(f"\n{'─'*80}")
        print(f"🔥 买入确认 (日线看多 + 30分钟线确认)")
        print(f"{'─'*80}")
        print(f"{'代码':>10} {'名称':>12} {'日线得分':>8} {'价格':>8} {'量比':>6} {'RSI':>5} {'操作':>14} {'原因'}")
        for r in buy_signals:
            print(f"{r['symbol']:>10} {r['name']:>12} {r['daily_score']:>+8.4f} "
                  f"{r['price']:>8.2f} {r['vol_ratio']:>5.1f}x {r['rsi']:>4.0f} "
                  f"{r['action']:>12}  {r['reason']}")

    if sell_signals:
        print(f"\n{'─'*80}")
        print(f"📉 卖出确认 (日线看空 + 30分钟线确认)")
        print(f"{'─'*80}")
        print(f"{'代码':>10} {'名称':>12} {'日线得分':>8} {'价格':>8} {'量比':>6} {'RSI':>5} {'操作':>14} {'原因'}")
        for r in sell_signals:
            print(f"{r['symbol']:>10} {r['name']:>12} {r['daily_score']:>+8.4f} "
                  f"{r['price']:>8.2f} {r['vol_ratio']:>5.1f}x {r['rsi']:>4.0f} "
                  f"{r['action']:>12}  {r['reason']}")

    if watch_signals:
        print(f"\n{'─'*80}")
        print(f"👀 等待确认 (日线有信号, 30分钟线待确认)")
        print(f"{'─'*80}")
        print(f"{'代码':>10} {'名称':>12} {'日线信号':>12} {'价格':>8} {'操作':>14} {'原因'}")
        for r in watch_signals[:10]:
            print(f"{r['symbol']:>10} {r['name']:>12} {r['daily_signal']:>12} "
                  f"{r['price']:>8.2f} {r['action']:>12}  {r['reason']}")

    if hold_signals:
        print(f"\n{'─'*80}")
        print(f"⏸️ 持有观望")
        print(f"{'─'*80}")
        print(f"{'代码':>10} {'名称':>12} {'价格':>8} {'操作':>14}")
        for r in hold_signals[:5]:
            print(f"{r['symbol']:>10} {r['name']:>12} {r['price']:>8.2f} {r['action']:>12}")


def main():
    parser = argparse.ArgumentParser(description='30分钟线买卖时机确认')
    parser.add_argument('--watch', action='store_true', help='持续监控模式')
    parser.add_argument('--symbol', type=str, help='单只股票代码')
    parser.add_argument('--top', type=int, default=20, help='检查Top-N买入信号')
    parser.add_argument('--interval', type=int, default=1800, help='监控间隔(秒), 默认1800=30分钟')
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)

    if args.watch:
        print(f"🔄 持续监控模式, 每 {args.interval} 秒检查一次...")
        print(f"   按 Ctrl+C 停止\n")
        try:
            while True:
                results = run_check(conn, top_n=args.top)
                print_results(results)
                print(f"\n⏰ 下次检查: {(datetime.now() + timedelta(seconds=args.interval)).strftime('%H:%M:%S')}")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n👋 停止监控")
    else:
        if args.symbol:
            daily_signals = load_daily_signals(conn)
            sig = daily_signals.get(args.symbol, {})
            if not sig:
                print(f"❌ {args.symbol} 不在预测结果中")
                conn.close()
                return
            result = check_stock(conn, args.symbol, sig)
            results = [result] if result else []
        else:
            results = run_check(conn, top_n=args.top)

        print_results(results)

    conn.close()


if __name__ == '__main__':
    main()