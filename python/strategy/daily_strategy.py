#!/usr/bin/env python3
"""
日线多因子选股策略 v2 — 每日信号 + 大事件监控

特性:
  - 每日收盘后计算因子得分，排名选股
  - 买入信号: 排名前 10% + 不持仓 + 有仓位空位
  - 卖出信号: 排名跌出前 50% / 止损-10% / 止盈+30%且排名跌出前30%
  - 大事件监控: 涨跌停、跳空缺口、成交量异动、均线交叉
  - 持仓上限 5 只，等权分配
  - 持仓最短持有 5 天，避免频繁交易

用法:
  python strategy/daily_strategy.py                  # 生成今日信号
  python strategy/daily_strategy.py --backtest 2024  # 回测
  python strategy/daily_strategy.py --notify          # 推送飞书
  python strategy/daily_strategy.py --monitor          # 持续监控模式
"""

import os, sys, json, sqlite3, argparse, warnings, time as _time
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from config_loader import get_db_path

DB_PATH = get_db_path()
SIGNAL_FILE = os.path.join(PROJECT_ROOT, 'data', 'latest_signals.json')

# ===================== 策略参数 =====================
MAX_POSITIONS = 5
TOP_BUY_PCT = 0.10
SELL_RANK_PCT = 0.50
STOP_LOSS = -0.10
TAKE_PROFIT = 0.30
TAKE_PROFIT_RANK = 0.30
MIN_HOLD_DAYS = 5           # 最短持有天数
INITIAL_CAPITAL = 500000    # 回测初始资金

# 因子权重
FACTOR_WEIGHTS = {
    'momentum_20': 0.25,
    'momentum_60': 0.20,
    'volatility_20': 0.15,
    'trend_strength': 0.20,
    'volume_ratio': 0.10,
    'rsi_14': 0.10,
}

# 大事件阈值
EVENT_THRESHOLDS = {
    'gap_pct': 0.05,        # 跳空 > 5%
    'vol_spike': 3.0,       # 成交量 > 3倍均量
    'limit_pct': 0.095,     # 接近涨跌停 (>9.5%)
    'ma_cross': True,       # 均线交叉
}


# ===================== 数据加载 (优化版) =====================
def load_all_daily_data(conn):
    """一次性加载所有日线数据到 DataFrame"""
    df = pd.read_sql("""
        SELECT symbol, date, open, high, low, close, volume
        FROM kline_daily ORDER BY symbol, date
    """, conn)
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    return df


def get_stock_names(conn):
    """获取所有股票名称映射"""
    rows = conn.execute("SELECT symbol, name FROM stock_info").fetchall()
    return {r['symbol']: r['name'] for r in rows}


# ===================== 因子批量计算 =====================
def compute_factors_for_symbol(group):
    """计算单只股票的因子"""
    group = group.sort_values('date').set_index('date')
    if len(group) < 120:
        return None

    close = group['close']
    vol = group['volume']
    f = pd.DataFrame(index=group.index)

    f['momentum_20'] = close / close.shift(20) - 1
    f['momentum_60'] = close / close.shift(60) - 1
    f['volatility_20'] = -close.pct_change().rolling(20).std()
    f['ma20'] = close.rolling(20).mean()
    f['trend_strength'] = (close - f['ma20']) / f['ma20']
    f['volume_ratio'] = vol.rolling(5).mean() / vol.rolling(20).mean()

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    f['rsi_14'] = 100 - (100 / (1 + rs))

    f['close'] = close
    f['open'] = group['open']
    f['high'] = group['high']
    f['low'] = group['low']
    f['volume'] = vol
    f['symbol'] = group['symbol'].iloc[0]

    return f


def compute_all_factors(df, batch_size=50):
    """
    分批计算因子，节省内存
    """
    results = {}
    symbols = sorted(df['symbol'].unique())
    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i:i+batch_size]
        batch = df[df['symbol'].isin(batch_symbols)]
        for sym, group in batch.groupby('symbol'):
            f = compute_factors_for_symbol(group)
            if f is not None:
                results[sym] = f
        # 释放内存
        del batch
        if (i + batch_size) % 100 == 0:
            print(f"  ... {min(i+batch_size, len(symbols))}/{len(symbols)} 只", flush=True)
    return results


# ===================== 信号生成 =====================
def compute_daily_scores(factors_dict, date):
    """计算某日的因子得分和排名"""
    rows = []
    for sym, df in factors_dict.items():
        if date not in df.index:
            continue
        row = df.loc[date]
        vals = {'symbol': sym, 'close': float(row['close'])}
        valid = 0
        for col in FACTOR_WEIGHTS:
            v = row.get(col)
            if pd.notna(v) and not np.isinf(v):
                vals[col] = float(v)
                valid += 1
            else:
                vals[col] = np.nan
        if valid >= 3:
            rows.append(vals)

    if not rows:
        return pd.DataFrame()

    df_s = pd.DataFrame(rows)
    for col in FACTOR_WEIGHTS:
        if col in df_s.columns:
            series = df_s[col].dropna()
            if len(series) > 0:
                mean, std = series.mean(), series.std()
                if std > 0:
                    df_s[f'{col}_z'] = (df_s[col] - mean) / std
                else:
                    df_s[f'{col}_z'] = 0.0
            else:
                df_s[f'{col}_z'] = 0.0

    df_s['score'] = sum(df_s.get(f'{c}_z', 0).fillna(0) * w for c, w in FACTOR_WEIGHTS.items())
    df_s['rank'] = df_s['score'].rank(ascending=False).astype(int)
    df_s['rank_pct'] = df_s['rank'] / len(df_s)
    return df_s.sort_values('score', ascending=False)


def generate_signals(scores, positions, name_map, today_str):
    """生成交易信号"""
    signals = []
    held = set(positions.keys())
    total = len(scores)
    if total == 0:
        return signals

    buy_threshold = int(total * TOP_BUY_PCT)
    sell_threshold = int(total * SELL_RANK_PCT)

    # 卖出检查
    for _, row in scores.iterrows():
        sym = row['symbol']
        if sym not in held:
            continue
        pos = positions[sym]
        pnl = (row['close'] - pos['cost_price']) / pos['cost_price']
        hold_days = (datetime.strptime(today_str, '%Y-%m-%d') -
                      datetime.strptime(pos['entry_date'], '%Y-%m-%d')).days

        reason = None
        if pnl <= STOP_LOSS:
            reason = f'止损 {pnl:.1%}'
        elif pnl >= TAKE_PROFIT and row['rank_pct'] > TAKE_PROFIT_RANK:
            reason = f'止盈 {pnl:.1%}'
        elif row['rank'] > sell_threshold and hold_days >= MIN_HOLD_DAYS:
            reason = f'排名下滑 {row["rank"]}/{total}'

        if reason:
            signals.append({
                'action': 'SELL', 'symbol': sym,
                'name': name_map.get(sym, sym),
                'price': round(row['close'], 2),
                'score': round(row['score'], 3),
                'rank': f"{row['rank']}/{total}",
                'pnl': f"{pnl:.1%}", 'reason': reason,
            })
            held.discard(sym)

    # 买入检查
    available = MAX_POSITIONS - len(held)
    if available > 0:
        for _, row in scores.iterrows():
            if available <= 0:
                break
            sym = row['symbol']
            if sym in held:
                continue
            if row['rank'] > buy_threshold:
                continue
            signals.append({
                'action': 'BUY', 'symbol': sym,
                'name': name_map.get(sym, sym),
                'price': round(row['close'], 2),
                'score': round(row['score'], 3),
                'rank': f"{row['rank']}/{total}",
                'pnl': '-', 'reason': f'排名前{TOP_BUY_PCT:.0%}',
            })
            available -= 1

    return signals


# ===================== 大事件监控 =====================
def check_events(factors_dict, positions, name_map, date):
    """检测持仓和关注股票的大事件"""
    events = []
    held = set(positions.keys())

    for sym, df in factors_dict.items():
        if date not in df.index:
            continue
        idx = df.index.get_loc(date)
        if idx < 1:
            continue

        today = df.iloc[idx]
        yesterday = df.iloc[idx - 1]
        close = today['close']
        prev_close = yesterday['close']

        # 1. 跳空缺口
        if prev_close > 0:
            gap = (today['open'] - yesterday['close']) / yesterday['close']
            if abs(gap) >= EVENT_THRESHOLDS['gap_pct']:
                direction = '向上跳空' if gap > 0 else '向下跳空'
                events.append({
                    'symbol': sym, 'name': name_map.get(sym, sym),
                    'type': '跳空缺口',
                    'detail': f'{direction} {gap:+.1%}',
                    'price': round(close, 2),
                    'is_holding': sym in held,
                })

        # 2. 涨跌停
        if prev_close > 0:
            chg = (close - prev_close) / prev_close
            if abs(chg) >= EVENT_THRESHOLDS['limit_pct']:
                direction = '涨停' if chg > 0 else '跌停'
                events.append({
                    'symbol': sym, 'name': name_map.get(sym, sym),
                    'type': direction,
                    'detail': f'{chg:+.1%}',
                    'price': round(close, 2),
                    'is_holding': sym in held,
                })

        # 3. 成交量异动
        if 'volume' in df.columns and idx >= 20:
            avg_vol = df['volume'].iloc[idx-20:idx].mean()
            if avg_vol > 0 and today['volume'] / avg_vol >= EVENT_THRESHOLDS['vol_spike']:
                events.append({
                    'symbol': sym, 'name': name_map.get(sym, sym),
                    'type': '放量异动',
                    'detail': f'量比 {today["volume"]/avg_vol:.1f}x',
                    'price': round(close, 2),
                    'is_holding': sym in held,
                })

        # 4. 均线交叉 (MA20 × MA60)
        if idx >= 60 and 'ma20' in df.columns:
            ma20 = df['ma20'].iloc[idx]
            ma60 = df['close'].rolling(60).mean().iloc[idx]
            if idx >= 1:
                ma20_prev = df['ma20'].iloc[idx-1]
                ma60_prev = df['close'].rolling(60).mean().iloc[idx-1]
                if pd.notna(ma20) and pd.notna(ma60) and pd.notna(ma20_prev) and pd.notna(ma60_prev):
                    if ma20_prev <= ma60_prev and ma20 > ma60:
                        events.append({
                            'symbol': sym, 'name': name_map.get(sym, sym),
                            'type': '金叉',
                            'detail': 'MA20 上穿 MA60',
                            'price': round(close, 2),
                            'is_holding': sym in held,
                        })
                    elif ma20_prev >= ma60_prev and ma20 < ma60:
                        events.append({
                            'symbol': sym, 'name': name_map.get(sym, sym),
                            'type': '死叉',
                            'detail': 'MA20 下穿 MA60',
                            'price': round(close, 2),
                            'is_holding': sym in held,
                        })

    return events


# ===================== 回测 (优化版) =====================
def backtest_optimized(factors_dict, start_date='2024-01-01', end_date=None):
    """快速回测：预计算因子，按日检查信号"""
    # 收集所有交易日
    all_dates = sorted(set(
        d for df in factors_dict.values() for d in df.index
        if str(d) >= start_date
    ))
    if end_date:
        all_dates = [d for d in all_dates if str(d) <= end_date]

    # 每天检查（但仅周五调仓，减少交易噪音）
    rebalance_dates = [d for d in all_dates if d.weekday() == 4]

    positions = {}
    cash = INITIAL_CAPITAL
    trades = []
    portfolio_values = []

    for date in all_dates:
        date_str = str(date).split(' ')[0]

        # 每日记录净值
        holdings_value = 0
        for sym, pos in list(positions.items()):
            if sym in factors_dict and date in factors_dict[sym].index:
                price = float(factors_dict[sym].loc[date, 'close'])
                holdings_value += pos['shares'] * price

        total_value = cash + holdings_value
        portfolio_values.append({'date': date_str, 'value': total_value})

        # 只在周五调仓
        if date not in rebalance_dates:
            continue

        scores = compute_daily_scores(factors_dict, date)
        if scores.empty:
            continue

        total = len(scores)
        buy_rank = int(total * TOP_BUY_PCT)
        sell_rank = int(total * SELL_RANK_PCT)

        # 卖出
        to_remove = []
        for _, row in scores.iterrows():
            sym = row['symbol']
            if sym not in positions:
                continue
            pos = positions[sym]
            pnl = (row['close'] - pos['cost']) / pos['cost']
            hold_days = (date - pos['entry_date']).days

            sell = False
            reason = ''
            if pnl <= STOP_LOSS:
                sell, reason = True, '止损'
            elif pnl >= TAKE_PROFIT and row['rank'] > int(total * TAKE_PROFIT_RANK):
                sell, reason = True, '止盈'
            elif row['rank'] > sell_rank and hold_days >= MIN_HOLD_DAYS:
                sell, reason = True, '排名下滑'

            if sell:
                value = pos['shares'] * row['close']
                cash += value
                trades.append({
                    'date': date_str, 'symbol': sym, 'action': 'SELL',
                    'price': round(row['close'], 2), 'shares': pos['shares'],
                    'pnl': f"{pnl:.1%}", 'reason': reason,
                })
                to_remove.append(sym)

        for sym in to_remove:
            del positions[sym]

        # 买入 (等权)
        available = MAX_POSITIONS - len(positions)
        if available > 0 and cash > 0:
            total_equity = cash + holdings_value
            per_stock = max(total_equity / MAX_POSITIONS, 10000)
            for _, row in scores.iterrows():
                if available <= 0:
                    break
                sym = row['symbol']
                if sym in positions or row['rank'] > buy_rank:
                    continue
                price = row['close']
                shares = int(per_stock / price / 100) * 100
                if shares <= 0:
                    continue
                cost = shares * price
                if cost > cash * 0.5:  # 单只不超过剩余现金 50%
                    continue
                cash -= cost
                positions[sym] = {
                    'cost': price, 'shares': shares,
                    'entry_date': date,
                }
                trades.append({
                    'date': date_str, 'symbol': sym, 'action': 'BUY',
                    'price': round(price, 2), 'shares': shares,
                    'pnl': '-', 'reason': '排名买入',
                })
                available -= 1

    # 输出结果
    if not portfolio_values:
        print("无交易记录")
        return

    df_pv = pd.DataFrame(portfolio_values)
    df_pv['date'] = pd.to_datetime(df_pv['date'])
    df_pv.set_index('date', inplace=True)

    initial = INITIAL_CAPITAL
    final = df_pv['value'].iloc[-1]
    total_return = (final - initial) / initial
    days = (df_pv.index[-1] - df_pv.index[0]).days
    annual_return = (1 + total_return) ** (365 / max(days, 1)) - 1

    returns = df_pv['value'].pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    max_dd = (df_pv['value'] / df_pv['value'].cummax() - 1).min()

    buys = [t for t in trades if t['action'] == 'BUY']
    sells = [t for t in trades if t['action'] == 'SELL']
    win_sells = [t for t in sells if float(t['pnl'].replace('%', '')) > 0]

    print(f"\n{'='*60}")
    print(f"📊 回测 ({start_date} ~ {str(df_pv.index[-1])[:10]})")
    print(f"{'='*60}")
    print(f"  初始资金: ¥{initial:,.0f}")
    print(f"  最终资金: ¥{final:,.0f}")
    print(f"  总收益率: {total_return:+.2%}")
    print(f"  年化收益: {annual_return:+.2%}")
    print(f"  夏普比率: {sharpe:.2f}")
    print(f"  最大回撤: {max_dd:.2%}")
    print(f"  年化波动: {returns.std() * np.sqrt(252):.2%}")
    print(f"  交易次数: {len(trades)} (买{len(buys)} 卖{len(sells)})")
    if sells:
        print(f"  胜率: {len(win_sells)/len(sells):.1%} ({len(win_sells)}/{len(sells)})")

    print(f"\n📋 最近 10 笔交易:")
    for t in trades[-10:]:
        icon = '🟢' if t['action'] == 'BUY' else '🔴'
        print(f"  {icon} {t['date']} {t['action']:4s} {t['symbol']:12s} "
              f"@{t['price']:.2f} x{t['shares']:6d}  {t['pnl']:>8s}  {t['reason']}")

    return df_pv, trades


# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser(description='日线多因子选股策略 v2')
    parser.add_argument('--backtest', type=str, nargs='?', const='2024',
                        help='回测模式，年份')
    parser.add_argument('--notify', action='store_true', help='推送飞书通知')
    parser.add_argument('--monitor', action='store_true', help='持续监控模式')
    parser.add_argument('--events', action='store_true', help='仅检查大事件')
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print("📡 加载数据...", flush=True)
    df_all = load_all_daily_data(conn)
    name_map = get_stock_names(conn)
    print(f"  {len(df_all):,} 行, {df_all['symbol'].nunique()} 只股票", flush=True)

    print("📊 计算因子...", flush=True)
    factors_dict = compute_all_factors(df_all)
    print(f"  {len(factors_dict)} 只有效因子数据", flush=True)

    if args.backtest:
        start = f"{args.backtest}-01-01"
        backtest_optimized(factors_dict, start_date=start)
        conn.close()
        return

    # 获取最新交易日
    all_dates = sorted(set(d for df in factors_dict.values() for d in df.index))
    latest_date = all_dates[-1]
    today_str = str(latest_date).split(' ')[0]
    print(f"📅 最新交易日: {today_str}", flush=True)

    # 计算评分
    scores = compute_daily_scores(factors_dict, latest_date)
    if scores.empty:
        print("❌ 无有效数据")
        conn.close()
        return

    # 加载持仓
    positions = {}
    rows = conn.execute(
        "SELECT symbol, stock_name, shares, cost_price, entry_date FROM positions"
    ).fetchall()
    for r in rows:
        positions[r['symbol']] = {
            'cost_price': r['cost_price'],
            'shares': r['shares'],
            'name': r['stock_name'],
            'entry_date': r['entry_date'] or '2026-01-01',
        }

    # 生成信号
    signals = generate_signals(scores, positions, name_map, today_str)

    # 大事件
    events = check_events(factors_dict, positions, name_map, latest_date)

    # 输出
    print(f"\n{'='*60}")
    print(f"📊 策略信号 ({today_str})")
    print(f"{'='*60}")
    print(f"  股票池: {len(scores)} 只 | 持仓: {len(positions)} 只 | 信号: {len(signals)} 条")

    if signals:
        print(f"\n📋 交易信号:")
        for s in signals:
            icon = '🟢 买入' if s['action'] == 'BUY' else '🔴 卖出'
            print(f"  {icon} | {s['symbol']:12s} {s['name'][:8]:8s} | "
                  f"@{s['price']:.2f} | 得分:{s['score']:.3f} | "
                  f"排名:{s['rank']} | 盈亏:{s['pnl']} | {s['reason']}")
    else:
        print(f"\n  无交易信号")

    # 持仓大事件
    holding_events = [e for e in events if e['is_holding']]
    if holding_events:
        print(f"\n⚠️ 持仓大事件:")
        for e in holding_events:
            print(f"  {e['symbol']} {e['name']} | {e['type']} | {e['detail']}")

    # Top 15
    print(f"\n🏆 综合得分 Top-15:")
    for _, row in scores.head(15).iterrows():
        held = ' ★' if row['symbol'] in positions else ''
        name = name_map.get(row['symbol'], '')[:8]
        print(f"  {row['rank']:3d}. {row['symbol']:12s} {name:8s} "
              f"得分:{row['score']:.3f}  价格:{row['close']:.2f}{held}")

    # 保存信号
    os.makedirs(os.path.dirname(SIGNAL_FILE), exist_ok=True)
    with open(SIGNAL_FILE, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'date': today_str,
            'signals': signals,
            'events': [e for e in events if e['is_holding']],
            'top_15': [{'symbol': r['symbol'], 'score': round(r['score'], 3),
                        'close': round(r['close'], 2), 'rank': r['rank']}
                       for _, r in scores.head(15).iterrows()],
            'positions': {s: {'name': p['name'], 'cost': p['cost_price']}
                          for s, p in positions.items()},
        }, f, ensure_ascii=False, indent=2)

    # 飞书推送
    if args.notify:
        notify_feishu(signals, events, scores, positions, name_map, today_str)

    conn.close()
    print(f"\n✅ 信号已保存: {SIGNAL_FILE}")


def notify_feishu(signals, events, scores, positions, name_map, today_str):
    """生成飞书推送文本"""
    lines = [f"📊 量化信号 {today_str}"]

    # 信号
    if signals:
        buy = [s for s in signals if s['action'] == 'BUY']
        sell = [s for s in signals if s['action'] == 'SELL']
        if buy:
            lines.append(f"\n🟢 买入 ({len(buy)}):")
            for s in buy:
                lines.append(f"  {s['symbol']} {s['name']} @{s['price']} 排名{s['rank']}")
        if sell:
            lines.append(f"\n🔴 卖出 ({len(sell)}):")
            for s in sell:
                lines.append(f"  {s['symbol']} {s['name']} 盈亏{s['pnl']} | {s['reason']}")
    else:
        lines.append("无交易信号")

    # 持仓大事件
    holding_events = [e for e in events if e['is_holding']]
    if holding_events:
        lines.append(f"\n⚠️ 持仓预警:")
        for e in holding_events[:5]:
            lines.append(f"  {e['symbol']} {e['name']} | {e['type']} {e['detail']}")

    # 持仓概览
    if positions:
        lines.append(f"\n📈 持仓 ({len(positions)}/{MAX_POSITIONS}):")
        for sym, pos in positions.items():
            if sym in scores['symbol'].values:
                srow = scores[scores['symbol'] == sym].iloc[0]
                lines.append(f"  {sym} {pos['name']} @{pos['cost_price']:.2f} "
                             f"排名{srow['rank']}/{len(scores)}")

    text = '\n'.join(lines)
    print(f"\n📤 飞书推送:\n{text}")
    return text


if __name__ == '__main__':
    main()