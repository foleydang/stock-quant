#!/usr/bin/env python3
"""
日线选股推理 — 轻量版，直接从 SQLite 读取，不依赖 qlib
配合 intraday 模型 (XGBoost 30min) 做双模型择时

架构:
  日线模型 (本脚本) → 每日收盘后: "明天关注 X, Y, Z"
  30分钟模型 (infer.py) → 盘中: "X 现在是最佳入场时机"
"""

import os, sys, sqlite3, json, argparse, warnings
from datetime import datetime, timedelta
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from config_loader import get_db_path

DB_PATH = get_db_path()
SIGNAL_FILE = os.path.join(PROJECT_ROOT, 'data', 'daily_signals.json')


# ===================== 因子计算 =====================
def compute_features(df):
    """计算日线因子，返回最新一期"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    vol = df['volume'].values
    n = len(close)
    if n < 120:
        return None

    feats = {}

    # 动量 (多周期)
    for p in [5, 10, 20, 60]:
        if n > p:
            feats[f'mom_{p}d'] = close[-1] / close[-p-1] - 1

    # 波动率 (取反)
    if n >= 21:
        rets = np.diff(close[-21:]) / close[-21:-1]
        feats['vol_20d'] = -np.std(rets) if len(rets) > 0 else 0

    # 均线偏离
    for p in [5, 10, 20, 60]:
        if n >= p:
            ma = np.mean(close[-p:])
            feats[f'ma_dev_{p}d'] = (close[-1] - ma) / ma if ma > 0 else 0

    # 成交量比
    for p in [5, 20]:
        if n >= p:
            avg_vol = np.mean(vol[-p:])
            feats[f'vol_ratio_{p}d'] = vol[-1] / avg_vol if avg_vol > 0 else 1

    # RSI
    if n >= 15:
        delta = np.diff(close[-15:])
        gain = np.sum(delta[delta > 0]) if len(delta[delta > 0]) > 0 else 0
        loss = -np.sum(delta[delta < 0]) if len(delta[delta < 0]) > 0 else 0
        avg_gain = gain / 14
        avg_loss = loss / 14
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            feats['rsi_14'] = 100 - 100 / (1 + rs)
        else:
            feats['rsi_14'] = 100

    # ATR (波动率)
    if n >= 15:
        tr = np.maximum(high[-14:] - low[-14:],
                        np.abs(high[-14:] - close[-15:-1]),
                        np.abs(low[-14:] - close[-15:-1]))
        feats['atr_14'] = np.mean(tr)

    # 价格位置 (近期高低点)
    if n >= 20:
        hh = np.max(high[-20:])
        ll = np.min(low[-20:])
        feats['price_pos'] = (close[-1] - ll) / (hh - ll) if hh > ll else 0.5

    return feats


def compute_all_features(conn):
    """批量计算所有股票的最新因子"""
    symbols = [r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM kline_daily ORDER BY symbol"
    ).fetchall()]

    rows = []
    for sym in symbols:
        df = pd.read_sql(
            "SELECT date, open, high, low, close, volume FROM kline_daily "
            "WHERE symbol=? ORDER BY date", conn, params=(sym,)
        )
        if len(df) < 120:
            continue
        f = compute_features(df)
        if f is None:
            continue
        valid = sum(1 for v in f.values() if not np.isnan(v) and not np.isinf(v))
        if valid < 5:
            continue
        f['symbol'] = sym
        f['close'] = float(df['close'].iloc[-1])
        rows.append(f)

    return pd.DataFrame(rows)


def rank_stocks(df):
    """多因子排名打分"""
    feature_cols = [c for c in df.columns if c not in ['symbol', 'close']]
    df = df.copy()

    # 方向调整: 波动率/ATR 越低越好 (已取反)
    # 其余因子越高越好

    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[f'{col}_z'] = (df[col] - mean) / std
        else:
            df[f'{col}_z'] = 0

    # 综合得分
    z_cols = [f'{c}_z' for c in feature_cols]
    df['score'] = df[z_cols].mean(axis=1)
    df['rank'] = df['score'].rank(ascending=False).astype(int)
    return df.sort_values('score', ascending=False)


def get_stock_names(conn):
    rows = conn.execute("SELECT symbol, name FROM stock_info").fetchall()
    return {r[0]: r[1] for r in rows}


def get_positions(conn):
    rows = conn.execute(
        "SELECT symbol, stock_name, shares, cost_price FROM positions"
    ).fetchall()
    return {r[0]: {'name': r[1], 'shares': r[2], 'cost': r[3]} for r in rows}


def generate_signals(scores, positions, name_map, top_k=5):
    """生成买入/卖出信号"""
    signals = []
    held = set(positions.keys())
    total = len(scores)

    # 卖出检查: 持仓排名跌出前 50%
    sell_rank = int(total * 0.5)
    for _, row in scores.iterrows():
        sym = row['symbol']
        if sym not in held:
            continue
        pos = positions[sym]
        pnl = (row['close'] - pos['cost']) / pos['cost'] if pos['cost'] > 0 else 0
        if row['rank'] > sell_rank:
            signals.append({
                'action': 'SELL', 'symbol': sym,
                'name': name_map.get(sym, sym),
                'price': round(row['close'], 2),
                'score': round(row['score'], 3),
                'rank': f"{row['rank']}/{total}",
                'pnl': f"{pnl:.1%}",
                'reason': '排名下滑'
            })
            held.discard(sym)

    # 买入: Top-K 中不持仓的
    available = 5 - len(held)
    for _, row in scores.iterrows():
        if available <= 0:
            break
        sym = row['symbol']
        if sym in held:
            continue
        if row['rank'] > top_k * 2:  # 至少 Top-10
            break
        signals.append({
            'action': 'BUY', 'symbol': sym,
            'name': name_map.get(sym, sym),
            'price': round(row['close'], 2),
            'score': round(row['score'], 3),
            'rank': f"{row['rank']}/{total}",
            'pnl': '-',
            'reason': '综合排名靠前'
        })
        available -= 1

    return signals


def main():
    parser = argparse.ArgumentParser(description='日线选股推理')
    parser.add_argument('--top-k', type=int, default=5, help='Top-K')
    parser.add_argument('--json', action='store_true', help='JSON 输出')
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print(f"📡 计算日线因子...", flush=True)
    df = compute_all_features(conn)
    print(f"  {len(df)} 只股票", flush=True)

    scores = rank_stocks(df)
    name_map = get_stock_names(conn)
    positions = get_positions(conn)
    signals = generate_signals(scores, positions, name_map, args.top_k)

    today = datetime.now().strftime('%Y-%m-%d')
    print(f"\n{'='*60}")
    print(f"📊 日线选股 ({today})")
    print(f"{'='*60}")
    print(f"  股票池: {len(scores)} 只 | 持仓: {len(positions)} 只 | 信号: {len(signals)} 条")

    if signals:
        print(f"\n📋 信号:")
        for s in signals:
            icon = '🟢 买入' if s['action'] == 'BUY' else '🔴 卖出'
            print(f"  {icon} | {s['symbol']:12s} {s['name'][:8]:8s} | "
                  f"@{s['price']:.2f} | 得分:{s['score']:.3f} | "
                  f"排名:{s['rank']} | {s['reason']}")

    print(f"\n🏆 Top-{args.top_k}:")
    for _, row in scores.head(args.top_k).iterrows():
        held = ' ★持仓' if row['symbol'] in positions else ''
        name = name_map.get(row['symbol'], '')[:8]
        print(f"  {row['rank']:3d}. {row['symbol']:12s} {name:8s} "
              f"得分:{row['score']:.3f}  价格:{row['close']:.2f}{held}")

    # 保存
    os.makedirs(os.path.dirname(SIGNAL_FILE), exist_ok=True)
    result = {
        'date': today,
        'timestamp': datetime.now().isoformat(),
        'signals': signals,
        'top5': [{'symbol': r['symbol'], 'score': round(r['score'], 3),
                  'close': round(r['close'], 2), 'rank': r['rank']}
                 for _, r in scores.head(5).iterrows()],
        'positions': {s: {'name': p['name'], 'cost': p['cost'], 'pnl': f"{(scores[scores['symbol']==s]['close'].values[0] - p['cost'])/p['cost']:.1%}" if s in scores['symbol'].values else 'N/A'}
                      for s, p in positions.items()},
    }
    with open(SIGNAL_FILE, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    conn.close()
    print(f"\n✅ 信号已保存: {SIGNAL_FILE}")


if __name__ == '__main__':
    main()