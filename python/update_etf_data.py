#!/usr/bin/env python3
"""数据更新器 — 新浪 API (免 token) + akshare (港股)。

东财 push2his 频繁挂掉, 改用新浪 JSONP 接口, 更稳定。

用法:
  python update_etf_data.py                 # 更新默认标的(ETF + A股持仓)
  python update_etf_data.py --hk            # 同时拉取港股成分股
  python update_etf_data.py --daily-only
  python update_etf_data.py --symbols 300124.SZ 600048.SH
"""
import os, sys, time, json, sqlite3, argparse
import requests
import pandas as pd
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))            # python/
DB_PATH = os.path.join(ROOT, 'data', 'stock_data.db')

# 默认关注标的
DEFAULT_SYMBOLS = [
    '159792.SZ',   # 港股通互联网ETF (主仓)
    '513050.SH',   # 中概互联网ETF
    '513330.SH',   # 恒生互联网ETF
    '159607.SZ',   # 中概互联网ETF(易方达)
    '300124.SZ',   # 汇川技术
    '300015.SZ',   # 爱尔眼科
    '600048.SH',   # 保利发展
]

# HSTECH 成分股 (用 akshare 拉取)
HK_COMPONENTS = [
    '00700.HK', '09988.HK', '03690.HK', '09618.HK', '01024.HK',
    '01810.HK', '09999.HK', '09888.HK', '02015.HK', '00981.HK',
]

# 新浪 API 符号映射
def to_sina_symbol(sym: str) -> str:
    """159792.SZ -> sz159792, 513050.SH -> sh513050"""
    code = sym[:6]
    if sym.endswith('.SZ'):
        return f'sz{code}'
    if sym.endswith('.SH'):
        return f'sh{code}'
    raise ValueError(f'unsupported symbol {sym}')


def fetch_sina_daily(symbol: str):
    """通过新浪 API 拉取日线。symbol 如 'sz300124'。返回 list[dict]."""
    url = 'https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData'
    r = requests.get(url, params={'symbol': symbol, 'scale': 240, 'ma': 'no', 'datalen': 800},
                     timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
    data = r.json()
    if not data or not isinstance(data, list):
        return []
    rows = []
    for d in data:
        rows.append(dict(
            date=d['day'],
            open=float(d['open']), high=float(d['high']),
            low=float(d['low']), close=float(d['close']),
            volume=float(d['volume'])))
    return rows


def fetch_hk_klines(symbol: str):
    """通过 akshare 拉取港股日线。symbol 如 '00700.HK'。"""
    import akshare as ak
    code = symbol[:5]
    df = ak.stock_hk_daily(symbol=code, adjust='qfq')
    df = df.reset_index()
    rows = []
    for _, row in df.iterrows():
        rows.append(dict(
            date=str(row['date'])[:10],
            open=float(row['open']), high=float(row['high']),
            low=float(row['low']), close=float(row['close']),
            volume=float(row['volume'])))
    return rows


def ensure_south_flow_table(conn):
    """南向资金净流入(港股通)历史表。南向=内地资金买港股,是港股通互联网ETF 的核心驱动。"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS south_flow (
            trade_date TEXT PRIMARY KEY,
            net_buy    REAL,   -- 当日成交净买额(亿元, 正=净流入港股)
            buy_amt    REAL,   -- 买入成交额
            sell_amt   REAL,   -- 卖出成交额
            updated_at TEXT
        )
    """)
    conn.commit()


def fetch_south_flow():
    """akshare 拉取南向资金历史日序列。返回 list[dict]。"""
    import akshare as ak
    df = ak.stock_hsgt_hist_em(symbol='南向资金')
    df = df.dropna(subset=['当日成交净买额']).copy()
    df['日期'] = pd.to_datetime(df['日期']).dt.strftime('%Y-%m-%d')
    rows = []
    for _, r in df.iterrows():
        rows.append(dict(
            trade_date=r['日期'],
            net_buy=float(r['当日成交净买额']),
            buy_amt=float(r.get('买入成交额', 0) or 0),
            sell_amt=float(r.get('卖出成交额', 0) or 0),
        ))
    return rows


def update_south_flow(conn):
    """全量 upsert 南向资金历史(只插新日期,已存在不覆盖,除非 net_buy 为空)。"""
    ensure_south_flow_table(conn)
    try:
        rows = fetch_south_flow()
    except Exception as e:
        print(f"  南向资金拉取失败: {e}")
        return 0
    n = 0
    for r in rows:
        conn.execute("""
            INSERT INTO south_flow(trade_date, net_buy, buy_amt, sell_amt, updated_at)
            VALUES(?,?,?,?,?)
            ON CONFLICT(trade_date) DO UPDATE SET
                net_buy=excluded.net_buy, buy_amt=excluded.buy_amt,
                sell_amt=excluded.sell_amt, updated_at=excluded.updated_at
            WHERE south_flow.net_buy IS NULL OR south_flow.net_buy != excluded.net_buy
        """, (r['trade_date'], r['net_buy'], r['buy_amt'], r['sell_amt'],
              datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        n += 1
    conn.commit()
    return n


def upsert_daily(conn, symbol, rows, replace_all=True):
    cur = conn.cursor()
    if replace_all:
        cur.execute("DELETE FROM kline_daily WHERE symbol=?", (symbol,))
    n = 0
    for r in rows:
        cur.execute(
            "INSERT OR REPLACE INTO kline_daily (symbol,date,open,high,low,close,volume,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (symbol, r['date'], r['open'], r['high'], r['low'], r['close'],
             r['volume'], datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        n += 1
    conn.commit()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', nargs='*', default=None,
                    help='手动指定标的(默认: 默认列表+持仓动态合并)')
    ap.add_argument('--daily-only', action='store_true')
    ap.add_argument('--hk', action='store_true', help='同时拉取港股成分股(akshare)')
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    print(f"DB: {DB_PATH} ({os.path.getsize(DB_PATH)/1e9:.2f}GB)")

    # 南向资金(港股通净流入) — 港股通互联网ETF 的核心资金面驱动
    print("\n--- 南向资金 (akshare 南向资金历史) ---")
    try:
        nsouth = update_south_flow(conn)
        latest = conn.execute("SELECT MAX(trade_date), net_buy FROM south_flow").fetchone()
        print(f"  南向资金: upsert {nsouth} 行 | 最新 {latest[0]} net={latest[1]:.2f}亿")
    except Exception as e:
        print(f"  南向资金更新失败: {e}")

    # 动态合并标的: 默认列表 + 持仓中的股票
    symbols = list(args.symbols) if args.symbols else list(DEFAULT_SYMBOLS)
    if args.symbols is None:
        try:
            cur = conn.cursor()
            cur.execute('SELECT DISTINCT symbol FROM positions')
            for r in cur.fetchall():
                if r[0] not in symbols:
                    symbols.append(r[0])
                    print(f"  + 从持仓添加: {r[0]}")
        except Exception:
            pass

    # A股 + ETF (新浪 API)
    print("\n--- A股/ETF (新浪) ---")
    for sym in symbols:
        try:
            sina_sym = to_sina_symbol(sym)
            rows = fetch_sina_daily(sina_sym)
            if rows:
                n = upsert_daily(conn, sym, rows, replace_all=True)
                print(f"  {sym}: {len(rows)} bars ({rows[0]['date']}~{rows[-1]['date']})")
            else:
                print(f"  {sym}: 无数据")
        except Exception as e:
            print(f"  {sym}: 错误 {e}")
        time.sleep(0.3)

    # 港股成分股 (akshare)
    if args.hk:
        print("\n--- 港股成分股 (akshare) ---")
        for sym in HK_COMPONENTS:
            try:
                rows = fetch_hk_klines(sym)
                if rows:
                    n = upsert_daily(conn, sym, rows, replace_all=True)
                    print(f"  {sym}: {len(rows)} bars ({rows[0]['date']}~{rows[-1]['date']})")
            except Exception as e:
                print(f"  {sym}: 错误 {e}")
            time.sleep(0.5)

    conn.close()
    print("\n完成")


if __name__ == '__main__':
    main()