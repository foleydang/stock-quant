#!/usr/bin/env python3
"""
港股 + ETF 日线数据同步（yfinance）
- 港股：3690.HK, 0700.HK, 9988.HK（及用户持仓中的港股）
- ETF：159792.SZ 等（Tushare 不覆盖的 ETF）
- 写入 kline_daily 表，与 Tushare 数据格式一致

用法:
  /root/miniconda3/bin/python scripts/sync_hk_etf.py
"""

import sys, os, sqlite3
import yfinance as yf
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'python', 'data', 'stock_data.db')

# 需要同步的品种（港股 + Tushare 不覆盖的 ETF）
SYMBOLS = [
    # 港股
    '3690.HK',   # 美团-W
    '0700.HK',   # 腾讯控股
    '9988.HK',   # 阿里巴巴-W
    # ETF（Tushare pro.daily 不返回 ETF）
    '159792.SZ', # 港股通互联网ETF
    # 从持仓表动态补充港股
]

def get_position_symbols(conn):
    """从持仓表获取港股和 ETF 品种"""
    extra = []
    try:
        rows = conn.execute(
            "SELECT symbol FROM positions WHERE symbol LIKE '%.HK' OR symbol LIKE '159%' OR symbol LIKE '51%'"
        ).fetchall()
        for r in rows:
            if r[0] not in SYMBOLS:
                extra.append(r[0])
    except Exception:
        pass
    return extra


def sync_one(conn, symbol, days=30):
    """同步单个品种最近 N 天的日线数据"""
    try:
        ticker = yf.Ticker(symbol)
        end = datetime.now() + timedelta(days=1)  # yfinance end 是 exclusive
        start = end - timedelta(days=days + 1)
        df = ticker.history(start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))

        if df is None or len(df) == 0:
            return 0

        new = 0
        for idx, row in df.iterrows():
            date_str = idx.strftime('%Y-%m-%d')
            # 检查是否已存在
            exists = conn.execute(
                "SELECT 1 FROM kline_daily WHERE symbol=? AND date=?",
                (symbol, date_str)
            ).fetchone()
            if exists:
                continue

            conn.execute(
                """INSERT INTO kline_daily (symbol, date, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (symbol, date_str,
                 float(row['Open']), float(row['High']),
                 float(row['Low']), float(row['Close']),
                 int(row['Volume']))
            )
            new += 1

        conn.commit()
        return new
    except Exception as e:
        print(f"  {symbol} 错误: {e}")
        return 0


def main():
    conn = sqlite3.connect(DB_PATH)
    
    # 动态添加持仓中的港股/ETF
    extra = get_position_symbols(conn)
    all_symbols = list(SYMBOLS) + extra
    # 去重
    seen = set()
    all_symbols = [x for x in all_symbols if not (x in seen or seen.add(x))]

    print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📡 yfinance 同步 {len(all_symbols)} 个品种: {all_symbols}")

    total_new = 0
    for sym in all_symbols:
        n = sync_one(conn, sym)
        if n > 0:
            print(f"  {sym}: +{n} 条")
        total_new += n

    # 验证
    print()
    for sym in all_symbols:
        row = conn.execute(
            "SELECT date, close FROM kline_daily WHERE symbol=? ORDER BY date DESC LIMIT 1",
            (sym,)
        ).fetchone()
        if row:
            print(f"  {sym}: 最新 {row[0]} close={row[1]:.3f}")
        else:
            print(f"  {sym}: 无数据")

    conn.close()
    print(f"\n✅ 港股/ETF 同步完成: +{total_new} 条")


if __name__ == '__main__':
    main()