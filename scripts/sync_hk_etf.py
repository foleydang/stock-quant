#!/usr/bin/env python3
"""
港股 + ETF 日线数据同步 (yfinance)
- HSTECH 成分股 (港股通互联网ETF 159792 的底仓): 10 只, 全量历史
- ETF: 159792.SZ 等 (Tushare/新浪 不覆盖的)
- 持仓中的港股动态补充
- 写入 kline_daily 表, 与 Tushare 数据格式一致

符号约定 (关键):
  yfinance 只认 4 位 (0700.HK), 但 etf159792_model.py 的 HSTECH_COMPONENTS
  用 5 位 (00700.HK) 查 DB。故 yf 用 4 位抓, DB 存 5 位 (db = yf 前补 '0')。

用法:
  /root/miniconda3/bin/python scripts/sync_hk_etf.py
"""

import sys, os, sqlite3
import yfinance as yf
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'python', 'data', 'stock_data.db')

# HSTECH 成分股 (与 etf159792_model.py 的 HSTECH_COMPONENTS 一致, 5位 DB 符号)
# (db_symbol, yf_symbol, 名称, 权重) — yf_symbol 是 yfinance 认的 4 位形式
HSTECH_COMPONENTS = [
    ('00700.HK', '0700.HK', '腾讯',   0.09),
    ('09988.HK', '9988.HK', '阿里',   0.08),
    ('03690.HK', '3690.HK', '美团',   0.08),
    ('09618.HK', '9618.HK', '京东',   0.07),
    ('01024.HK', '1024.HK', '快手',   0.07),
    ('01810.HK', '1810.HK', '小米',   0.07),
    ('09999.HK', '9999.HK', '网易',   0.06),
    ('09888.HK', '9888.HK', '百度',   0.05),
    ('02015.HK', '2015.HK', '理想',   0.05),
    ('00981.HK', '0981.HK', '中芯',   0.04),
]

# Tushare/新浪 不覆盖的 ETF (yfinance 对 A股 ETF 支持有限, 仅供补漏)
ETF_SYMBOLS = ['159792.SZ']

# 港股成分股回填起始日 (ETF 159792 成立于 2021-09, 成分股需更长历史算 ma60)
HK_START = '2019-01-01'


def get_position_symbols(conn):
    """从持仓表获取港股和 ETF 品种 (4位/5位都收, 之后统一存 5位)"""
    extra = []
    try:
        rows = conn.execute(
            "SELECT symbol FROM positions WHERE symbol LIKE '%.HK' OR symbol LIKE '159%' OR symbol LIKE '51%'"
        ).fetchall()
        for r in rows:
            if r[0] not in extra:
                extra.append(r[0])
    except Exception:
        pass
    return extra


def to_yf_symbol(symbol):
    """DB 5位 -> yfinance 4位 (去首个 0); 非 .HK 原样返回。"""
    if symbol.endswith('.HK') and symbol.startswith('0') and len(symbol) == 9:  # 00700.HK
        return symbol[1:]
    return symbol


def to_db_symbol(symbol):
    """yfinance 4位 -> DB 5位 (前补 0); 0981.HK -> 00981.HK。"""
    if symbol.endswith('.HK') and not symbol.startswith('0') and len(symbol) == 8:  # 0700.HK
        return '0' + symbol
    return symbol


def sync_full(conn, yf_symbol, db_symbol, start=HK_START):
    """抓全量历史 (从 start 至今), INSERT OR REPLACE 进 kline_daily, 存 db_symbol。"""
    try:
        ticker = yf.Ticker(yf_symbol)
        end = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        df = ticker.history(start=start, end=end)
        if df is None or len(df) == 0:
            return 0, f"{yf_symbol} 无数据"
        new = 0
        for idx, row in df.iterrows():
            date_str = idx.strftime('%Y-%m-%d')
            # NaN 检查 (yfinance 偶发全 NaN 行)
            if any(row[c] != row[c] for c in ['Open', 'High', 'Low', 'Close']):  # NaN != NaN
                continue
            vol = row['Volume']
            vol = int(vol) if vol == vol else 0  # NaN -> 0
            conn.execute(
                """INSERT OR REPLACE INTO kline_daily
                   (symbol, date, open, high, low, close, volume, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (db_symbol, date_str,
                 float(row['Open']), float(row['High']),
                 float(row['Low']), float(row['Close']), vol,
                 datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            new += 1
        conn.commit()
        rng = f"{df.index[0].strftime('%Y-%m-%d')}~{df.index[-1].strftime('%Y-%m-%d')}"
        return new, rng
    except Exception as e:
        return 0, f"{yf_symbol} 错误: {e}"


def main():
    conn = sqlite3.connect(DB_PATH)

    print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📡 1/2 HSTECH 成分股 ({len(HSTECH_COMPONENTS)} 只, 全量历史 from {HK_START})")
    for db_sym, yf_sym, name, _w in HSTECH_COMPONENTS:
        n, info = sync_full(conn, yf_sym, db_sym, start=HK_START)
        print(f"  {db_sym} {name}: +{n} 条 ({info})")

    # ETF + 持仓动态补充
    extra = get_position_symbols(conn)
    etf_and_pos = [s for s in (ETF_SYMBOLS + extra)
                   if not s.endswith('.HK')]  # 港股持仓上面已覆盖成分股; 其余单独抓
    # 港股持仓若不在成分股清单, 也抓
    hk_pos = [s for s in extra if s.endswith('.HK') and s not in [c[0] for c in HSTECH_COMPONENTS]]

    print(f"\n📡 2/2 港股持仓 + ETF: {hk_pos + etf_and_pos}")
    for sym in hk_pos + etf_and_pos:
        yf_sym = to_yf_symbol(sym) if sym.endswith('.HK') else sym
        db_sym = to_db_symbol(sym) if sym.endswith('.HK') else sym
        n, info = sync_full(conn, yf_sym, db_sym, start=HK_START)
        print(f"  {db_sym}: +{n} 条 ({info})")

    # 验证
    print("\n最新数据:")
    for db_sym, _yf, name, _w in HSTECH_COMPONENTS:
        row = conn.execute(
            "SELECT date, close FROM kline_daily WHERE symbol=? ORDER BY date DESC LIMIT 1",
            (db_sym,)).fetchone()
        print(f"  {db_sym} {name}: {'最新 ' + row[0] + ' close=' + f'{row[1]:.2f}' if row else '无数据'}")

    conn.close()
    print(f"\n✅ 港股/ETF 同步完成")


if __name__ == '__main__':
    main()
