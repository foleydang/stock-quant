#!/usr/bin/env python3
"""补全个股日线数据 (baostock)

用法: nohup python3 -u fetch_daily_kline.py > /tmp/daily_fetch.log 2>&1 &
预计: 372只×13年×1次(按年拉)=4836次, 不限频约1-2小时
"""

import baostock as bs
import sqlite3
import time
import json
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'stock_data.db')
PROGRESS_FILE = '/tmp/daily_progress.json'

conn = sqlite3.connect(DB_PATH)

# 创建日线表
conn.execute("""
    CREATE TABLE IF NOT EXISTS kline_daily (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        date TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        UNIQUE(symbol, date)
    )
""")
conn.execute("CREATE INDEX IF NOT EXISTS idx_kline_daily_symbol ON kline_daily(symbol)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_kline_daily_date ON kline_daily(date)")
conn.commit()

# ts_code -> baostock: 000001.SZ -> sz.000001
def to_bs_code(ts_code):
    prefix = ts_code[:6]
    if ts_code.endswith('.SZ'): return f"sz.{prefix}"
    elif ts_code.endswith('.SH'): return f"sh.{prefix}"
    return None

# 进度
progress = {}
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        d = json.load(f)
        if isinstance(d.get('done'), list): 
            d['done'] = set(d['done'])
        progress = d
if 'done' not in progress:
    progress['done'] = set()
if 'total_new' not in progress:
    progress['total_new'] = 0

def save_p():
    d = {k: (list(v) if isinstance(v, set) else v) for k, v in progress.items()}
    with open(PROGRESS_FILE, 'w') as f: json.dump(d, f)

save_counter = [0]
def checkpoint():
    save_counter[0] += 1
    if save_counter[0] % 30 == 0:
        save_p()

# 登录
lg = bs.login()
print(f"登录: {lg.error_msg}")

# 获取股票
symbols = [r[0] for r in conn.execute(
    "SELECT DISTINCT symbol FROM kline_30m ORDER BY symbol").fetchall()]
print(f"股票: {len(symbols)}只")

# 按年拉: 2014-2026
years = list(range(2014, 2027))

total = len(symbols) * len(years)
print(f"预计: {total}次调用\n")

for i, sym in enumerate(symbols):
    bs_code = to_bs_code(sym)
    if not bs_code: continue

    for year in years:
        key = f"{sym}_{year}"
        if key in progress['done']:
            continue

        try:
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume",
                start_date=f'{year}-01-01',
                end_date=f'{year}-12-31',
                frequency="d",
                adjustflag="2"  # 前复权
            )

            rows = []
            while (rs.error_code == '0') and rs.next():
                rows.append(rs.get_row_data())

            new = 0
            for row in rows:
                if row[0] == '': continue
                if not conn.execute("SELECT 1 FROM kline_daily WHERE symbol=? AND date=?",
                                    (sym, row[0])).fetchone():
                    conn.execute(
                        "INSERT OR IGNORE INTO kline_daily (symbol,date,open,high,low,close,volume) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (sym, row[0],
                         float(row[1] or 0), float(row[2] or 0),
                         float(row[3] or 0), float(row[4] or 0),
                         float(row[5] or 0)))
                    new += 1

            if new > 0:
                conn.commit()

            progress['total_new'] += new
            progress['done'].add(key)
            checkpoint()

            if new > 0 and i % 50 == 0:
                pct = len(progress['done']) / total * 100
                print(f"[{pct:.1f}%] {sym} {year}: {len(rows)}条(新增{new})")

        except Exception as e:
            print(f"  ❌ {sym} {year}: {e}")
            time.sleep(1)

    if (i + 1) % 50 == 0:
        cnt = conn.execute("SELECT COUNT(*) FROM kline_daily").fetchone()[0]
        dates = conn.execute("SELECT MIN(date), MAX(date) FROM kline_daily").fetchone()
        print(f"[{i+1}/{len(symbols)}] 日线:{cnt:,}条 ({dates[0]}~{dates[1]}) 新增:{progress['total_new']}")

conn.commit()
f = conn.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM kline_daily").fetchone()
s = conn.execute("SELECT COUNT(DISTINCT symbol) FROM kline_daily").fetchone()
print(f"\n✅ 日线: {f[0]:,}条 ({f[1]}~{f[2]}), {s[0]}只")
print(f"总新增: {progress['total_new']}")

bs.logout()
conn.close()
if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)