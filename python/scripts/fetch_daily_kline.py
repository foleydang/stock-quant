#!/usr/bin/env python3
"""拉取个股日线数据 - baostock不限频

覆盖: 2014-2026, 372只沪深300成分股
用途: 日线模型训练(预测未来1-3天涨跌)

用法: nohup python3 -u fetch_daily_kline.py > /tmp/daily_log.txt 2>&1 &
预计: 372只 × 12年 = 约5分钟(不限频)
"""
import baostock as bs, sqlite3, time, json, os, sys
from datetime import datetime

DB_PATH = '/root/github/stock-quant/python/data/stock_data.db'
PROGRESS_FILE = '/tmp/daily_progress.json'
conn = sqlite3.connect(DB_PATH)

# 创建日线K线表(如果不存在)
conn.execute('''CREATE TABLE IF NOT EXISTS kline_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    trade_date TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    pct_chg REAL,
    created_at TEXT,
    updated_at TEXT,
    UNIQUE(symbol, trade_date)
)''')
conn.commit()

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f: return json.load(f)
    return {'done': [], 'total_new': 0}

def save_progress(p):
    with open(PROGRESS_FILE, 'w') as f: json.dump(p, f)

progress = load_progress()

lg = bs.login()
print(f"baostock登录: {lg.error_msg}")

# 获取股票列表(从30分钟K线表取)
symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM kline_30m ORDER BY symbol").fetchall()]
print(f"股票数: {len(symbols)}只")

# ts_code -> baostock code
def to_bs_code(ts_code):
    prefix = ts_code[:6]
    if ts_code.endswith('.SZ'): return f"sz.{prefix}"
    elif ts_code.endswith('.SH'): return f"sh.{prefix}"
    return None

for i, sym in enumerate(symbols):
    if sym in progress['done']:
        continue
    
    bs_code = to_bs_code(sym)
    if not bs_code: continue
    
    # 拉取2014-2026全部日线数据
    rs = bs.query_history_k_data_plus(bs_code, 
        "date,open,high,low,close,volume,pctChg",
        start_date='2014-01-01', end_date='2026-06-30', 
        frequency="d", adjustflag="1")  # 前复权
    
    rows = []
    while (rs.error_code == '0') and rs.next(): rows.append(rs.get_row_data())
    
    new = 0
    for row in rows:
        td = row[0]  # date
        if not td: continue
        if not conn.execute("SELECT 1 FROM kline_daily WHERE symbol=? AND trade_date=?", (sym, td)).fetchone():
            conn.execute("""INSERT OR IGNORE INTO kline_daily 
                (symbol, trade_date, open, high, low, close, volume, pct_chg)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (sym, td, float(row[1] or 0), float(row[2] or 0), float(row[3] or 0),
                 float(row[4] or 0), float(row[5] or 0), float(row[6] or 0)))
            new += 1
    
    conn.commit()
    progress['done'].append(sym)
    progress['total_new'] += new
    save_progress(progress)
    
    if (i+1) % 20 == 0 or new > 200:
        print(f"[{i+1}/{len(symbols)}] {sym}: {len(rows)}条(新增{new}) 累计{progress['total_new']}")

conn.commit()
f = conn.execute("SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM kline_daily").fetchone()
print(f"\n✅ 日线数据: {f[0]}条 ({f[1]} ~ {f[2]})")
print(f"总新增: {progress['total_new']}")

bs.logout()
conn.close()
os.remove(PROGRESS_FILE)