#!/usr/bin/env python3
"""baostock补全数据 - 不限频

1. 个股30分钟K线 (2020-2024, 每只约7700条)
2. 大盘日线 (2014-2022)
3. 北向数据已有2649条(2014-2026), 不需要补

用法: nohup python3 -u fetch_baostock_history.py > /tmp/baostock_log.txt 2>&1 &
预计: 372只×5年×8次(按季度拉)=14900次, 不限频约2-3小时
"""
import baostock as bs, sqlite3, time, json, os, sys
from datetime import datetime

DB_PATH = '/root/github/stock-quant/python/data/stock_data.db'
PROGRESS_FILE = '/tmp/baostock_progress.json'
conn = sqlite3.connect(DB_PATH)

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            d = json.load(f)
            # 转换为set加速查找, 再存回set格式
            if isinstance(d.get('kline_done'), list):
                d['kline_done'] = set(d['kline_done'])
            if isinstance(d.get('hs300_done'), list):
                d['hs300_done'] = set(d['hs300_done'])
            return d
    return {'hs300_done': set(), 'kline_done': set(), 'total_new': 0, 'total_fail': 0}

def save_progress(p):
    # 只每隔50次保存, 不用每次stock×quarter都写
    # 写入时转回list (set不可JSON序列化)
    d = {k: (list(v) if isinstance(v, set) else v) for k, v in p.items()}
    with open(PROGRESS_FILE, 'w') as f: json.dump(d, f)

progress = load_progress()
_save_counter = [0]  # 用list包装以便在嵌套函数中修改

def save_if_needed():
    _save_counter[0] += 1
    if _save_counter[0] % 50 == 0:
        save_progress(progress)

lg = bs.login()
print(f"baostock登录: {lg.error_msg}")

# ====== 1. 补大盘日线 (2014-2022) ======
print("=== 补大盘日线 (2014-2022) ===")
years = range(2014, 2023)

for year in years:
    if str(year) in progress['hs300_done']:
        print(f"{year}: 已完成"); continue
    
    rs = bs.query_history_k_data_plus("sh.000300", "date,open,high,low,close,volume,pctChg",
        start_date=f'{year}-01-01', end_date=f'{year}-12-31', frequency="d", adjustflag="1")
    rows = []
    while (rs.error_code == '0') and rs.next(): rows.append(rs.get_row_data())
    
    new = 0
    for row in rows:
        td = row[0]  # date
        if not conn.execute("SELECT 1 FROM hs300_daily WHERE trade_date=?", (td,)).fetchone():
            conn.execute("INSERT OR IGNORE INTO hs300_daily VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (td, float(row[1] or 0), float(row[4] or 0), float(row[2] or 0), float(row[3] or 0),
                 float(row[5] or 0), 0, float(row[6] or 0), 0, 0, 0, 0))
            new += 1
    
    conn.commit()
    progress['hs300_done'].add(str(year))
    progress['total_new'] += new
    save_if_needed()
    print(f"{year}: {len(rows)}条(新增{new})")

f = conn.execute("SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM hs300_daily").fetchone()
print(f"✅ 大盘数据: {f[0]}条 ({f[1]} ~ {f[2]})")

# ====== 2. 补个股30分钟K线 (2020-2024) ======
print("\n=== 补个股30分钟K线 (2020-2024) ===")

# ts_code -> baostock code: 000001.SZ -> sz.000001, 600519.SH -> sh.600519
def to_bs_code(ts_code):
    prefix = ts_code[:6]
    if ts_code.endswith('.SZ'): return f"sz.{prefix}"
    elif ts_code.endswith('.SH'): return f"sh.{prefix}"
    return None

symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM kline_30m ORDER BY symbol").fetchall()]
print(f"股票数: {len(symbols)}只")

# 按季度拉取(每只每年4次, 避免单次数据量太大)
quarters = [
    ('2020-01-01', '2020-06-30'), ('2020-07-01', '2020-12-31'),
    ('2021-01-01', '2021-06-30'), ('2021-07-01', '2021-12-31'),
    ('2022-01-01', '2022-06-30'), ('2022-07-01', '2022-12-31'),
    ('2023-01-01', '2023-06-30'), ('2023-07-01', '2023-12-31'),
    ('2024-01-01', '2024-06-30'), ('2024-07-01', '2024-12-31'),
]

total_calls = len(symbols) * len(quarters)
print(f"预计调用: {total_calls}次 (不限频, 约2-3小时)")

for i, sym in enumerate(symbols):
    bs_code = to_bs_code(sym)
    if not bs_code: continue
    
    for start, end in quarters:
        key = f"{sym}_{start[:7]}"
        if key in progress['kline_done']: continue
        
        rs = bs.query_history_k_data_plus(bs_code, "date,time,open,high,low,close,volume",
            start_date=start, end_date=end, frequency="30")
        
        rows = []
        while (rs.error_code == '0') and rs.next(): rows.append(rs.get_row_data())
        
        new = 0
        for row in rows:
            # baostock时间: date=2024-01-02, time=20240102100000000
            dt_str = f"{row[0]} {row[1]}"  # 组合成 2024-01-02 100000000
            # 转换: 20240102100000000 -> 10:00:00
            t = row[1]
            if len(t) >= 12:
                hour = t[8:10]
                minute = t[10:12]
                dt_str = f"{row[0]} {hour}:{minute}:00"
            else:
                continue
            
            if not conn.execute("SELECT 1 FROM kline_30m WHERE symbol=? AND date=?", (sym, dt_str)).fetchone():
                conn.execute("INSERT OR IGNORE INTO kline_30m (symbol,date,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?)",
                    (sym, dt_str, float(row[2] or 0), float(row[3] or 0),
                     float(row[4] or 0), float(row[5] or 0),
                     int(row[6] or 0)))
                new += 1
        
        if new > 0: conn.commit()
        progress['kline_done'].add(key)
        progress['total_new'] += new
        save_if_needed()
        
        if i % 20 == 0 and new > 0:
            done_pct = len(progress['kline_done']) / total_calls * 100
            print(f"[{done_pct:.1f}%] {sym} {start[:7]}: {len(rows)}条(新增{new})")
    
    if (i+1) % 50 == 0:
        f = conn.execute("SELECT COUNT(*) FROM kline_30m").fetchone()[0]
        print(f"[{(i+1)/len(symbols)*100:.1f}%] 总K线:{f} 新增:{progress['total_new']}")

conn.commit()
f = conn.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM kline_30m").fetchone()
print(f"\n✅ K线数据: {f[0]}条 ({f[1]} ~ {f[2]})")
print(f"总新增: {progress['total_new']}, 失败: {progress['total_fail']}")

bs.logout()
conn.close()
os.remove(PROGRESS_FILE)