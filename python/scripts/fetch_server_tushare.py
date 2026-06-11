#!/usr/bin/env python3
import os
"""服务器后台补数据脚本 - tushare版

策略: 372只×3年(2022-2024) = 1116次调用, 1次/分钟 ≈ 18.6小时
服务器24h运行, nohup后台跑, 进度文件跟踪

用法: nohup python3 fetch_server_tushare.py > /tmp/fetch_log.txt 2>&1 &
"""
import tushare as ts, sqlite3, time, json, os, sys
from datetime import datetime

TUSHARE_TOKEN = '7a9014b18909e8cbce5109d7175f7b21ce37354eaff2371db0da2c58'
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'stock_data.db')
PROGRESS_FILE = '/tmp/fetch_progress.json'

ts.set_token(TUSHARE_TOKEN)
conn = sqlite3.connect(DB_PATH)

# 加载进度(断线重试)
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {'hs300_done': [], 'kline_done': [], 'total_new': 0, 'total_fail': 0}

def save_progress(p):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(p, f)

progress = load_progress()

# ====== 1. 补大盘数据 (2014-2022) ======
print("=== 补大盘数据 (2014-2022) ===")
years = [str(y) for y in range(2014, 2023)]

for year in years:
    if year in progress['hs300_done']:
        print(f"{year}: 已完成, 跳过")
        continue
    
    try:
        pro = ts.pro_api()
        df = pro.index_daily(ts_code='399300.SZ', start_date=f'{year}0101', end_date=f'{year}1231')
        if df is None or len(df) == 0:
            print(f"{year}: 无数据")
            progress['hs300_done'].append(year)
            save_progress(progress)
            time.sleep(61)
            continue
        
        new = 0
        for _, row in df.iterrows():
            td = str(row['trade_date'])
            if not conn.execute("SELECT 1 FROM hs300_daily WHERE trade_date=?", (td,)).fetchone():
                conn.execute("INSERT OR IGNORE INTO hs300_daily VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (td, float(row.get('open',0) or 0), float(row.get('close',0) or 0),
                     float(row.get('high',0) or 0), float(row.get('low',0) or 0),
                     float(row.get('vol',0) or 0), float(row.get('amount',0) or 0),
                     float(row.get('pct_chg',0) or 0), 0, 0, 0, 0))
                new += 1
        conn.commit()
        progress['hs300_done'].append(year)
        progress['total_new'] += new
        save_progress(progress)
        print(f"{year}: {len(df)}条(新增{new})")
    except Exception as e:
        print(f"{year}错误: {e}")
        if '频率超限' in str(e):
            print(f"限频等待65秒后重试...")
            time.sleep(65)
            continue  # 重试当年
    
    time.sleep(65)  # 限频1次/分钟+缓冲

f = conn.execute("SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM hs300_daily").fetchone()
print(f"✅ 大盘数据: {f[0]}条 ({f[1]} ~ {f[2]})")

# ====== 2. 补30分钟K线 (2022-2024) ======
print("\n=== 补30分钟K线 (2022-2024) ===")
symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM kline_30m ORDER BY symbol").fetchall()]
print(f"股票数: {len(symbols)}只")

kline_years = ['2022', '2023', '2024']
total_calls = len(symbols) * len(kline_years)
print(f"预计调用: {total_calls}次 ≈ {total_calls/60:.1f}小时")

for i, sym in enumerate(symbols):
    for year in kline_years:
        key = f"{sym}_{year}"
        if key in progress['kline_done']:
            continue
        
        try:
            df = ts.pro_bar(ts_code=sym, freq='30min', 
                           start_date=f'{year}0101', end_date=f'{year}1231')
            if df is None or len(df) == 0:
                progress['kline_done'].append(key)
                save_progress(progress)
                time.sleep(61)
                continue
            
            new = 0
            for _, row in df.iterrows():
                trade_time = str(row.get('trade_time', '')).strip()
                if not trade_time: continue
                # 日期格式: 2022-02-28 15:00:00
                if not conn.execute("SELECT 1 FROM kline_30m WHERE symbol=? AND date=?", (sym, trade_time)).fetchone():
                    conn.execute("INSERT OR IGNORE INTO kline_30m VALUES (?,?,?,?,?,?,?)",
                        (sym, trade_time, float(row.get('open',0) or 0), 
                         float(row.get('close',0) or 0), float(row.get('high',0) or 0),
                         float(row.get('low',0) or 0), float(row.get('vol',0) or 0)))
                    new += 1
            
            if new > 0: conn.commit()
            progress['kline_done'].append(key)
            progress['total_new'] += new
            save_progress(progress)
            
            done_pct = len(progress['kline_done']) / total_calls * 100
            if i % 10 == 0 or new > 50:
                print(f"[{done_pct:.1f}%] {sym} {year}: {len(df)}条(新增{new})")
        
        except Exception as e:
            progress['total_fail'] += 1
            save_progress(progress)
            if '频率超限' in str(e):
                print(f"限频等待65秒后重试 {key}...")
                time.sleep(65)
                continue  # 重试这个key
            if progress['total_fail'] <= 20:
                print(f"{key}错误: {e}")
        
        time.sleep(65)  # 限频1次/分钟+缓冲

conn.commit()
f = conn.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM kline_30m").fetchone()
print(f"\n✅ K线数据: {f[0]}条 ({f[1]} ~ {f[2]})")
print(f"总新增: {progress['total_new']}, 失败: {progress['total_fail']}")
conn.close()
os.remove(PROGRESS_FILE)  # 完成后删除进度文件