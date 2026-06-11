#!/usr/bin/env python3
"""新浪财经批量拉取30分钟K线 - 不限频，每只5001条

覆盖范围: 2023-11 ~ 2026-06 (约2.5年)
372只股票，几分钟完成

用法: python3 fetch_sina_kline.py
"""
import requests, sqlite3, json, time, os, sys
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/stock_data.db')
conn = sqlite3.connect(DB_PATH)

# 获取所有股票代码
symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM kline_30m ORDER BY symbol").fetchall()]
print(f"股票数: {len(symbols)}只")

# 新浪代码转换: 000001.SZ -> sz000001, 600519.SH -> sh600519
def to_sina_code(ts_code):
    if ts_code.endswith('.SZ'):
        return f"sz{ts_code[:6]}"
    elif ts_code.endswith('.SH'):
        return f"sh{ts_code[:6]}"
    return None

total_new = 0
total_fail = 0

for i, sym in enumerate(symbols):
    sina_code = to_sina_code(sym)
    if not sina_code: continue
    
    try:
        url = f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={sina_code}&scale=30&ma=no&datalen=5001"
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            total_fail += 1
            continue
        
        data = r.json()
        if not data:
            total_fail += 1
            continue
        
        new = 0
        for row in data:
            dt = row.get('day', '')
            if not dt: continue
            # 新浪时间格式: 2024-01-02 10:00:00 -> 直接用
            if not conn.execute("SELECT 1 FROM kline_30m WHERE symbol=? AND date=?", (sym, dt)).fetchone():
                conn.execute("INSERT OR IGNORE INTO kline_30m (symbol,date,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?)",
                    (sym, dt, float(row.get('open',0)), float(row.get('high',0)),
                     float(row.get('low',0)), float(row.get('close',0)),
                     int(row.get('volume',0))))
                new += 1
        
        total_new += new
        if new > 0: conn.commit()
        
        if (i+1) % 50 == 0 or new > 100:
            print(f"[{i+1}/{len(symbols)}] {sym}: {len(data)}条(新增{new}) 累计新增{total_new}")
        
        time.sleep(0.5)  # 防止被封，每只间隔0.5秒
    
    except Exception as e:
        total_fail += 1
        if total_fail <= 10:
            print(f"{sym}错误: {e}")

conn.commit()
f = conn.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM kline_30m").fetchone()
print(f"\n✅ K线数据(新浪): {f[0]}条 ({f[1]} ~ {f[2]})")
print(f"新增: {total_new}, 失败: {total_fail}")
conn.close()