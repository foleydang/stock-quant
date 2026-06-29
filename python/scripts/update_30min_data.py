#!/usr/bin/env python3
"""
更新30分钟K线数据（从新浪API拉取，用于qlib模型）
"""
import sys, os, time, sqlite3, requests, json
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import get_db_path

DB_PATH = get_db_path()
print(f"DB: {DB_PATH}")

conn = sqlite3.connect(DB_PATH)

# 获取所有symbol
symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM kline_30m").fetchall()]
print(f"共 {len(symbols)} 只股票需要更新30min数据")

total_new = 0
for i, sym in enumerate(symbols):
    code = sym[:6]
    if sym.endswith('.SZ'):
        sina_code = f'sz{code}'
    elif sym.endswith('.SH'):
        sina_code = f'sh{code}'
    else:
        continue  # skip HK

    try:
        url = "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"
        params = {"symbol": sina_code, "scale": "30", "datalen": 20}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            continue

        data = json.loads(r.text)
        if not isinstance(data, list) or len(data) == 0:
            continue

        new = 0
        for row in data:
            trade_time = row.get('day', '')
            # 归一化格式: 确保为 %Y-%m-%d %H:%M:%S
            if trade_time and len(trade_time) == 16:
                trade_time += ':00'
            if not trade_time:
                continue
            # Check if exists
            if not conn.execute("SELECT 1 FROM kline_30m WHERE symbol=? AND date=?", (sym, trade_time)).fetchone():
                conn.execute(
                    "INSERT OR IGNORE INTO kline_30m VALUES (?,?,?,?,?,?,?)",
                    (sym, trade_time,
                     float(row.get('open', 0) or 0),
                     float(row.get('close', 0) or 0),
                     float(row.get('high', 0) or 0),
                     float(row.get('low', 0) or 0),
                     float(row.get('volume', 0) or 0))
                )
                new += 1

        conn.commit()
        total_new += new
        if new > 0:
            print(f"  [{i+1}/{len(symbols)}] {sym}: +{new}条")

    except Exception as e:
        if i < 5:
            print(f"  [{i+1}] {sym} 错误: {e}")

    time.sleep(0.1)  # 避免请求过快

conn.close()
print(f"\n✅ 完成: 新增 {total_new} 条30min数据")

# 更新qlib
if total_new > 0:
    print("更新qlib数据...")
    os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'qlib_pipeline'))
    os.system(f"{sys.executable} convert_data.py --db {DB_PATH} --freq 30min")