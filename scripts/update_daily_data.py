#!/usr/bin/env python3
"""每日数据更新脚本 - 由 cron 在每晚20:00执行

1. 从 Tushare 更新持仓股+自选股的日K线到 DB
2. 更新 stock_info 表
3. 更新 positions 表的最新市值（从腾讯API获取实时收盘价）
"""

import os
import sys
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(PROJECT_ROOT, 'python')
DATA_DIR = os.path.join(PYTHON_DIR, 'data')
DB_PATH = os.path.join(DATA_DIR, 'stock_data.db')
AGENT_DIR = os.path.join(PROJECT_ROOT, 'agent')

sys.path.insert(0, PYTHON_DIR)
sys.path.insert(0, AGENT_DIR)

from config import TUSHARE_TOKEN, WATCHLIST

import tushare as ts
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 1. 更新持仓股 + 自选股的日K线
symbols_to_update = []

# 从 positions 表获取持仓股
cursor.execute("SELECT symbol FROM positions")
for row in cursor.fetchall():
    symbols_to_update.append(row[0])

# 从 config.yaml 获取自选股
for w in WATCHLIST:
    symbols_to_update.append(w.get('symbol'))

# 也加上 stock_info 里的所有股票
cursor.execute("SELECT symbol FROM stock_info")
for row in cursor.fetchall():
    if row[0] not in symbols_to_update:
        symbols_to_update.append(row[0])

symbols_to_update = list(set(symbols_to_update))
print(f"[{datetime.now()}] 需更新 {len(symbols_to_update)} 只股票的K线数据")

# 使用批量API按日期拉取（比逐只股票拉高效得多）
# 优先拉最近2-3个交易日
today = datetime.now()
end_date = today.strftime('%Y%m%d')
import time as _time

for offset in range(1, 4):
    trade_date = (today - timedelta(days=offset)).strftime('%Y%m%d')
    try:
        df = pro.daily(trade_date=trade_date)
        if df is None or df.empty:
            continue
        
        df_t = df.rename(columns={'ts_code': 'symbol', 'trade_date': 'date', 'vol': 'volume'})
        df_t = df_t[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']]
        df_t['date'] = df_t['date'].apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}")
        
        # 只保留symbols_to_update中的股票（跳过港股）
        target_symbols = set(s for s in symbols_to_update if '.HK' not in s)
        df_filtered = df_t[df_t['symbol'].isin(target_symbols)]
        
        if not df_filtered.empty:
            # 增量插入（忽略已存在的）
            existing = set(cursor.execute("SELECT symbol, date FROM kline_daily WHERE date=?", (df_filtered['date'].iloc[0],)).fetchall())
            df_new = df_filtered[~df_filtered.apply(lambda r: (r['symbol'], r['date']) in existing, axis=1)]
            
            if not df_new.empty:
                df_new.to_sql('kline_daily', conn, if_exists='append', index=False, method='multi')
                conn.commit()
                print(f"  ✓ 批量 {trade_date}: +{len(df_new)} 条")
        
        _time.sleep(65)  # Tushare限频：1次/分钟
        
    except Exception as e:
        print(f"  ✗ 批量 {trade_date}: {str(e)[:80]}")
        _time.sleep(65)

conn.commit()

# 2. 更新 stock_info（每月更新即可，这里做增量）
try:
    # 获取最新股票列表
    df_stock = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry')
    if df_stock is not None and not df_stock.empty:
        for _, r in df_stock.iterrows():
            cursor.execute(
                "INSERT OR REPLACE INTO stock_info (symbol, code, name, market, updated_at) VALUES (?, ?, ?, ?, ?)",
                (r['ts_code'], r['symbol'], r['name'], r['ts_code'].split('.')[-1], datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            )
        conn.commit()
        print(f"✓ stock_info 已更新: {len(df_stock)} 条")
except Exception as e:
    print(f"✗ stock_info 更新失败: {e}")

# 3. 更新 positions 表的 current_price（用腾讯API收盘价）
try:
    from data.data_handler import DataHandler
    dh = DataHandler(force_refresh=True)
    
    cursor.execute("SELECT symbol FROM positions")
    position_symbols = [r[0] for r in cursor.fetchall()]
    
    realtime = dh.get_realtime_prices(position_symbols) if position_symbols else {}
    
    for symbol, price_info in realtime.items():
        if price_info and 'price' in price_info:
            cursor.execute("UPDATE positions SET current_price=? WHERE symbol=?",
                          (price_info['price'], symbol))
    conn.commit()
    print(f"✓ positions 实时价已更新")
except Exception as e:
    print(f"✗ positions 更新失败: {e}")

# 关闭
conn.commit()

# 4. 更新北向资金（从东方财富API）
print(f"\n更新北向资金数据...")
try:
    import requests
    url = "https://push2his.eastmoney.com/api/qt/kamt.kline/get"
    params = {
        "fields1": "f1,f3,f5",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63",
        "klt": "101", "lmt": "30",
        "ut": "7eea3ed2ced24c2974d3210a0be1e25",
    }
    r = requests.get(url, params=params, timeout=15)
    data = r.json().get("data", {})
    hk2sh = data.get("hk2sh", [])
    hk2sz = data.get("hk2sz", [])
    
    north_count = 0
    for line in hk2sh[-5:]:
        parts = line.split(",")
        if len(parts) >= 4:
            date, net, buy, cum = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
            # 找同日深股通
            sz_line = [l for l in hk2sz if l.startswith(date)]
            if sz_line:
                sz_parts = sz_line[0].split(",")
                sz_net, sz_buy, sz_cum = float(sz_parts[1]), float(sz_parts[2]), float(sz_parts[3])
                total_net = net + sz_net
                total_buy = buy + sz_buy
                cursor.execute("INSERT OR REPLACE INTO north_flow (trade_date, north_net, north_buy, north_cum, sz_net, sz_buy, sz_cum, total_net, total_buy, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (date, net, buy, cum, sz_net, sz_buy, sz_cum, total_net, total_buy, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                north_count += 1
    conn.commit()
    print(f"✓ 北向资金更新: {north_count} 条")
except Exception as e:
    print(f"✗ 北向资金更新失败: {e}")

# 5. 更新大盘日线指标（从kline_30m聚合）
print(f"\n更新大盘指标...")
try:
    df_30m = pd.read_sql("SELECT symbol, date, close, volume FROM kline_30m WHERE date >= ?", conn, params=[(today - timedelta(days=3)).strftime('%Y-%m-%d')])
    df_30m['trade_date'] = df_30m['date'].str[:10]
    
    daily = df_30m.groupby(['symbol', 'trade_date']).agg(
        first_close=('close', 'first'), last_close=('close', 'last'), total_volume=('volume', 'sum')
    ).reset_index()
    daily['pct_chg'] = (daily['last_close'] - daily['first_close']) / daily['first_close'] * 100
    
    market = daily.groupby('trade_date').agg(
        avg_pct=('pct_chg', 'mean'), up=('pct_chg', lambda x: int((x > 0).sum())), down=('pct_chg', lambda x: int((x < 0).sum())), stocks=('symbol', 'count'), vol=('total_volume', 'sum')
    ).reset_index()
    
    for _, row in market.iterrows():
        cursor.execute("INSERT OR REPLACE INTO hs300_daily (trade_date, avg_pct_chg, up_count, down_count, stock_count, volume) VALUES (?, ?, ?, ?, ?, ?)",
            (row['trade_date'], row['avg_pct'], row['up'], row['down'], row['stocks'], row['vol']))
    conn.commit()
    print(f"✓ 大盘指标更新: {len(market)} 天")
except Exception as e:
    print(f"✗ 大盘指标更新失败: {e}")

# 关闭
conn.close()

# 6. 更新30分钟K线数据 + 重建qlib bin（每日收盘后执行）
print(f"\n更新30分钟K线数据...")
try:
    import subprocess
    result = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'update_qlib_data.py')],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode == 0:
        for line in result.stdout.strip().split('\n'):
            if '✅' in line:
                print(line)
    else:
        print(f"✗ 30min数据更新失败: {result.stderr[-200:]}")
except Exception as e:
    print(f"✗ 30min数据更新异常: {e}")

print(f"[{datetime.now()}] 每日数据更新完成")
