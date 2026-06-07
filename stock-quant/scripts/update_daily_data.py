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

# 也加上 stock_info 里有的热门股（top 50 by volume）
cursor.execute("SELECT symbol FROM stock_info ORDER BY ROWID LIMIT 50")
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
conn.close()
print(f"[{datetime.now()}] 每日数据更新完成")
