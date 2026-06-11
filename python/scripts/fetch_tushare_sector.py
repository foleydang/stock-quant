#!/usr/bin/env python3
"""Tushare板块映射拉取 - 用stock_basic的industry字段覆盖BaoStock关键词映射的缺陷"""

import sqlite3
import tushare as ts
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'stock_data.db')
TUSHARE_TOKEN = '7a9014b18909e8cbce5109d7175f7b21ce37354eaff2371db0da2c58'

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

conn = sqlite3.connect(DB_PATH)

print("拉取Tushare stock_basic (含industry字段)...")
try:
    df = pro.stock_basic(exchange='', list_status='L', 
        fields='ts_code,symbol,name,industry,market,list_date')
    print(f"获取: {len(df)} 条, {df.industry.nunique()} 个行业")
except Exception as e:
    print(f"错误: {e}")
    conn.close()
    exit(1)

# 写入stock_sector表（覆盖关键词映射的不精确数据）
count = 0
for _, row in df.iterrows():
    symbol = row['ts_code']  # 600036.SH
    name = row['name']
    industry = row.get('industry', '其他') or '其他'
    
    conn.execute(
        """INSERT OR REPLACE INTO stock_sector 
        (symbol, name, industry, sector_code, updated_at)
        VALUES (?, ?, ?, ?, ?)""",
        (symbol, name, industry, industry,
         datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    )
    count += 1

conn.commit()

# 统计
total = conn.execute("SELECT COUNT(*) FROM stock_sector").fetchone()[0]
others = conn.execute("SELECT COUNT(*) FROM stock_sector WHERE industry='其他' OR industry IS NULL").fetchone()[0]
industries = conn.execute("SELECT COUNT(DISTINCT industry) FROM stock_sector WHERE industry != '其他'").fetchone()[0]
symbols_30m = conn.execute("SELECT COUNT(DISTINCT symbol) FROM kline_30m").fetchone()[0]
matched = conn.execute("""
    SELECT COUNT(DISTINCT k.symbol) FROM kline_30m k 
    JOIN stock_sector s ON k.symbol = s.symbol WHERE s.industry != '其他'
""").fetchone()[0]

print(f"\n写入: {count} 条")
print(f"stock_sector总数: {total}")
print(f"'其他'类: {others} 只 (从212→{others})")
print(f"行业数: {industries}")
print(f"板块映射匹配率: {matched}/{symbols_30m} ({matched/symbols_30m*100:.0f}%)")

# 验证几个之前是"其他"的股票
print("\n之前是'其他'的股票验证:")
for sym in ['601318.SH', '000009.SZ', '000100.SZ']:
    r = conn.execute("SELECT name, industry FROM stock_sector WHERE symbol=?", (sym,)).fetchone()
    print(f"  {sym}: {r}")

conn.close()
print("✅ 完成")