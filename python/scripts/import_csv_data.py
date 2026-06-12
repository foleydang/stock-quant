#!/usr/bin/env python3
"""
CSV数据导入 - 读取Mac传来的3个CSV文件，写入DB

运行方式:
  python3 import_csv_data.py
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
DB_PATH = os.path.join(DATA_DIR, 'stock_data.db')

conn = sqlite3.connect(DB_PATH)

# ========================================
# 1. 导入 north_flow.csv
# ========================================
print("=" * 50)
print("1. 导入北向资金")
print("=" * 50)

csv_path = f'{DATA_DIR}/north_flow.csv'
try:
    df = pd.read_csv(csv_path)
    print(f"  CSV: {len(df)} 条")
    print(f"  范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    
    count = 0
    for _, row in df.iterrows():
        conn.execute(
            """INSERT OR REPLACE INTO north_flow 
            (trade_date, north_net, north_buy, north_cum, sz_net, sz_buy, sz_cum,
             total_net, total_buy, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (row['trade_date'], 
             row.get('north_net'), row.get('north_buy'), row.get('north_cum'),
             row.get('sz_net'), row.get('sz_buy'), row.get('sz_cum'),
             row.get('total_net'), row.get('total_buy'),
             datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        )
        count += 1
    
    conn.commit()
    print(f"  ✅ 导入: {count} 条")
    
    # 验证
    valid = conn.execute("SELECT COUNT(*) FROM north_flow WHERE total_net IS NOT NULL AND total_net != 0").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM north_flow").fetchone()[0]
    print(f"  有效: {valid}/{total}")
    
except FileNotFoundError:
    print(f"  ❌ 文件不存在: {csv_path}")
except Exception as e:
    print(f"  ❌ 错误: {e}")

# ========================================
# 2. 导入 stock_sector.csv
# ========================================
print("\n" + "=" * 50)
print("2. 导入行业映射")
print("=" * 50)

csv_path = f'{DATA_DIR}/stock_sector.csv'
try:
    df = pd.read_csv(csv_path)
    print(f"  CSV: {len(df)} 条, {df['industry'].nunique()} 个行业")
    
    count = 0
    for _, row in df.iterrows():
        industry = row.get('industry', '其他') or '其他'
        conn.execute(
            """INSERT OR REPLACE INTO stock_sector 
            (symbol, name, industry, sector_code, updated_at)
            VALUES (?, ?, ?, ?, ?)""",
            (row['symbol'], row['name'], industry, industry,
             datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        )
        count += 1
    
    conn.commit()
    print(f"  ✅ 导入: {count} 条")
    
    # 统计
    others = conn.execute("SELECT COUNT(*) FROM stock_sector WHERE industry='其他' OR industry IS NULL").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM stock_sector").fetchone()[0]
    symbols_30m = conn.execute("SELECT COUNT(DISTINCT symbol) FROM kline_30m").fetchone()[0]
    matched = conn.execute("""
        SELECT COUNT(DISTINCT k.symbol) FROM kline_30m k 
        JOIN stock_sector s ON k.symbol = s.symbol WHERE s.industry != '其他'
    """).fetchone()[0]
    
    print(f"  '其他': {others} 只")
    print(f"  板块匹配率: {matched}/{symbols_30m} ({matched/symbols_30m*100:.0f}%)")
    
    # 验证关键股票
    for sym in ['600036.SH', '601318.SH', '000001.SZ']:
        r = conn.execute("SELECT name, industry FROM stock_sector WHERE symbol=?", (sym,)).fetchone()
        print(f"    {sym}: {r}")
    
except FileNotFoundError:
    print(f"  ❌ 文件不存在: {csv_path}")
except Exception as e:
    print(f"  ❌ 错误: {e}")

# ========================================
# 3. 导入 hs300_daily.csv (可选，服务器已有)
# ========================================
print("\n" + "=" * 50)
print("3. 导入沪深300日线（可选）")
print("=" * 50)

csv_path = f'{DATA_DIR}/hs300_daily.csv'
try:
    df = pd.read_csv(csv_path)
    print(f"  CSV: {len(df)} 条")
    
    count = 0
    for _, row in df.iterrows():
        conn.execute(
            """INSERT OR REPLACE INTO hs300_daily 
            (trade_date, open, close, high, low, volume, amount, pct_chg)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (row['trade_date'], row.get('open'), row.get('close'),
             row.get('high'), row.get('low'), row.get('volume'),
             row.get('amount'), row.get('pct_chg'))
        )
        count += 1
    
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM hs300_daily").fetchone()[0]
    print(f"  ✅ 导入: {count} 条, 共: {total} 条")
    
except FileNotFoundError:
    print(f"  ⚠️ 文件不存在（服务器已有BaoStock数据，可跳过）")
except Exception as e:
    print(f"  ❌ 错误: {e}")

# ========================================
# 最终验证
# ========================================
print("\n" + "=" * 50)
print("最终数据状态")
print("=" * 50)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'strategy'))
from strategy.features import MarketFeatureEngineer

# 各表数据量
for table in ['north_flow', 'stock_sector', 'hs300_daily', 'kline_daily']:
    cnt = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  {table}: {cnt} 条")

# 北向资金日期匹配
kline_days = conn.execute("SELECT COUNT(DISTINCT SUBSTR(date,1,10)) FROM kline_30m").fetchone()[0]
north_match = conn.execute("""
    SELECT COUNT(DISTINCT SUBSTR(date,1,10)) FROM kline_30m 
    WHERE SUBSTR(date,1,10) IN (
        SELECT trade_date FROM north_flow WHERE total_net IS NOT NULL AND total_net != 0
    )
""").fetchone()[0]
print(f"  北向资金匹配: {north_match}/{kline_days} ({north_match/kline_days*100:.0f}%)")

# 板块映射匹配率
matched_sector = conn.execute("""
    SELECT COUNT(DISTINCT k.symbol) FROM kline_30m k 
    JOIN stock_sector s ON k.symbol = s.symbol WHERE s.industry != '其他'
""").fetchone()[0]
print(f"  板块映射匹配: {matched_sector}/{kline_days} 只 ({matched_sector/kline_days*100:.0f}%)")

# 市场特征验证
df_test = pd.read_sql(
    "SELECT date, open, close, high, low, volume FROM kline_30m "
    "WHERE symbol='600036.SH' AND date >= '2025-06-01' ORDER BY date",
    conn
)
feats = MarketFeatureEngineer.calculate_market_features(df_test, symbol='600036.SH')
print(f"\n  市场特征有效性:")
for col in feats.columns:
    nonzero = (feats[col] != 0).sum()
    pct = nonzero / len(feats) * 100
    mean = feats[col].mean()
    status = "✅" if nonzero > len(feats) * 0.5 else ("⚠️" if nonzero > 0 else "❌")
    print(f"    {status} {col}: {pct:.0f}% 非零, mean={mean:.4f}")

conn.close()
print("\n✅ 导入完成")