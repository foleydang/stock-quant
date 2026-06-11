#!/usr/bin/env python3
"""从 CSV 文件导入数据到 SQLite"""

import os
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'data', 'stock_data.db')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 创建数据库和表
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 股票基本信息表
cursor.execute('''
    CREATE TABLE IF NOT EXISTS stock_info (
        symbol TEXT PRIMARY KEY,
        code TEXT,
        name TEXT,
        market TEXT,
        update_time TEXT
    )
''')

# 30分钟K线数据表
cursor.execute('''
    CREATE TABLE IF NOT EXISTS kline_30m (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        date TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        created_at TEXT,
        UNIQUE(symbol, date)
    )
''')

# 创建索引
cursor.execute('CREATE INDEX IF NOT EXISTS idx_kline_symbol ON kline_30m(symbol)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_kline_date ON kline_30m(date)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_kline_symbol_date ON kline_30m(symbol, date)')

conn.commit()
print(f"数据库创建完成: {DB_PATH}")

# 导入所有 CSV 文件
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('_30m.csv')]
print(f"找到 {len(csv_files)} 个 CSV 文件")

for csv_file in csv_files:
    symbol = csv_file.replace('_30m.csv', '')
    csv_path = os.path.join(DATA_DIR, csv_file)
    
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        
        # 插入 K 线数据
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT OR IGNORE INTO kline_30m (symbol, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, row['date'], row['open'], row['high'], row['low'], row['close'], row['volume']))
        
        # 插入股票信息
        market = 'SH' if symbol.endswith('.SH') else 'SZ' if symbol.endswith('.SZ') else 'HK'
        code = symbol.split('.')[0]
        cursor.execute('''
            INSERT OR IGNORE INTO stock_info (symbol, code, market, update_time)
            VALUES (?, ?, ?, datetime('now'))
        ''', (symbol, code, market))
        
        print(f"导入 {symbol}: {len(df)} 条记录")
    except Exception as e:
        print(f"导入 {csv_file} 失败: {e}")

conn.commit()
conn.close()

# 检查结果
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM stock_info')
stock_count = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM kline_30m')
kline_count = cursor.fetchone()[0]
conn.close()

print(f"\n导入完成: {stock_count} 只股票, {kline_count} 条K线数据")
