#!/usr/bin/env python3
"""
CSV数据转存到SQLite数据库
- 将所有30分钟K线CSV导入数据库
- 将日线数据导入数据库
- 保留原CSV文件作为备份
"""

import os
import sys
import sqlite3
import pandas as pd
from datetime import datetime
from glob import glob

# 数据库路径
DB_PATH = os.path.join(os.path.dirname(__file__), '../data/stock_data.db')
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')


def init_database():
    """初始化/扩展数据库表结构"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. 股票基本信息表（已存在）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_info (
            symbol TEXT PRIMARY KEY,
            code TEXT,
            name TEXT,
            market TEXT,
            update_time TEXT
        )
    ''')

    # 2. 30分钟K线表（已存在）
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

    # 3. 日线K线表（新增）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS kline_daily (
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

    # 4. 数据源记录表（新增）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_source (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            source_file TEXT,
            data_type TEXT,
            record_count INTEGER,
            import_time TEXT
        )
    ''')

    # 创建索引
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_kline_30m_symbol ON kline_30m(symbol)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_kline_30m_date ON kline_30m(date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_kline_daily_symbol ON kline_daily(symbol)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_kline_daily_date ON kline_daily(date)')

    conn.commit()
    conn.close()
    print("数据库表结构初始化完成")


def import_30m_csv():
    """导入所有30分钟K线CSV文件"""
    csv_files = glob(os.path.join(DATA_DIR, '*_30m.csv'))
    print(f"\n发现 {len(csv_files)} 个30分钟K线文件")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    total_imported = 0
    total_skipped = 0

    for csv_file in csv_files:
        # 从文件名提取股票代码
        filename = os.path.basename(csv_file)
        symbol = filename.replace('_30m.csv', '')

        try:
            df = pd.read_csv(csv_file)

            if df.empty:
                continue

            # 确保列名正确
            if 'day' in df.columns:
                df = df.rename(columns={'day': 'date'})

            # 标准化日期格式
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M:%S')

            records = []
            for _, row in df.iterrows():
                records.append((
                    symbol,
                    row['date'],
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    int(row['volume']),
                    now
                ))

            # 批量插入
            cursor.executemany('''
                INSERT OR IGNORE INTO kline_30m
                (symbol, date, open, high, low, close, volume, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', records)

            imported = cursor.rowcount
            skipped = len(records) - imported

            total_imported += imported
            total_skipped += skipped

            # 记录数据源
            cursor.execute('''
                INSERT INTO data_source (symbol, source_file, data_type, record_count, import_time)
                VALUES (?, ?, ?, ?, ?)
            ''', (symbol, filename, '30m', imported, now))

            print(f"  {symbol}: 导入 {imported} 条, 跳过 {skipped} 条(已存在)")

        except Exception as e:
            print(f"  {symbol}: 导入失败 - {e}")

    conn.commit()
    conn.close()

    print(f"\n30分钟数据导入完成: 新增 {total_imported} 条, 跳过 {total_skipped} 条")
    return total_imported


def import_daily_csv():
    """导入日线数据（processed.csv）"""
    csv_files = glob(os.path.join(DATA_DIR, '*_processed.csv'))
    print(f"\n发现 {len(csv_files)} 个日线数据文件")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    total_imported = 0

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        symbol = filename.replace('_processed.csv', '')

        try:
            df = pd.read_csv(csv_file)

            if df.empty:
                continue

            # 标准化日期格式
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

            records = []
            for _, row in df.iterrows():
                records.append((
                    symbol,
                    row['date'],
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    int(row['volume']),
                    now
                ))

            cursor.executemany('''
                INSERT OR IGNORE INTO kline_daily
                (symbol, date, open, high, low, close, volume, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', records)

            imported = cursor.rowcount
            total_imported += imported

            # 记录数据源
            cursor.execute('''
                INSERT INTO data_source (symbol, source_file, data_type, record_count, import_time)
                VALUES (?, ?, ?, ?, ?)
            ''', (symbol, filename, 'daily', imported, now))

            if imported > 0:
                print(f"  {symbol}: 导入 {imported} 条日线数据")

        except Exception as e:
            print(f"  {symbol}: 导入失败 - {e}")

    conn.commit()
    conn.close()

    print(f"\n日线数据导入完成: 共 {total_imported} 条")
    return total_imported


def update_stock_info():
    """更新股票信息表"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    # 从已有数据中提取股票信息
    cursor.execute('''
        INSERT OR IGNORE INTO stock_info (symbol, code, name, market, update_time)
        SELECT DISTINCT
            symbol,
            SUBSTR(symbol, 1, 6) as code,
            '' as name,
            CASE
                WHEN symbol LIKE '%.SH' THEN 'sh'
                WHEN symbol LIKE '%.SZ' THEN 'sz'
                WHEN symbol LIKE '%.HK' THEN 'hk'
                ELSE 'unknown'
            END as market,
            ? as update_time
        FROM kline_30m
    ''', (now,))

    cursor.execute('''
        INSERT OR IGNORE INTO stock_info (symbol, code, name, market, update_time)
        SELECT DISTINCT
            symbol,
            SUBSTR(symbol, 1, 6) as code,
            '' as name,
            CASE
                WHEN symbol LIKE '%.SH' THEN 'sh'
                WHEN symbol LIKE '%.SZ' THEN 'sz'
                WHEN symbol LIKE '%.HK' THEN 'hk'
                ELSE 'unknown'
            END as market,
            ? as update_time
        FROM kline_daily
    ''', (now,))

    cursor.execute('SELECT COUNT(*) FROM stock_info')
    stock_count = cursor.fetchone()[0]

    conn.commit()
    conn.close()

    print(f"\n股票信息表更新完成: 共 {stock_count} 只股票")


def show_final_stats():
    """显示最终统计"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("\n" + "=" * 60)
    print("数据库统计")
    print("=" * 60)

    # 股票数量
    cursor.execute('SELECT COUNT(*) FROM stock_info')
    print(f"股票数量: {cursor.fetchone()[0]}")

    # 30分钟数据
    cursor.execute('SELECT COUNT(*) FROM kline_30m')
    kline_30m_count = cursor.fetchone()[0]
    print(f"30分钟K线: {kline_30m_count:,} 条")

    cursor.execute('SELECT COUNT(DISTINCT symbol) FROM kline_30m')
    print(f"  - 覆盖股票: {cursor.fetchone()[0]} 只")

    cursor.execute('SELECT MIN(date), MAX(date) FROM kline_30m')
    min_date, max_date = cursor.fetchone()
    print(f"  - 时间范围: {min_date} ~ {max_date}")

    # 日线数据
    cursor.execute('SELECT COUNT(*) FROM kline_daily')
    daily_count = cursor.fetchone()[0]
    if daily_count > 0:
        print(f"\n日线K线: {daily_count:,} 条")
        cursor.execute('SELECT COUNT(DISTINCT symbol) FROM kline_daily')
        print(f"  - 覆盖股票: {cursor.fetchone()[0]} 只")

    # 数据源记录
    cursor.execute('SELECT COUNT(*) FROM data_source')
    print(f"\n数据源记录: {cursor.fetchone()[0]} 条")

    conn.close()
    print("=" * 60)


def main():
    print("=" * 60)
    print("CSV数据转存到SQLite数据库")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 初始化数据库
    init_database()

    # 2. 导入30分钟数据
    import_30m_csv()

    # 3. 导入日线数据
    import_daily_csv()

    # 4. 更新股票信息
    update_stock_info()

    # 5. 显示统计
    show_final_stats()

    print(f"\n数据库位置: {DB_PATH}")
    print("\nNavicat 连接信息:")
    print("  连接名: stock_quant")
    print("  类型: SQLite")
    print(f"  数据库文件: {DB_PATH}")
    print("\n注意: CSV文件已保留作为备份，确认数据无误后可手动删除")


if __name__ == "__main__":
    main()