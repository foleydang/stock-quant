#!/usr/bin/env python3
"""
导入沪深300日线数据作为市场背景特征
数据来源: DataHandler (tushare)
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

DB_PATH = os.path.join(BASE_DIR, 'data/stock_data.db')
DAILY_TABLE = 'kline_daily'


def create_daily_table(conn):
    """创建日线数据表"""
    cursor = conn.cursor()
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {DAILY_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            UNIQUE(symbol, date)
        )
    ''')
    conn.commit()
    print(f'✓ 表 {DAILY_TABLE} 已创建')


def fetch_and_store_daily(conn, symbols=None):
    """获取日线数据并存入数据库 (用tushare)"""
    from data.data_handler import DataHandler
    handler = DataHandler()

    if symbols is None:
        # 用已有个股的30分钟数据聚合出日线，而非依赖tushare
        # 先尝试tushare指数，失败就用聚合方法
        symbols = ['000300.SH']  # 沪深300指数

    cursor = conn.cursor()
    total_new = 0

    for i, symbol in enumerate(symbols):
        print(f'[{i+1}/{len(symbols)}] 获取 {symbol} 日线数据...')
        try:
            df = handler._fetch_from_tushare(symbol, days=500)
            if df is None or len(df) == 0:
                print(f'  ⚠ {symbol}: 无数据(可能tushare未配置或指数不支持)')
                continue

            # 获取已有最新日期
            cursor.execute(f'SELECT MAX(date) FROM {DAILY_TABLE} WHERE symbol=?', (symbol,))
            latest = cursor.fetchone()[0]

            new_count = 0
            for _, row in df.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
                if latest is None or date_str > latest:
                    try:
                        cursor.execute(f'''
                            INSERT OR IGNORE INTO {DAILY_TABLE} (symbol, date, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (symbol, date_str, float(row['open']), float(row['high']),
                              float(row['low']), float(row['close']), float(row['volume'])))
                        new_count += 1
                    except Exception:
                        pass

            total_new += new_count
            print(f'  ✓ {symbol}: +{new_count}条新数据 (总共{len(df)}条)')

        except Exception as e:
            print(f'  ✗ {symbol}: {e}')

    conn.commit()
    print(f'\n=== 日线数据更新完成: 新增 {total_new} 条 ===')


def add_market_features_to_30m(conn, symbol_30m: str, df_30m: pd.DataFrame) -> pd.DataFrame:
    """
    为30分钟K线添加日线市场背景特征
    包括: 指数日收益率、市场趋势信号
    """
    query = f'SELECT date, close FROM {DAILY_TABLE} WHERE symbol="000300.SH" ORDER BY date'
    df_daily = pd.read_sql_query(query, conn)

    if df_daily.empty:
        return df_30m

    df_daily['date'] = pd.to_datetime(df_daily['date'], format='mixed')
    df_daily['daily_return'] = df_daily['close'].pct_change()
    df_daily['daily_date'] = df_daily['date'].dt.date

    # 创建日期->日收益率的映射
    daily_return_map = dict(zip(df_daily['daily_date'], df_daily['daily_return']))

    # 添加市场背景特征
    df_30m_dates = pd.to_datetime(df_30m['date'], format='mixed').dt.date
    df_30m['market_daily_return'] = df_30m_dates.map(daily_return_map).fillna(0)
    df_30m['market_daily_return_lag1'] = df_30m['market_daily_return'].shift(8)
    df_30m['market_daily_return_lag2'] = df_30m['market_daily_return'].shift(16)
    df_30m['market_ma5_return'] = df_30m['market_daily_return'].rolling(40).mean()
    df_30m['market_trend'] = (df_30m['market_daily_return'] > 0).astype(int)

    return df_30m


if __name__ == '__main__':
    conn = sqlite3.connect(DB_PATH)
    create_daily_table(conn)
    fetch_and_store_daily(conn)
    conn.close()
    print('✅ 日线数据导入完成')