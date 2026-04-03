#!/usr/bin/env python3
"""
沪深300数据增量更新脚本
保留历史数据，只插入新数据
"""

import os
import sys
import sqlite3
import time
import random
from datetime import datetime, timedelta

BASE_DIR = '/Users/foleydang/github/stock-quant/stock-quant/python'
sys.path.insert(0, BASE_DIR)

DB_PATH = f'{BASE_DIR}/data/stock_data.db'

from data.data_handler import DataHandler


def get_latest_date(conn, symbol):
    """获取数据库中该股票最新日期"""
    cursor = conn.cursor()
    cursor.execute('SELECT MAX(date) FROM kline_30m WHERE symbol = ?', (symbol,))
    result = cursor.fetchone()
    return result[0] if result and result[0] else None


def insert_new_data(conn, symbol, df, latest_date):
    """只插入新数据，保留历史数据"""
    if df is None or len(df) == 0:
        return 0

    cursor = conn.cursor()
    new_count = 0

    for _, row in df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(row['date'], 'strftime') else str(row['date'])

        # 只插入比最新日期更新的数据
        if latest_date is None or date_str > latest_date:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO kline_30m (symbol, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, date_str, float(row['open']), float(row['high']),
                      float(row['low']), float(row['close']), float(row['volume'])))
                new_count += 1
            except:
                pass

    conn.commit()
    return new_count


def update_hs300():
    """增量更新沪深300数据"""
    print("=" * 70)
    print(f"沪深300增量更新 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    handler = DataHandler(force_refresh=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 获取股票列表（只处理A股）
    cursor.execute('SELECT symbol, name FROM stock_info WHERE symbol NOT LIKE "%.HK"')
    stocks = cursor.fetchall()

    print(f"\n共 {len(stocks)} 只A股待更新\n")

    total_new = 0
    success = 0
    skipped = 0

    for i, (symbol, name) in enumerate(stocks, 1):
        # 获取数据库最新日期
        latest_date = get_latest_date(conn, symbol)

        # 获取数据
        df = handler.fetch_stock_data(symbol, force_refresh=True)

        if df is not None and len(df) > 0:
            new_count = insert_new_data(conn, symbol, df, latest_date)

            if new_count > 0:
                print(f"[{i}/{len(stocks)}] {name}({symbol}): +{new_count}条新数据")
                total_new += new_count
            else:
                print(f"[{i}/{len(stocks)}] {name}({symbol}): 已是最新")

            success += 1
        else:
            print(f"[{i}/{len(stocks)}] {name}({symbol}): 获取失败，保留历史数据")
            skipped += 1

        # 短暂延时
        time.sleep(random.uniform(0.2, 0.5))

        # 每50只显示进度
        if i % 50 == 0:
            print(f"  --- 已处理 {i}/{len(stocks)}, 新增:{total_new}条 ---")

    conn.close()

    print(f"\n{'='*70}")
    print(f"更新完成: 成功 {success}, 跳过 {skipped}, 新增数据 {total_new} 条")
    print(f"历史数据已保留")
    print(f"{'='*70}")

    return success, total_new


def update_positions():
    """更新持仓股票数据"""
    print("=" * 70)
    print(f"持仓股票更新 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    handler = DataHandler(force_refresh=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 获取持仓股票
    cursor.execute('SELECT symbol, stock_name FROM positions')
    positions = cursor.fetchall()

    for symbol, name in positions:
        latest_date = get_latest_date(conn, symbol)
        df = handler.fetch_stock_data(symbol, force_refresh=True)

        if df is not None and len(df) > 0:
            new_count = insert_new_data(conn, symbol, df, latest_date)
            print(f"  {name}({symbol}): +{new_count}条新数据" if new_count > 0 else f"  {name}({symbol}): 已是最新")
        else:
            print(f"  {name}({symbol}): 获取失败，保留历史数据")

        time.sleep(1)

    conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='增量更新股票数据')
    parser.add_argument('--positions', action='store_true', help='只更新持仓股票')
    args = parser.parse_args()

    if args.positions:
        update_positions()
    else:
        update_hs300()