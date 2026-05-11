"""分钟K线累积器 - 用实时行情构建自己的分钟K线数据库"""

import os
import sys
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import time

BASE_DIR = '/root/github/stock-quant/stock-quant/python'
DB_PATH = f'{BASE_DIR}/data/stock_data.db'

sys.path.insert(0, BASE_DIR)
from data.data_handler import DataHandler


class KlineAccumulator:
    """累积实时行情构建分钟K线"""
    
    def __init__(self):
        self.db_path = DB_PATH
        self.data_handler = DataHandler()
    
    def accumulate_realtime(self, symbols):
        """
        获取实时价格并写入分钟K线表
        
        每30分钟调用一次，记录当前时间的K线数据
        """
        # 获取实时价格
        prices = self.data_handler.get_realtime_prices(symbols)
        
        if not prices:
            print("获取实时价格失败")
            return 0
        
        now = datetime.now()
        # 30分钟K线时间戳（向下取整到30分钟）
        kline_time = now.replace(minute=now.minute // 30 * 30, second=0, microsecond=0)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        count = 0
        for symbol, data in prices.items():
            # 构建K线数据
            # 注意：实时数据只有当前价，open/high/low 需要从历史或近似
            open_price = data.get('open', data['price'])
            high_price = max(open_price, data['price'])
            low_price = min(open_price, data['price'])
            close_price = data['price']
            volume = data.get('volume', 0)
            
            # 检查是否已有该时间点的数据
            cursor.execute('''
                SELECT id FROM kline_30m WHERE symbol=? AND date=?
            ''', (symbol, kline_time.strftime('%Y-%m-%d %H:%M:%S')))
            
            existing = cursor.fetchone()
            
            if existing:
                # 更现有数据（更新 high/low/close/volume）
                cursor.execute('''
                    UPDATE kline_30m SET 
                        high=?, low=?, close=?, volume=?,
                        updated_at=?
                    WHERE symbol=? AND date=?
                ''', (high_price, low_price, close_price, volume,
                      now.isoformat(), symbol, kline_time.strftime('%Y-%m-%d %H:%M:%S')))
            else:
                # 插入新数据
                cursor.execute('''
                    INSERT INTO kline_30m (symbol, date, open, high, low, close, volume, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, kline_time.strftime('%Y-%m-%d %H:%M:%S'),
                      open_price, high_price, low_price, close_price, volume,
                      now.isoformat()))
            
            count += 1
        
        conn.commit()
        conn.close()
        
        print(f"✓ 累积 {count}/{len(symbols)} 条K线数据 @ {kline_time.strftime('%H:%M')}")
        return count
    
    def get_kline_stats(self, symbol):
        """获取K线统计"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*), MIN(date), MAX(date) FROM kline_30m WHERE symbol=?
        ''', (symbol,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0]:
            return {'count': row[0], 'min_date': row[1], 'max_date': row[2]}
        return None


def main():
    """主函数 - 用于定时累积"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='+', help='股票代码列表')
    args = parser.parse_args()
    
    acc = KlineAccumulator()
    
    # 默认持仓列表
    if not args.symbols:
        positions = ['300124.SZ', '600048.SH', '3690.HK', '300015.SZ', '159792.SZ',
                     '9988.HK']  # 加上关注的阿里巴巴
    else:
        positions = args.symbols
    
    print(f"累积K线: {positions}")
    count = acc.accumulate_realtime(positions)
    
    # 显示统计
    for sym in positions[:3]:
        stats = acc.get_kline_stats(sym)
        if stats:
            print(f"  {sym}: {stats['count']}条, 最新 {stats['max_date']}")


if __name__ == "__main__":
    main()
