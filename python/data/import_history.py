import os
"""导入足够的历史数据用于特征计算"""

import tushare as ts
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import time

DB_PATH = 'data/stock_data.db'

ts.set_token(os.getenv('TUSHARE_TOKEN', ''))
pro = ts.pro_api()

def import_history(symbol, days=365):
    """导入历史日线数据"""
    start = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
    end = datetime.now().strftime('%Y%m%d')
    
    print(f"{symbol}: 获取 {days} 天历史...")
    
    # Tushare 可能有限制，分批获取
    df = pro.daily(ts_code=symbol, start_date=start, end_date=end)
    
    if df is not None and not df.empty:
        df = df.rename(columns={'trade_date': 'date', 'vol': 'volume'})
        df['date'] = pd.to_datetime(df['date'], format='mixed')
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        df = df.sort_values('date').reset_index(drop=True)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 删除旧数据
        cursor.execute('DELETE FROM kline_30m WHERE symbol=?', (symbol,))
        
        # 写入新数据
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT INTO kline_30m (symbol, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, row['date'].strftime('%Y-%m-%d %H:%M:%S'),
                  float(row['open']), float(row['high']), float(row['low']),
                  float(row['close']), float(row['volume'])))
        
        conn.commit()
        conn.close()
        
        print(f"  ✓ 导入 {len(df)} 条")
        return len(df)
    
    print(f"  ✗ 获取失败")
    return 0

def main():
    # 持仓列表
    positions = ['300124.SZ', '600048.SH', '300015.SZ', '3690.HK', '159792.SZ']
    
    # A股：导入365天历史
    for symbol in positions:
        if symbol.endswith('.SZ') or symbol.endswith('.SH'):
            import_history(symbol, days=365)
            time.sleep(0.5)  # 速率限制
    
    # 检查数据量
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("\n=== 最终数据量 ===")
    for symbol in positions:
        cursor.execute('SELECT COUNT(*) FROM kline_30m WHERE symbol=?', (symbol,))
        cnt = cursor.fetchone()[0]
        print(f"{symbol}: {cnt} 条")
    
    conn.close()

if __name__ == "__main__":
    main()
