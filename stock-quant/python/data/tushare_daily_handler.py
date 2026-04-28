"""
Tushare 日线数据处理器 - 免费、不限频、服务器可用
适合：日线级别监控、回测
"""

import tushare as ts
import pandas as pd
import os
from datetime import datetime, timedelta

# 你的 Tushare Token
TS_TOKEN = '7a9014b18909e8cbce5109d7175f7b21ce37354eaff2371db0da2c58'
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

DATA_DIR = os.path.dirname(__file__)

def fetch_daily_data(symbol: str, days: int = 60):
    """
    获取日线数据 - 免费、不限频
    
    Args:
        symbol: 如 '600036.SH', '000001.SZ'
        days: 天数
    
    Returns:
        DataFrame
    """
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
    
    try:
        df = pro.daily(ts_code=symbol, start_date=start_date, end_date=end_date)
        
        if df is not None and not df.empty:
            # 转换格式
            df = df.rename(columns={
                'trade_date': 'date',
                'vol': 'volume'
            })
            df['date'] = pd.to_datetime(df['date'])
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            df = df.sort_values('date').reset_index(drop=True)
            
            # 保存缓存
            cache_path = os.path.join(DATA_DIR, f'{symbol}_daily.csv')
            df.to_csv(cache_path, index=False)
            
            return df
    except Exception as e:
        print(f'Tushare 日线获取失败 {symbol}: {e}')
    
    return None


def batch_update_daily(symbols: list):
    """批量更新日线数据"""
    results = {}
    for symbol in symbols:
        df = fetch_daily_data(symbol, days=90)
        if df is not None:
            results[symbol] = df
            print(f'{symbol}: ✅ {len(df)} 条')
        else:
            print(f'{symbol}: ❌ 失败')
    return results


if __name__ == '__main__':
    # 测试
    test_symbols = ['600036.SH', '000001.SZ', '300750.SZ']
    batch_update_daily(test_symbols)
