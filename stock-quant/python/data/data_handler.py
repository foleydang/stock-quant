"""
数据处理器 - 使用香港网关获取东方财富数据

解决：国内服务器 IP 被东方财富封锁，通过香港服务器转发
"""

import os
import sys
import pandas as pd
import sqlite3
import requests
from datetime import datetime, timedelta
import time
import random

# ========== 配置 ==========
# 香港网关地址（香港服务器部署后填入）
HK_GATEWAY_URL = 'http://香港服务器IP:5000'  # TODO: 替换为实际 IP

DB_PATH = os.path.join(os.path.dirname(__file__), 'stock_data.db')
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')

# Tushare 备用（免费日线）
TUSHARE_TOKEN = '7a9014b18909e8cbce5109d7175f7b21ce37354eaff2371db0da2c58'
try:
    import tushare as ts
    ts.set_token(TUSHARE_TOKEN)
    TUSHARE_PRO = ts.pro_api()
    TUSHARE_AVAILABLE = True
except:
    TUSHARE_AVAILABLE = False

_last_request_time = 0
_request_interval = 0.3


def _rate_limit():
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _request_interval:
        time.sleep(_request_interval - elapsed)
    _last_request_time = time.time()


class DataHandler:
    def __init__(self, force_refresh=False):
        self.force_refresh = force_refresh
        self.db_path = DB_PATH
        self.data_dir = DATA_DIR
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def fetch_stock_data(self, symbol, days=60, force_refresh=None):
        """获取股票数据"""
        cache_path = os.path.join(self.data_dir, f"{symbol}_30m.csv")
        should_refresh = force_refresh if force_refresh is not None else self.force_refresh

        # 从数据库读取
        if not should_refresh:
            df = self._load_from_db(symbol)
            if df is not None and len(df) > 100:
                return df

        # 检查缓存（30分钟有效）
        if not should_refresh and os.path.exists(cache_path):
            mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if datetime.now() - mod_time < timedelta(minutes=30):
                df = pd.read_csv(cache_path)
                df['date'] = pd.to_datetime(df['date'])
                return df

        # 港股
        if symbol.endswith('.HK'):
            return self._fetch_hk_stock(symbol, cache_path)

        # A股
        if symbol.endswith('.SZ') or symbol.endswith('.SH'):
            return self._fetch_a_stock(symbol, cache_path)

        return None

    def _load_from_db(self, symbol):
        """从 SQLite 加载"""
        if not os.path.exists(self.db_path):
            return None
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(
                'SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol = ? ORDER BY date',
                conn, params=(symbol,)
            )
            conn.close()
            if df.empty:
                return None
            df['date'] = pd.to_datetime(df['date'])
            return df
        except:
            return None

    def _fetch_from_hk_gateway(self, symbol, klt='30'):
        """通过香港网关获取数据"""
        code = symbol.replace('.SH', '').replace('.SZ', '')
        
        _rate_limit()
        
        try:
            url = f'{HK_GATEWAY_URL}/api/kline'
            r = requests.get(url, params={'symbol': code, 'klt': klt}, timeout=15)
            data = r.json()
            
            if data.get('success') and data.get('klines'):
                # 解析 K 线数据
                lines = data['klines']
                df = pd.DataFrame([line.split(',') for line in lines],
                    columns=['date', 'open', 'close', 'high', 'low', 'volume', 
                             'amount', 'amplitude', 'change_pct', 'change', 'turnover'])
                df['date'] = pd.to_datetime(df['date'])
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                df = df.sort_values('date').reset_index(drop=True)
                return df
        except Exception as e:
            sys.stderr.write(f'香港网关失败 {symbol}: {e}\n')
        
        return None

    def _fetch_a_stock(self, symbol, cache_path):
        """获取 A 股数据"""
        # 方法1: 香港网关（实时分钟数据）
        df = self._fetch_from_hk_gateway(symbol, klt='30')
        if df is not None and not df.empty:
            df.to_csv(cache_path, index=False)
            return df
        
        # 方法2: Tushare 日线（免费备用）
        if TUSHARE_AVAILABLE:
            try:
                code = symbol[:6]
                start = (datetime.now() - timedelta(days=60)).strftime('%Y%m%d')
                end = datetime.now().strftime('%Y%m%d')
                df = TUSHARE_PRO.daily(ts_code=symbol, start_date=start, end_date=end)
                if df is not None and not df.empty:
                    df = df.rename(columns={'trade_date': 'date', 'vol': 'volume'})
                    df['date'] = pd.to_datetime(df['date'])
                    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                    df = df.sort_values('date').reset_index(drop=True)
                    df.to_csv(cache_path, index=False)
                    return df
            except Exception as e:
                sys.stderr.write(f'Tushare 失败 {symbol}: {e}\n')
        
        # 方法3: 本地缓存
        return self._use_cache(cache_path)

    def _fetch_hk_stock(self, symbol, cache_path):
        """获取港股数据"""
        # 香港网关支持港股吗？需要测试
        # 目前返回缓存
        return self._use_cache(cache_path)

    def _use_cache(self, cache_path):
        """使用缓存"""
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            df['date'] = pd.to_datetime(df['date'])
            return df
        return None

    def fetch_batch_stocks(self, symbols, force_refresh=True):
        """批量获取"""
        results = {}
        for symbol in symbols:
            df = self.fetch_stock_data(symbol, force_refresh=force_refresh)
            if df is not None:
                results[symbol] = df
        return results

    def get_stock_list(self):
        """获取股票列表"""
        if not os.path.exists(self.db_path):
            return []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT symbol, name FROM stock_info')
            stocks = [{'symbol': r[0], 'name': r[1]} for r in cursor.fetchall()]
            conn.close()
            return stocks
        except:
            return []

    def get_data_stats(self):
        """数据统计"""
        if not os.path.exists(self.db_path):
            return None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM stock_info')
            stock_count = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM kline_30m')
            kline_count = cursor.fetchone()[0]
            cursor.execute('SELECT MIN(date), MAX(date) FROM kline_30m')
            min_date, max_date = cursor.fetchone()
            conn.close()
            return {'stock_count': stock_count, 'kline_count': kline_count,
                    'min_date': min_date, 'max_date': max_date}
        except:
            return None
