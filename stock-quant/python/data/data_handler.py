"""数据处理器 - 支持A股/港股/ETF"""

import os
import sys
import pandas as pd
import sqlite3
import requests
import re
from datetime import datetime, timedelta
import time

DB_PATH = os.path.join(os.path.dirname(__file__), 'stock_data.db')
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')

# 腾讯财经 API（港股/ETF/A股实时行情）
TENCENT_QUOTE_API = 'http://qt.gtimg.cn/q='

# Tushare Token（A股日线备用）
TUSHARE_TOKEN = '7a9014b18909e8cbce5109d7175f7b21ce37354eaff2371db0da2c58'


class DataHandler:
    def __init__(self, force_refresh=False):
        self.force_refresh = force_refresh
        self.db_path = DB_PATH
        self.data_dir = DATA_DIR
        self.last_fetch_status = {}
        self.tushare_pro = None
        self._init_tushare()
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def _init_tushare(self):
        try:
            import tushare as ts
            ts.set_token(TUSHARE_TOKEN)
            self.tushare_pro = ts.pro_api()
        except Exception as e:
            sys.stderr.write(f"Tushare 初始化失败: {e}\n")

    def fetch_stock_data(self, symbol, days=60, force_refresh=None):
        """获取股票数据"""
        cache_path = os.path.join(self.data_dir, f"{symbol}_daily.csv")
        should_refresh = force_refresh if force_refresh is not None else self.force_refresh

        # 从数据库读取（如果数据足够且新鲜）
        if not should_refresh:
            df = self._load_from_db(symbol)
            # 特征计算需要至少200条数据
            if df is not None and len(df) >= 200 and self.is_data_fresh(df):
                return df

        # 检查缓存（4小时有效）
        if not should_refresh and os.path.exists(cache_path):
            mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if datetime.now() - mod_time < timedelta(hours=4):
                df = pd.read_csv(cache_path)
                df['date'] = pd.to_datetime(df['date'])
                if self.is_data_fresh(df):
                    return df

        # 根据类型选择数据源
        if symbol.endswith('.HK'):
            return self._fetch_hk_stock(symbol, cache_path)
        
        if '.SZ' in symbol or '.SH' in symbol:
            # ETF 判断：代码以 51/15/16/18 开头
            code = symbol[:6]
            if code.startswith(('51', '15', '16', '18')):
                return self._fetch_etf(symbol, cache_path)
            return self._fetch_a_stock(symbol, cache_path)

        return None

    def _load_from_db(self, symbol):
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

    # ==================== 腾讯财经 API（实时行情） ====================
    
    def _fetch_from_tencent(self, symbol):
        """从腾讯财经获取实时行情（港股/ETF/A股）"""
        # 提取代码部分：3690.HK -> 3690, 600036.SH -> 600036
        code = symbol.split('.')[0]
        
        # 构建查询参数
        if symbol.endswith('.HK'):
            # 港股代码需要补齐5位，如 3690 -> hk03690
            hk_code = code.zfill(5)  # 补齐5位: 3690 -> 03690
            query = f"hk{hk_code}"
        elif symbol.endswith('.SZ'):
            query = f"sz{code}"
        elif symbol.endswith('.SH'):
            query = f"sh{code}"
        else:
            return None
        
        try:
            url = f"{TENCENT_QUOTE_API}{query}"
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            
            # 解析数据: v_xxxx="字段1~字段2~..."
            match = re.search(r'v_\w+="(.+)"', r.text)
            if not match:
                return None
            
            fields = match.group(1).split('~')
            if len(fields) < 7:
                return None
            
            # 提取关键数据
            name = fields[1]
            current_price = float(fields[3])
            prev_close = float(fields[4])
            open_price = float(fields[5])
            volume = float(fields[6])
            
            # 高低价需要从买卖盘提取，或用当前价近似
            high_price = current_price  # 可从其他字段提取
            low_price = current_price
            
            # 构建单条数据（今日）
            now = datetime.now()
            df = pd.DataFrame([{
                'date': now,
                'open': open_price,
                'high': max(open_price, current_price),
                'low': min(open_price, current_price),
                'close': current_price,
                'volume': volume
            }])
            
            self.last_fetch_status[symbol] = {
                'success': True,
                'source': 'tencent_realtime',
                'name': name,
                'price': current_price,
                'prev_close': prev_close,
                'change_pct': (current_price - prev_close) / prev_close * 100,
                'fetch_time': now.isoformat()
            }
            
            return df
            
        except Exception as e:
            sys.stderr.write(f"腾讯财经失败 {symbol}: {e}\n")
            self.last_fetch_status[symbol] = {
                'success': False,
                'source': 'tencent',
                'error': str(e),
                'fetch_time': datetime.now().isoformat()
            }
        return None

    def _fetch_hk_stock(self, symbol, cache_path):
        """获取港股数据 - 使用腾讯财经"""
        df = self._fetch_from_tencent(symbol)
        if df is not None and not df.empty:
            # 港股只有实时数据，没有历史K线
            # 合并缓存的历史数据
            cached = self._use_cache(cache_path)
            if cached is not None and not cached.empty:
                # 添加今日数据到历史
                df = pd.concat([cached, df]).tail(60).reset_index(drop=True)
            df.to_csv(cache_path, index=False)
            return df
        return self._use_cache(cache_path)

    def _fetch_etf(self, symbol, cache_path):
        """获取ETF数据 - 使用腾讯财经"""
        df = self._fetch_from_tencent(symbol)
        if df is not None and not df.empty:
            cached = self._use_cache(cache_path)
            if cached is not None and not cached.empty:
                df = pd.concat([cached, df]).tail(60).reset_index(drop=True)
            df.to_csv(cache_path, index=False)
            return df
        return self._use_cache(cache_path)

    def _fetch_a_stock(self, symbol, cache_path):
        """获取A股数据 - Tushare日线 + 腾讯实时"""
        # 方法1: Tushare 日线
        df = self._fetch_from_tushare(symbol, days=days)  # 使用传入的天数
        if df is not None and not df.empty:
            df.to_csv(cache_path, index=False)
            return df
        
        # 方法2: 腾讯实时（备用）
        df = self._fetch_from_tencent(symbol)
        if df is not None and not df.empty:
            cached = self._use_cache(cache_path)
            if cached is not None and not cached.empty:
                df = pd.concat([cached, df]).tail(60).reset_index(drop=True)
            df.to_csv(cache_path, index=False)
            return df
        
        return self._use_cache(cache_path)

    def _fetch_from_tushare(self, symbol, days=365):  # 默认获取1年历史
        if self.tushare_pro is None:
            return None
        try:
            start = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
            end = datetime.now().strftime('%Y%m%d')
            df = self.tushare_pro.daily(ts_code=symbol, start_date=start, end_date=end)
            if df is not None and not df.empty:
                df = df.rename(columns={'trade_date': 'date', 'vol': 'volume'})
                df['date'] = pd.to_datetime(df['date'])
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                df = df.sort_values('date').reset_index(drop=True)
                self.last_fetch_status[symbol] = {
                    'success': True,
                    'source': 'tushare_daily',
                    'count': len(df),
                    'last_date': df['date'].iloc[-1].strftime('%Y-%m-%d'),
                    'fetch_time': datetime.now().isoformat()
                }
                return df
        except Exception as e:
            sys.stderr.write(f"Tushare 失败 {symbol}: {e}\n")
        return None

    def _use_cache(self, cache_path):
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            df['date'] = pd.to_datetime(df['date'])
            return df
        return None

    def fetch_batch_stocks(self, symbols, force_refresh=True):
        """批量获取"""
        results = {}
        for i, symbol in enumerate(symbols):
            df = self.fetch_stock_data(symbol, force_refresh=force_refresh)
            if df is not None and not df.empty:
                results[symbol] = df
            time.sleep(0.3)
        return results

    def get_realtime_prices(self, symbols):
        """批量获取实时价格（腾讯财经，适合监控）"""
        # 构建查询字符串
        queries = []
        for symbol in symbols:
            code = symbol.split('.')[0]
            if symbol.endswith('.HK'):
                queries.append(f"hk0{code}")
            elif symbol.endswith('.SZ'):
                queries.append(f"sz{code}")
            elif symbol.endswith('.SH'):
                queries.append(f"sh{code}")
        
        if not queries:
            return {}
        
        try:
            url = f"{TENCENT_QUOTE_API}{','.join(queries)}"
            r = requests.get(url, timeout=15)
            
            prices = {}
            for line in r.text.strip().split('\n'):
                match = re.search(r'v_(\w+)="(.+)"', line)
                if match:
                    query_key = match.group(1)
                    fields = match.group(2).split('~')
                    if len(fields) >= 7:
                        # 解析查询键获取原始代码
                        # hk03690 -> 3690.HK, sz159792 -> 159792.SZ
                        if query_key.startswith('hk'):
                            # hk03690 -> 去掉前导0 -> 3690.HK
                            code = query_key[2:].lstrip('0')  # 03690 -> 3690
                            if not code:
                                code = query_key[2:]  # 如果全是0，保留原样
                            symbol = f"{code}.HK"
                        elif query_key.startswith('sz'):
                            code = query_key[2:]
                            symbol = f"{code}.SZ"
                        elif query_key.startswith('sh'):
                            code = query_key[2:]
                            symbol = f"{code}.SH"
                        
                        prices[symbol] = {
                            'name': fields[1],
                            'price': float(fields[3]),
                            'prev_close': float(fields[4]),
                            'open': float(fields[5]),
                            'volume': float(fields[6]),
                            'change_pct': (float(fields[3]) - float(fields[4])) / float(fields[4]) * 100
                        }
            
            return prices
        except Exception as e:
            sys.stderr.write(f"批量获取失败: {e}\n")
        return {}

    def get_stock_list(self):
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

    def is_data_fresh(self, df, max_age_hours=None):
        if df is None or df.empty:
            return False
        
        last_date = pd.to_datetime(df['date'].iloc[-1])
        now = datetime.now()
        
        # 如果是今日实时数据，认为是新鲜的
        if last_date.date() == now.date():
            return True
        
        # 日线数据是交易日 15:00 的收盘数据
        last_date_with_time = datetime(last_date.year, last_date.month, last_date.day, 15, 0)
        age_hours = (now - last_date_with_time).total_seconds() / 3600
        
        today_weekday = now.weekday()
        if max_age_hours is None:
            if today_weekday == 6:  # 周日
                max_age_hours = 72
            elif today_weekday == 0:  # 周一
                max_age_hours = 72
            elif today_weekday == 5:  # 周六
                max_age_hours = 48
            else:
                max_age_hours = 24
        
        return age_hours < max_age_hours

    def get_fetch_report(self):
        return self.last_fetch_status
