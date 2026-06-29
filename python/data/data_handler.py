"""数据处理器 - 支持A股/港股/ETF

数据流：DB → API（不再使用CSV缓存）
- 查询优先从 SQLite DB 读取
- DB数据不新鲜时从 API 拉取，并写入 DB
"""

import os
import sys
import pandas as pd
import sqlite3
import requests
import re
from datetime import datetime, timedelta
import time

DB_PATH = os.path.join(os.path.dirname(__file__), 'stock_data.db')
DATA_DIR = os.path.join(os.path.dirname(__file__))

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

    def _init_tushare(self):
        try:
            import tushare as ts
            ts.set_token(TUSHARE_TOKEN)
            self.tushare_pro = ts.pro_api()
        except Exception as e:
            sys.stderr.write(f"Tushare 初始化失败: {e}\n")

    def fetch_stock_data(self, symbol, days=60, force_refresh=None):
        """获取股票数据（DB优先，API补充）

        1. 先从 DB 读，数据够且新鲜就直接返回
        2. DB不够或不新鲜 → 从 API 拉取，写入 DB 后返回
        """
        should_refresh = force_refresh if force_refresh is not None else self.force_refresh

        # 从数据库读取（如果数据足够且新鲜）
        if not should_refresh:
            df = self._load_from_db(symbol)
            # 特征计算需要至少200条数据
            if df is not None and len(df) >= 200 and self.is_data_fresh(df):
                return df

        # DB数据不够或不新鲜，从API拉取
        if symbol.endswith('.HK'):
            return self._fetch_hk_stock(symbol)
        
        if '.SZ' in symbol or '.SH' in symbol:
            code = symbol[:6]
            if code.startswith(('51', '15', '16', '18')):
                return self._fetch_etf(symbol)
            return self._fetch_a_stock(symbol)

        return None

    def _load_from_db(self, symbol, min_rows=200):
        """从数据库加载股票数据"""
        if not os.path.exists(self.db_path):
            return None
        try:
            conn = sqlite3.connect(self.db_path)
            # 优先查30分钟线（数据量更丰富）
            df = pd.read_sql_query(
                'SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol = ? ORDER BY date',
                conn, params=(symbol,)
            )
            if df.empty or len(df) < min_rows:
                # fallback到日线
                df = pd.read_sql_query(
                    'SELECT date, open, high, low, close, volume FROM kline_daily WHERE symbol = ? ORDER BY date',
                    conn, params=(symbol,)
                )
            conn.close()
            if df.empty:
                return None
            df['date'] = pd.to_datetime(df['date'], format='mixed')
            return df
        except Exception as e:
            return None

    def _save_to_db(self, symbol, df, table='kline_daily'):
        """将数据写入数据库"""
        if df is None or df.empty:
            return
        try:
            conn = sqlite3.connect(self.db_path)
            # 逐行写入，利用 UNIQUE(symbol, date) 做去重
            for _, row in df.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else str(row['date'])
                conn.execute(
                    f'INSERT OR REPLACE INTO {table} (symbol, date, open, high, low, close, volume, updated_at) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                    (symbol, date_str, float(row['open']), float(row['high']),
                     float(row['low']), float(row['close']), float(row['volume']),
                     datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                )
            conn.commit()
            conn.close()
        except Exception as e:
            sys.stderr.write(f"写入DB失败 {symbol}: {e}\n")

    # ==================== 腾讯财经 API（实时行情） ====================
    
    def _fetch_from_tencent(self, symbol):
        """从腾讯财经获取实时行情（港股/ETF/A股）"""
        code = symbol.split('.')[0]
        
        if symbol.endswith('.HK'):
            hk_code = code.zfill(5)
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
            
            match = re.search(r'v_\w+="(.+)"', r.text)
            if not match:
                return None
            
            fields = match.group(1).split('~')
            if len(fields) < 7:
                return None
            
            name = fields[1]
            current_price = float(fields[3])
            prev_close = float(fields[4])
            open_price = float(fields[5])
            volume = float(fields[6])
            
            try:
                high_price = float(fields[33]) if len(fields) > 33 and fields[33] else current_price
                low_price = float(fields[34]) if len(fields) > 34 and fields[34] else current_price
            except:
                high_price = current_price
                low_price = current_price
            
            now = datetime.now()
            df = pd.DataFrame([{
                'date': now,
                'open': open_price,
                'high': high_price,
                'low': low_price,
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

    def _fetch_hk_stock(self, symbol):
        """获取港股数据 - 腾讯实时 → 合并DB历史 → 写回DB"""
        realtime_df = self._fetch_from_tencent(symbol)
        if realtime_df is not None and not realtime_df.empty:
            # 合并DB中的历史数据
            history_df = self._load_from_db(symbol, min_rows=0)
            if history_df is not None and not history_df.empty:
                df = pd.concat([history_df, realtime_df]).drop_duplicates(
                    subset=['date'], keep='last'
                ).tail(60).reset_index(drop=True)
            else:
                df = realtime_df
            self._save_to_db(symbol, df)
            return df
        # API失败，fallback到DB历史
        return self._load_from_db(symbol, min_rows=0)

    def _fetch_etf(self, symbol):
        """获取ETF数据 - 腾讯实时 → 合并DB历史 → 写回DB"""
        realtime_df = self._fetch_from_tencent(symbol)
        if realtime_df is not None and not realtime_df.empty:
            history_df = self._load_from_db(symbol, min_rows=0)
            if history_df is not None and not history_df.empty:
                df = pd.concat([history_df, realtime_df]).drop_duplicates(
                    subset=['date'], keep='last'
                ).tail(60).reset_index(drop=True)
            else:
                df = realtime_df
            self._save_to_db(symbol, df)
            return df
        return self._load_from_db(symbol, min_rows=0)

    def _fetch_a_stock(self, symbol):
        """获取A股数据 - Tushare日线 → 写入DB → 返回"""
        # 方法1: Tushare 日线
        df = self._fetch_from_tushare(symbol)
        if df is not None and not df.empty:
            self._save_to_db(symbol, df)
            return df
        
        # 方法2: 腾讯实时（备用） → 合并DB历史
        realtime_df = self._fetch_from_tencent(symbol)
        if realtime_df is not None and not realtime_df.empty:
            history_df = self._load_from_db(symbol, min_rows=0)
            if history_df is not None and not history_df.empty:
                df = pd.concat([history_df, realtime_df]).drop_duplicates(
                    subset=['date'], keep='last'
                ).tail(60).reset_index(drop=True)
            else:
                df = realtime_df
            self._save_to_db(symbol, df)
            return df
        
        # 全失败，fallback到DB历史
        return self._load_from_db(symbol, min_rows=0)

    def _fetch_from_tushare(self, symbol, days=365):
        if self.tushare_pro is None:
            return None
        try:
            start = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
            end = datetime.now().strftime('%Y%m%d')
            df = self.tushare_pro.daily(ts_code=symbol, start_date=start, end_date=end)
            if df is not None and not df.empty:
                df = df.rename(columns={'trade_date': 'date', 'vol': 'volume'})
                df['date'] = pd.to_datetime(df['date'], format='mixed')
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

    def fetch_real_30min_kline(self, symbol, count=20):
        """Fetch real 30min K-line from Sina Finance API"""
        if not (symbol.endswith(".SZ") or symbol.endswith(".SH")):
            return None
        code = symbol[:6]
        prefix = "sz" if symbol.endswith(".SZ") else "sh"
        sina_code = prefix + code
        try:
            url = "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"
            params = {"symbol": sina_code, "scale": "30", "datalen": count}
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                import json
                data = json.loads(r.text)
                if isinstance(data, list) and len(data) > 0:
                    df = pd.DataFrame(data)
                    df = df.rename(columns={"day": "date"})
                    df["date"] = pd.to_datetime(df["date"], format="mixed")
                    for col in ["open", "high", "low", "close", "volume"]:
                        df[col] = df[col].astype(float)
                    df = df[["date", "open", "high", "low", "close", "volume"]]
                    df = df.sort_values("date").reset_index(drop=True)
                    self.last_fetch_status[symbol] = {
                        "success": True, "source": "sina_30min",
                        "count": len(df),
                        "last_date": df["date"].iloc[-1].strftime("%Y-%m-%d %H:%M"),
                        "fetch_time": datetime.now().isoformat()
                    }
                    return df
        except Exception as e:
            sys.stderr.write(f"Sina 30min failed {symbol}: {e}\n")
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
        queries = []
        symbol_to_query = {}
        for symbol in symbols:
            code = symbol.split('.')[0]
            if symbol.endswith('.HK'):
                hk_code = code.zfill(5)
                query = f"hk{hk_code}"
                queries.append(query)
                symbol_to_query[query] = symbol
            elif symbol.endswith('.SZ'):
                query = f"sz{code}"
                queries.append(query)
                symbol_to_query[query] = symbol
            elif symbol.endswith('.SH'):
                query = f"sh{code}"
                queries.append(query)
                symbol_to_query[query] = symbol
        
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
                        symbol = symbol_to_query.get(query_key)
                        if not symbol:
                            if query_key.startswith('hk'):
                                code = query_key[2:].lstrip('0')
                                if not code:
                                    code = query_key[2:]
                                symbol = f"{code}.HK"
                            elif query_key.startswith('sz'):
                                symbol = f"{query_key[2:]}.SZ"
                            elif query_key.startswith('sh'):
                                symbol = f"{query_key[2:]}.SH"
                        
                        if symbol:
                            prices[symbol] = {
                                "name": fields[1],
                                "price": float(fields[3]),
                                "prev_close": float(fields[4]),
                                "open": float(fields[5]),
                                "high": float(fields[33]) if len(fields) > 33 and fields[33] else float(fields[3]),
                                "low": float(fields[34]) if len(fields) > 34 and fields[34] else float(fields[3]),
                                "volume": float(fields[6]),
                                "time": fields[30],
                                "change_pct": (float(fields[3]) - float(fields[4])) / float(fields[4]) * 100
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
        except Exception as e:
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
        except Exception as e:
            return None

    def is_data_fresh(self, df, max_age_hours=None):
        if df is None or df.empty:
            return False
        
        last_date = pd.to_datetime(df['date'], format='mixed').iloc[-1]
        now = datetime.now()
        
        if last_date.date() == now.date():
            return True
        
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