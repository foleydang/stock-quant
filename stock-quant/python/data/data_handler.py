import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import time
import sqlite3

# Tushare Pro Token（用户提供）
TUSHARE_TOKEN = '7a9014b18909e8cbce5109d7175f7b21ce37354eaff2371db0da2c58'

# 尝试导入 tushare
try:
    import tushare as ts
    ts.set_token(TUSHARE_TOKEN)
    TUSHARE_PRO = ts.pro_api()
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False
    TUSHARE_PRO = None

# 尝试导入 akshare 作为备用数据源
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

# 全局请求锁，防止并发请求 - 限流在接口层内部
_last_request_time = 0
_min_request_interval = 6.0  # 最小请求间隔（秒）- 增加到6秒避免限流
_max_retries = 2  # 最大重试次数（减少以快速跳过失败的股票）
_max_consecutive_failures = 0  # 连续失败计数
_backoff_multiplier = 1.0  # 退避乘数

# SQLite 数据库路径
DB_PATH = os.path.join(os.path.dirname(__file__), 'stock_data.db')


def _rate_limit():
    """全局请求限流 - 在接口层内部，调用层无需关心"""
    global _last_request_time, _max_consecutive_failures, _backoff_multiplier

    elapsed = time.time() - _last_request_time

    # 计算动态等待时间：基础间隔 * 退避乘数 + 随机抖动（增加范围）
    base_wait = _min_request_interval * _backoff_multiplier
    jitter = random.uniform(1.0, 3.0)  # 增加抖动范围到1-3秒
    wait_time = max(base_wait + jitter - elapsed, 0)

    if wait_time > 0:
        time.sleep(wait_time)

    _last_request_time = time.time()


def _on_success():
    """请求成功时重置退避"""
    global _max_consecutive_failures, _backoff_multiplier
    _max_consecutive_failures = 0
    _backoff_multiplier = 1.0


def _on_failure():
    """请求失败时增加退避"""
    global _max_consecutive_failures, _backoff_multiplier
    _max_consecutive_failures += 1
    # 每次失败增加0.5倍退避，最多3倍
    _backoff_multiplier = min(1.0 + _max_consecutive_failures * 0.5, 3.0)


class DataHandler:
    """
    数据获取处理器

    特点：
    - 优先从 SQLite 数据库读取
    - A股和港股使用 akshare 获取 30 分钟级别数据
    - 全局限流在接口层内部，调用层无需关心延时
    - 支持 30 分钟缓存
    """

    def __init__(self, force_refresh=False):
        self.data_dir = os.path.join(os.path.dirname(__file__), '../data')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.force_refresh = force_refresh
        self.db_path = DB_PATH

    def fetch_stock_data(self, symbol, days=60, force_refresh=None):
        """
        获取股票数据（30分钟级别）

        优先从 SQLite 数据库读取，如果没有则从网络获取

        Args:
            symbol: 股票代码，如 '300015.SZ', '3690.HK'
            days: 获取天数（用于计算需要的30分钟K线数量）
            force_refresh: 是否强制刷新
        """
        cache_path = os.path.join(self.data_dir, f"{symbol}_30m.csv")

        # 判断是否需要刷新
        should_refresh = force_refresh if force_refresh is not None else self.force_refresh

        # 优先从数据库读取
        if not should_refresh:
            df = self._load_from_db(symbol)
            if df is not None and len(df) > 100:
                return df

        # 检查缓存（30分钟有效期）
        if not should_refresh and os.path.exists(cache_path):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if datetime.now() - file_mod_time < timedelta(minutes=30):
                df = pd.read_csv(cache_path)
                df['date'] = pd.to_datetime(df['date'])
                return df

        # 港股
        if symbol.endswith('.HK'):
            return self._fetch_hk_stock_30m(symbol, cache_path)

        # A 股
        if symbol.endswith('.SZ') or symbol.endswith('.SH'):
            return self._fetch_a_stock_30m(symbol, cache_path)

        # 其他不支持
        return None

    def _load_from_db(self, symbol: str) -> pd.DataFrame:
        """从 SQLite 数据库加载数据"""
        if not os.path.exists(self.db_path):
            return None

        try:
            conn = sqlite3.connect(self.db_path)

            # 查询数据
            query = '''
                SELECT date, open, high, low, close, volume
                FROM kline_30m
                WHERE symbol = ?
                ORDER BY date
            '''
            df = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()

            if df.empty:
                return None

            df['date'] = pd.to_datetime(df['date'])
            return df

        except Exception as e:
            return None

    def _fetch_a_stock_30m(self, symbol, cache_path):
        """获取 A 股 30 分钟数据 - 新浪接口优先，东方财富备用"""
        code = symbol[:6]

        if AKSHARE_AVAILABLE:
            # 方法1: 新浪接口
            try:
                _rate_limit()

                market = 'sh' if code.startswith('6') else 'sz'
                sina_code = f"{market}{code}"

                df = ak.stock_zh_a_minute(symbol=sina_code, period='30')

                if df is not None and not df.empty:
                    df = df.rename(columns={'day': 'date'})
                    df['date'] = pd.to_datetime(df['date'])
                    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                    df = df.sort_values('date').reset_index(drop=True)
                    df.to_csv(cache_path, index=False)
                    _on_success()
                    return df

            except Exception as e:
                sys.stderr.write(f"akshare 新浪接口失败 {symbol}: {e}\n")

            # 方法2: 东方财富接口（备用，支持创业板和ETF）
            try:
                _rate_limit()

                df = ak.stock_zh_a_hist_min_em(symbol=code, period='30', adjust='qfq')

                if df is not None and not df.empty:
                    # 重命名列
                    df = df.rename(columns={
                        '时间': 'date',
                        '开盘': 'open',
                        '收盘': 'close',
                        '最高': 'high',
                        '最低': 'low',
                        '成交量': 'volume'
                    })
                    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date').reset_index(drop=True)
                    df.to_csv(cache_path, index=False)
                    _on_success()
                    sys.stderr.write(f"东方财富接口成功 {symbol}\n")
                    return df

            except Exception as e:
                sys.stderr.write(f"akshare 东方财富接口失败 {symbol}: {e}\n")
                _on_failure()

        return self._use_cache_or_none(cache_path)

    def _fetch_hk_stock_30m(self, symbol, cache_path):
        """使用 akshare 获取港股 30 分钟数据"""
        # 转换代码格式：3690.HK -> 03690
        hk_code = symbol.replace('.HK', '')
        while len(hk_code) < 5:
            hk_code = '0' + hk_code

        if not AKSHARE_AVAILABLE:
            return self._use_cache_or_none(cache_path)

        max_retries = 2  # 减少重试次数，快速失败使用缓存
        for attempt in range(max_retries):
            try:
                # 全局限流 - 在接口层内部
                _rate_limit()

                # 获取港股 30 分钟数据（使用 stock_hk_hist_min_em 接口）
                df = ak.stock_hk_hist_min_em(
                    symbol=hk_code,
                    period='30',
                    adjust='qfq'
                )

                if df is not None and not df.empty:
                    # 重命名列
                    df = df.rename(columns={
                        '时间': 'date',
                        '开盘': 'open',
                        '收盘': 'close',
                        '最高': 'high',
                        '最低': 'low',
                        '成交量': 'volume'
                    })

                    df['date'] = pd.to_datetime(df['date'])
                    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                    df = df.sort_values('date')
                    df = df.reset_index(drop=True)

                    # 保存缓存
                    df.to_csv(cache_path, index=False)
                    _on_success()  # 重置退避
                    return df

            except Exception as e:
                _on_failure()  # 增加退避
                sys.stderr.write(f"获取港股 30m 数据失败 {symbol} (尝试 {attempt+1}/{max_retries}): {e}\n")

                if attempt < max_retries - 1:
                    # 短暂等待后重试
                    time.sleep(random.uniform(2, 4))

        return self._use_cache_or_none(cache_path)

    def _use_cache_or_none(self, cache_path):
        """使用缓存或返回 None"""
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            df['date'] = pd.to_datetime(df['date'])
            return df
        return None

    def fetch_batch_stocks(self, symbols, force_refresh=True):
        """
        批量获取多只股票数据

        注意：延时已在接口层内部处理，调用层无需关心
        """
        results = {}
        for symbol in symbols:
            df = self.fetch_stock_data(symbol, force_refresh=force_refresh)
            if df is not None:
                results[symbol] = df
        return results

    def get_stock_list(self):
        """从数据库获取股票列表"""
        if not os.path.exists(self.db_path):
            return []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT symbol, name FROM stock_info')
            stocks = [{'symbol': row[0], 'name': row[1]} for row in cursor.fetchall()]
            conn.close()
            return stocks
        except:
            return []

    def get_data_stats(self):
        """获取数据库统计信息"""
        if not os.path.exists(self.db_path):
            return None

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 股票数量
            cursor.execute('SELECT COUNT(*) FROM stock_info')
            stock_count = cursor.fetchone()[0]

            # K线数据量
            cursor.execute('SELECT COUNT(*) FROM kline_30m')
            kline_count = cursor.fetchone()[0]

            # 数据时间范围
            cursor.execute('SELECT MIN(date), MAX(date) FROM kline_30m')
            min_date, max_date = cursor.fetchone()

            conn.close()

            return {
                'stock_count': stock_count,
                'kline_count': kline_count,
                'min_date': min_date,
                'max_date': max_date
            }
        except:
            return None