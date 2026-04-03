#!/usr/bin/env python3
"""
使用 Sina/Eastmoney/AKShare 获取股票实时数据
- A 股：Sina/Eastmoney 接口
- 港股：AKShare 接口（更稳定）
这些接口比 Tushare 经典接口更稳定
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class StockDataHandler:
    """使用 Sina/Eastmoney 的数据处理器"""

    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), '../data')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # 用户代理
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://finance.sina.com.cn/'
        }

    def _convert_symbol_sina(self, symbol: str) -> str:
        """转换股票代码为 Sina 格式"""
        if symbol.endswith('.SZ'):
            return f"sz{symbol[:6]}"
        elif symbol.endswith('.SH'):
            return f"sh{symbol[:6]}"
        elif symbol.endswith('.HK'):
            # 港股 - 去掉后缀，保留 5 位代码（如 03690）
            code = symbol.replace('.HK', '').zfill(5)
            return f"hk{code}"
        return symbol

    def _convert_symbol_em(self, symbol: str) -> str:
        """转换股票代码为 Eastmoney 格式"""
        if symbol.endswith('.SZ'):
            return f'0.{symbol[:6]}'  # 深交所
        elif symbol.endswith('.SH'):
            return f'1.{symbol[:6]}'  # 上交所
        elif symbol.endswith('.HK'):
            # 港股
            code = symbol.replace('.HK', '').zfill(5)
            return f'120.{code}'  # 港股市场
        return symbol

    def fetch_spot(self, symbol: str) -> Optional[Dict]:
        """
        获取股票实时行情（使用 Sina 接口）

        Args:
            symbol: 股票代码，如 '300015.SZ'

        Returns:
            包含实时行情数据的字典
        """
        try:
            sina_code = self._convert_symbol_sina(symbol)
            url = f"https://hq.sinajs.cn/list={sina_code}"

            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            # 解析返回数据
            # 格式：var hq_str_sz000001="平安银行，8.80,8.80,8.75,8.85,8.74,8.75,8.76,322200,28323465,..."
            text = response.text

            if '=' not in text:
                return None

            parts = text.split('=')
            if len(parts) < 2:
                return None

            data_str = parts[1].strip().strip('"')
            if not data_str or data_str == '':
                return None

            fields = data_str.split(',')

            if len(fields) < 32:
                return None

            # 解析字段
            # 0: 股票名，1: 今日开盘，2: 昨收，3: 当前价，4: 今日最高，5: 今日最低
            # 6: 买一价，7: 卖一价，8: 成交量 (手)，9: 成交额 (元)
            name = fields[0]
            open_price = float(fields[1]) if fields[1] else 0
            pre_close = float(fields[2]) if fields[2] else 0
            price = float(fields[3]) if fields[3] else 0
            high = float(fields[4]) if fields[4] else 0
            low = float(fields[5]) if fields[5] else 0
            bid = float(fields[6]) if fields[6] else 0
            ask = float(fields[7]) if fields[7] else 0
            volume = float(fields[8]) if fields[8] else 0  # 手
            amount = float(fields[9]) if fields[9] else 0  # 元

            # 日期时间
            date_str = fields[30] if len(fields) > 30 else ''
            time_str = fields[31] if len(fields) > 31 else ''

            return {
                'symbol': symbol[:6],
                'name': name,
                'price': price,
                'open': open_price,
                'high': high,
                'low': low,
                'close': pre_close,  # 昨收
                'volume': volume * 100,  # 转换为股数
                'amount': amount,
                'bid': bid,
                'ask': ask,
                'change': price - pre_close,
                'change_pct': ((price - pre_close) / pre_close * 100) if pre_close else 0,
                'date': date_str,
                'time': time_str,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            sys.stderr.write(f"获取实时行情失败 {symbol}: {e}\n")
            return None

    def fetch_history(self, symbol: str, days: int = 90) -> Optional[pd.DataFrame]:
        """
        获取股票历史行情（使用 Sina 日线数据）

        Args:
            symbol: 股票代码
            days: 获取天数

        Returns:
            DataFrame 包含历史数据
        """
        try:
            # 先获取股票所属市场
            code = symbol[:6]

            # 使用 Sina 的历史数据接口
            # 这个接口返回的是复权后的数据
            year = datetime.now().year
            data_frames = []

            # 获取最近 2 年的数据
            for y in [year, year - 1, year - 2]:
                url = f"https://money.finance.sina.com.cn/corp/go.php/vMS_FuQuanMarketHistoryForStock/stockid/{code}.phtml"
                params = {
                    'symbol': code,
                    'year': y
                }

                response = requests.get(url, headers=self.headers, params=params, timeout=10)
                if response.status_code == 200:
                    # 解析 HTML 表格
                    try:
                        tables = pd.read_html(response.text)
                        if tables:
                            df = tables[0]
                            data_frames.append(df)
                    except:
                        pass

                time.sleep(0.5)  # 避免请求过快

            if not data_frames:
                # 如果历史数据获取失败，返回空 DataFrame
                return pd.DataFrame()

            df = pd.concat(data_frames, ignore_index=True)

            if df.empty:
                return pd.DataFrame()

            # 重命名列
            if len(df.columns) >= 6:
                df = df.rename(columns={
                    df.columns[0]: 'date',
                    df.columns[1]: 'open',
                    df.columns[2]: 'high',
                    df.columns[3]: 'low',
                    df.columns[4]: 'close',
                    df.columns[5]: 'volume'
                })
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date', ascending=False)
                df = df.head(days)
                df = df.sort_values('date')
                df = df.reset_index(drop=True)

            # 保存到本地
            file_path = os.path.join(self.data_dir, f"{symbol}_sina.csv")
            df.to_csv(file_path, index=False)

            return df

        except Exception as e:
            sys.stderr.write(f"获取历史数据失败 {symbol}: {e}\n")
            return pd.DataFrame()

    def fetch_history_eastmoney(self, symbol: str, days: int = 90, period: str = 'daily') -> Optional[pd.DataFrame]:
        """
        获取股票历史行情（使用 Eastmoney 接口 - 备选）

        Args:
            symbol: 股票代码
            days: 获取天数
            period: 数据周期 ('daily', '30m', '60m')

        Returns:
            DataFrame 包含历史数据
        """
        import time

        # 重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Eastmoney 接口
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')

                # 使用转换函数获取 secid
                secid = self._convert_symbol_em(symbol)

                url = f"http://77.push2his.eastmoney.com/api/qt/stock/kline/get"

                # 设置 K 线周期
                klt_map = {
                    'daily': '101',    # 日线
                    '30m': '30',       # 30 分钟
                    '60m': '60',       # 60 分钟
                    '5m': '5',         # 5 分钟
                    '15m': '15',       # 15 分钟
                }
                klt = klt_map.get(period, '101')

                # 分钟线数据获取更多数据
                if period in ['30m', '60m']:
                    # 获取最近 30 天的分钟线数据（足够生成信号）
                    days = max(days, 30)
                    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')

                params = {
                    'secid': secid,
                    'fields1': 'f1,f2,f3,f4,f5,f6',
                    'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
                    'klt': klt,
                    'fqt': '1',    # 前复权
                    'beg': start_date,
                    'end': end_date,
                }

                response = requests.get(url, headers=self.headers, params=params, timeout=15)
                response.raise_for_status()

                data = response.json()

                if data.get('data') is None or data['data'].get('klines') is None:
                    return pd.DataFrame()

                klines = data['data']['klines']

                # 解析 K 线数据
                records = []
                for line in klines:
                    parts = line.split(',')
                    if len(parts) >= 6:
                        records.append({
                            'date': parts[0],
                            'open': float(parts[1]),
                            'high': float(parts[2]),
                            'low': float(parts[3]),
                            'close': float(parts[4]),
                            'volume': float(parts[5])
                        })

                df = pd.DataFrame(records)
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    df = df.reset_index(drop=True)

                    # 保存到本地
                    file_suffix = period if period in ['30m', '60m', '5m', '15m'] else 'em'
                    file_path = os.path.join(self.data_dir, f"{symbol}_{file_suffix}.csv")
                    df.to_csv(file_path, index=False)

                return df

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    sys.stderr.write(f"获取历史数据失败 (Eastmoney) {symbol}: {e}\n")
                    # 尝试使用 akshare 获取分钟线数据
                    if period in ['30m', '60m']:
                        return self.fetch_history_akshare_minute(symbol, period=period)
                    return pd.DataFrame()

        return pd.DataFrame()

    def fetch_history_akshare_minute(self, symbol: str, period: str = '30m') -> Optional[pd.DataFrame]:
        """
        使用 AKShare 获取分钟线数据（Sina 数据源）

        Args:
            symbol: 股票代码
            period: 周期 ('30m', '60m', '5m', '15m')

        Returns:
            DataFrame 包含历史数据
        """
        try:
            import akshare as ak

            # 转换为 Sina 格式的股票代码 (shXXXXXX 或 szXXXXXX)
            if symbol.endswith('.SZ'):
                sina_code = f"sz{symbol[:6]}"
            elif symbol.endswith('.SH'):
                sina_code = f"sh{symbol[:6]}"
            else:
                sina_code = symbol[:6]

            # AKShare 获取分钟数据 (使用 stock_zh_a_minute 函数)
            # period 参数：1, 5, 15, 30, 60 分钟
            ak_period = period.replace('m', '')
            df = ak.stock_zh_a_minute(symbol=sina_code, period=ak_period)  # adjust not supported for minute data

            if df is None or len(df) == 0:
                return pd.DataFrame()

            # 重命名列为标准格式
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']

            # 转换为数值类型
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 日期格式化
            df['date'] = pd.to_datetime(df['date'])

            # 添加涨跌额/涨跌幅
            df['change'] = df['close'] - df['close'].shift(1)
            df['pct_change'] = df['close'].pct_change() * 100

            return df.tail(100)

        except Exception as e:
            sys.stderr.write(f"获取分钟线数据失败 (AKShare) {symbol}: {e}\n")
            return pd.DataFrame()

    def fetch_spot_akshare(self, symbol: str) -> Optional[Dict]:
        """
        获取港股实时行情（使用 AKShare 接口）

        Args:
            symbol: 港股代码，如 '0700.HK'

        Returns:
            包含实时行情数据的字典
        """
        try:
            import akshare as ak

            # 获取港股实时行情
            code = symbol.replace('.HK', '')
            df = ak.stock_hk_daily(symbol=code)

            if df is None or df.empty:
                return None

            # 获取最新数据
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest

            # 计算涨跌幅
            close = float(latest['close'])
            pre_close = float(prev['close'])
            change = close - pre_close
            change_pct = (change / pre_close * 100) if pre_close else 0

            # 日期字段 - 直接访问列
            date_str = str(latest.get('date', datetime.now().strftime('%Y-%m-%d')))

            return {
                'symbol': symbol,
                'name': f'HK{code}',
                'price': close,
                'open': float(latest['open']),
                'high': float(latest['high']),
                'low': float(latest['low']),
                'close': pre_close,  # 昨收
                'volume': float(latest['volume']),
                'amount': float(latest['amount']) if 'amount' in latest else 0,
                'change': change,
                'change_pct': change_pct,
                'date': date_str,
                'time': '16:00',  # 港股收盘时间
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            sys.stderr.write(f"获取港股实时行情失败 {symbol}: {e}\n")
            return None

    def fetch_history_akshare(self, symbol: str, days: int = 90) -> Optional[pd.DataFrame]:
        """
        获取港股历史行情（使用 AKShare 接口）

        Args:
            symbol: 港股代码，如 '0700.HK'
            days: 获取天数

        Returns:
            DataFrame 包含历史数据
        """
        try:
            import akshare as ak

            code = symbol.replace('.HK', '')
            df = ak.stock_hk_daily(symbol=code)

            if df is None or df.empty:
                return pd.DataFrame()

            # 重命名列
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })

            # 确保日期格式正确
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=False)
            df = df.head(days)
            df = df.sort_values('date')
            df = df.reset_index(drop=True)

            # 保存到本地
            file_path = os.path.join(self.data_dir, f"{symbol}_ak.csv")
            df.to_csv(file_path, index=False)

            return df

        except Exception as e:
            sys.stderr.write(f"获取港股历史数据失败 {symbol}: {e}\n")
            return pd.DataFrame()

    def get_watchlist_data(self, watchlist: List[Dict], fetch_history: bool = True, period: str = 'daily') -> Dict:
        """
        批量获取股票池数据

        Args:
            watchlist: 股票列表
            fetch_history: 是否获取历史数据
            period: 数据周期 ('daily', '30m', '60m')

        Returns:
            股票数据字典
        """
        results = {}

        for i, stock in enumerate(watchlist):
            symbol = stock['symbol']
            name = stock.get('name', '')

            # 去掉后缀获取代码
            if symbol.endswith('.SZ') or symbol.endswith('.SH'):
                code = symbol[:6]
            elif symbol.endswith('.HK'):
                code = symbol.replace('.HK', '')
            else:
                code = symbol

            print(f"获取数据：{name} ({code})...")

            # 港股使用 AKShare，A 股使用 Sina/Eastmoney
            is_hk = symbol.endswith('.HK')

            # 获取实时行情
            if is_hk:
                spot_data = self.fetch_spot_akshare(symbol)
            else:
                spot_data = self.fetch_spot(symbol)

            if spot_data:
                # 更新股票名称
                spot_data['name'] = name
                results[symbol] = spot_data
                print(f"  ✓ 实时价格：{spot_data['price']:.2f}, 涨跌幅：{spot_data['change_pct']:.2f}%")

                # 获取历史数据
                if fetch_history:
                    if is_hk:
                        # 港股使用 AKShare（目前只有日线数据）
                        hist_df = self.fetch_history_akshare(symbol)
                    else:
                        # A 股根据周期选择接口
                        if period in ['30m', '60m', '5m', '15m']:
                            # 分钟线数据使用 Eastmoney
                            hist_df = self.fetch_history_eastmoney(symbol, period=period)
                            cache_suffix = f"{period}"
                        else:
                            # 日线数据
                            hist_df = self.fetch_history_eastmoney(symbol)
                            cache_suffix = "em"

                        if hist_df is None or hist_df.empty:
                            hist_df = self.fetch_history(symbol)

                    # 如果网络获取失败，尝试加载本地缓存
                    if hist_df is None or hist_df.empty:
                        if is_hk:
                            # 港股需要补齐前导零
                            cache_code = code.zfill(5)  # 港股代码补齐 5 位，如 03690
                            cache_file = os.path.join(self.data_dir, f"{cache_code}.HK_ak.csv")
                        else:
                            # A 股根据周期选择缓存文件
                            cache_file = os.path.join(self.data_dir, f"{symbol}_{period}.csv")

                        if os.path.exists(cache_file):
                            try:
                                hist_df = pd.read_csv(cache_file)
                                hist_df['date'] = pd.to_datetime(hist_df['date'])
                                print(f"  ✓ 从缓存加载历史数据：{len(hist_df)} 条")
                            except Exception as e:
                                print(f"  ⚠️ 缓存加载失败：{e}")
                                hist_df = None

                    if hist_df is not None and not hist_df.empty:
                        print(f"  ✓ 历史数据：{len(hist_df)} 条")
                        spot_data['history'] = hist_df
                    else:
                        print(f"  ⚠️ 历史数据获取失败")
                        spot_data['history'] = None
            else:
                # 实时行情获取失败，尝试加载缓存数据
                print(f"  ✗ 获取失败，尝试加载缓存...")

                # 港股需要补齐前导零
                if is_hk:
                    cache_code = code.zfill(5)  # 港股代码补齐 5 位，如 03690
                    cache_file = os.path.join(self.data_dir, f"{cache_code}.HK_ak.csv")
                else:
                    cache_file = os.path.join(self.data_dir, f"{symbol}_em.csv")

                if os.path.exists(cache_file):
                    try:
                        hist_df = pd.read_csv(cache_file)
                        hist_df['date'] = pd.to_datetime(hist_df['date'])
                        latest = hist_df.iloc[-1]
                        prev_close = float(hist_df.iloc[-2]['close']) if len(hist_df) > 1 else float(latest['close'])
                        results[symbol] = {
                            'name': name,
                            'symbol': symbol,
                            'price': float(latest['close']),
                            'open': float(latest['open']),
                            'high': float(latest['high']),
                            'low': float(latest['low']),
                            'close': prev_close,
                            'volume': float(latest['volume']),
                            'change_pct': ((latest['close'] - prev_close) / prev_close * 100) if prev_close else 0,
                            'date': str(latest['date']),
                            'time': '15:00' if not is_hk else '16:00',
                            'timestamp': datetime.now().isoformat(),
                            'history': hist_df
                        }
                        print(f"  ✓ 从缓存加载：{latest['close']}")
                    except Exception as e:
                        print(f"  ✗ 缓存加载失败：{e}")
                        results[symbol] = {'name': name, 'price': 0, 'history': None}
                else:
                    print(f"  ✗ 缓存文件不存在：{cache_file}")
                    results[symbol] = {'name': name, 'price': 0, 'history': None}

            # 避免请求过快
            if i < len(watchlist) - 1:
                time.sleep(0.5)

        return results


def test_data():
    """测试数据获取"""
    handler = StockDataHandler()

    # 测试股票池
    watchlist = [
        {'symbol': '300015.SZ', 'name': '爱尔眼科'},
        {'symbol': '300124.SZ', 'name': '汇川技术'},
        {'symbol': '600048.SH', 'name': '保利发展'},
    ]

    print("=" * 60)
    print("测试 Sina/Eastmoney 数据获取")
    print("=" * 60)

    results = handler.get_watchlist_data(watchlist)

    print("\n" + "=" * 60)
    print("数据汇总")
    print("=" * 60)

    for symbol, data in results.items():
        print(f"\n{data.get('name', symbol)} ({symbol})")
        print(f"  最新价：{data['price']:.2f}")
        print(f"  昨收价：{data['close']:.2f}")
        print(f"  涨跌幅：{data['change_pct']:.2f}%")
        print(f"  时间：{data['date']} {data['time']}")

    return results


if __name__ == "__main__":
    test_data()
