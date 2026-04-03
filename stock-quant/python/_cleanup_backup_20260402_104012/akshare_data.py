#!/usr/bin/env python3
"""
使用 akshare 获取 A 股实时数据
支持实时行情、历史行情、港股数据
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("警告：akshare 未安装，请运行：pip install akshare")


class AkshareDataHandler:
    """使用 akshare 的数据处理器"""

    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), '../data')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def fetch_stock_spot(self, symbol: str) -> Optional[Dict]:
        """
        获取股票实时行情

        Args:
            symbol: 股票代码，如 '300015'

        Returns:
            包含实时行情数据的字典
        """
        if not AKSHARE_AVAILABLE:
            return None

        try:
            # 获取实时行情（东方财富接口）
            df = ak.stock_zh_a_spot_em()

            # 查找指定股票
            stock_data = df[df['代码'] == symbol]

            if stock_data.empty:
                return None

            row = stock_data.iloc[0]

            return {
                'symbol': symbol,
                'name': row.get('名称', ''),
                'price': float(row.get('最新价', 0)),
                'open': float(row.get('今开', 0)),
                'high': float(row.get('最高', 0)),
                'low': float(row.get('最低', 0)),
                'close': float(row.get('昨收', 0)),
                'volume': float(row.get('成交量', 0)),
                'amount': float(row.get('成交额', 0)),
                'change': float(row.get('涨跌额', 0)),
                'change_pct': float(row.get('涨跌幅', 0)),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            sys.stderr.write(f"获取实时行情失败 {symbol}: {e}\n")
            return None

    def fetch_stock_history(self, symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
        """
        获取股票历史行情

        Args:
            symbol: 股票代码，如 '300015'
            days: 获取天数

        Returns:
            DataFrame 包含历史数据
        """
        if not AKSHARE_AVAILABLE:
            return None

        try:
            # 计算开始日期
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            start_str = start_date.strftime('%Y%m%d')

            # 获取历史数据（前复权）
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period='daily',
                start_date=start_str,
                adjust='qfq'
            )

            if df.empty:
                return None

            # 重命名列以匹配现有格式
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount'
            })

            df['date'] = pd.to_datetime(df['date'])
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            df = df.sort_values('date')
            df = df.reset_index(drop=True)

            # 保存到本地
            file_path = os.path.join(self.data_dir, f"{symbol}_akshare.csv")
            df.to_csv(file_path, index=False)

            return df

        except Exception as e:
            sys.stderr.write(f"获取历史数据失败 {symbol}: {e}\n")
            return None

    def fetch_hk_stock(self, symbol: str) -> Optional[Dict]:
        """
        获取港股实时行情

        Args:
            symbol: 港股代码，如 '03690'

        Returns:
            包含实时行情数据的字典
        """
        if not AKSHARE_AVAILABLE:
            return None

        try:
            # 获取港股实时行情
            df = ak.stock_hk_daily_em(symbol=symbol)

            if df.empty:
                return None

            row = df.iloc[-1]

            return {
                'symbol': symbol,
                'price': float(row.get('最新价', 0)),
                'open': float(row.get('开盘价', 0)),
                'high': float(row.get('最高价', 0)),
                'low': float(row.get('最低价', 0)),
                'close': float(row.get('昨收价', 0)),
                'volume': float(row.get('成交量', 0)),
                'change': float(row.get('涨跌额', 0)),
                'change_pct': float(row.get('涨跌幅', 0)),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            sys.stderr.write(f"获取港股数据失败 {symbol}: {e}\n")
            return None

    def get_watchlist_data(self, watchlist: List[Dict]) -> Dict[str, Dict]:
        """
        批量获取股票池数据

        Args:
            watchlist: 股票列表 [{'symbol': '300015', 'name': '爱尔眼科'}, ...]

        Returns:
            股票数据字典
        """
        results = {}

        for stock in watchlist:
            symbol = stock['symbol']
            name = stock.get('name', '')

            print(f"获取数据：{name} ({symbol})...")

            # A 股
            if symbol.endswith('.SZ') or symbol.endswith('.SH'):
                symbol_code = symbol[:6]  # 去掉后缀

                # 获取实时行情
                spot_data = self.fetch_stock_spot(symbol_code)
                if spot_data:
                    results[symbol] = spot_data
                    print(f"  ✓ 实时价格：{spot_data['price']:.2f}")

                # 获取历史数据
                hist_data = self.fetch_stock_history(symbol_code)
                if hist_data is not None:
                    print(f"  ✓ 历史数据：{len(hist_data)} 条")

            # 港股
            elif symbol.endswith('.HK'):
                # 去掉 .HK 后缀，转换为港股代码格式
                hk_code = symbol.replace('.HK', '')
                if not hk_code.startswith('0'):
                    hk_code = '0' + hk_code

                hk_data = self.fetch_hk_stock(hk_code)
                if hk_data:
                    results[symbol] = hk_data
                    print(f"  ✓ 实时价格：{hk_data['price']:.2f}")

        return results


def test_data():
    """测试数据获取"""
    handler = AkshareDataHandler()

    # 测试股票池
    watchlist = [
        {'symbol': '300015.SZ', 'name': '爱尔眼科'},
        {'symbol': '300124.SZ', 'name': '汇川技术'},
        {'symbol': '600048.SH', 'name': '保利发展'},
    ]

    print("=" * 60)
    print("测试 akshare 数据获取")
    print("=" * 60)

    results = handler.get_watchlist_data(watchlist)

    print("\n" + "=" * 60)
    print("数据汇总")
    print("=" * 60)

    for symbol, data in results.items():
        print(f"\n{data.get('name', symbol)} ({symbol})")
        print(f"  最新价：{data.get('price', 'N/A')}")
        print(f"  涨跌幅：{data.get('change_pct', 'N/A')}%")
        print(f"  成交量：{data.get('volume', 'N/A')}")
        print(f"  时间：{data.get('timestamp', 'N/A')}")

    return results


if __name__ == "__main__":
    test_data()
