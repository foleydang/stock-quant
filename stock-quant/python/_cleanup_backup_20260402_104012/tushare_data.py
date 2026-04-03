#!/usr/bin/env python3
"""
使用 Tushare 获取 A 股实时数据
Tushare 是一个免费稳定的 A 股数据源
https://tushare.pro/
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False
    print("警告：tushare 未安装，请运行：pip install tushare")


class TushareDataHandler:
    """使用 tushare 的数据处理器"""

    def __init__(self, token: str = None):
        """
        初始化

        Args:
            token: Tushare Pro token（可选，用于高级接口）
        """
        self.data_dir = os.path.join(os.path.dirname(__file__), '../data')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # 如果有 token，设置 pro 接口
        if token:
            ts.set_token(token)
            self.pro = ts.pro_api()
        else:
            self.pro = None

    def fetch_spot(self, symbol: str) -> Optional[Dict]:
        """
        获取股票实时行情（使用经典接口，无需 token）

        Args:
            symbol: 股票代码，如 '300015'（不含后缀）

        Returns:
            包含实时行情数据的字典
        """
        if not TUSHARE_AVAILABLE:
            return None

        try:
            # 使用经典接口获取实时行情
            df = ts.get_realtime_quotes(symbol)

            if df.empty:
                return None

            row = df.iloc[0]

            # 获取历史数据用于策略计算
            hist_df = self.fetch_history(symbol)

            return {
                'symbol': symbol,
                'name': row.get('name', ''),
                'price': float(row.get('price', 0)),
                'open': float(row.get('open', 0)),
                'high': float(row.get('high', 0)),
                'low': float(row.get('low', 0)),
                'close': float(row.get('pre_close', 0)),  # 昨收
                'volume': float(row.get('volume', 0)),
                'amount': float(row.get('amount', 0)),
                'bid': float(row.get('bid', 0)),
                'ask': float(row.get('ask', 0)),
                'bid1_volume': int(row.get('b1_v', 0)),
                'bid1_price': float(row.get('b1_p', 0)),
                'ask1_volume': int(row.get('a1_v', 0)),
                'ask1_price': float(row.get('a1_p', 0)),
                'time': row.get('time', ''),
                'date': row.get('date', ''),
                'timestamp': datetime.now().isoformat(),
                'history': hist_df  # 附加历史数据
            }

        except Exception as e:
            sys.stderr.write(f"获取实时行情失败 {symbol}: {e}\n")
            return None

    def fetch_history(self, symbol: str, days: int = 90) -> Optional[pd.DataFrame]:
        """
        获取股票历史行情

        Args:
            symbol: 股票代码
            days: 获取天数

        Returns:
            DataFrame 包含历史数据
        """
        if not TUSHARE_AVAILABLE:
            return None

        try:
            # 计算日期范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # 使用经典接口获取历史数据
            df = ts.get_hist_data(
                code=symbol,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            )

            if df is None or df.empty:
                return None

            df = df.reset_index()
            df = df.rename(columns={'date': 'date', 'index': 'date'})

            # 重命名列以匹配现有格式
            df = df.rename(columns={
                'open': 'open',
                'close': 'close',
                'high': 'high',
                'low': 'low',
                'volume': 'volume',
            })

            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')

            # 保存到本地
            file_path = os.path.join(self.data_dir, f"{symbol}_tushare.csv")
            df.to_csv(file_path, index=False)

            return df

        except Exception as e:
            sys.stderr.write(f"获取历史数据失败 {symbol}: {e}\n")
            return None

    def get_watchlist_data(self, watchlist: List[Dict]) -> Dict:
        """
        批量获取股票池数据

        Args:
            watchlist: 股票列表 [{'symbol': '300015.SZ', 'name': '爱尔眼科'}, ...]

        Returns:
            股票数据字典
        """
        results = {}

        for stock in watchlist:
            symbol = stock['symbol']
            name = stock.get('name', '')

            # 去掉后缀获取代码
            if symbol.endswith('.SZ') or symbol.endswith('.SH'):
                code = symbol[:6]
            else:
                code = symbol

            print(f"获取数据：{name} ({code})...")

            # 获取实时行情（包含历史数据）
            data = self.fetch_spot(code)

            if data:
                results[symbol] = data
                print(f"  ✓ 实时价格：{data['price']:.2f}, 涨跌幅：{(data['price'] - data['close']) / data['close'] * 100:.2f}%")
                if data.get('history') is not None:
                    print(f"  ✓ 历史数据：{len(data['history'])} 条")
            else:
                print(f"  ✗ 获取失败")

        return results


def test_data():
    """测试数据获取"""
    handler = TushareDataHandler()

    # 测试股票池
    watchlist = [
        {'symbol': '300015.SZ', 'name': '爱尔眼科'},
        {'symbol': '300124.SZ', 'name': '汇川技术'},
        {'symbol': '600048.SH', 'name': '保利发展'},
    ]

    print("=" * 60)
    print("测试 Tushare 数据获取")
    print("=" * 60)

    results = handler.get_watchlist_data(watchlist)

    print("\n" + "=" * 60)
    print("数据汇总")
    print("=" * 60)

    for symbol, data in results.items():
        print(f"\n{data.get('name', symbol)} ({symbol})")
        print(f"  最新价：{data['price']:.2f}")
        print(f"  昨收价：{data['close']:.2f}")
        print(f"  涨跌幅：{((data['price'] - data['close']) / data['close'] * 100):.2f}%")
        print(f"  成交量：{data['volume']}")
        print(f"  买一价：{data['bid1_price']} ({data['bid1_volume']}手)")
        print(f"  卖一价：{data['ask1_price']} ({data['ask1_volume']}手)")
        print(f"  时间：{data['date']} {data['time']}")

    return results


if __name__ == "__main__":
    test_data()
