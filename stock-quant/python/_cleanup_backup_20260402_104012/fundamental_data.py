#!/usr/bin/env python3
"""
基本面数据收集
获取股票的 PE、PB、ROE 等基本面指标
"""

import os
import sys
import pandas as pd
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("警告：akshare 未安装")


class FundamentalDataHandler:
    """基本面数据处理器"""

    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), '../data/fundamental_cache')

        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def fetch_stock_fundamental(self, symbol: str, retry: int = 3) -> Optional[Dict]:
        """
        获取股票基本面数据

        Args:
            symbol: 股票代码（如 '300015'）
            retry: 重试次数

        Returns:
            基本面数据字典
        """
        if not AKSHARE_AVAILABLE:
            return None

        import time

        for attempt in range(retry):
            try:
                # 方法 1：获取个股实时行情（包含基本面数据）
                df = ak.stock_zh_a_spot_em()

                if df.empty:
                    return None

                # 查找指定股票
                stock_data = df[df['代码'] == symbol]

                if stock_data.empty:
                    return None

                row = stock_data.iloc[0]

                return {
                    'symbol': symbol,
                    'update_date': datetime.now().strftime('%Y-%m-%d'),
                    # 估值指标
                    'pe_ttm': float(row.get('市盈率 - 动态', 0) or 0),
                    'pe_static': float(row.get('市盈率 - 静态', 0) or 0),
                    'pb': float(row.get('市净率', 0) or 0),
                    'ps_ttm': float(row.get('市销率', 0) or 0),
                    # 盈利能力（从财务指标获取）
                    'roe': float(row.get('净资产收益率', 0) or 0),
                    # 每股指标
                    'eps': float(row.get('每股收益', 0) or 0),
                    'bvps': float(row.get('每股净资产', 0) or 0),
                }

            except Exception as e:
                if attempt < retry - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    print(f"获取基本面数据失败 {symbol}: {e}")
                    return None

        return None

    def fetch_industry_comparison(self, symbol: str) -> Optional[Dict]:
        """
        获取行业对比数据

        Args:
            symbol: 股票代码

        Returns:
            行业对比数据
        """
        if not AKSHARE_AVAILABLE:
            return None

        try:
            # 获取行业板块数据
            df = ak.stock_board_industry_name_em()

            if df.empty:
                return None

            # 这里简化处理，返回市场平均值作为参考
            return {
                'market_avg_pe': 25.0,  # 市场平均 PE
                'market_avg_pb': 3.0,   # 市场平均 PB
                'market_avg_roe': 10.0, # 市场平均 ROE
            }

        except Exception as e:
            return None

    def get_cached_fundamental(self, symbol: str, days: int = 7) -> Optional[Dict]:
        """
        获取缓存的基本面数据

        Args:
            symbol: 股票代码
            days: 缓存有效期（天）

        Returns:
            缓存数据或 None
        """
        cache_file = os.path.join(self.cache_dir, f"{symbol}.pkl")

        if not os.path.exists(cache_file):
            return None

        try:
            # 检查文件修改时间
            mtime = os.path.getmtime(cache_file)
            age_days = (datetime.now().timestamp() - mtime) / (24 * 3600)

            if age_days > days:
                return None  # 缓存过期

            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data

        except Exception:
            return None

    def save_fundamental(self, symbol: str, data: Dict):
        """保存基本面数据到缓存"""
        cache_file = os.path.join(self.cache_dir, f"{symbol}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

    def fetch_and_cache(self, symbol: str, force_refresh: bool = False) -> Optional[Dict]:
        """
        获取并缓存基本面数据

        Args:
            symbol: 股票代码
            force_refresh: 强制刷新

        Returns:
            基本面数据
        """
        # 尝试从缓存获取
        if not force_refresh:
            cached = self.get_cached_fundamental(symbol, days=7)
            if cached:
                return cached

        # 获取新数据
        data = self.fetch_stock_fundamental(symbol)

        if data:
            self.save_fundamental(symbol, data)
            return data

        # 返回缓存（即使过期）
        return self.get_cached_fundamental(symbol, days=30)


def calculate_fundamental_features(fundamental: Dict) -> Dict:
    """
    从基本面数据计算特征

    Args:
        fundamental: 基本面数据字典

    Returns:
        特征字典
    """
    if not fundamental:
        return {}

    features = {}

    # 原始估值指标
    features['pe_ttm'] = fundamental.get('pe_ttm', 0)
    features['pb'] = fundamental.get('pb', 0)
    features['ps_ttm'] = fundamental.get('ps_ttm', 0)

    # 盈利能力
    features['roe'] = fundamental.get('roe', 0)
    features['roa'] = fundamental.get('roa', 0)
    features['gross_margin'] = fundamental.get('gross_margin', 0)
    features['net_margin'] = fundamental.get('net_margin', 0)

    # 成长能力
    features['revenue_growth'] = fundamental.get('revenue_growth', 0)
    features['profit_growth'] = fundamental.get('profit_growth', 0)

    # 偿债能力
    features['debt_ratio'] = fundamental.get('debt_ratio', 0)
    features['current_ratio'] = fundamental.get('current_ratio', 0)

    # 相对估值（与市场平均比较）
    market_pe = 25.0
    market_pb = 3.0
    market_roe = 10.0

    if market_pe > 0 and features['pe_ttm'] > 0:
        features['pe_relative'] = features['pe_ttm'] / market_pe
    else:
        features['pe_relative'] = 1.0

    if market_pb > 0 and features['pb'] > 0:
        features['pb_relative'] = features['pb'] / market_pb
    else:
        features['pb_relative'] = 1.0

    features['roe_relative'] = features['roe'] / market_roe if market_roe > 0 else 1.0

    # PEG 指标（PE / 盈利增长率）
    if features['profit_growth'] > 0:
        features['peg'] = features['pe_ttm'] / features['profit_growth']
    else:
        features['peg'] = 999  # 负增长时设为极大值

    return features


def collect_all_fundamentals(symbols: List[str]) -> pd.DataFrame:
    """
    批量收集基本面数据

    Args:
        symbols: 股票代码列表

    Returns:
        DataFrame 包含所有股票的基本面特征
    """
    handler = FundamentalDataHandler()
    all_features = []

    print(f"开始收集 {len(symbols)} 只股票的基本面数据...")

    for i, symbol in enumerate(symbols):
        if (i + 1) % 10 == 0:
            print(f"[{i + 1}/{len(symbols)}] 获取 {symbol}...")

        data = handler.fetch_and_cache(symbol)

        if data:
            features = calculate_fundamental_features(data)
            features['symbol'] = symbol
            all_features.append(features)
        else:
            print(f"  ✗ {symbol} 获取失败")

    df = pd.DataFrame(all_features)
    print(f"完成！成功 {len(df)}/{len(symbols)} 只股票")

    return df


if __name__ == "__main__":
    # 测试
    test_symbols = ['300015', '300124', '600048', '600519']

    handler = FundamentalDataHandler()

    print("=" * 60)
    print("基本面数据测试")
    print("=" * 60)

    for symbol in test_symbols:
        print(f"\n{symbol}:")
        data = handler.fetch_and_cache(symbol)
        if data:
            features = calculate_fundamental_features(data)
            print(f"  PE(TTM): {features.get('pe_ttm', 'N/A')}")
            print(f"  PB: {features.get('pb', 'N/A')}")
            print(f"  ROE: {features.get('roe', 'N/A')}%")
            print(f"  营收增长率：{features.get('revenue_growth', 'N/A')}%")
        else:
            print("  获取失败")
