#!/usr/bin/env python3
"""
基本面特征工程
将基本面数据融入 LightGBM 预测模型
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Optional
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 默认基本面数据（用于无法获取实时数据时）
DEFAULT_FUNDAMENTAL = {
    # 估值指标
    'pe_ttm': 20.0,
    'pb': 2.5,
    'ps_ttm': 3.0,
    # 盈利能力
    'roe': 10.0,
    'roa': 5.0,
    'gross_margin': 30.0,
    'net_margin': 15.0,
    # 成长能力
    'revenue_growth': 10.0,
    'profit_growth': 15.0,
    # 偿债能力
    'debt_ratio': 50.0,
    'current_ratio': 2.0,
}


class FundamentalFeatureEngineer:
    """基本面特征工程类"""

    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), '../data/fundamental_cache')

        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def get_fundamental_data(self, symbol: str) -> Dict:
        """
        获取股票基本面数据（优先从缓存）

        Args:
            symbol: 股票代码

        Returns:
            基本面数据字典
        """
        cache_file = os.path.join(self.cache_dir, f"{symbol}.pkl")

        # 尝试从缓存加载
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                # 检查缓存是否过期（7 天）
                mtime = os.path.getmtime(cache_file)
                age_days = (datetime.now().timestamp() - mtime) / (24 * 3600)
                if age_days <= 7:
                    return data
            except:
                pass

        # 使用默认数据
        return {**DEFAULT_FUNDAMENTAL, 'symbol': symbol, 'update_date': datetime.now().strftime('%Y-%m-%d')}

    def calculate_features(self, fundamental: Dict) -> Dict:
        """
        从基本面数据计算特征

        Args:
            fundamental: 基本面数据

        Returns:
            特征字典
        """
        features = {}

        # === 原始估值指标 ===
        features['pe_ttm'] = fundamental.get('pe_ttm', 20.0)
        features['pb'] = fundamental.get('pb', 2.5)
        features['ps_ttm'] = fundamental.get('ps_ttm', 3.0)

        # === 盈利能力 ===
        features['roe'] = fundamental.get('roe', 10.0)
        features['roa'] = fundamental.get('roa', 5.0)
        features['gross_margin'] = fundamental.get('gross_margin', 30.0)
        features['net_margin'] = fundamental.get('net_margin', 15.0)

        # === 成长能力 ===
        features['revenue_growth'] = fundamental.get('revenue_growth', 10.0)
        features['profit_growth'] = fundamental.get('profit_growth', 15.0)

        # === 偿债能力 ===
        features['debt_ratio'] = fundamental.get('debt_ratio', 50.0)
        features['current_ratio'] = fundamental.get('current_ratio', 2.0)

        # === 相对估值特征 ===
        # 市场平均值（可配置）
        market_pe = 25.0
        market_pb = 3.0
        market_roe = 10.0
        market_growth = 15.0

        # PE 相对值（<1 表示低估，>1 表示高估）
        if features['pe_ttm'] > 0:
            features['pe_relative'] = features['pe_ttm'] / market_pe
        else:
            features['pe_relative'] = 1.0

        # PB 相对值
        if features['pb'] > 0:
            features['pb_relative'] = features['pb'] / market_pb
        else:
            features['pb_relative'] = 1.0

        # ROE 相对值（>1 表示优于市场平均）
        features['roe_relative'] = features['roe'] / market_roe if market_roe > 0 else 1.0

        # 成长相对值
        features['growth_relative'] = features['profit_growth'] / market_growth if market_growth > 0 else 1.0

        # === PEG 指标 ===
        # PEG = PE / 盈利增长率，<1 表示低估
        if features['profit_growth'] > 0:
            features['peg'] = features['pe_ttm'] / features['profit_growth']
        else:
            features['peg'] = 999  # 负增长时设为极大值

        # === 综合评分 ===
        # 价值评分（低 PE、低 PB 得分高）
        value_score = 0
        if 0 < features['pe_ttm'] < 15:
            value_score += 1
        elif 0 < features['pe_ttm'] < 25:
            value_score += 0.5
        if 0 < features['pb'] < 2:
            value_score += 1
        elif 0 < features['pb'] < 4:
            value_score += 0.5
        features['value_score'] = value_score  # 0-2 分

        # 质量评分（高 ROE、高利润率得分高）
        quality_score = 0
        if features['roe'] > 15:
            quality_score += 1
        elif features['roe'] > 8:
            quality_score += 0.5
        if features['net_margin'] > 20:
            quality_score += 1
        elif features['net_margin'] > 10:
            quality_score += 0.5
        features['quality_score'] = quality_score  # 0-2 分

        # 成长评分（高增长得分高）
        growth_score = 0
        if features['profit_growth'] > 30:
            growth_score += 1
        elif features['profit_growth'] > 15:
            growth_score += 0.5
        if features['revenue_growth'] > 20:
            growth_score += 1
        elif features['revenue_growth'] > 10:
            growth_score += 0.5
        features['growth_score'] = growth_score  # 0-2 分

        return features

    def merge_with_technical(self, technical_features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        将基本面特征与技术面特征合并

        Args:
            technical_features: 技术面特征 DataFrame
            symbol: 股票代码

        Returns:
            合并后的特征 DataFrame
        """
        # 获取基本面特征
        fundamental = self.get_fundamental_data(symbol)
        fund_features = self.calculate_features(fundamental)

        # 将基本面特征添加到每一行
        for key, value in fund_features.items():
            technical_features[f'fund_{key}'] = value

        return technical_features


def add_fundamental_features_to_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    为股票数据添加基本面特征

    Args:
        df: OHLCV 数据
        symbol: 股票代码

    Returns:
        包含基本面特征的数据
    """
    engineer = FundamentalFeatureEngineer()
    fundamental = engineer.get_fundamental_data(symbol)
    fund_features = engineer.calculate_features(fundamental)

    # 为每一行添加相同的基本面特征
    for key, value in fund_features.items():
        df[f'fund_{key}'] = value

    return df


# 测试
if __name__ == "__main__":
    engineer = FundamentalFeatureEngineer()

    print("=" * 60)
    print("基本面特征工程测试")
    print("=" * 60)

    test_symbols = ['300015', '300124', '600048', '600519']

    for symbol in test_symbols:
        print(f"\n{symbol}:")
        fundamental = engineer.get_fundamental_data(symbol)
        features = engineer.calculate_features(fundamental)

        # 打印关键特征
        print(f"  PE(TTM): {features['pe_ttm']:.2f}")
        print(f"  PB: {features['pb']:.2f}")
        print(f"  ROE: {features['roe']:.2f}%")
        print(f"  PEG: {features['peg']:.2f}")
        print(f"  价值评分：{features['value_score']}")
        print(f"  质量评分：{features['quality_score']}")
        print(f"  成长评分：{features['growth_score']}")
