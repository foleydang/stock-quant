#!/usr/bin/env python3
"""
宏观特征模块 v2 — 为模型提供市场环境上下文

设计原则:
  - 宏观特征对全市场所有股票相同 (market-wide)
  - 计算一次, 所有股票复用 (通过 FeaturePipeline 缓存)
  - 包含宏观×个股交互特征 (如 beta, 弹性)

特征类别:
  1. 利率环境 (SHIBOR, 国债收益率, 期限利差)
  2. 汇率环境 (USD/CNY, 变化率, 波动率)
  3. 市场状态 (沪深300收益/波动/成交量, 市场广度)
  4. 宏观×个股交互 (beta, 相关性, 相对大盘alpha)
"""

import numpy as np
import pandas as pd
import sqlite3
import os
from typing import Dict, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/stock_data.db')


class MacroFeatures:
    """宏观特征计算器"""

    # 缓存: 避免重复读取数据库
    _cache = None
    _cache_conn = None

    @classmethod
    def load_macro_data(cls) -> pd.DataFrame:
        """从数据库加载宏观数据 (带缓存)"""
        if cls._cache is not None:
            return cls._cache

        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql("SELECT * FROM macro_daily ORDER BY trade_date", conn)
            conn.close()
            if len(df) > 0:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.set_index('trade_date').sort_index()
                cls._cache = df
                return df
        except Exception:
            pass
        return pd.DataFrame()

    @classmethod
    def calculate(cls, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """计算宏观特征

        Args:
            df: 股票日线数据 (需要 date, close, open, high, low 列)
            symbol: 股票代码

        Returns:
            DataFrame with macro features (index 与 df 对齐)
        """
        f = pd.DataFrame(index=df.index)

        if 'date' not in df.columns:
            return f

        macro = cls.load_macro_data()
        if len(macro) == 0:
            return f

        df_dates = pd.to_datetime(df['date'])

        # ====== 1. 利率环境 ======
        if 'shibor_on' in macro.columns:
            f['macro_shibor_on'] = cls._map_series(df_dates, macro['shibor_on'])
            f['macro_shibor_on_chg'] = f['macro_shibor_on'].diff(5)
            f['macro_shibor_on_ma10'] = f['macro_shibor_on'].rolling(10, min_periods=1).mean()
            f['macro_shibor_on_vs_ma'] = f['macro_shibor_on'] / (f['macro_shibor_on_ma10'] + 1e-10)

            # 流动性紧张信号 (隔夜利率 > 2.5%)
            f['macro_liquidity_tight'] = (f['macro_shibor_on'] > 2.5).astype(int)

        if 'shibor_1w' in macro.columns:
            f['macro_shibor_1w'] = cls._map_series(df_dates, macro['shibor_1w'])
            f['macro_shibor_1w_chg'] = f['macro_shibor_1w'].diff(5)

        if 'shibor_1m' in macro.columns:
            f['macro_shibor_1m'] = cls._map_series(df_dates, macro['shibor_1m'])
        if 'shibor_3m' in macro.columns:
            f['macro_shibor_3m'] = cls._map_series(df_dates, macro['shibor_3m'])

        # 期限利差
        if 'shibor_1m' in macro.columns and 'shibor_on' in macro.columns:
            f['macro_shibor_spread'] = f['macro_shibor_1m'] - f['macro_shibor_on']
            f['macro_shibor_spread_chg'] = f['macro_shibor_spread'].diff(10)

        # 国债收益率
        if 'cn_10y_yield' in macro.columns:
            f['macro_cn_10y'] = cls._map_series(df_dates, macro['cn_10y_yield'])
            f['macro_cn_10y_chg'] = f['macro_cn_10y'].diff(10)
            f['macro_cn_10y_ma20'] = f['macro_cn_10y'].rolling(20, min_periods=1).mean()
            f['macro_cn_10y_vs_ma'] = f['macro_cn_10y'] / (f['macro_cn_10y_ma20'] + 1e-10)
            # 利率方向 (上升=紧缩, 下降=宽松)
            f['macro_yield_trend'] = (f['macro_cn_10y'] > f['macro_cn_10y_ma20']).astype(int)

        # ====== 2. 汇率环境 ======
        if 'usdcny' in macro.columns:
            f['macro_usdcny'] = cls._map_series(df_dates, macro['usdcny'])
            f['macro_usdcny_chg'] = f['macro_usdcny'].pct_change(5)  # 正=人民币贬值
            f['macro_usdcny_ma20'] = f['macro_usdcny'].rolling(20, min_periods=1).mean()
            f['macro_usdcny_vs_ma'] = f['macro_usdcny'] / (f['macro_usdcny_ma20'] + 1e-10)
            # 汇率波动率
            f['macro_usdcny_vol'] = f['macro_usdcny'].pct_change().rolling(20, min_periods=1).std()
            # 升值/贬值信号
            f['macro_cny_appreciate'] = (f['macro_usdcny_chg'] < -0.002).astype(int)  # 5日升值>0.2%
            f['macro_cny_depreciate'] = (f['macro_usdcny_chg'] > 0.005).astype(int)  # 5日贬值>0.5%

        # ====== 3. 市场状态 (沪深300) ======
        if 'hs300_close' in macro.columns:
            f['macro_hs300_close'] = cls._map_series(df_dates, macro['hs300_close'])
            hs300_ret = f['macro_hs300_close'].pct_change()

            # 市场收益
            f['macro_hs300_ret_5'] = hs300_ret.rolling(5, min_periods=1).sum()
            f['macro_hs300_ret_20'] = hs300_ret.rolling(20, min_periods=1).sum()
            f['macro_hs300_ret_60'] = hs300_ret.rolling(60, min_periods=1).sum()

            # 市场波动率
            f['macro_hs300_vol_5'] = hs300_ret.rolling(5, min_periods=1).std()
            f['macro_hs300_vol_20'] = hs300_ret.rolling(20, min_periods=1).std()
            f['macro_hs300_vol_60'] = hs300_ret.rolling(60, min_periods=1).std()

            # 波动率状态 (高波动 vs 低波动)
            f['macro_hs300_vol_regime'] = f['macro_hs300_vol_20'] / (f['macro_hs300_vol_60'] + 1e-10)
            f['macro_hs300_vol_high'] = (f['macro_hs300_vol_regime'] > 1.3).astype(int)

            # 市场趋势
            f['macro_hs300_above_ma20'] = (f['macro_hs300_close'] > f['macro_hs300_close'].rolling(20).mean()).astype(int)
            f['macro_hs300_above_ma60'] = (f['macro_hs300_close'] > f['macro_hs300_close'].rolling(60).mean()).astype(int)
            f['macro_hs300_above_ma200'] = (f['macro_hs300_close'] > f['macro_hs300_close'].rolling(200).mean()).astype(int)

            # 市场状态分类: 0=熊, 1=震荡, 2=牛
            f['macro_market_regime'] = (
                f['macro_hs300_above_ma20'] + f['macro_hs300_above_ma60'] + f['macro_hs300_above_ma200']
            )

        if 'hs300_volume' in macro.columns:
            f['macro_hs300_volume'] = cls._map_series(df_dates, macro['hs300_volume'])
            f['macro_hs300_vol_ma20'] = f['macro_hs300_volume'].rolling(20, min_periods=1).mean()
            f['macro_hs300_vol_ratio'] = f['macro_hs300_volume'] / (f['macro_hs300_vol_ma20'] + 1e-10)
            f['macro_mkt_active'] = (f['macro_hs300_vol_ratio'] > 1.2).astype(int)

        # 市场广度
        if 'market_breadth' in macro.columns:
            f['macro_breadth'] = cls._map_series(df_dates, macro['market_breadth'])
            f['macro_breadth_ma5'] = f['macro_breadth'].rolling(5, min_periods=1).mean()
            f['macro_breadth_chg'] = f['macro_breadth'].diff(5)
            f['macro_breadth_strong'] = (f['macro_breadth_ma5'] > 0.55).astype(int)

        # ====== 4. 宏观×个股交互 ======
        if 'close' in df.columns and 'hs300_close' in macro.columns:
            close = df['close'].astype(float)
            stock_ret = close.pct_change()
            hs300_ret = f['macro_hs300_close'].pct_change()

            # 滚动 beta (60日)
            cov = stock_ret.rolling(60, min_periods=20).cov(hs300_ret)
            var = hs300_ret.rolling(60, min_periods=20).var()
            f['macro_stock_beta'] = cov / (var + 1e-10)

            # 滚动相关性
            f['macro_stock_corr'] = stock_ret.rolling(60, min_periods=20).corr(hs300_ret)

            # 个股相对大盘 alpha
            f['macro_stock_alpha_5'] = (stock_ret - hs300_ret).rolling(5, min_periods=1).sum()
            f['macro_stock_alpha_20'] = (stock_ret - hs300_ret).rolling(20, min_periods=1).sum()

            # 相对强弱 (RS)
            rs = (1 + stock_ret).rolling(20, min_periods=1).apply(lambda x: np.prod(1 + x) - 1, raw=True)
            rs_mkt = (1 + hs300_ret).rolling(20, min_periods=1).apply(lambda x: np.prod(1 + x) - 1, raw=True)
            f['macro_stock_rs'] = rs - rs_mkt

            # 个股在高低波动环境下的表现差异
            f['macro_stock_vol_elastic'] = stock_ret.rolling(60, min_periods=20).std() / (
                hs300_ret.rolling(60, min_periods=20).std() + 1e-10
            )

        # 填充NaN
        f = f.fillna(method='ffill').fillna(0)

        return f

    @staticmethod
    def _map_series(df_dates, macro_series):
        """将宏观数据映射到股票日期, 前向填充"""
        result = pd.Series(np.nan, index=range(len(df_dates)))
        macro_dates = macro_series.index
        macro_values = macro_series.values

        for i, d in enumerate(df_dates):
            mask = macro_dates <= d
            if mask.any():
                result.iloc[i] = macro_values[np.argmax(mask)]

        return result.fillna(method='ffill')