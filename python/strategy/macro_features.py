#!/usr/bin/env python3
"""
宏观特征模块 v3 — 为模型提供市场环境上下文

数据源: macro_daily 表 (SHIBOR/国债/汇率/沪深300/中美利差)

特征类别:
  1. 利率环境 (SHIBOR, 国债收益率, 期限利差, 中美利差)
  2. 汇率环境 (USD/CNY在岸, USD/CNH离岸, 变化率, 波动率)
  3. 市场状态 (沪深300收益/波动/成交量)
  4. 宏观×个股交互 (beta, 相关性, 相对大盘alpha)
"""

import numpy as np
import pandas as pd
import sqlite3
import os
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/stock_data.db')


class MacroFeatures:
    """宏观特征计算器"""

    _cache = None

    @classmethod
    def load_macro_data(cls) -> pd.DataFrame:
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
        """计算宏观特征 (index 与 df 对齐)"""
        f = pd.DataFrame(index=df.index)
        if 'date' not in df.columns:
            return f

        macro = cls.load_macro_data()
        if len(macro) == 0:
            return f

        df_dates = pd.to_datetime(df['date'], format='mixed')

        # ====== 1. 利率环境 ======
        # SHIBOR
        if 'shibor_on' in macro.columns:
            f['macro_shibor_on'] = cls._map(df_dates, macro['shibor_on'])
            f['macro_shibor_on_chg'] = f['macro_shibor_on'].diff(5)
            f['macro_shibor_on_ma10'] = f['macro_shibor_on'].rolling(10, min_periods=1).mean()
            f['macro_shibor_on_vs_ma'] = f['macro_shibor_on'] / (f['macro_shibor_on_ma10'] + 1e-10)
            f['macro_liquidity_tight'] = (f['macro_shibor_on'] > 2.5).astype(int)

        if 'shibor_1w' in macro.columns:
            f['macro_shibor_1w'] = cls._map(df_dates, macro['shibor_1w'])
            f['macro_shibor_1w_chg'] = f['macro_shibor_1w'].diff(5)

        if 'shibor_1m' in macro.columns:
            f['macro_shibor_1m'] = cls._map(df_dates, macro['shibor_1m'])
        if 'shibor_3m' in macro.columns:
            f['macro_shibor_3m'] = cls._map(df_dates, macro['shibor_3m'])

        # 期限利差
        if 'shibor_1m' in macro.columns and 'shibor_on' in macro.columns:
            f['macro_shibor_spread'] = f['macro_shibor_1m'] - f['macro_shibor_on']
            f['macro_shibor_spread_chg'] = f['macro_shibor_spread'].diff(10)

        # 中国国债收益率
        if 'cn_10y' in macro.columns:
            f['macro_cn_10y'] = cls._map(df_dates, macro['cn_10y'])
            f['macro_cn_10y_chg'] = f['macro_cn_10y'].diff(10)
            f['macro_cn_10y_ma20'] = f['macro_cn_10y'].rolling(20, min_periods=1).mean()
            f['macro_cn_10y_vs_ma'] = f['macro_cn_10y'] / (f['macro_cn_10y_ma20'] + 1e-10)
            f['macro_yield_trend'] = (f['macro_cn_10y'] > f['macro_cn_10y_ma20']).astype(int)

        if 'cn_2y' in macro.columns:
            f['macro_cn_2y'] = cls._map(df_dates, macro['cn_2y'])
            # 国债期限利差 (10Y-2Y, 正=陡峭, 负=倒挂)
            f['macro_cn_slope'] = f['macro_cn_10y'] - f['macro_cn_2y']
            f['macro_cn_slope_chg'] = f['macro_cn_slope'].diff(20)

        # 中美利差 (v3 新增)
        if 'cn_us_spread' in macro.columns:
            f['macro_cn_us_spread'] = cls._map(df_dates, macro['cn_us_spread'])
            f['macro_cn_us_spread_chg'] = f['macro_cn_us_spread'].diff(20)
            f['macro_cn_us_spread_ma20'] = f['macro_cn_us_spread'].rolling(20, min_periods=1).mean()
            # 利差收窄/倒挂信号 (中美利差 < 0 = 资金外流压力)
            f['macro_spread_negative'] = (f['macro_cn_us_spread'] < 0).astype(int)
            f['macro_spread_narrowing'] = (f['macro_cn_us_spread_chg'] < -0.2).astype(int)

        if 'us_10y' in macro.columns:
            f['macro_us_10y'] = cls._map(df_dates, macro['us_10y'])
            f['macro_us_10y_chg'] = f['macro_us_10y'].diff(10)

        # ====== 2. 汇率环境 ======
        if 'usdcny' in macro.columns:
            f['macro_usdcny'] = cls._map(df_dates, macro['usdcny'])
            f['macro_usdcny_chg'] = f['macro_usdcny'].pct_change(5)
            f['macro_usdcny_ma20'] = f['macro_usdcny'].rolling(20, min_periods=1).mean()
            f['macro_usdcny_vs_ma'] = f['macro_usdcny'] / (f['macro_usdcny_ma20'] + 1e-10)
            f['macro_usdcny_vol'] = f['macro_usdcny'].pct_change().rolling(20, min_periods=1).std()
            f['macro_cny_appreciate'] = (f['macro_usdcny_chg'] < -0.002).astype(int)
            f['macro_cny_depreciate'] = (f['macro_usdcny_chg'] > 0.005).astype(int)

        if 'usdcnh' in macro.columns:
            f['macro_usdcnh'] = cls._map(df_dates, macro['usdcnh'])
            f['macro_usdcnh_chg'] = f['macro_usdcnh'].pct_change(5)
            # 在岸离岸价差 (正=离岸更弱, 贬值预期)
            if 'usdcny' in macro.columns:
                f['macro_cny_cnh_spread'] = f['macro_usdcnh'] - f['macro_usdcny']
                f['macro_cny_cnh_spread_wide'] = (abs(f['macro_cny_cnh_spread']) > 0.02).astype(int)

        # ====== 3. 市场状态 (沪深300) ======
        if 'hs300_close' in macro.columns:
            f['macro_hs300_close'] = cls._map(df_dates, macro['hs300_close'])
            hs300_ret = f['macro_hs300_close'].pct_change()

            f['macro_hs300_ret_5'] = hs300_ret.rolling(5, min_periods=1).sum()
            f['macro_hs300_ret_20'] = hs300_ret.rolling(20, min_periods=1).sum()
            f['macro_hs300_ret_60'] = hs300_ret.rolling(60, min_periods=1).sum()

            f['macro_hs300_vol_5'] = hs300_ret.rolling(5, min_periods=1).std()
            f['macro_hs300_vol_20'] = hs300_ret.rolling(20, min_periods=1).std()
            f['macro_hs300_vol_60'] = hs300_ret.rolling(60, min_periods=1).std()

            f['macro_hs300_vol_regime'] = f['macro_hs300_vol_20'] / (f['macro_hs300_vol_60'] + 1e-10)
            f['macro_hs300_vol_high'] = (f['macro_hs300_vol_regime'] > 1.3).astype(int)

            f['macro_hs300_above_ma20'] = (f['macro_hs300_close'] > f['macro_hs300_close'].rolling(20).mean()).astype(int)
            f['macro_hs300_above_ma60'] = (f['macro_hs300_close'] > f['macro_hs300_close'].rolling(60).mean()).astype(int)
            f['macro_hs300_above_ma200'] = (f['macro_hs300_close'] > f['macro_hs300_close'].rolling(200).mean()).astype(int)

            f['macro_market_regime'] = (
                f['macro_hs300_above_ma20'] + f['macro_hs300_above_ma60'] + f['macro_hs300_above_ma200']
            )

        if 'hs300_volume' in macro.columns:
            f['macro_hs300_volume'] = cls._map(df_dates, macro['hs300_volume'])
            f['macro_hs300_vol_ma20'] = f['macro_hs300_volume'].rolling(20, min_periods=1).mean()
            f['macro_hs300_vol_ratio'] = f['macro_hs300_volume'] / (f['macro_hs300_vol_ma20'] + 1e-10)
            f['macro_mkt_active'] = (f['macro_hs300_vol_ratio'] > 1.2).astype(int)

        # ====== 4. 宏观×个股交互 ======
        if 'close' in df.columns and 'hs300_close' in macro.columns:
            close = df['close'].astype(float)
            stock_ret = close.pct_change()
            hs300_ret = f['macro_hs300_close'].pct_change()

            cov = stock_ret.rolling(60, min_periods=20).cov(hs300_ret)
            var = hs300_ret.rolling(60, min_periods=20).var()
            f['macro_stock_beta'] = cov / (var + 1e-10)

            f['macro_stock_corr'] = stock_ret.rolling(60, min_periods=20).corr(hs300_ret)
            f['macro_stock_alpha_5'] = (stock_ret - hs300_ret).rolling(5, min_periods=1).sum()
            f['macro_stock_alpha_20'] = (stock_ret - hs300_ret).rolling(20, min_periods=1).sum()

            rs = (1 + stock_ret).rolling(20, min_periods=1).apply(lambda x: np.prod(1 + x) - 1, raw=True)
            rs_mkt = (1 + hs300_ret).rolling(20, min_periods=1).apply(lambda x: np.prod(1 + x) - 1, raw=True)
            f['macro_stock_rs'] = rs - rs_mkt

            f['macro_stock_vol_elastic'] = stock_ret.rolling(60, min_periods=20).std() / (
                hs300_ret.rolling(60, min_periods=20).std() + 1e-10)

        return f.fillna(method='ffill').fillna(0)

    @staticmethod
    def _map(df_dates, macro_series):
        """前向填充映射: 找每个 stock date 对应的最近宏观数据"""
        result = pd.Series(np.nan, index=range(len(df_dates)))
        md = macro_series.index
        mv = macro_series.values
        md_vals = md.values
        for i, d in enumerate(df_dates):
            idx = np.searchsorted(md_vals, np.datetime64(d), side='right') - 1
            if idx >= 0:
                result.iloc[i] = mv[idx]
        return result.fillna(method='ffill')