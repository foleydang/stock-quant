"""
基本面特征 v2 — 财务数据衍生特征

数据来源: fundamental_daily 表 (stock_financial_abstract_ths)
特征: ROE, 营收增速, 净利增速, 估值, 市值

设计原则:
  1. 财务数据按季度发布, 前向填充到每日
  2. 用变化率和历史分位替代绝对值
  3. 结合日线价格计算 PB_proxy = close / bv_per_share
"""

import os, sqlite3
import numpy as np
import pandas as pd
from typing import Optional


class FundamentalFeatures:
    """基本面特征 (财务数据)"""

    def __init__(self):
        self._data = None
        self._db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data/stock_data.db'
        )

    def _load(self):
        """延迟加载全量基本面数据"""
        if self._data is not None:
            return self._data
        conn = sqlite3.connect(self._db_path)
        try:
            df = pd.read_sql("SELECT * FROM fundamental_daily", conn)
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            self._data = df
        except Exception:
            self._data = pd.DataFrame()
        finally:
            conn.close()
        return self._data

    def get_stock_data(self, symbol: str) -> pd.DataFrame:
        df = self._load()
        if len(df) == 0:
            return pd.DataFrame()
        return df[df['symbol'] == symbol].copy()

    def calculate(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """计算基本面特征 (对齐到日线日期)"""
        f = pd.DataFrame(index=df.index)
        fund = self.get_stock_data(symbol)
        if len(fund) == 0:
            return f

        fund = fund.set_index('trade_date').sort_index()
        dates = df['date'].values

        # 对齐: 财务数据按季度, 前向填充到每日
        cols = ['roe', 'revenue_yoy', 'net_profit_yoy', 'debt_ratio', 'bv_per_share', 'eps']
        aligned = {}
        for col in cols:
            val = self._align(fund, col, dates)
            if val is not None:
                aligned[col] = val

        if not aligned:
            return f

        # --- 盈利能力 ---
        if 'roe' in aligned:
            roe = aligned['roe']
            f['fund_roe'] = roe
            f['fund_roe_chg'] = roe.diff()  # 季度变化
            f['fund_roe_high'] = (roe > 0.15).astype(float)  # ROE>15% 高盈利

        # --- 成长性 ---
        if 'revenue_yoy' in aligned:
            rev = aligned['revenue_yoy']
            f['fund_rev_yoy'] = rev
            f['fund_rev_accel'] = rev.diff()  # 营收加速

        if 'net_profit_yoy' in aligned:
            np_yoy = aligned['net_profit_yoy']
            f['fund_np_yoy'] = np_yoy
            f['fund_np_accel'] = np_yoy.diff()  # 净利加速

        # --- 估值 ---
        if 'bv_per_share' in aligned:
            close = df['close'].astype(float).values
            bv = aligned['bv_per_share'].values
            mask = bv > 0
            pb = np.full(len(close), np.nan)
            pb[mask] = close[mask] / bv[mask]
            pb = pd.Series(pb, index=df.index).fillna(method='ffill').fillna(2.0)
            f['fund_pb_proxy'] = pb
            f['fund_pb_pct'] = pb.rolling(250, min_periods=20).apply(
                lambda x: (x.iloc[-1] < x).mean(), raw=False
            ).fillna(0.5)  # PB 越低越好, 所以用 <

        if 'eps' in aligned:
            eps = aligned['eps']
            close = df['close'].astype(float).values
            # PE proxy = close / eps (TTM)
            eps_val = eps.values
            mask = eps_val > 0
            pe = np.full(len(close), np.nan)
            pe[mask] = close[mask] / eps_val[mask]
            pe = pd.Series(pe, index=df.index).fillna(method='ffill').fillna(20.0)
            f['fund_pe_proxy'] = pe
            f['fund_pe_pct'] = pe.rolling(250, min_periods=20).apply(
                lambda x: (x.iloc[-1] < x).mean(), raw=False
            ).fillna(0.5)

        # --- 财务健康 ---
        if 'debt_ratio' in aligned:
            dr = aligned['debt_ratio']
            f['fund_debt_ratio'] = dr
            f['fund_low_debt'] = (dr < 0.5).astype(float)  # 低负债

        # --- 市值特征 (从日线) ---
        close = df['close'].astype(float).values
        volume = df['volume'].astype(float).values
        mv_proxy = pd.Series(close * volume / 1e8, index=df.index)  # 亿
        f['fund_mv'] = mv_proxy
        f['fund_mv_chg_1m'] = mv_proxy.pct_change(20)
        f['fund_mv_chg_3m'] = mv_proxy.pct_change(60)
        f['fund_mv_rank'] = mv_proxy.rolling(250, min_periods=20).apply(
            lambda x: (x.iloc[-1] > x).mean(), raw=False
        ).fillna(0.5)

        return f.fillna(0)

    def _align(self, fund: pd.DataFrame, col: str, dates) -> Optional[pd.Series]:
        """将财务数据对齐到日线日期 (前向填充)"""
        if col not in fund.columns:
            return None
        series = fund[col].reindex(
            fund.index.union(pd.DatetimeIndex(dates))
        )
        series = series.ffill()
        result = series.reindex(pd.DatetimeIndex(dates))
        return pd.Series(result.values, index=pd.RangeIndex(len(dates)))


# 延迟加载单例
_fundamental_features: Optional[FundamentalFeatures] = None


def _get_fundamental_features() -> FundamentalFeatures:
    global _fundamental_features
    if _fundamental_features is None:
        _fundamental_features = FundamentalFeatures()
    return _fundamental_features