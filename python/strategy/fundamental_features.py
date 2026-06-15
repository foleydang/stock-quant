"""
基本面特征 v1 — PE/PB/市值/ROE 衍生特征

设计原则:
  1. 所有特征截面可比 (行业中性暂不处理, 后期可加)
  2. 用历史分位数替代绝对值, 避免量纲差异
  3. 变化率特征捕捉估值变动趋势
"""

import os, sqlite3
import numpy as np
import pandas as pd
from typing import Optional


class FundamentalFeatures:
    """基本面特征 (PE/PB/市值等)

    数据来源: fundamental_daily 表
    """

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
        """获取单只股票的基本面历史"""
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

        # 提取基本字段
        pe = self._align(fund, 'pe_ttm', dates)
        pb = self._align(fund, 'pb', dates)

        if pe is None or pb is None:
            return f

        # --- PE 特征 ---
        f['fund_pe'] = pe
        f['fund_pe_pct_1y'] = self._rolling_pct(pe, 250)  # 1年百分位
        f['fund_pe_chg_1m'] = pe.pct_change(20)            # 1月变化率
        f['fund_pe_chg_3m'] = pe.pct_change(60)            # 3月变化率
        f['fund_pe_ma5'] = pe.rolling(5).mean()
        f['fund_pe_ma20'] = pe.rolling(20).mean()
        f['fund_pe_ma20_dev'] = pe / pe.rolling(20).mean() - 1  # 偏离20日均值

        # --- PB 特征 ---
        f['fund_pb'] = pb
        f['fund_pb_pct_1y'] = self._rolling_pct(pb, 250)
        f['fund_pb_chg_1m'] = pb.pct_change(20)
        f['fund_pb_chg_3m'] = pb.pct_change(60)
        f['fund_pb_ma20_dev'] = pb / pb.rolling(20).mean() - 1

        # --- PE/PB 关系 ---
        f['fund_pe_pb_ratio'] = pe / (pb + 1e-10)

        # --- 市值特征 (从 price × volume 估算) ---
        # 用日线 close 和 volume 的乘积作为市值代理
        close = df['close'].astype(float).values
        volume = df['volume'].astype(float).values
        mv_proxy = pd.Series(close * volume / 1e8, index=df.index)  # 亿
        f['fund_mv_proxy'] = mv_proxy
        f['fund_mv_chg_1m'] = mv_proxy.pct_change(20)
        f['fund_mv_chg_3m'] = mv_proxy.pct_change(60)
        f['fund_mv_ma20_dev'] = mv_proxy / mv_proxy.rolling(20).mean() - 1

        return f.fillna(0)

    def _align(self, fund: pd.DataFrame, col: str, dates) -> Optional[pd.Series]:
        """将基本面数据对齐到日线日期"""
        if col not in fund.columns:
            return None
        series = fund[col].reindex(fund.index.union(pd.DatetimeIndex(dates)))
        series = series.ffill()  # 前向填充 (基本面数据非每日更新)
        result = series.reindex(pd.DatetimeIndex(dates))
        return pd.Series(result.values, index=pd.RangeIndex(len(dates)))

    def _rolling_pct(self, series: pd.Series, window: int) -> pd.Series:
        """滚动百分位: 当前值在窗口内的排位"""
        pct = series.rolling(window, min_periods=20).apply(
            lambda x: (x.iloc[-1] > x).mean(), raw=False
        )
        return pct.fillna(0.5)


# 延迟加载单例
_fundamental_features: Optional[FundamentalFeatures] = None


def _get_fundamental_features() -> FundamentalFeatures:
    global _fundamental_features
    if _fundamental_features is None:
        _fundamental_features = FundamentalFeatures()
    return _fundamental_features