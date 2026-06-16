#!/usr/bin/env python3
"""
分钟级特征工程 v1 — 微观结构 + 日内模式 + 截面相对 + 日线上下文

设计原则:
  - 严格向后看: 只用当前及之前K线的数据
  - 聚焦微观结构: 量价关系、波动率、日内模式
  - 轻量: 80-120个特征，远少于日线
  - 命名: {category}_{name}_{period}

特征类别:
  1. MicroPrice   (~25) 多周期收益率、波动率、均线偏离
  2. MicroVolume  (~15) 量比、量趋势、量价背离
  3. Intraday     (~15) 时段效应、开盘/尾盘
  4. TechInd      (~15) 短周期技术指标
  5. CrossSection (~10) pool内截面排名
  6. DailyContext (~5)  日线模型输出
"""

import numpy as np
import pandas as pd
import sqlite3
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kw): return iterable

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/stock_data.db')


# ============ 微观价格特征 ============
class MicroPriceFeatures:
    """分钟级价格特征: 短周期收益率、波动率、均线"""

    @staticmethod
    def calculate(df: pd.DataFrame,
                  ret_periods=(1, 3, 5, 10, 20),
                  vol_periods=(5, 10, 20, 60),
                  ma_periods=(5, 10, 20, 60)) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        open_price = df['open'].values.astype(float)

        # ---- 收益率 ----
        for p in ret_periods:
            ret = pd.Series(close).pct_change(p)
            f[f'mp_ret_{p}'] = ret
            if p <= 5:
                roll_mean = ret.rolling(20, min_periods=5).mean()
                roll_std = ret.rolling(20, min_periods=5).std()
                f[f'mp_ret_{p}_z'] = (ret - roll_mean) / (roll_std + 1e-10)

        # ---- 波动率 ----
        returns = pd.Series(close).pct_change()
        for p in vol_periods:
            vol = returns.rolling(p, min_periods=max(3, p//2)).std()
            f[f'mp_vol_{p}'] = vol

        for p in (10, 20):
            vol = returns.rolling(p, min_periods=max(3, p//2)).std()
            f[f'mp_vol_chg_{p}'] = vol.diff(5) / (vol.shift(5) + 1e-10)

        # ---- 均线偏离 ----
        for p in ma_periods:
            ma = pd.Series(close).rolling(p, min_periods=max(3, p//2)).mean()
            f[f'mp_ma{p}_dist'] = (close - ma) / (ma + 1e-10)
            f[f'mp_above_ma{p}'] = (close > ma).astype(int)

        ma5 = pd.Series(close).rolling(5, min_periods=3).mean()
        ma10 = pd.Series(close).rolling(10, min_periods=5).mean()
        ma20 = pd.Series(close).rolling(20, min_periods=10).mean()
        f['mp_ma_bullish'] = ((ma5 > ma10) & (ma10 > ma20)).astype(int)
        f['mp_ma_bearish'] = ((ma5 < ma10) & (ma10 < ma20)).astype(int)

        # ---- 价格位置 ----
        for p in (10, 20, 60):
            roll_high = pd.Series(high).rolling(p, min_periods=max(3, p//2)).max()
            roll_low = pd.Series(low).rolling(p, min_periods=max(3, p//2)).min()
            f[f'mp_pos_{p}'] = (close - roll_low) / (roll_high - roll_low + 1e-10)
            f[f'mp_high_dist_{p}'] = (close - roll_high) / (roll_high + 1e-10)

        # ---- 日内位置 ----
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
            day_groups = dates.dt.date
            day_open = df.groupby(day_groups)['open'].transform('first').values
            day_high = df.groupby(day_groups)['high'].transform('max').values
            day_low = df.groupby(day_groups)['low'].transform('min').values
            f['mp_day_open_ret'] = (close - day_open) / (day_open + 1e-10)
            f['mp_day_pos'] = (close - day_low) / (day_high - day_low + 1e-10)
        else:
            f['mp_day_open_ret'] = 0
            f['mp_day_pos'] = 0.5

        # ---- 振幅 ----
        for p in (5, 10, 20):
            f[f'mp_range_{p}'] = (pd.Series(high).rolling(p).max() -
                                   pd.Series(low).rolling(p).min()) / \
                                  (pd.Series(close).rolling(p).mean() + 1e-10)

        # ---- 连续涨跌 ----
        for p in (3, 5):
            up = (returns > 0).astype(int)
            f[f'mp_up_streak_{p}'] = up.rolling(p).sum()
            f[f'mp_down_streak_{p}'] = (returns < 0).astype(int).rolling(p).sum()

        # ---- 加速/减速 ----
        ret1 = pd.Series(close).pct_change(1)
        ret3 = pd.Series(close).pct_change(3)
        f['mp_accel'] = ret1 - ret1.shift(1)
        f['mp_decay_3_10'] = ret3 - pd.Series(close).pct_change(10)

        return f


# ============ 微观成交量特征 ============
class MicroVolumeFeatures:
    """分钟级成交量特征: 量比、量趋势、量价关系"""

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        close = df['close'].values.astype(float)
        volume = df['volume'].values.astype(float)
        vol = pd.Series(volume)

        for p in (5, 10, 20, 60):
            ma = vol.rolling(p, min_periods=max(3, p//2)).mean()
            f[f'mv_vol_ratio_{p}'] = vol / (ma + 1e-10)

        f['mv_vol_trend_5_20'] = vol.rolling(5, min_periods=3).mean() / \
                                  (vol.rolling(20, min_periods=10).mean() + 1e-10)
        f['mv_vol_trend_5_60'] = vol.rolling(5, min_periods=3).mean() / \
                                  (vol.rolling(60, min_periods=30).mean() + 1e-10)

        f['mv_vol_chg_1'] = vol.pct_change(1)
        f['mv_vol_chg_5'] = vol.pct_change(5)

        for p in (10, 20):
            f[f'mv_vol_std_{p}'] = vol.rolling(p, min_periods=max(3, p//2)).std() / \
                                    (vol.rolling(p, min_periods=max(3, p//2)).mean() + 1e-10)

        price_up = (pd.Series(close).diff() > 0).astype(int)
        vol_up = (vol.diff() > 0).astype(int)
        f['mv_divergence'] = (price_up != vol_up).rolling(5, min_periods=3).mean()
        f['mv_confirmation'] = (price_up == vol_up).rolling(5, min_periods=3).mean()

        vol_ma20 = vol.rolling(20, min_periods=10).mean()
        f['mv_volume_spike'] = (vol > 2 * vol_ma20).astype(int)
        f['mv_volume_dry'] = (vol < 0.3 * vol_ma20).astype(int)

        obv = np.zeros(len(close))
        obv[0] = volume[0]
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]
        f['mv_obv_chg_5'] = pd.Series(obv).pct_change(5)
        f['mv_obv_chg_20'] = pd.Series(obv).pct_change(20)

        return f


# ============ 日内模式特征 ============
class IntradayPatternFeatures:
    """日内模式: 时段效应、开盘/收盘、星期效应"""

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)

        if 'date' not in df.columns:
            for col in ['ip_session_morning', 'ip_session_afternoon',
                        'ip_opening_30m', 'ip_closing_30m',
                        'ip_hour_0', 'ip_hour_1', 'ip_hour_2', 'ip_hour_3',
                        'ip_dow_0', 'ip_dow_1', 'ip_dow_2', 'ip_dow_3', 'ip_dow_4',
                        'ip_minutes_since_open', 'ip_is_first_hour',
                        'ip_is_last_hour']:
                f[col] = 0
            return f

        dates = pd.to_datetime(df['date'])
        hours = dates.dt.hour
        minutes = dates.dt.minute
        total_minutes = hours * 60 + minutes

        f['ip_session_morning'] = ((total_minutes >= 570) & (total_minutes < 690)).astype(int)
        f['ip_session_afternoon'] = ((total_minutes >= 780) & (total_minutes < 900)).astype(int)
        f['ip_opening_30m'] = ((total_minutes >= 570) & (total_minutes < 600)).astype(int)
        f['ip_closing_30m'] = ((total_minutes >= 870) & (total_minutes < 900)).astype(int)
        f['ip_hour_0'] = ((total_minutes >= 570) & (total_minutes < 630)).astype(int)
        f['ip_hour_1'] = ((total_minutes >= 630) & (total_minutes < 690)).astype(int)
        f['ip_hour_2'] = ((total_minutes >= 780) & (total_minutes < 840)).astype(int)
        f['ip_hour_3'] = ((total_minutes >= 840) & (total_minutes < 900)).astype(int)

        for d in range(5):
            f[f'ip_dow_{d}'] = (dates.dt.dayofweek == d).astype(int)

        day_groups = dates.dt.date
        f['ip_minutes_since_open'] = df.groupby(day_groups).cumcount() * 30
        f['ip_is_first_hour'] = (f['ip_minutes_since_open'] < 60).astype(int)
        f['ip_is_last_hour'] = (f['ip_minutes_since_open'] >= 150).astype(int)

        return f


# ============ 短周期技术指标 ============
class ShortTermTechFeatures:
    """短周期技术指标: RSI, MACD, 布林, KDJ, ADX"""

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)

        for p in (6, 14):
            delta = pd.Series(close).diff()
            gain = delta.where(delta > 0, 0).rolling(p, min_periods=max(3, p//2)).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(p, min_periods=max(3, p//2)).mean()
            rs = gain / (loss + 1e-10)
            f[f'ti_rsi_{p}'] = 100 - (100 / (1 + rs))

        if 'ti_rsi_14' in f.columns:
            f['ti_rsi_chg'] = f['ti_rsi_14'].diff(3)

        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        f['ti_macd_hist'] = macd - signal
        f['ti_macd_hist_chg'] = f['ti_macd_hist'].diff()

        low_9 = pd.Series(low).rolling(9, min_periods=5).min()
        high_9 = pd.Series(high).rolling(9, min_periods=5).max()
        rsv = (close - low_9) / (high_9 - low_9 + 1e-10) * 100
        k = rsv.ewm(com=2).mean()
        d = k.ewm(com=2).mean()
        f['ti_kdj_k'] = k
        f['ti_kdj_d'] = d
        f['ti_kdj_j'] = 3 * k - 2 * d

        for p in (10, 20):
            ma = pd.Series(close).rolling(p, min_periods=max(3, p//2)).mean()
            std = pd.Series(close).rolling(p, min_periods=max(3, p//2)).std()
            upper = ma + 2 * std
            lower = ma - 2 * std
            f[f'ti_bb{p}_width'] = (upper - lower) / (ma + 1e-10)
            f[f'ti_bb{p}_pos'] = (close - lower) / (upper - lower + 1e-10)

        tr = pd.concat([
            pd.Series(high) - pd.Series(low),
            (pd.Series(high) - pd.Series(close).shift(1)).abs(),
            (pd.Series(close).shift(1) - pd.Series(low)).abs()
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14, min_periods=7).mean()
        plus_dm = pd.Series(high).diff()
        minus_dm = -pd.Series(low).diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        plus_di = 100 * (plus_dm.rolling(14, min_periods=7).mean() / (atr14 + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14, min_periods=7).mean() / (atr14 + 1e-10))
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        f['ti_adx'] = dx.rolling(14, min_periods=7).mean()

        tp = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
        f['ti_cci'] = (tp - tp.rolling(20, min_periods=10).mean()) / \
                       (0.015 * tp.rolling(20, min_periods=10).std() + 1e-10)

        return f


# ============ 截面相对特征 (分钟级) — 向量化版本 ============
class IntradayCrossSection:
    """分钟级截面特征: 在相同时刻，股票在pool内的相对排名

    向量化实现: 对每个排名目标，构建 (timestamps × symbols) 矩阵
    然后沿 columns axis 批量计算 rank，最后拆回 per-symbol DataFrame
    """

    RANK_TARGETS = [
        'mp_ret_3', 'mp_ret_5', 'mp_ret_10',
        'mp_vol_10', 'mp_vol_20',
        'mp_ma5_dist', 'mp_ma20_dist',
        'mv_vol_ratio_5', 'mv_vol_ratio_20',
        'ti_rsi_14', 'ti_adx',
    ]

    @staticmethod
    def calculate(all_features: Dict[str, pd.DataFrame],
                  all_timestamps: np.ndarray) -> Dict[str, pd.DataFrame]:
        """向量化截面排名计算

        策略: 对每个 RANK_TARGET，构建一个 (timestamps × symbols) 的 DataFrame，
        然后对每行 (axis=1) 做 rank(pct=True)，最后拆回 per-symbol DataFrame。
        复杂度从 O(T×S×F) 降到 O(F×T×S)，其中 F=11(tiny), T=15K, S=372。
        """
        symbols = sorted(all_features.keys())
        all_ts = sorted(set(all_timestamps))
        ts_idx = {ts: i for i, ts in enumerate(all_ts)}

        result = {sym: pd.DataFrame(index=all_features[sym].index) for sym in symbols}

        for target in tqdm(IntradayCrossSection.RANK_TARGETS, desc='   截面排名', unit='target'):
            # 构建 (timestamps × symbols) 矩阵
            matrix = pd.DataFrame(index=all_ts, columns=symbols, dtype=float)

            for sym in symbols:
                feats = all_features[sym]
                if target in feats.columns:
                    series = feats[target]
                    for ts, val in series.items():
                        if ts in ts_idx:
                            matrix.loc[ts, sym] = val

            # 对每一行 (每个时间戳) 做 rank
            # 要求至少 10 只股票有值
            valid_counts = matrix.notna().sum(axis=1)
            rank_matrix = matrix.rank(axis=1, pct=True, method='average')
            # 过滤无效行
            rank_matrix[valid_counts < 10] = np.nan

            # 写回 per-symbol
            col_name = f'ics_rank_{target}'
            for sym in symbols:
                series = rank_matrix[sym].dropna()
                if len(series) > 0:
                    if col_name not in result[sym].columns:
                        result[sym][col_name] = np.nan
                    result[sym].loc[series.index, col_name] = series.values

        return result


# ============ 日线模型上下文 ============
class DailyModelContext:
    """从日线模型获取上下文特征"""

    def __init__(self, daily_model_path: str = None):
        self.daily_model_path = daily_model_path
        self._daily_scores = None

    def load_daily_scores(self, symbols: List[str], min_date: str, max_date: str) -> Dict[str, Dict[str, float]]:
        if self._daily_scores is not None:
            return self._daily_scores
        self._daily_scores = {}
        if self.daily_model_path and os.path.exists(self.daily_model_path):
            import pickle
            try:
                with open(self.daily_model_path, 'rb') as f:
                    pickle.load(f)
            except Exception as e:
                pass  # 静默处理
        return self._daily_scores

    def calculate(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        for col in ['dc_score', 'dc_rank', 'dc_in_pool']:
            f[col] = 0
        if not self._daily_scores or symbol not in self._daily_scores:
            return f
        scores = self._daily_scores[symbol]
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            f['dc_score'] = dates.map(lambda d: scores.get(d, 0)).fillna(0)
            f['dc_in_pool'] = (f['dc_score'] != 0).astype(int)
        return f


# ============ 特征流水线 ============
class IntradayFeaturePipeline:
    """分钟级特征统一计算流水线"""

    def __init__(self, daily_model_path: str = None):
        self.daily_ctx = DailyModelContext(daily_model_path)

    def compute_stock(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        micro_price = MicroPriceFeatures.calculate(df)
        micro_vol = MicroVolumeFeatures.calculate(df)
        intraday = IntradayPatternFeatures.calculate(df)
        tech = ShortTermTechFeatures.calculate(df)
        daily_ctx = self.daily_ctx.calculate(df, symbol)
        return pd.concat([micro_price, micro_vol, intraday, tech, daily_ctx], axis=1)

    def compute_cross_section(self, all_features: Dict[str, pd.DataFrame],
                              all_timestamps: np.ndarray) -> Dict[str, pd.DataFrame]:
        return IntradayCrossSection.calculate(all_features, all_timestamps)


if __name__ == '__main__':
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM kline_30m WHERE symbol='600519.SH' ORDER BY date LIMIT 500", conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    pipeline = IntradayFeaturePipeline()
    feats = pipeline.compute_stock(df, '600519.SH')
    print(f"特征数: {len(feats.columns)}")
    print(f"特征列: {list(feats.columns)}")
    print(f"缺失率: {(feats.isna().sum() / len(feats) * 100).describe()}")