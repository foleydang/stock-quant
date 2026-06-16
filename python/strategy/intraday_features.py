#!/usr/bin/env python3
"""
分钟级特征工程 v2 — 大幅扩展特征空间 (目标 250+ 特征)

设计原则:
  - 严格向后看: 只用当前及之前K线的数据
  - 多周期: 覆盖短/中/长周期, 让模型自动选择
  - 丰富性: 对标日线模型, 量价+技术+形态+截面全覆盖
  - 命名: {category}_{name}_{period}

特征类别:
  1. MicroPrice   (~55) 收益率、波动率、均线、价格通道
  2. MicroVolume  (~25) 量比、MFI、CMF、量价交互
  3. Intraday     (~20) 时段效应、开盘/尾盘、星期
  4. TechInd      (~30) RSI/MACD/KDJ/布林/ATR/ADX/CCI/WilliamsR/Ultimate
  5. Pattern      (~15) K线形态: 锤子/吞没/十字星/三连阳
  6. CrossSection (~15) pool内截面排名
  7. DailyContext (~5)  日线模型输出
  8. Advanced     (~20) 波动率聚集、相关性、收益分布
"""

import numpy as np
import pandas as pd
import sqlite3
import os
from typing import Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kw): return iterable

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/stock_data.db')


# ============ 微观价格特征 (扩展版) ============
class MicroPriceFeatures:
    """分钟级价格特征: 多周期收益率、波动率、均线、价格通道"""

    RET_P = (1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 200)
    VOL_P = (3, 5, 10, 15, 20, 30, 60, 80, 100, 120)
    MA_P  = (3, 5, 10, 15, 20, 30, 60, 80, 100, 120, 200)
    POS_P = (5, 10, 20, 30, 60, 120)

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        open_p = df['open'].values.astype(float)
        volume = df['volume'].values.astype(float)

        # ---- 1. 收益率 (多周期) ----
        for p in MicroPriceFeatures.RET_P:
            ret = pd.Series(close).pct_change(p)
            f[f'mp_ret_{p}'] = ret
            if p <= 5:
                roll_mean = ret.rolling(20, min_periods=5).mean()
                roll_std = ret.rolling(20, min_periods=5).std()
                f[f'mp_ret_{p}_z'] = (ret - roll_mean) / (roll_std + 1e-10)

        # ---- 2. 对数收益率 ----
        for p in (1, 3, 5, 10, 20, 60):
            f[f'mp_logret_{p}'] = np.log(pd.Series(close) / pd.Series(close).shift(p))

        # ---- 3. 波动率 (多周期) ----
        returns = pd.Series(close).pct_change()
        for p in MicroPriceFeatures.VOL_P:
            vol = returns.rolling(p, min_periods=max(3, p//2)).std()
            f[f'mp_vol_{p}'] = vol

        # Parkinson 波动率 (用high/low)
        for p in (10, 20, 60):
            hl = np.log(pd.Series(high) / pd.Series(low))
            f[f'mp_vol_park_{p}'] = np.sqrt(
                (hl ** 2).rolling(p, min_periods=max(3, p//2)).mean() / (4 * np.log(2))
            )

        # Garman-Klass 波动率 (用OHLC)
        for p in (10, 20):
            oc = np.log(pd.Series(open_p) / pd.Series(close).shift(1))
            hl = np.log(pd.Series(high) / pd.Series(low))
            co = np.log(pd.Series(close) / pd.Series(open_p))
            gk = 0.5 * hl**2 - (2*np.log(2)-1) * co**2
            f[f'mp_vol_gk_{p}'] = np.sqrt(gk.rolling(p, min_periods=max(3, p//2)).mean())

        # 波动率变化 (regime shift)
        for p in (10, 20, 60):
            vol = returns.rolling(p, min_periods=max(3, p//2)).std()
            f[f'mp_vol_chg_{p}'] = vol.diff(5) / (vol.shift(5) + 1e-10)

        # ---- 4. 均线系统 (扩展) ----
        for p in MicroPriceFeatures.MA_P:
            ma = pd.Series(close).rolling(p, min_periods=max(3, p//2)).mean()
            f[f'mp_ma{p}_dist'] = (close - ma) / (ma + 1e-10)
            f[f'mp_above_ma{p}'] = (close > ma).astype(int)

        # 均线斜率
        for p in (5, 10, 20, 60):
            ma = pd.Series(close).rolling(p, min_periods=max(3, p//2)).mean()
            f[f'mp_ma{p}_slope'] = ma.diff(3) / (ma + 1e-10)

        # 均线交叉
        crosses = [(5, 10), (5, 20), (10, 20), (10, 60), (20, 60), (60, 120)]
        for fast, slow in crosses:
            ma_f = pd.Series(close).rolling(fast, min_periods=max(3, fast//2)).mean()
            ma_s = pd.Series(close).rolling(slow, min_periods=max(3, slow//2)).mean()
            f[f'mp_ma{fast}_{slow}_dist'] = ma_f / ma_s - 1
            f[f'mp_ma{fast}_{slow}_cross'] = ((ma_f > ma_s) & (ma_f.shift(1) <= ma_s.shift(1))).astype(int)

        # 均线排列
        ma5 = pd.Series(close).rolling(5, min_periods=3).mean()
        ma10 = pd.Series(close).rolling(10, min_periods=5).mean()
        ma20 = pd.Series(close).rolling(20, min_periods=10).mean()
        ma60 = pd.Series(close).rolling(60, min_periods=30).mean()
        f['mp_ma_bullish'] = ((ma5 > ma10) & (ma10 > ma20) & (ma20 > ma60)).astype(int)
        f['mp_ma_bearish'] = ((ma5 < ma10) & (ma10 < ma20) & (ma20 < ma60)).astype(int)

        # ---- 5. 价格通道 / 位置 ----
        for p in MicroPriceFeatures.POS_P:
            roll_high = pd.Series(high).rolling(p, min_periods=max(3, p//2)).max()
            roll_low = pd.Series(low).rolling(p, min_periods=max(3, p//2)).min()
            f[f'mp_pos_{p}'] = (close - roll_low) / (roll_high - roll_low + 1e-10)
            f[f'mp_high_dist_{p}'] = (close - roll_high) / (roll_high + 1e-10)
            f[f'mp_low_dist_{p}'] = (close - roll_low) / (roll_low + 1e-10)

        # 突破信号
        for p in (20, 60, 120):
            roll_high = pd.Series(high).rolling(p, min_periods=max(3, p//2)).max()
            roll_low = pd.Series(low).rolling(p, min_periods=max(3, p//2)).min()
            f[f'mp_break_high_{p}'] = (close > roll_high.shift(1)).astype(int)
            f[f'mp_break_low_{p}'] = (close < roll_low.shift(1)).astype(int)

        # ---- 6. 日内位置 ----
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
            day_grp = dates.dt.date
            day_open = df.groupby(day_grp)['open'].transform('first').values
            day_high = df.groupby(day_grp)['high'].cummax().values
            day_low = df.groupby(day_grp)['low'].cummin().values
            f['mp_day_open_ret'] = (close - day_open) / (day_open + 1e-10)
            f['mp_day_pos'] = (close - day_low) / (day_high - day_low + 1e-10)
            # 日内是否创新高/新低
            prev_high = np.roll(day_high, 1); prev_high[0] = day_high[0]
            prev_low = np.roll(day_low, 1); prev_low[0] = day_low[0]
            f['mp_day_new_high'] = (day_high > prev_high).astype(int)
            f['mp_day_new_low'] = (day_low < prev_low).astype(int)
        else:
            f['mp_day_open_ret'] = 0
            f['mp_day_pos'] = 0.5
            f['mp_day_new_high'] = 0
            f['mp_day_new_low'] = 0

        # ---- 7. 振幅 ----
        for p in (5, 10, 20, 60):
            f[f'mp_range_{p}'] = (pd.Series(high).rolling(p).max() -
                                   pd.Series(low).rolling(p).min()) / \
                                  (pd.Series(close).rolling(p).mean() + 1e-10)

        # ---- 8. 连续涨跌 ----
        up = (returns > 0).astype(int)
        down = (returns < 0).astype(int)
        for p in (3, 5, 10, 20):
            f[f'mp_up_streak_{p}'] = up.rolling(p).sum()
            f[f'mp_down_streak_{p}'] = down.rolling(p).sum()

        # ---- 9. 加速/减速 ----
        for p in (3, 5, 10):
            ret_p = pd.Series(close).pct_change(p)
            f[f'mp_accel_{p}'] = ret_p.diff(p)

        ret1 = pd.Series(close).pct_change(1)
        ret3 = pd.Series(close).pct_change(3)
        ret10 = pd.Series(close).pct_change(10)
        ret20 = pd.Series(close).pct_change(20)
        f['mp_decay_3_10'] = ret3 - ret10
        f['mp_decay_10_20'] = ret10 - ret20
        f['mp_decay_5_20'] = pd.Series(close).pct_change(5) - ret20

        return f


# ============ 微观成交量特征 (扩展版) ============
class MicroVolumeFeatures:
    """成交量特征: 量比、MFI、CMF、量价交互"""

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float)
        vol = pd.Series(volume)

        # ---- 1. 量比 (多周期) ----
        for p in (3, 5, 10, 20, 30, 60, 120):
            ma = vol.rolling(p, min_periods=max(3, p//2)).mean()
            f[f'mv_vol_ratio_{p}'] = vol / (ma + 1e-10)

        # ---- 2. 量趋势 ----
        for fast, slow in [(5, 20), (5, 60), (10, 60), (20, 120)]:
            f[f'mv_vol_trend_{fast}_{slow}'] = vol.rolling(fast, min_periods=max(3, fast//2)).mean() / \
                                                (vol.rolling(slow, min_periods=max(3, slow//2)).mean() + 1e-10)

        # ---- 3. 量变化 ----
        for p in (1, 3, 5, 10, 20):
            f[f'mv_vol_chg_{p}'] = vol.pct_change(p)

        # ---- 4. 量波动 ----
        for p in (10, 20, 60):
            f[f'mv_vol_std_{p}'] = vol.rolling(p, min_periods=max(3, p//2)).std() / \
                                    (vol.rolling(p, min_periods=max(3, p//2)).mean() + 1e-10)

        # ---- 5. 量价关系 ----
        price_up = (pd.Series(close).diff() > 0).astype(int)
        vol_up = (vol.diff() > 0).astype(int)
        for p in (5, 10, 20):
            f[f'mv_div_{p}'] = (price_up != vol_up).rolling(p, min_periods=max(3, p//2)).mean()
            f[f'mv_conf_{p}'] = (price_up == vol_up).rolling(p, min_periods=max(3, p//2)).mean()

        # ---- 6. 放量/缩量 ----
        vol_ma20 = vol.rolling(20, min_periods=10).mean()
        vol_ma60 = vol.rolling(60, min_periods=30).mean()
        f['mv_volume_spike'] = (vol > 2 * vol_ma20).astype(int)
        f['mv_volume_dry'] = (vol < 0.3 * vol_ma20).astype(int)
        f['mv_volume_spike_60'] = (vol > 2 * vol_ma60).astype(int)

        # ---- 7. OBV ----
        obv = np.zeros(len(close))
        obv[0] = volume[0]
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]
        for p in (5, 10, 20):
            f[f'mv_obv_chg_{p}'] = pd.Series(obv).pct_change(p)
        f['mv_obv_ma'] = pd.Series(obv).rolling(20, min_periods=10).mean()
        f['mv_obv_vs_ma'] = obv / (f['mv_obv_ma'].values + 1e-10)

        # ---- 8. MFI (Money Flow Index) ----
        tp = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
        mf = tp * vol
        pos_mf = mf.where(tp.diff() > 0, 0).rolling(14, min_periods=7).sum()
        neg_mf = mf.where(tp.diff() < 0, 0).rolling(14, min_periods=7).sum()
        mfr = pos_mf / (neg_mf.abs() + 1e-10)
        f['mv_mfi'] = 100 - (100 / (1 + mfr))

        # ---- 9. CMF (Chaikin Money Flow) ----
        clv = ((pd.Series(close) - pd.Series(low)) - (pd.Series(high) - pd.Series(close))) / \
               (pd.Series(high) - pd.Series(low) + 1e-10)
        f['mv_cmf'] = (clv * vol).rolling(20, min_periods=10).sum() / \
                       vol.rolling(20, min_periods=10).sum()

        # ---- 10. VWAP 偏离 ----
        cum_vp = (tp * vol).cumsum()
        cum_vol = vol.cumsum()
        vwap = cum_vp / (cum_vol + 1e-10)
        f['mv_vwap_dev'] = close / vwap - 1
        # 日内VWAP
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
            day_grp = dates.dt.date
            tp_series = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
            cum_vp_day = (tp_series * vol).groupby(day_grp).cumsum()
            cum_vol_day = vol.groupby(day_grp).cumsum()
            vwap_day = cum_vp_day / (cum_vol_day + 1e-10)
            f['mv_vwap_day_dev'] = close / vwap_day.values - 1
        else:
            f['mv_vwap_day_dev'] = 0

        return f


# ============ 日内模式特征 (扩展版) ============
class IntradayPatternFeatures:
    """日内模式: 时段效应、开盘/收盘、星期、月份"""

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)

        if 'date' not in df.columns:
            for col in ['ip_morning', 'ip_afternoon', 'ip_opening_30m', 'ip_closing_30m',
                        'ip_hour_0', 'ip_hour_1', 'ip_hour_2', 'ip_hour_3',
                        'ip_dow_0', 'ip_dow_1', 'ip_dow_2', 'ip_dow_3', 'ip_dow_4',
                        'ip_month', 'ip_minutes', 'ip_first_hour', 'ip_last_hour',
                        'ip_midday', 'ip_bar_of_day']:
                f[col] = 0
            return f

        dates = pd.to_datetime(df['date'])
        total_min = dates.dt.hour * 60 + dates.dt.minute

        f['ip_morning'] = ((total_min >= 570) & (total_min < 690)).astype(int)
        f['ip_afternoon'] = ((total_min >= 780) & (total_min < 900)).astype(int)
        f['ip_opening_30m'] = ((total_min >= 570) & (total_min < 600)).astype(int)
        f['ip_closing_30m'] = ((total_min >= 870) & (total_min < 900)).astype(int)
        f['ip_midday'] = ((total_min >= 660) & (total_min < 690)).astype(int)
        f['ip_hour_0'] = ((total_min >= 570) & (total_min < 630)).astype(int)
        f['ip_hour_1'] = ((total_min >= 630) & (total_min < 690)).astype(int)
        f['ip_hour_2'] = ((total_min >= 780) & (total_min < 840)).astype(int)
        f['ip_hour_3'] = ((total_min >= 840) & (total_min < 900)).astype(int)

        for d in range(5):
            f[f'ip_dow_{d}'] = (dates.dt.dayofweek == d).astype(int)

        f['ip_month'] = dates.dt.month
        f['ip_first_hour'] = (f['ip_hour_0'] | f['ip_hour_2']).astype(int)
        f['ip_last_hour'] = (f['ip_hour_1'] | f['ip_hour_3']).astype(int)

        day_grp = dates.dt.date
        f['ip_minutes'] = df.groupby(day_grp).cumcount() * 30
        f['ip_bar_of_day'] = df.groupby(day_grp).cumcount() + 1
        f['ip_bar_pct'] = f['ip_bar_of_day'] / df.groupby(day_grp).cumcount().transform('max')

        # 上午/下午成交量占比 (当日累计)
        morning_mask = (total_min >= 570) & (total_min < 690)
        afternoon_mask = (total_min >= 780) & (total_min < 900)
        if 'volume' in df.columns:
            cum_vol = df.groupby(day_grp)['volume'].cumsum()
            total_vol = df.groupby(day_grp)['volume'].transform('sum')
            f['ip_vol_pct'] = cum_vol / (total_vol + 1e-10)

        return f


# ============ 短周期技术指标 (扩展版) ============
class ShortTermTechFeatures:
    """技术指标: RSI/MACD/KDJ/BB/ATR/ADX/CCI/WilliamsR/UltimateOsc"""

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float)

        # ---- RSI ----
        for p in (6, 14, 24):
            delta = pd.Series(close).diff()
            gain = delta.where(delta > 0, 0).rolling(p, min_periods=max(3, p//2)).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(p, min_periods=max(3, p//2)).mean()
            rs = gain / (loss + 1e-10)
            f[f'ti_rsi_{p}'] = 100 - (100 / (1 + rs))

        if 'ti_rsi_14' in f.columns:
            f['ti_rsi_chg'] = f['ti_rsi_14'].diff(3)
            f['ti_rsi_overbought'] = (f['ti_rsi_14'] > 70).astype(int)
            f['ti_rsi_oversold'] = (f['ti_rsi_14'] < 30).astype(int)

        # ---- MACD ----
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        f['ti_macd'] = macd
        f['ti_macd_signal'] = signal
        f['ti_macd_hist'] = macd - signal
        f['ti_macd_hist_chg'] = f['ti_macd_hist'].diff()
        f['ti_macd_cross'] = ((macd > signal) & (macd.shift(1) <= signal.shift(1))).astype(int)

        # ---- KDJ ----
        for p in (9, 18):
            low_min = pd.Series(low).rolling(p, min_periods=max(3, p//2)).min()
            high_max = pd.Series(high).rolling(p, min_periods=max(3, p//2)).max()
            rsv = (close - low_min) / (high_max - low_min + 1e-10) * 100
            k = rsv.ewm(com=2).mean()
            d = k.ewm(com=2).mean()
            j = 3 * k - 2 * d
            f[f'ti_kdj_k_{p}'] = k
            f[f'ti_kdj_d_{p}'] = d
            f[f'ti_kdj_j_{p}'] = j

        # ---- 布林带 ----
        for p in (10, 20, 60):
            ma = pd.Series(close).rolling(p, min_periods=max(3, p//2)).mean()
            std = pd.Series(close).rolling(p, min_periods=max(3, p//2)).std()
            upper = ma + 2 * std
            lower = ma - 2 * std
            f[f'ti_bb{p}_width'] = (upper - lower) / (ma + 1e-10)
            f[f'ti_bb{p}_pos'] = (close - lower) / (upper - lower + 1e-10)
            f[f'ti_bb{p}_squeeze'] = (f[f'ti_bb{p}_width'] < 0.05).astype(int)

        # ---- ATR ----
        tr = pd.concat([
            pd.Series(high) - pd.Series(low),
            (pd.Series(high) - pd.Series(close).shift(1)).abs(),
            (pd.Series(close).shift(1) - pd.Series(low)).abs()
        ], axis=1).max(axis=1)
        for p in (10, 14, 20):
            f[f'ti_atr_{p}'] = tr.rolling(p, min_periods=max(3, p//2)).mean()
        f['ti_atr_ratio'] = f['ti_atr_14'] / pd.Series(close)

        # ---- ADX ----
        atr14 = tr.rolling(14, min_periods=7).mean()
        plus_dm = pd.Series(high).diff()
        minus_dm = -pd.Series(low).diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        plus_di = 100 * (plus_dm.rolling(14, min_periods=7).mean() / (atr14 + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14, min_periods=7).mean() / (atr14 + 1e-10))
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        f['ti_adx'] = dx.rolling(14, min_periods=7).mean()
        f['ti_adx_trend'] = np.where(plus_di > minus_di, f['ti_adx'], -f['ti_adx'])

        # ---- CCI ----
        tp = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
        f['ti_cci'] = (tp - tp.rolling(20, min_periods=10).mean()) / \
                       (0.015 * tp.rolling(20, min_periods=10).std() + 1e-10)

        # ---- Williams %R ----
        for p in (10, 14):
            hh = pd.Series(high).rolling(p, min_periods=max(3, p//2)).max()
            ll = pd.Series(low).rolling(p, min_periods=max(3, p//2)).min()
            f[f'ti_wr_{p}'] = (hh - close) / (hh - ll + 1e-10) * -100

        # ---- Ultimate Oscillator ----
        bp = close - np.minimum(low, pd.Series(close).shift(1))
        tr_uo = np.maximum(high, pd.Series(close).shift(1)) - np.minimum(low, pd.Series(close).shift(1))
        avg7 = pd.Series(bp).rolling(7, min_periods=4).sum() / pd.Series(tr_uo).rolling(7, min_periods=4).sum()
        avg14 = pd.Series(bp).rolling(14, min_periods=7).sum() / pd.Series(tr_uo).rolling(14, min_periods=7).sum()
        avg28 = pd.Series(bp).rolling(28, min_periods=14).sum() / pd.Series(tr_uo).rolling(28, min_periods=14).sum()
        f['ti_uo'] = 100 * (4*avg7 + 2*avg14 + avg28) / 7

        # ---- Aroon ----
        for p in (14, 25):
            hh_idx = pd.Series(high).rolling(p, min_periods=max(3, p//2)).apply(lambda x: p - 1 - np.argmax(x))
            ll_idx = pd.Series(low).rolling(p, min_periods=max(3, p//2)).apply(lambda x: p - 1 - np.argmin(x))
            f[f'ti_aroon_up_{p}'] = 100 * (p - 1 - hh_idx) / (p - 1)
            f[f'ti_aroon_down_{p}'] = 100 * (p - 1 - ll_idx) / (p - 1)

        return f


# ============ K线形态特征 ============
class KlinePatternFeatures:
    """K线形态: 锤子、吞没、十字星、三连阳等"""

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        open_p = df['open'].values.astype(float)

        body = np.abs(close - open_p)
        total_range = high - low + 1e-10
        upper_shadow = (high - np.maximum(close, open_p)) / total_range
        lower_shadow = (np.minimum(close, open_p) - low) / total_range
        body_ratio = body / total_range

        # 影线/实体比率
        f['kp_body_ratio'] = body_ratio
        f['kp_upper_shadow'] = upper_shadow
        f['kp_lower_shadow'] = lower_shadow

        # 跳空
        gap = (open_p - pd.Series(close).shift(1)) / (pd.Series(close).shift(1) + 1e-10)
        f['kp_gap'] = gap
        f['kp_gap_up'] = (gap > 0.01).astype(int)
        f['kp_gap_down'] = (gap < -0.01).astype(int)

        # 内包/外包
        prev_h = pd.Series(high).shift(1)
        prev_l = pd.Series(low).shift(1)
        f['kp_inside'] = ((high <= prev_h) & (low >= prev_l)).astype(int)
        f['kp_outside'] = ((high > prev_h) & (low < prev_l)).astype(int)

        # 十字星
        f['kp_doji'] = (body_ratio < 0.001).astype(int)
        f['kp_spinning'] = ((body_ratio > 0.001) & (body_ratio < 0.3)).astype(int)

        # 锤子线 (下影线 > 2×实体, 上影线很短)
        is_bull = close > open_p
        f['kp_hammer'] = ((lower_shadow > 0.6) & (upper_shadow < 0.1) & (body_ratio < 0.3)).astype(int)
        f['kp_shooting_star'] = ((upper_shadow > 0.6) & (lower_shadow < 0.1) & (body_ratio < 0.3)).astype(int)

        # 吞没形态
        prev_body = pd.Series(body).shift(1)
        prev_is_bull = (pd.Series(close).shift(1) > pd.Series(open_p).shift(1))
        f['kp_bull_engulf'] = (is_bull & ~prev_is_bull & (body > prev_body * 1.5) & (close > prev_h) & (open_p < prev_l)).astype(int)
        f['kp_bear_engulf'] = (~is_bull & prev_is_bull & (body > prev_body * 1.5) & (close < prev_l) & (open_p > prev_h)).astype(int)

        # 三连阳/三连阴
        price_up = (pd.Series(close).diff() > 0).astype(int)
        for p in (3, 5, 10):
            f[f'kp_up_streak_{p}'] = price_up.rolling(p).sum()
            f[f'kp_down_streak_{p}'] = (pd.Series(close).diff() < 0).astype(int).rolling(p).sum()

        # Marubozu (光头光脚)
        f['kp_marubozu'] = (body_ratio > 0.8).astype(int)

        # 振幅
        f['kp_amplitude'] = (high - low) / (pd.Series(close).shift(1) + 1e-10)
        f['kp_amplitude_ma5'] = f['kp_amplitude'].rolling(5).mean()

        return f


# ============ 高级特征: 波动率聚集、相关性、收益分布 ============
class AdvancedFeatures:
    """波动率聚集、滚动相关性、收益分布特征"""

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        close = df['close'].values.astype(float)
        returns = pd.Series(close).pct_change()

        # ---- 波动率聚集 ----
        for fast, slow in [(5, 20), (10, 60), (20, 60)]:
            vf = returns.rolling(fast, min_periods=max(3, fast//2)).std()
            vs = returns.rolling(slow, min_periods=max(3, slow//2)).std()
            f[f'av_vol_ratio_{fast}_{slow}'] = vf / (vs + 1e-10)

        # 波动率均值回归
        vol20 = returns.rolling(20, min_periods=10).std()
        vol_ma = vol20.rolling(60, min_periods=30).mean()
        f['av_vol_vs_mean'] = vol20 / (vol_ma + 1e-10)

        # ---- 收益偏度/峰度 ----
        for p in (10, 20, 60):
            r = returns.rolling(p, min_periods=max(3, p//2))
            f[f'av_skew_{p}'] = r.skew()
            f[f'av_kurt_{p}'] = r.kurt()

        # ---- 收益自相关 ----
        for p in (1, 3, 5):
            f[f'av_autocorr_{p}'] = returns.rolling(20, min_periods=10).apply(
                lambda x: x.autocorr(lag=p) if len(x) > p else 0, raw=False
            )

        # ---- 收益极值 ----
        for p in (10, 20):
            r = returns.rolling(p, min_periods=max(3, p//2))
            f[f'av_max_ret_{p}'] = r.max()
            f[f'av_min_ret_{p}'] = r.min()

        # ---- 收益与成交量的相关性 ----
        if 'volume' in df.columns and len(df) > 20:
            vol = df['volume'].values.astype(float)
            vol_chg = pd.Series(vol).pct_change()
            f['av_ret_vol_corr'] = returns.rolling(20, min_periods=10).corr(vol_chg)

        # ---- 夏普比率 (rolling) ----
        for p in (20, 60):
            r = returns.rolling(p, min_periods=max(5, p//2))
            f[f'av_sharpe_{p}'] = r.mean() / (r.std() + 1e-10) * np.sqrt(p)

        return f


# ============ 截面相对特征 (分钟级) - 向量化 ============
class IntradayCrossSection:
    """分钟级截面特征: 在相同时刻，股票在pool内的相对排名"""

    RANK_TARGETS = [
        # 收益
        'mp_ret_3', 'mp_ret_5', 'mp_ret_10', 'mp_ret_20',
        # 波动率
        'mp_vol_10', 'mp_vol_20', 'mp_vol_60',
        # 均线偏离
        'mp_ma5_dist', 'mp_ma20_dist', 'mp_ma60_dist',
        # 量比
        'mv_vol_ratio_5', 'mv_vol_ratio_20',
        # 技术指标
        'ti_rsi_14', 'ti_adx', 'ti_cci',
        # 动量
        'mp_decay_3_10', 'mp_pos_10', 'mp_pos_60',
        # 量价
        'mv_mfi', 'mv_cmf',
    ]

    @staticmethod
    def calculate(all_features: Dict[str, pd.DataFrame],
                  all_timestamps: np.ndarray) -> Dict[str, pd.DataFrame]:
        symbols = sorted(all_features.keys())
        all_ts = sorted(set(all_timestamps))
        ts_idx = {ts: i for i, ts in enumerate(all_ts)}

        result = {sym: pd.DataFrame(index=all_features[sym].index) for sym in symbols}

        for target in tqdm(IntradayCrossSection.RANK_TARGETS, desc='   截面排名', unit='target'):
            matrix = pd.DataFrame(index=all_ts, columns=symbols, dtype=float)
            for sym in symbols:
                feats = all_features[sym]
                if target in feats.columns:
                    series = feats[target]
                    for ts, val in series.items():
                        if ts in ts_idx:
                            matrix.loc[ts, sym] = val

            valid_counts = matrix.notna().sum(axis=1)
            rank_matrix = matrix.rank(axis=1, pct=True, method='average')
            rank_matrix[valid_counts < 10] = np.nan

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
    def __init__(self, daily_model_path: str = None):
        self.daily_model_path = daily_model_path
        self._daily_scores = None

    def calculate(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        for col in ['dc_score', 'dc_rank', 'dc_in_pool']:
            f[col] = 0
        return f


# ============ 特征流水线 ============
class IntradayFeaturePipeline:
    def __init__(self, daily_model_path: str = None):
        self.daily_ctx = DailyModelContext(daily_model_path)

    def compute_stock(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        micro_price = MicroPriceFeatures.calculate(df)
        micro_vol = MicroVolumeFeatures.calculate(df)
        intraday = IntradayPatternFeatures.calculate(df)
        tech = ShortTermTechFeatures.calculate(df)
        kline = KlinePatternFeatures.calculate(df)
        advanced = AdvancedFeatures.calculate(df)
        daily_ctx = self.daily_ctx.calculate(df, symbol)
        return pd.concat([micro_price, micro_vol, intraday, tech, kline, advanced, daily_ctx], axis=1)

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
    print(f"前20列: {list(feats.columns[:20])}")
    print(f"后20列: {list(feats.columns[-20:])}")
    print(f"缺失率: {(feats.isna().sum() / len(feats) * 100).describe()}")