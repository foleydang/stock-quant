#!/usr/bin/env python3
"""
分钟级特征工程 v3 — 大幅扩展特征空间 (目标 460+ 特征)

v3 新增:
  - 更多收益率/波动率/均线周期 (覆盖更细粒度+更长周期)
  - 新指标: ROC/MOM/TRIX/BOP/StochRSI
  - 截面z-score (不只是rank)
  - 量价弹性、成交量集中度、异常成交量
  - 赫斯特指数、更多高级统计
  - 三白兵/启明星等K线组合形态
  - 日内时段×价格位置交互

设计原则:
  - 严格向后看: 只用当前及之前K线的数据
  - 多周期: 覆盖短/中/长周期, 让模型自动选择
  - 丰富性: 量价+技术+形态+截面+高级统计全覆盖
  - 命名: {category}_{name}_{period}

特征类别:
  1. MicroPrice   (~190) 收益率、波动率、均线、价格通道 (v3扩展)
  2. MicroVolume  (~60)  量比、MFI、CMF、量价交互、弹性 (v3扩展)
  3. Intraday     (~35)  时段效应、开盘/尾盘、星期、时段×位置 (v3扩展)
  4. TechInd      (~70)  RSI/MACD/KDJ/布林/ATR/ADX/CCI/ROC/MOM/TRIX/BOP (v3扩展)
  5. Pattern      (~25)  K线形态: 锤子/吞没/十字星/三白兵/启明星 (v3扩展)
  6. CrossSection (~50)  pool内截面排名 + z-score (v3扩展)
  7. DailyContext (~5)   日线模型输出
  8. Advanced     (~35)  波动率聚集、赫斯特指数、收益分布 (v3扩展)
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


def _mp(p):
    """Safe min_periods: ensures min_periods <= window"""
    return min(p, max(3, p // 2))


# ============ 微观价格特征 (v3扩展版) ============
class MicroPriceFeatures:
    """分钟级价格特征: 多周期收益率、波动率、均线、价格通道 (v3: 更细粒度+更长周期)"""

    RET_P = (1, 2, 3, 4, 5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 75, 80, 100, 120, 150, 200)
    RET_Z_P = (1, 2, 3, 4, 5, 8, 10, 12, 15, 20)  # 计算z-score的周期
    VOL_P = (2, 3, 5, 8, 10, 15, 20, 30, 50, 60, 80, 100, 120, 150)
    MA_P  = (3, 5, 8, 10, 15, 20, 25, 30, 50, 60, 80, 100, 120, 150, 200)
    POS_P = (3, 5, 8, 10, 15, 20, 30, 40, 60, 80, 120)

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        open_p = df['open'].values.astype(float)
        volume = df['volume'].values.astype(float)

        # ---- 1. 收益率 (多周期, v3扩展) ----
        for p in MicroPriceFeatures.RET_P:
            ret = pd.Series(close).pct_change(p)
            f[f'mp_ret_{p}'] = ret

        # z-score (v3: 扩展到更多周期)
        for p in MicroPriceFeatures.RET_Z_P:
            ret = pd.Series(close).pct_change(p)
            roll_mean = ret.rolling(20, min_periods=5).mean()
            roll_std = ret.rolling(20, min_periods=5).std()
            f[f'mp_ret_{p}_z'] = (ret - roll_mean) / (roll_std + 1e-10)

        # ---- 2. 对数收益率 (v3扩展) ----
        for p in (1, 2, 3, 5, 10, 15, 20, 30, 60, 120):
            f[f'mp_logret_{p}'] = np.log(pd.Series(close) / pd.Series(close).shift(p))

        # ---- 3. 波动率 (多周期, v3扩展) ----
        returns = pd.Series(close).pct_change()
        for p in MicroPriceFeatures.VOL_P:
            mp = min(p, _mp(p))
            vol = returns.rolling(p, min_periods=mp).std()
            f[f'mp_vol_{p}'] = vol

        # Parkinson 波动率 (v3扩展)
        for p in (5, 10, 20, 30, 60):
            hl = np.log(pd.Series(high) / pd.Series(low))
            f[f'mp_vol_park_{p}'] = np.sqrt(
                (hl ** 2).rolling(p, min_periods=_mp(p)).mean() / (4 * np.log(2))
            )

        # Garman-Klass 波动率 (v3扩展)
        for p in (5, 10, 20, 30, 60):
            oc = np.log(pd.Series(open_p) / pd.Series(close).shift(1))
            hl = np.log(pd.Series(high) / pd.Series(low))
            co = np.log(pd.Series(close) / pd.Series(open_p))
            gk = 0.5 * hl**2 - (2*np.log(2)-1) * co**2
            f[f'mp_vol_gk_{p}'] = np.sqrt(gk.rolling(p, min_periods=_mp(p)).mean())

        # 波动率变化 (v3扩展)
        for p in (5, 10, 20, 60, 120):
            vol = returns.rolling(p, min_periods=_mp(p)).std()
            f[f'mp_vol_chg_{p}'] = vol.diff(5) / (vol.shift(5) + 1e-10)

        # ---- 4. 均线系统 (v3扩展) ----
        for p in MicroPriceFeatures.MA_P:
            ma = pd.Series(close).rolling(p, min_periods=_mp(p)).mean()
            f[f'mp_ma{p}_dist'] = (close - ma) / (ma + 1e-10)
            f[f'mp_above_ma{p}'] = (close > ma).astype(int)

        # 均线斜率 (v3扩展)
        for p in (3, 5, 10, 20, 30, 60, 120):
            ma = pd.Series(close).rolling(p, min_periods=_mp(p)).mean()
            f[f'mp_ma{p}_slope'] = ma.diff(3) / (ma + 1e-10)

        # 均线交叉 (v3扩展)
        crosses = [(3, 10), (5, 10), (5, 20), (5, 60), (10, 20), (10, 60), (20, 60), (20, 120), (60, 120)]
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
        ma120 = pd.Series(close).rolling(120, min_periods=60).mean()
        f['mp_ma_bullish'] = ((ma5 > ma10) & (ma10 > ma20) & (ma20 > ma60)).astype(int)
        f['mp_ma_bearish'] = ((ma5 < ma10) & (ma10 < ma20) & (ma20 < ma60)).astype(int)
        f['mp_ma_bullish_strong'] = ((ma5 > ma10) & (ma10 > ma20) & (ma20 > ma60) & (ma60 > ma120)).astype(int)
        f['mp_ma_bearish_strong'] = ((ma5 < ma10) & (ma10 < ma20) & (ma20 < ma60) & (ma60 < ma120)).astype(int)

        # ---- 5. 价格通道 / 位置 (v3扩展) ----
        for p in MicroPriceFeatures.POS_P:
            roll_high = pd.Series(high).rolling(p, min_periods=_mp(p)).max()
            roll_low = pd.Series(low).rolling(p, min_periods=_mp(p)).min()
            f[f'mp_pos_{p}'] = (close - roll_low) / (roll_high - roll_low + 1e-10)
            f[f'mp_high_dist_{p}'] = (close - roll_high) / (roll_high + 1e-10)
            f[f'mp_low_dist_{p}'] = (close - roll_low) / (roll_low + 1e-10)

        # 突破信号 (v3扩展)
        for p in (10, 20, 60, 120):
            roll_high = pd.Series(high).rolling(p, min_periods=_mp(p)).max()
            roll_low = pd.Series(low).rolling(p, min_periods=_mp(p)).min()
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
            prev_high = np.roll(day_high, 1); prev_high[0] = day_high[0]
            prev_low = np.roll(day_low, 1); prev_low[0] = day_low[0]
            f['mp_day_new_high'] = (day_high > prev_high).astype(int)
            f['mp_day_new_low'] = (day_low < prev_low).astype(int)
            # v3新增: 日内回撤
            f['mp_day_drawdown'] = (day_high - close) / (day_high + 1e-10)
        else:
            f['mp_day_open_ret'] = 0
            f['mp_day_pos'] = 0.5
            f['mp_day_new_high'] = 0
            f['mp_day_new_low'] = 0
            f['mp_day_drawdown'] = 0

        # ---- 7. 振幅 (v3扩展) ----
        for p in (3, 5, 10, 20, 60, 120):
            f[f'mp_range_{p}'] = (pd.Series(high).rolling(p).max() -
                                   pd.Series(low).rolling(p).min()) / \
                                  (pd.Series(close).rolling(p).mean() + 1e-10)

        # ---- 8. 连续涨跌 (v3扩展) ----
        up = (returns > 0).astype(int)
        down = (returns < 0).astype(int)
        for p in (3, 5, 10, 20, 60):
            f[f'mp_up_streak_{p}'] = up.rolling(p).sum()
            f[f'mp_down_streak_{p}'] = down.rolling(p).sum()
        # v3新增: 净涨跌比
        for p in (5, 10, 20):
            f[f'mp_up_down_ratio_{p}'] = (up.rolling(p).sum() + 1) / (down.rolling(p).sum() + 1)

        # ---- 9. 加速/减速 (v3扩展) ----
        for p in (2, 3, 5, 10, 20):
            ret_p = pd.Series(close).pct_change(p)
            f[f'mp_accel_{p}'] = ret_p.diff(p)

        ret1 = pd.Series(close).pct_change(1)
        ret3 = pd.Series(close).pct_change(3)
        ret5 = pd.Series(close).pct_change(5)
        ret10 = pd.Series(close).pct_change(10)
        ret20 = pd.Series(close).pct_change(20)
        ret60 = pd.Series(close).pct_change(60)
        f['mp_decay_3_10'] = ret3 - ret10
        f['mp_decay_10_20'] = ret10 - ret20
        f['mp_decay_5_20'] = ret5 - ret20
        f['mp_decay_20_60'] = ret20 - ret60  # v3新增
        f['mp_decay_1_5'] = ret1 - ret5      # v3新增

        # ---- 10. 开盘价特征 (v3新增) ----
        f['mp_open_ret'] = (open_p - pd.Series(close).shift(1)) / (pd.Series(close).shift(1) + 1e-10)
        f['mp_open_vs_prev_close'] = (open_p / pd.Series(close).shift(1)) - 1

        return f


# ============ 微观成交量特征 (v3扩展版) ============
class MicroVolumeFeatures:
    """成交量特征: 量比、MFI、CMF、量价交互、弹性、集中度 (v3扩展)"""

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float)
        vol = pd.Series(volume)

        # ---- 1. 量比 (v3扩展) ----
        for p in (2, 3, 4, 5, 8, 10, 15, 20, 30, 40, 60, 80, 120):
            ma = vol.rolling(p, min_periods=_mp(p)).mean()
            f[f'mv_vol_ratio_{p}'] = vol / (ma + 1e-10)

        # ---- 2. 量趋势 (v3扩展) ----
        for fast, slow in [(3, 20), (5, 20), (5, 60), (10, 60), (10, 120), (20, 60), (20, 120)]:
            f[f'mv_vol_trend_{fast}_{slow}'] = vol.rolling(fast, min_periods=max(3, fast//2)).mean() / \
                                                (vol.rolling(slow, min_periods=max(3, slow//2)).mean() + 1e-10)

        # ---- 3. 量变化 (v3扩展) ----
        for p in (1, 2, 3, 5, 10, 20):
            f[f'mv_vol_chg_{p}'] = vol.pct_change(p)

        # ---- 4. 量波动 (v3扩展) ----
        for p in (5, 10, 20, 30, 60):
            f[f'mv_vol_std_{p}'] = vol.rolling(p, min_periods=_mp(p)).std() / \
                                    (vol.rolling(p, min_periods=_mp(p)).mean() + 1e-10)

        # ---- 5. 量价关系 (v3扩展) ----
        price_up = (pd.Series(close).diff() > 0).astype(int)
        vol_up = (vol.diff() > 0).astype(int)
        for p in (3, 5, 10, 20):
            f[f'mv_div_{p}'] = (price_up != vol_up).rolling(p, min_periods=_mp(p)).mean()
            f[f'mv_conf_{p}'] = (price_up == vol_up).rolling(p, min_periods=_mp(p)).mean()

        # ---- 6. 放量/缩量 (v3增强) ----
        vol_ma20 = vol.rolling(20, min_periods=10).mean()
        vol_ma60 = vol.rolling(60, min_periods=30).mean()
        vol_ma120 = vol.rolling(120, min_periods=60).mean()
        f['mv_volume_spike'] = (vol > 2 * vol_ma20).astype(int)
        f['mv_volume_dry'] = (vol < 0.3 * vol_ma20).astype(int)
        f['mv_volume_spike_60'] = (vol > 2 * vol_ma60).astype(int)
        f['mv_volume_spike_120'] = (vol > 2 * vol_ma120).astype(int)  # v3新增
        f['mv_volume_spike_ratio'] = (vol / (vol_ma20 + 1e-10)).clip(0, 5)  # v3新增: 连续值

        # ---- 7. OBV (v3增强) ----
        obv = np.zeros(len(close))
        obv[0] = volume[0]
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]
        obv_s = pd.Series(obv)
        for p in (3, 5, 10, 20):
            f[f'mv_obv_chg_{p}'] = obv_s.pct_change(p).clip(-5, 5)
        obv_ma20 = obv_s.rolling(20, min_periods=10).mean()
        f['mv_obv_vs_ma'] = obv / (obv_ma20.values + 1e-10)
        obv_ma5 = obv_s.rolling(5, min_periods=3).mean()
        obv_ma60 = obv_s.rolling(60, min_periods=30).mean()
        f['mv_obv_trend'] = obv_ma5 / (obv_ma60 + 1e-10) - 1
        f['mv_obv_slope'] = (obv_s.diff(5) / (obv_ma20 + 1e-10)).clip(-1, 1)
        # v3新增: OBV背离 (价格新高但OBV不新高)
        price_high_20 = pd.Series(close).rolling(20, min_periods=10).max()
        obv_high_20 = obv_s.rolling(20, min_periods=10).max()
        f['mv_obv_divergence'] = ((close >= price_high_20 * 0.99) & (obv < obv_high_20 * 0.95)).astype(int)

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

        # ---- 11. 量价弹性 (v3新增) ----
        # 价格变化率 / 成交量变化率, 反映量价敏感度
        for p in (5, 10, 20):
            price_chg = pd.Series(close).pct_change(p)
            vol_chg_p = vol.pct_change(p)
            elasticity = price_chg / (vol_chg_p.abs() + 1e-10)
            f[f'mv_elasticity_{p}'] = elasticity.clip(-10, 10)

        # ---- 12. 成交量集中度 (v3新增) ----
        # 最大单根K线成交量 / 总成交量
        for p in (20, 60):
            max_vol = vol.rolling(p, min_periods=_mp(p)).max()
            sum_vol = vol.rolling(p, min_periods=_mp(p)).sum()
            f[f'mv_concentration_{p}'] = max_vol / (sum_vol + 1e-10)

        # ---- 13. 异常成交量 (v3新增) ----
        # 当前成交量偏离均值的标准差倍数
        for p in (20, 60):
            vol_mean = vol.rolling(p, min_periods=_mp(p)).mean()
            vol_std_p = vol.rolling(p, min_periods=_mp(p)).std()
            f[f'mv_vol_zscore_{p}'] = ((vol - vol_mean) / (vol_std_p + 1e-10)).clip(-5, 5)

        # ---- 14. 主动买卖量 (v3新增) ----
        # 上涨时的成交量 vs 下跌时的成交量
        for p in (5, 10, 20):
            buy_vol = (vol * (pd.Series(close).diff() > 0)).rolling(p, min_periods=_mp(p)).sum()
            sell_vol = (vol * (pd.Series(close).diff() < 0)).rolling(p, min_periods=_mp(p)).sum()
            f[f'mv_buy_vol_ratio_{p}'] = buy_vol / (buy_vol + sell_vol + 1e-10)

        return f


# ============ 日内模式特征 (v3扩展版) ============
class IntradayPatternFeatures:
    """日内模式: 时段效应、开盘/收盘、星期、月份、时段×位置 (v3扩展)"""

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)

        if 'date' not in df.columns:
            for col in ['ip_morning', 'ip_afternoon', 'ip_opening_30m', 'ip_closing_30m',
                        'ip_hour_0', 'ip_hour_1', 'ip_hour_2', 'ip_hour_3',
                        'ip_dow_0', 'ip_dow_1', 'ip_dow_2', 'ip_dow_3', 'ip_dow_4',
                        'ip_month', 'ip_minutes', 'ip_first_hour', 'ip_last_hour',
                        'ip_midday', 'ip_bar_of_day', 'ip_bar_pct', 'ip_vol_pct']:
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
        max_bars = f['ip_bar_of_day'].groupby(day_grp).transform('max')
        f['ip_bar_pct'] = f['ip_bar_of_day'] / (max_bars + 1e-10)

        # 上午/下午成交量占比 (当日累计)
        morning_mask = (total_min >= 570) & (total_min < 690)
        if 'volume' in df.columns:
            cum_vol = df.groupby(day_grp)['volume'].cumsum()
            total_vol = df.groupby(day_grp)['volume'].transform('sum')
            f['ip_vol_pct'] = cum_vol / (total_vol + 1e-10)

        # ---- v3新增: 时段收益 ----
        close = df['close'].values.astype(float)
        # 过去30分钟收益
        bar_ret_1 = pd.Series(close).pct_change(1)
        f['ip_ret_30m'] = bar_ret_1
        # 过去60分钟收益
        f['ip_ret_60m'] = pd.Series(close).pct_change(2)
        # 过去120分钟收益
        f['ip_ret_120m'] = pd.Series(close).pct_change(4)

        # ---- v3新增: 开盘至今收益 ----
        if 'open' in df.columns:
            day_open = df.groupby(day_grp)['open'].transform('first').values
            f['ip_open_to_now'] = (close - day_open) / (day_open + 1e-10)

        # ---- v3新增: 日内振幅 (从开盘到当前) ----
        if 'high' in df.columns and 'low' in df.columns:
            day_high = df.groupby(day_grp)['high'].cummax().values
            day_low = df.groupby(day_grp)['low'].cummin().values
            f['ip_day_amplitude'] = (day_high - day_low) / (day_open + 1e-10)

        # ---- v3新增: 上午/下午成交量差 ----
        if 'volume' in df.columns:
            morning_vol = df['volume'].where(morning_mask, 0).groupby(day_grp).cumsum()
            afternoon_vol = df['volume'].where(~morning_mask, 0).groupby(day_grp).cumsum()
            f['ip_morning_vol_ratio'] = morning_vol / (morning_vol + afternoon_vol + 1e-10)

        # ---- v3新增: 时段×星期交互 ----
        f['ip_morning_monday'] = (f['ip_morning'] & f['ip_dow_0']).astype(int)
        f['ip_morning_friday'] = (f['ip_morning'] & f['ip_dow_4']).astype(int)
        f['ip_afternoon_monday'] = (f['ip_afternoon'] & f['ip_dow_0']).astype(int)
        f['ip_afternoon_friday'] = (f['ip_afternoon'] & f['ip_dow_4']).astype(int)
        f['ip_opening_monday'] = (f['ip_opening_30m'] & f['ip_dow_0']).astype(int)
        f['ip_closing_friday'] = (f['ip_closing_30m'] & f['ip_dow_4']).astype(int)

        return f


# ============ 短周期技术指标 (v3扩展版) ============
class ShortTermTechFeatures:
    """技术指标: RSI/MACD/KDJ/BB/ATR/ADX/CCI/ROC/MOM/TRIX/BOP/StochRSI (v3扩展)"""

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        open_p = df['open'].values.astype(float)
        volume = df['volume'].values.astype(float)

        # ---- RSI (v3扩展) ----
        for p in (6, 9, 14, 21, 24):
            delta = pd.Series(close).diff()
            gain = delta.where(delta > 0, 0).rolling(p, min_periods=_mp(p)).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(p, min_periods=_mp(p)).mean()
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

        # ---- KDJ (v3扩展) ----
        for p in (5, 9, 18, 27):
            low_min = pd.Series(low).rolling(p, min_periods=_mp(p)).min()
            high_max = pd.Series(high).rolling(p, min_periods=_mp(p)).max()
            rsv = (close - low_min) / (high_max - low_min + 1e-10) * 100
            k = rsv.ewm(com=2).mean()
            d = k.ewm(com=2).mean()
            j = 3 * k - 2 * d
            f[f'ti_kdj_k_{p}'] = k
            f[f'ti_kdj_d_{p}'] = d
            f[f'ti_kdj_j_{p}'] = j

        # ---- 布林带 (v3扩展) ----
        for p in (10, 20, 60, 120):
            ma = pd.Series(close).rolling(p, min_periods=_mp(p)).mean()
            std = pd.Series(close).rolling(p, min_periods=_mp(p)).std()
            upper = ma + 2 * std
            lower = ma - 2 * std
            f[f'ti_bb{p}_width'] = (upper - lower) / (ma + 1e-10)
            f[f'ti_bb{p}_pos'] = (close - lower) / (upper - lower + 1e-10)
            f[f'ti_bb{p}_squeeze'] = (f[f'ti_bb{p}_width'] < 0.05).astype(int)

        # ---- ATR (v3扩展) ----
        tr = pd.concat([
            pd.Series(high) - pd.Series(low),
            (pd.Series(high) - pd.Series(close).shift(1)).abs(),
            (pd.Series(close).shift(1) - pd.Series(low)).abs()
        ], axis=1).max(axis=1)
        for p in (7, 10, 14, 20, 30):
            f[f'ti_atr_{p}'] = tr.rolling(p, min_periods=_mp(p)).mean()
        f['ti_atr_ratio'] = f['ti_atr_14'] / pd.Series(close)

        # ---- ADX (v3扩展) ----
        for adx_p in (7, 14, 25):
            atr_p = tr.rolling(adx_p, min_periods=max(3, adx_p//2)).mean()
            plus_dm = pd.Series(high).diff()
            minus_dm = -pd.Series(low).diff()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            plus_di = 100 * (plus_dm.rolling(adx_p, min_periods=max(3, adx_p//2)).mean() / (atr_p + 1e-10))
            minus_di = 100 * (minus_dm.rolling(adx_p, min_periods=max(3, adx_p//2)).mean() / (atr_p + 1e-10))
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            f[f'ti_adx_{adx_p}'] = dx.rolling(adx_p, min_periods=max(3, adx_p//2)).mean()
            f[f'ti_adx_trend_{adx_p}'] = np.where(plus_di > minus_di, f[f'ti_adx_{adx_p}'], -f[f'ti_adx_{adx_p}'])

        # 保持向后兼容
        if 'ti_adx_14' in f.columns:
            f['ti_adx'] = f['ti_adx_14']
            f['ti_adx_trend'] = f['ti_adx_trend_14']

        # ---- CCI (v3扩展) ----
        tp = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
        for p in (14, 20, 50):
            f[f'ti_cci_{p}'] = (tp - tp.rolling(p, min_periods=_mp(p)).mean()) / \
                               (0.015 * tp.rolling(p, min_periods=_mp(p)).std() + 1e-10)
        # 保持向后兼容
        f['ti_cci'] = f['ti_cci_20']

        # ---- Williams %R (v3扩展) ----
        for p in (6, 10, 14, 20):
            hh = pd.Series(high).rolling(p, min_periods=_mp(p)).max()
            ll = pd.Series(low).rolling(p, min_periods=_mp(p)).min()
            f[f'ti_wr_{p}'] = (hh - close) / (hh - ll + 1e-10) * -100

        # ---- Ultimate Oscillator ----
        bp = close - np.minimum(low, pd.Series(close).shift(1))
        tr_uo = np.maximum(high, pd.Series(close).shift(1)) - np.minimum(low, pd.Series(close).shift(1))
        avg7 = pd.Series(bp).rolling(7, min_periods=4).sum() / pd.Series(tr_uo).rolling(7, min_periods=4).sum()
        avg14 = pd.Series(bp).rolling(14, min_periods=7).sum() / pd.Series(tr_uo).rolling(14, min_periods=7).sum()
        avg28 = pd.Series(bp).rolling(28, min_periods=14).sum() / pd.Series(tr_uo).rolling(28, min_periods=14).sum()
        f['ti_uo'] = 100 * (4*avg7 + 2*avg14 + avg28) / 7

        # ---- Aroon (v3扩展) ----
        for p in (7, 14, 25, 50):
            hh_idx = pd.Series(high).rolling(p, min_periods=_mp(p)).apply(lambda x: p - 1 - np.argmax(x))
            ll_idx = pd.Series(low).rolling(p, min_periods=_mp(p)).apply(lambda x: p - 1 - np.argmin(x))
            f[f'ti_aroon_up_{p}'] = 100 * (p - 1 - hh_idx) / (p - 1)
            f[f'ti_aroon_down_{p}'] = 100 * (p - 1 - ll_idx) / (p - 1)

        # ---- v3新增: ROC (Rate of Change) ----
        for p in (5, 10, 20, 60):
            f[f'ti_roc_{p}'] = (close - pd.Series(close).shift(p)) / (pd.Series(close).shift(p) + 1e-10) * 100

        # ---- v3新增: MOM (Momentum) ----
        for p in (5, 10, 20):
            f[f'ti_mom_{p}'] = close - pd.Series(close).shift(p)

        # ---- v3新增: TRIX ----
        for p in (12, 20):
            ema1 = pd.Series(close).ewm(span=p, adjust=False).mean()
            ema2 = ema1.ewm(span=p, adjust=False).mean()
            ema3 = ema2.ewm(span=p, adjust=False).mean()
            f[f'ti_trix_{p}'] = ema3.pct_change() * 100

        # ---- v3新增: BOP (Balance of Power) ----
        f['ti_bop'] = (close - pd.Series(open_p)) / (pd.Series(high) - pd.Series(low) + 1e-10)
        f['ti_bop_ma5'] = f['ti_bop'].rolling(5, min_periods=3).mean()

        # ---- v3新增: StochRSI ----
        if 'ti_rsi_14' in f.columns:
            rsi14 = f['ti_rsi_14']
            rsi_min = rsi14.rolling(14, min_periods=7).min()
            rsi_max = rsi14.rolling(14, min_periods=7).max()
            f['ti_stochrsi'] = (rsi14 - rsi_min) / (rsi_max - rsi_min + 1e-10)
            f['ti_stochrsi_k'] = f['ti_stochrsi'].rolling(3, min_periods=2).mean()
            f['ti_stochrsi_d'] = f['ti_stochrsi_k'].rolling(3, min_periods=2).mean()

        return f


# ============ K线形态特征 (v3扩展版) ============
class KlinePatternFeatures:
    """K线形态: 锤子、吞没、十字星、三白兵/三乌鸦、启明星/黄昏星 (v3扩展)"""

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

        # 锤子线
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

        # Marubozu
        f['kp_marubozu'] = (body_ratio > 0.8).astype(int)

        # 振幅
        f['kp_amplitude'] = (high - low) / (pd.Series(close).shift(1) + 1e-10)
        f['kp_amplitude_ma5'] = f['kp_amplitude'].rolling(5).mean()

        # ---- v3新增: 三白兵 (Three White Soldiers) ----
        c_shift = pd.Series(close)
        o_shift = pd.Series(open_p)
        three_white = ((c_shift > o_shift) &
                       (c_shift.shift(1) > o_shift.shift(1)) &
                       (c_shift.shift(2) > o_shift.shift(2)) &
                       (c_shift > c_shift.shift(1)) &
                       (c_shift.shift(1) > c_shift.shift(2)))
        f['kp_three_white'] = three_white.astype(int)

        # ---- v3新增: 三乌鸦 (Three Black Crows) ----
        three_black = ((c_shift < o_shift) &
                       (c_shift.shift(1) < o_shift.shift(1)) &
                       (c_shift.shift(2) < o_shift.shift(2)) &
                       (c_shift < c_shift.shift(1)) &
                       (c_shift.shift(1) < c_shift.shift(2)))
        f['kp_three_black'] = three_black.astype(int)

        # ---- v3新增: 启明星 (Morning Star) ----
        br_s = pd.Series(body_ratio)
        morning_star = ((c_shift.shift(2) < o_shift.shift(2)) &
                        (br_s.shift(2) > 0.3) &
                        (br_s.shift(1) < 0.3) &
                        (c_shift.shift(1) < c_shift.shift(2)) &
                        (c_shift > o_shift) &
                        (br_s > 0.3) &
                        (c_shift > c_shift.shift(1)))
        f['kp_morning_star'] = morning_star.astype(int)

        # ---- v3新增: 黄昏星 (Evening Star) ----
        evening_star = ((c_shift.shift(2) > o_shift.shift(2)) &
                        (br_s.shift(2) > 0.3) &
                        (br_s.shift(1) < 0.3) &
                        (c_shift.shift(1) > c_shift.shift(2)) &
                        (c_shift < o_shift) &
                        (br_s > 0.3) &
                        (c_shift < c_shift.shift(1)))
        f['kp_evening_star'] = evening_star.astype(int)

        # ---- v3新增: 身怀六甲 (Harami) ----
        f['kp_bull_harami'] = (is_bull & ~prev_is_bull & (body < prev_body * 0.5) & (open_p > prev_l) & (close < prev_h)).astype(int)
        f['kp_bear_harami'] = (~is_bull & prev_is_bull & (body < prev_body * 0.5) & (open_p < prev_h) & (close > prev_l)).astype(int)

        return f


# ============ 高级特征: 波动率聚集、赫斯特指数、收益分布 (v3扩展) ============
class AdvancedFeatures:
    """波动率聚集、滚动相关性、赫斯特指数、收益分布特征 (v3扩展)"""

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        close = df['close'].values.astype(float)
        returns = pd.Series(close).pct_change()

        # ---- 波动率聚集 (v3扩展) ----
        for fast, slow in [(3, 20), (5, 20), (5, 60), (10, 20), (10, 60), (20, 60), (20, 120)]:
            vf = returns.rolling(fast, min_periods=max(3, fast//2)).std()
            vs = returns.rolling(slow, min_periods=max(3, slow//2)).std()
            f[f'av_vol_ratio_{fast}_{slow}'] = vf / (vs + 1e-10)

        # 波动率均值回归
        vol20 = returns.rolling(20, min_periods=10).std()
        vol_ma = vol20.rolling(60, min_periods=30).mean()
        f['av_vol_vs_mean'] = vol20 / (vol_ma + 1e-10)

        # ---- 收益偏度/峰度 (v3扩展) ----
        for p in (10, 20, 60, 120):
            r = returns.rolling(p, min_periods=_mp(p))
            f[f'av_skew_{p}'] = r.skew()
            f[f'av_kurt_{p}'] = r.kurt()

        # ---- 收益自相关 (v3扩展) ----
        for p in (1, 2, 3, 5, 10):
            f[f'av_autocorr_{p}'] = returns.rolling(20, min_periods=10).apply(
                lambda x: x.autocorr(lag=p) if len(x) > p else 0, raw=False
            )

        # ---- 收益极值 (v3扩展) ----
        for p in (5, 10, 20, 60):
            r = returns.rolling(p, min_periods=_mp(p))
            f[f'av_max_ret_{p}'] = r.max()
            f[f'av_min_ret_{p}'] = r.min()

        # ---- 收益与成交量的相关性 ----
        if 'volume' in df.columns and len(df) > 20:
            vol = df['volume'].values.astype(float)
            vol_chg = pd.Series(vol).pct_change()
            f['av_ret_vol_corr'] = returns.rolling(20, min_periods=10).corr(vol_chg)

        # ---- 夏普比率 (rolling, v3扩展) ----
        for p in (20, 60, 120):
            r = returns.rolling(p, min_periods=max(5, p//2))
            f[f'av_sharpe_{p}'] = r.mean() / (r.std() + 1e-10) * np.sqrt(p)

        # ---- v3新增: 赫斯特指数 (Hurst Exponent) ----
        # 使用简化版: 重标极差法, 窗口内计算log(RS) vs log(n)
        def hurst_approx(series, window=60):
            """近似赫斯特指数: 用滚动窗口内极差/标准差比值"""
            if len(series) < window:
                return np.full(len(series), 0.5)
            result = np.full(len(series), np.nan)
            for i in range(window, len(series)):
                seg = series[max(0, i-window):i]
                if len(seg) < 10:
                    continue
                # 简化: R/S 方法
                mean_adj = seg - np.nanmean(seg)
                cum_dev = np.cumsum(mean_adj)
                R = np.nanmax(cum_dev) - np.nanmin(cum_dev)
                S = np.nanstd(seg)
                if S > 0:
                    # log(R/S) / log(n) 近似Hurst
                    result[i] = np.log(R / S) / np.log(len(seg))
            return np.clip(np.nan_to_num(result, nan=0.5), 0, 1)

        f['av_hurst_60'] = hurst_approx(close, 60)
        f['av_hurst_120'] = hurst_approx(close, 120)

        # ---- v3新增: 收益-波动率相关性 ----
        for p in (20, 60):
            ret = returns.rolling(p, min_periods=_mp(p))
            vol = returns.rolling(p, min_periods=_mp(p)).std()
            f[f'av_ret_vol_corr_{p}'] = ret.corr(vol)

        # ---- v3新增: 波动率期限结构 ----
        vol5 = returns.rolling(5, min_periods=3).std()
        vol20_2 = returns.rolling(20, min_periods=10).std()
        vol60 = returns.rolling(60, min_periods=30).std()
        f['av_vol_term_5_20'] = vol5 / (vol20_2 + 1e-10)
        f['av_vol_term_20_60'] = vol20_2 / (vol60 + 1e-10)

        return f


# ============ 截面相对特征 (分钟级, v3扩展) ============
class IntradayCrossSection:
    """分钟级截面特征: 在相同时刻，股票在pool内的相对排名 + z-score (v3扩展)"""

    RANK_TARGETS = [
        # 收益
        'mp_ret_1', 'mp_ret_3', 'mp_ret_5', 'mp_ret_10', 'mp_ret_15', 'mp_ret_20', 'mp_ret_60',
        # 波动率
        'mp_vol_5', 'mp_vol_10', 'mp_vol_20', 'mp_vol_60',
        # 均线偏离
        'mp_ma5_dist', 'mp_ma10_dist', 'mp_ma20_dist', 'mp_ma60_dist', 'mp_ma120_dist',
        # 量比
        'mv_vol_ratio_5', 'mv_vol_ratio_10', 'mv_vol_ratio_20', 'mv_vol_ratio_60',
        # 技术指标
        'ti_rsi_14', 'ti_adx', 'ti_cci',
        'ti_roc_10', 'ti_roc_20',  # v3新增
        'ti_mom_10', 'ti_mom_20',  # v3新增
        'ti_wr_14',                 # v3新增
        # 动量/位置
        'mp_decay_3_10', 'mp_pos_10', 'mp_pos_60',
        'mp_accel_5', 'mp_accel_10',  # v3新增
        # 量价
        'mv_mfi', 'mv_cmf',
        'mv_elasticity_10', 'mv_elasticity_20',  # v3新增
        # 日内
        'mp_day_open_ret', 'mp_day_pos',  # v3新增
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

            # v3: rank + z-score
            rank_matrix = matrix.rank(axis=1, pct=True, method='average')
            rank_matrix[valid_counts < 10] = np.nan

            col_rank = f'ics_rank_{target}'
            for sym in symbols:
                series = rank_matrix[sym].dropna()
                if len(series) > 0:
                    if col_rank not in result[sym].columns:
                        result[sym][col_rank] = np.nan
                    result[sym].loc[series.index, col_rank] = series.values

            # v3新增: z-score截面
            row_mean = matrix.mean(axis=1)
            row_std = matrix.std(axis=1)
            z_matrix = matrix.sub(row_mean, axis=0).div(row_std + 1e-10, axis=0)
            z_matrix[valid_counts < 10] = np.nan

            col_z = f'ics_z_{target}'
            for sym in symbols:
                series = z_matrix[sym].dropna()
                if len(series) > 0:
                    if col_z not in result[sym].columns:
                        result[sym][col_z] = np.nan
                    result[sym].loc[series.index, col_z] = series.values

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