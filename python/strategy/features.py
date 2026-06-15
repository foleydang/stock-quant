#!/usr/bin/env python3
"""
特征工程 v5 — 对标 Qlib Alpha158/360 + 截面增强 + 宏观环境 + LSTM时序

设计原则:
  - 严格向后看: 所有特征只用 ≤ 当前日期的数据
  - 层次化: 价格 → 成交量 → 形态 → 动量 → 截面 → 交互 → 市场
  - 可配置: 通过 periods 参数控制特征计算周期
  - 命名规范: {category}_{name}_{period} (如 price_ret_5, vol_ratio_20)

特征类别:
  1. Price   (~120) 收益率、波动率、均线、RSI、MACD、KDJ、布林、ATR、ADX
  2. Volume  (~30)  成交量比率、OBV、量趋势、换手率
  3. Pattern (~15)  K线形态、影线、跳空、突破
  4. Momentum(~15)  动量加速度、衰减、二阶变化
  5. CrossSection(~20) 截面排名 (行业内/全市场)
  6. Interaction(~15) 特征交互 (量价共振等)
  7. Market  (~15)  北向资金、大盘、板块
  8. Sentiment(~10) 情绪数据 (龙虎榜、涨跌停等)
"""

import numpy as np
import pandas as pd
import sqlite3
import os
from typing import Dict, List, Optional, Tuple

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/stock_data.db')

# 延迟导入, 避免循环依赖
def _get_macro_features():
    from strategy.macro_features import MacroFeatures
    return MacroFeatures

# ============ 默认周期配置 ============
DEFAULT_PRICE_PERIODS = [1, 2, 3, 5, 10, 15, 20, 30, 40, 60, 80, 100, 120, 200, 250]
DEFAULT_VOL_PERIODS = [5, 10, 20, 30, 60, 120]
DEFAULT_MA_PERIODS = [5, 10, 20, 30, 60, 80, 100, 120, 200, 250]
DEFAULT_MA_CROSSES = [(5, 10), (10, 20), (20, 60), (60, 120), (120, 200)]
DEFAULT_RSI_PERIODS = [6, 14, 24, 50, 100]
DEFAULT_BB_PERIODS = [20, 30, 60]
DEFAULT_ATR_PERIODS = [10, 14, 20, 60]
DEFAULT_VOLATILITY_PERIODS = [5, 10, 20, 30, 40, 60, 80, 100, 120, 200]


# ============ 价格特征 ============
class PriceFeatures:
    """价格特征: 收益率、波动率、均线、技术指标"""

    @staticmethod
    def calculate(df: pd.DataFrame,
                  ret_periods: List[int] = None,
                  ma_periods: List[int] = None,
                  vol_periods: List[int] = None,
                  rsi_periods: List[int] = None,
                  bb_periods: List[int] = None,
                  atr_periods: List[int] = None) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)

        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float)
        open_price = df['open'].values.astype(float)

        ret_periods = ret_periods or DEFAULT_PRICE_PERIODS
        ma_periods = ma_periods or DEFAULT_MA_PERIODS
        vol_periods = vol_periods or DEFAULT_VOLATILITY_PERIODS
        rsi_periods = rsi_periods or DEFAULT_RSI_PERIODS
        bb_periods = bb_periods or DEFAULT_BB_PERIODS
        atr_periods = atr_periods or DEFAULT_ATR_PERIODS

        # ---- 1. 收益率 ----
        for p in ret_periods:
            f[f'price_ret_{p}'] = pd.Series(close).pct_change(p)

        # ---- 2. 对数收益率 ----
        for p in [1, 3, 5, 10, 20, 60]:
            f[f'price_logret_{p}'] = np.log(pd.Series(close) / pd.Series(close).shift(p))

        # ---- 3. 波动率 ----
        returns = pd.Series(close).pct_change()
        for p in vol_periods:
            f[f'price_vol_{p}'] = returns.rolling(p).std()

        # Parkinson 波动率
        f['price_parkinson_vol'] = np.sqrt(
            (np.log(pd.Series(high) / pd.Series(low)) ** 2).rolling(20).mean() / (4 * np.log(2))
        )

        # 波动率变化率 (regime change)
        for p in [20, 60]:
            vol = returns.rolling(p).std()
            f[f'price_vol_chg_{p}'] = vol.diff(5) / (vol.shift(5) + 1e-10)

        # ---- 4. 均线系统 ----
        for p in ma_periods:
            ma = pd.Series(close).rolling(p).mean()
            f[f'price_ma{p}_ratio'] = close / ma - 1
            f[f'price_above_ma{p}'] = (close > ma).astype(int)

        # 均线斜率 (趋势方向)
        for p in [5, 10, 20, 60]:
            ma = pd.Series(close).rolling(p).mean()
            f[f'price_ma{p}_slope'] = ma.diff(5) / (ma + 1e-10)

        # 均线距离
        for fast, slow in DEFAULT_MA_CROSSES:
            ma_f = pd.Series(close).rolling(fast).mean()
            ma_s = pd.Series(close).rolling(slow).mean()
            f[f'price_ma{fast}_{slow}_dist'] = ma_f / ma_s - 1

        # ---- 5. RSI ----
        for p in rsi_periods:
            delta = pd.Series(close).diff()
            gain = delta.where(delta > 0, 0).rolling(p).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(p).mean()
            rs = gain / (loss + 1e-10)
            f[f'price_rsi_{p}'] = 100 - (100 / (1 + rs))

        # RSI 变化率
        rsi14 = f['price_rsi_14']
        f['price_rsi_14_chg'] = rsi14.diff(3)

        # ---- 6. MACD ----
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, adjust=False).mean()
        f['price_macd'] = macd_line
        f['price_macd_signal'] = signal
        f['price_macd_hist'] = macd_line - signal
        f['price_macd_hist_chg'] = f['price_macd_hist'].diff()

        # ---- 7. KDJ ----
        low_min = pd.Series(low).rolling(9).min()
        high_max = pd.Series(high).rolling(9).max()
        rsv = (close - low_min) / (high_max - low_min + 1e-10) * 100
        k = rsv.ewm(com=2).mean()
        d = k.ewm(com=2).mean()
        f['price_kdj_k'] = k
        f['price_kdj_d'] = d
        f['price_kdj_j'] = 3 * k - 2 * d
        f['price_kdj_kd_dist'] = k - d

        # ---- 8. 布林带 ----
        for p in bb_periods:
            ma = pd.Series(close).rolling(p).mean()
            std = pd.Series(close).rolling(p).std()
            upper = ma + 2 * std
            lower = ma - 2 * std
            f[f'price_bb{p}_width'] = (upper - lower) / (ma + 1e-10)
            f[f'price_bb{p}_pos'] = (close - lower) / (upper - lower + 1e-10)

        # ---- 9. ATR ----
        tr = pd.concat([
            pd.Series(high) - pd.Series(low),
            (pd.Series(high) - pd.Series(close).shift(1)).abs(),
            (pd.Series(close).shift(1) - pd.Series(low)).abs()
        ], axis=1).max(axis=1)
        for p in atr_periods:
            f[f'price_atr_{p}'] = tr.rolling(p).mean()
        f['price_atr_ratio'] = f['price_atr_14'] / pd.Series(close)

        # ---- 10. ADX (趋势强度) ----
        plus_dm = pd.Series(high).diff()
        minus_dm = -pd.Series(low).diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        atr14 = f['price_atr_14']
        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr14 + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr14 + 1e-10))
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        f['price_adx'] = dx.rolling(14).mean()
        f['price_adx_trend'] = np.where(plus_di > minus_di, f['price_adx'], -f['price_adx'])

        # ---- 11. CCI ----
        tp = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
        f['price_cci'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-10)

        # ---- 12. 价格通道 ----
        for p in [10, 20, 60, 120]:
            high_roll = pd.Series(high).rolling(p).max()
            low_roll = pd.Series(low).rolling(p).min()
            f[f'price_pos_{p}'] = (close - low_roll) / (high_roll - low_roll + 1e-10)
            f[f'price_high_dist_{p}'] = (close - high_roll) / (high_roll + 1e-10)

        # ---- 13. 突破信号 ----
        for p in [20, 60]:
            high_roll = pd.Series(high).rolling(p).max()
            f[f'price_breakout_{p}'] = (close > high_roll.shift(1)).astype(int)

        # ---- 14. 收益偏度/峰度 (rolling) ----
        for p in [20, 60]:
            r = returns.rolling(p)
            f[f'price_skew_{p}'] = r.skew()
            f[f'price_kurt_{p}'] = r.kurt()

        return f


# ============ 成交量特征 ============
class VolumeFeatures:
    """成交量特征: 量比、OBV、量趋势、换手率"""

    @staticmethod
    def calculate(df: pd.DataFrame,
                  vol_periods: List[int] = None) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)

        close = df['close'].values.astype(float)
        volume = df['volume'].values.astype(float)

        vol_periods = vol_periods or DEFAULT_VOL_PERIODS
        vol = pd.Series(volume)

        # ---- 1. 量比 ----
        for p in vol_periods:
            ma = vol.rolling(p).mean()
            f[f'vol_ratio_{p}'] = vol / (ma + 1e-10)
            f[f'vol_ma_{p}'] = ma

        # ---- 2. 量变化率 ----
        f['vol_chg'] = vol.pct_change()
        f['vol_chg_5'] = vol.pct_change(5)
        f['vol_chg_20'] = vol.pct_change(20)

        # ---- 3. 量趋势 ----
        f['vol_trend_5_20'] = vol.rolling(5).mean() / (vol.rolling(20).mean() + 1e-10)
        f['vol_trend_5_60'] = vol.rolling(5).mean() / (vol.rolling(60).mean() + 1e-10)

        # ---- 4. 量波动 ----
        for p in [10, 20, 60]:
            f[f'vol_std_{p}'] = vol.rolling(p).std() / (vol.rolling(p).mean() + 1e-10)

        # ---- 5. OBV ----
        obv = np.zeros(len(close))
        obv[0] = volume[0]
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        f['vol_obv'] = obv
        f['vol_obv_chg'] = pd.Series(obv).pct_change(10)
        f['vol_obv_ma'] = pd.Series(obv).rolling(10).mean()

        # ---- 6. 量价关系 ----
        price_up = (pd.Series(close).diff() > 0).astype(int)
        vol_up = (vol.diff() > 0).astype(int)
        f['vol_price_div'] = (price_up != vol_up).rolling(5).mean()  # 背离
        f['vol_price_conf'] = (price_up == vol_up).rolling(5).mean()  # 确认

        # ---- 7. 换手率特征 (如果有) ----
        if 'turnover' in df.columns:
            to = df['turnover'].values.astype(float)
            for p in [5, 10, 20]:
                f[f'vol_turnover_ma{p}'] = pd.Series(to).rolling(p).mean()
            f['vol_turnover_chg'] = pd.Series(to).pct_change(5)

        return f


# ============ K线形态特征 ============
class PatternFeatures:
    """K线形态特征: 影线、实体、跳空、内包"""

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)

        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        open_price = df['open'].values.astype(float)

        # ---- 1. 影线/实体比率 ----
        body = np.abs(close - open_price)
        total_range = high - low + 1e-10
        f['pat_body_ratio'] = body / total_range
        f['pat_upper_shadow'] = (high - np.maximum(close, open_price)) / total_range
        f['pat_lower_shadow'] = (np.minimum(close, open_price) - low) / total_range

        # ---- 2. 跳空 ----
        f['pat_gap'] = (open_price - pd.Series(close).shift(1)) / (pd.Series(close).shift(1) + 1e-10)
        f['pat_gap_up'] = (f['pat_gap'] > 0.01).astype(int)
        f['pat_gap_down'] = (f['pat_gap'] < -0.01).astype(int)

        # ---- 3. 内包/外包 ----
        prev_high = pd.Series(high).shift(1)
        prev_low = pd.Series(low).shift(1)
        f['pat_inside'] = ((high <= prev_high) & (low >= prev_low)).astype(int)
        f['pat_outside'] = ((high > prev_high) & (low < prev_low)).astype(int)

        # ---- 4. 十字星 ----
        f['pat_doji'] = (body / total_range < 0.001).astype(int)

        # ---- 5. 连续涨跌 ----
        for p in [3, 5, 10]:
            price_up = (pd.Series(close).diff() > 0).astype(int)
            f[f'pat_up_streak_{p}'] = price_up.rolling(p).sum()
            f[f'pat_down_streak_{p}'] = (pd.Series(close).diff() < 0).astype(int).rolling(p).sum()

        # ---- 6. 振幅 ----
        f['pat_amplitude'] = (high - low) / (pd.Series(close).shift(1) + 1e-10)
        f['pat_amplitude_ma5'] = f['pat_amplitude'].rolling(5).mean()

        return f


# ============ 动量/加速度特征 ============
class MomentumFeatures:
    """动量特征: 二阶变化、加速度、衰减"""

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        close = df['close'].values.astype(float)

        # ---- 1. 动量 ----
        for p in [3, 5, 10, 20, 60]:
            f[f'mom_momentum_{p}'] = close - pd.Series(close).shift(p)

        # ---- 2. 动量加速度 (二阶) ----
        for p in [3, 5, 10, 20]:
            ret = pd.Series(close).pct_change(p)
            f[f'mom_accel_{p}'] = ret.diff(3)

        # ---- 3. 动量衰减 ----
        mom_short = pd.Series(close).pct_change(5)
        mom_long = pd.Series(close).pct_change(20)
        f['mom_decay_5_20'] = mom_short - mom_long
        mom_short2 = pd.Series(close).pct_change(10)
        mom_long2 = pd.Series(close).pct_change(60)
        f['mom_decay_10_60'] = mom_short2 - mom_long2

        # ---- 4. 波动率聚集 ----
        returns = pd.Series(close).pct_change()
        vol5 = returns.rolling(5).std()
        vol20 = returns.rolling(20).std()
        vol60 = returns.rolling(60).std()
        f['mom_vol_ratio_5_20'] = vol5 / (vol20 + 1e-10)
        f['mom_vol_ratio_20_60'] = vol20 / (vol60 + 1e-10)

        # ---- 5. 波动率均值回归 ----
        vol_ma = vol20.rolling(60).mean()
        f['mom_vol_vs_mean'] = vol20 / (vol_ma + 1e-10)

        # ---- 6. Hurst 指数 (趋势 vs 均值回归) ----
        # 简化版: 用 log(RS_20/RS_5) / log(20/5) 近似
        for p in [20, 60]:
            r = returns.rolling(p)
            rs = r.max() - r.min()
            f[f'mom_range_{p}'] = rs / (r.std() * np.sqrt(p) + 1e-10)

        return f


# ============ 截面排名特征 ============
class CrossSectionFeatures:
    """截面排名特征: 在给定日期内，对所有股票的特征做排名

    使用方法:
      1. 先对所有股票独立计算 Price/Volume/Pattern/Momentum 特征
      2. 调用 CrossSectionFeatures.calculate(all_features_dict, all_dates)
         all_features_dict: {symbol: DataFrame(index=date, columns=features)}
      3. 返回同样结构的截面排名特征
    """

    # 哪些特征做截面排名 (选择有比较意义的)
    RANK_TARGETS = [
        'price_ret_5', 'price_ret_20', 'price_ret_60',
        'price_vol_20', 'price_vol_60',
        'price_ma5_ratio', 'price_ma20_ratio', 'price_ma60_ratio',
        'price_rsi_14', 'price_rsi_50',
        'price_atr_ratio', 'price_bb20_pos',
        'vol_ratio_5', 'vol_ratio_20',
        'mom_momentum_20', 'mom_decay_5_20',
    ]

    @staticmethod
    def calculate(all_features: Dict[str, pd.DataFrame],
                  all_dates: List) -> Dict[str, pd.DataFrame]:
        """计算截面排名特征

        Args:
            all_features: {symbol: DataFrame(index=date, columns=features)}
            all_dates: 所有日期列表

        Returns:
            {symbol: DataFrame(截面排名特征)}
        """
        # 初始化结果
        result = {sym: pd.DataFrame(index=feats.index) for sym, feats in all_features.items()}

        # 确保 all_dates 是 sorted unique
        all_dates = sorted(set(all_dates))

        for date in all_dates:
            # 收集该日期所有股票的可用特征
            date_data = {}
            for sym, feats in all_features.items():
                if date in feats.index:
                    row = feats.loc[date]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]  # 取第一个匹配
                    if not bool(row.isna().all()):
                        date_data[sym] = row

            if len(date_data) < 10:  # 至少需要10只股票才有意义
                continue

            # 构建该日期的特征矩阵
            symbols = list(date_data.keys())
            for target in CrossSectionFeatures.RANK_TARGETS:
                values = []
                for sym in symbols:
                    val = date_data[sym].get(target, np.nan)
                    values.append(val)

                values = np.array(values, dtype=float)
                valid = ~np.isnan(values)

                if valid.sum() < 5:
                    continue

                # 排名 (0~1, 值越大排名越高)
                ranks = np.full(len(values), np.nan)
                ranks[valid] = pd.Series(values[valid]).rank(pct=True).values

                # 写入结果
                for i, sym in enumerate(symbols):
                    col_name = f'cs_rank_{target}'
                    if col_name not in result[sym].columns:
                        result[sym][col_name] = np.nan
                    result[sym].loc[date, col_name] = ranks[i]

        return result


# ============ 特征交互 ============
class InteractionFeatures:
    """特征交互: 量价共振、波动率×成交量等"""

    @staticmethod
    def calculate(features: pd.DataFrame) -> pd.DataFrame:
        """基于已有特征计算交互特征

        Args:
            features: 包含 price/vol/pat/mom 原始特征的 DataFrame
        """
        f = pd.DataFrame(index=features.index)

        # 可用特征映射
        cols = set(features.columns)

        # ---- 量价共振 ----
        for ret_p, vol_p in [(5, 5), (20, 20), (60, 20)]:
            ret_col = f'price_ret_{ret_p}'
            vol_col = f'vol_ratio_{vol_p}'
            if ret_col in cols and vol_col in cols:
                f[f'interact_ret{ret_p}_vol{vol_p}'] = features[ret_col] * features[vol_col]

        # ---- 波动率×成交量 ----
        if 'price_vol_20' in cols and 'vol_ratio_20' in cols:
            f['interact_vol20_volratio20'] = features['price_vol_20'] * features['vol_ratio_20']

        # ---- RSI×动量 ----
        if 'price_rsi_14' in cols and 'price_ret_5' in cols:
            f['interact_rsi14_ret5'] = features['price_rsi_14'] * features['price_ret_5']

        # ---- 均线突破×放量 ----
        if 'price_ma5_ratio' in cols and 'vol_ratio_5' in cols:
            f['interact_ma5_vol5'] = features['price_ma5_ratio'] * features['vol_ratio_5']

        if 'price_ma20_ratio' in cols and 'vol_ratio_20' in cols:
            f['interact_ma20_vol20'] = features['price_ma20_ratio'] * features['vol_ratio_20']

        # ---- 波动率×收益 ----
        if 'price_vol_20' in cols and 'price_ret_20' in cols:
            f['interact_vol20_ret20'] = features['price_vol_20'] * features['price_ret_20']

        # ---- 趋势×成交量 ----
        if 'price_adx' in cols and 'vol_ratio_20' in cols:
            f['interact_adx_vol20'] = features['price_adx'] * features['vol_ratio_20']

        # ---- 布林位置×成交量 ----
        if 'price_bb20_pos' in cols and 'vol_ratio_5' in cols:
            f['interact_bb20_vol5'] = features['price_bb20_pos'] * features['vol_ratio_5']

        # ---- 动量衰减×成交量 ----
        if 'mom_decay_5_20' in cols and 'vol_ratio_20' in cols:
            f['interact_decay_vol'] = features['mom_decay_5_20'] * features['vol_ratio_20']

        # ---- 跳空×成交量 ----
        if 'pat_gap' in cols and 'vol_ratio_5' in cols:
            f['interact_gap_vol5'] = features['pat_gap'] * features['vol_ratio_5']

        return f


# ============ 情绪特征 ============
class SentimentFeatures:
    """情绪特征: 涨跌停、融资融券、龙虎榜、量比、异常收益"""

    @staticmethod
    def calculate(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)

        if 'date' not in df.columns or symbol is None:
            for col in ['sent_limit_up', 'sent_limit_down', 'sent_consecutive_limit',
                        'sent_vol_ratio', 'sent_abnormal_ret', 'sent_margin_chg',
                        'sent_short_balance', 'sent_lhb_ret_5d', 'sent_lhb_flag',
                        'sent_limit_any', 'sent_vol_ratio_ma5', 'sent_margin_cum3']:
                f[col] = 0
            return f

        df_dates = pd.to_datetime(df['date'])
        trade_dates_ymd = df_dates.dt.strftime('%Y-%m-%d')

        try:
            conn = sqlite3.connect(DB_PATH)
            min_d, max_d = trade_dates_ymd.min(), trade_dates_ymd.max()
            sent = pd.read_sql(
                "SELECT symbol, trade_date, is_limit_up, is_limit_down, "
                "consecutive_limit_up, vol_ratio_20, abnormal_ret, "
                "margin_balance_chg, short_balance, lhb_ret_5d, lhb_flag "
                "FROM sentiment_daily "
                f"WHERE symbol='{symbol}' AND trade_date >= '{min_d}' "
                f"AND trade_date <= '{max_d}' ORDER BY trade_date", conn)
            conn.close()

            if len(sent) > 0:
                sent_map = sent.set_index('trade_date')

                f['sent_limit_up'] = trade_dates_ymd.map(
                    lambda d: float(sent_map.loc[d, 'is_limit_up']) if d in sent_map.index else 0).fillna(0)
                f['sent_limit_down'] = trade_dates_ymd.map(
                    lambda d: float(sent_map.loc[d, 'is_limit_down']) if d in sent_map.index else 0).fillna(0)
                f['sent_consecutive_limit'] = trade_dates_ymd.map(
                    lambda d: float(sent_map.loc[d, 'consecutive_limit_up']) if d in sent_map.index else 0).fillna(0)

                f['sent_vol_ratio'] = trade_dates_ymd.map(
                    lambda d: float(sent_map.loc[d, 'vol_ratio_20']) if d in sent_map.index else 1).fillna(1)
                f['sent_vol_ratio'] = f['sent_vol_ratio'].clip(0, 50)

                f['sent_abnormal_ret'] = trade_dates_ymd.map(
                    lambda d: float(sent_map.loc[d, 'abnormal_ret']) if d in sent_map.index else 0).fillna(0)

                f['sent_margin_chg'] = trade_dates_ymd.map(
                    lambda d: float(sent_map.loc[d, 'margin_balance_chg']) if d in sent_map.index else 0).fillna(0)
                f['sent_short_balance'] = trade_dates_ymd.map(
                    lambda d: float(sent_map.loc[d, 'short_balance']) if d in sent_map.index else 0).fillna(0)

                f['sent_lhb_ret_5d'] = trade_dates_ymd.map(
                    lambda d: float(sent_map.loc[d, 'lhb_ret_5d']) if d in sent_map.index else 0).fillna(0)
                f['sent_lhb_flag'] = trade_dates_ymd.map(
                    lambda d: float(sent_map.loc[d, 'lhb_flag']) if d in sent_map.index else 0).fillna(0)

                f['sent_limit_any'] = ((f['sent_limit_up'] > 0) | (f['sent_limit_down'] > 0)).astype(int)
                f['sent_vol_ratio_ma5'] = f['sent_vol_ratio'].rolling(5, min_periods=1).mean()
                f['sent_margin_cum3'] = f['sent_margin_chg'].rolling(3, min_periods=1).sum()
            else:
                for col in ['sent_limit_up', 'sent_limit_down', 'sent_consecutive_limit',
                            'sent_vol_ratio', 'sent_abnormal_ret', 'sent_margin_chg',
                            'sent_short_balance', 'sent_lhb_ret_5d', 'sent_lhb_flag',
                            'sent_limit_any', 'sent_vol_ratio_ma5', 'sent_margin_cum3']:
                    f[col] = 0
        except Exception:
            for col in ['sent_limit_up', 'sent_limit_down', 'sent_consecutive_limit',
                        'sent_vol_ratio', 'sent_abnormal_ret', 'sent_margin_chg',
                        'sent_short_balance', 'sent_lhb_ret_5d', 'sent_lhb_flag',
                        'sent_limit_any', 'sent_vol_ratio_ma5', 'sent_margin_cum3']:
                f[col] = 0

        return f


# ============ 市场特征 ============
class MarketFeatures:
    """市场特征: 北向资金、大盘、板块"""

    @staticmethod
    def calculate(df: pd.DataFrame, symbol: str = None,
                  north_shift_days: int = 0) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)

        if 'date' not in df.columns:
            return f

        df_dates = pd.to_datetime(df['date'])
        trade_dates_ymd = df_dates.dt.strftime('%Y-%m-%d')
        trade_dates_raw8 = df_dates.dt.strftime('%Y%m%d')

        # ---- 大盘数据 (沪深300) ----
        market_pct = None
        try:
            conn = sqlite3.connect(DB_PATH)
            min_d, max_d = trade_dates_raw8.min(), trade_dates_raw8.max()
            hs300 = pd.read_sql(
                f"SELECT trade_date, pct_chg, avg_pct_chg, volume, up_count "
                f"FROM hs300_daily "
                f"WHERE trade_date >= '{min_d}' AND trade_date <= '{max_d}'", conn)
            conn.close()
            if len(hs300) > 0:
                pct_col = 'pct_chg' if 'pct_chg' in hs300.columns else 'avg_pct_chg'
                mkt_map = dict(zip(hs300['trade_date'], hs300[pct_col]))
                market_pct = trade_dates_raw8.map(mkt_map).fillna(0) / 100

                if 'volume' in hs300.columns:
                    vol_map = dict(zip(hs300['trade_date'], hs300['volume'].astype(float)))
                    market_volume = pd.Series(trade_dates_raw8.map(vol_map).fillna(0).values, index=f.index)
                    vol_ma20 = market_volume.rolling(20, min_periods=1).mean()
                    f['mkt_hs300_volume_chg'] = (market_volume - vol_ma20) / (vol_ma20 + 1e-10)

                if 'up_count' in hs300.columns:
                    up_map = dict(zip(hs300['trade_date'], hs300['up_count'].astype(float)))
                    up_series = pd.Series(trade_dates_raw8.map(up_map).fillna(0).values, index=f.index)
                    f['mkt_breadth'] = up_series / 300
                    f['mkt_breadth_ma5'] = f['mkt_breadth'].rolling(5, min_periods=1).mean()
        except Exception:
            pass

        if market_pct is not None:
            mkt_vol = pd.Series(market_pct).rolling(20, min_periods=1).std()
            f['mkt_hs300_volatility'] = mkt_vol.values

        # ---- 北向资金 (可选滞后) ----
        north_flow_abs = None
        north_sh_net = None
        north_sz_net = None
        north_buy_ratio = None
        try:
            conn = sqlite3.connect(DB_PATH)
            min_ymd, max_ymd = trade_dates_ymd.min(), trade_dates_ymd.max()
            north_df = pd.read_sql(
                "SELECT trade_date, total_net, north_net, sz_net, total_buy, total_sell "
                "FROM north_flow "
                "WHERE total_net IS NOT NULL AND total_net != 0 "
                f"AND trade_date >= '{min_ymd}' AND trade_date <= '{max_ymd}'", conn)
            conn.close()

            if len(north_df) > 0:
                if north_shift_days > 0:
                    from datetime import timedelta
                    north_df['trade_date'] = (pd.to_datetime(north_df['trade_date'])
                                              + timedelta(days=north_shift_days)).dt.strftime('%Y-%m-%d')

                north_df['total_net_billion'] = north_df['total_net'] / 10000
                north_df = north_df[north_df['total_net_billion'].abs() < 500]
                north_map = dict(zip(north_df['trade_date'], north_df['total_net_billion']))
                north_mapped = trade_dates_ymd.map(north_map)
                if north_mapped.notna().sum() / len(north_mapped) >= 0.5:
                    north_flow_abs = north_mapped.fillna(0)

                sh_map = dict(zip(north_df['trade_date'], north_df['north_net'] / 10000))
                sh_mapped = trade_dates_ymd.map(sh_map).fillna(0)
                if sh_mapped.notna().sum() / len(sh_mapped) >= 0.5:
                    north_sh_net = sh_mapped

                sz_map = dict(zip(north_df['trade_date'], north_df['sz_net'] / 10000))
                sz_mapped = trade_dates_ymd.map(sz_map).fillna(0)
                if sz_mapped.notna().sum() / len(sz_mapped) >= 0.5:
                    north_sz_net = sz_mapped

                north_df['buy_sell_ratio'] = (
                    north_df['total_buy'].astype(float) /
                    (north_df['total_sell'].astype(float) + 1)
                ).clip(0, 5)
                ratio_map = dict(zip(north_df['trade_date'], north_df['buy_sell_ratio']))
                ratio_mapped = trade_dates_ymd.map(ratio_map).fillna(1)
                if ratio_mapped.notna().sum() / len(ratio_mapped) >= 0.5:
                    north_buy_ratio = ratio_mapped
        except Exception:
            pass

        if north_flow_abs is not None:
            north_ma = north_flow_abs.rolling(10, min_periods=1).mean()
            f['mkt_north_surprise'] = (north_flow_abs - north_ma) / (north_ma.abs() + 1e-6)
            f['mkt_north_cum5'] = north_flow_abs.rolling(5, min_periods=1).sum()
            f['mkt_north_cum10'] = north_flow_abs.rolling(10, min_periods=1).sum()
            f['mkt_north_dir'] = (north_flow_abs > 0).astype(int)
            f['mkt_north_dir_streak'] = f['mkt_north_dir'].rolling(5, min_periods=1).sum()
        else:
            for col in ['mkt_north_surprise', 'mkt_north_cum5', 'mkt_north_cum10',
                        'mkt_north_dir', 'mkt_north_dir_streak']:
                f[col] = 0

        if north_sh_net is not None:
            f['mkt_north_sh_net'] = north_sh_net
            f['mkt_north_sh_ma5'] = north_sh_net.rolling(5, min_periods=1).mean()
        else:
            f['mkt_north_sh_net'] = 0
            f['mkt_north_sh_ma5'] = 0

        if north_sz_net is not None:
            f['mkt_north_sz_net'] = north_sz_net
            f['mkt_north_sz_ma5'] = north_sz_net.rolling(5, min_periods=1).mean()
        else:
            f['mkt_north_sz_net'] = 0
            f['mkt_north_sz_ma5'] = 0

        if north_buy_ratio is not None:
            f['mkt_north_buy_ratio'] = north_buy_ratio
            f['mkt_north_buy_ratio_ma5'] = north_buy_ratio.rolling(5, min_periods=1).mean()
        else:
            f['mkt_north_buy_ratio'] = 1
            f['mkt_north_buy_ratio_ma5'] = 1

        # ---- 个股 vs 大盘 ----
        if 'close' in df.columns and market_pct is not None:
            stock_pct = df['close'].pct_change()
            f['mkt_alpha'] = stock_pct - market_pct
            f['mkt_alpha_cum3'] = f['mkt_alpha'].rolling(3, min_periods=1).sum()
            f['mkt_alpha_cum5'] = f['mkt_alpha'].rolling(5, min_periods=1).sum()
            f['mkt_contra_up'] = ((stock_pct > 0) & (market_pct < 0)).astype(int)
            f['mkt_contra_down'] = ((stock_pct < 0) & (market_pct > 0)).astype(int)
            stock_vol = stock_pct.rolling(20, min_periods=1).std()
            market_vol = pd.Series(market_pct).rolling(20, min_periods=1).std()
            f['mkt_vol_ratio'] = stock_vol / (market_vol.values + 1e-10)
            if 'open' in df.columns:
                intraday = (df['close'] - df['open']) / (df['open'] + 1e-10)
                f['mkt_intraday_alpha'] = intraday - market_pct
        else:
            for col in ['mkt_alpha', 'mkt_alpha_cum3', 'mkt_alpha_cum5', 'mkt_contra_up',
                        'mkt_contra_down', 'mkt_vol_ratio', 'mkt_intraday_alpha']:
                f[col] = 0

        # ---- 板块 ----
        try:
            if symbol:
                conn = sqlite3.connect(DB_PATH)
                row = conn.execute("SELECT industry FROM stock_sector WHERE symbol=?", (symbol,)).fetchone()
                conn.close()
                industry = row[0] if row else '其他'
            else:
                industry = '其他'
        except Exception:
            industry = '其他'
        strong = any(kw in industry for kw in
                     ['电子', '计算机', '通信', '软件', '医药', '医疗', '电力设备',
                      '电气', '军工', '国防', '汽车', '新能源', '半导体', '芯片', '金融', '保险'])
        f['mkt_sector_strong'] = 1 if strong else 0

        return f


# ============ 特征流水线 ============
class MacroInteractionFeatures:
    """宏观×个股交互特征: 在不同宏观环境下个股特征的表现差异"""

    @staticmethod
    def calculate(features: pd.DataFrame) -> pd.DataFrame:
        """计算宏观环境下的个股特征交互"""
        f = pd.DataFrame(index=features.index)
        cols = set(features.columns)

        # 波动率环境 × 个股波动率
        if 'macro_hs300_vol_high' in cols and 'price_vol_20' in cols:
            f['mi_vol_regime_x_vol'] = features['macro_hs300_vol_high'] * features['price_vol_20']

        # 流动性环境 × 成交量
        if 'macro_liquidity_tight' in cols and 'vol_ratio_20' in cols:
            f['mi_liquidity_x_vol'] = features['macro_liquidity_tight'] * features['vol_ratio_20']

        # 汇率环境 × 北向资金
        if 'macro_cny_depreciate' in cols and 'mkt_north_cum5' in cols:
            f['mi_cny_x_north'] = features['macro_cny_depreciate'] * features['mkt_north_cum5']

        # 市场状态 × 个股动量
        if 'macro_market_regime' in cols and 'price_ret_20' in cols:
            f['mi_regime_x_ret'] = features['macro_market_regime'] * features['price_ret_20']

        # 市场波动率 × 个股beta
        if 'macro_hs300_vol_20' in cols and 'macro_stock_beta' in cols:
            f['mi_mktvol_x_beta'] = features['macro_hs300_vol_20'] * features['macro_stock_beta']

        # 利率变化 × 高beta股票
        if 'macro_cn_10y_chg' in cols and 'macro_stock_beta' in cols:
            f['mi_yield_x_beta'] = features['macro_cn_10y_chg'] * features['macro_stock_beta']

        # 市场广度 × 个股相对强弱
        if 'macro_breadth_strong' in cols and 'macro_stock_rs' in cols:
            f['mi_breadth_x_rs'] = features['macro_breadth_strong'] * features['macro_stock_rs']

        return f


class FeaturePipeline:
    """统一特征计算流水线"""

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or {}
        self.north_shift = self.cfg.get('north_shift_days', 0)
        self._macro_features = None
        self._lstm_embeddings = None

    def _get_macro(self):
        """延迟加载宏观特征类"""
        if self._macro_features is None:
            self._macro_features = _get_macro_features()
        return self._macro_features

    def _load_lstm(self):
        """延迟加载 LSTM embeddings"""
        if self._lstm_embeddings is not None:
            return self._lstm_embeddings
        import pickle
        emb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'data/lstm_embeddings.pkl')
        try:
            with open(emb_path, 'rb') as f:
                self._lstm_embeddings = pickle.load(f)
            print(f"   ✅ LSTM embeddings 已加载 ({len(self._lstm_embeddings)} 只股票)")
        except FileNotFoundError:
            self._lstm_embeddings = {}
        return self._lstm_embeddings

    def compute_stock(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """计算单只股票的所有特征 (含宏观 + LSTM)"""
        price = PriceFeatures.calculate(df)
        volume = VolumeFeatures.calculate(df)
        pattern = PatternFeatures.calculate(df)
        momentum = MomentumFeatures.calculate(df)
        market = MarketFeatures.calculate(df, symbol=symbol, north_shift_days=self.north_shift)
        sentiment = SentimentFeatures.calculate(df, symbol=symbol)

        # 宏观特征 (v8 新增)
        macro = self._get_macro().calculate(df, symbol)

        base = pd.concat([price, volume, pattern, momentum, market, sentiment, macro], axis=1)
        interact = InteractionFeatures.calculate(base)

        # 宏观交互特征 (v8 新增)
        macro_interact = MacroInteractionFeatures.calculate(base)

        # LSTM 时序特征 (v9 新增)
        lstm_feats = self._compute_lstm_features(df, symbol)

        return pd.concat([base, interact, macro_interact, lstm_feats], axis=1)

    def _compute_lstm_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """从预计算 embedding 中提取 LSTM 特征 (向量化)"""
        embeddings = self._load_lstm()
        if symbol not in embeddings:
            return pd.DataFrame(index=df.index)

        emb_dict = embeddings[symbol]  # {date_str: np.array(64,)}
        dates = df['date'].values
        dim = 64
        arr = np.zeros((len(dates), dim), dtype=np.float32)

        for i, d in enumerate(dates):
            d_str = str(pd.Timestamp(d))[:10]
            if d_str in emb_dict:
                arr[i] = emb_dict[d_str]

        return pd.DataFrame(arr, index=df.index, columns=[f'lstm_{j}' for j in range(dim)])

    def compute_cross_section(self, all_stock_features: Dict[str, pd.DataFrame],
                              all_dates: List) -> Dict[str, pd.DataFrame]:
        """计算截面排名特征 (需要所有股票的特征)"""
        return CrossSectionFeatures.calculate(all_stock_features, all_dates)

    def merge_sentiment(self, features: pd.DataFrame, df: pd.DataFrame,
                        symbol: str, sent_df: pd.DataFrame) -> pd.DataFrame:
        """合并情绪特征"""
        if len(sent_df) == 0:
            return features

        dates = df['date'].dt.strftime('%Y-%m-%d')
        sent = sent_df[sent_df['symbol'] == symbol].set_index('date')
        for col in sent.columns:
            if col not in ('symbol', 'date'):
                features[f'sent_{col}'] = dates.map(
                    lambda d: sent.loc[d, col] if d in sent.index else 0
                ).fillna(0).values
        return features

    def get_feature_names(self) -> List[str]:
        """返回特征名称列表 (用于对齐)"""
        return None  # 运行时动态获取


# ============ 兼容旧接口 ============
# 保留旧类名，避免破坏现有代码
EnhancedFeatureEngineer = PriceFeatures
AdvancedFeatureEngineer = MomentumFeatures
MarketFeatureEngineer = MarketFeatures

# 旧 API 兼容方法
def _patch_for_compat():
    """为旧类添加旧方法名别名和类属性"""
    PriceFeatures.calculate_features = staticmethod(PriceFeatures.calculate)
    PriceFeatures.FEATURE_NAMES = None
    MomentumFeatures.calculate_advanced_features = staticmethod(MomentumFeatures.calculate)
    MarketFeatures.calculate_market_features = staticmethod(MarketFeatures.calculate)
    MarketFeatures.MARKET_FEATURE_NAMES = None

_patch_for_compat()

# 旧常量 (保留兼容)
ZERO_IMP_FEATURES = []
TIME_FEATURES = ['morning_session', 'afternoon_session', 'is_month_end']

# 兼容旧的 compute_features (train.py 中使用)
def compute_features(df: pd.DataFrame, symbol: str, cfg: dict) -> pd.DataFrame:
    """兼容旧接口: 计算增强特征"""
    pipeline = FeaturePipeline(cfg)
    return pipeline.compute_stock(df, symbol)


if __name__ == '__main__':
    # 测试
    import yfinance as yf
    test_df = yf.download('AAPL', period='1y', auto_adjust=False)
    if isinstance(test_df.columns, pd.MultiIndex):
        test_df.columns = test_df.columns.droplevel(1)
    test_df.columns = [c.lower() for c in test_df.columns]
    test_df = test_df.reset_index()
    test_df.columns = [c.lower() for c in test_df.columns]

    pipeline = FeaturePipeline()
    features = pipeline.compute_stock(test_df, 'AAPL')
    print(f"特征数: {len(features.columns)}")
    print(f"特征名: {list(features.columns)[:20]}...")
    print(f"缺失率: {(features.isna().sum() / len(features) * 100).describe()}")