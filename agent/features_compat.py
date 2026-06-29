#!/usr/bin/env python3
"""特征工程兼容层 — 从 train_lgb_enhanced.py 提取, 保证与 v8 模型特征一致"""
import numpy as np
import pandas as pd

class EnhancedFeatureEngineer:
    """增强版特征工程"""

    FEATURE_NAMES = None

    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算增强版特征（50+特征）
        """
        features = pd.DataFrame(index=df.index)

        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float)
        open_price = df['open'].values.astype(float)

        # ========================================
        # 1. 收益率特征 (10个)
        # ========================================
        for period in [1, 2, 3, 5, 10, 15, 20, 30, 40, 60]:
            features[f'return_{period}'] = pd.Series(close).pct_change(period)

        # ========================================
        # 2. 对数收益率 (5个)
        # ========================================
        for period in [1, 3, 5, 10, 20]:
            features[f'log_return_{period}'] = np.log(pd.Series(close) / pd.Series(close).shift(period))

        # ========================================
        # 3. 波动率特征 (8个)
        # ========================================
        returns = pd.Series(close).pct_change()
        for period in [5, 10, 20, 30, 40, 60, 80, 100]:
            features[f'volatility_{period}'] = returns.rolling(period).std()

        # Parkinson 波动率
        features['parkinson_vol'] = np.sqrt(
            (np.log(pd.Series(high) / pd.Series(low)) ** 2).rolling(20).mean() / (4 * np.log(2))
        )

        # ========================================
        # 4. 均线系统 (16个)
        # ========================================
        for period in [5, 10, 20, 30, 60, 80, 100, 120]:
            ma = pd.Series(close).rolling(period).mean()
            features[f'ma{period}_ratio'] = close / ma - 1
            features[f'price_above_ma{period}'] = (close > ma).astype(int)

        # 均线交叉
        for fast, slow in [(5, 10), (10, 20), (20, 60), (60, 120)]:
            ma_fast = pd.Series(close).rolling(fast).mean()
            ma_slow = pd.Series(close).rolling(slow).mean()
            features[f'ma{fast}_ma{slow}'] = ma_fast / ma_slow - 1
            features[f'ma{fast}_cross_ma{slow}'] = ((ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))).astype(int)

        # ========================================
        # 5. RSI 系列 (4个)
        # ========================================
        for period in [6, 14, 24, 50]:
            delta = pd.Series(close).diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # RSI 背离
        rsi_14 = features['rsi_14']
        price_change = pd.Series(close, index=df.index).diff(20)
        rsi_change = rsi_14.diff(20)
        features['rsi_divergence'] = np.where(
            (price_change.values < 0) & (rsi_change.values > 0), 1,
            np.where((price_change.values > 0) & (rsi_change.values < 0), -1, 0)
        )

        # ========================================
        # 6. MACD (4个)
        # ========================================
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal

        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = histogram
        features['macd_hist_slope'] = histogram.diff()

        # MACD 交叉信号
        features['macd_cross'] = ((macd > signal) & (macd.shift(1) <= signal.shift(1))).astype(int)

        # ========================================
        # 7. KDJ (5个)
        # ========================================
        low_min = pd.Series(low).rolling(9).min()
        high_max = pd.Series(high).rolling(9).max()
        rsv = (close - low_min) / (high_max - low_min + 1e-10) * 100

        features['kdj_k'] = rsv.ewm(com=2).mean()
        features['kdj_d'] = features['kdj_k'].ewm(com=2).mean()
        features['kdj_j'] = 3 * features['kdj_k'] - 2 * features['kdj_d']
        features['kdj_cross'] = features['kdj_k'] - features['kdj_d']
        features['kdj_cross_signal'] = ((features['kdj_k'] > features['kdj_d']) &
                                        (features['kdj_k'].shift(1) <= features['kdj_d'].shift(1))).astype(int)

        # ========================================
        # 8. 布林带 (5个)
        # ========================================
        for period in [20, 30]:
            ma = pd.Series(close).rolling(period).mean()
            std = pd.Series(close).rolling(period).std()
            upper = ma + 2 * std
            lower = ma - 2 * std

            features[f'bb_upper_{period}'] = (upper - close) / close
            features[f'bb_lower_{period}'] = (close - lower) / close
            features[f'bb_width_{period}'] = (upper - lower) / ma
            features[f'bb_position_{period}'] = (close - lower) / (upper - lower + 1e-10)

        # ========================================
        # 9. ATR (3个)
        # ========================================
        tr = pd.concat([
            pd.Series(high) - pd.Series(low),
            pd.Series(high) - pd.Series(close).shift(1),
            pd.Series(close).shift(1) - pd.Series(low)
        ], axis=1).max(axis=1)

        for period in [10, 14, 20]:
            features[f'atr_{period}'] = tr.rolling(period).mean()

        features['atr_ratio'] = features['atr_14'] / pd.Series(close)

        # ========================================
        # 10. 成交量特征 (10个)
        # ========================================
        vol = pd.Series(volume)

        for period in [5, 10, 20, 30, 60]:
            features[f'volume_ma{period}'] = vol.rolling(period).mean()
            features[f'volume_ratio_{period}'] = vol / (features[f'volume_ma{period}'] + 1e-10)

        # 成交量变化率
        features['volume_change'] = vol.pct_change()
        features['volume_acceleration'] = features['volume_change'].diff()

        # OBV
        obv = np.zeros(len(close))
        obv[0] = volume[0]
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]

        features['obv_ma10'] = pd.Series(obv).rolling(10).mean()
        features['obv_ma30'] = pd.Series(obv).rolling(30).mean()
        features['obv_trend'] = pd.Series(obv).diff(10)

        # ========================================
        # 11. 价格形态 (8个)
        # ========================================
        # 影线
        features['upper_shadow'] = (high - np.maximum(open_price, close)) / (close + 1e-10)
        features['lower_shadow'] = (np.minimum(open_price, close) - low) / (close + 1e-10)
        features['body_size'] = np.abs(close - open_price) / (close + 1e-10)

        # 跳空
        features['gap'] = (open_price - pd.Series(close).shift(1)) / (pd.Series(close).shift(1) + 1e-10)

        # 价格位置
        for period in [10, 20, 60]:
            high_roll = pd.Series(high).rolling(period).max()
            low_roll = pd.Series(low).rolling(period).min()
            features[f'price_position_{period}'] = (close - low_roll) / (high_roll - low_roll + 1e-10)
            features[f'high_{period}_ratio'] = (close - high_roll) / (high_roll + 1e-10)

        # ========================================
        # 12. 动量指标 (4个)
        # ========================================
        # 动量
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = close - pd.Series(close).shift(period)

        # CCI
        tp = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
        features['cci'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())

        # ========================================
        # 13. 交易时段特征 (3个，去掉伪规律日历特征)
        # ========================================
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
            # 只保留交易时段（早盘/尾盘行为确实不同）和月末效应
            features['morning_session'] = ((dates.dt.hour >= 9) & (dates.dt.hour < 12)).astype(int)
            features['afternoon_session'] = ((dates.dt.hour >= 13) & (dates.dt.hour < 15)).astype(int)
            features['is_month_end'] = dates.dt.is_month_end.astype(int)
            # 去掉 hour/minute/day_of_week/day_of_month（v1时排Top6-8，是伪规律）

        # ========================================
        # 14. 趋势强度 (3个)
        # ========================================
        # ADX
        plus_dm = pd.Series(high).diff()
        minus_dm = pd.Series(low).diff() * -1
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr = features['atr_14']
        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr + 1e-10))
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        features['adx'] = dx.rolling(14).mean()

        # 趋势方向
        features['trend_direction'] = np.where(
            plus_di.values > minus_di.values, 1, np.where(plus_di.values < minus_di.values, -1, 0)
        )

        # 趋势强度
        features['trend_strength'] = features['adx'] * features['trend_direction']

        # 缓存特征名称
        if EnhancedFeatureEngineer.FEATURE_NAMES is None:
            EnhancedFeatureEngineer.FEATURE_NAMES = features.columns.tolist()

        return features


