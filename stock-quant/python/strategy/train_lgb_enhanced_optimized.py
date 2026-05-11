"""
优化版特征计算 - 使用pd.concat避免DataFrame碎片化
"""
import pandas as pd
import numpy as np

class EnhancedFeatureEngineerOptimized:
    """优化版特征工程"""
    
    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """使用concat批量添加特征，避免碎片化"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        features_dict = {}
        
        # 收益率特征（批量计算）
        for period in [1, 5, 10, 20]:
            features_dict[f'return_{period}'] = pd.Series(close).pct_change(period)
            features_dict[f'log_return_{period}'] = np.log(close / np.roll(close, period))
        
        # 均线特征
        for period in [5, 10, 20, 60, 120]:
            ma = pd.Series(close).rolling(period).mean()
            features_dict[f'ma{period}'] = ma
            features_dict[f'ma{period}_ratio'] = close / ma
        
        # 技术指标
        features_dict['volatility_10'] = pd.Series(close).pct_change().rolling(10).std()
        features_dict['volatility_20'] = pd.Series(close).pct_change().rolling(20).std()
        
        # RSI
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = delta.where(delta < 0, 0).rolling(14).mean()
        features_dict['rsi_14'] = 100 - (100 / (1 + gain / loss))
        
        # MACD
        ema12 = pd.Series(close).ewm(span=12).mean()
        ema26 = pd.Series(close).ewm(span=26).mean()
        features_dict['macd'] = ema12 - ema26
        features_dict['macd_signal'] = features_dict['macd'].ewm(span=9).mean()
        
        # 布林带
        ma20 = pd.Series(close).rolling(20).mean()
        std20 = pd.Series(close).rolling(20).std()
        features_dict['boll_upper'] = ma20 + 2 * std20
        features_dict['boll_lower'] = ma20 - 2 * std20
        features_dict['boll_ratio'] = (close - ma20) / (2 * std20)
        
        # KDJ
        lowest = pd.Series(low).rolling(9).min()
        highest = pd.Series(high).rolling(9).max()
        rsv = (close - lowest) / (highest - lowest) * 100
        features_dict['k'] = rsv.ewm(com=2).mean()
        features_dict['d'] = features_dict['k'].ewm(com=2).mean()
        features_dict['j'] = 3 * features_dict['k'] - 2 * features_dict['d']
        
        # 成交量特征
        features_dict['volume_ratio'] = pd.Series(volume) / pd.Series(volume).rolling(20).mean()
        
        # 时间特征
        dates = pd.to_datetime(df['date'])
        features_dict['hour'] = dates.dt.hour
        features_dict['minute'] = dates.dt.minute
        features_dict['weekday'] = dates.dt.weekday
        
        # 使用concat一次性合并（避免碎片化）
        features = pd.concat(features_dict, axis=1)
        
        # 清理NaN
        features = features.fillna(0)
        
        return features
