#!/usr/bin/env python3
"""
LightGBM 股票走势预测器
预测 3 日收益率，作为交易策略的额外评分因子
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入基本面特征工程
from data.fundamental_feature import FundamentalFeatureEngineer


class EnhancedFeatureEngineer:
    """增强版特征工程（与 train_full_optimized.py 中的一致）"""

    @staticmethod
    def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """计算增强的技术指标"""
        features = pd.DataFrame(index=df.index)
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        open_price = df['open'].values

        # === 基础特征 ===
        features['return_1d'] = pd.Series(close).pct_change(1)
        features['return_3d'] = pd.Series(close).pct_change(3)
        features['return_5d'] = pd.Series(close).pct_change(5)
        features['return_10d'] = pd.Series(close).pct_change(10)

        features['volatility_5d'] = pd.Series(features['return_1d']).rolling(5).std()
        features['volatility_10d'] = pd.Series(features['return_1d']).rolling(10).std()

        features['price_ma5_ratio'] = close / pd.Series(close).rolling(5).mean() - 1
        features['price_ma10_ratio'] = close / pd.Series(close).rolling(10).mean() - 1
        features['price_ma20_ratio'] = close / pd.Series(close).rolling(20).mean() - 1
        features['ma5_ma10'] = pd.Series(close).rolling(5).mean() / pd.Series(close).rolling(10).mean() - 1
        features['ma10_ma20'] = pd.Series(close).rolling(10).mean() / pd.Series(close).rolling(20).mean() - 1
        features['ma20_ma60'] = pd.Series(close).rolling(20).mean() / pd.Series(close).rolling(60).mean() - 1

        # === KDJ ===
        low_min = pd.Series(low).rolling(9).min()
        high_max = pd.Series(high).rolling(9).max()
        rsv = (close - low_min) / (high_max - low_min + 1e-10) * 100
        features['kdj_k'] = rsv.ewm(com=2).mean()
        features['kdj_d'] = features['kdj_k'].ewm(com=2).mean()
        features['kdj_j'] = 3 * features['kdj_k'] - 2 * features['kdj_d']
        features['kdj_cross'] = features['kdj_k'] - features['kdj_d']

        # === DDI ===
        dmz = np.maximum(np.abs(high - pd.Series(low).shift(1)), np.abs(high - low))
        dmf = np.maximum(np.abs(pd.Series(low).shift(1) - low), np.abs(high - low))
        diz = dmz.rolling(14).mean()
        dif = dmf.rolling(14).mean()
        features['ddi'] = (diz - dif) / (diz + dif + 1e-10)

        # === CCI ===
        tp = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
        ma = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        features['cci'] = (tp - ma) / (0.015 * mad + 1e-10)

        # === MFI ===
        tp = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
        positive_flow = tp * pd.Series(volume) * (tp > tp.shift(1)).astype(int)
        negative_flow = tp * pd.Series(volume) * (tp < tp.shift(1)).astype(int)
        mfi_ratio = positive_flow.rolling(14).sum() / (negative_flow.rolling(14).sum() + 1e-10)
        features['mfi'] = 100 - (100 / (1 + mfi_ratio))
        features['money_flow'] = (pd.Series(close).diff() * pd.Series(volume) / 1e6).rolling(5).sum()

        # === ATR ===
        tr = pd.concat([
            pd.Series(high) - pd.Series(low),
            pd.Series(high) - pd.Series(close).shift(1),
            pd.Series(close).shift(1) - pd.Series(low)
        ], axis=1).max(axis=1)
        features['atr'] = tr.rolling(14).mean()
        features['atr_ratio'] = features['atr'] / pd.Series(close)

        # === 成交量 ===
        features['volume_ratio'] = pd.Series(volume).rolling(5).mean() / pd.Series(volume).rolling(20).mean()

        obv = np.zeros(len(close))
        obv[0] = volume[0]
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        features['obv_change_5d'] = pd.Series(obv).pct_change(5)

        # === K 线形态 ===
        features['upper_shadow'] = (high - np.maximum(open_price, close)) / (close + 1e-10)
        features['lower_shadow'] = (np.minimum(open_price, close) - low) / (close + 1e-10)
        features['body_size'] = np.abs(close - open_price) / (close + 1e-10)
        features['gap'] = (open_price - pd.Series(close).shift(1)) / pd.Series(close).shift(1)

        # === 布林带 ===
        ma20 = pd.Series(close).rolling(20).mean()
        std20 = pd.Series(close).rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        features['bb_position'] = (close - lower) / (upper - lower + 1e-10)
        features['bb_width'] = (upper - lower) / ma20

        # === RSI ===
        delta = pd.Series(close).diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features['rsi'] = 100 - (100 / (1 + rs))
        rs6 = (delta.where(delta > 0, 0)).rolling(6).mean() / ((-delta.where(delta < 0, 0)).rolling(6).mean() + 1e-10)
        features['rsi_6d'] = 100 - (100 / (1 + rs6))
        rs24 = (delta.where(delta > 0, 0)).rolling(24).mean() / ((-delta.where(delta < 0, 0)).rolling(24).mean() + 1e-10)
        features['rsi_24d'] = 100 - (100 / (1 + rs24))

        # === MACD ===
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = macd - signal
        features['macd_slope'] = macd.diff()

        # === 价格极值 ===
        features['highest_10d'] = (close - pd.Series(high).rolling(10).max()) / pd.Series(high).rolling(10).max()
        features['lowest_10d'] = (close - pd.Series(low).rolling(10).min()) / pd.Series(low).rolling(10).min()
        features['highest_20d'] = (close - pd.Series(high).rolling(20).max()) / pd.Series(high).rolling(20).max()
        features['lowest_20d'] = (close - pd.Series(low).rolling(20).min()) / pd.Series(low).rolling(20).min()

        # === SAR (简化) ===
        sar = np.zeros(len(close))
        ep = low[0]
        trend = 1
        af = 0.02
        for i in range(2, len(close)):
            if trend == 1:
                sar[i] = sar[i-1] + af * (ep - sar[i-1]) if i > 1 and sar[i-1] > 0 else low[0]
                if low[i] < sar[i]:
                    trend = -1
                    af = 0.02
            else:
                sar[i] = sar[i-1] + af * (ep - sar[i-1]) if i > 1 and sar[i-1] > 0 else high[0]
                if high[i] > sar[i]:
                    trend = 1
                    af = 0.02
            if trend == 1:
                ep = max(ep, high[i])
            else:
                ep = min(ep, low[i])
            af = min(0.2, af + 0.02)
        features['sar_position'] = (close - sar) / (close + 1e-10)

        return features

    @staticmethod
    def add_fundamental_features(features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """添加基本面特征"""
        try:
            fund_engineer = FundamentalFeatureEngineer()
            fundamental = fund_engineer.get_fundamental_data(symbol)
            if fundamental is not None and len(fundamental) > 0:
                fund_features = fund_engineer.calculate_features(fundamental)
                for key, value in fund_features.items():
                    if value is not None:
                        features[f'fund_{key}'] = value
        except Exception:
            pass
        return features


class FeatureEngineer:
    """特征工程类"""

    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算预测所需的特征

        特征类别:
        1. 技术指标：RSI, MACD, KDJ, 布林带等
        2. 价格动量：收益率，波动率
        3. 成交量特征：OBV, 量比
        4. 趋势特征：均线关系
        """
        features = pd.DataFrame(index=df.index)

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        # === 价格动量特征 ===
        # 日收益率
        features['return_1d'] = pd.Series(close).pct_change(1)
        features['return_3d'] = pd.Series(close).pct_change(3)
        features['return_5d'] = pd.Series(close).pct_change(5)
        features['return_10d'] = pd.Series(close).pct_change(10)

        # 波动率
        features['volatility_5d'] = pd.Series(features['return_1d']).rolling(5).std()
        features['volatility_10d'] = pd.Series(features['return_1d']).rolling(10).std()

        # 价格位置
        features['price_ma5_ratio'] = close / pd.Series(close).rolling(5).mean() - 1
        features['price_ma10_ratio'] = close / pd.Series(close).rolling(10).mean() - 1
        features['price_ma20_ratio'] = close / pd.Series(close).rolling(20).mean() - 1

        # === 技术指标 ===
        # RSI
        delta = pd.Series(close).diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = macd - signal

        # 布林带位置
        ma20 = pd.Series(close).rolling(20).mean()
        std20 = pd.Series(close).rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        features['bb_position'] = (close - lower) / (upper - lower + 1e-10)

        # === 成交量特征 ===
        # 量比
        features['volume_ratio'] = pd.Series(volume).rolling(5).mean() / \
                                   pd.Series(volume).rolling(20).mean()

        # OBV 变化率
        obv = np.zeros(len(close))
        obv[0] = volume[0]
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        features['obv_change_5d'] = pd.Series(obv).pct_change(5)

        # === 趋势特征 ===
        # 均线排列
        ma5 = pd.Series(close).rolling(5).mean()
        ma10 = pd.Series(close).rolling(10).mean()
        ma20 = pd.Series(close).rolling(20).mean()
        ma60 = pd.Series(close).rolling(60).mean()

        features['ma5_ma10'] = ma5 / ma10 - 1
        features['ma10_ma20'] = ma10 / ma20 - 1
        features['ma20_ma60'] = ma20 / ma60 - 1

        # 最高/最低位置
        features['highest_10d'] = (close - pd.Series(high).rolling(10).max()) / \
                                  pd.Series(high).rolling(10).max()
        features['lowest_10d'] = (close - pd.Series(low).rolling(10).min()) / \
                                 pd.Series(low).rolling(10).min()

        # 填充 NaN 值
        features = features.fillna(0)

        return features

    @staticmethod
    def add_fundamental_features(features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        添加基本面特征到技术面特征

        Args:
            features: 技术面特征 DataFrame
            symbol: 股票代码

        Returns:
            合并后的特征 DataFrame
        """
        fund_engineer = FundamentalFeatureEngineer()
        fundamental = fund_engineer.get_fundamental_data(symbol)
        fund_features = fund_engineer.calculate_features(fundamental)

        # 将基本面特征添加到每一行
        for key, value in fund_features.items():
            features[f'fund_{key}'] = value

        return features

    @staticmethod
    def calculate_target(df: pd.DataFrame, horizon: int = 3) -> np.ndarray:
        """
        计算预测目标：未来 N 日收益率

        Args:
            df: 数据 DataFrame
            horizon: 预测周期（默认 3 日）

        Returns:
            未来 N 日收益率数组
        """
        close = df['close'].values
        target = np.zeros(len(close))

        for i in range(len(close) - horizon):
            target[i] = (close[i + horizon] - close[i]) / close[i]

        # 最后几天的目标设为 0（实际不会用到）
        target[-horizon:] = 0

        return target


class LGBPredictor:
    """LightGBM 预测器 - 支持 30 分钟级别模型"""

    def __init__(self, model_dir: str = None):
        """
        初始化预测器

        Args:
            model_dir: 模型保存目录
        """
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), '../models/lgb_enhanced')

        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.model = None
        self.feature_names = None

        # 优先加载 30 分钟级别模型
        self._load_model()

    def _load_model(self):
        """加载模型"""
        # 优先加载 30 分钟级别模型
        model_30m_path = os.path.join(os.path.dirname(__file__), '../models/lgb_30m/model_30m.pkl')
        if os.path.exists(model_30m_path):
            with open(model_30m_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data.get('model')
                self.model_type = '30m'
                return

        # 次优：加载原有增强模型
        best_model_path = os.path.join(self.model_dir, 'zz500_full_optimized.pkl')
        if os.path.exists(best_model_path):
            with open(best_model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data.get('lgb_model')
                self.feature_names = model_data.get('feature_names', [])
                self.model_type = 'enhanced'
                return

        # 尝试加载通用模型
        generic_path = os.path.join(self.model_dir, 'generic_lgb.pkl')
        if os.path.exists(generic_path):
            with open(generic_path, 'rb') as f:
                self.model = pickle.load(f)
                self.model_type = 'generic'
                return

        self.model_type = None

    def _calculate_features_30m(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 30 分钟级别的特征"""
        features = pd.DataFrame(index=df.index)

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        open_price = df['open'].values

        # 价格动量
        features['return_1'] = pd.Series(close).pct_change(1)
        features['return_2'] = pd.Series(close).pct_change(2)
        features['return_3'] = pd.Series(close).pct_change(3)
        features['return_5'] = pd.Series(close).pct_change(5)
        features['return_10'] = pd.Series(close).pct_change(10)

        # 波动率
        features['volatility_5'] = pd.Series(features['return_1']).rolling(5).std()
        features['volatility_10'] = pd.Series(features['return_1']).rolling(10).std()
        features['volatility_20'] = pd.Series(features['return_1']).rolling(20).std()

        # 均线系统
        features['ma5_ratio'] = close / pd.Series(close).rolling(5).mean() - 1
        features['ma10_ratio'] = close / pd.Series(close).rolling(10).mean() - 1
        features['ma20_ratio'] = close / pd.Series(close).rolling(20).mean() - 1
        features['ma60_ratio'] = close / pd.Series(close).rolling(60).mean() - 1

        ma5 = pd.Series(close).rolling(5).mean()
        ma10 = pd.Series(close).rolling(10).mean()
        ma20 = pd.Series(close).rolling(20).mean()
        ma60 = pd.Series(close).rolling(60).mean()

        features['ma5_ma10'] = ma5 / ma10 - 1
        features['ma10_ma20'] = ma10 / ma20 - 1
        features['ma20_ma60'] = ma20 / ma60 - 1

        # RSI
        delta = pd.Series(close).diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        features['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        gain6 = (delta.where(delta > 0, 0)).rolling(6).mean()
        loss6 = (-delta.where(delta < 0, 0)).rolling(6).mean()
        features['rsi_6'] = 100 - (100 / (1 + gain6 / (loss6 + 1e-10)))

        # MACD
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = macd - signal
        features['macd_slope'] = macd.diff()

        # KDJ
        low_min = pd.Series(low).rolling(9).min()
        high_max = pd.Series(high).rolling(9).max()
        rsv = (close - low_min) / (high_max - low_min + 1e-10) * 100
        features['kdj_k'] = rsv.ewm(com=2).mean()
        features['kdj_d'] = features['kdj_k'].ewm(com=2).mean()
        features['kdj_j'] = 3 * features['kdj_k'] - 2 * features['kdj_d']
        features['kdj_cross'] = features['kdj_k'] - features['kdj_d']

        # 布林带
        ma20_bb = pd.Series(close).rolling(20).mean()
        std20 = pd.Series(close).rolling(20).std()
        upper = ma20_bb + 2 * std20
        lower = ma20_bb - 2 * std20
        features['bb_upper'] = (upper - close) / close
        features['bb_lower'] = (close - lower) / close
        features['bb_width'] = (upper - lower) / ma20_bb
        features['bb_position'] = (close - lower) / (upper - lower + 1e-10)

        # ATR
        tr = pd.concat([
            pd.Series(high) - pd.Series(low),
            pd.Series(high) - pd.Series(close).shift(1),
            pd.Series(close).shift(1) - pd.Series(low)
        ], axis=1).max(axis=1)
        features['atr'] = tr.rolling(14).mean()
        features['atr_ratio'] = features['atr'] / pd.Series(close)

        # 成交量
        features['volume_ma5'] = pd.Series(volume).rolling(5).mean()
        features['volume_ma10'] = pd.Series(volume).rolling(10).mean()
        features['volume_ratio'] = features['volume_ma5'] / (features['volume_ma10'] + 1e-10)

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
        features['obv_ma5'] = pd.Series(obv).rolling(5).mean()
        features['obv_ma10'] = pd.Series(obv).rolling(10).mean()
        features['obv_ratio'] = features['obv_ma5'] / (features['obv_ma10'] + 1e-10)

        # K 线形态
        features['upper_shadow'] = (high - np.maximum(open_price, close)) / (close + 1e-10)
        features['lower_shadow'] = (np.minimum(open_price, close) - low) / (close + 1e-10)
        features['body_size'] = np.abs(close - open_price) / (close + 1e-10)
        features['gap'] = (open_price - pd.Series(close).shift(1)) / (pd.Series(close).shift(1) + 1e-10)

        # 价格位置
        features['high_10'] = (close - pd.Series(high).rolling(10).max()) / (pd.Series(high).rolling(10).max() + 1e-10)
        features['low_10'] = (close - pd.Series(low).rolling(10).min()) / (pd.Series(low).rolling(10).min() + 1e-10)
        features['high_20'] = (close - pd.Series(high).rolling(20).max()) / (pd.Series(high).rolling(20).max() + 1e-10)
        features['low_20'] = (close - pd.Series(low).rolling(20).min()) / (pd.Series(low).rolling(20).min() + 1e-10)

        # 时间特征
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
            features['hour'] = dates.dt.hour
            features['minute'] = dates.dt.minute
            features['day_of_week'] = dates.dt.dayofweek
            features['is_open'] = ((dates.dt.hour == 9) | (dates.dt.hour == 13)).astype(int)
            features['is_close'] = ((dates.dt.hour == 11) | (dates.dt.hour == 14) | (dates.dt.hour == 15)).astype(int)

        # 填充 NaN
        features = features.fillna(0)

        return features

    def prepare_data(self, df: pd.DataFrame, symbol: str = 'default') -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练数据

        Args:
            df: 包含 OHLCV 的 DataFrame
            symbol: 股票代码（用于添加基本面特征）

        Returns:
            (特征矩阵，目标向量)
        """
        # 计算技术面特征
        features = FeatureEngineer.calculate_features(df)

        # 添加基本面特征
        features = FeatureEngineer.add_fundamental_features(features, symbol)

        # 计算目标
        target = FeatureEngineer.calculate_target(df, horizon=3)

        return features.values, target

    def train(self, df: pd.DataFrame, symbol: str = 'default') -> Dict:
        """
        训练模型

        Args:
            df: 训练数据
            symbol: 股票代码（用于保存模型）

        Returns:
            训练结果字典
        """
        print(f"开始训练 {symbol} 的 LightGBM 模型...")

        # 准备数据
        X, y = self.prepare_data(df)

        # 移除目标为 0 的样本（最后几天的无效数据）
        non_zero_mask = y != 0
        X = X[non_zero_mask]
        y = y[non_zero_mask]

        if len(X) < 50:
            print(f"  ⚠️ 数据量不足 ({len(X)} 条)，使用默认模型")
            return {'status': 'insufficient_data'}

        # 转换目标为分类问题（涨/跌/震荡）
        # 收益率 > 1%: 涨 (1)
        # 收益率 < -1%: 跌 (-1)
        # 其他：震荡 (0)
        y_class = np.zeros(len(y))
        y_class[y > 0.01] = 1
        y_class[y < -0.01] = -1

        # 检查是否有足够的类别
        unique_classes = np.unique(y_class)
        if len(unique_classes) < 2:
            print(f"  ⚠️ 数据类别单一 ({unique_classes})，使用通用模型")
            return {'status': 'insufficient_classes'}

        # 如果只有两个类别，使用二分类
        n_classes = len(unique_classes)
        if n_classes == 2:
            # 合并震荡到接近的类别
            y_class = np.where(y > 0, 1, -1)
            n_classes = 2

        # 划分训练集测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_class, test_size=0.2, random_state=42, shuffle=False
        )

        # 训练模型
        model = lgb.LGBMClassifier(
            objective='multiclass' if n_classes > 2 else 'binary',
            num_class=n_classes if n_classes > 2 else None,
            metric='multi_logloss' if n_classes > 2 else 'binary_logloss',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            n_estimators=100,
            max_depth=6,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1
        )

        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(10, verbose=False)]
            )

            # 评估
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print(f"  ✓ 训练完成，准确率：{accuracy:.2%}")
        except Exception as e:
            print(f"  ⚠️ 训练失败：{e}，使用通用模型")
            return {'status': 'training_failed', 'error': str(e)}

            # 保存模型
            model_path = os.path.join(self.model_dir, f"{symbol}_lgb.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            # 保存特征名称（包含基本面特征）
            self.feature_names = [
                # 技术面特征
                'return_1d', 'return_3d', 'return_5d', 'return_10d',
                'volatility_5d', 'volatility_10d',
                'price_ma5_ratio', 'price_ma10_ratio', 'price_ma20_ratio',
                'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_position',
                'volume_ratio', 'obv_change_5d',
                'ma5_ma10', 'ma10_ma20', 'ma20_ma60',
                'highest_10d', 'lowest_10d',
                # 基本面特征
                'fund_pe_ttm', 'fund_pb', 'fund_ps_ttm',
                'fund_roe', 'fund_roa', 'fund_gross_margin', 'fund_net_margin',
                'fund_revenue_growth', 'fund_profit_growth',
                'fund_debt_ratio', 'fund_current_ratio',
                'fund_pe_relative', 'fund_pb_relative', 'fund_roe_relative', 'fund_growth_relative',
                'fund_peg',
                'fund_value_score', 'fund_quality_score', 'fund_growth_score'
            ]

            feature_path = os.path.join(self.model_dir, f"{symbol}_features.pkl")
            with open(feature_path, 'wb') as f:
                pickle.dump(self.feature_names, f)

            return {
                'status': 'success',
                'accuracy': accuracy,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'n_classes': int(n_classes)
            }
        except Exception as e:
            print(f"  ⚠️ 训练异常：{e}，使用通用模型")
            return {'status': 'training_error', 'error': str(e)}

    def predict(self, df: pd.DataFrame, symbol: str = 'default') -> Optional[Dict]:
        """
        预测未来走势

        Args:
            df: 历史数据（30 分钟级别）
            symbol: 股票代码

        Returns:
            预测结果字典
        """
        # 加载模型（如果尚未加载）
        if self.model is None:
            self._load_model()

        if self.model is None:
            return None

        # 根据模型类型计算特征
        if self.model_type == '30m':
            # 使用 30 分钟级别特征
            features = self._calculate_features_30m(df)
        else:
            # 使用原有特征
            features = EnhancedFeatureEngineer.calculate_technical_features(df)
            features = FeatureEngineer.add_fundamental_features(features, symbol)

        # 填充 NaN 并获取最新特征
        features = features.fillna(0)

        # 获取特征名称列表（与训练时一致）
        if hasattr(self.model, 'feature_name_'):
            feature_names = self.model.feature_name_
        else:
            feature_names = features.columns.tolist()

        # 确保特征顺序与训练时一致
        latest_features = features[feature_names].iloc[-1:].values

        # 预测
        prediction = self.model.predict(latest_features)[0]
        proba = self.model.predict_proba(latest_features)[0]

        # 映射预测结果（0=下跌，1=上涨）
        pred_map = {0: '下跌', 1: '上涨'}
        pred_label = pred_map.get(int(prediction), '未知')

        # 概率字典
        proba_dict = {
            '下跌': float(proba[0]),
            '上涨': float(proba[1]) if len(proba) > 1 else 0.0,
            '震荡': 0.0
        }

        # 计算置信度
        confidence = max(proba)

        return {
            'prediction': int(prediction),
            'label': pred_label,
            'confidence': confidence,
            'probabilities': proba_dict
        }

    def get_signal_score(self, df: pd.DataFrame, symbol: str = 'default') -> float:
        """
        获取预测评分（用于融入交易策略）

        返回 -2 到 +2 的分数：
        - 强烈看涨：+2
        - 看涨：+1
        - 震荡：0
        - 看跌：-1
        - 强烈看跌：-2

        Args:
            df: 历史数据
            symbol: 股票代码

        Returns:
            评分 (-2 到 +2)
        """
        result = self.predict(df, symbol)

        if result is None:
            return 0  # 无预测结果时返回中性分

        prediction = result['prediction']
        confidence = result['confidence']
        probabilities = result.get('probabilities', {})

        # 获取上涨和下跌的概率
        up_prob = probabilities.get('上涨', 0.5)
        down_prob = probabilities.get('下跌', 0.5)

        # 二分类评分逻辑（0=下跌，1=上涨）
        if prediction == 1:  # 上涨
            if confidence > 0.65:
                return 2  # 强烈看涨
            elif confidence > 0.55:
                return 1.5
            else:
                return 1  # 看涨
        else:  # 下跌
            if confidence > 0.65:
                return -2  # 强烈看跌
            elif confidence > 0.55:
                return -1.5
            else:
                return -1  # 看跌

    def train_generic_model(self, df: pd.DataFrame) -> Dict:
        """
        训练通用模型（使用所有股票数据）

        Args:
            df: 合并的训练数据

        Returns:
            训练结果
        """
        print("训练通用 LightGBM 模型...")

        X, y = self.prepare_data(df)

        # 过滤无效数据（先过滤）
        non_zero_mask = y != 0
        X = X[non_zero_mask]
        y = y[non_zero_mask]

        if len(X) < 100:
            print(f"  ⚠️ 数据量不足 ({len(X)} 条)")
            return {'status': 'insufficient_data'}

        # 转换为目标分类（在过滤后）
        y_class = np.zeros(len(y))
        y_class[y > 0.02] = 1
        y_class[y < -0.02] = -1

        # 检查类别数量
        unique_classes = np.unique(y_class)
        n_classes = len(unique_classes)

        if n_classes < 2:
            print(f"  ⚠️ 数据类别单一 ({unique_classes})")
            return {'status': 'insufficient_classes'}

        # 如果只有两个类别，重新映射为 0 和 1
        if n_classes == 2:
            y_class = np.where(y_class == -1, 0, 1)  # -1->0, 1->1
            n_classes = 2

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_class, test_size=0.2, random_state=42
        )

        # 训练模型 - 根据类别数选择二分类或多分类
        model = lgb.LGBMClassifier(
            objective='multiclass' if n_classes > 2 else 'binary',
            num_class=n_classes if n_classes > 2 else None,
            metric='multi_logloss' if n_classes > 2 else 'binary_logloss',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            n_estimators=100,
            max_depth=6,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(10, verbose=False)]
        )

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"  ✓ 通用模型训练完成，准确率：{accuracy:.2%}")

        # 保存通用模型
        model_path = os.path.join(self.model_dir, 'generic_lgb.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        self.feature_names = [
            # 技术面特征
            'return_1d', 'return_3d', 'return_5d', 'return_10d',
            'volatility_5d', 'volatility_10d',
            'price_ma5_ratio', 'price_ma10_ratio', 'price_ma20_ratio',
            'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_position',
            'volume_ratio', 'obv_change_5d',
            'ma5_ma10', 'ma10_ma20', 'ma20_ma60',
            'highest_10d', 'lowest_10d',
            # 基本面特征
            'fund_pe_ttm', 'fund_pb', 'fund_ps_ttm',
            'fund_roe', 'fund_roa', 'fund_gross_margin', 'fund_net_margin',
            'fund_revenue_growth', 'fund_profit_growth',
            'fund_debt_ratio', 'fund_current_ratio',
            'fund_pe_relative', 'fund_pb_relative', 'fund_roe_relative', 'fund_growth_relative',
            'fund_peg',
            'fund_value_score', 'fund_quality_score', 'fund_growth_score'
        ]

        feature_path = os.path.join(self.model_dir, 'generic_features.pkl')
        with open(feature_path, 'wb') as f:
            pickle.dump(self.feature_names, f)

        return {
            'status': 'success',
            'accuracy': accuracy,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }


def train_all_models():
    """训练所有股票的模型"""
    from data.stock_data import StockDataHandler

    handler = StockDataHandler()
    predictor = LGBPredictor()

    # 训练数据
    watchlist = [
        {'symbol': '300015.SZ', 'name': '爱尔眼科'},
        {'symbol': '300124.SZ', 'name': '汇川技术'},
        {'symbol': '600048.SH', 'name': '保利发展'},
        {'symbol': '600519.SH', 'name': '贵州茅台'},
    ]

    all_data = []

    for stock in watchlist:
        symbol = stock['symbol']
        print(f"\n处理 {symbol}...")

        # 获取历史数据
        data = handler.get_watchlist_data([stock], fetch_history=True)
        if symbol in data and data[symbol].get('history') is not None:
            df = data[symbol]['history']
            all_data.append(df)

            # 训练个股模型
            predictor.train(df, symbol)

    # 训练通用模型
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        predictor.train_generic_model(combined_df)

    print("\n所有模型训练完成!")


if __name__ == "__main__":
    train_all_models()
