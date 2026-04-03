#!/usr/bin/env python3
"""
使用 30 分钟级别数据训练 LightGBM 模型
预测未来 3 根 K 线（90 分钟）的走势
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_handler import DataHandler, _rate_limit, AKSHARE_AVAILABLE

# 监控股票池
TRAINING_STOCKS = [
    # A 股
    {"symbol": "300015.SZ", "name": "爱尔眼科"},
    {"symbol": "300124.SZ", "name": "汇川技术"},
    {"symbol": "600048.SH", "name": "保利发展"},
    {"symbol": "600519.SH", "name": "贵州茅台"},
    {"symbol": "000001.SZ", "name": "平安银行"},
    {"symbol": "000333.SZ", "name": "美的集团"},
    {"symbol": "000858.SZ", "name": "五粮液"},
    {"symbol": "002594.SZ", "name": "比亚迪"},
    {"symbol": "601318.SH", "name": "中国平安"},
    {"symbol": "601398.SH", "name": "工商银行"},
    # 港股
    {"symbol": "3690.HK", "name": "美团-W"},
    {"symbol": "0700.HK", "name": "腾讯控股"},
    {"symbol": "9988.HK", "name": "阿里巴巴-W"},
]


class FeatureEngineer30m:
    """30 分钟级别特征工程"""

    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 30 分钟级别的技术特征
        """
        features = pd.DataFrame(index=df.index)

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        open_price = df['open'].values

        # === 价格动量特征 ===
        # 收益率（1-10 根 K 线）
        features['return_1'] = pd.Series(close).pct_change(1)
        features['return_2'] = pd.Series(close).pct_change(2)
        features['return_3'] = pd.Series(close).pct_change(3)
        features['return_5'] = pd.Series(close).pct_change(5)
        features['return_10'] = pd.Series(close).pct_change(10)

        # 波动率
        features['volatility_5'] = pd.Series(features['return_1']).rolling(5).std()
        features['volatility_10'] = pd.Series(features['return_1']).rolling(10).std()
        features['volatility_20'] = pd.Series(features['return_1']).rolling(20).std()

        # === 均线系统 ===
        features['ma5_ratio'] = close / pd.Series(close).rolling(5).mean() - 1
        features['ma10_ratio'] = close / pd.Series(close).rolling(10).mean() - 1
        features['ma20_ratio'] = close / pd.Series(close).rolling(20).mean() - 1
        features['ma60_ratio'] = close / pd.Series(close).rolling(60).mean() - 1

        # 均线交叉
        ma5 = pd.Series(close).rolling(5).mean()
        ma10 = pd.Series(close).rolling(10).mean()
        ma20 = pd.Series(close).rolling(20).mean()
        ma60 = pd.Series(close).rolling(60).mean()

        features['ma5_ma10'] = ma5 / ma10 - 1
        features['ma10_ma20'] = ma10 / ma20 - 1
        features['ma20_ma60'] = ma20 / ma60 - 1

        # === RSI ===
        delta = pd.Series(close).diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features['rsi_14'] = 100 - (100 / (1 + rs))

        # RSI 6
        gain6 = (delta.where(delta > 0, 0)).rolling(6).mean()
        loss6 = (-delta.where(delta < 0, 0)).rolling(6).mean()
        features['rsi_6'] = 100 - (100 / (1 + gain6 / (loss6 + 1e-10)))

        # === MACD ===
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = macd - signal
        features['macd_slope'] = macd.diff()

        # === KDJ ===
        low_min = pd.Series(low).rolling(9).min()
        high_max = pd.Series(high).rolling(9).max()
        rsv = (close - low_min) / (high_max - low_min + 1e-10) * 100
        features['kdj_k'] = rsv.ewm(com=2).mean()
        features['kdj_d'] = features['kdj_k'].ewm(com=2).mean()
        features['kdj_j'] = 3 * features['kdj_k'] - 2 * features['kdj_d']
        features['kdj_cross'] = features['kdj_k'] - features['kdj_d']

        # === 布林带 ===
        ma20_bb = pd.Series(close).rolling(20).mean()
        std20 = pd.Series(close).rolling(20).std()
        upper = ma20_bb + 2 * std20
        lower = ma20_bb - 2 * std20
        features['bb_upper'] = (upper - close) / close
        features['bb_lower'] = (close - lower) / close
        features['bb_width'] = (upper - lower) / ma20_bb
        features['bb_position'] = (close - lower) / (upper - lower + 1e-10)

        # === ATR ===
        tr = pd.concat([
            pd.Series(high) - pd.Series(low),
            pd.Series(high) - pd.Series(close).shift(1),
            pd.Series(close).shift(1) - pd.Series(low)
        ], axis=1).max(axis=1)
        features['atr'] = tr.rolling(14).mean()
        features['atr_ratio'] = features['atr'] / pd.Series(close)

        # === 成交量特征 ===
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

        # === K 线形态 ===
        features['upper_shadow'] = (high - np.maximum(open_price, close)) / (close + 1e-10)
        features['lower_shadow'] = (np.minimum(open_price, close) - low) / (close + 1e-10)
        features['body_size'] = np.abs(close - open_price) / (close + 1e-10)
        features['gap'] = (open_price - pd.Series(close).shift(1)) / (pd.Series(close).shift(1) + 1e-10)

        # === 价格位置 ===
        features['high_10'] = (close - pd.Series(high).rolling(10).max()) / (pd.Series(high).rolling(10).max() + 1e-10)
        features['low_10'] = (close - pd.Series(low).rolling(10).min()) / (pd.Series(low).rolling(10).min() + 1e-10)
        features['high_20'] = (close - pd.Series(high).rolling(20).max()) / (pd.Series(high).rolling(20).max() + 1e-10)
        features['low_20'] = (close - pd.Series(low).rolling(20).min()) / (pd.Series(low).rolling(20).min() + 1e-10)

        # === 时间特征 ===
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
            features['hour'] = dates.dt.hour
            features['minute'] = dates.dt.minute
            features['day_of_week'] = dates.dt.dayofweek
            # 是否是开盘/收盘时段
            features['is_open'] = ((dates.dt.hour == 9) | (dates.dt.hour == 13)).astype(int)
            features['is_close'] = ((dates.dt.hour == 11) | (dates.dt.hour == 14) | (dates.dt.hour == 15)).astype(int)

        return features

    @staticmethod
    def calculate_target(df: pd.DataFrame, horizon: int = 3) -> np.ndarray:
        """
        计算预测目标：未来 N 根 K 线的收益率

        Args:
            df: 数据 DataFrame
            horizon: 预测周期（默认 3 根 K 线 = 90 分钟）

        Returns:
            目标数组
        """
        close = df['close'].values
        target = np.zeros(len(close))

        for i in range(len(close) - horizon):
            target[i] = (close[i + horizon] - close[i]) / close[i]

        return target


def fetch_training_data(stocks: List[Dict]) -> Dict[str, pd.DataFrame]:
    """
    获取训练数据（30 分钟级别）

    注意：延时已在 DataHandler 内部处理
    """
    handler = DataHandler(force_refresh=True)
    all_data = {}

    print(f"开始获取 {len(stocks)} 只股票的 30 分钟数据...")

    for i, stock in enumerate(stocks):
        symbol = stock['symbol']
        name = stock['name']

        print(f"[{i+1}/{len(stocks)}] {name} ({symbol})...")

        # 获取数据（延时在 DataHandler 内部处理）
        df = handler.fetch_stock_data(symbol, force_refresh=True)

        if df is not None and len(df) >= 60:
            all_data[symbol] = df
            print(f"  ✓ {len(df)} 条数据, 最新价格: {df['close'].iloc[-1]:.2f}")
        else:
            print(f"  ✗ 数据不足")

    return all_data


def prepare_training_data(all_data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
    """
    准备训练数据
    """
    all_features = []
    all_targets = []

    for symbol, df in all_data.items():
        # 计算特征
        features = FeatureEngineer30m.calculate_features(df)

        # 计算目标（未来 3 根 K 线收益率）
        target = FeatureEngineer30m.calculate_target(df, horizon=3)

        # 过滤无效数据
        valid_mask = ~(features.isna().any(axis=1)) & (target != 0)
        features = features[valid_mask]
        target = target[valid_mask]

        if len(features) > 50:
            all_features.append(features.values)
            all_targets.append(target)

    if not all_features:
        return None, None

    X = np.vstack(all_features)
    y = np.concatenate(all_targets)

    return X, y


def train_model(X: np.ndarray, y: np.ndarray) -> Dict:
    """
    训练 LightGBM 模型
    """
    # 转换为分类问题
    # 收益率 > 1%: 上涨 (1)
    # 收益率 < -1%: 下跌 (0)
    y_class = np.zeros(len(y))
    y_class[y > 0.008] = 1  # 0.8% 以上算上涨
    y_class[y < -0.008] = 0  # -0.8% 以下算下跌

    # 过滤中间值（震荡）
    valid_mask = (y > 0.008) | (y < -0.008)
    X = X[valid_mask]
    y_class = y_class[valid_mask]

    print(f"\n训练数据统计:")
    print(f"  总样本数: {len(X)}")
    print(f"  上涨样本: {np.sum(y_class == 1)}")
    print(f"  下跌样本: {np.sum(y_class == 0)}")

    # 划分训练集和测试集（时序数据不能用随机划分）
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_class[:split_idx], y_class[split_idx:]

    # LightGBM 参数
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 200,
        'max_depth': 6,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    }

    print(f"\n开始训练 LightGBM 模型...")

    model = lgb.LGBMClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(20, verbose=False)]
    )

    # 评估
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n模型评估:")
    print(f"  准确率: {accuracy:.2%}")
    print(f"\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['下跌', '上涨']))

    return {
        'model': model,
        'accuracy': accuracy,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_importance': dict(zip(
            FeatureEngineer30m.calculate_features(
                pd.DataFrame({'close': [1], 'high': [1], 'low': [1], 'volume': [1], 'open': [1]})
            ).columns,
            model.feature_importances_
        ))
    }


def save_model(model_data: Dict, model_dir: str):
    """保存模型"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 保存模型
    model_path = os.path.join(model_dir, 'model_30m.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n模型已保存到: {model_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("LightGBM 模型训练 - 30 分钟级别")
    print("=" * 60)
    print(f"预测目标: 未来 3 根 K 线（90 分钟）走势")
    print(f"训练股票: {len(TRAINING_STOCKS)} 只")
    print("=" * 60)

    # 获取数据
    all_data = fetch_training_data(TRAINING_STOCKS)

    if not all_data:
        print("未获取到任何数据，退出训练")
        return

    # 准备训练数据
    X, y = prepare_training_data(all_data)

    if X is None or len(X) < 100:
        print(f"训练数据不足 ({len(X) if X is not None else 0} 条)，退出训练")
        return

    # 训练模型
    model_data = train_model(X, y)

    # 保存模型
    model_dir = os.path.join(os.path.dirname(__file__), '../models/lgb_30m')
    save_model(model_data, model_dir)

    # 显示特征重要性
    print("\n特征重要性 Top 10:")
    importance = sorted(model_data['feature_importance'].items(), key=lambda x: x[1], reverse=True)
    for name, score in importance[:10]:
        print(f"  {name}: {score}")

    print("\n训练完成!")


if __name__ == "__main__":
    main()