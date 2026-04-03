#!/usr/bin/env python3
"""
使用中证 500 成分股 30 分钟级别数据训练 LightGBM 模型
预测未来 3 根 K 线（90 分钟）的走势

用法：
    python3 strategy/train_lgb_30m_full.py
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FeatureEngineer30m:
    """30 分钟级别特征工程"""

    FEATURE_NAMES = None  # 缓存特征名称

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

        # 缓存特征名称
        if FeatureEngineer30m.FEATURE_NAMES is None:
            FeatureEngineer30m.FEATURE_NAMES = features.columns.tolist()

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


def load_cached_data(cache_dir: str = None, use_existing_csv: bool = True) -> Dict[str, pd.DataFrame]:
    """
    加载缓存数据

    Args:
        cache_dir: 缓存目录（中证500 pkl 文件）
        use_existing_csv: 是否使用现有 CSV 文件（data 目录下的 _30m.csv）

    Returns:
        股票数据字典 {symbol: DataFrame}
    """
    all_data = {}

    # 首先尝试加载现有的 CSV 文件
    if use_existing_csv:
        data_dir = os.path.join(os.path.dirname(__file__), '../data')
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('_30m.csv')]

        print(f"加载现有 CSV 数据：{len(csv_files)} 个文件")

        for i, file in enumerate(csv_files):
            symbol = file.replace('_30m.csv', '')
            csv_file = os.path.join(data_dir, file)

            try:
                df = pd.read_csv(csv_file)
                if df is not None and len(df) >= 60:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date').reset_index(drop=True)
                    all_data[symbol] = df
                    print(f"  ✓ {symbol}: {len(df)} 条数据")
            except Exception as e:
                print(f"  加载失败 {symbol}: {e}")

    # 如果现有数据不足，尝试加载中证500缓存
    if len(all_data) < 50 and cache_dir:
        if not os.path.exists(cache_dir):
            print(f"缓存目录不存在: {cache_dir}")
            return all_data

        pkl_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]

        print(f"加载中证500缓存数据：{len(pkl_files)} 个文件")

        for i, file in enumerate(pkl_files):
            symbol = file.replace('.pkl', '')
            if symbol in all_data:  # 跳过已加载的
                continue
            cache_file = os.path.join(cache_dir, file)

            try:
                with open(cache_file, 'rb') as f:
                    df = pickle.load(f)

                if df is not None and len(df) >= 60:
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.sort_values('date').reset_index(drop=True)
                    all_data[symbol] = df
                    if (i + 1) % 50 == 0:
                        print(f"  已加载 {i + 1}/{len(pkl_files)} 个文件")
            except Exception as e:
                print(f"  加载失败 {symbol}: {e}")

    print(f"成功加载 {len(all_data)} 只股票数据")
    return all_data


def prepare_training_data(all_data: Dict[str, pd.DataFrame], horizon: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    准备训练数据

    Args:
        all_data: 所有股票数据
        horizon: 预测周期
    """
    all_features = []
    all_targets = []
    all_symbols = []  # 记录每个样本对应的股票代码

    print("计算特征和目标...")

    for i, (symbol, df) in enumerate(all_data.items()):
        try:
            # 计算特征
            features = FeatureEngineer30m.calculate_features(df)

            # 计算目标（未来 3 根 K 线收益率）
            target = FeatureEngineer30m.calculate_target(df, horizon=horizon)

            # 过滤无效数据（前60行特征不完整，后horizon行目标无法计算）
            valid_mask = ~(features.isna().any(axis=1))
            valid_mask[:60] = False  # 前60行特征不完整
            valid_mask[-horizon:] = False  # 最后几行目标无法计算

            features_valid = features[valid_mask]
            target_valid = target[valid_mask]

            if len(features_valid) > 30:
                all_features.append(features_valid.values)
                all_targets.append(target_valid)
                all_symbols.extend([symbol] * len(features_valid))

        except Exception as e:
            print(f"  特征计算失败 {symbol}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/{len(all_data)} 只股票")

    if not all_features:
        return None, None, None

    X = np.vstack(all_features)
    y = np.concatenate(all_targets)
    symbols = np.array(all_symbols)

    print(f"总样本数: {len(X)}")
    return X, y, symbols


def train_model(X: np.ndarray, y: np.ndarray, symbols: np.ndarray = None) -> Dict:
    """
    训练 LightGBM 模型

    使用交叉验证防止过拟合
    """
    # 转换为分类问题
    # 收益率 > 0.8%: 上涨 (1)
    # 收益率 < -0.8%: 下跌 (0)
    # 中间值过滤（震荡市场）
    y_class = np.zeros(len(y))
    y_class[y > 0.008] = 1  # 0.8% 以上算上涨
    y_class[y < -0.008] = 0  # -0.8% 以下算下跌

    # 过滤中间值（震荡）
    valid_mask = (y > 0.008) | (y < -0.008)
    X_filtered = X[valid_mask]
    y_filtered = y_class[valid_mask]

    if symbols is not None:
        symbols_filtered = symbols[valid_mask]
    else:
        symbols_filtered = None

    print(f"\n训练数据统计:")
    print(f"  总样本数: {len(X_filtered)}")
    print(f"  上涨样本: {np.sum(y_filtered == 1)} ({np.sum(y_filtered == 1)/len(X_filtered)*100:.1f}%)")
    print(f"  下跌样本: {np.sum(y_filtered == 0)} ({np.sum(y_filtered == 0)/len(X_filtered)*100:.1f}%)")

    # 使用时序交叉验证（5折）
    tscv = TimeSeriesSplit(n_splits=5)

    # LightGBM 参数（优化后）
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
        'n_estimators': 300,
        'max_depth': 6,
        'min_child_samples': 30,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42
    }

    print(f"\n开始训练 LightGBM 模型（5折交叉验证）...")

    # 交叉验证
    cv_scores = []
    models = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_filtered)):
        X_train, X_test = X_filtered[train_idx], X_filtered[test_idx]
        y_train, y_test = y_filtered[train_idx], y_filtered[test_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(20, verbose=False)]
        )

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores.append(accuracy)
        models.append(model)

        print(f"  Fold {fold + 1}: Accuracy = {accuracy:.4f}")

    avg_accuracy = np.mean(cv_scores)
    print(f"\n平均交叉验证准确率: {avg_accuracy:.4f}")

    # 使用最后一折的模型作为最终模型（或取平均）
    final_model = models[-1]

    # 在全部数据上评估
    y_pred_all = final_model.predict(X_filtered)
    overall_accuracy = accuracy_score(y_filtered, y_pred_all)

    print(f"\n整体评估:")
    print(f"  准确率: {overall_accuracy:.2%}")
    print(f"\n分类报告:")
    print(classification_report(y_filtered, y_pred_all, target_names=['下跌', '上涨']))

    # 混淆矩阵
    cm = confusion_matrix(y_filtered, y_pred_all)
    print(f"\n混淆矩阵:")
    print(f"  预测下跌 | 实际下跌: {cm[0][0]}, 实际上涨: {cm[0][1]}")
    print(f"  预测上涨 | 实际下跌: {cm[1][0]}, 实际上涨: {cm[1][1]}")

    return {
        'model': final_model,
        'cv_accuracy': avg_accuracy,
        'overall_accuracy': overall_accuracy,
        'cv_scores': cv_scores,
        'train_samples': len(X_filtered),
        'test_samples': len(X_filtered) - int(len(X_filtered) * 0.8),
        'feature_importance': dict(zip(
            FeatureEngineer30m.FEATURE_NAMES if FeatureEngineer30m.FEATURE_NAMES else [],
            final_model.feature_importances_
        )),
        'feature_names': FeatureEngineer30m.FEATURE_NAMES,
        'params': params
    }


def save_model(model_data: Dict, model_dir: str):
    """保存模型"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 保存模型
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n模型已保存到: {model_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("LightGBM 模型训练 - 30 分钟级别")
    print("=" * 60)
    print(f"预测目标: 未来 3 根 K 线（90 分钟）走势")
    print("=" * 60)

    # 加载缓存数据（优先使用现有 CSV，其次使用中证500缓存）
    cache_dir = os.path.join(os.path.dirname(__file__), '../data/zz500_cache')
    all_data = load_cached_data(cache_dir, use_existing_csv=True)

    if not all_data:
        print("未加载到任何数据，退出训练")
        return

    # 准备训练数据
    X, y, symbols = prepare_training_data(all_data, horizon=3)

    if X is None or len(X) < 100:
        print(f"训练数据不足 ({len(X) if X is not None else 0} 条)，退出训练")
        return

    # 训练模型
    model_data = train_model(X, y, symbols)

    # 保存模型
    model_dir = os.path.join(os.path.dirname(__file__), '../models/lgb_zz500')
    save_model(model_data, model_dir)

    # 显示特征重要性
    print("\n特征重要性 Top 15:")
    importance = sorted(model_data['feature_importance'].items(), key=lambda x: x[1], reverse=True)
    for name, score in importance[:15]:
        print(f"  {name}: {score}")

    print("\n训练完成!")
    print(f"交叉验证平均准确率: {model_data['cv_accuracy']:.2%}")


if __name__ == "__main__":
    main()