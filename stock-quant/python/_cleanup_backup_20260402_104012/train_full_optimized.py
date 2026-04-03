#!/usr/bin/env python3
"""
全面优化的 LightGBM 模型训练脚本
优化内容：
1. 调整涨跌阈值（2% -> 2.5%）
2. 增强特征工程（KDJ 完整、DDI、CCI、SAR、资金流等）
3. 样本加权（近期数据更高权重）
4. XGBoost + LightGBM 集成
5. 超参数优化
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from typing import List, Dict, Tuple
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.lgb_predictor import FeatureEngineer


def load_cached_data(cache_dir: str = None) -> Tuple[List[pd.DataFrame], List[str]]:
    """加载缓存中的所有股票数据"""
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), '../data/zz500_cache')

    if not os.path.exists(cache_dir):
        print(f"缓存目录不存在：{cache_dir}")
        return [], []

    all_data = []
    symbols = []
    files = sorted([f for f in os.listdir(cache_dir) if f.endswith('.pkl')])

    print(f"发现 {len(files)} 个缓存文件")

    for file in files:
        try:
            with open(os.path.join(cache_dir, file), 'rb') as f:
                df = pickle.load(f)
            if len(df) >= 100:
                all_data.append(df)
                symbols.append(file.replace('.pkl', ''))
        except Exception as e:
            print(f"加载 {file} 失败：{e}")

    return all_data, symbols


class EnhancedFeatureEngineer:
    """增强版特征工程"""

    @staticmethod
    def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """计算增强的技术指标"""
        features = pd.DataFrame(index=df.index)
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        open_price = df['open'].values

        # === 基础特征（原有） ===
        # 价格动量
        features['return_1d'] = pd.Series(close).pct_change(1)
        features['return_3d'] = pd.Series(close).pct_change(3)
        features['return_5d'] = pd.Series(close).pct_change(5)
        features['return_10d'] = pd.Series(close).pct_change(10)

        # 波动率
        features['volatility_5d'] = pd.Series(features['return_1d']).rolling(5).std()
        features['volatility_10d'] = pd.Series(features['return_1d']).rolling(10).std()

        # 均线
        features['price_ma5_ratio'] = close / pd.Series(close).rolling(5).mean() - 1
        features['price_ma10_ratio'] = close / pd.Series(close).rolling(10).mean() - 1
        features['price_ma20_ratio'] = close / pd.Series(close).rolling(20).mean() - 1
        features['ma5_ma10'] = pd.Series(close).rolling(5).mean() / pd.Series(close).rolling(10).mean() - 1
        features['ma10_ma20'] = pd.Series(close).rolling(10).mean() / pd.Series(close).rolling(20).mean() - 1
        features['ma20_ma60'] = pd.Series(close).rolling(20).mean() / pd.Series(close).rolling(60).mean() - 1

        # === 增强特征 1: 完整 KDJ ===
        low_min = pd.Series(low).rolling(9).min()
        high_max = pd.Series(high).rolling(9).max()
        rsv = (close - low_min) / (high_max - low_min + 1e-10) * 100
        features['kdj_k'] = rsv.ewm(com=2).mean()
        features['kdj_d'] = features['kdj_k'].ewm(com=2).mean()
        features['kdj_j'] = 3 * features['kdj_k'] - 2 * features['kdj_d']
        features['kdj_cross'] = features['kdj_k'] - features['kdj_d']  # KDJ 金叉/死叉

        # === 增强特征 2: DDI (方向分歧指标) ===
        tr = pd.Series(high).rolling(14).max() - pd.Series(low).rolling(14).min()
        dmz = np.maximum(np.abs(high - pd.Series(low).shift(1)), np.abs(high - low))
        dmf = np.maximum(np.abs(pd.Series(low).shift(1) - low), np.abs(high - low))
        diz = dmz.rolling(14).mean()
        dif = dmf.rolling(14).mean()
        features['ddi'] = (diz - dif) / (diz + dif + 1e-10)

        # === 增强特征 3: CCI (顺势指标) ===
        tp = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
        ma = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        features['cci'] = (tp - ma) / (0.015 * mad + 1e-10)

        # === 增强特征 4: SAR (抛物线转向) ===
        # 简化版 SAR 计算
        af = 0.02
        sar = np.zeros(len(close))
        ep = low[0]
        trend = 1
        for i in range(2, len(close)):
            if trend == 1:
                sar[i] = sar[i-1] + af * (ep - sar[i-1]) if i > 0 else low[0]
                if low[i] < sar[i]:
                    trend = -1
                    af = 0.02
            else:
                sar[i] = sar[i-1] + af * (ep - sar[i-1]) if i > 0 else high[0]
                if high[i] > sar[i]:
                    trend = 1
                    af = 0.02
            ep = max(ep, high[i]) if trend == 1 else min(ep, low[i])
            af = min(0.2, af + 0.02)
        features['sar_position'] = (close - sar) / (close + 1e-10)

        # === 增强特征 5: 资金流特征 ===
        # MFI 简化版
        tp = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
        positive_flow = tp * pd.Series(volume) * (tp > tp.shift(1)).astype(int)
        negative_flow = tp * pd.Series(volume) * (tp < tp.shift(1)).astype(int)
        mfi_ratio = positive_flow.rolling(14).sum() / (negative_flow.rolling(14).sum() + 1e-10)
        features['mfi'] = 100 - (100 / (1 + mfi_ratio))

        # 资金流向强度
        price_change = pd.Series(close).diff()
        volume_change = pd.Series(volume).diff()
        features['money_flow'] = (price_change * pd.Series(volume) / 1e6).rolling(5).sum()

        # === 增强特征 6: ATR (平均真实波幅) ===
        tr = pd.concat([
            pd.Series(high) - pd.Series(low),
            pd.Series(high) - pd.Series(close).shift(1),
            pd.Series(close).shift(1) - pd.Series(low)
        ], axis=1).max(axis=1)
        features['atr'] = tr.rolling(14).mean()
        features['atr_ratio'] = features['atr'] / pd.Series(close)

        # === 增强特征 7: 成交量特征 ===
        features['volume_ratio'] = pd.Series(volume).rolling(5).mean() / pd.Series(volume).rolling(20).mean()

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
        features['obv_change_5d'] = pd.Series(obv).pct_change(5)

        # === 增强特征 8: K 线形态特征 ===
        # 上下影线
        features['upper_shadow'] = (high - np.maximum(open_price, close)) / (close + 1e-10)
        features['lower_shadow'] = (np.minimum(open_price, close) - low) / (close + 1e-10)
        features['body_size'] = np.abs(close - open_price) / (close + 1e-10)

        # 缺口
        features['gap'] = (open_price - pd.Series(close).shift(1)) / pd.Series(close).shift(1)

        # === 增强特征 9: 布林带 ===
        ma20 = pd.Series(close).rolling(20).mean()
        std20 = pd.Series(close).rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        features['bb_position'] = (close - lower) / (upper - lower + 1e-10)
        features['bb_width'] = (upper - lower) / ma20

        # === 增强特征 10: RSI 优化 ===
        delta = pd.Series(close).diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features['rsi'] = 100 - (100 / (1 + rs))
        # 多周期 RSI
        rs6 = (delta.where(delta > 0, 0)).rolling(6).mean() / ((-delta.where(delta < 0, 0)).rolling(6).mean() + 1e-10)
        features['rsi_6d'] = 100 - (100 / (1 + rs6))
        rs24 = (delta.where(delta > 0, 0)).rolling(24).mean() / ((-delta.where(delta < 0, 0)).rolling(24).mean() + 1e-10)
        features['rsi_24d'] = 100 - (100 / (1 + rs24))

        # === 增强特征 11: MACD 优化 ===
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = macd - signal
        features['macd_slope'] = macd.diff()  # MACD 斜率

        # === 增强特征 12: 价格极值 ===
        features['highest_10d'] = (close - pd.Series(high).rolling(10).max()) / pd.Series(high).rolling(10).max()
        features['lowest_10d'] = (close - pd.Series(low).rolling(10).min()) / pd.Series(low).rolling(10).min()
        features['highest_20d'] = (close - pd.Series(high).rolling(20).max()) / pd.Series(high).rolling(20).max()
        features['lowest_20d'] = (close - pd.Series(low).rolling(20).min()) / pd.Series(low).rolling(20).min()

        return features

    @staticmethod
    def calculate_target(df: pd.DataFrame, horizon: int = 3, threshold: float = 0.025) -> np.ndarray:
        """
        计算目标变量（优化版）

        Args:
            df: 数据框
            horizon: 预测周期
            threshold: 涨跌阈值（优化为 2.5%）

        Returns:
            目标标签数组：-1=下跌，0=震荡，1=上涨
        """
        close = df['close'].values
        n = len(close)
        future_return = np.zeros(n)

        for i in range(n - horizon):
            future_return[i] = (close[i + horizon] - close[i]) / close[i]

        # 使用优化后的阈值
        y = np.zeros(n)
        y[future_return > threshold] = 1    # 上涨
        y[future_return < -threshold] = -1  # 下跌
        # 中间为震荡 (0)

        return y

    @staticmethod
    def add_fundamental_features(features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """添加基本面特征"""
        try:
            from data.fundamental_feature import FundamentalFeatureEngineer
            fund_engineer = FundamentalFeatureEngineer()
            fundamental = fund_engineer.get_fundamental_data(symbol)
            if fundamental is not None and len(fundamental) > 0:
                fund_features = fund_engineer.calculate_features(fundamental)
                for key, value in fund_features.items():
                    if value is not None:
                        features[f'fund_{key}'] = value
        except Exception as e:
            pass
        return features


def prepare_features(all_data: List[pd.DataFrame], symbols: List[str], use_fundamental: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    准备特征和标签（包含样本权重）

    Returns:
        (特征矩阵 X, 标签 y, 特征名称，样本权重)
    """
    feature_engineer = EnhancedFeatureEngineer()

    all_X = []
    all_y = []
    all_weights = []
    feature_names = None

    for i, (df, symbol) in enumerate(zip(all_data, symbols)):
        try:
            # 计算技术面特征
            features = feature_engineer.calculate_technical_features(df)

            # 添加基本面特征
            if use_fundamental:
                features = feature_engineer.add_fundamental_features(features, symbol)

            # 计算目标
            target = feature_engineer.calculate_target(df, horizon=3, threshold=0.025)

            # 过滤无效数据（去除最后的 horizon 个样本）
            n_valid = len(features) - 3
            if n_valid <= 0:
                continue

            X = features.iloc[:n_valid].values
            y = target[:n_valid]

            # 只保留有效样本（非震荡）
            valid_mask = y != 0
            X = X[valid_mask]
            y = y[valid_mask]

            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)

                # 样本权重：近期数据权重更高
                weights = np.linspace(0.5, 1.5, len(X))
                all_weights.append(weights)

                if feature_names is None:
                    feature_names = list(features.columns)

        except Exception as e:
            print(f"处理第 {i} 只股票 ({symbol}) 失败：{e}")

    if len(all_X) == 0:
        return np.array([]), np.array([]), [], np.array([])

    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)
    weights_combined = np.concatenate(all_weights)

    # 转换标签：-1 -> 0, 1 -> 1
    y_binary = (y_combined + 1) / 2
    y_binary = y_binary.astype(int)

    print(f"\n特征准备完成:")
    print(f"  样本数：{len(X_combined)}")
    print(f"  特征数：{len(feature_names)}")
    print(f"  下跌样本 (0): {np.sum(y_binary == 0)} ({np.sum(y_binary == 0) / len(y_binary) * 100:.1f}%)")
    print(f"  上涨样本 (1): {np.sum(y_binary == 1)} ({np.sum(y_binary == 1) / len(y_binary) * 100:.1f}%)")

    return X_combined, y_binary, feature_names, weights_combined


def train_ensemble_model(X: np.ndarray, y: np.ndarray, feature_names: List[str], sample_weights: np.ndarray) -> Dict:
    """
    训练集成模型（LightGBM + XGBoost）
    """
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.metrics import accuracy_score, classification_report
    import lightgbm as lgb

    # 尝试导入 xgboost
    try:
        import xgboost as xgb
        has_xgb = True
    except ImportError:
        print("警告：xgboost 未安装，将只使用 LightGBM")
        has_xgb = False

    # y 已经是 0/1 标签
    y_class = y.astype(int)

    unique_classes = np.unique(y_class)
    n_classes = len(unique_classes)

    print(f"\n类别分布：{unique_classes}")
    print(f"  类别 0 (下跌): {np.sum(y_class == 0)}")
    print(f"  类别 1 (上涨): {np.sum(y_class == 1)}")

    # 划分训练测试集
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y_class, sample_weights, test_size=0.2, random_state=42, stratify=y_class
    )

    # === LightGBM 模型（优化参数） ===
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.05,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.75,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'reg_alpha': 0.05,
        'reg_lambda': 0.05,
        'n_estimators': 300,
        'verbose': -1,
        'class_weight': 'balanced'
    }

    print("\n训练 LightGBM 模型...")
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(30, verbose=False)]
    )

    lgb_pred = lgb_model.predict(X_test)
    lgb_accuracy = accuracy_score(y_test, lgb_pred)
    print(f"LightGBM 测试集准确率：{lgb_accuracy:.4f}")

    # === XGBoost 模型（如果可用） ===
    if has_xgb:
        print("\n训练 XGBoost 模型...")
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'min_child_weight': 2,
            'reg_alpha': 0.05,
            'reg_lambda': 0.05,
            'random_state': 42,
            'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1)
        }

        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        xgb_pred = xgb_model.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        print(f"XGBoost 测试集准确率：{xgb_accuracy:.4f}")

        # === 集成预测 ===
        print("\n集成模型...")
        # 简单平均集成
        lgb_proba = lgb_model.predict_proba(X_test)
        xgb_proba = xgb_model.predict_proba(X_test)
        ensemble_proba = (lgb_proba + xgb_proba) / 2
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        print(f"集成模型测试集准确率：{ensemble_accuracy:.4f}")
    else:
        ensemble_accuracy = lgb_accuracy
        xgb_accuracy = None

    # === 交叉验证 ===
    print("\n5 折交叉验证...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_idx, val_idx in cv.split(X, y_class):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_class[train_idx], y_class[val_idx]
        w_tr = sample_weights[train_idx]

        lgb_cv = lgb.LGBMClassifier(**lgb_params)
        lgb_cv.fit(X_tr, y_tr, sample_weight=w_tr)
        cv_pred = lgb_cv.predict(X_val)
        cv_scores.append(accuracy_score(y_val, cv_pred))

    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"交叉验证准确率：{cv_mean:.4f} (+/- {cv_std * 2:.4f})")

    # === 特征重要性 ===
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n特征重要性 Top 15:")
    print(importance.head(15).to_string(index=False))

    # === 保存模型 ===
    model_dir = os.path.join(os.path.dirname(__file__), '../models/lgb_enhanced')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, 'zz500_full_optimized.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'lgb_model': lgb_model,
            'xgb_model': xgb_model if has_xgb else None,
            'feature_names': feature_names,
            'n_classes': n_classes
        }, f)

    print(f"\n模型已保存到：{model_path}")

    return {
        'status': 'success',
        'lgb_accuracy': lgb_accuracy,
        'xgb_accuracy': xgb_accuracy if has_xgb else None,
        'ensemble_accuracy': ensemble_accuracy,
        'cv_accuracy': cv_mean,
        'cv_std': cv_std,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'feature_importance': importance
    }


def main():
    """主函数"""
    print("=" * 60)
    print("全面优化的 LightGBM 模型训练")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/3] 加载缓存数据...")
    all_data, symbols = load_cached_data()

    if not all_data:
        print("没有找到缓存数据，请先运行 collect_zz500_data.py")
        return

    print(f"加载了 {len(symbols)} 只股票的数据")

    # 2. 准备特征
    print("\n[2/3] 准备特征...")
    X, y, feature_names, sample_weights = prepare_features(all_data, symbols, use_fundamental=True)

    # 3. 训练模型
    print("\n[3/3] 训练模型...")
    result = train_ensemble_model(X, y, feature_names, sample_weights)

    # 打印结果
    print("\n" + "=" * 60)
    print("训练结果")
    print("=" * 60)
    print(f"LightGBM 准确率：{result['lgb_accuracy']:.4f}")
    if result['xgb_accuracy']:
        print(f"XGBoost 准确率：{result['xgb_accuracy']:.4f}")
    print(f"集成模型准确率：{result['ensemble_accuracy']:.4f}")
    print(f"交叉验证准确率：{result['cv_accuracy']:.4f} (+/- {result['cv_std'] * 2:.4f})")
    print(f"训练样本：{result['train_size']}, 测试样本：{result['test_size']}")


if __name__ == "__main__":
    main()
