#!/usr/bin/env python3
"""
从缓存数据训练增强版 LightGBM 模型
使用已收集的中证 500 成分股历史数据
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from typing import List
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.lgb_predictor import LGBPredictor, FeatureEngineer


def load_cached_data(cache_dir: str = None) -> List[pd.DataFrame]:
    """
    加载缓存中的所有股票数据

    Args:
        cache_dir: 缓存目录

    Returns:
        DataFrame 列表
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), '../data/zz500_cache')

    if not os.path.exists(cache_dir):
        print(f"缓存目录不存在：{cache_dir}")
        return []

    all_data = []
    files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]

    print(f"发现 {len(files)} 个缓存文件")

    for file in files:
        try:
            with open(os.path.join(cache_dir, file), 'rb') as f:
                df = pickle.load(f)
            if len(df) >= 100:  # 至少 100 条数据
                all_data.append(df)
        except Exception as e:
            print(f"加载 {file} 失败：{e}")

    return all_data


def train_from_cache():
    """从缓存数据训练模型"""
    print("=" * 60)
    print("从缓存数据训练增强版 LightGBM 模型")
    print("=" * 60)

    # 加载缓存数据
    all_data = load_cached_data()

    if not all_data:
        print("没有可用的缓存数据")
        return

    # 合并数据
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n合并后总数据量：{len(combined)} 条")
    print(f"股票数量：{len(all_data)} 只")

    # 训练模型
    predictor = LGBPredictor(model_dir='./models/lgb_enhanced')

    # 准备数据
    X, y = predictor.prepare_data(combined)

    # 过滤无效数据
    non_zero_mask = y != 0
    X = X[non_zero_mask]
    y = y[non_zero_mask]

    print(f"\n有效样本数：{len(X)}")

    if len(X) < 500:
        print(f"数据量不足，无法训练")
        return

    # 转换为目标分类
    y_class = np.zeros(len(y))
    y_class[y > 0.02] = 1
    y_class[y < -0.02] = -1

    # 检查类别
    unique_classes = np.unique(y_class)
    n_classes = len(unique_classes)

    print(f"类别分布：{unique_classes}")

    if n_classes < 2:
        print("类别单一，无法训练")
        return

    # 处理二分类情况
    if n_classes == 2:
        y_class = np.where(y_class == -1, 0, 1)

    # 划分数据集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )

    print(f"\n训练集：{len(X_train)} 样本")
    print(f"测试集：{len(X_test)} 样本")

    # 训练模型
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, classification_report

    model = lgb.LGBMClassifier(
        objective='multiclass' if n_classes > 2 else 'binary',
        num_class=n_classes if n_classes > 2 else None,
        metric='multi_logloss' if n_classes > 2 else 'binary_logloss',
        boosting_type='gbdt',
        num_leaves=63,
        learning_rate=0.03,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        n_estimators=200,
        max_depth=8,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=0.1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(20, verbose=False)]
    )

    # 评估
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"测试集准确率：{accuracy:.2%}")
    print(f"{'='*60}")

    # 特征重要性
    feature_names = [
        'return_1d', 'return_3d', 'return_5d', 'return_10d',
        'volatility_5d', 'volatility_10d',
        'price_ma5_ratio', 'price_ma10_ratio', 'price_ma20_ratio',
        'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_position',
        'volume_ratio', 'obv_change_5d',
        'ma5_ma10', 'ma10_ma20', 'ma20_ma60',
        'highest_10d', 'lowest_10d'
    ]

    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n特征重要性 Top 10:")
    print(importance.head(10).to_string(index=False))

    # 保存模型
    model_path = os.path.join(predictor.model_dir, 'zz500_cache.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n模型已保存到：{model_path}")

    feature_path = os.path.join(predictor.model_dir, 'zz500_cache_features.pkl')
    with open(feature_path, 'wb') as f:
        pickle.dump(feature_names, f)

    print("\n训练完成!")
    return accuracy


if __name__ == "__main__":
    train_from_cache()
