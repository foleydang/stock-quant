#!/usr/bin/env python3
"""
从缓存数据训练增强版 LightGBM 模型（带超参数调优）
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
    """
    加载缓存中的所有股票数据

    Returns:
        (DataFrame 列表，股票代码列表)
    """
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
            if len(df) >= 100:  # 至少 100 条数据
                all_data.append(df)
                symbols.append(file.replace('.pkl', ''))
        except Exception as e:
            print(f"加载 {file} 失败：{e}")

    return all_data, symbols


def prepare_features(all_data: List[pd.DataFrame], symbols: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    准备特征和标签（包含基本面特征）

    Returns:
        (特征矩阵 X, 标签 y, 特征名称)
    """
    feature_engineer = FeatureEngineer()

    all_X = []
    all_y = []

    for i, (df, symbol) in enumerate(zip(all_data, symbols)):
        try:
            # 计算技术面特征
            features = feature_engineer.calculate_features(df)
            # 添加基本面特征
            features = feature_engineer.add_fundamental_features(features, symbol)
            # 计算目标
            target = feature_engineer.calculate_target(df, horizon=3)

            # 过滤无效数据
            valid_mask = target != 0
            X = features.values[valid_mask]
            y = target[valid_mask]

            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
        except Exception as e:
            print(f"处理第 {i} 只股票失败：{e}")

    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)

    feature_names = list(features.columns)

    return X_combined, y_combined, feature_names


def train_with_cv(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict:
    """
    使用交叉验证训练模型

    Returns:
        训练结果
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    import lightgbm as lgb

    # 转换为分类标签
    y_class = np.zeros(len(y))
    y_class[y > 0.02] = 1
    y_class[y < -0.02] = -1

    # 处理二分类情况
    unique_classes = np.unique(y_class)
    n_classes = len(unique_classes)

    if n_classes < 2:
        print("类别单一，无法训练")
        return None

    if n_classes == 2:
        y_class = np.where(y_class == -1, 0, 1)

    print(f"\n类别分布：{unique_classes}")
    print(f"样本数：{len(X)}")

    # 模型参数
    params = {
        'objective': 'multiclass' if n_classes > 2 else 'binary',
        'num_class': n_classes if n_classes > 2 else None,
        'metric': 'multi_logloss' if n_classes > 2 else 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'max_depth': 8,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 30,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_estimators': 200,
        'verbose': -1
    }

    model = lgb.LGBMClassifier(**params)

    # 交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y_class, cv=cv, scoring='accuracy')

    print(f"\n交叉验证结果：")
    print(f"  平均准确率：{scores.mean():.2%} (+/- {scores.std() * 2:.2%})")
    print(f"  各折分数：{['{:.2%}'.format(s) for s in scores]}")

    # 训练最终模型
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(20, verbose=False)]
    )

    # 测试集评估
    y_pred = model.predict(X_test)
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n测试集准确率：{accuracy:.2%}")
    print(f"训练集大小：{len(X_train)}, 测试集大小：{len(X_test)}")

    # 特征重要性
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n特征重要性 Top 10:")
    print(importance.head(10).to_string(index=False))

    # 保存模型
    model_dir = os.path.join(os.path.dirname(__file__), '../models/lgb_enhanced')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, 'zz500_cv_optimized.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    feature_path = os.path.join(model_dir, 'zz500_cv_features.pkl')
    with open(feature_path, 'wb') as f:
        pickle.dump(feature_names, f)

    print(f"\n模型已保存到：{model_path}")

    return {
        'status': 'success',
        'cv_accuracy': scores.mean(),
        'test_accuracy': accuracy,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'feature_importance': importance
    }


def main():
    """主函数"""
    print("=" * 60)
    print("增强版 LightGBM 模型训练（带交叉验证）")
    print("=" * 60)

    # 加载缓存数据
    all_data, symbols = load_cached_data()

    if not all_data:
        print("没有可用的缓存数据")
        return

    print(f"加载了 {len(symbols)} 只股票的数据")
    total_samples = sum(len(df) for df in all_data)
    print(f"总样本数：{total_samples}")

    # 准备特征
    print("\n准备特征（包含基本面特征）...")
    X, y, feature_names = prepare_features(all_data, symbols)

    print(f"特征矩阵形状：{X.shape}")
    print(f"标签数组形状：{y.shape}")

    # 训练模型
    result = train_with_cv(X, y, feature_names)

    if result:
        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)
        print(f"\n模型性能总结:")
        print(f"  交叉验证准确率：{result['cv_accuracy']:.2%}")
        print(f"  测试集准确率：{result['test_accuracy']:.2%}")
        print(f"  训练样本：{result['train_size']}")
        print(f"  测试样本：{result['test_size']}")
    else:
        print("\n训练失败")


if __name__ == "__main__":
    main()
