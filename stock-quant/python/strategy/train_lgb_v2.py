#!/usr/bin/env python3
"""
LGBM 模型优化重训练 v2
优化点:
1. 移除时间特征（day_of_week等）避免统计偏差
2. 提高涨跌阈值到1.5%减少噪声标签
3. 加入早停防止过拟合
4. 使用最新数据训练
5. 滚动验证评估真实表现
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

# 导入特征工程
from strategy.train_lgb_enhanced import EnhancedFeatureEngineer


# === 优化1: 移除时间特征的特征名列表 ===
TIME_FEATURES = ['day_of_week', 'day_of_month', 'hour', 'minute', 'is_morning',
                  'is_afternoon', 'is_first_hour', 'is_last_hour']

def filter_time_features(feature_names: List[str], features_df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    """移除时间特征，避免统计偏差"""
    keep_cols = [c for c in feature_names if c not in TIME_FEATURES]
    return keep_cols, features_df[keep_cols]


# === 优化2: 提高阈值，减少噪声 ===
def calculate_target_v2(df: pd.DataFrame, horizon: int = 3, threshold: float = 0.015) -> np.ndarray:
    """
    改进版目标计算
    - threshold从0.8%提高到1.5%
    - 涨跌超过1.5%才标记，中间震荡标记为-1过滤掉
    - 这样模型只预测"有明显趋势"的K线
    """
    close = df['close'].values
    target = np.zeros(len(close))

    for i in range(len(close) - horizon):
        ret = (close[i + horizon] - close[i]) / close[i]
        if ret > threshold:
            target[i] = 1   # 明确上涨
        elif ret < -threshold:
            target[i] = 0   # 明确下跌
        else:
            target[i] = -1  # 震荡区间，不参与训练

    return target


def load_data_from_db(db_path: str) -> Dict[str, pd.DataFrame]:
    """从数据库加载所有股票30分钟数据"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM kline_30m WHERE LENGTH(symbol) > 5")
    symbols = [r[0] for r in cursor.fetchall()]
    conn.close()

    all_data = {}
    for symbol in symbols:
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(
                'SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol=? ORDER BY date',
                conn, params=(symbol,)
            )
            conn.close()
            if len(df) > 200:
                df['date'] = pd.to_datetime(df['date'])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                all_data[symbol] = df
        except Exception as e:
            pass

    return all_data


def prepare_training_data_v2(all_data: Dict[str, pd.DataFrame], horizon: int = 3, threshold: float = 0.015) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """优化版数据准备：移除时间特征、提高阈值"""
    all_features = []
    all_targets = []

    # 先获取完整特征名
    sample_df = list(all_data.values())[0]
    full_features = EnhancedFeatureEngineer.calculate_features(sample_df)
    full_names = list(full_features.columns)
    filtered_names, _ = filter_time_features(full_names, full_features)

    logger.info(f"特征数: {len(full_names)} → {len(filtered_names)} (移除{len(full_names)-len(filtered_names)}个时间特征)")

    for i, (symbol, df) in enumerate(all_data.items()):
        try:
            features = EnhancedFeatureEngineer.calculate_features(df)
            target = calculate_target_v2(df, horizon=horizon, threshold=threshold)

            # 过滤无效数据
            valid_mask = ~(features.isna().any(axis=1)) & (target >= 0)
            features_valid = features[valid_mask]
            target_valid = target[valid_mask].astype(int)

            # 过滤前120行 + 移除时间特征
            features_valid = features_valid.iloc[120:]
            target_valid = target_valid[120:]
            _, features_filtered = filter_time_features(filtered_names, features_valid)

            # 重新检查NaN
            features_filtered = features_filtered.fillna(0)

            if len(features_filtered) > 30:
                all_features.append(features_filtered.values)
                all_targets.append(target_valid)
        except Exception as e:
            logger.warning(f"  {symbol}: {e}")

        if (i + 1) % 50 == 0:
            logger.info(f"  已处理 {i + 1}/{len(all_data)} 只股票")

    if not all_features:
        return None, None, None

    X = np.vstack(all_features)
    y = np.concatenate(all_targets)

    logger.info(f"总样本: {len(X)}")
    logger.info(f"  上涨: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
    logger.info(f"  下跌: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    logger.info(f"  (原阈值0.8%时震荡样本被过滤)")

    return X, y, filtered_names


def train_model_v2(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict:
    """优化版训练：早停 + 更合理的参数"""
    tscv = TimeSeriesSplit(n_splits=5)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,          # 从63降到31，减少过拟合
        'learning_rate': 0.02,     # 从0.03降到0.02，更稳
        'n_estimators': 1000,      # 设置上限，靠早停决定
        'max_depth': 6,            # 从8降到6
        'min_child_samples': 50,   # 从30提高到50，更保守
        'feature_fraction': 0.7,   # 从0.8降到0.7
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'reg_alpha': 0.3,          # 从0.1提高到0.3，更强正则化
        'reg_lambda': 0.3,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1,
    }

    logger.info(f"\n训练 LightGBM 模型（5折交叉验证 + 早停）...")
    logger.info(f"参数: num_leaves={params['num_leaves']}, max_depth={params['max_depth']}, lr={params['learning_rate']}")

    cv_scores = []
    best_iterations = []
    models = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)]
        )

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores.append(accuracy)
        best_iterations.append(model.best_iteration_)
        models.append(model)

        logger.info(f"  Fold {fold + 1}: Accuracy = {accuracy:.4f}, Best iteration = {model.best_iteration_}")

    avg_accuracy = np.mean(cv_scores)
    avg_best_iter = int(np.mean(best_iterations))
    logger.info(f"\n平均CV准确率: {avg_accuracy:.4f}")
    logger.info(f"平均最优迭代: {avg_best_iter}")

    # 用平均最优迭代重新训练最终模型
    final_params = params.copy()
    final_params['n_estimators'] = avg_best_iter
    logger.info(f"\n用最优迭代({avg_best_iter})训练最终模型...")

    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X, y)

    # 评估
    y_pred_all = final_model.predict(X)
    logger.info(f"\n整体评估:")
    logger.info(f"  准确率: {accuracy_score(y, y_pred_all):.2%}")
    logger.info(classification_report(y, y_pred_all, target_names=['下跌', '上涨']))

    # 特征重要性分析
    importance = final_model.feature_importances_
    top_indices = np.argsort(importance)[::-1][:15]
    logger.info(f"\nTop 15 重要特征:")
    for i in top_indices:
        logger.info(f"  {feature_names[i]}: {importance[i]}")

    zero_count = sum(1 for x in importance if x == 0)
    logger.info(f"\n零重要性特征: {zero_count}/{len(importance)}")

    return {
        'model': final_model,
        'cv_accuracy': avg_accuracy,
        'cv_scores': cv_scores,
        'feature_importance': dict(zip(feature_names, importance)),
        'feature_names': feature_names,
        'params': final_params,
        'best_iteration': avg_best_iter,
        'train_samples': len(X),
    }


def save_model(model_data: Dict, model_dir: str):
    """保存模型和元数据"""
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model_data, f)

    # 更新metadata
    metadata = {
        "model_name": "lgb_hs300_v2",
        "version": "2.0",
        "train_date": datetime.now().strftime('%Y-%m-%d'),
        "train_data": {
            "symbols": "沪深300成分股",
            "period": "30分钟K线",
            "threshold": "1.5%",
            "horizon": "3根K线(90分钟)",
        },
        "improvements": [
            "移除时间特征(day_of_week等)避免统计偏差",
            "涨跌阈值从0.8%提高到1.5%减少噪声",
            "num_leaves从63降到31",
            "max_depth从8降到6",
            "正则化alpha/lambda从0.1提高到0.3",
            "加入早停(50轮)防止过拟合",
        ],
        "performance": {
            "accuracy": round(model_data['cv_accuracy'], 4),
            "cv_scores": model_data['cv_scores'],
        },
        "hyperparameters": model_data['params'],
    }

    with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
        import json
        json.dump(metadata, f, indent=2)

    logger.info(f"\n模型已保存到 {model_dir}")


def main():
    from config_loader import get_db_path
    db_path = get_db_path()

    logger.info("=== LGBM 模型优化重训练 v2 ===")
    logger.info("优化: 移除时间特征 + 提高阈值1.5% + 早停 + 更强正则化")

    # 加载数据
    all_data = load_data_from_db(db_path)
    logger.info(f"加载了 {len(all_data)} 只股票")

    if not all_data:
        logger.error("未加载到任何数据")
        return

    # 准备训练数据
    X, y, feature_names = prepare_training_data_v2(all_data, horizon=3, threshold=0.015)

    if X is None or len(X) < 500:
        logger.error(f"训练数据不足 ({len(X) if X is not None else 0} 条)")
        return

    # 训练模型
    model_data = train_model_v2(X, y, feature_names)

    # 保存模型
    model_dir = os.path.join(os.path.dirname(__file__), '../models/lgb_hs300')
    save_model(model_data, model_dir)

    logger.info("\n✅ 训练完成！请重启 stock-api 使新模型生效。")


if __name__ == '__main__':
    main()