#!/usr/bin/env python3
"""
大规模 LightGBM 模型训练
使用中证 500 成分股 3 年历史数据训练通用模型
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pickle

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("警告：akshare 未安装，请运行：pip install akshare")

from strategy.lgb_predictor import LGBPredictor, FeatureEngineer


def fetch_zz500_constituents() -> List[Dict]:
    """
    获取中证 500 成分股列表

    Returns:
        成分股列表 [{'symbol': '000001', 'name': '平安银行'}, ...]
    """
    try:
        # 获取中证 500 成分股
        df = ak.index_stock_cons(symbol="000905")

        constituents = []
        for _, row in df.iterrows():
            constituents.append({
                'symbol': row.get('品种代码', ''),
                'name': row.get('品种名称', '')
            })

        print(f"获取到 {len(constituents)} 只中证 500 成分股")
        return constituents

    except Exception as e:
        print(f"获取中证 500 成分股失败：{e}")
        return []


def fetch_stock_history_akshare(symbol: str, years: int = 3, retry: int = 3) -> Optional[pd.DataFrame]:
    """
    获取股票历史数据（使用 akshare）

    Args:
        symbol: 股票代码
        years: 获取年数
        retry: 重试次数

    Returns:
        DataFrame 包含历史 OHLCV 数据
    """
    if not AKSHARE_AVAILABLE:
        return None

    import time

    for attempt in range(retry):
        try:
            # 计算开始日期
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)

            start_str = start_date.strftime('%Y%m%d')

            # 获取历史数据（前复权）
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period='daily',
                start_date=start_str,
                adjust='qfq'
            )

            if df.empty:
                return None

            # 重命名列以匹配现有格式
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount'
            })

            df['date'] = pd.to_datetime(df['date'])
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            df = df.sort_values('date')
            df = df.reset_index(drop=True)

            return df

        except Exception as e:
            if attempt < retry - 1:
                time.sleep(1)  # 重试前等待
            else:
                return None

    return None


def collect_training_data(
    constituents: List[Dict],
    years: int = 3,
    min_samples: int = 100,
    cache_dir: str = None
) -> pd.DataFrame:
    """
    批量收集成分股历史数据

    Args:
        constituents: 成分股列表
        years: 获取年数
        min_samples: 最少样本数
        cache_dir: 缓存目录

    Returns:
        合并的 DataFrame
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), '../data/zz500_cache')

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    import time

    all_data = []
    successful_count = 0
    failed_count = 0

    for i, stock in enumerate(constituents):
        symbol = stock['symbol']
        name = stock.get('name', '')

        # 进度显示
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[{i + 1}/{len(constituents)}] 获取 {name} ({symbol})...")

        # 检查缓存
        cache_file = os.path.join(cache_dir, f"{symbol}.pkl")
        df = None

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    df = pickle.load(f)
                if len(df) >= min_samples:
                    print(f"  ✓ 从缓存加载：{len(df)} 条")
                    all_data.append(df)
                    successful_count += 1
                    continue
                else:
                    os.remove(cache_file)  # 删除无效缓存
            except:
                pass

        # 获取新数据
        df = fetch_stock_history_akshare(symbol, years=years)

        if df is not None and len(df) >= min_samples:
            # 保存到缓存
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
            print(f"  ✓ 获取成功：{len(df)} 条")
            all_data.append(df)
            successful_count += 1
        else:
            failed_count += 1

        # 每 10 个显示一次进度
        if (i + 1) % 10 == 0:
            print(f"  进度：{successful_count} 成功，{failed_count} 失败")
            time.sleep(0.5)  # 添加延时避免被限

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"\n总计：{len(combined)} 条数据，成功 {successful_count}/{len(constituents)} 只股票")
        return combined
    else:
        return pd.DataFrame()


def train_enhanced_model(df: pd.DataFrame) -> Dict:
    """
    使用大规模数据训练增强模型

    Args:
        df: 合并的训练数据

    Returns:
        训练结果
    """
    print("\n" + "=" * 60)
    print("开始训练增强版 LightGBM 模型")
    print("=" * 60)

    predictor = LGBPredictor(model_dir='./models/lgb_enhanced')

    # 准备数据
    X, y = predictor.prepare_data(df)

    # 过滤无效数据
    non_zero_mask = y != 0
    X = X[non_zero_mask]
    y = y[non_zero_mask]

    print(f"\n有效样本数：{len(X)}")

    if len(X) < 500:
        print(f"数据量不足，无法训练")
        return {'status': 'insufficient_data'}

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
        return {'status': 'insufficient_classes'}

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
        num_leaves=63,  # 更大的树
        learning_rate=0.03,  # 更小的学习率
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        n_estimators=200,  # 更多迭代次数
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
    model_path = os.path.join(predictor.model_dir, 'zz500_enhanced.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n模型已保存到：{model_path}")

    feature_path = os.path.join(predictor.model_dir, 'zz500_features.pkl')
    with open(feature_path, 'wb') as f:
        pickle.dump(feature_names, f)

    return {
        'status': 'success',
        'accuracy': accuracy,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_classes': n_classes,
        'feature_importance': importance
    }


def main():
    """主函数"""
    print("=" * 60)
    print("大规模 LightGBM 模型训练 - 中证 500 成分股")
    print("=" * 60)

    if not AKSHARE_AVAILABLE:
        print("错误：akshare 未安装")
        return

    # 1. 获取成分股列表
    constituents = fetch_zz500_constituents()
    if not constituents:
        print("无法获取成分股列表")
        return

    # 2. 收集历史数据（全部 500 只）
    print("\n收集历史数据（预计需要 5-10 分钟）...")
    combined_data = collect_training_data(constituents, years=3)

    if combined_data.empty:
        print("未能收集到足够的数据")
        return

    # 3. 训练模型
    result = train_enhanced_model(combined_data)

    if result['status'] == 'success':
        print("\n训练完成!")
        # 输出模型使用指南
        print("\n" + "=" * 60)
        print("模型使用指南")
        print("=" * 60)
        print("要将增强模型集成到交易策略中，请修改:")
        print("  intraday_strategy.py 中的 LGBPredictor 初始化")
        print("  将 model_dir 改为 './models/lgb_enhanced'")
        print("=" * 60)
    else:
        print(f"\n训练失败：{result['status']}")


if __name__ == "__main__":
    main()
