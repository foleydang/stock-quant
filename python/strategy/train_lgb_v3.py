#!/usr/bin/env python3
"""
LGBM 模型 v3 - 全面优化版 (Mac本地训练)

优化点:
1. 新增特征: 成交量剖面、价格动量加速度、市场广度指标
2. Optuna超参数搜索 (利用24G内存)
3. Bagging集成 (5个LGBM子模型投票)
4. 自适应阈值 (按波动率动态调整涨跌阈值)
5. 滚动验证评估
6. 特征选择 (去除低重要性特征)

输出: 单个pickle文件，推理内存<10MB，2G服务器可运行
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import pickle
import sqlite3
import lightgbm as lgb
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

# 尝试导入optuna（如果没有就跳过超参搜索）
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("⚠ optuna未安装，跳过超参搜索。安装: pip install optuna")

from strategy.train_lgb_enhanced import EnhancedFeatureEngineer

# ============ 配置 ============
DB_PATH = os.path.join(os.path.dirname(__file__), '../data/stock_data.db')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models/lgb_hs300')
HORIZON = 3               # 预测3根K线后(90分钟)
BASE_THRESHOLD = 0.010     # 阈值1.0% (从1.8%降低,保留更多有效样本)
N_BAGGING = 3              # 3个子模型(验证最优,回测16.88%)
TIME_FEATURES = ['day_of_week', 'day_of_month', 'hour', 'minute',
                  'is_morning', 'is_afternoon', 'is_first_hour', 'is_last_hour']
# 零重要性特征 - 首轮训练确认无用，直接剔除
ZERO_IMP_FEATURES = [
    'price_above_ma5', 'price_above_ma10', 'price_above_ma20',
    'price_above_ma30', 'price_above_ma60', 'price_above_ma80',
    'price_above_ma100', 'price_above_ma120',
    'ma5_cross_ma10', 'ma10_cross_ma20', 'ma20_cross_ma60', 'ma60_cross_ma120',
    'macd_cross', 'kdj_cross_signal', 'inside_bar', 'breakout_20', 'trend_direction',
]


# ============ 新增特征 ============
class AdvancedFeatureEngineer:
    """高级特征工程 - 在EnhancedFeatureEngineer基础上增加"""

    @staticmethod
    def calculate_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
        """计算高级特征"""
        adv = pd.DataFrame(index=df.index)
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float)
        open_price = df['open'].values.astype(float)

        # === 1. 价格动量加速度 (5个) ===
        for period in [3, 5, 10, 20]:
            ret = pd.Series(close).pct_change(period)
            adv[f'momentum_accel_{period}'] = ret.diff(3)  # 动量加速度

        # 动量衰减 (近期动量 vs 远期动量差异)
        mom_short = pd.Series(close).pct_change(5)
        mom_long = pd.Series(close).pct_change(20)
        adv['momentum_decay'] = mom_short - mom_long

        # === 2. 成交量剖面 (6个) ===
        vol = pd.Series(volume)
        for period in [5, 10, 20]:
            vol_ma = vol.rolling(period).mean()
            adv[f'vol_ratio_{period}'] = volume / vol_ma.values
            adv[f'vol_std_{period}'] = vol.rolling(period).std() / vol_ma.values

        # 成交量趋势 (成交量是否在增加)
        adv['vol_trend'] = vol.rolling(5).mean() / vol.rolling(20).mean().values

        # 成交量价格背离
        price_up = (pd.Series(close).diff() > 0).astype(int)
        vol_up = (vol.diff() > 0).astype(int)
        adv['vol_price_divergence'] = (price_up != vol_up).rolling(5).mean()

        # === 3. 价格形态 (4个) ===
        # 内包/外包K线
        prev_high = pd.Series(high).shift(1)
        prev_low = pd.Series(low).shift(1)
        adv['inside_bar'] = ((high <= prev_high) & (low >= prev_low)).astype(int)
        adv['outside_bar'] = ((high > prev_high) & (low < prev_low)).astype(int)

        # K线实体比例 (红/绿K线)
        body = close - open_price
        total_range = high - low + 1e-10
        adv['body_ratio'] = body / total_range
        adv['adv_upper_shadow'] = (high - np.maximum(close, open_price)) / total_range

        # === 4. 波动率聚类 (3个) ===
        # 波动率变化率
        returns = pd.Series(close).pct_change()
        vol20 = returns.rolling(20).std()
        vol5 = returns.rolling(5).std()
        adv['vol_ratio_5_20'] = vol5 / vol20.values

        # 波动率均值回归信号
        vol_ma = vol20.rolling(60).mean()
        adv['vol_vs_mean'] = vol20 / vol_ma.values

        # 高低波动率状态 (1=高波动, 0=低波动)
        vol_median = vol20.rolling(120).median()
        adv['high_vol_state'] = (vol20 > vol_median).astype(int)

        # === 5. 支撑/压力相对位置 (3个) ===
        for period in [20, 60]:
            period_high = pd.Series(high).rolling(period).max()
            period_low = pd.Series(low).rolling(period).min()
            period_range = period_high - period_low + 1e-10
            adv[f'adv_price_position_{period}'] = (close - period_low) / period_range.values

        # 突破信号 (价格突破20日高点)
        adv['breakout_20'] = (close > pd.Series(high).rolling(20).max().shift(1)).astype(int)

        # === 6. 连续涨跌 (2个) ===
        up_days = pd.Series(close).diff()
        adv['consecutive_up'] = (up_days > 0).rolling(5).sum()
        adv['consecutive_down'] = (up_days < 0).rolling(5).sum()

        return adv


# ============ 目标计算 ============
def calculate_target_adaptive(df: pd.DataFrame, horizon: int = 3, base_threshold: float = 0.012) -> np.ndarray:
    """
    自适应阈值目标计算
    - 高波动股票: threshold放大 (阈值更高, 过滤更多噪声)
    - 低波动股票: threshold缩小 (捕捉更多趋势)
    """
    close = df['close'].values
    returns = pd.Series(close).pct_change()
    recent_vol = returns.rolling(60).std().values  # 近期波动率

    # 计算全局波动率中位数作为基准
    vol_median = np.nanmedian(recent_vol)

    target = np.zeros(len(close))

    for i in range(len(close) - horizon):
        if np.isnan(recent_vol[i]) or np.isnan(vol_median):
            # 无波动率数据时用基础阈值
            threshold = base_threshold
        else:
            # 自适应: 高波动时放大阈值, 低波动时缩小
            vol_ratio = recent_vol[i] / (vol_median + 1e-10)
            threshold = base_threshold * max(0.8, min(2.0, vol_ratio))

        ret = (close[i + horizon] - close[i]) / close[i]
        if ret > threshold:
            target[i] = 1   # 明确上涨
        elif ret < -threshold:
            target[i] = 0   # 明确下跌
        else:
            target[i] = -1  # 震荡, 不参与训练

    return target


# ============ 数据加载 ============
def load_all_data(db_path: str) -> Dict[str, pd.DataFrame]:
    """从数据库加载所有股票数据"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM kline_30m")
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
        except Exception:
            pass

    print(f"加载了 {len(all_data)} 只股票")
    return all_data


# ============ 特征计算 ============
def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算基础 + 高级特征, 过滤时间特征和零重要性特征"""
    base_features = EnhancedFeatureEngineer.calculate_features(df)
    adv_features = AdvancedFeatureEngineer.calculate_advanced_features(df)

    # 合并
    all_features = pd.concat([base_features, adv_features], axis=1)

    # 过滤时间特征 + 零重要性特征
    drop_cols = TIME_FEATURES + ZERO_IMP_FEATURES
    keep_cols = [c for c in all_features.columns if c not in drop_cols]
    all_features = all_features[keep_cols]

    return all_features


# ============ 数据准备 ============
def prepare_data(all_data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """准备训练数据"""
    # 先算特征名
    sample_df = list(all_data.values())[0]
    sample_features = compute_all_features(sample_df)
    feature_names = list(sample_features.columns)
    print(f"总特征数: {len(feature_names)} (基础110 + 高级23 - 时间4)")

    all_X = []
    all_y = []
    min_history = 150  # 需要更多历史(高级特征需要更长窗口)

    for i, (symbol, df) in enumerate(all_data.items()):
        try:
            features = compute_all_features(df)
            target = calculate_target_adaptive(df, horizon=HORIZON, base_threshold=BASE_THRESHOLD)

            # 确保 valid_mask 是 numpy array 以正确索引
            mask_features = features.isna().any(axis=1).values  # ndarray
            mask_target = (target >= 0)  # ndarray
            valid_mask = ~mask_features & mask_target  # ndarray
            features_valid = features.iloc[valid_mask].iloc[min_history:]  # 先用 iloc 选择有效行
            target_valid = target[valid_mask][min_history:]  # ndarray 直接切片
            features_valid = features_valid.fillna(0)

            if len(features_valid) > 30:
                all_X.append(features_valid.values)
                all_y.append(target_valid)  # target_valid 已经是 ndarray
        except Exception as e:
            if i < 5:
                print(f"  {symbol}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  已处理 {i + 1}/{len(all_data)} 只股票")

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    print(f"总样本: {len(X)}")
    print(f"  上涨: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
    print(f"  下跌: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")

    return X, y, feature_names


# ============ Optuna超参搜索 ============
def optimize_hyperparams(X: np.ndarray, y: np.ndarray, n_trials: int = 50) -> Dict:
    """Optuna超参数搜索"""
    if not HAS_OPTUNA:
        print("跳过Optuna搜索，使用默认参数")
        return {
            'num_leaves': 31,
            'max_depth': 6,
            'learning_rate': 0.02,
            'n_estimators': 500,
            'min_child_samples': 50,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'reg_alpha': 0.3,
            'reg_lambda': 0.3,
        }

    print(f"\n=== Optuna超参搜索 ({n_trials}次试验) ===")

    tscv = TimeSeriesSplit(n_splits=3)  # 搜索用3折加速

    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
            'n_estimators': 500,
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
            'bagging_freq': 5,
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1,
        }

        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      callbacks=[lgb.early_stopping(30, verbose=False)])

            y_pred = model.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best['n_estimators'] = 500  # 最终训练时会用早停决定
    best['objective'] = 'binary'
    best['metric'] = 'binary_logloss'
    best['boosting_type'] = 'gbdt'
    best['bagging_freq'] = 5
    best['verbose'] = -1
    best['random_state'] = 42
    best['n_jobs'] = -1

    print(f"\n最优参数:")
    for k, v in best.items():
        print(f"  {k}: {v}")
    print(f"最优CV准确率: {study.best_value:.4f}")

    return best


# ============ Bagging集成训练 ============
def train_ensemble(X: np.ndarray, y: np.ndarray, params: Dict, feature_names: List[str], n_models: int = 5) -> Dict:
    """训练Bagging集成 (n个子模型投票)"""
    print(f"\n=== 训练Bagging集成 ({n_models}个子模型) ===")

    tscv = TimeSeriesSplit(n_splits=5)
    models = []
    best_iterations = []

    # 每个子模型用不同的数据子集(行采样) + 不同的特征子集(列采样)
    for m_idx in range(n_models):
        print(f"\n子模型 {m_idx + 1}/{n_models}:")

        # 微调参数：每个子模型稍有不同
        model_params = params.copy()
        model_params['random_state'] = 42 + m_idx * 7
        # 不同的列采样率
        model_params['feature_fraction'] = min(0.9, params['feature_fraction'] + (m_idx % 3) * 0.05)

        # 交叉验证找最优迭代
        cv_scores = []
        fold_best_iters = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = lgb.LGBMClassifier(**model_params)
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)])

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cv_scores.append(acc)
            fold_best_iters.append(model.best_iteration_)
            print(f"  Fold {fold + 1}: Acc={acc:.4f}, BestIter={model.best_iteration_}")

        avg_acc = np.mean(cv_scores)
        avg_iter = int(np.mean(fold_best_iters))
        print(f"  平均: Acc={avg_acc:.4f}, BestIter={avg_iter}")

        # 用最优迭代训练最终子模型
        final_params = model_params.copy()
        final_params['n_estimators'] = avg_iter
        final_model = lgb.LGBMClassifier(**final_params)
        final_model.fit(X, y)

        models.append(final_model)
        best_iterations.append(avg_iter)

    # 集成评估
    print(f"\n=== 集成评估 ===")
    # 投票预测
    all_preds = []
    for model in models:
        all_preds.append(model.predict(X))

    # 多数投票
    vote_preds = np.array(all_preds).T  # (n_samples, n_models)
    ensemble_pred = np.apply_along_axis(
        lambda x: Counter(x).most_common(1)[0][0], axis=1, arr=vote_preds
    )

    ensemble_acc = accuracy_score(y, ensemble_pred)
    print(f"  单模型准确率范围: {min(accuracy_score(y, p) for p in all_preds):.2%} ~ {max(accuracy_score(y, p) for p in all_preds):.2%}")
    print(f"  集成投票准确率: {ensemble_acc:.2%}")
    print(classification_report(y, ensemble_pred, target_names=['下跌', '上涨']))

    # 特征重要性 (平均)
    avg_importance = np.mean([m.feature_importances_ for m in models], axis=0)
    top_indices = np.argsort(avg_importance)[::-1][:20]
    print(f"\nTop 20 重要特征 (集成平均):")
    for i in top_indices:
        print(f"  {feature_names[i]}: {avg_importance[i]:.0f}")

    zero_count = sum(1 for x in avg_importance if x < 1)
    print(f"\n低重要性特征(<1): {zero_count}/{len(feature_names)}")

    # 特征选择: 建议移除低重要性特征
    keep_features = [feature_names[i] for i in range(len(feature_names)) if avg_importance[i] >= 1]
    print(f"建议保留特征: {len(keep_features)}/{len(feature_names)}")

    return {
        'models': models,
        'ensemble_accuracy': ensemble_acc,
        'best_iterations': best_iterations,
        'feature_names': feature_names,
        'keep_features': keep_features,
        'avg_importance': avg_importance,
        'params': params,
        'n_models': n_models,
        'horizon': HORIZON,
        'threshold': BASE_THRESHOLD,
        'train_samples': len(X),
        'train_date': datetime.now().strftime('%Y-%m-%d'),
    }


# ============ 模型保存 ============
def save_model(model_data: Dict, model_dir: str):
    """保存模型 (单个文件，推理轻量)"""
    os.makedirs(model_dir, exist_ok=True)

    # 保存为单个pickle (推理时只加载这一个文件)
    with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model_data, f)

    # 保存metadata
    metadata = {
        "model_name": "lgb_hs300_v3",
        "version": "3.0",
        "train_date": model_data['train_date'],
        "architecture": "Bagging ensemble (5 LGBM models, majority vote)",
        "inference_memory_mb": "~15 (5个LGBM模型，2G服务器可运行)",
        "improvements": [
            "新增23个高级特征(动量加速度/成交量剖面/价格形态/波动率聚类/支撑压力)",
            "Optuna超参数搜索",
            "Bagging集成(5个子模型投票)",
            "自适应阈值(按波动率动态调整)",
            "滚动验证评估",
        ],
        "target_definition": {
            "horizon": f"{HORIZON}根K线 ({HORIZON*30}分钟)",
            "threshold": f"自适应, 基础{BASE_THRESHOLD*100}%",
        },
        "performance": {
            "ensemble_accuracy": round(model_data['ensemble_accuracy'], 4),
            "n_models": model_data['n_models'],
            "best_iterations": model_data['best_iterations'],
            "total_features": len(model_data['feature_names']),
            "recommended_features": len(model_data['keep_features']),
            "train_samples": model_data['train_samples'],
        },
        "hyperparameters": model_data['params'],
    }

    with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
        import json
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 模型已保存到 {model_dir}")
    print(f"   model.pkl 大小: {os.path.getsize(os.path.join(model_dir, 'model.pkl')) / 1024 / 1024:.1f} MB")
    print(f"   推理内存预估: ~15MB (2G服务器完全可运行)")


# ============ 主流程 ============
def main():
    db_path = DB_PATH
    if not os.path.exists(db_path):
        # 尝试相对路径
        db_path = os.path.join(os.path.dirname(__file__), 'data/stock_data.db')

    print("=" * 60)
    print("  LGBM v3 - 全面优化版 (Mac本地训练)")
    print("  特征: 基础110 + 高级23 = 133")
    print("  集成: 5个LGBM投票")
    print("  推理: ~15MB, 2G服务器可运行")
    print("=" * 60)

    # 1. 加载数据
    all_data = load_all_data(db_path)
    if not all_data:
        print("❌ 未加载到数据")
        return

    # 2. 准备特征
    X, y, feature_names = prepare_data(all_data)
    if len(X) < 500:
        print(f"❌ 数据不足: {len(X)} 条")
        return

    # 3. Optuna超参搜索
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='快速模式(10次搜索)')
    parser.add_argument('--trials', type=int, default=200, help='Optuna搜索次数(默认200)')
    args = parser.parse_args()
    n_trials = 10 if args.quick else args.trials
    print(f"Optuna搜索: {n_trials}次 {'(快速)' if args.quick else '(全量)'}")
    best_params = optimize_hyperparams(X, y, n_trials=n_trials)

    # 4. Bagging集成训练
    model_data = train_ensemble(X, y, best_params, feature_names, n_models=N_BAGGING)

    # 5. 保存
    save_model(model_data, MODEL_DIR)

    print("\n🎉 训练完成！把 model.pkl 丢给小土豆部署即可。")


if __name__ == '__main__':
    main()