#!/usr/bin/env python3
"""
v8 回归模型训练 — 预测未来收益率（非涨跌方向）

核心改进:
1. 分类 → 回归: 预测未来3根K线的收益率, 不是涨跌方向
2. 跨截面排序: 对所有股票打分, 买Top N, 这才是赚钱逻辑
3. Purged K-Fold: gap=3, 防止数据泄漏
4. 纯30分钟特征: 不用日级别数据
5. 特征选择: 相关度过滤 + SelectFromModel
6. 模型版本管理: 保存 v8 带版本号

业界对标: WorldQuant/Citadel 的截面排序(strategy) + 择时(execution) 双层架构
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
import lightgbm as lgb
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 复用 v7 的特征工程
from strategy.train_lgb_enhanced import (
    EnhancedFeatureEngineer,
    MarketFeatureEngineer,
    load_data,
    load_data_from_db,
)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/stock_data.db')


def prepare_training_data_regression(
    all_data: Dict[str, pd.DataFrame],
    horizon: int = 3
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    准备回归训练数据
    
    Args:
        all_data: 股票数据字典 {symbol: DataFrame}
        horizon: 预测周期（3根K线 = 90分钟）
    
    Returns:
        X: 特征矩阵
        y: 目标收益率（连续值）
        feature_names: 特征名列表
        symbols: 每条样本对应的股票代码（用于后续排序）
    """
    all_features = []
    all_targets = []
    all_symbols = []
    success_count = 0
    fail_count = 0
    feature_names = None

    print("计算特征(回归目标, 纯30分钟级别)...")

    for i, (symbol, df) in enumerate(all_data.items()):
        try:
            # 只用纯技术特征
            features = EnhancedFeatureEngineer.calculate_features(df)

            if feature_names is None and len(features.columns) > 0:
                feature_names = features.columns.tolist()
                print(f"  特征数: {len(feature_names)}")

            # 回归目标: 未来3根K线的收益率
            close = df['close'].values
            target = np.zeros(len(close))
            for j in range(len(close) - horizon):
                target[j] = (close[j + horizon] - close[j]) / close[j]
            # 未来有NaN(最后horizon根K线)
            target[-horizon:] = np.nan

            # NaN处理
            features = features.fillna(method='ffill').fillna(0)
            valid_mask = ~np.isnan(target)
            features_valid = features[valid_mask]
            target_valid = target[valid_mask]

            # 过滤前120行（特征不完整）
            if len(features_valid) > 120:
                features_valid = features_valid.iloc[120:]
                target_valid = target_valid[120:]

            if len(features_valid) > 50:
                all_features.append(features_valid.values)
                all_targets.append(target_valid)
                all_symbols.extend([symbol] * len(features_valid))
                success_count += 1
            else:
                fail_count += 1

        except Exception as e:
            fail_count += 1

        if (i + 1) % 50 == 0:
            print(f"  已处理 {i + 1}/{len(all_data)} 只股票 (成功{success_count}, 失败{fail_count})")

    if not all_features:
        return None, None, None, None

    X = np.vstack(all_features)
    y = np.concatenate(all_targets)

    # 去除极端值（>10% 或 <-10%的收益率，可能是数据错误）
    valid = (np.abs(y) < 0.1)
    X = X[valid]
    y = y[valid]
    all_symbols = [s for s, v in zip(all_symbols, valid) if v]

    print(f"\n数据准备完成: 成功{success_count}只, 失败{fail_count}只")
    print(f"总样本数: {len(X)} (去除极端值后)")
    print(f"目标分布: mean={y.mean():.4f}, std={y.std():.4f}, "
          f"min={y.min():.4f}, max={y.max():.4f}")
    print(f"特征数: {len(feature_names) if feature_names else 'N/A'}")

    return X, y, feature_names, all_symbols


def train_model_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str] = None
) -> Dict:
    """
    训练回归模型 (v8)

    评估指标:
    - RMSE: 预测误差
    - MAE: 平均绝对误差
    - Spearman: 排序相关性（最重要！决定能否选出好股票）
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tscv = TimeSeriesSplit(n_splits=5, gap=3)

    # ====== Optuna 搜索超参数 ======
    print("\n🔍 Optuna超参数搜索 (50轮, 回归目标=RMSE)")
    print("  搜索: num_leaves, max_depth, learning_rate, feature_fraction,")
    print("        bagging_fraction, bagging_freq, min_child_samples,")
    print("        reg_alpha, reg_lambda, min_gain_to_split")

    split_idx = int(len(X) * 0.8)
    X_search, y_search = X[:split_idx], y[:split_idx]

    def objective(trial):
        params = {
            'objective': 'regression_l1',  # MAE loss, 对异常值更鲁棒
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'n_jobs': -1,
            'random_state': 42,
            'n_estimators': 2000,

            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 5.0),
        }

        mini_tscv = TimeSeriesSplit(n_splits=3, gap=3)
        scores = []
        for train_idx, test_idx in mini_tscv.split(X_search):
            X_tr, X_te = X_search[train_idx], X_search[test_idx]
            y_tr, y_te = y_search[train_idx], y_search[test_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_te, y_te)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            y_pred = model.predict(X_te)
            # 用 Spearman 排序相关性（目标函数）
            if len(y_pred) > 2:
                corr, _ = spearmanr(y_pred, y_te)
                if not np.isnan(corr):
                    scores.append(corr)
                else:
                    scores.append(0)
            else:
                scores.append(0)

        return np.mean(scores) if scores else 0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    best_params = study.best_params
    best_value = study.best_value

    print(f"\n✅ Optuna搜索完成!")
    print(f"  最优Spearman: {best_value:.4f}")
    print(f"  最优参数:")
    for k, v in best_params.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.6f}")
        else:
            print(f"    {k}: {v}")

    # ====== 用最优参数做5折Purged交叉验证 ======
    best_lgbm_params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42,
        'n_estimators': 2000,
    }
    best_lgbm_params.update(best_params)

    print(f"\n用最优参数做5折Purged交叉验证...")

    cv_scores = []
    fold_rmse = []
    fold_mae = []
    fold_spearman = []
    models = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgb.LGBMRegressor(**best_lgbm_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        corr, _ = spearmanr(y_pred, y_test)
        if np.isnan(corr):
            corr = 0

        fold_rmse.append(rmse)
        fold_mae.append(mae)
        fold_spearman.append(corr)
        models.append(model)

        print(f"  Fold {fold + 1}: RMSE={rmse:.4f}, MAE={mae:.4f}, Spearman={corr:.4f}")

    avg_rmse = np.mean(fold_rmse)
    avg_mae = np.mean(fold_mae)
    avg_spearman = np.mean(fold_spearman)

    print(f"\n平均: RMSE={avg_rmse:.4f}, MAE={avg_mae:.4f}, Spearman={avg_spearman:.4f}")

    final_model = models[-1]

    # ====== 特征去冗余 ======
    if feature_names is not None and len(feature_names) > 20:
        valid_start = int(len(X) * 0.8)
        X_train_part = X[:valid_start]
        corr_matrix = np.corrcoef(X_train_part.T)
        high_corr_pairs = []
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                if abs(corr_matrix[i, j]) > 0.95:
                    high_corr_pairs.append((i, j, feature_names[i], feature_names[j], abs(corr_matrix[i, j])))

        if high_corr_pairs:
            to_remove = set()
            for i, j, fn_i, fn_j, corr in high_corr_pairs:
                if i in to_remove or j in to_remove:
                    continue
                var_i = np.var(X_train_part[:, i])
                var_j = np.var(X_train_part[:, j])
                to_remove.add(i if var_i < var_j else j)

            if to_remove:
                keep_mask = np.ones(len(feature_names), dtype=bool)
                keep_mask[list(to_remove)] = False
                removed_corr = [feature_names[i] for i in sorted(to_remove)]

                print(f"\n特征去冗余 (corr > 0.95):")
                print(f"  删除: {', '.join(removed_corr[:8])}"
                      + (f' ...等{len(removed_corr)}个' if len(removed_corr) > 8 else ''))
                print(f"  保留: {sum(keep_mask)}/{len(feature_names)} 个特征")

                X = X[:, keep_mask]
                feature_names = [fn for fn, m in zip(feature_names, keep_mask) if m]

    # ====== 特征选择 ======
    if feature_names is not None and len(feature_names) > 0:
        valid_start = int(len(X) * 0.8)
        X_full_train = X[:valid_start]
        y_full_train = y[:valid_start]

        selector_model = lgb.LGBMRegressor(**best_lgbm_params)
        selector_model.fit(X_full_train[:-50], y_full_train[:-50],
                          eval_set=[(X_full_train[-50:], y_full_train[-50:])],
                          callbacks=[lgb.early_stopping(30, verbose=False)])

        selector = SelectFromModel(selector_model, threshold='median', prefit=True)
        X_selected = selector.transform(X)
        selected_mask = selector.get_support()

        original_count = len(feature_names)
        selected_feature_names = [fn for fn, mask in zip(feature_names, selected_mask) if mask]
        removed_features = [fn for fn, mask in zip(feature_names, selected_mask) if not mask]

        print(f"\n特征选择 (threshold=median):")
        print(f"  保留: {len(selected_feature_names)}/{original_count} 个特征")
        print(f"  删除: {', '.join(removed_features[:10])}"
              + (f' ...等{len(removed_features)}个' if len(removed_features) > 10 else ''))

        feature_names = selected_feature_names

        # 用精选特征重新训练最终模型
        final_model = lgb.LGBMRegressor(**best_lgbm_params)
        final_model.fit(X_selected[:valid_start], y[:valid_start],
                       eval_set=[(X_selected[valid_start:], y[valid_start:])],
                       callbacks=[lgb.early_stopping(50, verbose=False)])

        # 最终评估
        y_pred_final = final_model.predict(X_selected[valid_start:])
        final_rmse = np.sqrt(mean_squared_error(y[valid_start:], y_pred_final))
        final_mae = mean_absolute_error(y[valid_start:], y_pred_final)
        final_spearman, _ = spearmanr(y_pred_final, y[valid_start:])
        if np.isnan(final_spearman):
            final_spearman = 0

        print(f"\n最终评估 (后20%验证集):")
        print(f"  RMSE={final_rmse:.4f}, MAE={final_mae:.4f}, Spearman={final_spearman:.4f}")

    # 特征重要性
    if feature_names and len(feature_names) == len(final_model.feature_importances_):
        feature_importance = dict(zip(feature_names, final_model.feature_importances_))
    else:
        feature_importance = {}

    return {
        'model': final_model,
        'model_type': 'regression',
        'cv_rmse': avg_rmse,
        'cv_mae': avg_mae,
        'cv_spearman': avg_spearman,
        'final_rmse': final_rmse if 'final_rmse' in dir() else avg_rmse,
        'final_mae': final_mae if 'final_mae' in dir() else avg_mae,
        'final_spearman': final_spearman if 'final_spearman' in dir() else avg_spearman,
        'feature_importance': feature_importance,
        'feature_names': feature_names,
        'selected_features': feature_names,
        'params': best_lgbm_params,
        'train_samples': len(X),
        'train_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_version': 'v8',
        'description': 'v8: 回归模型(预测收益率)+截面排序, 纯30分钟特征, Purged K-Fold',
        'horizon': 3,
    }


def save_model(model_data: Dict, model_dir: str):
    """保存模型"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    version = model_data.get('model_version', 'unknown')

    # 主模型文件
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    # 版本化备份
    backup_path = os.path.join(model_dir, f'model_{version}.pkl')
    with open(backup_path, 'wb') as f:
        pickle.dump(model_data, f)

    # 元信息
    meta_path = os.path.join(model_dir, f'model_{version}_meta.json')
    meta = {
        'version': version,
        'model_type': model_data.get('model_type', 'regression'),
        'train_time': model_data.get('train_time', ''),
        'cv_rmse': float(model_data.get('cv_rmse', 0)),
        'cv_mae': float(model_data.get('cv_mae', 0)),
        'cv_spearman': float(model_data.get('cv_spearman', 0)),
        'final_spearman': float(model_data.get('final_spearman', 0)),
        'feature_count': len(model_data.get('feature_names', [])),
        'train_samples': model_data.get('train_samples', 0),
        'description': model_data.get('description', ''),
        'horizon': model_data.get('horizon', 3),
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n模型已保存: {model_path}")
    print(f"  备份: {backup_path}")
    print(f"  元信息: {meta_path}")


def main():
    print("=" * 60)
    print("v8 回归模型训练 — 预测未来收益率")
    print("=" * 60)
    print(f"数据源: SQLite 数据库")
    print(f"特征: 纯30分钟级别（无日级别泄漏）")
    print(f"预测目标: 未来3根K线（90分钟）的收益率")
    print(f"评估指标: Spearman排序相关性（决定能否选出好股票）")
    print("=" * 60)

    db_path = os.path.join(os.path.dirname(__file__), '../data/stock_data.db')
    all_data = load_data_from_db(db_path)

    if not all_data:
        print("未加载到任何数据")
        return

    print(f"\n加载了 {len(all_data)} 只股票，开始训练...")

    X, y, feature_names, symbols = prepare_training_data_regression(all_data, horizon=3)

    if X is None or len(X) < 500:
        print(f"训练数据不足 ({len(X) if X is not None else 0} 条)")
        return

    model_data = train_model_regression(X, y, feature_names=feature_names)

    # 显示特征重要性
    print("\n特征重要性 Top 20:")
    importance = sorted(model_data['feature_importance'].items(), key=lambda x: x[1], reverse=True)
    for name, score in importance[:20]:
        print(f"  {name}: {score}")

    # 保存模型
    model_dir = os.path.join(os.path.dirname(__file__), '../models/lgb_hs300')
    save_model(model_data, model_dir)

    print("\n训练完成!")
    print(f"Spearman排序相关性: {model_data['final_spearman']:.4f}")
    print(f"→ 这个值决定了截面排序策略能否选出好股票")
    print(f"→ 业界标准: 0.05-0.15 即可盈利, 0.15+ 优秀")


if __name__ == '__main__':
    main()