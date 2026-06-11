#!/usr/bin/env python3
"""
v8 回归模型训练 — 内存优化版
使用 np.memmap 写入磁盘, 避免 list+vstack 双倍内存
"""

import os, sys, pickle, gc, json
import numpy as np
import pandas as pd
import sqlite3
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel
from typing import Dict, List, Tuple, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lgb_enhanced import EnhancedFeatureEngineer

DB_PATH = os.path.join(os.path.dirname(__file__), '../data/stock_data.db')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models/lgb_hs300')
TEMP_FEATURES = '/tmp/v8_features.dat'
TEMP_TARGETS = '/tmp/v8_targets.dat'


def load_data_from_db(db_path: str) -> Dict[str, pd.DataFrame]:
    """加载所有股票数据"""
    conn = sqlite3.connect(db_path)
    symbols = [r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM kline_30m ORDER BY symbol").fetchall()]
    print(f"数据库中共有 {len(symbols)} 只股票")

    all_data = {}
    for i, sym in enumerate(symbols):
        df = pd.read_sql_query(
            'SELECT date, open, high, low, close, volume FROM kline_30m '
            'WHERE symbol=? ORDER BY date', conn, params=(sym,))
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            all_data[sym] = df
        if (i + 1) % 50 == 0:
            print(f"  已加载 {i + 1}/{len(symbols)} 只股票")

    conn.close()
    print(f"成功加载 {len(all_data)} 只股票数据\n")
    return all_data


def prepare_training_data_memmap(
    all_data: Dict[str, pd.DataFrame],
    horizon: int = 3,
    start_date: str = '2025-01-01'
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]], Optional[List[str]]]:
    """
    使用 memmap 磁盘写入避免双倍内存
    
    流程:
    1. 第一遍: 统计总行数
    2. 第二遍: 写入 memmap 文件
    3. 从 memmap 加载 (mmap 模式, 不占用 RAM)
    """
    feature_path = '/tmp/v8_X.dat'
    target_path = '/tmp/v8_y.dat'
    feature_names = None
    success_count = 0
    fail_count = 0

    # ====== 第一遍: 统计总行数 ======
    print("第1遍: 统计样本数...")
    total_rows = 0
    stock_rows = {}

    for i, (symbol, df) in enumerate(all_data.items()):
        df = df[df['date'] >= pd.to_datetime(start_date)]
        if len(df) < 150:
            continue

        try:
            features = EnhancedFeatureEngineer.calculate_features(df)
            if feature_names is None:
                feature_names = list(features.columns)

            close = df['close'].values
            target = np.zeros(len(close))
            for j in range(len(close) - horizon):
                target[j] = (close[j + horizon] - close[j]) / close[j]
            target[-horizon:] = np.nan

            features = features.ffill().fillna(0)
            valid = ~np.isnan(target)
            fv = features[valid].iloc[120:]
            tv = target[valid][120:]

            if len(fv) > 50:
                stock_rows[symbol] = len(fv)
                total_rows += len(fv)
                success_count += 1
        except Exception:
            fail_count += 1
            continue

        if (i + 1) % 50 == 0:
            print(f"  扫描 {i + 1}/{len(all_data)} (已统计 {total_rows:,} 行)")

    if total_rows == 0:
        print("无数据!")
        return None, None, None, None

    n_features = len(feature_names)
    print(f"\n统计完成: {success_count}只股票, {total_rows:,}行, {n_features}特征")
    print(f"预计内存: {total_rows * n_features * 4 / 1024**2:.0f} MB (memmap, 磁盘映射)")

    # ====== 第二遍: 写入 memmap ======
    print("\n第2遍: 写入 memmap...")
    X_mmap = np.memmap(feature_path, dtype=np.float32, mode='w+', shape=(total_rows, n_features))
    y_mmap = np.memmap(target_path, dtype=np.float32, mode='w+', shape=(total_rows,))

    symbols_list = []
    offset = 0
    for i, (symbol, df) in enumerate(all_data.items()):
        if symbol not in stock_rows:
            continue

        n = stock_rows[symbol]
        try:
            df = df[df['date'] >= pd.to_datetime(start_date)]
            features = EnhancedFeatureEngineer.calculate_features(df)

            close = df['close'].values
            target = np.zeros(len(close))
            for j in range(len(close) - horizon):
                target[j] = (close[j + horizon] - close[j]) / close[j]
            target[-horizon:] = np.nan

            features = features.ffill().fillna(0)
            valid = ~np.isnan(target)
            fv = features[valid].iloc[120:]
            tv = target[valid][120:]

            X_mmap[offset:offset + n] = fv.values.astype(np.float32)
            y_mmap[offset:offset + n] = tv.astype(np.float32)
            symbols_list.extend([symbol] * n)
            offset += n
        except Exception:
            pass

        if (i + 1) % 100 == 0:
            print(f"  写入 {i + 1}/{len(all_data)} ({offset/total_rows*100:.0f}%)")

    X_mmap.flush()
    y_mmap.flush()
    del X_mmap, y_mmap
    gc.collect()

    # ====== 加载为普通 ndarray（mmap 读取模式，内存友好） ======
    print(f"\n加载数据 (mmap 模式)...")
    X = np.memmap(feature_path, dtype=np.float32, mode='r', shape=(total_rows, n_features))
    y = np.memmap(target_path, dtype=np.float32, mode='r', shape=(total_rows,))

    # 去除极端值
    print("过滤极端值...")
    valid = np.abs(y) < 0.1
    X = np.array(X[valid], dtype=np.float32)
    y = np.array(y[valid], dtype=np.float32)
    symbols_list = [s for s, v in zip(symbols_list, valid) if v]

    # 清理临时文件
    for f in [feature_path, target_path]:
        if os.path.exists(f):
            os.remove(f)

    gc.collect()

    print(f"\n数据准备完成: {len(X):,} 样本, {X.shape[1]} 特征")
    print(f"目标分布: mean={y.mean():.4f}, std={y.std():.4f}, "
          f"min={y.min():.4f}, max={y.max():.4f}")

    return X, y, feature_names, symbols_list


def train_model_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_trials: int = 30
) -> Dict:
    """训练回归模型"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_features_original = len(feature_names)

    # ====== 特征选择: 相关性过滤 + SelectFromModel ======
    print("\n特征选择...")
    
    # 1. 相关性过滤
    corr_threshold = 0.95
    corr = pd.DataFrame(X, columns=feature_names).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        high_corr = upper.index[upper[col] > corr_threshold].tolist()
        for hc in high_corr:
            # 保留方差更大的
            if X[:, feature_names.index(col)].var() > X[:, feature_names.index(hc)].var():
                to_drop.add(hc)
            else:
                to_drop.add(col)

    if to_drop:
        keep_idx = [i for i, f in enumerate(feature_names) if f not in to_drop]
        X_filtered = X[:, keep_idx]
        feature_names_filtered = [feature_names[i] for i in keep_idx]
        print(f"  相关性过滤: {n_features_original} → {len(feature_names_filtered)} "
              f"(移除 {len(to_drop)} 个高相关特征)")
    else:
        X_filtered = X
        feature_names_filtered = feature_names

    # 2. SelectFromModel
    print("  SelectFromModel...")
    selector = SelectFromModel(
        lgb.LGBMRegressor(
            n_estimators=100, objective='regression_l1',
            verbose=-1, random_state=42, n_jobs=-1
        ),
        threshold='median'
    )
    selector.fit(X_filtered, y)
    selected_mask = selector.get_support()
    X_selected = X_filtered[:, selected_mask]
    feature_names_selected = [feature_names_filtered[i] for i, m in enumerate(selected_mask) if m]
    print(f"  SelectFromModel: {len(feature_names_filtered)} → {len(feature_names_selected)} 特征")

    # ====== Optuna 搜索 ======
    print(f"\n🔍 Optuna超参数搜索 ({n_trials}轮)")
    tscv = TimeSeriesSplit(n_splits=5, gap=3)

    split_idx = int(len(X_selected) * 0.8)
    X_search, y_search = X_selected[:split_idx], y[:split_idx]

    def objective(trial):
        params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'verbose': -1, 'n_jobs': -1,
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

        scores = []
        mini_tscv = TimeSeriesSplit(n_splits=3, gap=3)
        for tr_idx, te_idx in mini_tscv.split(X_search):
            X_tr, X_te = X_search[tr_idx], X_search[te_idx]
            y_tr, y_te = y_search[tr_idx], y_search[te_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr,
                     eval_set=[(X_te, y_te)],
                     callbacks=[lgb.early_stopping(50, verbose=False)])
            y_pred = model.predict(X_te)
            if len(y_pred) > 2:
                c, _ = spearmanr(y_pred, y_te)
                scores.append(c if not np.isnan(c) else 0)
            else:
                scores.append(0)
        return np.mean(scores) if scores else 0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n✅ Optuna完成! 最优Spearman: {study.best_value:.4f}")

    # ====== 5折 Purged CV ======
    best_params = {
        'objective': 'regression_l1', 'metric': 'mae',
        'boosting_type': 'gbdt', 'verbose': -1, 'n_jobs': -1,
        'random_state': 42, 'n_estimators': 2000,
    }
    best_params.update(study.best_params)

    print("5折Purged交叉验证...")
    cv_scores, cv_mae = [], []
    feature_importance = {}

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_selected)):
        X_tr, X_te = X_selected[tr_idx], X_selected[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        model = lgb.LGBMRegressor(**best_params)
        model.fit(X_tr, y_tr,
                 eval_set=[(X_te, y_te)],
                 callbacks=[lgb.early_stopping(50, verbose=False)])

        y_pred = model.predict(X_te)
        c, _ = spearmanr(y_pred, y_te)
        mae = np.mean(np.abs(y_pred - y_te))
        cv_scores.append(c if not np.isnan(c) else 0)
        cv_mae.append(mae)

        for f, imp in zip(feature_names_selected, model.feature_importances_):
            feature_importance[f] = feature_importance.get(f, 0) + imp

        print(f"  Fold {fold+1}: Spearman={c:.4f}, MAE={mae:.4f}")

    mean_spearman = np.mean(cv_scores)
    mean_mae = np.mean(cv_mae)

    print(f"\nCV结果: Spearman={mean_spearman:.4f}±{np.std(cv_scores):.4f}, "
          f"MAE={mean_mae:.4f}±{np.std(cv_mae):.4f}")

    # 最终模型
    print("训练最终模型...")
    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_selected, y)

    return {
        'model': final_model,
        'feature_names': feature_names_selected,
        'feature_importance': dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)),
        'cv_spearman': mean_spearman,
        'cv_mae': mean_mae,
        'cv_spearman_std': np.std(cv_scores),
        'best_params': study.best_params,
        'best_spearman': study.best_value,
        'model_type': 'regression',
        'model_version': 'v8',
        'n_features_selected': len(feature_names_selected),
        'n_features_original': n_features_original,
        'horizon': 3,
        'trained_at': datetime.now().isoformat(),
        'spearman_guideline': '0.05-0.15=profitable, 0.15+=excellent',
    }


def save_model(model_data: Dict, model_dir: str):
    """保存模型"""
    os.makedirs(model_dir, exist_ok=True)

    # model.pkl
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    # model_v8.pkl
    v8_path = os.path.join(model_dir, 'model_v8.pkl')
    with open(v8_path, 'wb') as f:
        pickle.dump(model_data, f)

    # meta.json
    meta = {k: v for k, v in model_data.items()
            if k not in ('model', 'feature_importance')}
    meta['feature_importance_top20'] = dict(
        list(model_data['feature_importance'].items())[:20])
    meta_path = os.path.join(model_dir, 'model_v8_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n模型已保存: {model_path}")
    print(f"备份: {v8_path}")
    print(f"元数据: {meta_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', default='2025-01-01')
    parser.add_argument('--trials', type=int, default=30)
    args = parser.parse_args()

    print("=" * 60)
    print("v8 回归模型训练 (memmap优化)")
    print("=" * 60)
    print(f"数据起始: {args.start_date}")
    print(f"内存: memmap 磁盘写入, float32")
    print("=" * 60)

    all_data = load_data_from_db(DB_PATH)

    X, y, feature_names, symbols = prepare_training_data_memmap(
        all_data, horizon=3, start_date=args.start_date
    )

    if X is None or len(X) < 500:
        print(f"训练数据不足")
        return

    # 释放原始数据
    del all_data
    gc.collect()

    model_data = train_model_regression(X, y, feature_names, n_trials=args.trials)

    print(f"\n特征重要性 Top 20:")
    for name, score in list(model_data['feature_importance'].items())[:20]:
        print(f"  {name}: {score:.4f}")

    save_model(model_data, MODEL_DIR)

    print("\n训练完成!")
    print(f"Spearman: {model_data['cv_spearman']:.4f}")
    print(model_data['spearman_guideline'])


if __name__ == '__main__':
    main()