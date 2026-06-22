#!/usr/bin/env python3
"""
日线选股训练 — 100+ 特征 + LightGBM 排序模型
在 Mac 上运行，结果 scp 到服务器

用法:
  python qlib_pipeline/train_daily.py

输出:
  models/lgb_daily/
    ├── model.pkl          # sklearn Pipeline
    ├── model.txt          # LightGBM 原生格式
    ├── meta.json
    └── feature_names.json
"""

import os, sys, json, pickle, argparse, warnings
from datetime import datetime
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config_loader import get_db_path
from qlib_pipeline.features_daily import compute_features, FEATURE_NAMES

DB_PATH = get_db_path()


def load_data(conn):
    """加载所有股票日线数据"""
    symbols = [r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM kline_daily ORDER BY symbol"
    ).fetchall()]

    data = {}
    for sym in symbols:
        df = pd.read_sql(
            "SELECT date, open, high, low, close, volume FROM kline_daily "
            "WHERE symbol=? ORDER BY date", conn, params=(sym,)
        )
        if len(df) < 120:
            continue
        df['date'] = pd.to_datetime(df['date'].apply(
            lambda d: f"{str(d)[:4]}-{str(d)[4:6]}-{str(d)[6:8]}"
            if len(str(d)) == 8 else str(d)[:10]
        ))
        data[sym] = df
    return data


def build_dataset(data, target_horizon=5):
    """构建特征-标签数据集"""
    X_rows, y_rows, meta_rows = [], [], []

    for sym, df in data.items():
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        vol = df['volume'].values
        dates = df['date'].values

        for i in range(120, len(df) - target_horizon):
            feats = compute_features(close[:i+1], high[:i+1], low[:i+1], vol[:i+1])
            if feats is None:
                continue

            future_return = close[i + target_horizon] / close[i] - 1
            if np.isnan(future_return) or np.isinf(future_return):
                continue

            X_rows.append(feats)
            y_rows.append(future_return)
            meta_rows.append({
                'symbol': sym,
                'date': str(dates[i])[:10],
                'future_return': future_return,
            })

    X = pd.DataFrame(X_rows)
    for col in FEATURE_NAMES:
        if col not in X.columns:
            X[col] = 0.0
    X = X[FEATURE_NAMES].fillna(0).replace([np.inf, -np.inf], 0)

    return X, np.array(y_rows), pd.DataFrame(meta_rows)


def train_model(X_train, y_train, X_val, y_val, output_dir):
    """训练 LightGBM 排序模型"""
    os.makedirs(output_dir, exist_ok=True)

    model = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        num_leaves=128,
        learning_rate=0.03,
        n_estimators=1000,
        subsample=0.7,
        colsample_bytree=0.7,
        subsample_freq=1,
        reg_alpha=0.5,
        reg_lambda=0.5,
        min_child_samples=50,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model),
    ])

    n_train = len(X_train)
    n_val = len(X_val)

    print(f"  训练集: {n_train:,} 样本, {X_train.shape[1]} 特征")
    print(f"  验证集: {n_val:,} 样本")

    # 创建 group (每个日期一个 group，用于 lambdarank)
    # 简化：用样本数作为 group size
    train_group = [n_train]
    val_group = [n_val]

    t0 = datetime.now()
    pipeline.fit(
        X_train, y_train,
        model__group=train_group,
        model__eval_set=[(X_val, y_val)],
        model__eval_group=[val_group],
        model__eval_metric='ndcg',
        model__eval_at=[5, 10],
        model__callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100),
        ],
    )
    elapsed = (datetime.now() - t0).total_seconds()

    # 评估
    from scipy.stats import spearmanr
    y_pred = pipeline.predict(X_val)
    ic = np.corrcoef(y_pred, y_val)[0, 1]
    rank_ic, _ = spearmanr(y_pred, y_val)

    print(f"\n  IC={ic:.4f}, RankIC={rank_ic:.4f}, 耗时={elapsed:.0f}s")

    # 特征重要性
    importance = model.feature_importances_
    top_idx = np.argsort(importance)[-15:][::-1]
    print(f"  Top-15 特征:")
    for idx in top_idx:
        print(f"    {FEATURE_NAMES[idx]:20s}: {importance[idx]:.0f}")

    # 保存
    pipeline_path = os.path.join(output_dir, 'model.pkl')
    with open(pipeline_path, 'wb') as f:
        pickle.dump(pipeline, f)

    txt_path = os.path.join(output_dir, 'model.txt')
    model.booster_.save_model(txt_path)

    with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
        json.dump({'features': FEATURE_NAMES, 'horizon': target_horizon}, f)

    meta = {
        'model': 'LightGBMRanker',
        'horizon': target_horizon,
        'label': 'future_5d_return',
        'features': len(FEATURE_NAMES),
        'IC': round(float(ic), 4),
        'RankIC': round(float(rank_ic), 4),
        'train_time_s': round(elapsed),
        'train_samples': n_train,
        'val_samples': n_val,
        'timestamp': datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    size_mb = os.path.getsize(pipeline_path) / 1024 / 1024
    print(f"\n✅ 模型已导出: {output_dir}")
    print(f"   model.pkl ({size_mb:.1f}MB)")
    return meta


if __name__ == '__main__':
    import sqlite3
    from scipy import stats

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='models/lgb_daily')
    parser.add_argument('--horizon', type=int, default=5)
    args = parser.parse_args()

    target_horizon = args.horizon

    conn = sqlite3.connect(DB_PATH)

    print("📡 加载数据...")
    data = load_data(conn)
    print(f"  {len(data)} 只股票")

    print("🔧 构建特征...")
    X, y, meta = build_dataset(data, target_horizon=target_horizon)

    # 时间切分: 前70%训练, 后30%验证
    n = len(X)
    train_end = int(n * 0.7)

    X_train, y_train = X.iloc[:train_end], y[:train_end]
    X_val, y_val = X.iloc[train_end:], y[train_end:]

    print(f"  总样本: {n:,}")
    print(f"  训练: {len(X_train):,} | 验证: {len(X_val):,} | 特征: {X.shape[1]}")

    print("🤖 训练 LightGBMRanker...")
    train_model(X_train, y_train, X_val, y_val, args.output)

    conn.close()