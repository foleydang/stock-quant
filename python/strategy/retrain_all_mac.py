#!/usr/bin/env python3
"""
Mac 端一键重训脚本 — lgb_30m + v9 日线 + LSTM embeddings

用法:
  python strategy/retrain_all_mac.py                    # 完整重训
  python strategy/retrain_all_mac.py --quick             # 快速验证 (少股票, 少树)
  python strategy/retrain_all_mac.py --30m-only          # 只重训 30m 模型
  python strategy/retrain_all_mac.py --daily-only        # 只重训 日线模型
  python strategy/retrain_all_mac.py --no-lstm           # 不加 LSTM embeddings

输出:
  python/models/lgb_30m/model.pkl           # 30分钟模型 (309+64=373特征)
  python/models/lgb_30m/model.pkl.backup    # 旧模型备份
  ../models/lgb_hs300_enhanced/model.pkl    # 日线模型
  ../models/lgb_hs300_enhanced/model.pkl.backup  # 旧日线模型备份

数据要求:
  - data/stock_data.db (从 OSS 下载)
  - data/lstm_embeddings.pkl (自动生成, 或从服务器复制)
"""

import os, sys, pickle, sqlite3, time, gc, shutil, warnings, argparse
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'python'))

DB_PATH = os.path.join(ROOT, 'python', 'data', 'stock_data.db')
LSTM_EMB_PATH = os.path.join(ROOT, 'python', 'data', 'lstm_embeddings.pkl')
MODEL_30M_DIR = os.path.join(ROOT, 'python', 'models', 'lgb_30m')
MODEL_DAILY_DIR = os.path.join(ROOT, '..', 'models', 'lgb_hs300_enhanced')

from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

# ============ 配置 ============
LGBM_PARAMS = {
    'n_estimators': 2000, 'learning_rate': 0.005, 'num_leaves': 31,
    'max_depth': 6, 'min_child_samples': 300, 'subsample': 0.3,
    'subsample_freq': 1, 'colsample_bytree': 0.2, 'feature_fraction_bynode': 0.6,
    'reg_alpha': 1.0, 'reg_lambda': 10.0, 'min_split_gain': 0.05,
    'path_smooth': 15, 'n_jobs': -1, 'verbosity': -1, 'random_state': 42,
}

QUICK_PARAMS = {
    'n_estimators': 500, 'learning_rate': 0.05,
    'num_leaves': 31, 'max_depth': 4, 'n_jobs': -1, 'verbosity': -1,
}

N_SEEDS = 5  # ensemble 数量
HORIZON = 3  # 预测未来3期


def load_lstm_embeddings():
    """加载 LSTM embeddings"""
    if not os.path.exists(LSTM_EMB_PATH):
        print("⚠️ LSTM embeddings 未找到，将跳过 (使用 --no-lstm 可避免此警告)")
        return {}
    print(f"📦 加载 LSTM embeddings: {LSTM_EMB_PATH}")
    with open(LSTM_EMB_PATH, 'rb') as f:
        return pickle.load(f)


def prepare_30m_dataset(conn, lstm_embeddings=None, quick=False):
    """准备30分钟线训练数据 — FeaturePipeline + LSTM embeddings"""
    print("\n" + "=" * 60)
    print("📊 准备 30分钟线 训练数据")
    print("=" * 60)

    from strategy.features import FeaturePipeline, rename_features_for_model

    # 加载所有30m股票
    symbols = [r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM kline_30m WHERE symbol LIKE '%.SZ' OR symbol LIKE '%.SH' ORDER BY symbol"
    ).fetchall()]

    if quick:
        symbols = symbols[:50]
    print(f"   共 {len(symbols)} 只股票")

    pipeline = FeaturePipeline({
        'label': '30分钟', 'horizon': HORIZON, 'db_table': 'kline_30m',
        'min_history': 150, 'purged_gap': 3, 'north_shift_days': 0,
    })

    all_X, all_y, all_sym, all_date = [], [], [], []
    lstm_feat_count = 0

    for i, sym in enumerate(symbols):
        try:
            df = pd.read_sql(
                "SELECT date, open, high, low, close, volume FROM kline_30m "
                "WHERE symbol=? ORDER BY date", conn, params=(sym,))
            if len(df) < 200:
                continue
            df = df.sort_values('date').reset_index(drop=True)

            # FeaturePipeline 特征
            feats = pipeline.compute_stock(df, sym)
            feats = feats.ffill().fillna(0)

            # 添加 LSTM embeddings
            if lstm_embeddings and sym in lstm_embeddings:
                emb_dict = lstm_embeddings[sym]
                emb_cols = []
                dates = pd.to_datetime(df['date'].values)
                for j, d in enumerate(dates):
                    date_str = d.strftime('%Y-%m-%d')
                    if date_str in emb_dict:
                        emb = emb_dict[date_str]
                        if lstm_feat_count == 0:
                            lstm_feat_count = len(emb)
                        emb_cols.append(emb)
                    else:
                        emb_cols.append(np.zeros(lstm_feat_count or 64))
                if emb_cols:
                    emb_arr = np.array(emb_cols)
                    for k in range(emb_arr.shape[1]):
                        feats[f'lstm_{k}'] = emb_arr[:, k]

            # 标签: 未来3期收益率
            close = df['close'].values
            future_ret = np.full(len(close), np.nan)
            for j in range(len(close) - HORIZON):
                future_ret[j] = (close[j + HORIZON] - close[j]) / close[j]

            valid = ~np.isnan(future_ret)
            if valid.sum() < 50:
                continue

            all_X.append(feats[valid].values.astype(np.float32))
            all_y.append(future_ret[valid].astype(np.float32))
            all_sym.extend([sym] * valid.sum())
            all_date.extend(dates[valid])

            if (i + 1) % 50 == 0:
                print(f"   [{i+1}/{len(symbols)}] {len(all_X)} 只已处理, {sum(len(y) for y in all_y):,} 样本")

        except Exception as e:
            if i == 0:
                print(f"   ⚠️ {sym}: {e}")
            continue
        finally:
            gc.collect()

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    print(f"\n   最终: {X.shape[1]} 特征, {len(y):,} 样本, {len(set(all_sym))} 只股票")
    if lstm_feat_count:
        print(f"   含 {lstm_feat_count} 维 LSTM embeddings")

    return X, y, all_sym, all_date, list(feats.columns)


def prepare_daily_dataset(conn, lstm_embeddings=None, quick=False):
    """准备日线训练数据 — FeaturePipeline + LSTM embeddings"""
    print("\n" + "=" * 60)
    print("📊 准备 日线 训练数据")
    print("=" * 60)

    from strategy.features import FeaturePipeline, rename_features_for_model

    # 加载所有日线股票 (限A股)
    symbols = [r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM kline_daily WHERE symbol LIKE '%.SZ' OR symbol LIKE '%.SH' ORDER BY symbol"
    ).fetchall()]

    if quick:
        symbols = symbols[:100]
    print(f"   共 {len(symbols)} 只股票")

    pipeline = FeaturePipeline({
        'label': '日线', 'horizon': HORIZON, 'db_table': 'kline_daily',
        'min_history': 120, 'purged_gap': 3, 'north_shift_days': 1,
    })

    all_X, all_y, all_sym, all_date = [], [], [], []
    lstm_feat_count = 0

    for i, sym in enumerate(symbols):
        try:
            df = pd.read_sql(
                "SELECT date, open, high, low, close, volume FROM kline_daily "
                "WHERE symbol=? ORDER BY date", conn, params=(sym,))
            if len(df) < 200:
                continue
            df = df.sort_values('date').reset_index(drop=True)

            feats = pipeline.compute_stock(df, sym)
            feats = feats.ffill().fillna(0)

            # 添加 LSTM embeddings
            if lstm_embeddings and sym in lstm_embeddings:
                emb_dict = lstm_embeddings[sym]
                emb_cols = []
                dates = pd.to_datetime(df['date'].values)
                for j, d in enumerate(dates):
                    date_str = d.strftime('%Y-%m-%d')
                    if date_str in emb_dict:
                        emb = emb_dict[date_str]
                        if lstm_feat_count == 0:
                            lstm_feat_count = len(emb)
                        emb_cols.append(emb)
                    else:
                        emb_cols.append(np.zeros(lstm_feat_count or 64))
                if emb_cols:
                    emb_arr = np.array(emb_cols)
                    for k in range(emb_arr.shape[1]):
                        feats[f'lstm_{k}'] = emb_arr[:, k]

            close = df['close'].values
            future_ret = np.full(len(close), np.nan)
            for j in range(len(close) - HORIZON):
                future_ret[j] = (close[j + HORIZON] - close[j]) / close[j]

            valid = ~np.isnan(future_ret)
            if valid.sum() < 50:
                continue

            all_X.append(feats[valid].values.astype(np.float32))
            all_y.append(future_ret[valid].astype(np.float32))
            all_sym.extend([sym] * valid.sum())
            all_date.extend(dates[valid])

            if (i + 1) % 50 == 0:
                print(f"   [{i+1}/{len(symbols)}] {len(all_X)} 只已处理")

        except Exception as e:
            if i == 0:
                print(f"   ⚠️ {sym}: {e}")
            continue
        finally:
            gc.collect()

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    print(f"\n   最终: {X.shape[1]} 特征, {len(y):,} 样本, {len(set(all_sym))} 只股票")
    if lstm_feat_count:
        print(f"   含 {lstm_feat_count} 维 LSTM embeddings")

    return X, y, all_sym, all_date, list(feats.columns)


def train_ensemble(X, y, feature_names, model_dir, model_name, params, n_seeds, quick=False):
    """训练 LGBM ensemble"""
    print(f"\n🔧 训练 {model_name} ensemble ({n_seeds} 个模型)...")

    os.makedirs(model_dir, exist_ok=True)
    n_seeds_actual = 2 if quick else n_seeds

    models = []
    val_ic_list = []
    val_mse_list = []

    # 按时间切分: 前80%训练, 后20%验证
    split_idx = int(len(y) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"   训练集: {len(y_train):,} | 验证集: {len(y_val):,}")

    for seed in range(n_seeds_actual):
        t0 = time.time()
        p = params.copy()
        p['random_state'] = seed * 42

        model = LGBMRegressor(**p)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                   eval_metric='l2', callbacks=[])

        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        ic = spearmanr(y_val, y_pred)[0]

        models.append(model)
        val_ic_list.append(ic)
        val_mse_list.append(mse)

        elapsed = time.time() - t0
        print(f"   seed={seed}: IC={ic:.4f}, MSE={mse:.6f}, {elapsed:.0f}s")

    avg_ic = np.mean(val_ic_list)
    avg_mse = np.mean(val_mse_list)
    print(f"   平均 IC: {avg_ic:.4f} | 平均 MSE: {avg_mse:.6f}")

    return models, avg_ic, avg_mse, val_ic_list, val_mse_list


def save_model(models, feature_names, model_dir, model_name, ic, mse, ic_list, mse_list, n_seeds):
    """保存模型到 pkl"""
    model_path = os.path.join(model_dir, 'model.pkl')

    # 备份旧模型
    if os.path.exists(model_path):
        backup_path = model_path + '.backup'
        shutil.copy2(model_path, backup_path)
        print(f"   📦 备份旧模型: {backup_path}")

    model_data = {
        'models': models,
        'feature_names': feature_names,
        'n_models': len(models),
        'horizon': HORIZON,
        'model_type': 'lgbm_ensemble',
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'train_samples': len(models[0]._Booster),
        'seeds': list(range(len(models))),
        'n_trees_per_model': models[0].n_estimators_,
        'avg_n_trees': np.mean([m.n_estimators_ for m in models]),
        'val_ic_list': ic_list,
        'val_mse_list': mse_list,
        'test_ic': ic,
        'test_mse': mse,
        'params': models[0].get_params(),
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"   ✅ 模型已保存: {model_path} ({os.path.getsize(model_path)/1024/1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description='Mac 一键重训脚本')
    parser.add_argument('--quick', action='store_true', help='快速验证')
    parser.add_argument('--30m-only', action='store_true', help='只训练30m模型')
    parser.add_argument('--daily-only', action='store_true', help='只训练日线模型')
    parser.add_argument('--no-lstm', action='store_true', help='不加 LSTM embeddings')
    parser.add_argument('--db', type=str, default=DB_PATH, help='数据库路径')
    args = parser.parse_args()

    print("=" * 60)
    print("🔄 Mac 端一键重训")
    print(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   数据库: {args.db}")
    print(f"   快速模式: {args.quick}")
    print(f"   LSTM: {'关闭' if args.no_lstm else '启用'}")
    print("=" * 60)

    params = QUICK_PARAMS if args.quick else LGBM_PARAMS
    n_seeds = 2 if args.quick else N_SEEDS

    # 加载 LSTM embeddings
    lstm_embeddings = None if args.no_lstm else load_lstm_embeddings()

    conn = sqlite3.connect(args.db)

    # ========== 训练 30m 模型 ==========
    if not args.daily_only:
        print("\n" + "=" * 60)
        print("📈 第一阶段: 训练 lgb_30m 模型")
        print("=" * 60)

        X, y, syms, dates, feat_names = prepare_30m_dataset(conn, lstm_embeddings, args.quick)

        models, ic, mse, ic_list, mse_list = train_ensemble(
            X, y, feat_names, MODEL_30M_DIR, 'lgb_30m', params, n_seeds, args.quick)

        save_model(models, feat_names, MODEL_30M_DIR, 'lgb_30m', ic, mse, ic_list, mse_list, n_seeds)

        del X, y
        gc.collect()

    # ========== 训练 日线模型 ==========
    if not args.__dict__.get('30m_only'):
        print("\n" + "=" * 60)
        print("📈 第二阶段: 训练 v9 日线模型")
        print("=" * 60)

        X, y, syms, dates, feat_names = prepare_daily_dataset(conn, lstm_embeddings, args.quick)

        models, ic, mse, ic_list, mse_list = train_ensemble(
            X, y, feat_names, MODEL_DAILY_DIR, 'v9-daily', params, n_seeds, args.quick)

        save_model(models, feat_names, MODEL_DAILY_DIR, 'v9-daily', ic, mse, ic_list, mse_list, n_seeds)

    conn.close()

    print("\n" + "=" * 60)
    print("✅ 全部训练完成!")
    print("=" * 60)
    print(f"   30m模型: {MODEL_30M_DIR}/model.pkl")
    print(f"   日线模型: {MODEL_DAILY_DIR}/model.pkl")
    print(f"\n   上传到服务器:")
    print(f"   scp python/models/lgb_30m/model.pkl root@47.242.158.242:~/github/stock-quant/python/models/lgb_30m/")
    print(f"   scp ../models/lgb_hs300_enhanced/model.pkl root@47.242.158.242:~/github/stock-quant/models/lgb_hs300_enhanced/")


if __name__ == '__main__':
    main()