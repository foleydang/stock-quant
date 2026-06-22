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
from scipy.stats import rankdata, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config_loader import get_db_path
from qlib_pipeline.features_daily import compute_features_batch, compute_features, FEATURE_NAMES

# OHLCV 特征 (不含辅助特征)
OHLCV_FEATURES = [f for f in FEATURE_NAMES if not f.startswith(('fund_', 'macro_', 'north_', 'sent_', 'sector_'))]

DB_PATH = get_db_path()


def load_data(conn):
    """加载所有股票日线数据 + 辅助数据"""
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

    # 加载辅助数据
    aux = load_auxiliary(conn, data)
    return data, aux


def load_auxiliary(conn, data):
    """加载基本面、行业、宏观、情绪数据"""
    aux = {}

    # 基本面
    try:
        fund = pd.read_sql("SELECT * FROM fundamental_daily", conn)
        fund['trade_date'] = pd.to_datetime(fund['trade_date'])
        fund = fund.set_index(['symbol', 'trade_date']).sort_index()
        aux['fund'] = fund
        print(f"  基本面: {len(fund)} 条")
    except Exception:
        aux['fund'] = None

    # 行业
    try:
        sector = pd.read_sql("SELECT symbol, industry FROM stock_sector", conn)
        aux['sector'] = sector.set_index('symbol')['industry'].to_dict()
        print(f"  行业: {len(aux['sector'])} 只")
    except Exception:
        aux['sector'] = {}

    # 宏观
    try:
        macro = pd.read_sql(
            "SELECT trade_date, hs300_close, hs300_volume, shibor_1w, cn_10y, "
            "shibor_1m, cn_2y, cn_5y, cn_30y FROM macro_daily ORDER BY trade_date", conn)
        macro['trade_date'] = pd.to_datetime(macro['trade_date'])
        macro = macro.set_index('trade_date')
        aux['macro'] = macro
        print(f"  宏观: {len(macro)} 天")
    except Exception:
        aux['macro'] = None

    # 北向资金
    try:
        north = pd.read_sql(
            "SELECT trade_date, north_net, total_net FROM north_flow ORDER BY trade_date", conn)
        north['trade_date'] = pd.to_datetime(north['trade_date'])
        north = north.set_index('trade_date')
        aux['north'] = north
        print(f"  北向: {len(north)} 天")
    except Exception:
        aux['north'] = None

    # 情绪
    try:
        sent = pd.read_sql(
            "SELECT symbol, trade_date, is_limit_up, is_limit_down, vol_ratio_20 "
            "FROM sentiment_daily", conn)
        sent['trade_date'] = pd.to_datetime(sent['trade_date'])
        sent = sent.set_index(['symbol', 'trade_date']).sort_index()
        aux['sent'] = sent
        print(f"  情绪: {len(sent)} 条")
    except Exception:
        aux['sent'] = None

    return aux


CACHE_FILE = os.path.join(os.path.dirname(DB_PATH), 'features_cache_v3.parquet')


def build_dataset(data, aux, target_horizon=5, use_cache=True):
    """构建特征-标签数据集 (向量化批量计算，支持缓存)"""
    if use_cache and os.path.exists(CACHE_FILE):
        print(f"  📦 加载缓存: {CACHE_FILE}")
        df = pd.read_parquet(CACHE_FILE)
        y = df.pop('__label__').values
        dates = df.pop('__date__').values if '__date__' in df.columns else None
        return df, y, dates

    from tqdm import tqdm
    X_rows, y_rows, date_rows = [], [], []

    for sym, df in tqdm(data.items(), desc="  计算特征", unit="stock"):
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        vol = df['volume'].values

        # 批量计算该股票全部特征
        feats_batch = compute_features_batch(close, high, low, vol)
        if feats_batch is None:
            continue

        # 提取训练样本
        for i in range(120, len(df) - target_horizon):
            future_return = close[i + target_horizon] / close[i] - 1
            if np.isnan(future_return) or np.isinf(future_return):
                continue

            row = {}
            valid = True
            for k in OHLCV_FEATURES:
                v = feats_batch[k][i]
                if np.isnan(v) or np.isinf(v):
                    valid = False
                    break
                row[k] = v
            if not valid:
                continue

            # 添加辅助特征
            _add_aux_features(row, sym, dates[i], aux)

            X_rows.append(row)
            y_rows.append(future_return)
            date_rows.append(str(dates[i])[:10])

    X = pd.DataFrame(X_rows)
    X = X[FEATURE_NAMES].fillna(0).replace([np.inf, -np.inf], 0)
    y = np.array(y_rows)
    dates = np.array(date_rows)

    # 截面排名标签: 每个日期内，对 y 做排名归一化
    y_cs = _cs_rank(y, dates)

    # 保存缓存
    cache = X.copy()
    cache['__label__'] = y_cs
    cache['__date__'] = dates
    cache.to_parquet(CACHE_FILE, index=False)
    print(f"  💾 缓存已保存: {CACHE_FILE}")

    return X, y_cs, dates


def _cs_rank(y, dates):
    """截面排名: 每个日期内，对 y 排名并归一化到 [0, 1]"""
    y_ranked = np.zeros_like(y)
    unique_dates = np.unique(dates)
    for d in unique_dates:
        mask = dates == d
        if mask.sum() < 5:
            y_ranked[mask] = y[mask]  # 样本太少，保持原值
            continue
        ranks = rankdata(y[mask])
        y_ranked[mask] = (ranks - 1) / (len(ranks) - 1)  # 归一化到 [0, 1]
    return y_ranked


def train_model(X_train, y_train, X_val, y_val, output_dir, target_horizon=5):
    """训练 LightGBM 排序模型"""
    os.makedirs(output_dir, exist_ok=True)

    model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
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

    print(f"  训练集: {len(X_train):,} 样本, {X_train.shape[1]} 特征")
    print(f"  验证集: {len(X_val):,} 样本")

    t0 = datetime.now()
    pipeline.fit(
        X_train, y_train,
        model__eval_set=[(X_val, y_val)],
        model__eval_metric='rmse',
        model__callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100),
        ],
    )
    elapsed = (datetime.now() - t0).total_seconds()

    # 评估
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
        'model': 'LightGBM',
        'horizon': target_horizon,
        'label': 'cs_rank_5d',
        'features': len(FEATURE_NAMES),
        'IC': round(float(ic), 4),
        'RankIC': round(float(rank_ic), 4),
        'train_time_s': round(elapsed),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'timestamp': datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    size_mb = os.path.getsize(pipeline_path) / 1024 / 1024
    print(f"\n✅ 模型已导出: {output_dir}")
    print(f"   model.pkl ({size_mb:.1f}MB)")
    return meta


def _add_aux_features(row, sym, date, aux):
    """添加辅助特征到 feature dict"""
    date_str = pd.Timestamp(str(date)[:10])


if __name__ == '__main__':
    import sqlite3

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='models/lgb_daily')
    parser.add_argument('--horizon', type=int, default=5)
    args = parser.parse_args()

    target_horizon = args.horizon

    conn = sqlite3.connect(DB_PATH)

    print("📡 加载数据...")
    data, aux = load_data(conn)
    print(f"  {len(data)} 只股票")

    print("🔧 构建特征...")
    X, y, dates = build_dataset(data, aux, target_horizon=target_horizon)

    # 时间切分: 前70%训练, 后30%验证
    n = len(X)
    train_end = int(n * 0.7)

    X_train, y_train = X.iloc[:train_end], y[:train_end]
    X_val, y_val = X.iloc[train_end:], y[train_end:]

    print(f"  总样本: {n:,}")
    print(f"  训练: {len(X_train):,} | 验证: {len(X_val):,} | 特征: {X.shape[1]}")

    print("🤖 训练 LightGBM...")
    train_model(X_train, y_train, X_val, y_val, args.output, target_horizon)

    conn.close()

    # 基本面
    fund = aux.get('fund')
    if fund is not None and sym in fund.index:
        try:
            fund_stock = fund.loc[sym]
            fund_before = fund_stock[fund_stock.index <= date_str]
            if len(fund_before) > 0:
                latest = fund_before.iloc[-1]
                row['fund_roe'] = float(latest.get('roe', 0)) if pd.notna(latest.get('roe')) else 0
                row['fund_np_yoy'] = float(latest.get('net_profit_yoy', 0)) if pd.notna(latest.get('net_profit_yoy')) else 0
                row['fund_debt'] = float(latest.get('debt_ratio', 0)) if pd.notna(latest.get('debt_ratio')) else 0
            else:
                row['fund_roe'] = row['fund_np_yoy'] = row['fund_debt'] = 0
        except Exception:
            row['fund_roe'] = row['fund_np_yoy'] = row['fund_debt'] = 0
    else:
        row['fund_roe'] = row['fund_np_yoy'] = row['fund_debt'] = 0

    # 行业
    industry = aux.get('sector', {}).get(sym, '\u672a\u77e5')
    row['sector_code'] = float(hash(industry) % 100) / 100

    # 宏观
    macro = aux.get('macro')
    if macro is not None and date_str in macro.index:
        m = macro.loc[date_str]
        if macro.index.get_loc(date_str) > 0:
            prev = macro.iloc[macro.index.get_loc(date_str) - 1]['hs300_close']
            row['macro_hs300_chg'] = float((m['hs300_close'] - prev) / prev) if prev > 0 else 0
        else:
            row['macro_hs300_chg'] = 0
        row['macro_shibor_1w'] = float(m.get('shibor_1w', 0)) if pd.notna(m.get('shibor_1w')) else 0
        row['macro_cn_10y'] = float(m.get('cn_10y', 0)) if pd.notna(m.get('cn_10y')) else 0
    else:
        row['macro_hs300_chg'] = row['macro_shibor_1w'] = row['macro_cn_10y'] = 0

    # 北向
    north = aux.get('north')
    if north is not None and date_str in north.index:
        row['north_net'] = float(north.loc[date_str, 'north_net']) if pd.notna(north.loc[date_str, 'north_net']) else 0
    else:
        row['north_net'] = 0

    # 情绪
    sent = aux.get('sent')
    if sent is not None and sym in sent.index:
        try:
            sent_stock = sent.loc[sym]
            if date_str in sent_stock.index:
                s = sent_stock.loc[date_str]
                row['sent_limit_up'] = float(s.get('is_limit_up', 0)) if pd.notna(s.get('is_limit_up')) else 0
                row['sent_limit_down'] = float(s.get('is_limit_down', 0)) if pd.notna(s.get('is_limit_down')) else 0
                row['sent_vol_ratio'] = float(s.get('vol_ratio_20', 0)) if pd.notna(s.get('vol_ratio_20')) else 0
            else:
                row['sent_limit_up'] = row['sent_limit_down'] = row['sent_vol_ratio'] = 0
        except Exception:
            row['sent_limit_up'] = row['sent_limit_down'] = row['sent_vol_ratio'] = 0
    else:
        row['sent_limit_up'] = row['sent_limit_down'] = row['sent_vol_ratio'] = 0
