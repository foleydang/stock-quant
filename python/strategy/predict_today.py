#!/usr/bin/env python3
"""
今日预测 — 用最新模型对全部股票打分, 输出 Top/Bottom 候选

用法:
  python strategy/predict_today.py                  # 预测最新交易日
  python strategy/predict_today.py --date 2026-06-30
  python strategy/predict_today.py --top 20         # 显示 Top-20
"""

import os, sys, pickle, sqlite3, argparse, json
import numpy as np
import pandas as pd
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'python'))

DB_PATH = os.path.join(ROOT, 'data', 'stock_data.db')
MODEL_DIR = os.path.join(ROOT, '..', 'models', 'lgb_hs300_enhanced')

from strategy.train_enhanced import (
    load_kline_data, calculate_features,
)


def get_stock_names(conn):
    try:
        df = pd.read_sql("SELECT symbol, name FROM stock_info", conn)
        return dict(zip(df['symbol'], df['name']))
    except Exception:
        return {}


def predict_latest(model_path, date_str=None, top_n=15):
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    feature_names = model_data['feature_names']
    models = model_data.get('models', [])
    weights_raw = model_data.get('regime_weights', {})
    if 'all' in weights_raw:
        weights = weights_raw['all']
    else:
        weights = weights_raw if weights_raw else {k: 1.0/len(models) for k in range(len(models))}

    print(f"模型版本: {model_data.get('model_version')}")
    print(f"训练时间: {model_data.get('trained_at', 'unknown')}")
    print(f"特征数: {len(feature_names)} | 模型数: {len(models)}")

    conn = sqlite3.connect(DB_PATH)

    if date_str is None:
        latest = pd.read_sql(
            "SELECT MAX(date) as d FROM kline_daily", conn
        )['d'].iloc[0]
        date_str = str(latest)[:10]
    print(f"预测日期: {date_str}")

    df = load_kline_data(conn)
    name_map = get_stock_names(conn)
    conn.close()

    df['trade_date'] = pd.to_datetime(df['trade_date'])
    cutoff = pd.to_datetime(date_str)
    df = df[df['trade_date'] <= cutoff].copy()

    if len(df) == 0:
        print(f"❌ 没有日期 <= {date_str} 的数据")
        return

    conn = sqlite3.connect(DB_PATH)
    results = calculate_features(df, conn)
    conn.close()

    # 对每只股票, 取 pred_date 当天的特征
    target_date = pd.to_datetime(date_str)
    pred_rows = []
    pred_syms = []

    for sym, (stock_df, feats) in results.items():
        if len(stock_df) == 0:
            continue
        stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'])
        available = stock_df[stock_df['trade_date'] <= target_date]
        if len(available) == 0:
            continue
        pred_date = available['trade_date'].max()
        row_idx = available.index[available['trade_date'] == pred_date][0]
        # 在 feats 中找对应行 (feats 和 stock_df 行对齐)
        if row_idx not in feats.index:
            continue
        feat_row = feats.loc[row_idx].fillna(0).replace([np.inf, -np.inf], 0)
        pred_rows.append(feat_row)
        pred_syms.append(sym)

    if not pred_rows:
        print(f"❌ 没有可用样本")
        return

    # 构造特征矩阵, 按训练时顺序对齐
    feat_df = pd.DataFrame(pred_rows).fillna(0).replace([np.inf, -np.inf], 0)
    for c in feature_names:
        if c not in feat_df.columns:
            feat_df[c] = 0
    X = feat_df[feature_names].values

    actual_pred_date = pred_rows and (
        stock_df['date'].iloc[-1] if False else None
    )
    # 重新取实际预测日
    actual_date = None
    for sym, (stock_df, feats) in results.items():
        stock_df2 = stock_df.copy()
        stock_df2['trade_date'] = pd.to_datetime(stock_df2['trade_date'])
        avail = stock_df2[stock_df2['trade_date'] <= target_date]
        if len(avail) > 0:
            d = avail['trade_date'].max()
            if actual_date is None or d > actual_date:
                actual_date = d

    print(f"实际预测日: {actual_date.date() if actual_date else 'N/A'} | 样本数: {len(X)}")

    # 集成预测
    preds = np.zeros(len(X))
    keys = list(weights.keys())
    for i, model in enumerate(models):
        key = keys[i] if i < len(keys) else i
        w = weights.get(key, 0)
        if w == 0 and isinstance(key, str):
            # 尝试按索引
            w = list(weights.values())[i] if i < len(weights) else 0
        preds += model.predict(X) * w

    # 归一化权重检查
    w_sum = sum(weights.values())
    if abs(w_sum - 1.0) > 0.01:
        print(f"⚠️ 权重和={w_sum:.3f} (非1.0), 重新归一化")
        preds = preds / max(w_sum, 1e-9)

    sort_idx = np.argsort(-preds)

    print(f"\n{'='*70}")
    print(f"📈 Top-{top_n} 买入候选 (预测涨幅最高)")
    print(f"{'='*70}")
    print(f"{'排名':>4} {'代码':>10} {'名称':>15} {'得分':>8}")
    for rank, idx in enumerate(sort_idx[:top_n], 1):
        sym = pred_syms[idx]
        name = name_map.get(sym, '未知')
        print(f"{rank:>4} {sym:>10} {name:>15} {preds[idx]:>+8.4f}")

    print(f"\n{'='*70}")
    print(f"📉 Bottom-{top_n} 规避候选 (预测跌幅最大)")
    print(f"{'='*70}")
    print(f"{'排名':>4} {'代码':>10} {'名称':>15} {'得分':>8}")
    for rank, idx in enumerate(sort_idx[-top_n:][::-1], 1):
        sym = pred_syms[idx]
        name = name_map.get(sym, '未知')
        print(f"{rank:>4} {sym:>10} {name:>15} {preds[idx]:>+8.4f}")

    out_path = os.path.join(MODEL_DIR, f'prediction_{actual_date.strftime("%Y%m%d") if actual_date else "unknown"}.csv')
    result_df = pd.DataFrame({
        'rank': np.arange(1, len(preds) + 1),
        'symbol': [pred_syms[i] for i in sort_idx],
        'name': [name_map.get(pred_syms[i], '未知') for i in sort_idx],
        'score': preds[sort_idx],
    })
    result_df.to_csv(out_path, index=False)
    print(f"\n💾 完整结果保存到 {out_path}")
    print(f"   共 {len(preds)} 只股票打分")

    return result_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default=None, help='预测日期 YYYY-MM-DD')
    parser.add_argument('--top', type=int, default=15, help='显示 Top/Bottom 数量')
    args = parser.parse_args()

    model_path = os.path.join(MODEL_DIR, 'model.pkl')
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        sys.exit(1)

    predict_latest(model_path, date_str=args.date, top_n=args.top)
