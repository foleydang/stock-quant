#!/usr/bin/env python3
"""
30m 模型回测 — 用训练特征缓存, 快速验证最近1个月准确度

优化: 直接加载缓存的特征 (all_features), 不重新计算
      只需对每个时间戳做 ensemble 预测 + 对比实际收益
"""

import os
import sys
import pickle
import sqlite3
import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Dict, List

PYTHON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python')
sys.path.insert(0, PYTHON_DIR)
sys.path.insert(0, os.path.join(PYTHON_DIR, 'strategy'))

DB_PATH = os.path.join(PYTHON_DIR, 'data/stock_data.db')
MODEL_PATH = os.path.join(PYTHON_DIR, 'models/lgb_30m/model.pkl')
CACHE_DIR = os.path.join(PYTHON_DIR, 'models/.cache')
HORIZON = 3
SIGNAL_WINDOW = 3


def find_cache():
    for f in os.listdir(CACHE_DIR):
        if f.startswith('features_kline_30m_') and f.endswith('.pkl'):
            return os.path.join(CACHE_DIR, f)
    return None


def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def run_backtest():
    print("=" * 70)
    print("  30m 模型回测 — 最近1个月准确度 (使用特征缓存)")
    print("=" * 70)

    model_data = load_model()
    models = model_data['models']
    feature_names = model_data['feature_names']
    print(f"  模型: {len(models)}模型 Ensemble, "
          f"特征={len(feature_names)}, test_ic={model_data.get('test_ic', '?')}")

    cache_path = find_cache()
    if not cache_path:
        print("❌ 特征缓存不存在, 请先训练模型")
        return
    print(f"  加载特征缓存: {cache_path}")
    with open(cache_path, 'rb') as f:
        cached = pickle.load(f)
    all_features = cached['all_features']
    print(f"  缓存: {len(all_features)} 只股票")

    conn = sqlite3.connect(DB_PATH)

    # 取最近30天的所有时间戳
    df = pd.read_sql_query(
        "SELECT DISTINCT date FROM kline_30m ORDER BY date DESC LIMIT 2000", conn
    )
    df['date'] = pd.to_datetime(df['date'])
    max_date = df['date'].max()
    cutoff = max_date - timedelta(days=30)
    recent_ts = df[df['date'] >= cutoff]['date'].sort_values().values

    # 均匀采样10个时间戳
    n_checkpoints = 10
    if len(recent_ts) > n_checkpoints:
        idx = np.linspace(0, len(recent_ts) - 1, n_checkpoints, dtype=int)
        recent_ts = recent_ts[idx]
    recent_ts = sorted(recent_ts)
    print(f"  回测 {len(recent_ts)} 个时间戳: "
          f"{pd.Timestamp(recent_ts[0]).strftime('%Y-%m-%d')} → "
          f"{pd.Timestamp(recent_ts[-1]).strftime('%Y-%m-%d')}")
    print(f"  每个时间戳: 预测所有股票 → 对比实际{HORIZON}根K线收益\n")

    from scipy.stats import spearmanr

    all_ics = []
    all_top_rets = []
    all_bot_rets = []
    all_long_short = []
    sample_top_stocks = None

    for i, ts in enumerate(recent_ts):
        ts_pt = pd.Timestamp(ts)
        ts_str = ts_pt.strftime('%Y-%m-%d %H:%M')

        # 收集该时间戳所有股票的预测
        preds = {}
        for sym, feats in all_features.items():
            # 找到 <= ts 的最后一个行
            mask = feats.index <= ts
            if mask.sum() < SIGNAL_WINDOW:
                continue
            sub = feats.loc[mask]
            # 补齐缺失特征 (基本面等可能不在缓存里)
            for c in feature_names:
                if c not in sub.columns:
                    sub[c] = 0
            row_data = sub[feature_names].iloc[-SIGNAL_WINDOW:]
            if row_data.isna().all().all():
                continue
            vals = row_data.values.astype(np.float32)
            ps = [float(np.mean([m.predict(v.reshape(1, -1))[0] for m in models]))
                  for v in vals]
            ps = [p for p in ps if not np.isnan(p) and not np.isinf(p)]
            if ps:
                preds[sym] = float(np.mean(ps))

        # 取实际收益: ts 之后的 close 变化
        ts_next = ts_pt + timedelta(hours=2)
        ts_str_q = ts_pt.strftime('%Y-%m-%d %H:%M:%S')
        rets = {}
        for sym in preds:
            rows = conn.execute(
                "SELECT close FROM kline_30m WHERE symbol=? AND date >= ? "
                "ORDER BY date LIMIT ?", (sym, ts_str_q, HORIZON + 1)
            ).fetchall()
            if len(rows) >= HORIZON + 1:
                c0 = float(rows[0][0])
                c1 = float(rows[HORIZON][0])
                if c0 > 0:
                    rets[sym] = (c1 - c0) / c0

        common = [s for s in preds if s in rets]
        if len(common) < 20:
            print(f"  [{i+1}/{len(recent_ts)}] {ts_str}: 数据不足 ({len(common)} 只)")
            continue

        pred_arr = np.array([preds[s] for s in common])
        ret_arr = np.array([rets[s] for s in common])
        ic = spearmanr(pred_arr, ret_arr)[0]
        all_ics.append(ic)

        sort_idx = np.argsort(pred_arr)
        n = len(common)
        top_n = max(5, n // 20)
        top_ret = ret_arr[sort_idx[-top_n:]].mean()
        bot_ret = ret_arr[sort_idx[:top_n]].mean()
        all_top_rets.append(top_ret)
        all_bot_rets.append(bot_ret)
        all_long_short.append(top_ret - bot_ret)

        # 第一个时间戳显示 Top5 详情
        if sample_top_stocks is None:
            top_syms = [common[j] for j in sort_idx[-5:][::-1]]
            names = {}
            for s in top_syms:
                row = conn.execute("SELECT name FROM stock_info WHERE symbol=?", (s,)).fetchone()
                names[s] = row[0] if row and row[0] else s
            sample_top_stocks = [(s, names.get(s, s), preds[s], rets[s]) for s in top_syms]

        print(f"  [{i+1}/{len(recent_ts)}] {ts_str} | IC={ic:+.4f} | "
              f"Top{top_n}={top_ret:+.2%} Bot{top_n}={bot_ret:+.2%} "
              f"多空={top_ret-bot_ret:+.2%} ({len(common)}只)")

    # 显示第一个时间戳的 Top5 详情
    if sample_top_stocks:
        print(f"\n  📌 首个时间戳 Top5 预测股票:")
        for sym, name, pred, ret in sample_top_stocks:
            print(f"      {name:<8} {sym:<8} 预测={pred:+.4f} 实际{HORIZON}bar={ret:+.2%}")

    conn.close()

    print("\n" + "=" * 70)
    print("  📊 回测汇总")
    print("=" * 70)
    if all_ics:
        print(f"  平均 IC: {np.mean(all_ics):+.4f} "
              f"(正向={sum(1 for x in all_ics if x>0)}/{len(all_ics)})")
        icir = np.mean(all_ics) / np.std(all_ics) if np.std(all_ics) > 0 else 0
        print(f"  ICIR (稳定性): {icir:.3f}")
        print(f"  平均 Top收益: {np.mean(all_top_rets):+.2%}")
        print(f"  平均 Bot收益: {np.mean(all_bot_rets):+.2%}")
        print(f"  平均多空差:   {np.mean(all_long_short):+.2%}")
        win = sum(1 for x in all_long_short if x > 0)
        print(f"  多空胜率:     {win}/{len(all_long_short)} ({win/len(all_long_short):.0%})")


if __name__ == '__main__':
    run_backtest()
