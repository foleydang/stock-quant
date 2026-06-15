#!/usr/bin/env python3
"""v11 日线模型快速回测 — 基于截面排名信号"""
import sys, os, sqlite3, pickle, warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
from datetime import datetime

DB_PATH = os.path.join(ROOT, 'data/stock_data.db')
MODEL_PATH = os.path.join(ROOT, 'models/lgb_daily/model.pkl')

# 加载模型
print(f"📦 加载模型: {MODEL_PATH}")
with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)
models = model_data['models']
feature_names = model_data['feature_names']
print(f"   特征: {len(feature_names)} | 模型: {len(models)}")

# 加载数据
conn = sqlite3.connect(DB_PATH)
symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM kline_daily")]
print(f"📊 {len(symbols)} 只股票")

# 用 test 期间回测 (2025-03-24 ~ 2026-06-11)
TEST_START = '2025-03-24'
TEST_END = '2026-06-11'

# 计算特征 + 预测 (复用 FeaturePipeline)
from strategy.features import FeaturePipeline
pipeline = FeaturePipeline({})

all_preds = {}  # {symbol: {date_str: pred}}
all_returns = {}  # {symbol: {date_str: ret}}
horizon = 5
RETURN_CLIP = 0.20

print(f"🔧 计算特征 + 预测 (test: {TEST_START} ~ {TEST_END})...")
count = 0
for sym in symbols:
    df = pd.read_sql(f"SELECT * FROM kline_daily WHERE symbol=? ORDER BY date", conn, params=(sym,))
    if len(df) < 200:
        continue
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    df = df.sort_values('date').reset_index(drop=True)
    df = df.drop_duplicates(subset=['date']).reset_index(drop=True)

    try:
        feats = pipeline.compute_stock(df, sym)
        feats = feats.fillna(method='ffill').fillna(0)
        feats = feats.reindex(columns=feature_names, fill_value=0)
    except Exception:
        continue

    # 前向收益
    close = df['close'].values.astype(float)
    date_vals = df['date'].values

    # 预测
    X = feats.values.astype(np.float32)
    preds = np.zeros(len(X))
    for m in models:
        preds += m.predict(X) / len(models)

    # 日期索引
    date_strs = [str(pd.Timestamp(d))[:10] for d in date_vals]

    pred_map = {}
    ret_map = {}
    for i, d in enumerate(date_strs):
        if d >= TEST_START and d <= TEST_END:
            if i < len(close) - horizon and not np.isnan(preds[i]):
                ret = (close[i + horizon] - close[i]) / close[i]
                if abs(ret) < RETURN_CLIP:
                    pred_map[d] = float(preds[i])
                    ret_map[d] = float(ret)

    if pred_map:
        all_preds[sym] = pred_map
        all_returns[sym] = ret_map

    count += 1
    if count % 50 == 0:
        print(f"   {count}/{len(symbols)}")

conn.close()
print(f"   {count} 只股票完成")

# 截面回测: 每日选 top K 买入
print(f"\n📈 截面回测...")
all_dates = sorted(set().union(*[set(p.keys()) for p in all_preds.values()]))
print(f"   交易日: {len(all_dates)}")

TOP_K = 10
results = []
capital = 1.0  # 归一化

for d in all_dates:
    day_preds = []
    for sym in all_preds:
        if d in all_preds[sym] and d in all_returns[sym]:
            day_preds.append((sym, all_preds[sym][d], all_returns[sym][d]))

    if len(day_preds) < TOP_K:
        continue

    day_preds.sort(key=lambda x: -x[1])  # 按预测排名

    # 多空: 买入 top K, 卖出 bottom K
    long_ret = np.mean([x[2] for x in day_preds[:TOP_K]])
    short_ret = np.mean([x[2] for x in day_preds[-TOP_K:]])
    spread = long_ret - short_ret

    results.append({
        'date': d,
        'long_ret': long_ret,
        'short_ret': short_ret,
        'spread': spread,
        'market_ret': np.mean([x[2] for x in day_preds]),
        'n_stocks': len(day_preds)
    })

df_results = pd.DataFrame(results)
df_results['cum_spread'] = (1 + df_results['spread']).cumprod()
df_results['cum_long'] = (1 + df_results['long_ret']).cumprod()
df_results['cum_short'] = (1 + df_results['short_ret']).cumprod()

# IC 评估
preds_all = []
rets_all = []
for d in all_dates:
    day_preds = []
    for sym in all_preds:
        if d in all_preds[sym] and d in all_returns[sym]:
            day_preds.append((all_preds[sym][d], all_returns[sym][d]))
    if len(day_preds) >= 10:
        day_preds.sort(key=lambda x: -x[0])
        preds_all.extend([x[0] for x in day_preds])
        rets_all.extend([x[1] for x in day_preds])

from scipy.stats import spearmanr
ic, ic_p = spearmanr(preds_all, rets_all)
print(f"   IC: {ic:.4f} (p={ic_p:.2e})")

# 报告
print(f"\n{'='*60}")
print(f"🔙 截面回测结果 (test: {TEST_START} ~ {TEST_END})")
print(f"{'='*60}")
print(f"   Top {TOP_K} 做多 vs Bottom {TOP_K} 做空")
print(f"   交易日: {len(df_results)} 天")
print(f"   平均多空价差: {df_results['spread'].mean()*100:.2f}%/5天")
print(f"   多空胜率: {(df_results['spread']>0).mean()*100:.1f}%")
print(f"   累计多空收益: {(df_results['cum_spread'].iloc[-1]-1)*100:.1f}%")
print(f"   累计多头收益: {(df_results['cum_long'].iloc[-1]-1)*100:.1f}%")
print(f"   累计空头收益: {(df_results['cum_short'].iloc[-1]-1)*100:.1f}%")
print(f"   年化夏普: {df_results['spread'].mean()/df_results['spread'].std()*np.sqrt(250/5):.2f}")
print(f"   最大回撤: {(df_results['cum_spread']/df_results['cum_spread'].cummax()-1).min()*100:.1f}%")
print(f"   IC: {ic:.4f}")
print(f"{'='*60}")