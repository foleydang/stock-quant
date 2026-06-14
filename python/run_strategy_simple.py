#!/usr/bin/env python3
"""精简版策略执行"""
import os
import sys
import sqlite3
import pickle
import pandas as pd
from datetime import datetime

# 路径（动态获取）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'data/stock_data.db')
MODEL_PATH = os.path.join(BASE_DIR, 'models/lgb_hs300/model.pkl')

print("=" * 60)
print(f"LGBM策略执行 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# 1. 加载模型
print("\n[1] 加载模型...")
with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)
model = model_data.get('model')
print(f"    模型F1: {model_data.get('cv_f1', 0):.2%}")

# 2. 获取股票列表
print("\n[2] 获取股票池...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute('''
    SELECT symbol, COUNT(*) as cnt
    FROM kline_30m
    GROUP BY symbol
    HAVING cnt >= 500
    ORDER BY cnt DESC
    LIMIT 30
''')
stocks = cursor.fetchall()
print(f"    共 {len(stocks)} 只股票")

# 3. 导入特征工程
sys.path.insert(0, BASE_DIR)
from strategy.features import EnhancedFeatureEngineer

# 4. 回测
print("\n[3] 执行回测...")
results = []

for idx, (symbol, cnt) in enumerate(stocks):
    print(f"    [{idx+1}/{len(stocks)}] {symbol}...", end=" ")

    # 加载数据
    df = pd.read_sql_query('''
        SELECT date, open, high, low, close, volume
        FROM kline_30m WHERE symbol = ? ORDER BY date
    ''', conn, params=(symbol,))

    if len(df) < 500:
        print("数据不足")
        continue

    df['date'] = pd.to_datetime(df['date'])

    # 回测参数
    cash = 100000
    holding = 0
    cost = 0
    trades_list = []
    wins = 0

    # 每隔8个bar检查一次(4小时)
    for i in range(150, len(df), 8):
        price = float(df['close'].iloc[i])

        try:
            features = EnhancedFeatureEngineer.calculate_features(df.iloc[:i+1])
            if features.iloc[-1].isna().any():
                continue
            pred_ret = model.predict([features.iloc[-1].values])[0]
        except:
            continue

        # 买入
        if holding == 0 and pred_ret > 0.01:
            shares = int(cash * 0.9 / price / 100) * 100
            if shares >= 100:
                holding = shares
                cash -= shares * price
                cost = price
                trades_list.append({'type': 'BUY', 'price': price, 'shares': shares})

        # 卖出
        elif holding > 0 and (pred_ret < -0.01 or (price - cost) / cost > 0.10 or (price - cost) / cost < -0.08):
            cash += holding * price
            profit = (price - cost) * holding
            if profit > 0:
                wins += 1
            trades_list.append({'type': 'SELL', 'price': price, 'shares': holding, 'profit': profit})
            holding = 0

    # 平仓
    if holding > 0:
        cash += holding * float(df['close'].iloc[-1])
        if float(df['close'].iloc[-1]) > cost:
            wins += 1

    profit_rate = (cash - 100000) / 100000 * 100
    win_rate = wins / max(len([t for t in trades_list if t['type'] == 'SELL']), 1) * 100

    results.append({
        'symbol': symbol,
        'profit_rate': profit_rate,
        'win_rate': win_rate,
        'trades': len(trades_list),
        'final_value': cash
    })

    print(f"收益率 {profit_rate:.2f}%, 胜率 {win_rate:.1f}%")

conn.close()

# 5. 排序输出
results.sort(key=lambda x: x['profit_rate'], reverse=True)

print("\n" + "=" * 60)
print("选股结果 (按收益率排序)")
print("=" * 60)
print(f"{'排名':<6}{'股票':<15}{'收益率':<12}{'胜率':<10}{'交易次数':<10}")
print("-" * 60)
for i, r in enumerate(results[:10], 1):
    print(f"#{i:<5}{r['symbol']:<15}{r['profit_rate']:>8.2f}%{r['win_rate']:>10.1f}%{r['trades']:>10}次")

print("\n" + "=" * 60)
print("策略汇总")
print("=" * 60)
best = results[0]
print(f"最佳股票: {best['symbol']}")
print(f"收益率: {best['profit_rate']:.2f}%")
print(f"最终市值: ¥{best['final_value']:,.0f}")
print(f"盈亏: ¥{best['final_value'] - 100000:,.0f}")

# 保存结果
import json
result_file = f'{BASE_DIR}/logs/strategy_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
os.makedirs(os.path.dirname(result_file), exist_ok=True)
with open(result_file, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存: {result_file}")