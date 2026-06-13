#!/usr/bin/env python3
"""
日线模型样本外验证 — 真正的时间序列切分

关键修复:
  - 用前80%数据训练模型，后20%数据测试（真正样本外）
  - 不再用 resume 模型直接测，而是重新训练+验证
  - 北向资金数据 shift 1 天避免未来信息

用法:
  python strategy/oos_validate.py
"""

import sys, os, pickle, sqlite3, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategy.train import load_data, compute_features, load_sentiment

warnings.filterwarnings('ignore')

# ============ 配置 ============
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/stock_data.db')
PARAMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_params.json')
HORIZON = 5
TOP_K = 10
CAPITAL = 100000

BASE_CFG = {
    'horizon': HORIZON, 'min_history': 120, 'min_samples': 200,
    'features': 'enhanced+advanced+market',
}

FIXED_PARAMS = {
    'objective': 'regression_l1', 'metric': 'mae',
    'boosting_type': 'gbdt', 'verbosity': -1, 'random_state': 42,
    'force_row_wise': True, 'n_jobs': -1,
}


def load_params():
    with open(PARAMS_FILE) as f:
        return json.load(f).get('daily', {})


def oos_validate():
    import json

    print("=" * 60)
    print(" 日线模型 真正样本外验证")
    print(f" 前80%训练 → 后20%测试 | 预测{HORIZON}日收益率 | Top{TOP_K}")
    print("=" * 60)

    # 1. 加载数据 + 计算特征
    print("\n📊 加载数据 + 计算特征...")
    conn = sqlite3.connect(DB_PATH)
    data = load_data(DB_PATH, 'kline_daily')
    sent_df = load_sentiment(conn)
    has_sent = len(sent_df) > 0

    all_dates = None
    stock_features = {}
    stock_close = {}

    for sym, df in data.items():
        try:
            feats = compute_features(df, sym, BASE_CFG)

            if has_sent:
                dates = df['date'].dt.strftime('%Y-%m-%d')
                sent = sent_df[sent_df['symbol'] == sym].set_index('date')
                for col in sent.columns:
                    if col not in ('symbol', 'date'):
                        feats[f'sent_{col}'] = dates.map(
                            lambda d: sent.loc[d, col] if d in sent.index else 0
                        ).fillna(0).values

            feats = feats.fillna(method='ffill').fillna(0)
            feats.index = df['date'].values
            stock_features[sym] = feats
            stock_close[sym] = df.set_index('date')['close'].astype(float)

            if all_dates is None:
                all_dates = sorted(df['date'].unique())
        except Exception:
            continue

    conn.close()
    print(f"  {len(all_dates)} 个交易日, {len(stock_features)} 只股票")

    # 2. 构建训练/测试数据
    print("\n📦 构建训练/测试集...")
    train_cutoff = all_dates[int(len(all_dates) * 0.8)]
    print(f"  训练截止: {str(train_cutoff)[:10]}")

    X_train, y_train, X_test, y_test = [], [], [], []
    feature_names_ref = None

    for sym in stock_features:
        feats = stock_features[sym]
        close = stock_close[sym]

        if feature_names_ref is None:
            feature_names_ref = list(feats.columns)

        # 对齐特征
        for fn in feature_names_ref:
            if fn not in feats.columns:
                feats[fn] = 0
        feats = feats[feature_names_ref]

        # 目标
        target = np.full(len(close), np.nan)
        for j in range(len(close) - HORIZON):
            target[j] = (close.iloc[j + HORIZON] - close.iloc[j]) / close.iloc[j]

        valid = ~np.isnan(target)
        feats_v = feats[valid].values
        target_v = target[valid]
        dates_v = feats.index[valid]

        if len(feats_v) <= BASE_CFG['min_history'] + 50:
            continue

        # 切分
        train_mask = dates_v <= train_cutoff
        test_mask = dates_v > train_cutoff

        if train_mask.sum() >= 200:
            X_train.append(feats_v[train_mask])
            y_train.append(target_v[train_mask])
        if test_mask.sum() >= 10:
            X_test.append(feats_v[test_mask])
            y_test.append(target_v[test_mask])

    if not X_train or not X_test:
        print("❌ 训练或测试数据不足"); return

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)

    # 过滤极端值
    valid_train = np.abs(y_train) < 0.15
    X_train, y_train = X_train[valid_train], y_train[valid_train]
    valid_test = np.abs(y_test) < 0.15
    X_test, y_test = X_test[valid_test], y_test[valid_test]

    print(f"  训练: {len(X_train):,}条, 测试: {len(X_test):,}条, 特征: {len(feature_names_ref)}")

    # 3. 去冗余
    cm = np.corrcoef(X_train.T)
    rm = set()
    for i in range(len(feature_names_ref)):
        for j in range(i + 1, len(feature_names_ref)):
            if abs(cm[i, j]) > 0.95 and i not in rm and j not in rm:
                rm.add(j)
    if rm:
        keep = np.ones(len(feature_names_ref), dtype=bool)
        keep[list(rm)] = False
        X_train = X_train[:, keep]
        X_test = X_test[:, keep]
        feature_names_ref = [fn for fn, m in zip(feature_names_ref, keep) if m]
        print(f"  去冗余: {len(feature_names_ref)} 特征")

    # 4. 训练模型
    print("\n🏋️ 训练模型...")
    params = load_params()
    for k, v in FIXED_PARAMS.items():
        params.setdefault(k, v)
    params['n_estimators'] = 2000

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    print(f"  实际训练了 {model.best_iteration_ or 2000} 棵树")

    # 5. 评估 — 按日期分组
    print("\n📈 样本外评估...")
    # 找出测试集中的日期
    # 由于数据是按股票拼接的，我们需要按原始顺序评估
    # 简化：直接在全部测试集上算 Rank IC
    pred = model.predict(X_test)
    ic, pval = spearmanr(pred, y_test)
    if np.isnan(ic):
        ic = 0

    rmse = np.sqrt(np.mean((pred - y_test) ** 2))
    mae = np.mean(np.abs(pred - y_test))

    print(f"\n  测试集 Rank IC: {ic:.4f}")
    print(f"  测试集 RMSE: {rmse:.4f}")
    print(f"  测试集 MAE: {mae:.4f}")

    # 6. 模拟选股 (按日期)
    print(f"\n🎯 模拟选股 (每日 Top{TOP_K})...")

    # 重建日期索引
    test_dates = set()
    stocks_per_date = defaultdict(list)
    for sym in stock_features:
        feats = stock_features[sym]
        close = stock_close[sym]
        # 对齐特征
        for fn in feature_names_ref:
            if fn not in feats.columns:
                feats[fn] = 0
        feats = feats[feature_names_ref]

        for i, (date, row) in enumerate(feats.iterrows()):
            if date <= train_cutoff:
                continue
            if i + HORIZON >= len(close):
                continue
            if row.isna().any():
                continue
            pred_val = float(model.predict(row.values.reshape(1, -1))[0])
            actual = (close.iloc[i + HORIZON] - close.iloc[i]) / close.iloc[i]
            if abs(actual) > 0.15:
                continue
            test_dates.add(date)
            stocks_per_date[date].append((sym, pred_val, actual))

    test_dates = sorted(test_dates)
    print(f"  有效测试日: {len(test_dates)}")

    portfolio_value = CAPITAL
    portfolio_values = [CAPITAL]
    benchmark_value = CAPITAL
    benchmark_values = [CAPITAL]

    daily_ics = []

    for date in test_dates:
        stocks = stocks_per_date[date]
        if len(stocks) < TOP_K:
            continue

        preds = np.array([s[1] for s in stocks])
        actuals = np.array([s[2] for s in stocks])

        if len(preds) >= 10:
            ic_day, _ = spearmanr(preds, actuals)
            if not np.isnan(ic_day):
                daily_ics.append(ic_day)

        top_idx = np.argsort(preds)[-TOP_K:][::-1]
        top_ret = np.mean(actuals[top_idx])
        bm_ret = np.mean(actuals)

        trade_amount = portfolio_value * 0.95
        new_pf = sum(trade_amount / TOP_K * (1 + r) for r in actuals[top_idx]) + portfolio_value * 0.05
        new_bm = benchmark_value * (1 + bm_ret)

        portfolio_values.append(new_pf)
        portfolio_value = new_pf
        benchmark_values.append(new_bm)
        benchmark_value = new_bm

    # 7. 结果
    print("\n" + "=" * 60)
    print(" 📊 样本外结果")
    print("=" * 60)

    if not daily_ics:
        print("❌ 无有效数据"); return

    ic_mean = np.mean(daily_ics)
    ic_std = np.std(daily_ics)
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0
    ic_pos = sum(1 for x in daily_ics if x > 0) / len(daily_ics)

    strat_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    bm_returns = np.diff(benchmark_values) / benchmark_values[:-1]

    def sharpe(r):
        return np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0

    def max_dd(vals):
        peak = np.maximum.accumulate(vals)
        return np.min((vals - peak) / peak)

    print(f"\n  Rank IC (日频):")
    print(f"    均值: {ic_mean:.4f}  标准差: {ic_std:.4f}  IR: {ic_ir:.2f}")
    print(f"    正值率: {ic_pos:.1%}  ({len(daily_ics)}天)")

    print(f"\n  策略 vs 基准:")
    print(f"    策略累计收益: {(portfolio_value/CAPITAL - 1)*100:.2f}%")
    print(f"    基准累计收益: {(benchmark_value/CAPITAL - 1)*100:.2f}%")
    print(f"    超额收益:     {(portfolio_value/CAPITAL - benchmark_value/CAPITAL)*100:.2f}%")
    print(f"    策略 Sharpe:  {sharpe(strat_returns):.3f}")
    print(f"    基准 Sharpe:  {sharpe(bm_returns):.3f}")
    print(f"    策略 最大回撤: {max_dd(portfolio_values)*100:.2f}%")
    print(f"    基准 最大回撤: {max_dd(benchmark_values)*100:.2f}%")

    print(f"\n  📋 结论:")
    if ic_mean > 0.05 and portfolio_value > benchmark_value:
        print(f"  ✅ 样本外有效: IC={ic_mean:.4f}({ic_pos:.0%}正), 超额{(portfolio_value/CAPITAL - benchmark_value/CAPITAL)*100:.2f}%")
    elif ic_mean > 0.025:
        print(f"  ⚠️ 弱有效: IC={ic_mean:.4f}, 需优化")
    else:
        print(f"  ❌ 过拟合/失效: IC={ic_mean:.4f}")


if __name__ == '__main__':
    from collections import defaultdict
    oos_validate()