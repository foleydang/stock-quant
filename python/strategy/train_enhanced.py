#!/usr/bin/env python3
"""
增强版模型训练脚本 — 多特征 + 多模型 + 市场状态分治

新增功能:
  2. 特征增强: 板块轮动、北向资金、龙虎榜、融资融券、基本面
  3. 多模型集成: LGBM + XGBoost + CatBoost 加权ensemble
  4. 市场状态: 牛/熊/震荡分别训练，预测时自动识别

用法:
  python strategy/train_enhanced.py                    # 完整训练
  python strategy/train_enhanced.py --quick             # 快速验证(2模型,1000树)
  python strategy/train_enhanced.py --no-regime         # 不分区市场状态
  python strategy/train_enhanced.py --no-ensemble       # 不集成，只训练LGBM
"""

import os, sys, json, pickle, time, warnings, argparse, gc
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

# === 路径配置 ===
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'python'))

DB_PATH = os.path.join(ROOT, 'data', 'stock_data.db')
MODEL_DIR = os.path.join(ROOT, '..', 'models', 'lgb_hs300_enhanced')
os.makedirs(MODEL_DIR, exist_ok=True)

# === 模型参数 (V8 网格搜索最优: IC 0.0416→0.0664) ===
LGBM_PARAMS = {
    'n_estimators': 2000, 'learning_rate': 0.005, 'num_leaves': 31,
    'max_depth': 6, 'min_child_samples': 300, 'subsample': 0.3,
    'subsample_freq': 1, 'colsample_bytree': 0.2, 'feature_fraction_bynode': 0.6,
    'reg_alpha': 1.0, 'reg_lambda': 10.0, 'min_split_gain': 0.05,
    'path_smooth': 15, 'random_state': 42, 'n_jobs': 1, 'verbosity': -1
}

XGB_PARAMS = {
    'n_estimators': 2000, 'learning_rate': 0.005, 'max_depth': 6,
    'min_child_weight': 3, 'subsample': 0.3, 'colsample_bytree': 0.2,
    'reg_alpha': 1.0, 'reg_lambda': 10.0, 'gamma': 0.05,
    'random_state': 42, 'n_jobs': 1, 'verbosity': 0
}

CATBOOST_PARAMS = {
    'iterations': 2000, 'learning_rate': 0.005, 'depth': 6,
    'l2_leaf_reg': 10, 'random_seed': 42, 'verbose': 0,
    'thread_count': 1, 'subsample': 0.3, 'rsm': 0.2
}

QUICK_PARAMS = {
    'n_estimators': 500, 'learning_rate': 0.05,
    'num_leaves': 31, 'max_depth': 4
}

# === 数据加载 ===
def load_db():
    return sqlite3.connect(DB_PATH)

def load_kline_data(conn, max_stocks=0):
    """加载所有A股的30分钟K线，按日聚合"""
    print("📊 加载K线数据...")
    df = pd.read_sql("""
        SELECT t.symbol, t.trade_date,
               first_bar.open as open, t.high, t.low,
               last_bar.close as close, t.volume
        FROM (
            SELECT symbol, substr(date,1,10) as trade_date,
                   MAX(high) as high, MIN(low) as low,
                   SUM(volume) as volume,
                   MIN(date) as first_date, MAX(date) as last_date
            FROM kline_30m
            WHERE (symbol LIKE '%.SZ' OR symbol LIKE '%.SH')
            GROUP BY symbol, substr(date,1,10)
        ) t
        JOIN kline_30m first_bar ON first_bar.symbol = t.symbol AND first_bar.date = t.first_date
        JOIN kline_30m last_bar ON last_bar.symbol = t.symbol AND last_bar.date = t.last_date
        ORDER BY t.symbol, t.trade_date
    """, conn)
    if max_stocks > 0:
        top_symbols = pd.read_sql("SELECT symbol FROM (SELECT symbol, COUNT(*) as cnt FROM kline_30m WHERE (symbol LIKE '%.SZ' OR symbol LIKE '%.SH') GROUP BY symbol ORDER BY cnt DESC LIMIT " + str(max_stocks) + ")", conn)['symbol'].tolist()
        df = df[df['symbol'].isin(top_symbols)]
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df

def load_sector_data(conn):
    """加载板块映射"""
    print("📊 加载板块数据...")
    df = pd.read_sql("SELECT symbol, industry FROM stock_sector", conn)
    return df.set_index('symbol')['industry'].to_dict()

def load_north_flow(conn):
    """加载北向资金"""
    print("📊 加载北向资金...")
    df = pd.read_sql("""
        SELECT trade_date, north_net, total_net, total_buy, total_sell
        FROM north_flow ORDER BY trade_date
    """, conn)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df

def load_sentiment(conn):
    """加载情绪数据(龙虎榜+融资融券 从 sentiment_daily)"""
    print("📊 加载情绪数据...")
    df = pd.read_sql("""
        SELECT symbol, trade_date, lhb_flag, lhb_net_buy, lhb_net_buy_ratio,
               lhb_ret_5d, margin_balance_chg, short_balance
        FROM sentiment_daily ORDER BY trade_date
    """, conn)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df

def load_fundamental(conn):
    """加载基本面数据"""
    print("📊 加载基本面数据...")
    df = pd.read_sql("""
        SELECT symbol, trade_date, roe, debt_ratio, net_profit_yoy, revenue_yoy
        FROM fundamental_daily ORDER BY trade_date
    """, conn)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df

def load_sentiment_daily(conn):
    """加载每日情绪指标"""
    df = pd.read_sql("""
        SELECT symbol, trade_date, is_limit_up, is_limit_down,
               consecutive_limit_up, vol_ratio_20, abnormal_ret
        FROM sentiment_daily ORDER BY trade_date
    """, conn)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df


# === 特征工程 ===
def calculate_features(df, conn):
    """计算增强特征: V8 FeaturePipeline (254特征) + 情绪/基本面/北向/板块"""
    print("🔧 计算特征 (V8 FeaturePipeline + 增强特征)...")

    from strategy.features import FeaturePipeline
    pipeline = FeaturePipeline({'label': '日线', 'horizon': 3, 'db_table': 'kline_daily',
                                'min_history': 120, 'purged_gap': 3, 'north_shift_days': 1})

    results = {}
    symbols = sorted(df['symbol'].unique())

    # 加载辅助数据
    sector_map = load_sector_data(conn)
    north_flow = load_north_flow(conn)
    sentiment = load_sentiment(conn)
    fundamental = load_fundamental(conn)
    sent_daily = load_sentiment_daily(conn)

    for sym in symbols:
        stock_df = df[df['symbol'] == sym].copy().sort_values('trade_date')
        if len(stock_df) < 200:
            continue

        stock_df = stock_df.reset_index(drop=True)

        # 准备 FeaturePipeline 需要的格式 (列名 date, 含 datetime)
        fp_df = stock_df.rename(columns={'trade_date': 'date'}).copy()
        fp_df['date'] = pd.to_datetime(fp_df['date'])

        try:
            # V8 FeaturePipeline: 254+ 特征
            feats = pipeline.compute_stock(fp_df, sym)
            feats = feats.ffill().fillna(0)
            feats.index = stock_df.index  # 对齐回 stock_df
        except Exception as e:
            if len(results) == 0:
                print(f"  ⚠️ FeaturePipeline 失败 {sym}: {e}, 跳过")
            continue

        # === V9 独有增强特征 (叠加在 V8 之上) ===
        close = stock_df['close'].values
        volume = stock_df['volume'].values

        # 情绪特征
        sent_stock = sent_daily[sent_daily['symbol'] == sym].copy()
        if len(sent_stock) > 0:
            sent_stock = sent_stock.set_index('trade_date')
            for col in ['is_limit_up', 'is_limit_down', 'consecutive_limit_up', 'vol_ratio_20', 'abnormal_ret']:
                if col in sent_stock.columns:
                    mapped = stock_df['trade_date'].map(
                        lambda d: sent_stock.loc[d, col] if d in sent_stock.index else 0
                    ).fillna(0).values
                    feats[f'sent_{col}'] = mapped

        # 龙虎榜 + 融资融券
        sent_stock2 = sentiment[sentiment['symbol'] == sym].copy()
        if len(sent_stock2) > 0:
            sent_stock2 = sent_stock2.set_index('trade_date')
            for col in ['lhb_net_buy', 'lhb_ret_5d', 'lhb_net_buy_ratio',
                        'margin_balance_chg', 'short_balance']:
                if col in sent_stock2.columns:
                    feats[f'extra_{col}'] = stock_df['trade_date'].map(
                        lambda d: sent_stock2.loc[d, col] if d in sent_stock2.index else 0
                    ).fillna(0).values

        # 基本面
        fund_stock = fundamental[fundamental['symbol'] == sym].copy()
        if len(fund_stock) > 0:
            fund_stock = fund_stock.set_index('trade_date')
            for col in ['roe', 'debt_ratio', 'net_profit_yoy', 'revenue_yoy']:
                if col in fund_stock.columns:
                    feats[f'extra_fund_{col}'] = stock_df['trade_date'].map(
                        lambda d: fund_stock.loc[d, col] if d in fund_stock.index else 0
                    ).fillna(0).values

        # 北向资金
        if len(north_flow) > 0:
            nf = north_flow.set_index('trade_date')
            for col in ['north_net', 'total_net']:
                feats[f'extra_north_{col}'] = stock_df['trade_date'].map(
                    lambda d: nf.loc[d, col] if d in nf.index else 0
                ).fillna(0).values

        # 板块
        industry = sector_map.get(sym, '未知')
        feats['sector_code'] = hash(industry) % 100

        results[sym] = (stock_df, feats)
        if len(results) % 20 == 0:
            gc.collect()

    gc.collect()
    print(f"   特征计算完成: {len(results)} 只股票, 每只约 {len(list(results.values())[0][1].columns)} 特征")

    # === 截面排名特征 (对关键指标做截面排名) ===
    print("   计算截面排名特征...")
    cs_candidates = ['price_ret_1', 'price_ret_5', 'price_ret_20', 'price_vol_20',
                     'price_ma20_ratio', 'vol_ratio_20', 'price_rsi_14', 'price_adx',
                     'price_bb20_width', 'price_parkinson_vol', 'price_macd_hist',
                     'price_kdj_j', 'price_cci', 'price_atr_ratio']
    available_cs = [t for t in cs_candidates
                    if any(t in r[1].columns for r in results.values())]

    pieces = []
    for sym, (stock_df, feats) in results.items():
        cols = feats.columns.intersection(available_cs)
        if len(cols) == 0:
            continue
        sub = feats[cols].copy()
        sub = sub.loc[~sub.isna().all(axis=1)]
        if sub.empty:
            continue
        sub['_sym'] = sym
        sub['_date'] = stock_df['trade_date'].values
        pieces.append(sub)

    if pieces:
        big = pd.concat(pieces).set_index('_date')
        for t in available_cs:
            if t not in big.columns:
                continue
            big[f'cs_rank_{t}'] = big[t].groupby(level='_date').rank(pct=True)

        for sym, (stock_df, feats) in results.items():
            sub = big[big['_sym'] == sym].drop(columns='_sym')
            sub = sub.reindex(stock_df['trade_date'].values)
            for col in sub.columns:
                if col.startswith('cs_rank_'):
                    feats[col] = sub[col].values

    gc.collect()
    return results


def _calc_adx(high, low, close, period=14):
    """计算ADX趋势强度"""
    n = len(close)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        plus_dm[i] = up if up > down and up > 0 else 0
        minus_dm[i] = down if down > up and down > 0 else 0
    
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / (atr + 1e-9)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / (atr + 1e-9)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    adx = pd.Series(dx).ewm(span=period, adjust=False).mean()
    return adx.values


# === 市场状态识别 ===
def detect_market_regime(df, window=60):
    """识别市场状态: bull(牛市), bear(熊市), sideways(震荡)"""
    ma20 = df['close'].rolling(20).mean()
    ma60 = df['close'].rolling(60).mean()
    returns = df['close'].pct_change(window)
    
    regimes = np.full(len(df), 'sideways', dtype=object)
    
    # 牛市: 价格 > MA60 且 60日收益 > 5%
    bull_mask = (df['close'] > ma60) & (returns > 0.05)
    regimes[bull_mask] = 'bull'
    
    # 熊市: 价格 < MA60 且 60日收益 < -5%
    bear_mask = (df['close'] < ma60) & (returns < -0.05)
    regimes[bear_mask] = 'bear'
    
    # 震荡: 其余情况
    return regimes


# === 模型训练 (Bagging: 3 seeds × 3 模型类型 = 9 模型/regime) ===
BAG_SEEDS = [42, 123, 456]

def train_models(X_train, y_train, X_val, y_val, feature_names, quick=False):
    """训练 LGBM + XGBoost + CatBoost (各3个seed, 共9模型)"""
    models = {}   # {model_id: model}
    cv_scores = {}  # {model_id: ic}

    if quick:
        lgb_params = {**LGBM_PARAMS, **QUICK_PARAMS}
        xgb_params = {**XGB_PARAMS, **QUICK_PARAMS}
        cb_params = {**CATBOOST_PARAMS, 'iterations': 500}
    else:
        lgb_params = LGBM_PARAMS.copy()
        xgb_params = XGB_PARAMS.copy()
        cb_params = CATBOOST_PARAMS.copy()

    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor

    # LGBM bagging
    print(f"  🌳 训练 LGBM (3 seeds)...")
    for seed in BAG_SEEDS:
        p = {**lgb_params, 'random_state': seed}
        m = lgb.LGBMRegressor(**p)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              eval_metric='l1', callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        pred = m.predict(X_val)
        ic = np.corrcoef(pred, y_val)[0, 1] if len(pred) > 1 else 0
        mid = f'lgbm_{seed}'
        models[mid] = m
        cv_scores[mid] = ic
        print(f"     {mid}: IC={ic:.4f}")

    # XGBoost bagging
    print(f"  🌲 训练 XGBoost (3 seeds)...")
    for seed in BAG_SEEDS:
        p = {**xgb_params, 'random_state': seed}
        m = xgb.XGBRegressor(**p, early_stopping_rounds=50)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pred = m.predict(X_val)
        ic = np.corrcoef(pred, y_val)[0, 1] if len(pred) > 1 else 0
        mid = f'xgb_{seed}'
        models[mid] = m
        cv_scores[mid] = ic
        print(f"     {mid}: IC={ic:.4f}")

    # CatBoost bagging
    print(f"  🐱 训练 CatBoost (3 seeds)...")
    for seed in BAG_SEEDS:
        p = {**cb_params, 'random_seed': seed}
        m = CatBoostRegressor(**p)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              early_stopping_rounds=50, verbose=False)
        pred = m.predict(X_val)
        ic = np.corrcoef(pred, y_val)[0, 1] if len(pred) > 1 else 0
        mid = f'cb_{seed}'
        models[mid] = m
        cv_scores[mid] = ic
        print(f"     {mid}: IC={ic:.4f}")

    return models, cv_scores


def ensemble_weight(models, cv_scores):
    """按验证集IC加权, 归一化到总和=1"""
    total = sum(max(0, s) for s in cv_scores.values())
    if total == 0:
        return {k: 1.0/len(cv_scores) for k in cv_scores}
    weights = {k: max(0, s) / total for k, s in cv_scores.items()}
    # 归一化 (确保总和=1)
    w_sum = sum(weights.values())
    return {k: v/w_sum for k, v in weights.items()} if w_sum > 0 else weights


# === 主流程 ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='快速验证模式')
    parser.add_argument('--no-regime', action='store_true', help='不分区市场状态')
    parser.add_argument('--no-ensemble', action='store_true', help='不集成，只训练LGBM')
    parser.add_argument('--horizon', type=int, default=3, help='预测周期')
    parser.add_argument('--db', default=DB_PATH, help='数据库路径')
    parser.add_argument('--max-stocks', type=int, default=0, help='最大股票数')
    args = parser.parse_args()
    
    conn = sqlite3.connect(args.db)
    
    print("=" * 60)
    print("🚀 增强版模型训练")
    print(f"   预测周期: h={args.horizon}")
    print(f"   市场分治: {'否' if args.no_regime else '是'}")
    print(f"   多模型: {'否' if args.no_ensemble else '是'}")
    print(f"   快速模式: {'是' if args.quick else '否'}")
    print("=" * 60)
    
    # 1. 加载数据
    t0 = time.time()
    df = load_kline_data(conn, args.max_stocks)
    print(f"   共 {len(df['symbol'].unique())} 只股票, {len(df)} 条日线")
    
    # 2. 计算特征
    results = calculate_features(df, conn)
    print(f"   特征计算完成: {len(results)} 只股票")
    
    # 3. 准备训练数据
    print("\n📦 准备训练数据...")
    X_all, y_all, regimes_all, dates_all = [], [], [], []

    # 第一遍: 收集所有特征列的并集
    all_features = set()
    for sym, (stock_df, feats) in results.items():
        all_features.update(feats.columns)
    all_features = sorted(all_features)
    print(f"   统一特征数: {len(all_features)}")

    # 第一遍: 计算每只股票的绝对收益, 后面转截面排名
    stock_returns_by_date = {}  # {date: {symbol: return}}
    stock_data_dict = {}

    for sym, (stock_df, feats) in results.items():
        close = stock_df['close'].values
        target = np.full(len(close), np.nan)
        for i in range(len(close) - args.horizon):
            target[i] = (close[i + args.horizon] - close[i]) / (close[i] + 1e-9)

        dates = stock_df['trade_date'].values
        for i in range(len(close)):
            if not np.isnan(target[i]):
                d = pd.Timestamp(dates[i]).strftime('%Y-%m-%d')
                if d not in stock_returns_by_date:
                    stock_returns_by_date[d] = {}
                stock_returns_by_date[d][sym] = target[i]
        stock_data_dict[sym] = (stock_df, feats, target)

    # 第二遍: 截面排名 → 市场中性目标 (中心化到 -0.5 ~ 0.5)
    print("   计算截面排名目标 (市场中性)...")
    rank_targets = {}  # {sym: {date_str: rank_pct}}
    for sym in stock_data_dict:
        rank_targets[sym] = {}
    n_dates_ranked = 0
    for d, sym_rets in stock_returns_by_date.items():
        if len(sym_rets) < 10:
            continue
        rets = np.array(list(sym_rets.values()))
        market_ret = np.nanmean(rets)
        alpha_rets = rets - market_ret  # 去市场 beta
        ranks = pd.Series(alpha_rets).rank(pct=True).values - 0.5
        for i, sym in enumerate(sym_rets.keys()):
            rank_targets[sym][d] = float(ranks[i])
        n_dates_ranked += 1
    print(f"   截面排名完成: {n_dates_ranked} 天, {len(stock_data_dict)} 只股票")

    for sym, (stock_df, feats, abs_target) in stock_data_dict.items():
        # 目标: 截面排名分位 (市场中性)
        date_strs = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in stock_df['trade_date'].values]
        target = np.array([rank_targets[sym].get(d, np.nan) for d in date_strs])

        # 市场状态
        regime = detect_market_regime(stock_df)

        # 对齐到统一特征列 (缺失的填0)
        feats = feats.fillna(0).replace([np.inf, -np.inf], 0)
        for c in all_features:
            if c not in feats.columns:
                feats[c] = 0
        feats = feats[all_features]

        valid = ~np.isnan(target)
        valid[:120] = False  # 前120条作为历史

        X_all.append(feats[valid].values)
        y_all.append(target[valid])
        regimes_all.append(regime[valid])
        dates_all.append(stock_df['trade_date'].values[valid])

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    regimes_all = np.concatenate(regimes_all)
    dates_all = np.concatenate(dates_all)

    # 去除极端值 (排名目标已经在 [-0.5, 0.5], 不需要 clip)
    valid_mask = np.abs(y_all) < 0.5
    X_all = X_all[valid_mask]
    y_all = y_all[valid_mask]
    regimes_all = regimes_all[valid_mask]
    dates_all = dates_all[valid_mask]

    print(f"   总样本: {len(X_all)}")
    print(f"   特征数: {len(all_features)}")
    print(f"   牛市: {(regimes_all == 'bull').sum()}")
    print(f"   熊市: {(regimes_all == 'bear').sum()}")
    print(f"   震荡: {(regimes_all == 'sideways').sum()}")

    # 4. 训练/验证分割 (按时间切分)
    train_cutoff = np.percentile(dates_all.astype('datetime64[D]').astype(int), 80)
    train_cutoff = np.datetime64(int(train_cutoff), 'D')
    # purged gap: drop samples in [train_cutoff - horizon, train_cutoff) to avoid target leakage
    horizon = args.horizon
    purge_start = train_cutoff - np.timedelta64(horizon + 2, 'D')
    train_mask = dates_all < purge_start
    val_mask = dates_all >= train_cutoff

    X_train, X_val = X_all[train_mask], X_all[val_mask]
    y_train, y_val = y_all[train_mask], y_all[val_mask]
    r_train, r_val = regimes_all[train_mask], regimes_all[val_mask]

    n_purged = ((dates_all >= purge_start) & (dates_all < train_cutoff)).sum()
    print(f"\n  ⏰ 时间切分: train < {purge_start} | purge gap {n_purged} 条 | val >= {train_cutoff}")
    print(f"     train: {train_mask.sum()} 条 | val: {val_mask.sum()} 条")
    
    # 5. 训练模型
    print("\n🎯 训练模型...")
    all_models = {}
    all_cv_scores = {}
    all_weights = {}
    
    if args.no_regime:
        # 不分区市场状态
        models, cv_scores = train_models(X_train, y_train, X_val, y_val, all_features, args.quick)
        all_models['all'] = models
        all_cv_scores['all'] = cv_scores
        all_weights['all'] = ensemble_weight(models, cv_scores)
    else:
        # 按市场状态分别训练
        for regime in ['bull', 'bear', 'sideways']:
            mask_train = r_train == regime
            mask_val = r_val == regime
            
            if mask_train.sum() < 1000 or mask_val.sum() < 200:
                print(f"\n  ⚠️ {regime} 样本不足, 使用全部数据")
                models, cv_scores = train_models(X_train, y_train, X_val, y_val, all_features, args.quick)
            else:
                print(f"\n  📈 {regime} 市场 (train={mask_train.sum()}, val={mask_val.sum()})")
                models, cv_scores = train_models(
                    X_train[mask_train], y_train[mask_train],
                    X_val[mask_val], y_val[mask_val], all_features, args.quick
                )
            
            all_models[regime] = models
            all_cv_scores[regime] = cv_scores
            all_weights[regime] = ensemble_weight(models, cv_scores)
    
    # 6. 评估
    print("\n📊 评估结果...")
    
    if args.no_regime:
        for name, model in all_models['all'].items():
            pred = model.predict(X_val)
            ic = np.corrcoef(pred, y_val)[0, 1]
            mae = np.mean(np.abs(pred - y_val))
            print(f"  {name:10s}: IC={ic:.4f} MAE={mae:.4f} weight={all_weights['all'].get(name, 0):.3f}")
        
        # Ensemble
        ensemble_pred = np.zeros(len(y_val))
        for name, model in all_models['all'].items():
            ensemble_pred += model.predict(X_val) * all_weights['all'].get(name, 0)
        ic = np.corrcoef(ensemble_pred, y_val)[0, 1]
        mae = np.mean(np.abs(ensemble_pred - y_val))
        print(f"  {'Ensemble':10s}: IC={ic:.4f} MAE={mae:.4f}")
    else:
        for regime in ['bull', 'bear', 'sideways']:
            mask = r_val == regime
            if mask.sum() < 10:
                continue
            print(f"\n  [{regime}] val={mask.sum()}")

            # 各模型类型平均 IC
            for mtype in ['lgbm', 'xgb', 'cb']:
                type_ics = []
                for name, model in all_models[regime].items():
                    if name.startswith(mtype):
                        pred = model.predict(X_val[mask])
                        type_ics.append(np.corrcoef(pred, y_val[mask])[0, 1])
                if type_ics:
                    print(f"    {mtype:10s}: avg IC={np.mean(type_ics):.4f} ({len(type_ics)}模型)")

            # Ensemble
            ensemble_pred = np.zeros(mask.sum())
            total_w = 0
            for name, model in all_models[regime].items():
                w = all_weights[regime].get(name, 0)
                ensemble_pred += model.predict(X_val[mask]) * w
                total_w += w
            if total_w > 0:
                ensemble_pred /= total_w
            ic = np.corrcoef(ensemble_pred, y_val[mask])[0, 1]
            mae = np.mean(np.abs(ensemble_pred - y_val[mask]))
            print(f"    {'Ensemble':10s}: IC={ic:.4f} MAE={mae:.4f}")
    
    # 7. 保存模型
    print(f"\n💾 保存模型到 {MODEL_DIR}...")
    
    model_data = {
        'model_version': 'v9-enhanced',
        'model_type': 'regime_ensemble' if not args.no_regime else 'ensemble',
        'horizon': args.horizon,
        'feature_names': all_features,
        'n_features': len(all_features),
        'regime_models': all_models,
        'regime_weights': all_weights,
        'cv_scores': all_cv_scores,
        'trained_at': datetime.now().isoformat(),
        'market_regimes': not args.no_regime,
        'n_stocks': len(results),
        'n_samples': len(X_all),
    }
    
    # 兼容旧版预测API: 添加 models 列表 (取所有 bagging 模型)
    if not args.no_regime and not args.no_ensemble:
        # 把所有 regime 的所有模型展平成 models 列表
        flat_models = []
        for regime, models in all_models.items():
            flat_models.extend(models.values())
        model_data['models'] = flat_models
        model_data['model_types'] = ['lgbm', 'xgb', 'catboost']
    elif not args.no_ensemble:
        model_data['models'] = list(all_models['all'].values())
        model_data['model_types'] = list(all_models['all'].keys())
    else:
        # 取第一个模型
        first_model = list(all_models['all'].values())[0]
        model_data['model'] = first_model
    
    model_path = os.path.join(MODEL_DIR, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    # 保存特征名
    with open(os.path.join(MODEL_DIR, 'feature_names.json'), 'w') as f:
        json.dump(all_features, f)
    
    elapsed = time.time() - t0
    print(f"\n✅ 训练完成! 耗时 {elapsed:.0f}s")
    print(f"   模型路径: {model_path}")
    print(f"   特征数: {len(all_features)}")
    print(f"   样本数: {len(X_all)}")


if __name__ == '__main__':
    main()