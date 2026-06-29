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

DB_PATH = os.path.join(ROOT, 'python', 'data', 'stock_data.db')
MODEL_DIR = os.path.join(ROOT, 'python', 'models', 'lgb_hs300_enhanced')
os.makedirs(MODEL_DIR, exist_ok=True)

# === 模型参数 ===
LGBM_PARAMS = {
    'n_estimators': 2000, 'learning_rate': 0.03, 'num_leaves': 63,
    'max_depth': 6, 'min_child_samples': 50, 'subsample': 0.7,
    'colsample_bytree': 0.6, 'reg_alpha': 0.01, 'reg_lambda': 0.1,
    'random_state': 42, 'n_jobs': 1, 'verbosity': -1
}

XGB_PARAMS = {
    'n_estimators': 2000, 'learning_rate': 0.03, 'max_depth': 6,
    'min_child_weight': 3, 'subsample': 0.7, 'colsample_bytree': 0.6,
    'reg_alpha': 0.01, 'reg_lambda': 0.1, 'random_state': 42,
    'n_jobs': 1, 'verbosity': 0
}

CATBOOST_PARAMS = {
    'iterations': 2000, 'learning_rate': 0.03, 'depth': 6,
    'l2_leaf_reg': 3, 'random_seed': 42, 'verbose': 0,
    'thread_count': 1
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
        SELECT symbol, substr(date,1,10) as trade_date,
               MIN(open) as open, MAX(high) as high, MIN(low) as low,
               MAX(close) as close, SUM(volume) as volume
        FROM kline_30m
        WHERE (symbol LIKE '%.SZ' OR symbol LIKE '%.SH')
        GROUP BY symbol, substr(date,1,10)
        ORDER BY symbol, trade_date
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
    """计算增强特征：价格 + 成交量 + 情绪 + 板块 + 资金流"""
    print("🔧 计算特征...")
    
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
        feats = pd.DataFrame(index=stock_df.index)
        
        # === 价格特征 ===
        close = stock_df['close'].values
        high = stock_df['high'].values
        low = stock_df['low'].values
        volume = stock_df['volume'].values
        returns = np.diff(close, prepend=close[0]) / (close + 1e-9)
        returns[0] = 0
        
        # 收益率
        for p in [1, 2, 3, 5, 10, 20, 30, 60]:
            feats[f'return_{p}'] = pd.Series(close).pct_change(p).values
        
        # 波动率
        for p in [5, 10, 20, 30, 60]:
            feats[f'volatility_{p}'] = pd.Series(returns).rolling(p).std().values
        
        # MA比率
        for p in [5, 10, 20, 60, 120]:
            ma = pd.Series(close).rolling(p).mean()
            feats[f'ma{p}_ratio'] = close / (ma.values + 1e-9)
        
        # MA交叉
        feats['ma20_ma60'] = (pd.Series(close).rolling(20).mean() / 
                              (pd.Series(close).rolling(60).mean() + 1e-9)).values
        feats['ma60_ma120'] = (pd.Series(close).rolling(60).mean() / 
                               (pd.Series(close).rolling(120).mean() + 1e-9)).values
        
        # RSI
        for p in [14, 24, 50]:
            delta = pd.Series(close).diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(p).mean()
            avg_loss = loss.rolling(p).mean()
            rs = avg_gain / (avg_loss + 1e-9)
            feats[f'rsi_{p}'] = (100 - 100 / (1 + rs)).values
        
        # MACD
        ema12 = pd.Series(close).ewm(span=12).mean()
        ema26 = pd.Series(close).ewm(span=26).mean()
        feats['macd'] = (ema12 - ema26).values
        feats['macd_signal'] = feats['macd'].ewm(span=9).mean().values
        feats['macd_hist'] = (feats['macd'] - feats['macd_signal']).values
        
        # 布林带
        for p in [20, 30]:
            ma = pd.Series(close).rolling(p).mean()
            std = pd.Series(close).rolling(p).std()
            feats[f'bb_upper_{p}'] = (ma + 2 * std).values
            feats[f'bb_width_{p}'] = (4 * std / (ma + 1e-9)).values
        
        # ATR
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        feats['atr_10'] = pd.Series(tr).rolling(10).mean().values
        
        # 成交量特征
        feats['volume_ma5'] = volume / (pd.Series(volume).rolling(5).mean() + 1e-9)
        feats['volume_ratio_5'] = volume / (pd.Series(volume).rolling(5).mean().shift(1) + 1e-9)
        feats['volume_ratio_60'] = volume / (pd.Series(volume).rolling(60).mean() + 1e-9)
        
        # 影线
        body = np.abs(close - stock_df['open'].values)
        upper_shadow = high - np.maximum(close, stock_df['open'].values)
        lower_shadow = np.minimum(close, stock_df['open'].values) - low
        feats['upper_shadow'] = upper_shadow / (high - low + 1e-9)
        feats['lower_shadow'] = lower_shadow / (high - low + 1e-9)
        
        # 价格位置
        for p in [20, 60]:
            roll_high = pd.Series(high).rolling(p).max()
            roll_low = pd.Series(low).rolling(p).min()
            feats[f'price_position_{p}'] = (close - roll_low) / (roll_high - roll_low + 1e-9)
        
        # ADX (趋势强度)
        feats['adx'] = _calc_adx(high, low, close, period=14)
        
        # === 情绪特征 ===
        sent_stock = sent_daily[sent_daily['symbol'] == sym].copy()
        if len(sent_stock) > 0:
            sent_stock = sent_stock.set_index('trade_date')
            for col in ['is_limit_up', 'is_limit_down', 'consecutive_limit_up', 'vol_ratio_20', 'abnormal_ret']:
                if col in sent_stock.columns:
                    mapped = stock_df['trade_date'].map(
                        lambda d: sent_stock.loc[d, col] if d in sent_stock.index else 0
                    ).fillna(0).values
                    feats[f'sent_{col}'] = mapped
        
        # === 龙虎榜 + 融资融券特征 ===
        sent_stock2 = sentiment[sentiment['symbol'] == sym].copy()
        if len(sent_stock2) > 0:
            sent_stock2 = sent_stock2.set_index('trade_date')
            for col in ['lhb_net_buy', 'lhb_ret_5d', 'lhb_net_buy_ratio',
                        'margin_balance_chg', 'short_balance']:
                if col in sent_stock2.columns:
                    prefix = 'lhb_' if col.startswith('lhb') else 'margin_'
                    col_name = col if col.startswith('margin') else col
                    feats[f'{prefix}{col_name}'] = stock_df['trade_date'].map(
                        lambda d: sent_stock2.loc[d, col] if d in sent_stock2.index else 0
                    ).fillna(0).values
        
        # === 基本面特征 ===
        fund_stock = fundamental[fundamental['symbol'] == sym].copy()
        if len(fund_stock) > 0:
            fund_stock = fund_stock.set_index('trade_date')
            for col in ['roe', 'debt_ratio', 'net_profit_yoy', 'revenue_yoy']:
                if col in fund_stock.columns:
                    feats[f'fund_{col}'] = stock_df['trade_date'].map(
                        lambda d: fund_stock.loc[d, col] if d in fund_stock.index else 0
                    ).fillna(0).values
        
        # === 北向资金特征 ===
        if len(north_flow) > 0:
            nf = north_flow.set_index('trade_date')
            for col in ['north_net', 'total_net']:
                feats[f'north_{col}'] = stock_df['trade_date'].map(
                    lambda d: nf.loc[d, col] if d in nf.index else 0
                ).fillna(0).values
        
        # === 板块特征 ===
        industry = sector_map.get(sym, '未知')
        feats['sector_code'] = hash(industry) % 100  # 板块编码
        
        results[sym] = (stock_df, feats)
        if len(results) % 20 == 0:
            gc.collect()
    
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


# === 模型训练 ===
def train_models(X_train, y_train, X_val, y_val, feature_names, quick=False):
    """训练LGBM + XGBoost + CatBoost"""
    models = {}
    cv_scores = {}
    
    if quick:
        lgb_params = {**LGBM_PARAMS, **QUICK_PARAMS}
        xgb_params = {**XGB_PARAMS, **QUICK_PARAMS}
        cb_params = {**CATBOOST_PARAMS, 'iterations': 500}
    else:
        lgb_params = LGBM_PARAMS.copy()
        xgb_params = XGB_PARAMS.copy()
        cb_params = CATBOOST_PARAMS.copy()
    
    # LGBM
    print("  🌳 训练 LGBM...")
    import lightgbm as lgb
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  eval_metric='l1', callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    pred = lgb_model.predict(X_val)
    ic = np.corrcoef(pred, y_val)[0, 1] if len(pred) > 1 else 0
    cv_scores['lgbm'] = ic
    models['lgbm'] = lgb_model
    print(f"     Val IC: {ic:.4f}")
    
    # XGBoost
    print("  🌲 训练 XGBoost...")
    import xgboost as xgb
    xgb_model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred = xgb_model.predict(X_val)
    ic = np.corrcoef(pred, y_val)[0, 1] if len(pred) > 1 else 0
    cv_scores['xgb'] = ic
    models['xgb'] = xgb_model
    print(f"     Val IC: {ic:.4f}")
    
    # CatBoost
    try:
        print("  🐱 训练 CatBoost...")
        from catboost import CatBoostRegressor
        cb_model = CatBoostRegressor(**cb_params)
        cb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                     early_stopping_rounds=50, verbose=False)
        pred = cb_model.predict(X_val)
        ic = np.corrcoef(pred, y_val)[0, 1] if len(pred) > 1 else 0
        cv_scores['catboost'] = ic
        models['catboost'] = cb_model
        print(f"     Val IC: {ic:.4f}")
    except Exception as e:
        print(f"     CatBoost 跳过: {e}")
    
    return models, cv_scores


def ensemble_weight(models, cv_scores):
    """按验证集IC加权"""
    total = sum(max(0, s) for s in cv_scores.values())
    if total == 0:
        return {k: 1.0/len(cv_scores) for k in cv_scores}
    return {k: max(0, s) / total for k, s in cv_scores.items()}


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
    X_all, y_all, regimes_all = [], [], []
    all_features = None
    
    for sym, (stock_df, feats) in results.items():
        # 目标变量: horizon期后的收益率
        close = stock_df['close'].values
        target = np.zeros(len(close))
        for i in range(len(close) - args.horizon):
            target[i] = (close[i + args.horizon] - close[i]) / (close[i] + 1e-9)
        target[-args.horizon:] = np.nan
        
        # 市场状态
        regime = detect_market_regime(stock_df)
        
        # 填充NaN
        feats = feats.fillna(0).replace([np.inf, -np.inf], 0)
        valid = ~np.isnan(target)
        valid[:120] = False  # 前120条作为历史
        
        X_all.append(feats[valid].values)
        y_all.append(target[valid])
        regimes_all.append(regime[valid])
        
        if all_features is None:
            all_features = list(feats.columns)
    
    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    regimes_all = np.concatenate(regimes_all)
    
    # 去除极端值
    y_clip = np.clip(y_all, -0.2, 0.2)
    valid_mask = np.abs(y_clip) < 0.15
    X_all = X_all[valid_mask]
    y_all = y_all[valid_mask]
    regimes_all = regimes_all[valid_mask]
    
    print(f"   总样本: {len(X_all)}")
    print(f"   特征数: {len(all_features)}")
    print(f"   牛市: {(regimes_all == 'bull').sum()}")
    print(f"   熊市: {(regimes_all == 'bear').sum()}")
    print(f"   震荡: {(regimes_all == 'sideways').sum()}")
    
    # 4. 训练/验证分割 (时间序列)
    n_train = int(len(X_all) * 0.8)
    X_train, X_val = X_all[:n_train], X_all[n_train:]
    y_train, y_val = y_all[:n_train], y_all[n_train:]
    r_train, r_val = regimes_all[:n_train], regimes_all[n_train:]
    
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
            for name, model in all_models[regime].items():
                pred = model.predict(X_val[mask])
                ic = np.corrcoef(pred, y_val[mask])[0, 1]
                mae = np.mean(np.abs(pred - y_val[mask]))
                print(f"    {name:10s}: IC={ic:.4f} MAE={mae:.4f} weight={all_weights[regime].get(name, 0):.3f}")
            
            ensemble_pred = np.zeros(mask.sum())
            for name, model in all_models[regime].items():
                ensemble_pred += model.predict(X_val[mask]) * all_weights[regime].get(name, 0)
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
    
    # 兼容旧版预测API: 添加 model 字段
    if not args.no_regime and not args.no_ensemble:
        # 多模型集成: 保存为models列表
        model_data['models'] = list(all_models['bull']['lgbm'] for _ in range(1))  # 占位
        model_data['model_types'] = ['lgbm']
    elif not args.no_ensemble:
        model_data['models'] = list(all_models['all'].values())
        model_data['model_types'] = list(all_models['all'].keys())
    else:
        model_data['model'] = all_models['all']['lgbm']
    
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