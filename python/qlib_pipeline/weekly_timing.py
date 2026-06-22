#!/usr/bin/env python3
"""
周频择时模型 v3 — 回归预测 + 趋势信号
- 30分钟K线 → 周级别特征
- 多只股票合并为面板数据
- LightGBM 回归: 预测下周收益率
- 信号: 预测收益 > 阈值 买入, < -阈值 卖出
- 对比: 简单均线策略 vs ML策略 vs 买入持有
"""

import os, sys, warnings, io, pickle, json, argparse
import numpy as np
import pandas as pd

np.seterr(all='ignore')
warnings.filterwarnings('ignore')

_stderr_backup = sys.stderr
sys.stderr = io.StringIO()
try:
    import gym
except Exception:
    pass
finally:
    sys.stderr = _stderr_backup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

import qlib
from qlib.constant import REG_CN
from qlib.contrib.ops.high_freq import Cut, DayLast, FFillNan, IsNull
from qlib.data import D

BIN_DIR = os.path.expanduser('~/.qlib/qlib_data/cn_30min/bin')
FREQ = '30min'

TARGET_STOCKS = [
    '000001.SZ',  # 平安银行
    '600519.SH',  # 贵州茅台
    '000858.SZ',  # 五粮液
    '600036.SH',  # 招商银行
    '000333.SZ',  # 美的集团
]

WEEK_HORIZON = 1
TRAIN_END = '2024-12-31'
VAL_END = '2025-09-30'
TEST_END = '2026-06-16'

# 回归信号阈值: 预测周收益超过此值才操作
SIGNAL_THRESHOLD = 0.005  # 0.5% 周收益


def build_weekly_features(df_raw, stock_id):
    """从 30min K线 构建周级别特征"""
    df = df_raw.copy().sort_index()

    daily_raw = df.groupby(df.index.date).agg({
        '$open': 'first', '$high': 'max', '$low': 'min',
        '$close': 'last', '$volume': 'sum',
    })
    daily_raw.index = pd.to_datetime(daily_raw.index)

    close = daily_raw['$close']
    high = daily_raw['$high']
    low = daily_raw['$low']
    volume = daily_raw['$volume']
    ret_1d = close.pct_change()

    daily_feat = pd.DataFrame({
        'close': close, 'volume': volume, 'ret_1d': ret_1d,
        'high': high, 'low': low,
        'open': daily_raw['$open'],
        'hl_range': (high - low) / close,
        'oc_range': (close - daily_raw['$open']) / daily_raw['$open'],
    }, index=daily_raw.index)

    weekly = daily_feat.resample('W').agg({
        'close': 'last', 'volume': 'sum',
        'ret_1d': ['mean', 'std', 'last'],
        'high': 'max', 'low': 'min', 'open': 'first',
        'hl_range': 'mean', 'oc_range': 'mean',
    })
    weekly.columns = ['_'.join(c).strip() for c in weekly.columns]
    weekly = weekly.rename(columns={
        'close_last': 'close', 'volume_sum': 'volume',
        'ret_1d_mean': 'ret_mean', 'ret_1d_std': 'ret_std',
        'ret_1d_last': 'ret_last', 'high_max': 'high', 'low_min': 'low',
        'open_first': 'open', 'hl_range_mean': 'hl_range',
        'oc_range_mean': 'oc_range',
    })

    weekly['ret'] = weekly['close'].pct_change()

    # ── 趋势 ──
    for w in [2, 4, 8, 12, 26]:
        weekly[f'ma_{w}w'] = weekly['close'].rolling(w).mean() / weekly['close'] - 1
        weekly[f'ret_{w}w'] = weekly['close'].pct_change(w)
        weekly[f'vol_{w}w'] = weekly['ret'].rolling(w).std()

    # ── 动量 ──
    for w in [2, 4, 8]:
        weekly[f'mom_{w}w'] = weekly['ret'].rolling(w).mean()

    # ── 量价 ──
    weekly['vol_ratio_4w'] = weekly['volume'] / weekly['volume'].rolling(4).mean()
    weekly['vol_ratio_12w'] = weekly['volume'] / weekly['volume'].rolling(12).mean()
    weekly['vol_trend'] = weekly['volume'].rolling(4).mean() / weekly['volume'].rolling(12).mean()

    # ── RSI ──
    delta = weekly['ret'].fillna(0)
    gain, loss = delta.clip(lower=0), (-delta).clip(lower=0)
    for w in [2, 4, 8]:
        avg_gain = gain.rolling(w).mean()
        avg_loss = loss.rolling(w).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        weekly[f'rsi_{w}w'] = 100 - 100 / (1 + rs)

    # ── 布林带 ──
    for w in [4, 8]:
        ma = weekly['close'].rolling(w).mean()
        std = weekly['close'].rolling(w).std()
        weekly[f'bb_pct_{w}w'] = (weekly['close'] - (ma - 2 * std)) / (4 * std + 1e-8)

    # ── 周线振幅 ──
    weekly['week_range'] = (weekly['high'] - weekly['low']) / weekly['close']

    # ── 最近涨跌 ──
    for w in [1, 2, 3, 4]:
        weekly[f'up_{w}w_ago'] = (weekly['ret'].shift(w) > 0).astype(float)

    # ── 标签: 未来一周收益率 (回归) ──
    weekly['label'] = weekly['close'].shift(-WEEK_HORIZON) / weekly['close'] - 1

    weekly['stock_id'] = stock_id
    weekly['stock_code'] = weekly['stock_id'].astype('category').cat.codes

    weekly = weekly.dropna()

    exclude = ['close', 'volume', 'high', 'low', 'open', 'label', 'stock_id', 'ret']
    feat_cols = [c for c in weekly.columns if c not in exclude]
    return weekly[feat_cols + ['label', 'stock_id', 'ret', 'close']], feat_cols


def load_raw_data(stock, start='2015-01-01', end='2026-06-16'):
    instruments = [stock]
    fields = ['$open', '$high', '$low', '$close', '$volume']
    df = D.features(instruments, fields, start, end, freq='30min')
    df = df.swaplevel().sort_index()
    return df.xs(stock, level='instrument')


def build_panel(stocks, verbose=True):
    """构建面板数据集"""
    qlib.init(provider_uri=BIN_DIR, region=REG_CN, freq=FREQ,
              custom_ops=[Cut, DayLast, FFillNan, IsNull], expression_cache=None)

    all_data = []
    all_feat_cols = None

    for stock in stocks:
        raw = load_raw_data(stock)
        weekly, feat_cols = build_weekly_features(raw, stock)
        if all_feat_cols is None:
            all_feat_cols = feat_cols
        all_data.append(weekly)

    panel = pd.concat(all_data).sort_index()
    if verbose:
        print(f"面板数据: {len(panel)} 周 ({panel.index[0]} ~ {panel.index[-1]})")
        print(f"特征数: {len(all_feat_cols)}, 股票数: {len(stocks)}")
        print(f"标签均值: {panel['label'].mean()*100:.2f}%  std: {panel['label'].std()*100:.2f}%")

    return panel, all_feat_cols


def train_panel_model(panel, feat_cols, verbose=True):
    """训练面板回归模型"""
    import lightgbm as lgb

    train_mask = panel.index <= TRAIN_END
    val_mask = (panel.index > TRAIN_END) & (panel.index <= VAL_END)
    test_mask = panel.index > VAL_END

    df_train = panel[train_mask]
    df_val = panel[val_mask]
    df_test = panel[test_mask]

    if verbose:
        print(f"训练: {len(df_train)} 周 | 验证: {len(df_val)} 周 | 测试: {len(df_test)} 周")

    X_train = df_train[feat_cols].values
    y_train = df_train['label'].values
    X_val = df_val[feat_cols].values
    y_val = df_val['label'].values
    X_test = df_test[feat_cols].values
    y_test = df_test['label'].values

    # 过滤 NaN
    for arr_name, arr, y_arr, df_ref in [
        ('train', X_train, y_train, df_train),
        ('val', X_val, y_val, df_val),
        ('test', X_test, y_test, df_test)]:
        mask = ~np.isnan(arr).any(axis=1)
        if arr_name == 'train':
            X_train, y_train, df_train = arr[mask], y_arr[mask], df_ref.iloc[mask]
        elif arr_name == 'val':
            X_val, y_val, df_val = arr[mask], y_arr[mask], df_ref.iloc[mask]
        else:
            X_test, y_test, df_test = arr[mask], y_arr[mask], df_ref.iloc[mask]

    model = lgb.LGBMRegressor(
        objective='regression', metric='rmse',
        num_leaves=15, learning_rate=0.03, n_estimators=300,
        early_stopping_rounds=30, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=2.0, min_child_samples=30,
        verbosity=-1, random_state=42,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse')

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    def calc_ic(pred, label):
        mask = ~(np.isnan(pred) | np.isnan(label))
        if mask.sum() < 10:
            return np.nan, np.nan
        ic = np.corrcoef(pred[mask], label[mask])[0, 1]
        from scipy.stats import spearmanr
        rank_ic = spearmanr(pred[mask], label[mask])[0]
        return ic, rank_ic

    train_ic, train_ric = calc_ic(train_pred, y_train)
    val_ic, val_ric = calc_ic(val_pred, y_val)
    test_ic, test_ric = calc_ic(test_pred, y_test)

    importance = {name: float(imp) for name, imp in zip(feat_cols, model.feature_importances_)}
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:15]

    if verbose:
        print(f"\n  📊 面板模型评估 (回归):")
        print(f"    训练: IC={train_ic:.4f} RankIC={train_ric:.4f} "
              f"RMSE={np.sqrt(((train_pred - y_train)**2).mean())*100:.2f}%")
        print(f"    验证: IC={val_ic:.4f} RankIC={val_ric:.4f} "
              f"RMSE={np.sqrt(((val_pred - y_val)**2).mean())*100:.2f}%")
        print(f"    测试: IC={test_ic:.4f} RankIC={test_ric:.4f} "
              f"RMSE={np.sqrt(((test_pred - y_test)**2).mean())*100:.2f}%")
        print(f"    预测分布: min={test_pred.min()*100:.2f}% max={test_pred.max()*100:.2f}% "
              f"mean={test_pred.mean()*100:.2f}% std={test_pred.std()*100:.2f}%")
        print(f"    实际分布: min={y_test.min()*100:.2f}% max={y_test.max()*100:.2f}% "
              f"mean={y_test.mean()*100:.2f}% std={y_test.std()*100:.2f}%")
        n_buy = (test_pred > SIGNAL_THRESHOLD).sum()
        n_sell = (test_pred < -SIGNAL_THRESHOLD).sum()
        print(f"    信号: 买入={n_buy} 卖出={n_sell} 持有={len(test_pred)-n_buy-n_sell}")
        print(f"\n  📊 特征重要性 Top 15:")
        for name, imp in top_features:
            print(f"    {name:<25s} {imp:.4f}")

    return {
        'model': model, 'features': feat_cols, 'importance': importance,
        'train_ic': float(train_ic), 'train_ric': float(train_ric),
        'val_ic': float(val_ic), 'val_ric': float(val_ric),
        'test_ic': float(test_ic), 'test_ric': float(test_ric),
        'df_test': df_test, 'test_pred': test_pred,
    }


def backtest_per_stock(result, stocks, verbose=True):
    """按股票回测: ML策略 + 均线策略 + 买入持有"""
    df_test = result['df_test']
    test_pred = result['test_pred']

    print(f"\n{'='*70}")
    print(f" 📈 分股票回测 (ML策略 vs 均线策略 vs 买入持有)")
    print(f"{'='*70}")
    header = (f"{'股票':<12} {'周数':>5} "
              f"{'ML年化':>10} {'MA年化':>10} {'持有年化':>10} "
              f"{'ML夏普':>7} {'MA夏普':>7} "
              f"{'ML胜率':>7} {'ML交易':>6}")
    print(header)
    print(f"{'-'*len(header)}")

    bt_results = {}
    for stock in stocks:
        mask = df_test['stock_id'] == stock
        if mask.sum() < 5:
            continue

        mask_arr = mask.values
        pred = test_pred[mask_arr]
        ret = df_test.loc[mask, 'ret'].values
        close = df_test.loc[mask, 'close'].values
        n = len(pred)

        # ── ML 策略 ──
        ml_positions = np.zeros(n)
        for i in range(n):
            if pred[i] > SIGNAL_THRESHOLD:
                ml_positions[i] = 1.0
            elif pred[i] < -SIGNAL_THRESHOLD:
                ml_positions[i] = 0.0
            else:
                ml_positions[i] = 0.0  # 不确定时空仓

        ml_ret = ml_positions * ret
        ml_total = (1 + ml_ret).prod() - 1
        ml_annual = (1 + ml_total) ** (52 / n) - 1
        ml_vol = ml_ret.std() * np.sqrt(52)
        ml_sharpe = ml_annual / ml_vol if ml_vol > 0 else 0
        ml_trades = (np.diff(ml_positions, prepend=0) != 0).sum()
        trade_mask = ml_positions > 0
        ml_win = (ml_ret[trade_mask] > 0).mean() if trade_mask.sum() > 0 else 0

        # ── 均线策略 (MA crossover) ──
        if len(close) > 4:
            ma_short = pd.Series(close).rolling(4).mean().values
            ma_long = pd.Series(close).rolling(12).mean().values
            ma_positions = np.zeros(n)
            for i in range(n):
                if np.isnan(ma_short[i]) or np.isnan(ma_long[i]):
                    ma_positions[i] = 0
                elif ma_short[i] > ma_long[i]:
                    ma_positions[i] = 1.0
                else:
                    ma_positions[i] = 0.0
            ma_ret = ma_positions * ret
            ma_total = (1 + ma_ret).prod() - 1
            ma_annual = (1 + ma_total) ** (52 / n) - 1
            ma_vol = ma_ret.std() * np.sqrt(52)
            ma_sharpe = ma_annual / ma_vol if ma_vol > 0 else 0
        else:
            ma_annual, ma_sharpe = 0, 0

        # ── 买入持有 ──
        bh_total = (1 + ret).prod() - 1
        bh_annual = (1 + bh_total) ** (52 / n) - 1

        print(f"{stock:<12} {n:>5} "
              f"{ml_annual:>9.2%} {ma_annual:>9.2%} {bh_annual:>9.2%} "
              f"{ml_sharpe:>7.3f} {ma_sharpe:>7.3f} "
              f"{ml_win:>7.1%} {ml_trades:>6}")

        bt_results[stock] = {
            'n_weeks': n,
            'ml_annual_return': float(ml_annual),
            'ml_sharpe': float(ml_sharpe),
            'ml_trades': int(ml_trades),
            'ml_win_rate': float(ml_win),
            'ma_annual_return': float(ma_annual),
            'ma_sharpe': float(ma_sharpe),
            'bh_annual_return': float(bh_annual),
        }

    return bt_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stocks', default=','.join(TARGET_STOCKS))
    parser.add_argument('--output', default='models/weekly_timing')
    parser.add_argument('--threshold', type=float, default=SIGNAL_THRESHOLD)
    args = parser.parse_args()

    stocks = [s.strip() for s in args.stocks.split(',')]
    output_dir = os.path.join(ROOT, args.output)
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f" 📈 周频择时训练 v3 (回归预测)")
    print(f"   股票: {stocks}")
    print(f"   预测: 下周收益率")
    print(f"   信号: |预测收益| > {args.threshold*100:.1f}% 才操作")
    print(f"{'='*60}")

    print(f"\n🔍 加载数据...")
    panel, feat_cols = build_panel(stocks)

    print(f"\n▶ 训练面板回归模型...")
    result = train_panel_model(panel, feat_cols)

    bt_results = backtest_per_stock(result, stocks)

    # 保存
    print(f"\n💾 保存模型...")
    with open(os.path.join(output_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(result['model'], f)

    config = {
        'features': feat_cols,
        'importance': result['importance'],
        'stocks': stocks,
        'horizon': WEEK_HORIZON,
        'threshold': args.threshold,
        'train_end': TRAIN_END, 'val_end': VAL_END,
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    summary = {
        'train_ic': result['train_ic'], 'train_ric': result['train_ric'],
        'val_ic': result['val_ic'], 'val_ric': result['val_ric'],
        'test_ic': result['test_ic'], 'test_ric': result['test_ric'],
        'backtest': bt_results,
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✅ 模型已保存到: {output_dir}")


if __name__ == '__main__':
    main()