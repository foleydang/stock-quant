#!/usr/bin/env python3
"""
日线特征工程模块 — 训练和推理共用
确保特征计算完全一致
"""
import numpy as np


def compute_features(close, high, low, volume):
    """
    计算日线因子，返回 dict
    参数均为 numpy 数组，按时间升序排列
    """
    n = len(close)
    if n < 120:
        return None

    feats = {}
    latest_close = close[-1]

    # === 动量因子 (多周期) ===
    for p in [5, 10, 20, 60]:
        if n > p and close[-p-1] > 0:
            feats[f'mom_{p}d'] = latest_close / close[-p-1] - 1

    # === 波动率 (年化, 取反) ===
    if n >= 21:
        rets = np.diff(close[-21:]) / (close[-21:-1] + 1e-8)
        feats['vol_20d'] = -np.std(rets) * np.sqrt(252)

    # === 均线偏离 ===
    for p in [5, 10, 20, 60]:
        if n >= p:
            ma = np.mean(close[-p:])
            feats[f'ma_dev_{p}d'] = (latest_close - ma) / (ma + 1e-8)

    # === 均线交叉 ===
    if n >= 20:
        ma5 = np.mean(close[-5:])
        ma20 = np.mean(close[-20:])
        feats['ma5_ma20_ratio'] = ma5 / (ma20 + 1e-8) - 1

    if n >= 60:
        ma20 = np.mean(close[-20:])
        ma60 = np.mean(close[-60:])
        feats['ma20_ma60_ratio'] = ma20 / (ma60 + 1e-8) - 1

    # === 成交量因子 ===
    for p in [5, 20]:
        if n >= p:
            avg_vol = np.mean(volume[-p:])
            feats[f'vol_ratio_{p}d'] = volume[-1] / (avg_vol + 1e-8)

    # 量价关系
    if n >= 5:
        price_chg = latest_close - close[-6] if n >= 6 else 0
        vol_ratio = volume[-1] / (np.mean(volume[-5:]) + 1e-8)
        feats['vol_price'] = price_chg / (latest_close + 1e-8) * vol_ratio

    # === RSI ===
    if n >= 15:
        delta = np.diff(close[-15:])
        gain = np.sum(np.maximum(delta, 0)) / 14
        loss = np.sum(np.maximum(-delta, 0)) / 14
        if loss > 0:
            feats['rsi_14'] = 100 - 100 / (1 + gain / loss)
        else:
            feats['rsi_14'] = 100

    # === ATR (波动率) ===
    if n >= 15:
        tr = np.maximum(
            high[-14:] - low[-14:],
            np.abs(high[-14:] - close[-15:-1]),
            np.abs(low[-14:] - close[-15:-1])
        )
        feats['atr_pct'] = np.mean(tr) / (latest_close + 1e-8)

    # === 价格位置 ===
    if n >= 20:
        hh = np.max(high[-20:])
        ll = np.min(low[-20:])
        feats['price_pos'] = (latest_close - ll) / (hh - ll + 1e-8)

    if n >= 60:
        hh = np.max(high[-60:])
        ll = np.min(low[-60:])
        feats['price_pos_60'] = (latest_close - ll) / (hh - ll + 1e-8)

    # === 夏普比率 ===
    if n >= 21:
        rets = np.diff(close[-21:]) / (close[-21:-1] + 1e-8)
        mean_ret = np.mean(rets)
        std_ret = np.std(rets)
        feats['sharpe_20d'] = mean_ret / (std_ret + 1e-8) * np.sqrt(252)

    # === 涨跌比 ===
    if n >= 20:
        up_days = np.sum(np.diff(close[-20:]) > 0)
        feats['up_ratio'] = up_days / 19

    return feats


FEATURE_NAMES = sorted([
    'mom_5d', 'mom_10d', 'mom_20d', 'mom_60d',
    'vol_20d',
    'ma_dev_5d', 'ma_dev_10d', 'ma_dev_20d', 'ma_dev_60d',
    'ma5_ma20_ratio', 'ma20_ma60_ratio',
    'vol_ratio_5d', 'vol_ratio_20d', 'vol_price',
    'rsi_14', 'atr_pct',
    'price_pos', 'price_pos_60',
    'sharpe_20d', 'up_ratio',
])