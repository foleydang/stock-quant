#!/usr/bin/env python3
"""
日线特征工程 — 训练和推理共用 (Alpha158 级别)
从 OHLCV 计算 100+ 因子，确保训练推理完全一致
"""
import numpy as np
from scipy import stats as sp_stats


def compute_features(close, high, low, volume):
    """
    计算日线因子，返回 dict
    Args:
        close, high, low, volume: numpy 数组，按时间升序
    Returns:
        dict or None (数据不足)
    """
    n = len(close)
    if n < 120:
        return None

    feats = {}
    latest_close = close[-1]
    eps = 1e-8

    # ============================================================
    # 1. 收益率 (多周期)
    # ============================================================
    for p in [1, 2, 3, 5, 10, 20, 60, 120]:
        if n > p:
            feats[f'ret_{p}d'] = latest_close / (close[-p-1] + eps) - 1

    # ============================================================
    # 2. 波动率 (年化)
    # ============================================================
    for p in [5, 10, 20, 60]:
        if n >= p + 1:
            rets = np.diff(close[-p-1:]) / (close[-p-1:-1] + eps)
            feats[f'vol_{p}d'] = np.std(rets) * np.sqrt(252)

    # ============================================================
    # 3. 偏度/峰度
    # ============================================================
    for p in [20, 60]:
        if n >= p + 1:
            rets = np.diff(close[-p-1:]) / (close[-p-1:-1] + eps)
            if len(rets) > 3:
                feats[f'skew_{p}d'] = sp_stats.skew(rets)
                feats[f'kurt_{p}d'] = sp_stats.kurtosis(rets)

    # ============================================================
    # 4. 均线偏离
    # ============================================================
    for p in [5, 10, 20, 30, 60, 120]:
        if n >= p:
            ma = np.mean(close[-p:])
            feats[f'ma_dev_{p}d'] = (latest_close - ma) / (ma + eps)

    # ============================================================
    # 5. 均线交叉
    # ============================================================
    cross_pairs = [(5, 10), (5, 20), (5, 60), (10, 20), (10, 60), (20, 60)]
    for s, l in cross_pairs:
        if n >= l:
            ma_s = np.mean(close[-s:])
            ma_l = np.mean(close[-l:])
            feats[f'ma{s}_ma{l}'] = (ma_s - ma_l) / (ma_l + eps)

    # ============================================================
    # 6. 成交量因子
    # ============================================================
    for p in [5, 10, 20, 60]:
        if n >= p:
            avg_vol = np.mean(volume[-p:])
            feats[f'vol_ratio_{p}d'] = volume[-1] / (avg_vol + eps)
            feats[f'vol_std_{p}d'] = np.std(volume[-p:]) / (avg_vol + eps)

    # 量价关系
    if n >= 5:
        price_chg = close[-1] - close[-6] if n >= 6 else 0
        vol_chg = volume[-1] / (np.mean(volume[-5:]) + eps) - 1
        feats['vol_price_5d'] = price_chg / (latest_close + eps) * vol_chg

    # 量价相关性
    for p in [10, 20, 60]:
        if n >= p:
            rets = np.diff(close[-p:]) / (close[-p:-1] + eps)
            vol_chgs = np.diff(volume[-p:]) / (volume[-p:-1] + eps)
            if len(rets) > 2:
                feats[f'corr_rv_{p}d'] = np.corrcoef(rets, vol_chgs)[0, 1] if len(rets) > 1 else 0

    # ============================================================
    # 7. RSI (多周期)
    # ============================================================
    for p in [6, 14, 28]:
        if n >= p + 1:
            delta = np.diff(close[-p-1:])
            gain = np.sum(np.maximum(delta, 0))
            loss = np.sum(np.maximum(-delta, 0))
            avg_g = gain / p
            avg_l = loss / p
            if avg_l > 0:
                feats[f'rsi_{p}'] = 100 - 100 / (1 + avg_g / avg_l)
            else:
                feats[f'rsi_{p}'] = 100.0

    # ============================================================
    # 8. MACD
    # ============================================================
    if n >= 35:
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        dif = ema12 - ema26
        dea = _ema_series(dif, 9)
        macd = (dif - dea) * 2
        price = latest_close
        feats['macd_dif'] = dif[-1] / (price + eps)
        feats['macd_dea'] = dea[-1] / (price + eps)
        feats['macd_bar'] = macd[-1] / (price + eps)
        feats['macd_dif_chg'] = (dif[-1] - dif[-5]) / (price + eps) if n >= 40 else 0

    # ============================================================
    # 9. KDJ
    # ============================================================
    if n >= 9:
        for p in [9, 14]:
            if n >= p:
                ll = np.min(low[-p:])
                hh = np.max(high[-p:])
                rsv = (latest_close - ll) / (hh - ll + eps) * 100
                feats[f'kdj_rsv_{p}'] = rsv

    # ============================================================
    # 10. Bollinger Bands
    # ============================================================
    for p in [20, 60]:
        if n >= p:
            ma = np.mean(close[-p:])
            std = np.std(close[-p:])
            feats[f'bb_pct_{p}d'] = (latest_close - ma) / (2 * std + eps)  # %b
            feats[f'bb_width_{p}d'] = (2 * std) / (ma + eps)  # bandwidth

    # ============================================================
    # 11. ATR (真实波幅)
    # ============================================================
    for p in [14, 28]:
        if n >= p + 1:
            tr = np.maximum(
                high[-p:] - low[-p:],
                np.abs(high[-p:] - close[-p-1:-1]),
                np.abs(low[-p:] - close[-p-1:-1])
            )
            feats[f'atr_{p}d'] = np.mean(tr) / (latest_close + eps)

    # ============================================================
    # 12. 价格位置 (高低点)
    # ============================================================
    for p in [20, 60, 120]:
        if n >= p:
            hh = np.max(high[-p:])
            ll = np.min(low[-p:])
            feats[f'price_pos_{p}d'] = (latest_close - ll) / (hh - ll + eps)

    # ============================================================
    # 13. 最大回撤
    # ============================================================
    for p in [20, 60, 120]:
        if n >= p:
            rolling_max = np.maximum.accumulate(close[-p:])
            drawdown = (rolling_max - close[-p:]) / (rolling_max + eps)
            feats[f'max_dd_{p}d'] = np.max(drawdown)
            feats[f'mean_dd_{p}d'] = np.mean(drawdown)

    # ============================================================
    # 14. 夏普比率
    # ============================================================
    for p in [20, 60]:
        if n >= p + 1:
            rets = np.diff(close[-p-1:]) / (close[-p-1:-1] + eps)
            mean_ret = np.mean(rets)
            std_ret = np.std(rets)
            feats[f'sharpe_{p}d'] = mean_ret / (std_ret + eps) * np.sqrt(252)

    # ============================================================
    # 15. 涨跌比
    # ============================================================
    for p in [20, 60]:
        if n >= p + 1:
            up_days = np.sum(np.diff(close[-p-1:]) > 0)
            feats[f'up_ratio_{p}d'] = up_days / p

    # ============================================================
    # 16. 振幅
    # ============================================================
    for p in [5, 20]:
        if n >= p:
            amps = (high[-p:] - low[-p:]) / (close[-p:] + eps)
            feats[f'amp_{p}d'] = np.mean(amps)
            feats[f'amp_max_{p}d'] = np.max(amps)

    # ============================================================
    # 17. 跳空
    # ============================================================
    if n >= 2:
        feats['gap'] = (close[-1] - close[-2]) / (close[-2] + eps)
    if n >= 5:
        gaps = np.diff(close[-5:]) / (close[-5:-1] + eps)
        feats['gap_mean_5d'] = np.mean(gaps)

    # ============================================================
    # 18. OBV 变化
    # ============================================================
    if n >= 20:
        obv = np.zeros(n)
        for i in range(1, n):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        feats['obv_ma5'] = np.mean(obv[-5:]) / (np.mean(volume[-5:]) * 1000 + eps)
        feats['obv_ma20'] = np.mean(obv[-20:]) / (np.mean(volume[-20:]) * 1000 + eps)

    # ============================================================
    # 19. 威廉指标 WR
    # ============================================================
    for p in [14, 28]:
        if n >= p:
            hh = np.max(high[-p:])
            ll = np.min(low[-p:])
            feats[f'wr_{p}'] = (hh - latest_close) / (hh - ll + eps) * 100

    # ============================================================
    # 20. CCI (商品通道指数)
    # ============================================================
    for p in [14, 20]:
        if n >= p:
            tp = (high[-p:] + low[-p:] + close[-p:]) / 3
            ma_tp = np.mean(tp)
            md = np.mean(np.abs(tp - ma_tp))
            feats[f'cci_{p}'] = (tp[-1] - ma_tp) / (0.015 * md + eps)

    # ============================================================
    # 21. 价格动量 (ROC)
    # ============================================================
    for p in [3, 5, 10, 20, 60]:
        if n > p:
            feats[f'roc_{p}d'] = (latest_close - close[-p-1]) / (close[-p-1] + eps) * 100

    return feats


def _ema(data, span):
    """EMA"""
    alpha = 2 / (span + 1)
    result = np.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result


def _ema_series(data, span):
    alpha = 2 / (span + 1)
    result = np.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result


# 所有特征名 (排序)
FEATURE_NAMES = sorted([
    # 收益率
    'ret_1d', 'ret_2d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d', 'ret_120d',
    # 波动率
    'vol_5d', 'vol_10d', 'vol_20d', 'vol_60d',
    # 偏度/峰度
    'skew_20d', 'skew_60d', 'kurt_20d', 'kurt_60d',
    # 均线偏离
    'ma_dev_5d', 'ma_dev_10d', 'ma_dev_20d', 'ma_dev_30d', 'ma_dev_60d', 'ma_dev_120d',
    # 均线交叉
    'ma5_ma10', 'ma5_ma20', 'ma5_ma60', 'ma10_ma20', 'ma10_ma60', 'ma20_ma60',
    # 成交量
    'vol_ratio_5d', 'vol_ratio_10d', 'vol_ratio_20d', 'vol_ratio_60d',
    'vol_std_5d', 'vol_std_10d', 'vol_std_20d', 'vol_std_60d',
    'vol_price_5d',
    # 量价相关性
    'corr_rv_10d', 'corr_rv_20d', 'corr_rv_60d',
    # RSI
    'rsi_6', 'rsi_14', 'rsi_28',
    # MACD
    'macd_dif', 'macd_dea', 'macd_bar', 'macd_dif_chg',
    # KDJ
    'kdj_rsv_9', 'kdj_rsv_14',
    # Bollinger
    'bb_pct_20d', 'bb_pct_60d', 'bb_width_20d', 'bb_width_60d',
    # ATR
    'atr_14d', 'atr_28d',
    # 价格位置
    'price_pos_20d', 'price_pos_60d', 'price_pos_120d',
    # 回撤
    'max_dd_20d', 'max_dd_60d', 'max_dd_120d',
    'mean_dd_20d', 'mean_dd_60d', 'mean_dd_120d',
    # 夏普
    'sharpe_20d', 'sharpe_60d',
    # 涨跌比
    'up_ratio_20d', 'up_ratio_60d',
    # 振幅
    'amp_5d', 'amp_20d', 'amp_max_5d', 'amp_max_20d',
    # 跳空
    'gap', 'gap_mean_5d',
    # OBV
    'obv_ma5', 'obv_ma20',
    # 威廉
    'wr_14', 'wr_28',
    # CCI
    'cci_14', 'cci_20',
    # ROC
    'roc_3d', 'roc_5d', 'roc_10d', 'roc_20d', 'roc_60d',
])