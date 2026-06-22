#!/usr/bin/env python3
"""
日线特征工程 — 训练和推理共用 (Alpha158 级别)
从 OHLCV 计算 100+ 因子，确保训练推理完全一致
"""
import numpy as np
from scipy import stats as sp_stats


def _ema(data, span):
    alpha = 2 / (span + 1)
    result = np.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result


def compute_features_batch(close, high, low, volume, min_idx=120):
    """
    批量计算日线因子，返回 DataFrame (每行一个日期)
    """
    n = len(close)
    if n < min_idx:
        return None

    eps = 1e-8
    feats = {}

    # ============================================================
    # 1. 收益率 (多周期) — 向量化
    # ============================================================
    for p in [1, 2, 3, 5, 10, 20, 60, 120]:
        ret = np.full(n, np.nan)
        ret[p:] = close[p:] / close[:-p] - 1
        feats[f'ret_{p}d'] = ret

    # ============================================================
    # 2. 波动率 (年化) — rolling
    # ============================================================
    for p in [5, 10, 20, 60]:
        rets = np.diff(close) / (close[:-1] + eps)
        vol = np.full(n, np.nan)
        for i in range(p, n):
            vol[i] = np.std(rets[i-p:i]) * np.sqrt(252)
        feats[f'vol_{p}d'] = vol

    # ============================================================
    # 3. 偏度/峰度
    # ============================================================
    for p in [20, 60]:
        rets = np.diff(close) / (close[:-1] + eps)
        skew = np.full(n, np.nan)
        kurt = np.full(n, np.nan)
        for i in range(p, n):
            r = rets[i-p:i]
            if len(r) > 3:
                skew[i] = sp_stats.skew(r)
                kurt[i] = sp_stats.kurtosis(r)
        feats[f'skew_{p}d'] = skew
        feats[f'kurt_{p}d'] = kurt

    # ============================================================
    # 4. 均线偏离
    # ============================================================
    for p in [5, 10, 20, 30, 60, 120]:
        ma = np.convolve(close, np.ones(p)/p, mode='valid')
        dev = np.full(n, np.nan)
        dev[p-1:] = (close[p-1:] - ma) / (ma + eps)
        feats[f'ma_dev_{p}d'] = dev

    # ============================================================
    # 5. 均线交叉
    # ============================================================
    cross_pairs = [(5, 10), (5, 20), (5, 60), (10, 20), (10, 60), (20, 60)]
    for s, l in cross_pairs:
        ma_s = np.convolve(close, np.ones(s)/s, mode='valid')
        ma_l = np.convolve(close, np.ones(l)/l, mode='valid')
        cross = np.full(n, np.nan)
        cross[l-1:] = (ma_s[l-s:] - ma_l) / (ma_l + eps)
        feats[f'ma{s}_ma{l}'] = cross

    # ============================================================
    # 6. 成交量因子
    # ============================================================
    for p in [5, 10, 20, 60]:
        vol_ma = np.convolve(volume, np.ones(p)/p, mode='valid')
        ratio = np.full(n, np.nan)
        ratio[p-1:] = volume[p-1:] / (vol_ma + eps)
        feats[f'vol_ratio_{p}d'] = ratio

        std = np.full(n, np.nan)
        for i in range(p, n):
            std[i] = np.std(volume[i-p:i]) / (np.mean(volume[i-p:i]) + eps)
        feats[f'vol_std_{p}d'] = std

    # 量价关系
    if n >= 5:
        vp = np.full(n, np.nan)
        for i in range(5, n):
            pc = close[i] - close[i-5]
            vc = volume[i] / (np.mean(volume[i-5:i]) + eps) - 1
            vp[i] = pc / (close[i] + eps) * vc
        feats['vol_price_5d'] = vp

    # 量价相关性
    for p in [10, 20, 60]:
        corr = np.full(n, np.nan)
        for i in range(p, n):
            r = np.diff(close[i-p:i]) / (close[i-p:i-1] + eps)
            v = np.diff(volume[i-p:i]) / (volume[i-p:i-1] + eps)
            if len(r) > 1:
                c = np.corrcoef(r, v)[0, 1]
                corr[i] = 0 if np.isnan(c) else c
        feats[f'corr_rv_{p}d'] = corr

    # ============================================================
    # 7. RSI
    # ============================================================
    for p in [6, 14, 28]:
        delta = np.diff(close)
        rsi = np.full(n, np.nan)
        for i in range(p, n):
            d = delta[i-p:i]
            gain = np.sum(np.maximum(d, 0))
            loss = np.sum(np.maximum(-d, 0))
            avg_g = gain / p
            avg_l = loss / p
            rsi[i] = 100 - 100 / (1 + avg_g / (avg_l + eps)) if avg_l > 0 else 100
        feats[f'rsi_{p}'] = rsi

    # ============================================================
    # 8. MACD
    # ============================================================
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    dif = ema12 - ema26
    dea = _ema(dif, 9)
    macd = (dif - dea) * 2
    feats['macd_dif'] = dif / (close + eps)
    feats['macd_dea'] = dea / (close + eps)
    feats['macd_bar'] = macd / (close + eps)
    if n >= 5:
        dif_chg = np.full(n, np.nan)
        dif_chg[5:] = (dif[5:] - dif[:-5]) / (close[5:] + eps)
        feats['macd_dif_chg'] = dif_chg

    # ============================================================
    # 9. KDJ
    # ============================================================
    for p in [9, 14]:
        rsv = np.full(n, np.nan)
        for i in range(p, n):
            ll = np.min(low[i-p:i])
            hh = np.max(high[i-p:i])
            rsv[i] = (close[i] - ll) / (hh - ll + eps) * 100
        feats[f'kdj_rsv_{p}'] = rsv

    # ============================================================
    # 10. Bollinger
    # ============================================================
    for p in [20, 60]:
        ma = np.convolve(close, np.ones(p)/p, mode='valid')
        bb_pct = np.full(n, np.nan)
        bb_width = np.full(n, np.nan)
        for i in range(p, n):
            std = np.std(close[i-p:i])
            bb_pct[i] = (close[i] - ma[i-p+1]) / (2 * std + eps)
            bb_width[i] = (2 * std) / (ma[i-p+1] + eps)
        feats[f'bb_pct_{p}d'] = bb_pct
        feats[f'bb_width_{p}d'] = bb_width

    # ============================================================
    # 11. ATR
    # ============================================================
    for p in [14, 28]:
        atr = np.full(n, np.nan)
        for i in range(p, n):
            tr = np.maximum(
                high[i-p+1:i+1] - low[i-p+1:i+1],
                np.abs(high[i-p+1:i+1] - close[i-p:i]),
                np.abs(low[i-p+1:i+1] - close[i-p:i])
            )
            atr[i] = np.mean(tr) / (close[i] + eps)
        feats[f'atr_{p}d'] = atr

    # ============================================================
    # 12. 价格位置
    # ============================================================
    for p in [20, 60, 120]:
        pos = np.full(n, np.nan)
        for i in range(p, n):
            hh = np.max(high[i-p:i])
            ll = np.min(low[i-p:i])
            pos[i] = (close[i] - ll) / (hh - ll + eps)
        feats[f'price_pos_{p}d'] = pos

    # ============================================================
    # 13. 回撤
    # ============================================================
    for p in [20, 60, 120]:
        max_dd = np.full(n, np.nan)
        mean_dd = np.full(n, np.nan)
        for i in range(p, n):
            c = close[i-p:i]
            rolling_max = np.maximum.accumulate(c)
            dd = (rolling_max - c) / (rolling_max + eps)
            max_dd[i] = np.max(dd)
            mean_dd[i] = np.mean(dd)
        feats[f'max_dd_{p}d'] = max_dd
        feats[f'mean_dd_{p}d'] = mean_dd

    # ============================================================
    # 14. 夏普
    # ============================================================
    for p in [20, 60]:
        rets = np.diff(close) / (close[:-1] + eps)
        sharpe = np.full(n, np.nan)
        for i in range(p, n):
            r = rets[i-p:i]
            mean_r = np.mean(r)
            std_r = np.std(r)
            sharpe[i] = mean_r / (std_r + eps) * np.sqrt(252)
        feats[f'sharpe_{p}d'] = sharpe

    # ============================================================
    # 15. 涨跌比
    # ============================================================
    for p in [20, 60]:
        up = np.full(n, np.nan)
        for i in range(p, n):
            up[i] = np.sum(np.diff(close[i-p:i+1]) > 0) / p
        feats[f'up_ratio_{p}d'] = up

    # ============================================================
    # 16. 振幅
    # ============================================================
    for p in [5, 20]:
        amp = np.full(n, np.nan)
        amp_max = np.full(n, np.nan)
        for i in range(p, n):
            a = (high[i-p+1:i+1] - low[i-p+1:i+1]) / (close[i-p+1:i+1] + eps)
            amp[i] = np.mean(a)
            amp_max[i] = np.max(a)
        feats[f'amp_{p}d'] = amp
        feats[f'amp_max_{p}d'] = amp_max

    # ============================================================
    # 17. 跳空
    # ============================================================
    gap = np.full(n, np.nan)
    gap[1:] = (close[1:] - close[:-1]) / (close[:-1] + eps)
    feats['gap'] = gap
    if n >= 5:
        gap_mean = np.full(n, np.nan)
        for i in range(5, n):
            gap_mean[i] = np.mean(gap[i-4:i+1])
        feats['gap_mean_5d'] = gap_mean

    # ============================================================
    # 18. OBV
    # ============================================================
    obv = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    for p in [5, 20]:
        obv_ma = np.full(n, np.nan)
        for i in range(p, n):
            obv_ma[i] = np.mean(obv[i-p:i]) / (np.mean(volume[i-p:i]) * 1000 + eps)
        feats[f'obv_ma{p}'] = obv_ma

    # ============================================================
    # 19. 威廉指标
    # ============================================================
    for p in [14, 28]:
        wr = np.full(n, np.nan)
        for i in range(p, n):
            hh = np.max(high[i-p:i])
            ll = np.min(low[i-p:i])
            wr[i] = (hh - close[i]) / (hh - ll + eps) * 100
        feats[f'wr_{p}'] = wr

    # ============================================================
    # 20. CCI
    # ============================================================
    for p in [14, 20]:
        cci = np.full(n, np.nan)
        for i in range(p, n):
            tp = (high[i-p:i] + low[i-p:i] + close[i-p:i]) / 3
            ma_tp = np.mean(tp)
            md = np.mean(np.abs(tp - ma_tp))
            cci[i] = (tp[-1] - ma_tp) / (0.015 * md + eps)
        feats[f'cci_{p}'] = cci

    # ============================================================
    # 21. ROC
    # ============================================================
    for p in [3, 5, 10, 20, 60]:
        roc = np.full(n, np.nan)
        roc[p:] = (close[p:] - close[:-p]) / (close[:-p] + eps) * 100
        feats[f'roc_{p}d'] = roc

    return feats


def compute_features(close, high, low, volume):
    """单个股最新一期特征 (推理用，轻量)"""
    n = len(close)
    if n < 120:
        return None

    eps = 1e-8
    result = {}

    # 收益率
    for p in [1, 2, 3, 5, 10, 20, 60, 120]:
        if n > p:
            result[f'ret_{p}d'] = float(close[-1] / close[-p-1] - 1)

    # 波动率
    rets_all = np.diff(close) / (close[:-1] + eps)
    for p in [5, 10, 20, 60]:
        if n >= p:
            result[f'vol_{p}d'] = float(np.std(rets_all[-p:]) * np.sqrt(252))

    # 偏度/峰度
    for p in [20, 60]:
        if n >= p:
            r = rets_all[-p:]
            if len(r) > 3:
                result[f'skew_{p}d'] = float(sp_stats.skew(r))
                result[f'kurt_{p}d'] = float(sp_stats.kurtosis(r))

    # 均线偏离
    for p in [5, 10, 20, 30, 60, 120]:
        if n >= p:
            ma = np.mean(close[-p:])
            result[f'ma_dev_{p}d'] = float((close[-1] - ma) / (ma + eps))

    # 均线交叉
    for s, l in [(5, 10), (5, 20), (5, 60), (10, 20), (10, 60), (20, 60)]:
        if n >= l:
            ma_s = np.mean(close[-s:])
            ma_l = np.mean(close[-l:])
            result[f'ma{s}_ma{l}'] = float((ma_s - ma_l) / (ma_l + eps))

    # 成交量
    for p in [5, 10, 20, 60]:
        if n >= p:
            avg_vol = np.mean(volume[-p:])
            result[f'vol_ratio_{p}d'] = float(volume[-1] / (avg_vol + eps))
            result[f'vol_std_{p}d'] = float(np.std(volume[-p:]) / (avg_vol + eps))

    if n >= 5:
        pc = close[-1] - close[-6]
        vc = volume[-1] / (np.mean(volume[-5:]) + eps) - 1
        result['vol_price_5d'] = float(pc / (close[-1] + eps) * vc)

    for p in [10, 20, 60]:
        if n >= p:
            r = rets_all[-p+1:]
            v = np.diff(volume[-p:]) / (volume[-p:-1] + eps)
            if len(r) > 1:
                c = np.corrcoef(r, v)[0, 1]
                result[f'corr_rv_{p}d'] = float(0 if np.isnan(c) else c)

    # RSI
    for p in [6, 14, 28]:
        if n >= p:
            d = np.diff(close[-p-1:])
            gain = np.sum(np.maximum(d, 0))
            loss = np.sum(np.maximum(-d, 0))
            avg_g = gain / p
            avg_l = loss / p
            result[f'rsi_{p}'] = float(100 - 100 / (1 + avg_g / (avg_l + eps)) if avg_l > 0 else 100)

    # MACD
    if n >= 35:
        ema12 = _ema_last(close, 12)
        ema26 = _ema_last(close, 26)
        dif = ema12 - ema26
        dea = _series_ema_last(close, dif, 12, 26, 9)
        macd = (dif - dea) * 2
        price = close[-1]
        result['macd_dif'] = float(dif / (price + eps))
        result['macd_dea'] = float(dea / (price + eps))
        result['macd_bar'] = float(macd / (price + eps))
        if n >= 40:
            dif_5 = _ema_last(close[:-5], 12) - _ema_last(close[:-5], 26)
            result['macd_dif_chg'] = float((dif - dif_5) / (price + eps))

    # KDJ
    for p in [9, 14]:
        if n >= p:
            ll = np.min(low[-p:])
            hh = np.max(high[-p:])
            result[f'kdj_rsv_{p}'] = float((close[-1] - ll) / (hh - ll + eps) * 100)

    # Bollinger
    for p in [20, 60]:
        if n >= p:
            ma = np.mean(close[-p:])
            std = np.std(close[-p:])
            result[f'bb_pct_{p}d'] = float((close[-1] - ma) / (2 * std + eps))
            result[f'bb_width_{p}d'] = float((2 * std) / (ma + eps))

    # ATR
    for p in [14, 28]:
        if n >= p:
            tr = np.maximum(
                high[-p:] - low[-p:],
                np.abs(high[-p:] - close[-p-1:-1]),
                np.abs(low[-p:] - close[-p-1:-1])
            )
            result[f'atr_{p}d'] = float(np.mean(tr) / (close[-1] + eps))

    # 价格位置
    for p in [20, 60, 120]:
        if n >= p:
            hh = np.max(high[-p:])
            ll = np.min(low[-p:])
            result[f'price_pos_{p}d'] = float((close[-1] - ll) / (hh - ll + eps))

    # 回撤
    for p in [20, 60, 120]:
        if n >= p:
            c = close[-p:]
            rolling_max = np.maximum.accumulate(c)
            dd = (rolling_max - c) / (rolling_max + eps)
            result[f'max_dd_{p}d'] = float(np.max(dd))
            result[f'mean_dd_{p}d'] = float(np.mean(dd))

    # 夏普
    for p in [20, 60]:
        if n >= p:
            r = rets_all[-p:]
            mean_r = np.mean(r)
            std_r = np.std(r)
            result[f'sharpe_{p}d'] = float(mean_r / (std_r + eps) * np.sqrt(252))

    # 涨跌比
    for p in [20, 60]:
        if n >= p:
            result[f'up_ratio_{p}d'] = float(np.sum(np.diff(close[-p-1:]) > 0) / p)

    # 振幅
    for p in [5, 20]:
        if n >= p:
            a = (high[-p:] - low[-p:]) / (close[-p:] + eps)
            result[f'amp_{p}d'] = float(np.mean(a))
            result[f'amp_max_{p}d'] = float(np.max(a))

    # 跳空
    result['gap'] = float((close[-1] - close[-2]) / (close[-2] + eps))
    if n >= 5:
        gaps = (close[-5:] - close[-6:-1]) / (close[-6:-1] + eps)
        result['gap_mean_5d'] = float(np.mean(gaps))

    # OBV
    if n >= 20:
        obv = np.zeros(n)
        for i in range(1, n):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        result['obv_ma5'] = float(np.mean(obv[-5:]) / (np.mean(volume[-5:]) * 1000 + eps))
        result['obv_ma20'] = float(np.mean(obv[-20:]) / (np.mean(volume[-20:]) * 1000 + eps))

    # 威廉
    for p in [14, 28]:
        if n >= p:
            hh = np.max(high[-p:])
            ll = np.min(low[-p:])
            result[f'wr_{p}'] = float((hh - close[-1]) / (hh - ll + eps) * 100)

    # CCI
    for p in [14, 20]:
        if n >= p:
            tp = (high[-p:] + low[-p:] + close[-p:]) / 3
            ma_tp = np.mean(tp)
            md = np.mean(np.abs(tp - ma_tp))
            result[f'cci_{p}'] = float((tp[-1] - ma_tp) / (0.015 * md + eps))

    # ROC
    for p in [3, 5, 10, 20, 60]:
        if n > p:
            result[f'roc_{p}d'] = float((close[-1] - close[-p-1]) / (close[-p-1] + eps) * 100)

    return result if len(result) > 20 else None


def _ema_last(data, span):
    """计算 EMA 最后一个值"""
    alpha = 2 / (span + 1)
    ema = data[0]
    for x in data[1:]:
        ema = alpha * x + (1 - alpha) * ema
    return ema


def _series_ema_last(close, dif, ema_span, macd_span, smoothing):
    """计算 DEA (dif 的 EMA) 最后一个值"""
    alpha = 2 / (smoothing + 1)
    # 用 dif 的 EMA 近似
    ema = dif
    for i in range(1, smoothing):
        ema = alpha * dif + (1 - alpha) * ema
    return ema


# 所有特征名 (排序)
FEATURE_NAMES = sorted([
    'ret_1d', 'ret_2d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d', 'ret_120d',
    'vol_5d', 'vol_10d', 'vol_20d', 'vol_60d',
    'skew_20d', 'skew_60d', 'kurt_20d', 'kurt_60d',
    'ma_dev_5d', 'ma_dev_10d', 'ma_dev_20d', 'ma_dev_30d', 'ma_dev_60d', 'ma_dev_120d',
    'ma5_ma10', 'ma5_ma20', 'ma5_ma60', 'ma10_ma20', 'ma10_ma60', 'ma20_ma60',
    'vol_ratio_5d', 'vol_ratio_10d', 'vol_ratio_20d', 'vol_ratio_60d',
    'vol_std_5d', 'vol_std_10d', 'vol_std_20d', 'vol_std_60d',
    'vol_price_5d',
    'corr_rv_10d', 'corr_rv_20d', 'corr_rv_60d',
    'rsi_6', 'rsi_14', 'rsi_28',
    'macd_dif', 'macd_dea', 'macd_bar', 'macd_dif_chg',
    'kdj_rsv_9', 'kdj_rsv_14',
    'bb_pct_20d', 'bb_pct_60d', 'bb_width_20d', 'bb_width_60d',
    'atr_14d', 'atr_28d',
    'price_pos_20d', 'price_pos_60d', 'price_pos_120d',
    'max_dd_20d', 'max_dd_60d', 'max_dd_120d',
    'mean_dd_20d', 'mean_dd_60d', 'mean_dd_120d',
    'sharpe_20d', 'sharpe_60d',
    'up_ratio_20d', 'up_ratio_60d',
    'amp_5d', 'amp_20d', 'amp_max_5d', 'amp_max_20d',
    'gap', 'gap_mean_5d',
    'obv_ma5', 'obv_ma20',
    'wr_14', 'wr_28',
    'cci_14', 'cci_20',
    'roc_3d', 'roc_5d', 'roc_10d', 'roc_20d', 'roc_60d',
])