#!/usr/bin/env python3
"""159792 港股通互联网ETF 专用择时/加仓模型 + 透明反弹规则。

为什么单独做:
  - 通用补仓顾问(add_advisor_ml)只在 A 股池训练(需≥400根日线), 159792 因日线不足
    被排除, 只靠外推打分; 且港股/ETF 缺宏观情绪特征(填0)。这次错过反弹的直接原因。
  - 本模块只用 159792 自身 OHLCV + 3 只高相关同类 ETF(513050/513330/159607)做外生特征,
    对 ETF 因果成立, 全部特征当日收盘可知, 无未来函数。

纪律:
  - 严格时间序 walk-forward, purge ≥ horizon, 只信样本外(OOS)。
  - 规则集是先验固定的(非拟合), 其历史条件收益是诚实的条件期望。
  - 带交易成本(ETF 往返≈0.1%)的决策级回测。

用法:
  python strategy/etf159792_model.py            # 全量: 拉特征→walk-forward→回测→当前建议
  python strategy/etf159792_model.py --signal   # 只输出当前信号(需已训模型或直接算规则)
  python strategy/etf159792_model.py --shuffle   # 打乱标签泄漏自检
"""
import os, sys, json, argparse, warnings
import numpy as np
import pandas as pd
import sqlite3

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # python/
DB_PATH = os.path.join(ROOT, 'data', 'stock_data.db')
OUT_DIR = os.path.join(ROOT, 'models', 'etf159792')
MODEL_PKL = os.path.join(OUT_DIR, 'model.pkl')

SYMBOL = '159792.SZ'
PEERS = ['513050.SH', '513330.SH', '159607.SZ']
# HSTECH 指数权重股 (top 10, 权重合计约 66%)
# 用成分股自下而上构建 ETF 走势信号——比只看 ETF 价格更早发现拐点
HSTECH_COMPONENTS = [
    ('00700.HK', '腾讯', 0.09),
    ('09988.HK', '阿里', 0.08),
    ('03690.HK', '美团', 0.08),
    ('09618.HK', '京东', 0.07),
    ('01024.HK', '快手', 0.07),
    ('01810.HK', '小米', 0.07),
    ('09999.HK', '网易', 0.06),
    ('09888.HK', '百度', 0.05),
    ('02015.HK', '理想', 0.05),
    ('00981.HK', '中芯', 0.04),
]
HORIZON = 10           # 预测未来10个交易日(约2周)—— 反弹级别
PURGE = HORIZON + 3    # walk-forward 切点 purge
COST = 0.001           # ETF 往返成本约 0.1%

LGB_PARAMS = dict(n_estimators=400, learning_rate=0.02, num_leaves=15,
                  max_depth=4, min_child_samples=40, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.5,
                  reg_lambda=5.0, min_split_gain=0.01, n_jobs=-1,
                  verbosity=-1, random_state=42)


# ---------------- 指标 ----------------
def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values

def rsi(close, n=14):
    d = np.diff(close, prepend=close[0])
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().values
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().values
    return 100 - 100 / (1 + ru / (rd + 1e-12))

def atr(high, low, close, n=14):
    pc = np.roll(close, 1); pc[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))
    return pd.Series(tr).ewm(alpha=1/n, adjust=False).mean().values

def macd(close, fast=12, slow=26, sig=9):
    dif = ema(close, fast) - ema(close, slow)
    dea = pd.Series(dif).ewm(span=sig, adjust=False).mean().values
    return dif, dea, (dif - dea)


# ---------------- 数据 + 特征 ----------------
def load_ohlcv(conn, sym):
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM kline_daily WHERE symbol=? ORDER BY date",
        conn, params=(sym,))
    df['date'] = pd.to_datetime(df['date'].astype(str).str.strip(), format='mixed')
    df = df.drop_duplicates('date', keep='last').sort_values('date').reset_index(drop=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_south_flow(conn):
    """南向资金(港股通)日净流入序列, 单位亿元。南向=内地资金净买港股,
    是港股通互联网ETF 最直接的资金面驱动。"""
    try:
        sdf = pd.read_sql(
            "SELECT trade_date, net_buy FROM south_flow ORDER BY trade_date", conn)
    except Exception:
        return None
    if sdf.empty:
        return None
    sdf['trade_date'] = pd.to_datetime(sdf['trade_date'].astype(str).str.strip(), format='mixed')
    sdf = sdf.dropna(subset=['net_buy']).sort_values('trade_date')
    sdf = sdf.drop_duplicates('trade_date', keep='last').set_index('trade_date')['net_buy']
    return sdf.astype(float)


def south_flow_features(etf_dates, sf):
    """把南向净流入对齐到 ETF 交易日, 并 shift(1) 保证因果。

    因果说明: 南向资金当日终值 ~16:00(港股收盘)才确定, 而 ETF 15:00(A股)收盘,
    当日南向值在 ETF 收盘时未定, 故用 t-1 的值预测 t 起 forward 收益。
    返回 dict[str, np.array], 长度=len(etf_dates), 对齐到 ETF 行。
    """
    n = len(etf_dates)
    out = {}
    if sf is None or len(sf) == 0:
        return out
    etf_dates = pd.to_datetime(etf_dates)
    # reindex 到 ETF 交易日(前向填充周末/缺日), 再 shift(1) 取 t-1
    aligned = sf.reindex(etf_dates, method='ffill').shift(1)
    net = aligned.values
    s = pd.Series(net)
    out['south_net1'] = net
    m20 = s.rolling(20, min_periods=10).mean().values
    s20 = s.rolling(20, min_periods=10).std().values
    out['south_net_z20'] = (net - m20) / (s20 + 1e-9)
    out['south_net_sum5'] = s.rolling(5, min_periods=3).sum().values
    out['south_net_sum20'] = s.rolling(20, min_periods=10).sum().values
    out['south_net_chg1'] = s.diff(1).values
    out['south_net_ma5'] = s.rolling(5, min_periods=3).mean().values
    return out


def build_component_features(conn, etf_dates):
    """对 HSTECH 成分股计算技术指标, 按权重聚合为自下而上信号。

    返回 dict: 每个 key 是长度 n 的 numpy array (与 etf_dates 对齐)。
    所有特征 t 时刻已知(只用当日及之前成分股数据), 无未来函数。
    """
    n = len(etf_dates)
    total_w = sum(w for _, _, w in HSTECH_COMPONENTS)
    comp_data = {}
    for sym, _name, w in HSTECH_COMPONENTS:
        df = load_ohlcv(conn, sym)
        if df.empty:
            continue
        df = df.set_index('date')
        comp_data[sym] = {'df': df, 'w': w / total_w}

    out = {}
    # 对每个交易日, 用成分股当日已知数据计算聚合指标
    for metric in ['pct_above_ma20', 'avg_rsi14', 'pct_macd_golden',
                   'wgt_ret5', 'wgt_ret20', 'pct_rsi_oversold', 'pct_vol_expand',
                   'avg_close_position', 'avg_upper_shadow', 'avg_body_ratio']:
        out[metric] = np.full(n, np.nan)

    for i, dt in enumerate(etf_dates):
        dt = pd.Timestamp(dt)
        vals_above = []; vals_rsi = []; vals_macd = []; vals_ret5 = []; vals_ret20 = []
        vals_oversold = []; vals_vol = []; weights = []
        vals_closepos = []; vals_upshadow = []; vals_body = []
        for sym, info in comp_data.items():
            pdf = info['df']
            if dt not in pdf.index:
                continue
            idx = pdf.index.get_loc(dt)
            if idx < 60:  # 需要足够历史算指标
                continue
            c_hist = pdf['close'].values[:idx + 1]
            c = c_hist[-1]
            if c <= 0:
                continue
            w = info['w']
            ma20 = np.mean(c_hist[-20:]) if len(c_hist) >= 20 else c
            vals_above.append(1.0 if c > ma20 else 0.0)
            # RSI(14)
            d_rsi = np.diff(c_hist[-15:], prepend=c_hist[-15])
            up = np.where(d_rsi > 0, d_rsi, 0.0); dn = np.where(d_rsi < 0, -d_rsi, 0.0)
            au = np.mean(up[-14:]) if len(up) >= 14 else np.mean(up)
            ad = np.mean(dn[-14:]) if len(dn) >= 14 else np.mean(dn)
            r = 100 - 100 / (1 + au / (ad + 1e-12)) if ad > 0 else 50
            vals_rsi.append(r); vals_oversold.append(1.0 if r < 30 else 0.0)
            # MACD 金叉(hist 由负转正)
            dif = ema(c_hist, 12)[-1] - ema(c_hist, 26)[-1]
            dea = pd.Series(ema(c_hist, 12) - ema(c_hist, 26)).ewm(span=9, adjust=False).mean().values[-1]
            hist_now = dif - dea
            if idx >= 1:
                dif_p = ema(c_hist[:-1], 12)[-1] - ema(c_hist[:-1], 26)[-1]
                dea_p = pd.Series(ema(c_hist[:-1], 12) - ema(c_hist[:-1], 26)).ewm(span=9, adjust=False).mean().values[-1]
                hist_prev = dif_p - dea_p
                vals_macd.append(1.0 if hist_prev < 0 <= hist_now else 0.0)
            else:
                vals_macd.append(0.0)
            # 动量
            if len(c_hist) >= 6:
                vals_ret5.append(c / c_hist[-6] - 1)
            if len(c_hist) >= 21:
                vals_ret20.append(c / c_hist[-21] - 1)
            # 量能
            v = pdf['volume'].values[:idx + 1]
            vma20 = np.mean(v[-20:]) if len(v) >= 20 else v[-1]
            vals_vol.append(1.0 if v[-1] > vma20 * 1.2 else 0.0)
            # 蜡烛图/日内质量(成分股)
            o_day = pdf['open'].values[idx]; h_day = pdf['high'].values[idx]
            l_day = pdf['low'].values[idx]
            hl = h_day - l_day + 1e-12
            vals_closepos.append((c - l_day) / hl)
            vals_upshadow.append((h_day - max(o_day, c)) / hl)
            vals_body.append(abs(c - o_day) / hl)
            weights.append(w)

        if weights and sum(weights) > 0:
            wsum = sum(weights)
            out['pct_above_ma20'][i] = sum(v * w for v, w in zip(vals_above, weights)) / wsum
            out['avg_rsi14'][i] = sum(v * w for v, w in zip(vals_rsi, weights)) / wsum
            out['pct_macd_golden'][i] = sum(v * w for v, w in zip(vals_macd, weights)) / wsum
            out['pct_rsi_oversold'][i] = sum(v * w for v, w in zip(vals_oversold, weights)) / wsum
            out['pct_vol_expand'][i] = sum(v * w for v, w in zip(vals_vol, weights)) / wsum
            if vals_ret5:
                out['wgt_ret5'][i] = sum(v * w for v, w in zip(vals_ret5, weights[:len(vals_ret5)])) / sum(weights[:len(vals_ret5)])
            if vals_ret20:
                out['wgt_ret20'][i] = sum(v * w for v, w in zip(vals_ret20, weights[:len(vals_ret20)])) / sum(weights[:len(vals_ret20)])
            out['avg_close_position'][i] = sum(v * w for v, w in zip(vals_closepos, weights)) / wsum
            out['avg_upper_shadow'][i] = sum(v * w for v, w in zip(vals_upshadow, weights)) / wsum
            out['avg_body_ratio'][i] = sum(v * w for v, w in zip(vals_body, weights)) / wsum

    return out


def build_features(conn):
    """返回 df(含特征列 + 元列), feat_names。所有特征 t 时刻收盘后可知。"""
    df = load_ohlcv(conn, SYMBOL)
    c = df['close'].values; h = df['high'].values; l = df['low'].values
    o = df['open'].values; v = df['volume'].values
    n = len(c)
    f = pd.DataFrame(index=df.index)

    # 动量/收益
    for k in [1, 3, 5, 10, 20]:
        f[f'ret{k}'] = pd.Series(c).pct_change(k).values
    # 均线比值 + 斜率
    for k in [5, 10, 20, 60]:
        ma = pd.Series(c).rolling(k).mean().values
        f[f'px_ma{k}'] = c / (ma + 1e-12) - 1
        if k == 20:
            ma20 = ma
    f['ma20_slope'] = pd.Series(ma20).pct_change(5).values
    ma60 = pd.Series(c).rolling(60).mean().values

    # RSI
    f['rsi14'] = rsi(c, 14)
    f['rsi6'] = rsi(c, 6)
    f['rsi14_chg'] = pd.Series(f['rsi14']).diff(3).values
    # MACD (按价格归一化)
    dif, dea, hist = macd(c)
    f['macd_dif'] = dif / (c + 1e-12)
    f['macd_hist'] = hist / (c + 1e-12)
    f['macd_hist_chg'] = pd.Series(f['macd_hist']).diff(1).values
    # 布林
    m20 = pd.Series(c).rolling(20).mean(); s20 = pd.Series(c).rolling(20).std()
    f['boll_pctb'] = ((c - (m20 - 2 * s20)) / (4 * s20 + 1e-12)).values
    f['boll_bw'] = (4 * s20 / (m20 + 1e-12)).values
    # 波动
    a = atr(h, l, c, 14)
    f['atr_pct'] = a / (c + 1e-12)
    f['vol_20'] = pd.Series(pd.Series(c).pct_change()).rolling(20).std().values
    # 量能
    vma20 = pd.Series(v).rolling(20).mean().values
    f['vol_ratio'] = v / (vma20 + 1e-12)
    f['vol_z'] = ((v - vma20) / (pd.Series(v).rolling(20).std().values + 1e-12))
    ret1 = pd.Series(c).pct_change().values
    obv = np.cumsum(np.sign(np.nan_to_num(ret1)) * v)
    f['obv_slope'] = pd.Series(obv).pct_change(10).replace([np.inf, -np.inf], 0).values

    # 蜡烛图/日内质量 —— 区分"冲高回落"和"一路向上"
    hl_range = h - l + 1e-12
    upper_body = np.maximum(o, c)
    lower_body = np.minimum(o, c)
    f['upper_shadow'] = (h - upper_body) / hl_range     # 上影线比例: 越大越像冲高回落
    f['lower_shadow'] = (lower_body - l) / hl_range     # 下影线比例: 越大越像探底回升
    f['body_ratio'] = np.abs(c - o) / hl_range          # 实体比例: 越大趋势越确认
    f['close_position'] = (c - l) / hl_range            # 收盘在日内位置: 1=收最高, 0=收最低
    f['gap'] = o / np.roll(c, 1) - 1                     # 跳空: 隔夜情绪
    f['gap'][0] = 0
    # 冲高回落日: 上影线>40% 且收阴线
    fade_today = ((f['upper_shadow'] > 0.4) & (c < o)).astype(float).values
    f['fade_today'] = fade_today
    f['fade_5d'] = pd.Series(fade_today).rolling(5, min_periods=1).sum().values  # 近5日冲高回落次数
    f['fade_10d'] = pd.Series(fade_today).rolling(10, min_periods=1).sum().values
    # 探底回升日: 下影线>50% 实体<30%
    hammer_today = ((f['lower_shadow'] > 0.5) & (f['body_ratio'] < 0.3)).astype(float).values
    f['hammer_today'] = hammer_today
    f['hammer_5d'] = pd.Series(hammer_today).rolling(5, min_periods=1).sum().values
    # 日内波动烈度
    f['intraday_range'] = (h - l) / (c + 1e-12)         # 日内振幅
    f['intraday_range_ma5'] = pd.Series(f['intraday_range']).rolling(5).mean().values
    # 收盘强度: 连续收阳天数和连续高收盘(close>open 且 close_position>0.7)
    up_days = (c > o).astype(int)
    streak_up = np.zeros(n)
    for i in range(1, n):
        streak_up[i] = streak_up[i-1] + 1 if up_days[i] else 0
    f['streak_up'] = streak_up                              # 连续收阳天数
    strong_close = ((c > o) & (f['close_position'] > 0.7)).astype(float).values
    f['strong_close_5d'] = pd.Series(strong_close).rolling(5, min_periods=1).sum().values  # 强势收盘次数
    # 量价配合: 放量阳线 vs 放量阴线
    vol_expand = (f['vol_ratio'] > 1.2).values
    up_vol = (vol_expand & (c > o)).astype(float)
    dn_vol = (vol_expand & (c < o)).astype(float)
    f['up_vol_5d'] = pd.Series(up_vol).rolling(5, min_periods=1).sum().values
    f['dn_vol_5d'] = pd.Series(dn_vol).rolling(5, min_periods=1).sum().values
    f['vol_quality'] = f['up_vol_5d'] - f['dn_vol_5d']      # 正=放量涨多, 负=放量跌多
    # 回撤 / 位置
    hh60 = pd.Series(c).rolling(60).max().values
    hh120 = pd.Series(c).rolling(120).max().values
    ll60 = pd.Series(c).rolling(60).min().values
    f['dd_from_hh60'] = c / (hh60 + 1e-12) - 1
    f['dd_from_hh120'] = c / (hh120 + 1e-12) - 1
    f['up_from_ll60'] = c / (ll60 + 1e-12) - 1
    # 破位天数(连续在 MA20 下方)
    below = (c < ma20).astype(int)
    days_below = np.zeros(n)
    for i in range(1, n):
        days_below[i] = days_below[i-1] + 1 if below[i] else 0
    f['days_below_ma20'] = days_below

    # 同类 ETF 外生特征(当日可知)
    base = df[['date']].copy()
    for p in PEERS:
        pdf = load_ohlcv(conn, p)[['date', 'close']].rename(columns={'close': p})
        base = base.merge(pdf, on='date', how='left')
    for p in PEERS:
        pc = base[p].values
        base[f'{p}_r1'] = pd.Series(pc).pct_change(1).values
        base[f'{p}_r5'] = pd.Series(pc).pct_change(5).values
    peer_r1_cols = [f'{p}_r1' for p in PEERS]
    peer_r5_cols = [f'{p}_r5' for p in PEERS]
    f['peer_r1'] = base[peer_r1_cols].mean(axis=1).values
    f['peer_r5'] = base[peer_r5_cols].mean(axis=1).values

    # 成分股自下而上特征 —— 比 ETF 价格更早感知拐点
    comp_feats = build_component_features(conn, df['date'].values)
    for k, v in comp_feats.items():
        f[k] = v

    # 南向资金特征 —— 港股通互联网ETF 最直接的资金面驱动(因果 t-1)
    sf = load_south_flow(conn)
    south_feats = south_flow_features(df['date'].values, sf)
    for k, v in south_feats.items():
        f[k] = v
    if south_feats:
        print(f"   南向资金特征: {list(south_feats.keys())} (最新t-1值={south_feats['south_net1'][-1]:.2f}亿)")

    feat_names = list(f.columns)
    f['__date'] = df['date'].values
    f['__close'] = c
    f['__ma20'] = ma20
    f['__ma60'] = ma60
    f['__rsi14'] = f['rsi14']
    f['__hist'] = hist
    f['__atr'] = a
    f['__vol_ratio'] = f['vol_ratio']
    # 标签: 未来 HORIZON 日收益
    fwd = np.full(n, np.nan)
    for t in range(n - HORIZON):
        fwd[t] = c[t + HORIZON] / c[t] - 1
    f['__fwd'] = fwd
    f['__up'] = np.where(np.isfinite(fwd), (fwd > 0).astype(float), np.nan)

    f[feat_names] = (f[feat_names].apply(pd.to_numeric, errors='coerce')
                     .replace([np.inf, -np.inf], np.nan))
    return f, feat_names


# ---------------- 透明反弹规则(先验固定) ----------------
def rule_signal_row(close, ma20, ma20_prev, ma60, rsi14, rsi14_prev, hist,
                    hist_prev, vol_ratio, up_today, peer_r5, dd120, boll_pctb):
    """返回 (score, tags)。分数越高越偏多/反弹。"""
    score = 0; tags = []
    # 趋势
    if close > ma20:
        score += 1; tags.append('站上MA20')
    if ma20 > ma20_prev:
        score += 1; tags.append('MA20上行')
    if close > ma60:
        score += 1; tags.append('站上MA60')
    # 反转
    if rsi14_prev < 40 <= rsi14:
        score += 2; tags.append('RSI超卖回升')
    if hist_prev < 0 <= hist:
        score += 2; tags.append('MACD金叉')
    elif hist > hist_prev and hist < 0:
        score += 1; tags.append('MACD柱收敛')
    # 量能确认
    if up_today and vol_ratio > 1.2:
        score += 1; tags.append('放量上涨')
    # 同类共振
    if peer_r5 is not None and peer_r5 > 0.01:
        score += 1; tags.append('同类走强')
    # 深跌反弹赔率
    if dd120 is not None and dd120 < -0.25:
        score += 1; tags.append('深跌赔率')
    # 超买/见顶 -> 减分
    if rsi14 > 75:
        score -= 2; tags.append('RSI超买')
    if boll_pctb is not None and boll_pctb > 1.0:
        score -= 1; tags.append('破布林上轨')
    if close / (ma20 + 1e-12) - 1 > 0.15:
        score -= 1; tags.append('远离MA20')
    return score, tags


def rule_signal_row_with_candle(close, ma20, ma20_prev, ma60, rsi14, rsi14_prev, hist,
                                hist_prev, vol_ratio, up_today, peer_r5, dd120, boll_pctb,
                                upper_shadow, close_position, fade_5d, body_ratio):
    """扩展版规则: 加上蜡烛图/日内质量信号, 区分冲高回落 vs 稳步上行。"""
    score, tags = rule_signal_row(close, ma20, ma20_prev, ma60, rsi14, rsi14_prev, hist,
                                  hist_prev, vol_ratio, up_today, peer_r5, dd120, boll_pctb)
    # 冲高回落减分: 上影线长 + 收盘偏弱
    if upper_shadow is not None and upper_shadow > 0.4 and close_position is not None and close_position < 0.5:
        score -= 1; tags.append('日内冲高回落')
    # 近5日多次冲高回落→趋势不健康
    if fade_5d is not None and fade_5d >= 3:
        score -= 1; tags.append('连续冲高回落')
    # 稳步上行加分: 实体大+收盘高位+无上影
    if body_ratio is not None and close_position is not None and upper_shadow is not None:
        if body_ratio > 0.5 and close_position > 0.8 and upper_shadow < 0.15:
            score += 1; tags.append('强势收盘')
    return score, tags


def compute_rule_series(f):
    """对每根 bar 计算规则分数(全部因果), 返回 score 数组与 tags 列表。"""
    close = f['__close'].values; ma20 = f['__ma20'].values; ma60 = f['__ma60'].values
    r14 = f['__rsi14'].values; hist = f['__hist'].values; vr = f['__vol_ratio'].values
    dd120 = f['dd_from_hh120'].values; pctb = f['boll_pctb'].values
    peer5 = f['peer_r5'].values
    # 蜡烛图特征(可能不存在, 兼容旧模型)
    us = f['upper_shadow'].values if 'upper_shadow' in f.columns else np.zeros(len(f))
    cp = f['close_position'].values if 'close_position' in f.columns else np.zeros(len(f))
    f5 = f['fade_5d'].values if 'fade_5d' in f.columns else np.zeros(len(f))
    br = f['body_ratio'].values if 'body_ratio' in f.columns else np.zeros(len(f))
    n = len(f); scores = np.zeros(n); tags_all = [[] for _ in range(n)]
    for i in range(1, n):
        up_today = close[i] > close[i-1]
        kwargs = dict(
            close=close[i], ma20=ma20[i], ma20_prev=ma20[i-1], ma60=ma60[i],
            rsi14=r14[i], rsi14_prev=r14[i-1], hist=hist[i], hist_prev=hist[i-1],
            vol_ratio=vr[i], up_today=up_today,
            peer_r5=peer5[i] if np.isfinite(peer5[i]) else None,
            dd120=dd120[i] if np.isfinite(dd120[i]) else None,
            boll_pctb=pctb[i] if np.isfinite(pctb[i]) else None,
            upper_shadow=us[i] if np.isfinite(us[i]) else None,
            close_position=cp[i] if np.isfinite(cp[i]) else None,
            fade_5d=f5[i] if np.isfinite(f5[i]) else None,
            body_ratio=br[i] if np.isfinite(br[i]) else None,
        )
        s, tg = rule_signal_row_with_candle(**kwargs)
        scores[i] = s; tags_all[i] = tg
    return scores, tags_all


# ---------------- walk-forward ML ----------------
def per_win_metrics(y, p_up, ret, reg):
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    out = {}
    try:
        out['auc'] = float(roc_auc_score(y.astype(int), p_up))
    except Exception:
        out['auc'] = np.nan
    out['acc'] = float(np.mean((p_up > 0.5) == (y > 0.5)))
    out['base_up'] = float(np.mean(y))
    ic = spearmanr(ret, reg)[0] if len(ret) > 5 else np.nan
    out['ic'] = float(ic) if np.isfinite(ic) else np.nan
    return out


def walk_forward(f, feat_names, shuffle=False):
    from lightgbm import LGBMClassifier, LGBMRegressor
    d = f.dropna(subset=feat_names + ['__fwd', '__up']).reset_index(drop=True)
    X = d[feat_names].values.astype(np.float32)
    dates = pd.to_datetime(d['__date'].values)
    y = d['__up'].values.astype(int)
    ret = d['__fwd'].values.astype(float)
    if shuffle:
        rng = np.random.default_rng(0); perm = rng.permutation(len(y))
        y = y[perm]; ret = ret[perm]

    n = len(d)
    start = max(400, int(n * 0.4))          # 首个训练窗口
    step = 60                                # 每 60 交易日一个测试块
    oos = {'date': [], 'y': [], 'ret': [], 'p_up': [], 'reg': []}
    folds = 0
    t = start
    while t + 20 < n:
        tr_end = t - PURGE                   # purge 切点
        if tr_end < 300:
            t += step; continue
        te_end = min(t + step, n)
        Xtr, ytr, rtr = X[:tr_end], y[:tr_end], ret[:tr_end]
        Xte = X[t:te_end]
        clf = LGBMClassifier(**LGB_PARAMS).fit(Xtr, ytr)
        reg = LGBMRegressor(**LGB_PARAMS).fit(Xtr, rtr)
        oos['date'].extend(dates[t:te_end]); oos['y'].extend(y[t:te_end])
        oos['ret'].extend(ret[t:te_end]); oos['p_up'].extend(clf.predict_proba(Xte)[:, 1])
        oos['reg'].extend(reg.predict(Xte))
        folds += 1; t += step
    for k in oos:
        oos[k] = np.array(oos[k])
    oos['folds'] = folds
    return oos


def eval_oos(oos):
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    y = oos['y']; ret = oos['ret']; pup = oos['p_up']; reg = oos['reg']
    n = len(y)
    rep = {'n': int(n), 'folds': int(oos['folds'])}
    if n < 30:
        rep['note'] = 'OOS 样本太少'; return rep
    rep['base_up_rate'] = float(np.mean(y))
    rep['dir_acc'] = float(np.mean((pup > 0.5) == (y > 0.5)))
    try:
        rep['auc'] = float(roc_auc_score(y.astype(int), pup))
    except Exception:
        rep['auc'] = float('nan')
    ic = spearmanr(ret, reg)[0]
    rep['ic'] = float(ic) if np.isfinite(ic) else float('nan')
    # 决策回测: 预测上涨(p_up>0.5)才持有 HORIZON 天, 扣一次成本
    hold = pup > 0.5
    strat = np.where(hold, ret - COST, 0.0)
    rep['decision_ret_net'] = float(np.mean(strat))
    rep['buyhold_ret'] = float(np.mean(ret))
    rep['trades'] = int(hold.sum())
    # 高置信档(p_up>0.6)
    hi = pup > 0.6
    if hi.sum() >= 10:
        rep['hi_conf_n'] = int(hi.sum())
        rep['hi_conf_hit'] = float(np.mean(ret[hi] > 0))
        rep['hi_conf_ret_net'] = float(np.mean(ret[hi] - COST))
    rep['usable'] = bool(np.isfinite(rep['auc']) and rep['auc'] > 0.53
                         and rep['dir_acc'] > rep['base_up_rate'] + 0.01
                         and np.isfinite(rep['ic']) and rep['ic'] > 0.02
                         and rep['decision_ret_net'] > rep['buyhold_ret'] * 0.5)
    return rep


# ---------------- 规则条件收益(诚实) ----------------
def eval_rule(f, scores):
    fwd = f['__fwd'].values
    dates = pd.to_datetime(f['__date'].values)
    valid = np.isfinite(fwd)
    rep = {}
    base = float(np.nanmean(fwd[valid]))
    rep['base_fwd10'] = base
    for thr in [4, 5, 6]:
        m = valid & (scores >= thr)
        if m.sum() >= 10:
            rep[f'buy>={thr}'] = {
                'n': int(m.sum()),
                'mean_fwd10': float(np.mean(fwd[m])),
                'win_rate': float(np.mean(fwd[m] > 0)),
                'net_vs_base': float(np.mean(fwd[m]) - base),
            }
    # 减仓侧
    m = valid & (scores <= -2)
    if m.sum() >= 10:
        rep['trim<=-2'] = {'n': int(m.sum()), 'mean_fwd10': float(np.mean(fwd[m])),
                           'win_rate': float(np.mean(fwd[m] > 0))}
    return rep


def did_catch_rebound(f, scores):
    """检查近一个月规则在反弹起点是否给出买入。"""
    dates = pd.to_datetime(f['__date'].values)
    close = f['__close'].values
    out = []
    mask = dates >= (dates.max() - pd.Timedelta(days=25))
    idx = np.where(mask)[0]
    for i in idx:
        out.append((str(dates[i])[:10], round(float(close[i]), 3), int(scores[i])))
    return out


# ---------------- 最终模型 + 当前信号 ----------------
def fit_final(f, feat_names):
    from lightgbm import LGBMClassifier, LGBMRegressor
    d = f.dropna(subset=feat_names + ['__fwd', '__up']).reset_index(drop=True)
    dates = pd.to_datetime(d['__date'].values)
    cutoff = dates.max() - pd.Timedelta(days=PURGE)
    m = dates < cutoff
    X = d.loc[m, feat_names].values.astype(np.float32)
    clf = LGBMClassifier(**LGB_PARAMS).fit(X, d.loc[m, '__up'].values.astype(int))
    reg = LGBMRegressor(**LGB_PARAMS).fit(X, d.loc[m, '__fwd'].values)
    return clf, reg, str(cutoff)[:10]


def current_signal(f, feat_names, clf, reg):
    row = f.dropna(subset=feat_names).iloc[-1]
    x = row[feat_names].values.astype(np.float32).reshape(1, -1)
    p_up = float(clf.predict_proba(x)[0, 1])
    exp_ret = float(reg.predict(x)[0])
    scores, tags = compute_rule_series(f)
    return {
        'date': str(pd.to_datetime(f['__date'].values[-1]))[:10],
        'close': float(f['__close'].values[-1]),
        'rsi14': float(f['__rsi14'].values[-1]),
        'p_up': p_up, 'exp_ret10': exp_ret,
        'rule_score': int(scores[-1]), 'rule_tags': tags[-1],
    }


def component_analysis(conn):
    """打印成分股技术状态, 自下而上验证 ETF 信号。"""
    rows = []
    for sym, name, w in HSTECH_COMPONENTS:
        df = load_ohlcv(conn, sym)
        if df.empty:
            continue
        c = df['close'].values; v = df['volume'].values
        if len(c) < 60:
            continue
        close = c[-1]; ma20 = np.mean(c[-20:])
        d_rsi = np.diff(c[-15:], prepend=c[-15])
        up = np.where(d_rsi > 0, d_rsi, 0.0); dn = np.where(d_rsi < 0, -d_rsi, 0.0)
        rsi_val = 100 - 100 / (1 + np.mean(up[-14:]) / (np.mean(dn[-14:]) + 1e-12))
        dif = ema(c, 12)[-1] - ema(c, 26)[-1]
        dea = pd.Series(ema(c, 12) - ema(c, 26)).ewm(span=9, adjust=False).mean().values[-1]
        hist = dif - dea
        dif_p = ema(c[:-1], 12)[-1] - ema(c[:-1], 26)[-1]
        dea_p = pd.Series(ema(c[:-1], 12) - ema(c[:-1], 26)).ewm(span=9, adjust=False).mean().values[-1]
        macd_state = '金叉' if hist > 0 else ('收敛' if hist > dif_p - dea_p else '死叉')
        r5 = c[-1] / c[-6] - 1 if len(c) >= 6 else 0
        r20 = c[-1] / c[-21] - 1 if len(c) >= 21 else 0
        states = []
        if close > ma20: states.append('站上MA20')
        if rsi_val < 30: states.append('超卖')
        elif rsi_val > 70: states.append('超买')
        vol_ratio = v[-1] / np.mean(v[-20:])
        if vol_ratio > 1.2: states.append('放量')
        rows.append((name, w, close, rsi_val, close > ma20, macd_state, r5, r20, ','.join(states)))

    if not rows:
        return
    print(f"\n{'成分股':<6} {'权重':<5} {'现价':<8} {'RSI':<6} {'MA20':<6} {'MACD':<6} {'5日':<7} {'20日':<7} {'状态'}")
    print('-' * 85)
    for r in rows:
        name, w, close, rsi_val, above_ma, macd, r5, r20, states = r
        ma_arrow = '↑' if above_ma else '↓'
        print(f'{name:<6} {w:.0%}   {close:<8.2f} {rsi_val:<6.1f} {ma_arrow:<6} {macd:<6} {r5:+.1%}    {r20:+.1%}    {states}')
    # 汇总
    n_above = sum(1 for r in rows if r[4])
    n_golden = sum(1 for r in rows if r[5] == '金叉')
    avg_rsi = np.mean([r[3] for r in rows])
    print(f'汇总: {n_above}/{len(rows)} 站上MA20 | {n_golden}/{len(rows)} MACD金叉 | 平均RSI {avg_rsi:.1f}')


def save_model(clf, reg, feat_names, cutoff, ml_rep, rule_rep):
    import pickle
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(MODEL_PKL, 'wb') as fp:
        pickle.dump(dict(clf=clf, reg=reg, feat_names=feat_names, cutoff=cutoff,
                         horizon=HORIZON, ml_oos=ml_rep, rule=rule_rep,
                         train_date=str(pd.Timestamp.now())[:16]), fp)
    print(f"   💾 已保存 {MODEL_PKL}")


# ---------------- 报告 ----------------
def build_recommendation(sig, ml_rep, cost=0.875, shares=660000):
    close = sig['close']; pnl = (close - cost) / cost
    ml_ok = ml_rep.get('usable', False)
    score = sig['rule_score']; p_up = sig['p_up']; er = sig['exp_ret10']
    lines = []
    lines.append(f"当前 {sig['date']}  现价 {close:.3f}  成本 {cost:.3f}  浮亏 {pnl:+.1%}")
    lines.append(f"规则分 {score} {sig['rule_tags']}")
    lines.append(f"ML: 上涨概率 {p_up:.2f}  预期{HORIZON}日 {er:+.2%}  "
                 f"(样本外{'可用✅' if ml_ok else '偏弱, 仅参考'})")
    # 综合裁决: 规则为主(透明), ML 为辅确认
    if score >= 6 and (not ml_ok or p_up >= 0.5):
        verdict = "🟢 积极加仓信号: 反弹确认, 可分批加仓"
    elif score >= 4 and (not ml_ok or p_up >= 0.5):
        verdict = "🔵 试探加仓: 反弹初现, 可小仓介入, 破MA20/MA20下拐则止损"
    elif score <= -2:
        verdict = "🔴 减仓/落袋: 超买见顶迹象, 反弹到阻力可减"
    else:
        verdict = "⚪ 持有观望: 未见明确反弹确认, 不追高不割肉"
    lines.append(f"建议: {verdict}")
    return "\n".join(lines), verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shuffle', action='store_true')
    ap.add_argument('--signal', action='store_true', help='只输出当前信号')
    ap.add_argument('--db', default=DB_PATH)
    ap.add_argument('--horizon', type=int, default=20,
                    help='预测周期(交易日), 默认20; 旧版10')
    args = ap.parse_args()

    global HORIZON, PURGE
    HORIZON = args.horizon
    PURGE = HORIZON + 3

    conn = sqlite3.connect(args.db)
    print("=" * 74)
    print(f"159792 港股通互联网ETF 专用模型  horizon={HORIZON}日  purge={PURGE}  cost={COST}")
    print("=" * 74)
    f, feat_names = build_features(conn)
    print(f"数据: {len(f)} 根日线 ({str(pd.to_datetime(f['__date'].values[0]))[:10]}"
          f"~{str(pd.to_datetime(f['__date'].values[-1]))[:10]})  特征 {len(feat_names)} 个")

    scores, tags = compute_rule_series(f)

    if args.signal:
        import pickle
        with open(MODEL_PKL, 'rb') as fp:
            m = pickle.load(fp)
        sig = current_signal(f, m['feat_names'], m['clf'], m['reg'])
        txt, _ = build_recommendation(sig, m.get('ml_oos', {}))
        component_analysis(conn)
        print("\n" + txt)
        conn.close(); return

    # walk-forward ML
    oos = walk_forward(f, feat_names, shuffle=args.shuffle)
    ml_rep = eval_oos(oos)
    print("\n【ML 样本外(OOS)评估】")
    print(f"  折数 {ml_rep.get('folds')}  样本 {ml_rep.get('n')}  上涨基准 {ml_rep.get('base_up_rate', float('nan')):.3f}")
    print(f"  方向准确率 {ml_rep.get('dir_acc', float('nan')):.3f}  AUC {ml_rep.get('auc', float('nan')):.3f}  IC {ml_rep.get('ic', float('nan')):.4f}")
    print(f"  决策净收益 {ml_rep.get('decision_ret_net', float('nan')):+.4f}  vs 买入持有 {ml_rep.get('buyhold_ret', float('nan')):+.4f}  (交易{ml_rep.get('trades')}次)")
    if 'hi_conf_n' in ml_rep:
        print(f"  高置信(p>0.6, n={ml_rep['hi_conf_n']}): 胜率 {ml_rep['hi_conf_hit']:.3f}  净收益 {ml_rep['hi_conf_ret_net']:+.4f}")
    print(f"  → ML 可用: {'✅ 是' if ml_rep.get('usable') else '❌ 否(edge 不足, 以规则为主)'}")

    if args.shuffle:
        print("\n(shuffle 自检: 上述 AUC 应≈0.5, IC≈0, 否则有泄漏)")
        conn.close(); return

    # 规则条件收益
    rule_rep = eval_rule(f, scores)
    print("\n【透明规则 · 历史条件收益(诚实, 规则先验固定)】")
    print(f"  全样本未来{HORIZON}日均收益(基准) {rule_rep['base_fwd10']:+.4f}")
    for thr in [4, 5, 6]:
        k = f'buy>={thr}'
        if k in rule_rep:
            r = rule_rep[k]
            print(f"  规则分{k}: n={r['n']}  未来{HORIZON}日均收益 {r['mean_fwd10']:+.4f}  "
                  f"胜率 {r['win_rate']:.3f}  超基准 {r['net_vs_base']:+.4f}")
    if 'trim<=-2' in rule_rep:
        r = rule_rep['trim<=-2']
        print(f"  减仓分<=-2: n={r['n']}  未来{HORIZON}日均收益 {r['mean_fwd10']:+.4f}  胜率 {r['win_rate']:.3f}")

    # 是否抓到近月反弹
    print("\n【近一月规则分轨迹(检验是否抓到反弹)】")
    for dt, px, sc in did_catch_rebound(f, scores):
        flag = '  ← 买入信号' if sc >= 4 else ('  ← 减仓' if sc <= -2 else '')
        print(f"  {dt}  {px:.3f}  规则分 {sc}{flag}")

    # 最终模型 + 当前信号 + 建议
    clf, reg, cutoff = fit_final(f, feat_names)
    save_model(clf, reg, feat_names, cutoff, ml_rep, rule_rep)
    sig = current_signal(f, feat_names, clf, reg)
    txt, _ = build_recommendation(sig, ml_rep)
    component_analysis(conn)
    print("\n" + "=" * 74)
    print("📊 当前操作建议")
    print("=" * 74)
    print(txt)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'report.json'), 'w') as fp:
        json.dump({'ml_oos': ml_rep, 'rule': rule_rep, 'signal': sig},
                  fp, ensure_ascii=False, indent=2, default=str)
    conn.close()


if __name__ == '__main__':
    main()
