#!/usr/bin/env python3
"""
补仓顾问 ML — 方案2(单只涨跌预测) + 方案3(ATR三隘口 +EV 补仓决策)

严格纪律(防重蹈 IC 0.38 泄漏覆辙):
  1. 特征只用当下已知: 复用 FeaturePipeline.compute_stock(features.py:943),
     每只股票只用自身历史算特征。日频宏观/情绪/基本面在日线模型上因果成立。
  2. 绝不按行号/随机切分 —— 一律真时间 walk-forward。
  3. purge ≥ horizon(20交易日≈28日历天)—— 用 40 日历天 purge + embargo,
     剔除训练集里标签跨越切点、落入测试期的重叠样本。
  4. 只认样本外(OOS) + 带成本决策级回测,训练指标一律不信。

产出: python/models/add_advisor/{walkforward_summary.json, holdings_report.txt}

用法:
  python strategy/add_advisor_ml.py --quick     # 少股票少树, 冒烟自检
  python strategy/add_advisor_ml.py             # 完整 walk-forward + 持仓建议
  python strategy/add_advisor_ml.py --shuffle   # 打乱标签的泄漏自检(指标应塌回基准)
"""

import os, sys, json, argparse, warnings, time
import numpy as np
import pandas as pd
import sqlite3

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'python'))

DB_PATH = os.path.join(ROOT, 'data', 'stock_data.db')
OUT_DIR = os.path.join(ROOT, 'models', 'add_advisor')
MODEL_PKL = os.path.join(OUT_DIR, 'model.pkl')

import pickle
from strategy.features import FeaturePipeline
from lightgbm import LGBMRegressor, LGBMClassifier
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, mean_squared_error

# ============ 配置 ============
HORIZON = 20            # 预测周期: 20 交易日 ≈ 1 个月
ATR_TP = 2.5           # 止盈 = close + 2.5 × ATR14
ATR_SL = 1.5           # 止损 = close − 1.5 × ATR14
CAND_RSI = 40          # 候选态: RSI14 < 40
MIN_BARS = 400         # 进入训练池的最少日线根数
PURGE_DAYS = 40        # 日历天数, > horizon(≈28) + embargo, 防标签跨切点泄漏
TRAIN_MONTHS = 24
TEST_MONTHS = 3

# A股 0.13% 往返, 港股/ETF 0.3% 往返
COST_A = 0.0013
COST_HK = 0.003

HOLDINGS = [
    ('300124.SZ', '汇川技术', 1500, 65.883),
    ('600048.SH', '保利发展', 18600, 7.004),
    ('3690.HK',   '美团-W',   1300, 108.0),
    ('300015.SZ', '爱尔眼科', 8600, 11.391),
    ('159792.SZ', '港股通互联网ETF', 660000, 0.875),
]

REG_PARAMS = {
    'n_estimators': 600, 'learning_rate': 0.02, 'num_leaves': 31,
    'max_depth': 6, 'min_child_samples': 200, 'subsample': 0.7,
    'subsample_freq': 1, 'colsample_bytree': 0.5, 'reg_alpha': 1.0,
    'reg_lambda': 10.0, 'min_split_gain': 0.02, 'n_jobs': -1,
    'verbosity': -1, 'random_state': 42,
}
CLF_PARAMS = dict(REG_PARAMS)  # 同结构, 分类头

QUICK_PARAMS = {
    'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31,
    'max_depth': 5, 'n_jobs': -1, 'verbosity': -1, 'random_state': 42,
}


# ============ 原始指标(用于标签/候选态, 不经特征归一化) ============
def wilder_rsi(close, n=14):
    d = np.diff(close, prepend=close[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().values
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().values
    return 100 - 100 / (1 + ru / (rd + 1e-12))


def atr14(high, low, close, n=14):
    pc = np.roll(close, 1); pc[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))
    return pd.Series(tr).ewm(alpha=1/n, adjust=False).mean().values


def build_labels(o, h, l, c):
    """返回逐行标签数组(与输入等长, 未来不足处为 nan/False)。

    方案2: ret20(回归), sign(分类涨跌)
    方案3: tb_class(1=先触止盈 / 0=先触止损或到期未涨), tb_exit(实际退出收益)
    候选态: close<MA20 且 RSI14<40
    """
    n = len(c)
    atr = atr14(h, l, c)
    ma20 = pd.Series(c).rolling(20).mean().values
    rsi = wilder_rsi(c)

    ret20 = np.full(n, np.nan)
    tb_class = np.full(n, np.nan)
    tb_exit = np.full(n, np.nan)

    for t in range(n - HORIZON):
        base = c[t]
        ret20[t] = c[t + HORIZON] / base - 1.0
        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            # 无有效 ATR: 退化为纯到期收益符号
            tb_exit[t] = ret20[t]
            tb_class[t] = 1.0 if ret20[t] > 0 else 0.0
            continue
        tp = base + ATR_TP * a
        sl = base - ATR_SL * a
        hit = None
        for k in range(t + 1, t + HORIZON + 1):
            hi, lo = h[k], l[k]
            # 同一根内同时触及: 保守认定先触止损
            if lo <= sl:
                hit = ('SL', -ATR_SL * a / base); break
            if hi >= tp:
                hit = ('TP', ATR_TP * a / base); break
        if hit is None:
            r = c[t + HORIZON] / base - 1.0
            tb_class[t] = 1.0 if r > 0 else 0.0
            tb_exit[t] = r
        else:
            tb_class[t] = 1.0 if hit[0] == 'TP' else 0.0
            tb_exit[t] = hit[1]

    sign = np.where(np.isfinite(ret20), (ret20 > 0).astype(float), np.nan)
    candidate = (c < ma20) & (rsi < CAND_RSI)
    return dict(ret20=ret20, sign=sign, tb_class=tb_class, tb_exit=tb_exit,
                candidate=candidate, atr=atr, ma20=ma20, rsi=rsi)


# ============ 数据加载 + 特征池化 ============
def load_symbols(conn, quick, min_bars=MIN_BARS):
    rows = conn.execute(
        "SELECT symbol, COUNT(*) c FROM kline_daily "
        "WHERE symbol LIKE '%.SZ' OR symbol LIKE '%.SH' "
        "GROUP BY symbol HAVING c>=? ORDER BY symbol", (min_bars,)).fetchall()
    syms = [r[0] for r in rows]
    if quick:
        syms = syms[:40]
    return syms


def build_pool(conn, symbols, pipeline, quick):
    """逐股算特征+标签, 池化成一个大 DataFrame。

    返回: (df_all, feat_names)
      df_all 含特征列 + 元列 __sym/__date/__close/__ret20/__sign/
      __tbclass/__tbexit/__cand
    """
    frames = []
    feat_cols = set()
    t0 = time.time()
    for i, sym in enumerate(symbols):
        try:
            df = pd.read_sql(
                "SELECT date, open, high, low, close, volume FROM kline_daily "
                "WHERE symbol=? ORDER BY date", conn, params=(sym,))
            if len(df) < 200:
                continue
            df['date'] = pd.to_datetime(df['date'].astype(str).str.strip(),
                                        )
            df = (df.drop_duplicates('date', keep='last')
                    .sort_values('date').reset_index(drop=True))

            o = df['open'].values.astype(float)
            h = df['high'].values.astype(float)
            l = df['low'].values.astype(float)
            c = df['close'].values.astype(float)

            feats = pipeline.compute_stock(df, sym).ffill().fillna(0)
            lab = build_labels(o, h, l, c)

            feats = feats.reset_index(drop=True)
            feat_cols.update(feats.columns)
            feats['__sym'] = sym
            feats['__date'] = df['date'].values
            feats['__close'] = c
            feats['__ret20'] = lab['ret20']
            feats['__sign'] = lab['sign']
            feats['__tbclass'] = lab['tb_class']
            feats['__tbexit'] = lab['tb_exit']
            feats['__cand'] = lab['candidate']
            frames.append(feats)

            if (i + 1) % 50 == 0:
                print(f"   [{i+1}/{len(symbols)}] {len(frames)} 只已处理, "
                      f"{time.time()-t0:.0f}s")
        except Exception as e:
            if i < 3:
                print(f"   ⚠️ {sym}: {e}")
            continue

    feat_names = sorted(feat_cols)
    meta = ['__sym', '__date', '__close', '__ret20', '__sign',
            '__tbclass', '__tbexit', '__cand']
    aligned = []
    for f in frames:
        f = f.reindex(columns=feat_names + meta)
        aligned.append(f)
    df_all = pd.concat(aligned, ignore_index=True)
    df_all[feat_names] = (df_all[feat_names]
                          .apply(pd.to_numeric, errors='coerce')
                          .replace([np.inf, -np.inf], 0).fillna(0))
    print(f"   池化完成: {df_all.shape[0]:,} 行, {len(feat_names)} 特征, "
          f"{len(frames)} 只, {time.time()-t0:.0f}s")
    return df_all, feat_names


# ============ walk-forward 窗口 ============
def make_windows(dates):
    dmin, dmax = dates.min(), dates.max()
    purge = pd.Timedelta(days=PURGE_DAYS)
    train_off = pd.DateOffset(months=TRAIN_MONTHS)
    test_off = pd.DateOffset(months=TEST_MONTHS)
    wins = []
    test_start = dmin + train_off
    while test_start < dmax:
        test_end = test_start + test_off
        wins.append((dmin, test_start - purge, test_start, min(test_end, dmax)))
        test_start = test_end
    return wins


def per_date_rank_ic(dates, y_true, y_pred):
    """逐日横截面 spearman rank-IC 的均值与 ICIR"""
    df = pd.DataFrame({'d': dates, 't': y_true, 'p': y_pred})
    ics = []
    for _, g in df.groupby('d'):
        if len(g) >= 5 and g['t'].nunique() > 1 and g['p'].nunique() > 1:
            ic = spearmanr(g['t'], g['p'])[0]
            if np.isfinite(ic):
                ics.append(ic)
    if not ics:
        return np.nan, np.nan, 0
    ics = np.array(ics)
    icir = ics.mean() / (ics.std() + 1e-12)
    return float(ics.mean()), float(icir), len(ics)


# ============ walk-forward 主流程 ============
def run_walkforward(df_all, feat_names, params, shuffle=False):
    X_all = df_all[feat_names].values.astype(np.float32)
    dates = df_all['__date'].values.astype('datetime64[ns]')
    dser = pd.to_datetime(dates)
    sym = df_all['__sym'].values
    ret20 = df_all['__ret20'].values.astype(float)
    sign = df_all['__sign'].values
    tbclass = df_all['__tbclass'].values
    tbexit = df_all['__tbexit'].values
    cand = df_all['__cand'].values.astype(bool)

    if shuffle:
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(ret20))
        ret20, sign, tbclass, tbexit = ret20[perm], sign[perm], tbclass[perm], tbexit[perm]

    wins = make_windows(dser)
    print(f"\n🔁 walk-forward: {len(wins)} 个窗口 "
          f"(训练{TRAIN_MONTHS}月 / purge{PURGE_DAYS}天 / 测试{TEST_MONTHS}月)")

    # 汇总 OOS 预测(方案2回归)与决策记录
    oos = {'date': [], 'sym': [], 'ret20': [], 'sign': [],
           'reg_pred': [], 'sign_prob': [],
           'tbclass': [], 'tbexit': [], 'cand': [], 'tp_prob': []}

    for wi, (tr0, tr1, te0, te1) in enumerate(wins):
        tr = (dser >= tr0) & (dser < tr1) & np.isfinite(ret20) & np.isfinite(tbclass)
        te = (dser >= te0) & (dser < te1) & np.isfinite(ret20) & np.isfinite(tbclass)
        if tr.sum() < 5000 or te.sum() < 200:
            continue

        Xtr, Xte = X_all[tr], X_all[te]

        # 方案2: 回归 ret20 + 分类 sign
        reg = LGBMRegressor(**params).fit(Xtr, ret20[tr])
        clf_s = LGBMClassifier(**params).fit(Xtr, sign[tr].astype(int))
        # 方案3: 分类 三隘口先触止盈(全 episode 训练)
        clf_tb = LGBMClassifier(**params).fit(Xtr, tbclass[tr].astype(int))

        reg_p = reg.predict(Xte)
        sign_p = clf_s.predict_proba(Xte)[:, 1]
        tp_p = clf_tb.predict_proba(Xte)[:, 1]

        oos['date'].extend(dser[te]); oos['sym'].extend(sym[te])
        oos['ret20'].extend(ret20[te]); oos['sign'].extend(sign[te])
        oos['reg_pred'].extend(reg_p); oos['sign_prob'].extend(sign_p)
        oos['tbclass'].extend(tbclass[te]); oos['tbexit'].extend(tbexit[te])
        oos['cand'].extend(cand[te]); oos['tp_prob'].extend(tp_p)

        if (wi + 1) % 5 == 0 or wi == len(wins) - 1:
            print(f"   窗口 {wi+1}/{len(wins)}  train={tr.sum():,} "
                  f"test={te.sum():,}  @{str(te0)[:7]}")

    for k in oos:
        oos[k] = np.array(oos[k])
    return oos


# ============ 诚实带成本评估 ============
def evaluate(oos):
    rep = {}
    n = len(oos['ret20'])
    if n == 0:
        return {'error': 'no OOS samples'}

    # ---- 方案2 ----
    base_up = float(np.mean(oos['sign']))                 # 全样本上涨基准率
    pred_up = oos['sign_prob'] > 0.5
    acc = float(np.mean(pred_up == (oos['sign'] > 0.5)))
    try:
        auc = float(roc_auc_score(oos['sign'].astype(int), oos['sign_prob']))
    except Exception:
        auc = np.nan
    rmse = float(np.sqrt(mean_squared_error(oos['ret20'], oos['reg_pred'])))
    ic_mean, icir, ndays = per_date_rank_ic(oos['date'], oos['ret20'], oos['reg_pred'])

    # 决策回测: 预测上涨才持有20天(扣一次往返成本), 否则空仓
    hold = pred_up
    strat_ret = np.where(hold, oos['ret20'] - COST_A, 0.0)
    bh_ret = oos['ret20']  # 无差别买入持有(每个样本都进)
    rep['approach2'] = {
        'n': n, 'base_up_rate': base_up, 'dir_acc': acc, 'auc': auc,
        'reg_rmse': rmse, 'per_date_rankIC': ic_mean, 'ICIR': icir, 'n_days': ndays,
        'decision_mean_ret_net': float(np.mean(strat_ret)),
        'buyhold_mean_ret': float(np.mean(bh_ret)),
        'decision_trades': int(hold.sum()),
        'usable': bool(np.isfinite(auc) and auc > 0.53 and acc > base_up + 0.01
                       and np.isfinite(ic_mean) and ic_mean > 0.01),
    }

    # ---- 方案3: 仅在候选态子样本评估/应用 ----
    cm = oos['cand']
    c_tp = oos['tbclass'][cm]
    c_exit = oos['tbexit'][cm]
    c_prob = oos['tp_prob'][cm]
    nc = len(c_tp)
    if nc >= 50:
        base_tp = float(np.mean(c_tp))                    # 候选态里 TP-first 基准率
        base_exp = float(np.mean(c_exit))                 # 无差别买跌期望(未扣成本)
        # top decile 高概率档
        thr = np.quantile(c_prob, 0.9)
        top = c_prob >= thr
        top_prec = float(np.mean(c_tp[top])) if top.sum() else np.nan
        top_exp_net = float(np.mean(c_exit[top]) - COST_A) if top.sum() else np.nan
        # 决策回测: P(TP)>阈值才补, 对照"每次超卖都补"基线(均扣成本)
        add_ret = c_exit[top] - COST_A
        baseline_ret = c_exit - COST_A
        rep['approach3'] = {
            'n_candidate': nc, 'base_tp_rate': base_tp,
            'base_exit_exp_gross': base_exp,
            'top_decile_thr': float(thr), 'top_n': int(top.sum()),
            'top_precision': top_prec, 'top_exp_net': top_exp_net,
            'top_win_rate': float(np.mean(add_ret > 0)) if top.sum() else np.nan,
            'baseline_exp_net': float(np.mean(baseline_ret)),
            'usable': bool(top.sum() >= 20 and np.isfinite(top_exp_net)
                           and top_exp_net > 0 and top_exp_net > np.mean(baseline_ret)),
        }
        # 分年查衰减
        yr = pd.to_datetime(oos['date'][cm]).year
        by_year = {}
        for y in np.unique(yr):
            m = (yr == y) & top
            if m.sum() >= 10:
                by_year[int(y)] = {
                    'n': int(m.sum()),
                    'precision': float(np.mean(c_tp[m])),
                    'exp_net': float(np.mean(c_exit[m]) - COST_A),
                }
        rep['approach3']['by_year'] = by_year
    else:
        rep['approach3'] = {'n_candidate': nc, 'note': '候选态样本不足, 无法评估'}

    return rep


def print_report(rep):
    print("\n" + "=" * 74)
    print("📋 样本外(OOS)诚实评估 — 训练指标一律不看")
    print("=" * 74)
    a2 = rep.get('approach2', {})
    print("\n【方案2 · 单只涨跌预测】")
    print(f"  样本 {a2.get('n'):,} | 上涨基准率 {a2.get('base_up_rate'):.3f}")
    print(f"  方向准确率 {a2.get('dir_acc'):.3f} (需 > 基准+0.01)  AUC {a2.get('auc'):.3f}")
    print(f"  回归 per-date rank-IC {a2.get('per_date_rankIC'):.4f}  ICIR {a2.get('ICIR'):.2f}  ({a2.get('n_days')}天)")
    print(f"  决策净收益(扣成本) {a2.get('decision_mean_ret_net'):+.4f}  vs 买入持有 {a2.get('buyhold_mean_ret'):+.4f}")
    print(f"  → 可用: {'✅ 是' if a2.get('usable') else '❌ 否'}")

    a3 = rep.get('approach3', {})
    print("\n【方案3 · 候选态(超卖跌破MA20)三隘口补仓】")
    if a3.get('n_candidate', 0) < 50:
        print(f"  候选态样本 {a3.get('n_candidate')} 不足, 跳过")
    else:
        print(f"  候选态样本 {a3['n_candidate']:,} | TP-first 基准率 {a3['base_tp_rate']:.3f}")
        print(f"  无差别买跌期望(毛) {a3['base_exit_exp_gross']:+.4f}")
        print(f"  高概率档(top10%, n={a3['top_n']}): 精度 {a3['top_precision']:.3f}  "
              f"胜率 {a3['top_win_rate']:.3f}")
        print(f"  高概率档期望(扣成本) {a3['top_exp_net']:+.4f}  vs 无差别买跌基线 {a3['baseline_exp_net']:+.4f}")
        if a3.get('by_year'):
            print("  分年(高概率档): " + "  ".join(
                f"{y}:{v['exp_net']:+.3f}(n{v['n']})" for y, v in a3['by_year'].items()))
        print(f"  → 可用: {'✅ 是' if a3.get('usable') else '❌ 否'}")
    print("=" * 74)


# ============ 应用到 5 只持仓 ============
def fit_final(df_all, feat_names, params):
    """用截至最近(留 PURGE_DAYS embargo)的数据训最终模型"""
    dser = pd.to_datetime(df_all['__date'].values)
    cutoff = dser.max() - pd.Timedelta(days=PURGE_DAYS)
    m = (dser < cutoff) & np.isfinite(df_all['__ret20'].values) & np.isfinite(df_all['__tbclass'].values)
    X = df_all.loc[m, feat_names].values.astype(np.float32)
    reg = LGBMRegressor(**params).fit(X, df_all.loc[m, '__ret20'].values)
    clf_s = LGBMClassifier(**params).fit(X, df_all.loc[m, '__sign'].values.astype(int))
    clf_tb = LGBMClassifier(**params).fit(X, df_all.loc[m, '__tbclass'].values.astype(int))
    return reg, clf_s, clf_tb, str(cutoff)[:10]


def score_holding(conn, pipeline, sym, feat_names, reg, clf_s, clf_tb, tail=0):
    """对单只股票用最新一根 bar 打分。

    tail=0 读全历史(默认, 保证最后一行特征与训练逐位一致)。
    注意: tail>0 的截尾窗口对部分股票会改变最后一行特征(某些特征用
    expanding/全历史统计), 已 validate 出偏差, 故扫描一律 tail=0。
    """
    if tail and tail > 0:
        df = pd.read_sql(
            "SELECT date, open, high, low, close, volume FROM kline_daily "
            "WHERE symbol=? ORDER BY date DESC LIMIT ?", conn, params=(sym, int(tail)))
        df = df.iloc[::-1].reset_index(drop=True)
    else:
        df = pd.read_sql(
            "SELECT date, open, high, low, close, volume FROM kline_daily "
            "WHERE symbol=? ORDER BY date", conn, params=(sym,))
    if len(df) < 120:
        return None
    df['date'] = pd.to_datetime(df['date'].astype(str).str.strip())
    df = df.drop_duplicates('date', keep='last').sort_values('date').reset_index(drop=True)
    c = df['close'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)

    feats = pipeline.compute_stock(df, sym).ffill().fillna(0)
    x = feats.reindex(columns=feat_names, fill_value=0).iloc[[-1]]
    x = x.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], 0).fillna(0)
    xv = x.values.astype(np.float32)

    atr = atr14(h, l, c)[-1]
    ma20 = pd.Series(c).rolling(20).mean().values[-1]
    rsi = wilder_rsi(c)[-1]
    last = c[-1]
    cand = (last < ma20) and (rsi < CAND_RSI)

    return dict(
        sym=sym, last=last, atr=atr, ma20=ma20, rsi=rsi, cand=cand,
        date=str(df['date'].iloc[-1])[:10],
        reg=float(reg.predict(xv)[0]),
        pup=float(clf_s.predict_proba(xv)[0, 1]),
        ptp=float(clf_tb.predict_proba(xv)[0, 1]),
        tp_price=last + ATR_TP * atr, sl_price=last - ATR_SL * atr,
    )


def scan_universe(conn, pipeline, feat_names, reg, clf_s, clf_tb,
                  symbols=None, tail=0, min_bars=250, progress=None):
    """对整个 A 股票池逐只打分(低内存: 每只读完即 gc 释放)。

    模型真实 edge 只在 A 股(港股/ETF 缺宏观情绪特征), 故只扫 .SZ/.SH。
    tail=0 读全历史(保证特征与训练逐位一致); OOM 的真凶是 273MB lstm pkl,
    已用瘦身版(lstm_slim)解决, 全历史日线读取本身不占内存。
    返回 list[dict], 每项含 score_holding 的字段 + name(若 stock_info 有)。
    """
    import gc
    if symbols is None:
        rows = conn.execute(
            "SELECT symbol, COUNT(*) c FROM kline_daily "
            "WHERE symbol LIKE '%.SZ' OR symbol LIKE '%.SH' "
            "GROUP BY symbol HAVING c>=? ORDER BY symbol", (min_bars,)).fetchall()
        symbols = [r[0] for r in rows]

    name_map = {}
    try:
        name_map = {r[0]: r[1] for r in
                    conn.execute("SELECT symbol, name FROM stock_info").fetchall()}
    except Exception:
        pass

    out = []
    n = len(symbols)
    for i, sym in enumerate(symbols):
        try:
            s = score_holding(conn, pipeline, sym, feat_names, reg, clf_s, clf_tb, tail=tail)
        except Exception:
            s = None
        if s is not None:
            s['name'] = name_map.get(sym, sym)
            out.append(s)
        if progress and (i + 1) % 50 == 0:
            progress(i + 1, n, len(out))
        gc.collect()
    return out


def load_holdings(conn):
    """从 positions 表读持仓; 失败则回退到硬编码 HOLDINGS"""
    try:
        df = pd.read_sql(
            "SELECT symbol, stock_name, shares, cost_price FROM positions", conn)
        if len(df):
            return [(r['symbol'], r['stock_name'], r['shares'], r['cost_price'])
                    for _, r in df.iterrows()]
    except Exception:
        pass
    return HOLDINGS


def save_final_model(reg, clf_s, clf_tb, feat_names, cutoff, rep):
    os.makedirs(OUT_DIR, exist_ok=True)
    data = {
        'reg': reg, 'clf_s': clf_s, 'clf_tb': clf_tb,
        'feat_names': feat_names, 'cutoff': cutoff,
        'horizon': HORIZON, 'atr_tp': ATR_TP, 'atr_sl': ATR_SL, 'cand_rsi': CAND_RSI,
        'a2_usable': rep.get('approach2', {}).get('usable', False),
        'a3_usable': rep.get('approach3', {}).get('usable', False),
        'train_date': str(pd.Timestamp.now())[:16],
        'oos_summary': rep,
    }
    with open(MODEL_PKL, 'wb') as f:
        pickle.dump(data, f)
    print(f"   💾 最终模型已保存: {MODEL_PKL} ({os.path.getsize(MODEL_PKL)/1024/1024:.1f} MB)")


def load_final_model():
    if not os.path.exists(MODEL_PKL):
        raise FileNotFoundError(
            f"未找到模型 {MODEL_PKL} — 请先在 Mac 跑完整训练生成, 再 scp 到服务器")
    with open(MODEL_PKL, 'rb') as f:
        return pickle.load(f)


def holdings_report(conn, pipeline, feat_names, reg, clf_s, clf_tb, cutoff, rep,
                    holdings=None):
    a2_ok = rep.get('approach2', {}).get('usable', False)
    a3_ok = rep.get('approach3', {}).get('usable', False)
    if holdings is None:
        holdings = load_holdings(conn)
    lines = []
    lines.append("=" * 74)
    lines.append(f"📊 补仓顾问(ML) — 持仓体检  生成@{str(pd.Timestamp.now())[:16]}")
    lines.append(f"   最终模型训练截至 {cutoff} (留 {PURGE_DAYS}天 embargo)")
    lines.append(f"   预测周期 {HORIZON} 交易日 | 三隘口 TP=+{ATR_TP}×ATR / SL=−{ATR_SL}×ATR")
    lines.append(f"   模型可用性: 方案2 {'✅' if a2_ok else '❌薄/不可用'}  "
                 f"方案3 {'✅' if a3_ok else '❌薄/不可用'}")
    lines.append("   ⚠️ edge 薄 + 港股/ETF 无宏观情绪特征(填0)+ 5只样本少→结论靠池化外推, 仅辅助")
    lines.append("=" * 74)

    for sym, name, shares, cost in holdings:
        s = score_holding(conn, pipeline, sym, feat_names, reg, clf_s, clf_tb)
        lines.append("\n" + "-" * 74)
        if s is None:
            lines.append(f"  {name} ({sym}): 数据不足, 跳过")
            continue
        pnl = (s['last'] - cost) / cost * 100
        lines.append(f"  {name} ({sym})  数据@{s['date']}")
        lines.append(f"    现价 {s['last']:.3f}  成本 {cost:.3f}  盈亏 {pnl:+.1f}%  "
                     f"RSI {s['rsi']:.0f}  {'[补仓候选态: 跌破MA20+超卖]' if s['cand'] else '[非候选态]'}")
        lines.append(f"    方案2: 预测20日收益 {s['reg']:+.2%}  上涨概率 {s['pup']:.2f}")
        lines.append(f"    方案3: P(先触止盈) {s['ptp']:.2f}  "
                     f"止盈位 {s['tp_price']:.3f}  止损位 {s['sl_price']:.3f}")

        # 综合建议
        verdict = _verdict(s, a2_ok, a3_ok)
        lines.append(f"    建议: {verdict}")

    lines.append("\n" + "=" * 74)
    lines.append("原则: 模型只做辅助排序。edge 薄, 补仓与否仍以纪律(破位止损/不接飞刀)为先。")
    lines.append("=" * 74)
    txt = "\n".join(lines)
    print("\n" + txt)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'holdings_report.txt'), 'w') as f:
        f.write(txt)


def _verdict(s, a2_ok, a3_ok):
    if not s['cand']:
        base = "非候选态(未跌破MA20或未超卖), 不属'补仓'语境; "
    else:
        base = ""
    # 方案3 优先(补仓语境); 不可用则降级为观望
    if s['cand'] and a3_ok:
        if s['ptp'] >= 0.55 and s['reg'] > 0:
            return base + f"⚠️可小仓试探(P(TP)={s['ptp']:.2f}偏高), 严格按止损位{s['sl_price']:.3f}离场"
        return base + f"❌不补(P(TP)={s['ptp']:.2f}不占优), 等企稳再看"
    if a2_ok:
        if s['pup'] >= 0.55 and s['reg'] > 0:
            return base + f"偏多(上涨概率{s['pup']:.2f}), 但edge薄仅供参考"
        return base + f"偏空/中性(上涨概率{s['pup']:.2f}), 不宜补"
    return base + "模型edge不足以支撑补仓决策, 建议按规则(position_advisor)为准, 别赌反弹"


# ============ main ============
def score_only(db_path):
    """每日轻量模式: 加载已训模型, 只对当前持仓打分 (秒级, 不重训)。"""
    data = load_final_model()
    print("=" * 74)
    print(f"📅 每日打分模式  模型训练于 {data.get('train_date')} "
          f"(截至 {data.get('cutoff')})")
    print("=" * 74)
    conn = sqlite3.connect(db_path)
    pipeline = FeaturePipeline({
        'label': '日线', 'horizon': data['horizon'], 'db_table': 'kline_daily',
        'min_history': 120, 'purged_gap': PURGE_DAYS, 'north_shift_days': 1,
    })
    holdings = load_holdings(conn)
    holdings_report(conn, pipeline, data['feat_names'], data['reg'],
                    data['clf_s'], data['clf_tb'], data['cutoff'],
                    data.get('oos_summary', {}), holdings=holdings)
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='少股票少树冒烟')
    ap.add_argument('--shuffle', action='store_true', help='打乱标签泄漏自检')
    ap.add_argument('--score-only', action='store_true',
                    help='每日轻量: 加载已训模型只打分, 不重训')
    ap.add_argument('--db', default=DB_PATH)
    args = ap.parse_args()

    if args.score_only:
        score_only(args.db)
        return

    params = QUICK_PARAMS if args.quick else REG_PARAMS
    print("=" * 74)
    print(f"🔧 补仓顾问 ML  quick={args.quick}  shuffle={args.shuffle}")
    print(f"   horizon={HORIZON}  purge={PURGE_DAYS}d  pool>={MIN_BARS}bars")
    print("=" * 74)

    conn = sqlite3.connect(args.db)
    pipeline = FeaturePipeline({
        'label': '日线', 'horizon': HORIZON, 'db_table': 'kline_daily',
        'min_history': 120, 'purged_gap': PURGE_DAYS, 'north_shift_days': 1,
    })

    symbols = load_symbols(conn, args.quick)
    print(f"📈 训练池: {len(symbols)} 只 (A股 >= {MIN_BARS} 根日线)")

    df_all, feat_names = build_pool(conn, symbols, pipeline, args.quick)

    oos = run_walkforward(df_all, feat_names, params, shuffle=args.shuffle)
    rep = evaluate(oos)
    print_report(rep)

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = 'shuffle' if args.shuffle else ('quick' if args.quick else 'full')
    with open(os.path.join(OUT_DIR, f'walkforward_summary_{tag}.json'), 'w') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n💾 已保存 walkforward_summary_{tag}.json")

    if args.shuffle:
        print("\n(shuffle 自检模式: 上述指标应塌回基准/0, 否则说明仍有泄漏)")
        conn.close()
        return

    # 最终模型 + 保存 + 应用到持仓
    print("\n🏗️ 训练最终模型并应用到持仓...")
    reg, clf_s, clf_tb, cutoff = fit_final(df_all, feat_names, params)
    save_final_model(reg, clf_s, clf_tb, feat_names, cutoff, rep)
    holdings_report(conn, pipeline, feat_names, reg, clf_s, clf_tb, cutoff, rep)
    conn.close()


if __name__ == '__main__':
    main()
