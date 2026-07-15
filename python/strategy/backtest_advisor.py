#!/usr/bin/env python3
"""
诚实盈利回测 — 基于 add_advisor 模型的 OOS walk-forward 输出聚合组合净值曲线。

为什么这么做(诚实口径):
  - add_advisor 的 edge 是【横截面】的(rank-IC≈0.053), 不是绝对择时。
  - 单只 long-only 择时 OOS 净收益 +0.71%/笔 < 买入持有 +1.35%(A股上行漂移,
    择时踏空)—— 所以本脚本【如实】输出该对照曲线, 让它跑输, 不藏拙。
  - rank-IC 的正确变现 = 每 20 交易日横截面按预测收益排序, long 高分档
    (相对全市场 / 或 long-short 市场中性), 扣成本后复利。这才是"能不能赚钱"。
  - 方案3 候选态(跌破MA20+超卖) top-decile 扣成本 +2.2%/笔, 单列一条曲线。

关键防坑:
  - 【不重叠 rebalance】: horizon=20, 若每日 rebalance 则 20 日持有期重叠,
    复利会严重虚高。故只在【每 20 个交易日】取一个 rebalance 日, 用当日横截面
    的已实现 20 日收益(ret20 本就是前视已实现值)复利 → 期间不重叠, 复利合法。
  - 全部复用 run_walkforward 的 OOS 输出, 不额外训练; 服务器不跑, Mac 算好入 git。

产出:
  models/add_advisor/backtest_portfolio.json  (组合净值曲线+指标+分年, 小)
  models/add_advisor/backtest_signals.json    (每股 OOS 摘要+月度下采样序列)

用法:
  python strategy/backtest_advisor.py --quick   # 40只少树冒烟
  python strategy/backtest_advisor.py           # 全 A 股池完整回测
"""

import os, sys, json, argparse, warnings, time
import numpy as np
import pandas as pd
import sqlite3

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'python'))

from strategy.features import FeaturePipeline
from strategy.add_advisor_ml import (
    build_pool, run_walkforward, load_symbols,
    HORIZON, PURGE_DAYS, OUT_DIR,
    REG_PARAMS, QUICK_PARAMS,
)
from strategy.costs import roundtrip_frac

# 回测池是 A 股个股(见 caveat "仅A股"),往返成本用真实费率分数
# (佣金万2.5双边 + 印花0.05%卖出 + 过户0.001%双边 = 0.102%),
# 取代旧的粗略 flat 0.13%,与纸面引擎 costs.py 同源。
COST_A = roundtrip_frac(etf=False)

PORTFOLIO_JSON = os.path.join(OUT_DIR, 'backtest_portfolio.json')
SIGNALS_JSON = os.path.join(OUT_DIR, 'backtest_signals.json')
OOS_CACHE = os.path.join(OUT_DIR, 'oos_cache.pkl')  # 原始 OOS, 供 --reaggregate 免训重算

TOP_Q = 0.20          # 横截面前 20% 做多
BOT_Q = 0.20          # 后 20% 做空 (long-short)
MIN_NAMES = 20        # 一个 rebalance 日至少这么多股票才成截面
CAND_TOP_DECILE = 0.90  # 方案3 候选态 P(TP) 分位阈


# ============ 组合聚合 ============
def _curve_metrics(period_rets, dates, periods_per_year):
    """由每期(不重叠)净收益序列算净值曲线 + 指标。"""
    period_rets = np.asarray(period_rets, dtype=float)
    if len(period_rets) == 0:
        return {'curve': [], 'total_return': 0.0, 'annual_return': 0.0,
                'sharpe': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0,
                'avg_period_ret': 0.0, 'n_periods': 0}
    nav = np.cumprod(1.0 + period_rets)
    # 净值曲线从 1.0 基准起, 每期末记一个点
    curve = [{'date': str(pd.Timestamp(dates[0]))[:10], 'value': 1.0}]
    curve += [{'date': str(pd.Timestamp(d))[:10], 'value': round(float(v), 4)}
              for d, v in zip(dates, nav)]
    total = float(nav[-1] - 1.0)
    n = len(period_rets)
    years = n / periods_per_year if periods_per_year else 1.0
    annual = float((nav[-1]) ** (1.0 / max(years, 1e-9)) - 1.0) if nav[-1] > 0 else -1.0
    mu, sd = float(np.mean(period_rets)), float(np.std(period_rets))
    sharpe = float(mu / sd * np.sqrt(periods_per_year)) if sd > 1e-12 else 0.0
    peak = np.maximum.accumulate(nav)
    mdd = float(np.min(nav / peak - 1.0)) if len(nav) else 0.0
    return {
        'curve': curve,
        'total_return': round(total, 4),
        'annual_return': round(annual, 4),
        'sharpe': round(sharpe, 3),
        'max_drawdown': round(mdd, 4),
        'win_rate': round(float(np.mean(period_rets > 0)), 4),
        'avg_period_ret': round(mu, 5),
        'n_periods': n,
    }


def build_portfolio(oos):
    """由 OOS 输出构建各策略的不重叠 20 日 rebalance 净值曲线。"""
    date = pd.to_datetime(oos['date'])
    reg = oos['reg_pred'].astype(float)
    ret = oos['ret20'].astype(float)
    sign_prob = oos['sign_prob'].astype(float)
    cand = oos['cand'].astype(bool)
    tbexit = oos['tbexit'].astype(float)
    tp_prob = oos['tp_prob'].astype(float)

    uniq = np.array(sorted(pd.unique(date)))
    # 每 HORIZON 个交易日取一个不重叠 rebalance 日
    rb_dates = uniq[::HORIZON]

    top_r, ls_r, uni_r, cand_r = [], [], [], []
    kept_dates = []
    # 分年累积 (top-K 与 long-short 的每期净收益)
    by_year = {}

    for d in rb_dates:
        m = date.values == d
        r = ret[m]
        p = reg[m]
        if len(r) < MIN_NAMES:
            continue
        order = np.argsort(-p)  # 高分在前
        k = max(1, int(len(r) * TOP_Q))
        kb = max(1, int(len(r) * BOT_Q))
        top_idx = order[:k]
        bot_idx = order[-kb:]

        top_net = float(np.mean(r[top_idx])) - COST_A          # 做多高分, 扣一次往返
        bot_mean = float(np.mean(r[bot_idx]))
        ls_net = (float(np.mean(r[top_idx])) - bot_mean) - 2 * COST_A  # 多空扣双边
        uni_mean = float(np.mean(r))                            # 全市场等权基准(不扣)

        # 方案3: 候选态里 P(TP) top-decile, 用 tbexit 扣成本
        cm = cand[m]
        if cm.sum() >= 5:
            cp = tp_prob[m][cm]
            ce = tbexit[m][cm]
            thr = np.quantile(cp, CAND_TOP_DECILE)
            sel = cp >= thr
            cand_net = float(np.mean(ce[sel])) - COST_A if sel.sum() else 0.0
        else:
            cand_net = 0.0

        top_r.append(top_net); ls_r.append(ls_net); uni_r.append(uni_mean)
        cand_r.append(cand_net)
        kept_dates.append(d)

        y = int(pd.Timestamp(d).year)
        by_year.setdefault(y, {'top': [], 'ls': [], 'uni': []})
        by_year[y]['top'].append(top_net)
        by_year[y]['ls'].append(ls_net)
        by_year[y]['uni'].append(uni_mean)

    # 每年 HORIZON=20 交易日 → ~244/20 ≈ 12 期/年
    ppy = 244.0 / HORIZON
    strategies = {
        'top_k':   _curve_metrics(top_r, kept_dates, ppy),
        'long_short': _curve_metrics(ls_r, kept_dates, ppy),
        'universe': _curve_metrics(uni_r, kept_dates, ppy),
        'candidate_a3': _curve_metrics(cand_r, kept_dates, ppy),
    }

    # 诚实对照 (逐笔, 非复利): 真·单只择时 vs 无差别买入持有。
    # 用【全部 OOS 逐笔样本】(不是每期分散组合), 才是"单只该不该择时"的真问题。
    # 全池实测 ≈ 打平 (无择时 edge); edge 只在横截面 (见 top_k / long_short)。
    up_all = sign_prob > 0.5
    sn_timing = float(np.mean(ret[up_all]) - COST_A) if up_all.sum() else 0.0
    sn_buyhold = float(np.mean(ret))
    single_name = {
        'timing_net_per_trade': round(sn_timing, 5),
        'buyhold_net_per_trade': round(sn_buyhold, 5),
        'timing_edge_per_trade': round(sn_timing - sn_buyhold, 5),
        'n_trades': int(len(ret)),
        'n_timing_trades': int(up_all.sum()),
    }

    year_tbl = []
    for y in sorted(by_year):
        v = by_year[y]
        year_tbl.append({
            'year': y,
            'top_avg': round(float(np.mean(v['top'])), 5),
            'ls_avg': round(float(np.mean(v['ls'])), 5),
            'uni_avg': round(float(np.mean(v['uni'])), 5),
            'n_periods': len(v['top']),
        })

    top = strategies['top_k']
    uni = strategies['universe']
    return {
        'strategies': strategies,
        'single_name': single_name,
        'by_year': year_tbl,
        'headline': {
            'top_k_total': top['total_return'],
            'universe_total': uni['total_return'],
            'excess_total': round(top['total_return'] - uni['total_return'], 4),
            'long_short_total': strategies['long_short']['total_return'],
            'long_short_sharpe': strategies['long_short']['sharpe'],
            'top_k_excess_per_period': round(top['avg_period_ret'] - uni['avg_period_ret'], 5),
            'top_k_annual': top['annual_return'],
            'top_k_sharpe': top['sharpe'],
            'top_k_maxdd': top['max_drawdown'],
            'single_name_timing_edge': single_name['timing_edge_per_trade'],
        },
        'config': {
            'horizon': HORIZON, 'top_quantile': TOP_Q, 'bot_quantile': BOT_Q,
            'cost_roundtrip': COST_A, 'rebalance_days': HORIZON,
            'n_rebalances': len(kept_dates),
            'span': [str(pd.Timestamp(kept_dates[0]))[:10],
                     str(pd.Timestamp(kept_dates[-1]))[:10]] if kept_dates else [],
        },
        'caveat': ('edge 是横截面的(rank-IC≈0.05): 盈利来自"每20交易日 long 预测高分档 '
                   '相对全池等权(超额)"或 long-short 市场中性。⚠️绝对收益(top-K/基准 的总收益)'
                   '被幸存者偏差吹高(池=至今仍活跃的A股), 请看【超额/long-short】而非绝对数。'
                   '单只择时逐笔≈打平(无择时edge), 别指望单只择时赚钱。'
                   '仅A股, 已扣成本(往返0.13%), 不重叠20日 rebalance。'),
        'generated_at': time.strftime('%Y-%m-%d %H:%M'),
    }


def build_signals(oos, max_syms=None):
    """每股 OOS 摘要 + 月度下采样序列, 供 per-symbol 页。"""
    date = pd.to_datetime(oos['date'])
    sym = oos['sym']
    reg = oos['reg_pred'].astype(float)
    ret = oos['ret20'].astype(float)
    sign = oos['sign'].astype(float)
    sign_prob = oos['sign_prob'].astype(float)

    out = {}
    for s in pd.unique(sym):
        m = sym == s
        if m.sum() < 10:
            continue
        d = date[m]; rg = reg[m]; rt = ret[m]; sg = sign[m]; sp = sign_prob[m]
        pred_up = sp > 0.5
        dir_acc = float(np.mean(pred_up == (sg > 0.5)))
        hit_up = float(np.mean(rt[pred_up] > 0)) if pred_up.sum() else None
        mean_up_net = (float(np.mean(rt[pred_up])) - COST_A) if pred_up.sum() else None
        # 月度下采样序列 (控体积)
        dfm = pd.DataFrame({'d': d, 'reg': rg, 'ret': rt})
        dfm['ym'] = dfm['d'].dt.to_period('M')
        g = dfm.groupby('ym').last().reset_index()
        series = [{'date': str(row['d'])[:10],
                   'pred': round(float(row['reg']), 4),
                   'actual': round(float(row['ret']), 4)}
                  for _, row in g.iterrows()]
        out[str(s)] = {
            'n': int(m.sum()),
            'dir_acc': round(dir_acc, 4),
            'hit_rate_up': round(hit_up, 4) if hit_up is not None else None,
            'mean_ret_up_net': round(mean_up_net, 5) if mean_up_net is not None else None,
            'series': series,
        }
        if max_syms and len(out) >= max_syms:
            break
    return out


def main():
    import pickle
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='40只少树冒烟')
    ap.add_argument('--reaggregate', action='store_true',
                    help='免训: 读 oos_cache.pkl 重算 JSON (改了聚合口径后用)')
    ap.add_argument('--db', default=os.path.join(ROOT, 'data', 'stock_data.db'))
    args = ap.parse_args()

    params = QUICK_PARAMS if args.quick else REG_PARAMS
    print('=' * 74)
    print(f'📊 诚实盈利回测  quick={args.quick}  reaggregate={args.reaggregate}  '
          f'horizon={HORIZON}  cost={COST_A}')
    print('=' * 74)

    if args.reaggregate:
        if not os.path.exists(OOS_CACHE):
            print(f'❌ 无缓存 {OOS_CACHE}, 请先跑一次完整回测'); return
        with open(OOS_CACHE, 'rb') as f:
            oos = pickle.load(f)
        print(f'♻️  从缓存重算 ({len(oos["ret20"])} 条 OOS)')
    else:
        conn = sqlite3.connect(args.db)
        pipeline = FeaturePipeline({
            'label': '日线', 'horizon': HORIZON, 'db_table': 'kline_daily',
            'min_history': 120, 'purged_gap': PURGE_DAYS, 'north_shift_days': 1,
        })
        symbols = load_symbols(conn, args.quick)
        print(f'📈 池: {len(symbols)} 只')
        df_all, feat_names = build_pool(conn, symbols, pipeline, args.quick)
        conn.close()

        oos = run_walkforward(df_all, feat_names, params)
        if len(oos['ret20']) == 0:
            print('❌ 无 OOS 样本, 退出')
            return
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(OOS_CACHE, 'wb') as f:
            pickle.dump(oos, f)
        print(f'💾 OOS 已缓存: {OOS_CACHE}')

    print('\n🧮 聚合组合净值曲线...')
    portfolio = build_portfolio(oos)
    signals = build_signals(oos)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(PORTFOLIO_JSON, 'w') as f:
        json.dump(portfolio, f, ensure_ascii=False, default=str)
    with open(SIGNALS_JSON, 'w') as f:
        json.dump(signals, f, ensure_ascii=False, default=str)

    h = portfolio['headline']
    print('\n' + '=' * 74)
    print('📋 组合回测结果 (OOS, 扣成本, 不重叠20日 rebalance)')
    print('=' * 74)
    sn = portfolio['single_name']
    print(f"  横截面 top-K 总收益   {h['top_k_total']:+.1%}  年化 {h['top_k_annual']:+.1%}"
          f"  Sharpe {h['top_k_sharpe']:.2f}  最大回撤 {h['top_k_maxdd']:.1%}")
    print(f"  全市场等权(基准)总收益 {h['universe_total']:+.1%}   → 超额 {h['excess_total']:+.1%}"
          f"  (每期超额 {h['top_k_excess_per_period']:+.2%})  ⚠️绝对数含幸存者偏差")
    print(f"  long-short 总收益     {h['long_short_total']:+.1%}  Sharpe {h['long_short_sharpe']:.2f} (最可信)")
    print(f"  [诚实对照] 单只择时逐笔 {sn['timing_net_per_trade']:+.2%} vs 买入持有 "
          f"{sn['buyhold_net_per_trade']:+.2%} → 差 {sn['timing_edge_per_trade']:+.2%} (≈打平, 无择时edge)")
    print(f"\n💾 已保存:\n  {PORTFOLIO_JSON} ({os.path.getsize(PORTFOLIO_JSON)/1024:.0f} KB)")
    print(f"  {SIGNALS_JSON} ({os.path.getsize(SIGNALS_JSON)/1024:.0f} KB, {len(signals)} 只)")


if __name__ == '__main__':
    main()
