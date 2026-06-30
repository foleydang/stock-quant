#!/usr/bin/env python3
"""
Walk-forward 滚动训练 — 12个月训练 → purge 5天 → 1个月测试 → 滚动

用法:
  python strategy/train_walkforward.py                  # 完整滚动
  python strategy/train_walkforward.py --quick           # 快速模式 (少模型少树)
  python strategy/train_walkforward.py --train-months 6  # 缩短训练窗口
"""

import os, sys, pickle, sqlite3, time, argparse, json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'python'))

DB_PATH = os.path.join(ROOT, 'data', 'stock_data.db')
OUT_DIR = os.path.join(ROOT, '..', 'models', 'lgb_hs300_walkforward')
os.makedirs(OUT_DIR, exist_ok=True)

from strategy.train_enhanced import (
    load_kline_data, calculate_features, prepare_dataset,
    train_models, ensemble_weight, detect_market_regime,
)


def run_walkforward(train_months=12, test_months=1, purge_days=5,
                    quick=False, no_regime=False, horizon=3, max_stocks=0,
                    drop_macro=False):
    print("=" * 70)
    print(f"  Walk-Forward 滚动训练")
    print(f"  训练窗口: {train_months}个月 | 测试: {test_months}个月 | purge: {purge_days}天")
    print(f"  horizon={horizon} | quick={quick} | regime={'否' if no_regime else '是'}")
    print("=" * 70)

    t0 = time.time()
    conn = sqlite3.connect(DB_PATH)
    df = load_kline_data(conn, max_stocks)
    n_stocks = len(df['symbol'].unique())
    print(f"  加载 {n_stocks} 只股票, {len(df)} 条日线 ({time.time()-t0:.0f}s)")

    t0 = time.time()
    results = calculate_features(df, conn)
    conn.close()
    print(f"  特征计算完成: {len(results)} 只股票 ({time.time()-t0:.0f}s)")

    t0 = time.time()
    X_all, y_all, regimes_all, dates_all, all_features = prepare_dataset(
        results, horizon, db_path=DB_PATH, drop_macro=drop_macro)
    print(f"  数据准备: {len(X_all)} 样本, {len(all_features)} 特征 ({time.time()-t0:.0f}s)")

    dates_dt = pd.to_datetime(dates_all)
    min_date = dates_dt.min()
    max_date = dates_dt.max()
    print(f"  数据范围: {min_date.date()} → {max_date.date()}")

    # 生成滚动窗口
    train_offset = pd.DateOffset(months=train_months)
    test_offset = pd.DateOffset(months=test_months)
    purge_offset = pd.Timedelta(days=purge_days)

    windows = []
    # 第一个窗口: 从数据起点 + train_months 开始测试
    test_start = min_date + train_offset
    while test_start < max_date:
        test_end = min(test_start + test_offset, max_date)
        train_start = test_start - train_offset
        windows.append((train_start, test_start - purge_offset, test_start, test_end))
        test_start = test_end
    print(f"  共 {len(windows)} 个滚动窗口\n")

    # 运行每个窗口
    window_results = []
    all_test_preds = []
    all_test_actuals = []
    all_test_dates = []

    for wi, (tr_start, tr_end, te_start, te_end) in enumerate(windows):
        print(f"\n{'='*60}")
        print(f"  窗口 {wi+1}/{len(windows)}: "
              f"train [{tr_start.date()} → {tr_end.date()}] "
              f"test [{te_start.date()} → {te_end.date()}]")

        train_mask = (dates_dt >= tr_start) & (dates_dt <= tr_end)
        test_mask = (dates_dt >= te_start) & (dates_dt <= te_end)

        n_train = train_mask.sum()
        n_test = test_mask.sum()
        if n_train < 1000 or n_test < 50:
            print(f"  ⚠️ 样本不足 (train={n_train}, test={n_test}), 跳过")
            continue

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        r_train = regimes_all[train_mask]
        X_test = X_all[test_mask]
        y_test = y_all[test_mask]
        r_test = regimes_all[test_mask]
        d_test = dates_all[test_mask]

        print(f"  samples: train={n_train} test={n_test}")

        # 按 regime 训练
        w_models = {}
        w_weights = {}
        w_scores = {}

        if no_regime:
            # 分20%做验证
            n_val = max(200, int(n_train * 0.2))
            models, cv_scores = train_models(
                X_train[:-n_val], y_train[:-n_val],
                X_train[-n_val:], y_train[-n_val:],
                all_features, quick)
            w_models['all'] = models
            w_scores['all'] = cv_scores
            w_weights['all'] = ensemble_weight(models, cv_scores)
        else:
            for regime in ['bull', 'bear', 'sideways']:
                mask_tr = r_train == regime
                if mask_tr.sum() < 500:
                    print(f"    {regime}: 样本不足({mask_tr.sum()}), 用全部数据")
                    n_val = max(200, int(n_train * 0.2))
                    models, cv_scores = train_models(
                        X_train[:-n_val], y_train[:-n_val],
                        X_train[-n_val:], y_train[-n_val:],
                        all_features, quick)
                else:
                    Xr = X_train[mask_tr]
                    yr = y_train[mask_tr]
                    n_val = max(100, int(len(Xr) * 0.2))
                    print(f"    {regime}: train={len(Xr)-n_val}, val={n_val}")
                    models, cv_scores = train_models(
                        Xr[:-n_val], yr[:-n_val],
                        Xr[-n_val:], yr[-n_val:],
                        all_features, quick)
                w_models[regime] = models
                w_scores[regime] = cv_scores
                w_weights[regime] = ensemble_weight(models, cv_scores)

        # 预测
        preds = np.zeros(n_test)
        for i in range(n_test):
            regime = r_test[i]
            if no_regime:
                regime = 'all'
            if regime not in w_models:
                regime = 'sideways' if not no_regime else 'all'
            X_i = X_test[i:i+1]
            for name, model in w_models[regime].items():
                preds[i] += model.predict(X_i)[0] * w_weights[regime].get(name, 0)

        # 逐日 IC
        d_test_ts = pd.to_datetime(d_test)
        test_dates_unique = sorted(d_test_ts.unique())
        daily_ics = []
        for date_val in test_dates_unique:
            d_mask = np.array(d_test_ts == date_val)
            if d_mask.sum() < 10:
                continue
            ic = spearmanr(preds[d_mask], y_test[d_mask])[0]
            if not np.isnan(ic):
                daily_ics.append(ic)

        ic_mean = np.mean(daily_ics) if daily_ics else 0
        ic_std = np.std(daily_ics) if daily_ics else 1
        icir = ic_mean / ic_std if ic_std > 0 else 0

        # Top/Bottom 组
        sort_idx = np.argsort(preds)
        top_n = max(5, len(preds) // 20)
        top_ret = y_test[sort_idx[-top_n:]].mean()
        bot_ret = y_test[sort_idx[:top_n]].mean()

        print(f"  → IC={ic_mean:+.4f} ICIR={icir:.3f} "
              f"Top{top_n}={top_ret:+.2%} Bot{top_n}={bot_ret:+.2%} "
              f"多空={top_ret-bot_ret:+.2%} (日IC正向={sum(1 for x in daily_ics if x>0)}/{len(daily_ics)})")

        window_results.append({
            'window': wi + 1,
            'train_start': str(tr_start.date()),
            'train_end': str(tr_end.date()),
            'test_start': str(te_start.date()),
            'test_end': str(te_end.date()),
            'n_train': int(n_train),
            'n_test': int(n_test),
            'ic_mean': float(ic_mean),
            'icir': float(icir),
            'top_ret': float(top_ret),
            'bot_ret': float(bot_ret),
            'long_short': float(top_ret - bot_ret),
            'n_daily_ics': len(daily_ics),
            'ic_positive_rate': sum(1 for x in daily_ics if x > 0) / max(len(daily_ics), 1),
        })

        all_test_preds.extend(preds)
        all_test_actuals.extend(y_test)
        all_test_dates.extend(d_test)

        # 保存窗口模型
        window_model = {
            'model_version': 'v9-walkforward',
            'window': wi + 1,
            'horizon': horizon,
            'feature_names': all_features,
            'regime_models': w_models,
            'regime_weights': w_weights,
            'cv_scores': w_scores,
            'test_start': str(te_start.date()),
            'test_end': str(te_end.date()),
        }
        wpath = os.path.join(OUT_DIR, f'model_w{wi+1:02d}.pkl')
        with open(wpath, 'wb') as f:
            pickle.dump(window_model, f)

    # === 汇总 ===
    print(f"\n\n{'='*70}")
    print(f"  📊 Walk-Forward 汇总 ({len(window_results)} 个窗口)")
    print(f"{'='*70}")

    if not window_results:
        print("  ❌ 没有完成任何窗口")
        return

    ics = [w['ic_mean'] for w in window_results]
    icirs = [w['icir'] for w in window_results]
    ls = [w['long_short'] for w in window_results]

    print(f"  平均 IC:    {np.mean(ics):+.4f} (std={np.std(ics):.4f})")
    print(f"  聚合 ICIR:  {np.mean(ics)/np.std(ics):.3f}" if np.std(ics) > 0 else "  聚合 ICIR:  N/A")
    print(f"  窗口级 ICIR: {np.mean(icirs):.3f}")
    print(f"  平均多空差:  {np.mean(ls):+.2%}")
    print(f"  多空正向率:  {sum(1 for x in ls if x > 0)}/{len(ls)}")

    # 全样本外 IC
    all_p = np.array(all_test_preds)
    all_a = np.array(all_test_actuals)
    oos_ic = spearmanr(all_p, all_a)[0]
    oos_pearson = np.corrcoef(all_p, all_a)[0, 1]
    print(f"\n  全 OOS Spearman IC: {oos_ic:+.4f}")
    print(f"  全 OOS Pearson IC:  {oos_pearson:+.4f}")

    # 逐窗口详情
    print(f"\n  {'窗口':>4} {'测试区间':>24} {'IC':>8} {'ICIR':>8} {'多空':>8} {'IC正向':>8}")
    for w in window_results:
        print(f"  {w['window']:>4} {w['test_start']}→{w['test_end']} "
              f"{w['ic_mean']:>+8.4f} {w['icir']:>8.3f} {w['long_short']:>+8.2%} "
              f"{w['ic_positive_rate']:>8.0%}")

    # 保存最新窗口为 model.pkl (供 backtest 使用)
    latest_pkl = os.path.join(OUT_DIR, f'model_w{len(window_results):02d}.pkl')
    if os.path.exists(latest_pkl):
        import shutil
        shutil.copy(latest_pkl, os.path.join(OUT_DIR, 'model.pkl'))
        print(f"\n  最新窗口模型已复制为 {OUT_DIR}/model.pkl")

    # 保存汇总
    summary = {
        'train_months': train_months,
        'test_months': test_months,
        'horizon': horizon,
        'n_windows': len(window_results),
        'avg_ic': float(np.mean(ics)),
        'agg_icir': float(np.mean(ics) / np.std(ics)) if np.std(ics) > 0 else 0,
        'avg_long_short': float(np.mean(ls)),
        'oos_spearman_ic': float(oos_ic),
        'windows': window_results,
    }
    with open(os.path.join(OUT_DIR, 'walkforward_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  汇总保存到 {OUT_DIR}/walkforward_summary.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-months', type=int, default=12, help='训练窗口月数')
    parser.add_argument('--test-months', type=int, default=1, help='测试窗口月数')
    parser.add_argument('--purge-days', type=int, default=5, help='Purge gap 天数')
    parser.add_argument('--quick', action='store_true', help='快速模式')
    parser.add_argument('--no-regime', action='store_true', help='不分区市场状态')
    parser.add_argument('--horizon', type=int, default=3, help='预测周期')
    parser.add_argument('--max-stocks', type=int, default=0, help='最大股票数(0=全部)')
    parser.add_argument('--drop-macro', action='store_true', help='去除宏观/市场共性特征')
    args = parser.parse_args()
    run_walkforward(
        train_months=args.train_months,
        test_months=args.test_months,
        purge_days=args.purge_days,
        quick=args.quick,
        no_regime=args.no_regime,
        horizon=args.horizon,
        max_stocks=args.max_stocks,
        drop_macro=args.drop_macro,
    )
