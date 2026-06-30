#!/usr/bin/env python3
"""
特征剪枝测试 — 用不同数量的 Top-K 特征训练, 对比 IC/ICIR

读取 feature_importance.csv, 依次用 K=50/100/150/200/全部 子集训练模型,
输出各子集的验证集 IC, 找到最优特征数。

用法:
  python strategy/test_feature_pruning.py           # 完整测试
  python strategy/test_feature_pruning.py --quick    # 快速模式
"""

import os, sys, sqlite3, time, argparse, csv
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'python'))

DB_PATH = os.path.join(ROOT, 'data', 'stock_data.db')
MODEL_DIR = os.path.join(ROOT, '..', 'models', 'lgb_hs300_enhanced')
IMP_CSV = os.path.join(MODEL_DIR, 'feature_importance.csv')

from strategy.train_enhanced import (
    load_kline_data, calculate_features, prepare_dataset,
    train_models, ensemble_weight,
)


def load_feature_ranking():
    if not os.path.exists(IMP_CSV):
        print(f"❌ 未找到 {IMP_CSV}, 请先运行 train_enhanced.py 生成特征重要性")
        sys.exit(1)
    features = []
    with open(IMP_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            features.append(row['feature'])
    return features


def run_pruning_test(k_values=None, quick=False, horizon=3, max_stocks=0):
    if k_values is None:
        k_values = [50, 100, 150, 200]

    ranked_features = load_feature_ranking()
    total_features = len(ranked_features)
    k_values = [k for k in k_values if k < total_features]
    k_values.append(total_features)
    print(f"  共 {total_features} 个特征, 测试 K={k_values}")

    print("\n加载数据...")
    t0 = time.time()
    conn = sqlite3.connect(DB_PATH)
    df = load_kline_data(conn, max_stocks)
    results = calculate_features(df, conn)
    conn.close()

    X_all, y_all, regimes_all, dates_all, all_features = prepare_dataset(
        results, horizon, db_path=DB_PATH)
    print(f"  数据: {len(X_all)} 样本, {len(all_features)} 特征 ({time.time()-t0:.0f}s)")

    # 时间切分
    train_cutoff = np.percentile(dates_all.astype('datetime64[D]').astype(int), 80)
    train_cutoff = np.datetime64(int(train_cutoff), 'D')
    purge_start = train_cutoff - np.timedelta64(horizon + 2, 'D')
    train_mask = dates_all < purge_start
    val_mask = dates_all >= train_cutoff

    X_train_full, X_val_full = X_all[train_mask], X_all[val_mask]
    y_train, y_val = y_all[train_mask], y_all[val_mask]
    d_val = dates_all[val_mask]

    # 特征名→索引映射
    feat_to_idx = {f: i for i, f in enumerate(all_features)}

    results_table = []

    for k in k_values:
        print(f"\n{'='*60}")
        top_k_names = ranked_features[:k]
        label = f"Top-{k}" if k < total_features else f"全部({total_features})"
        print(f"  测试: {label} 特征")

        # 选列
        col_idx = [feat_to_idx[f] for f in top_k_names if f in feat_to_idx]
        if len(col_idx) < k:
            print(f"  ⚠️ 只找到 {len(col_idx)}/{k} 个特征在数据中")
        X_train_k = X_train_full[:, col_idx]
        X_val_k = X_val_full[:, col_idx]
        feat_k = [all_features[i] for i in col_idx]

        models, cv_scores = train_models(X_train_k, y_train, X_val_k, y_val, feat_k, quick)
        weights = ensemble_weight(models, cv_scores)

        # ensemble 预测
        preds = np.zeros(len(y_val))
        for name, model in models.items():
            preds += model.predict(X_val_k) * weights.get(name, 0)

        # 逐日 IC
        val_dates = pd.to_datetime(d_val)
        daily_ics = []
        for d in sorted(set(val_dates.strftime('%Y-%m-%d'))):
            d_mask = val_dates.strftime('%Y-%m-%d') == d
            if d_mask.sum() < 10:
                continue
            ic = spearmanr(preds[d_mask], y_val[d_mask])[0]
            if not np.isnan(ic):
                daily_ics.append(ic)

        ic_mean = np.mean(daily_ics) if daily_ics else 0
        ic_std = np.std(daily_ics) if daily_ics else 1
        icir = ic_mean / ic_std if ic_std > 0 else 0

        # Top/Bottom 组收益
        sort_idx = np.argsort(preds)
        top_n = max(5, len(preds) // 20)
        top_ret = y_val[sort_idx[-top_n:]].mean()
        bot_ret = y_val[sort_idx[:top_n]].mean()

        ic_pos_rate = sum(1 for x in daily_ics if x > 0) / max(len(daily_ics), 1)

        print(f"  → IC={ic_mean:+.4f} ICIR={icir:.3f} "
              f"Top={top_ret:+.2%} Bot={bot_ret:+.2%} 多空={top_ret-bot_ret:+.2%} "
              f"IC正向={ic_pos_rate:.0%}")

        results_table.append({
            'k': k,
            'label': label,
            'ic_mean': ic_mean,
            'icir': icir,
            'top_ret': top_ret,
            'bot_ret': bot_ret,
            'long_short': top_ret - bot_ret,
            'ic_pos_rate': ic_pos_rate,
        })

    # 汇总
    print(f"\n\n{'='*70}")
    print(f"  📊 特征剪枝测试汇总")
    print(f"{'='*70}")
    print(f"  {'特征数':>10} {'IC':>8} {'ICIR':>8} {'多空':>8} {'IC正向':>8}")
    best = max(results_table, key=lambda x: x['icir'])
    for r in results_table:
        marker = ' ← best' if r == best else ''
        print(f"  {r['label']:>10} {r['ic_mean']:>+8.4f} {r['icir']:>8.3f} "
              f"{r['long_short']:>+8.2%} {r['ic_pos_rate']:>8.0%}{marker}")

    print(f"\n  最优特征数: {best['label']} (ICIR={best['icir']:.3f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='快速模式')
    parser.add_argument('--horizon', type=int, default=3, help='预测周期')
    parser.add_argument('--max-stocks', type=int, default=0, help='最大股票数(0=全部)')
    parser.add_argument('--k', type=int, nargs='+', default=None, help='自定义K值')
    args = parser.parse_args()
    run_pruning_test(
        k_values=args.k,
        quick=args.quick,
        horizon=args.horizon,
        max_stocks=args.max_stocks,
    )
