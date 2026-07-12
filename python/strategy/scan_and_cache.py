#!/usr/bin/env python3
"""
add_advisor 全量扫描 → 写缓存 JSON
由 cron 定时调用, 或 /advisor/scan?refresh=1 触发后台执行

用法: python strategy/scan_and_cache.py [--limit 50]
"""
import sys, os, gc, time, json, sqlite3, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'api'))

from config_loader import get_db_path
from strategy.features import FeaturePipeline
from strategy.add_advisor_ml import load_final_model, scan_universe, PURGE_DAYS

CACHE_FILE = os.path.join(os.path.dirname(get_db_path()), 'advisor_scan.json')

# 分桶阈值
_BUCKETS = [
    (0.10, 'strong_buy', '强烈买入'),
    (0.25, 'buy', '买入'),
    (0.75, 'hold', '持有'),
    (0.90, 'sell', '卖出'),
    (1.01, 'strong_sell', '强烈卖出'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=50, help='最多扫描股票数 (默认50)')
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"add_advisor 全量扫描 (limit={args.limit})")
    print(f"{'='*60}")

    t0 = time.time()
    print("加载模型...")
    data = load_final_model()
    print(f"  horizon={data['horizon']}, features={len(data['feat_names'])}")

    print("初始化 FeaturePipeline (lstm_slim=True)...")
    pipeline = FeaturePipeline({
        'label': '日线', 'horizon': data['horizon'], 'db_table': 'kline_daily',
        'min_history': 120, 'purged_gap': PURGE_DAYS, 'north_shift_days': 1,
        'lstm_slim': True,
    })

    conn = sqlite3.connect(get_db_path())

    def progress(i, n, done):
        elapsed = time.time() - t0
        print(f"  [{i}/{n}] {done} scored, {elapsed:.0f}s")

    print("扫描全 A 股...")
    # 取前 N 只数据量最大的股票
    rows = conn.execute(
        "SELECT symbol, COUNT(*) c FROM kline_daily "
        "WHERE symbol LIKE '%.SZ' OR symbol LIKE '%.SH' "
        "GROUP BY symbol HAVING c>=120 ORDER BY c DESC LIMIT ?",
        (args.limit,)).fetchall()
    symbols = [r[0] for r in rows]
    print(f"  候选: {len(symbols)} 只")

    scored = scan_universe(
        conn, pipeline, data['feat_names'],
        data['reg'], data['clf_s'], data['clf_tb'],
        symbols=symbols, progress=progress,
    )
    conn.close()

    elapsed = time.time() - t0
    print(f"扫描完成: {len(scored)} 只, {elapsed:.0f}s")

    # 按 reg 排序 + 分桶
    items = [s for s in scored if s is not None]
    items.sort(key=lambda s: s['reg'], reverse=True)
    n = len(items)

    dist = {'strong_buy': 0, 'buy': 0, 'hold': 0, 'sell': 0, 'strong_sell': 0}
    signals = {'strong_buy': [], 'buy': [], 'sell': [], 'strong_sell': []}
    pred_date = ''
    for i, s in enumerate(items):
        q = (i + 0.5) / max(n, 1)
        key = 'strong_sell'
        label = '强烈卖出'
        for thr, k, lb in _BUCKETS:
            if q < thr:
                key, label = k, lb
                break
        dist[key] += 1
        pred_date = max(pred_date, s.get('date', '') or '')
        if key in signals:
            signals[key].append({
                'rank': i + 1,
                'symbol': s['sym'],
                'name': s.get('name', s['sym']),
                'score': round(s['reg'], 4),
                'signal': label,
                'upProb': round(s['pup'], 3),
                'tpProb': round(s['ptp'], 3),
                'candidate': bool(s['cand']),
            })

    pred_date = (pred_date or '')[:10].replace('-', '')
    payload = {
        'status': 'success',
        'predDate': pred_date,
        'totalStocks': n,
        'distribution': dist,
        'signals': signals,
        'trainDate': data.get('train_date'),
        'cutoff': data.get('cutoff'),
        'horizon': data.get('horizon'),
        'caveat': 'edge 薄(横截面 rank-IC≈0.05), 仅 A 股; 按预测20日收益相对排名分桶, 非绝对信号',
        'generatedAt': time.strftime('%Y-%m-%d %H:%M'),
    }

    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(payload, f, ensure_ascii=False)

    top3 = [(s.get('sym', s.get('symbol', '?')), f"{s['reg']*100:+.1f}%") for s in items[:3]]
    print(f"✅ 缓存已写入: {CACHE_FILE}")
    print(f"   Top-3: {top3}")


if __name__ == '__main__':
    main()