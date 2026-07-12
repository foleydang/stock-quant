#!/usr/bin/env python3
"""
补仓顾问全票池扫描 — 定时任务用 (cron)

对全 A 股票池用 add_advisor 模型打分, 截面排名分桶, 写入磁盘缓存。
供 /api/advisor/scan 接口读取, 6 小时 TTL。

用法:
  python strategy/scan_advisor.py [--limit 300] [--board all]
"""

import sys
import os
import json
import time
import sqlite3
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config_loader import get_db_path
from strategy.features import FeaturePipeline
from strategy.add_advisor_ml import (
    load_final_model, scan_universe, PURGE_DAYS,
)

# 缓存文件路径
_SCAN_CACHE = os.path.join(ROOT, 'data', 'advisor_scan.json')

# 分位分桶阈值
_BUCKETS = [
    (0.10, 'strong_buy', '强烈买入'),
    (0.25, 'buy', '买入'),
    (0.75, 'hold', '持有'),
    (0.90, 'sell', '卖出'),
    (1.01, 'strong_sell', '强烈卖出'),
]


def _get_symbols_for_board(conn, board, limit):
    """根据板块筛选股票池"""
    if board == 'cyb':
        pattern = "symbol LIKE '300%' OR symbol LIKE '301%'"
    elif board == 'kcb':
        pattern = "symbol LIKE '688%'"
    elif board == 'sh':
        pattern = "symbol LIKE '600%' OR symbol LIKE '601%' OR symbol LIKE '603%' OR symbol LIKE '605%'"
    elif board == 'sz':
        pattern = "symbol LIKE '000%' OR symbol LIKE '001%' OR symbol LIKE '002%' OR symbol LIKE '003%'"
    else:
        pattern = "(symbol LIKE '%.SZ' OR symbol LIKE '%.SH')"

    rows = conn.execute(
        f"SELECT symbol, COUNT(*) c FROM kline_daily "
        f"WHERE {pattern} "
        f"GROUP BY symbol HAVING c>=120 ORDER BY c DESC LIMIT ?",
        (limit,)).fetchall()
    return [r[0] for r in rows]


def _bucket_by_rank(idx, n):
    q = (idx + 0.5) / max(n, 1)
    for thr, key, label in _BUCKETS:
        if q < thr:
            return key, label
    return 'strong_sell', '强烈卖出'


def _build_scan_payload(scored, data):
    items = [s for s in scored if s is not None]
    items.sort(key=lambda s: s['reg'], reverse=True)
    n = len(items)

    dist = {'strong_buy': 0, 'buy': 0, 'hold': 0, 'sell': 0, 'strong_sell': 0}
    signals = {'strong_buy': [], 'buy': [], 'hold': [], 'sell': [], 'strong_sell': []}
    pred_date = ''
    for i, s in enumerate(items):
        key, label = _bucket_by_rank(i, n)
        dist[key] += 1
        pred_date = max(pred_date, s.get('date', '') or '')
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
    return {
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


def main():
    parser = argparse.ArgumentParser(description='补仓顾问全票池扫描')
    parser.add_argument('--limit', type=int, default=300, help='最多扫描股票数 (默认 300)')
    parser.add_argument('--board', type=str, default='all',
                        choices=['all', 'sh', 'sz', 'cyb', 'kcb'],
                        help='板块 (默认 all)')
    args = parser.parse_args()

    print(f"🔍 补仓顾问扫描开始: board={args.board}, limit={args.limit}")
    t0 = time.time()

    # 加载模型
    try:
        data = load_final_model()
    except FileNotFoundError:
        print("❌ 模型文件不存在: models/add_advisor/model.pkl")
        print("   请在 Mac 跑 python strategy/add_advisor_ml.py 训练后提交")
        sys.exit(1)

    print(f"   📦 模型加载完成 (训练日期: {data.get('train_date', '?')})")

    pipeline = FeaturePipeline({
        'label': '日线', 'horizon': data['horizon'], 'db_table': 'kline_daily',
        'min_history': 120, 'purged_gap': PURGE_DAYS, 'north_shift_days': 1,
        'lstm_slim': True,
    })

    conn = sqlite3.connect(get_db_path())
    try:
        symbols = _get_symbols_for_board(conn, args.board, args.limit)
        if not symbols:
            print(f"❌ 板块 {args.board} 无符合条件的股票")
            sys.exit(1)

        print(f"   📊 符合条件股票: {len(symbols)} 只, 开始逐只打分...")

        batch_size = 50
        all_scored = []
        total_batches = (len(symbols) + batch_size - 1) // batch_size
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            scored = scan_universe(conn, pipeline, data['feat_names'],
                                   data['reg'], data['clf_s'], data['clf_tb'],
                                   symbols=batch)
            all_scored.extend(scored)
            batch_num = i // batch_size + 1
            elapsed = time.time() - t0
            print(f"   ⏳ batch {batch_num}/{total_batches} "
                  f"({len(batch)}只, 累计 {len(all_scored)} 有效, "
                  f"{elapsed:.0f}s)")
    finally:
        conn.close()

    # 构建并写入缓存
    payload = _build_scan_payload(all_scored, data)
    payload['_cacheKey'] = f'{args.board}_{args.limit}'
    payload['board'] = args.board

    os.makedirs(os.path.dirname(_SCAN_CACHE), exist_ok=True)
    with open(_SCAN_CACHE, 'w') as f:
        json.dump(payload, f, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n✅ 扫描完成: {payload['totalStocks']} 只, 耗时 {elapsed:.0f}s")
    print(f"   📁 缓存: {_SCAN_CACHE}")
    print(f"   📊 分布: {payload['distribution']}")

    # 打印 top 10
    top = payload['signals'].get('strong_buy', [])[:5] + payload['signals'].get('buy', [])[:5]
    if top:
        print("\n   🏆 Top 买入信号:")
        for s in top[:10]:
            print(f"      {s['rank']:>4}. {s['symbol']:<12} {s['name']:<8} "
                  f"score={s['score']:.4f} up={s['upProb']:.3f} tp={s['tpProb']:.3f}")


if __name__ == '__main__':
    main()