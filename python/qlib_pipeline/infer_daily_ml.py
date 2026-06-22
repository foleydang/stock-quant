#!/usr/bin/env python3
"""
日线 ML 推理 — 加载 Mac 训练的 LightGBM 模型，输出每日信号
配合 30min 模型做双模型择时

用法:
  python qlib_pipeline/infer_daily_ml.py                    # 单次推理
  python qlib_pipeline/infer_daily_ml.py --top-k 10         # Top-10
  python qlib_pipeline/infer_daily_ml.py --with-intraday     # 含日内择时提示
"""

import os, sys, sqlite3, json, pickle, argparse, warnings
from datetime import datetime
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from config_loader import get_db_path
from qlib_pipeline.features_daily import compute_features, FEATURE_NAMES

# 复制辅助特征加载逻辑（轻量版）
from qlib_pipeline.train_daily import load_auxiliary, _add_aux_features

DB_PATH = get_db_path()
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'lgb_daily')
SIGNAL_FILE = os.path.join(PROJECT_ROOT, 'data', 'daily_ml_signals.json')


def load_model(model_dir=MODEL_DIR):
    """加载模型"""
    pkl_path = os.path.join(model_dir, 'model.pkl')
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"模型不存在: {pkl_path}\n请先在 Mac 上训练并同步模型")

    with open(pkl_path, 'rb') as f:
        pipeline = pickle.load(f)

    with open(os.path.join(model_dir, 'meta.json')) as f:
        meta = json.load(f)

    return pipeline, meta


def predict_all(conn, pipeline, aux):
    """对所有股票预测未来收益"""
    symbols = [r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM kline_daily ORDER BY symbol"
    ).fetchall()]

    results = []
    for sym in symbols:
        df = pd.read_sql(
            "SELECT date, open, high, low, close, volume FROM kline_daily "
            "WHERE symbol=? ORDER BY date", conn, params=(sym,)
        )
        if len(df) < 120:
            continue

        feats = compute_features(
            df['close'].values, df['high'].values,
            df['low'].values, df['volume'].values
        )
        if feats is None:
            continue

        # 添加辅助特征
        _add_aux_features(feats, sym, df['date'].iloc[-1], aux)

        X = pd.DataFrame([feats])
        for col in FEATURE_NAMES:
            if col not in X.columns:
                X[col] = 0.0
        X = X[FEATURE_NAMES].fillna(0).replace([np.inf, -np.inf], 0)

        pred = float(pipeline.predict(X)[0])
        if np.isnan(pred) or np.isinf(pred):
            continue

        results.append({
            'symbol': sym,
            'pred_return': pred,
            'close': float(df['close'].iloc[-1]),
        })

    df = pd.DataFrame(results)
    df['rank'] = df['pred_return'].rank(ascending=False).astype(int)
    return df.sort_values('pred_return', ascending=False)


def get_positions(conn):
    rows = conn.execute(
        "SELECT symbol, stock_name, shares, cost_price FROM positions"
    ).fetchall()
    return {r[0]: {'name': r[1], 'shares': r[2], 'cost': r[3]} for r in rows}


def get_names(conn):
    rows = conn.execute("SELECT symbol, name FROM stock_info").fetchall()
    return {r[0]: r[1] for r in rows}


def generate_signals(scores, positions, name_map, top_k=5):
    """生成交易信号"""
    signals = []
    held = set(positions.keys())
    total = len(scores)

    # 卖出: 持仓预测排名跌出前 50%
    sell_rank = max(int(total * 0.5), top_k * 3)
    for _, row in scores.iterrows():
        sym = row['symbol']
        if sym not in held:
            continue
        pos = positions[sym]
        pnl = (row['close'] - pos['cost']) / pos['cost'] if pos['cost'] > 0 else 0
        if row['rank'] > sell_rank:
            signals.append({
                'action': 'SELL', 'symbol': sym,
                'name': name_map.get(sym, sym),
                'price': round(row['close'], 2),
                'pred_return': round(row['pred_return'], 4),
                'rank': f"{row['rank']}/{total}",
                'pnl': f"{pnl:.1%}",
                'reason': 'ML排名下滑'
            })
            held.discard(sym)

    # 买入: Top-K 中不持仓的
    available = top_k - len(held)
    for _, row in scores.iterrows():
        if available <= 0:
            break
        sym = row['symbol']
        if sym in held:
            continue
        if row['rank'] > top_k * 3:  # 至少 Top-3K
            break
        signals.append({
            'action': 'BUY', 'symbol': sym,
            'name': name_map.get(sym, sym),
            'price': round(row['close'], 2),
            'pred_return': round(row['pred_return'], 4),
            'rank': f"{row['rank']}/{total}",
            'pnl': '-',
            'reason': 'ML预测最高'
        })
        available -= 1

    return signals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--model-dir', default=MODEL_DIR)
    parser.add_argument('--with-intraday', action='store_true',
                        help='附带日内择时提醒')
    args = parser.parse_args()

    pipeline, meta = load_model(args.model_dir)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # 加载辅助数据
    aux = load_auxiliary(conn)

    print(f"📡 预测中...", flush=True)
    scores = predict_all(conn, pipeline, aux)
    name_map = get_names(conn)
    positions = get_positions(conn)
    signals = generate_signals(scores, positions, name_map, args.top_k)

    today = datetime.now().strftime('%Y-%m-%d')
    print(f"\n{'='*60}")
    print(f"📊 日线 ML 信号 ({today})")
    print(f"  模型: {meta['model']} | IC={meta['IC']} | RankIC={meta['RankIC']} | "
          f"预测{meta['horizon']}天收益")
    print(f"  股票池: {len(scores)} 只 | 持仓: {len(positions)} 只 | 信号: {len(signals)} 条")

    if signals:
        print(f"\n📋 交易信号:")
        for s in signals:
            icon = '🟢 买入' if s['action'] == 'BUY' else '🔴 卖出'
            pred_pct = f"{s['pred_return']:+.2%}"
            print(f"  {icon} | {s['symbol']:12s} {s['name'][:8]:8s} | "
                  f"@{s['price']:.2f} | 预测:{pred_pct} | "
                  f"排名:{s['rank']} | {s['reason']}")

    print(f"\n🏆 ML 预测 Top-{args.top_k}:")
    for _, row in scores.head(args.top_k).iterrows():
        held = ' ★持仓' if row['symbol'] in positions else ''
        name = name_map.get(row['symbol'], '')[:8]
        print(f"  {row['rank']:3d}. {row['symbol']:12s} {name:8s} "
              f"预测:{row['pred_return']:+.2%}  价格:{row['close']:.2f}{held}")

    if args.with_intraday and signals:
        print(f"\n⏰ 日内择时提醒:")
        buy_signals = [s for s in signals if s['action'] == 'BUY']
        for s in buy_signals:
            print(f"  关注 {s['symbol']} {s['name']} — 盘中用 30min 模型择时入场")
        print(f"  运行: python qlib_pipeline/infer.py 监控这些股票")

    conn.close()

    os.makedirs(os.path.dirname(SIGNAL_FILE), exist_ok=True)
    with open(SIGNAL_FILE, 'w') as f:
        json.dump({
            'date': today, 'timestamp': datetime.now().isoformat(),
            'model': meta, 'signals': signals,
            'top5': [{'symbol': r['symbol'], 'pred_return': round(r['pred_return'], 4),
                      'close': round(r['close'], 2), 'rank': r['rank']}
                     for _, r in scores.head(5).iterrows()],
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 信号已保存: {SIGNAL_FILE}")


if __name__ == '__main__':
    main()