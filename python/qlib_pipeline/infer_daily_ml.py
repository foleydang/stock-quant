#!/usr/bin/env python3
"""
日线 ML 推理 v2 — 轻量化内存友好版
用法: python qlib_pipeline/infer_daily_ml.py [--top-k 5]
"""
import os, sys, sqlite3, json, argparse, warnings, gc
from datetime import datetime
import numpy as np, pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from config_loader import get_db_path

DB_PATH = get_db_path()
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'lgb_daily')
SIGNAL_FILE = os.path.join(PROJECT_ROOT, 'data', 'daily_ml_signals.json')


def load_model():
    """加载 LightGBM 原生格式 (内存友好)"""
    with open(os.path.join(MODEL_DIR, 'meta.json')) as f:
        meta = json.load(f)
    with open(os.path.join(MODEL_DIR, 'feature_names.json')) as f:
        fn = json.load(f)
    model = lgb.Booster(model_file=os.path.join(MODEL_DIR, 'model.txt'))
    return model, meta, fn['features']


def load_aux_light(conn):
    """轻量加载辅助数据 (只加载需要的数据)"""
    aux = {}

    # 基本面 (小)
    try:
        fund = pd.read_sql("SELECT symbol, trade_date, roe, net_profit_yoy, debt_ratio, revenue_yoy FROM fundamental_daily", conn)
        fund['trade_date'] = pd.to_datetime(fund['trade_date'])
        fund = fund.set_index(['symbol', 'trade_date']).sort_index()
        aux['fund'] = fund
    except Exception:
        aux['fund'] = None

    # 行业 (最小的)
    try:
        sector = pd.read_sql("SELECT symbol, industry FROM stock_sector", conn)
        aux['sector'] = sector.set_index('symbol')['industry'].to_dict()
    except Exception:
        aux['sector'] = {}

    # 宏观 (小)
    try:
        macro = pd.read_sql(
            "SELECT trade_date, hs300_close, shibor_1w, shibor_1m, cn_10y, "
            "cn_us_spread, usdcny, us_10y FROM macro_daily ORDER BY trade_date", conn)
        macro['trade_date'] = pd.to_datetime(macro['trade_date'])
        macro = macro.set_index('trade_date')
        aux['macro'] = macro
    except Exception:
        aux['macro'] = None

    # 北向 (小)
    try:
        north = pd.read_sql("SELECT trade_date, north_net, total_net FROM north_flow ORDER BY trade_date", conn)
        north['trade_date'] = pd.to_datetime(north['trade_date'])
        north = north.set_index('trade_date')
        aux['north'] = north
    except Exception:
        aux['north'] = None

    # 情绪 — 只加载最近 30 天 (避免 OOM)
    try:
        sent = pd.read_sql(
            "SELECT symbol, trade_date, is_limit_up, is_limit_down, vol_ratio_20, "
            "lhb_net_buy, margin_balance_chg, abnormal_ret FROM sentiment_daily "
            "WHERE trade_date >= date('now', '-30 days')", conn)
        sent['trade_date'] = pd.to_datetime(sent['trade_date'])
        sent = sent.set_index(['symbol', 'trade_date']).sort_index()
        aux['sent'] = sent
    except Exception:
        aux['sent'] = None

    return aux


def add_aux_features(row, sym, date, aux):
    """添加辅助特征到 row dict"""
    ds = pd.Timestamp(str(date)[:10])

    # 基本面
    fund = aux.get('fund')
    if fund is not None and sym in fund.index:
        try:
            fs = fund.loc[sym]
            fb = fs[fs.index <= ds]
            if len(fb) > 0:
                latest = fb.iloc[-1]
                row['fund_roe'] = float(latest.get('roe', 0) or 0)
                row['fund_np_yoy'] = float(latest.get('net_profit_yoy', 0) or 0)
                row['fund_debt'] = float(latest.get('debt_ratio', 0) or 0)
                row['fund_rev_yoy'] = float(latest.get('revenue_yoy', 0) or 0)
            else:
                row['fund_roe'] = row['fund_np_yoy'] = row['fund_debt'] = row['fund_rev_yoy'] = 0
        except Exception:
            row['fund_roe'] = row['fund_np_yoy'] = row['fund_debt'] = row['fund_rev_yoy'] = 0
    else:
        row['fund_roe'] = row['fund_np_yoy'] = row['fund_debt'] = row['fund_rev_yoy'] = 0

    industry = aux.get('sector', {}).get(sym, '未知')
    row['sector_code'] = float(hash(industry) % 100) / 100

    macro = aux.get('macro')
    if macro is not None and ds in macro.index:
        m = macro.loc[ds]
        if macro.index.get_loc(ds) > 0:
            prev = macro.iloc[macro.index.get_loc(ds) - 1]['hs300_close']
            row['macro_hs300_chg'] = float((m['hs300_close'] - prev) / prev) if prev > 0 else 0
        else:
            row['macro_hs300_chg'] = 0
        row['macro_shibor_1w'] = float(m.get('shibor_1w', 0) or 0)
        row['macro_shibor_1m'] = float(m.get('shibor_1m', 0) or 0)
        row['macro_cn_10y'] = float(m.get('cn_10y', 0) or 0)
        row['macro_us_10y'] = float(m.get('us_10y', 0) or 0)
        row['macro_cn_us_spread'] = float(m.get('cn_us_spread', 0) or 0)
        row['macro_usdcny'] = float(m.get('usdcny', 0) or 0)
    else:
        row['macro_hs300_chg'] = row['macro_shibor_1w'] = row['macro_shibor_1m'] = 0
        row['macro_cn_10y'] = row['macro_us_10y'] = row['macro_cn_us_spread'] = 0
        row['macro_usdcny'] = 0

    north = aux.get('north')
    if north is not None and ds in north.index:
        row['north_net'] = float(north.loc[ds, 'north_net'] or 0)
        row['north_total_net'] = float(north.loc[ds, 'total_net'] or 0)
    else:
        row['north_net'] = row['north_total_net'] = 0

    sent = aux.get('sent')
    if sent is not None and sym in sent.index:
        try:
            ss = sent.loc[sym]
            if ds in ss.index:
                s = ss.loc[ds]
                row['sent_limit_up'] = float(s.get('is_limit_up', 0) or 0)
                row['sent_limit_down'] = float(s.get('is_limit_down', 0) or 0)
                row['sent_vol_ratio'] = float(s.get('vol_ratio_20', 0) or 0)
                row['sent_lhb_net'] = float(s.get('lhb_net_buy', 0) or 0)
                row['sent_margin_chg'] = float(s.get('margin_balance_chg', 0) or 0)
                row['sent_abnormal_ret'] = float(s.get('abnormal_ret', 0) or 0)
                return
        except Exception:
            pass
    row['sent_limit_up'] = row['sent_limit_down'] = row['sent_vol_ratio'] = 0
    row['sent_lhb_net'] = row['sent_margin_chg'] = row['sent_abnormal_ret'] = 0


def predict_all(conn, model, feature_names, aux):
    """批量预测，内存友好"""
    from qlib_pipeline.features_daily import compute_features

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
        if feats is None or len(feats) < 20:
            continue

        add_aux_features(feats, sym, df['date'].iloc[-1], aux)

        X = pd.DataFrame([feats])
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[feature_names].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)

        pred = float(model.predict(X)[0])
        if np.isnan(pred) or np.isinf(pred):
            continue

        results.append({
            'symbol': sym,
            'score': pred,  # cs_rank [0,1], 越高越好
            'close': float(df['close'].iloc[-1]),
        })
        gc.collect()

    df = pd.DataFrame(results)
    if len(df) == 0:
        return df
    df['rank'] = df['score'].rank(ascending=False).astype(int)
    return df.sort_values('score', ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()

    print("📡 加载模型...", flush=True)
    model, meta, feature_names = load_model()
    print(f"   {meta['model']} | RankIC={meta['RankIC']} | {meta['features']}维")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print("📡 加载辅助数据...", flush=True)
    aux = load_aux_light(conn)

    print("📡 预测中...", flush=True)
    scores = predict_all(conn, model, feature_names, aux)

    if len(scores) == 0:
        print("⚠️ 无预测结果")
        conn.close()
        return

    # 持仓信息
    positions = {}
    try:
        rows = conn.execute("SELECT symbol, stock_name, shares, cost_price FROM positions").fetchall()
        positions = {r[0]: {'name': r[1], 'shares': r[2], 'cost': r[3]} for r in rows}
    except Exception:
        pass

    name_map = {}
    try:
        rows = conn.execute("SELECT symbol, name FROM stock_info").fetchall()
        name_map = {r[0]: r[1] for r in rows}
    except Exception:
        pass

    today = datetime.now().strftime('%Y-%m-%d')
    print(f"\n{'='*60}")
    print(f"📊 日线 ML 信号 ({today})")
    print(f"  模型: {meta['model']} | RankIC={meta['RankIC']} | 预测{meta['horizon']}天收益")
    print(f"  股票池: {len(scores)} 只 | 持仓: {len(positions)} 只")

    held = set(positions.keys())
    total = len(scores)

    # 信号
    signals = []
    sell_rank = max(int(total * 0.5), args.top_k * 3)
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
                'score': round(row["score"], 4),
                'rank': f"{row['rank']}/{total}",
                'pnl': f"{pnl:.1%}",
                'reason': 'ML排名下滑'
            })
            held.discard(sym)

    available = args.top_k - len(held)
    for _, row in scores.iterrows():
        if available <= 0:
            break
        sym = row['symbol']
        if sym in held or row['rank'] > args.top_k * 3:
            continue
        signals.append({
            'action': 'BUY', 'symbol': sym,
            'name': name_map.get(sym, sym),
            'price': round(row['close'], 2),
            'score': round(row["score"], 4),
            'rank': f"{row['rank']}/{total}",
            'pnl': '-',
            'reason': 'ML预测最高'
        })
        available -= 1

    if signals:
        print(f"\n📋 交易信号 ({len(signals)}):")
        for s in signals:
            icon = '🟢' if s['action'] == 'BUY' else '🔴'
            print(f"  {icon} {s['action']:4s} | {s['symbol']:12s} {s['name'][:8]:8s} | "
                  f"@{s['price']:.2f} | 预测:{s["score"]:.4f} | "
                  f"排名:{s['rank']} | {s['reason']}")

    print(f"\n🏆 ML 预测 Top-{args.top_k}:")
    for _, row in scores.head(args.top_k).iterrows():
        held_mark = ' ★' if row['symbol'] in positions else ''
        name = name_map.get(row['symbol'], '')[:8]
        print(f"  {row['rank']:3d}. {row['symbol']:12s} {name:8s} "
              f"预测:{row["score"]:.4f}  价格:{row['close']:.2f}{held_mark}")

    conn.close()

    # 保存
    os.makedirs(os.path.dirname(SIGNAL_FILE), exist_ok=True)
    with open(SIGNAL_FILE, 'w') as f:
        json.dump({
            'date': today, 'timestamp': datetime.now().isoformat(),
            'model': meta, 'signals': signals,
            'top5': [{'symbol': r['symbol'], 'score': round(r['score'], 4),
                      'close': round(r['close'], 2), 'rank': int(r['rank'])}
                     for _, r in scores.head(5).iterrows()],
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 信号已保存: {SIGNAL_FILE}")


if __name__ == '__main__':
    main()