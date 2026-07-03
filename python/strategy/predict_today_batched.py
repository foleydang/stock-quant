#!/usr/bin/env python3
"""
分批预测 — 内存友好版，逐批加载股票→计算特征→预测→释放内存

优化策略:
  1. 只加载最近 300 个交易日 (够算 MA200)，而非全部 12000 行 → 省 40x 内存
  2. 股票按批次 (默认 20 只/批) 处理
  3. 每批: 加载K线→计算特征→只保留最后一行→释放
  4. 全部批完成后, 计算截面排名→预测→输出

用法:
  python strategy/predict_today_batched.py                  # 全量预测
  python strategy/predict_today_batched.py --batch 10       # 每批10只(更省内存)
  python strategy/predict_today_batched.py --batch 30       # 每批30只(更快)
  python strategy/predict_today_batched.py --date 2026-06-30
  python strategy/predict_today_batched.py --top 30
  python strategy/predict_today_batched.py --dry-run        # 不预测, 只看内存占用
"""

import os, sys, pickle, sqlite3, argparse, json, gc, time, tracemalloc
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'python'))

DB_PATH = os.path.join(ROOT, 'data', 'stock_data.db')
MODEL_DIR = os.path.join(ROOT, '..', 'models', 'lgb_hs300_enhanced')

# 最大历史窗口: 300 个交易日 (够算 MA200, MA250, 再加一些缓冲)
MAX_HISTORY_DAYS = 300


def get_stock_list(conn, max_stocks=0):
    """获取所有股票列表"""
    sql = """
        SELECT DISTINCT symbol FROM kline_30m
        WHERE (symbol LIKE '%.SZ' OR symbol LIKE '%.SH')
        ORDER BY symbol
    """
    if max_stocks > 0:
        sql += f" LIMIT {max_stocks}"
    df = pd.read_sql(sql, conn)
    return df['symbol'].tolist()


def get_stock_names(conn):
    """获取股票名称映射"""
    try:
        df = pd.read_sql("SELECT symbol, name FROM stock_info", conn)
        return dict(zip(df['symbol'], df['name']))
    except Exception:
        return {}


def get_latest_date(conn):
    """获取最新交易日期"""
    df = pd.read_sql(
        "SELECT MAX(substr(date,1,10)) as d FROM kline_30m", conn
    )
    return str(df['d'].iloc[0])


def load_kline_batch(conn, symbols, target_date, max_days=MAX_HISTORY_DAYS):
    """
    只加载指定股票最近 max_days 个交易日的日线数据
    内存优化: SQL 层面用日期过滤, 只返回最近 ~300 天数据, 而非全部 12000 行
    """
    # 计算 cutoff date (保守估计: max_days * 2 个自然日覆盖 max_days 个交易日)
    cutoff = pd.to_datetime(target_date) - pd.Timedelta(days=max_days * 2)
    cutoff_str = cutoff.strftime('%Y-%m-%d')

    # 用字符串拼接 IN 子句 (SQLite params 不展开 IN 列表)
    sym_list = ','.join(f"'{s}'" for s in symbols)

    # 关键优化: WHERE date >= cutoff 在 GROUP BY 之前过滤, 大幅减少扫描行数
    sql = f"""
        SELECT t.symbol, t.trade_date,
               first_bar.open as open, t.high, t.low,
               last_bar.close as close, t.volume
        FROM (
            SELECT symbol, substr(date,1,10) as trade_date,
                   MAX(high) as high, MIN(low) as low,
                   SUM(volume) as volume,
                   MIN(date) as first_date, MAX(date) as last_date
            FROM kline_30m
            WHERE symbol IN ({sym_list})
              AND date >= '{cutoff_str}'
            GROUP BY symbol, substr(date,1,10)
        ) t
        JOIN kline_30m first_bar ON first_bar.symbol = t.symbol
            AND first_bar.date = t.first_date
        JOIN kline_30m last_bar ON last_bar.symbol = t.symbol
            AND last_bar.date = t.last_date
        ORDER BY t.symbol, t.trade_date
    """

    df = pd.read_sql(sql, conn)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df[df['trade_date'] <= target_date].copy()

    # 确保每只股票不超过 max_days 行
    df = df.sort_values(['symbol', 'trade_date'])
    df = df.groupby('symbol', group_keys=False).tail(max_days)
    df = df.reset_index(drop=True)

    return df


def compute_features_batch(df, conn, pipeline):
    """
    对一批股票计算特征, 只返回每只股票最后一行特征
    注意: FeaturePipeline.compute_stock 内部已处理情绪/市场/宏观特征,
          不需要额外加载 sentiment_daily (全表 969K 行, 会 OOM)
    返回: {symbol: (last_date, feature_row)}
    """
    results = {}
    symbols = sorted(df['symbol'].unique())

    for sym in symbols:
        stock_df = df[df['symbol'] == sym].copy().sort_values('trade_date')
        if len(stock_df) < 100:
            continue

        stock_df = stock_df.reset_index(drop=True)

        fp_df = stock_df.rename(columns={'trade_date': 'date'}).copy()
        fp_df['date'] = pd.to_datetime(fp_df['date'])

        try:
            feats = pipeline.compute_stock(fp_df, sym)
            feats = feats.ffill().fillna(0)
            feats.index = stock_df.index
        except Exception:
            continue

        # 只保留最后一行 (FeaturePipeline 已包含 sent_* 特征, 无需额外添加)
        last_idx = feats.index[-1]
        last_date = stock_df['trade_date'].iloc[-1]
        last_row = feats.loc[last_idx].copy()
        results[sym] = (last_date, last_row)

        # 每处理完一只就释放
        del feats, stock_df, fp_df

    gc.collect()
    return results


def compute_cross_sectional_ranks(all_results):
    """对收集到的所有股票的最后一行特征, 计算截面排名"""
    cs_candidates = ['price_ret_1', 'price_ret_5', 'price_ret_20', 'price_vol_20',
                     'price_ma20_ratio', 'vol_ratio_20', 'price_rsi_14', 'price_adx',
                     'price_bb20_width', 'price_parkinson_vol', 'price_macd_hist',
                     'price_kdj_j', 'price_cci', 'price_atr_ratio']

    rows = []
    syms = []
    for sym, (date_obj, feat_row) in all_results.items():
        rows.append(feat_row)
        syms.append(sym)

    if len(rows) == 0:
        return

    df = pd.DataFrame(rows, index=syms)
    available_cols = [c for c in cs_candidates if c in df.columns]

    for col in available_cols:
        df[f'cs_rank_{col}'] = df[col].rank(pct=True)

    for sym in syms:
        if sym in all_results:
            _, feat_row = all_results[sym]
            for col in df.columns:
                if col.startswith('cs_rank_'):
                    feat_row[col] = df.loc[sym, col]
            all_results[sym] = (all_results[sym][0], feat_row)


def score_to_signal(score, p10, p30, p70, p90):
    """得分转买卖信号
    - 得分不是概率, 是预测未来3日的截面相对收益
    - 基于全市场分位数划分信号区间
    """
    if score >= p90:
        return '🟢 强烈买入', 'strong_buy'
    elif score >= p70:
        return '🟢 买入', 'buy'
    elif score >= p30:
        return '🟡 持有', 'hold'
    elif score >= p10:
        return '🔴 卖出', 'sell'
    else:
        return '🔴 强烈卖出', 'strong_sell'


def format_memory(bytes_val):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}TB"


def predict_batched(model_path, date_str=None, top_n=15, batch_size=20, dry_run=False):
    t_start = time.time()

    print("=" * 70)
    print("🚀 分批预测 - 内存友好版 (仅加载最近{N}天)".format(N=MAX_HISTORY_DAYS))
    print("=" * 70)

    # === 1. 加载模型 ===
    print(f"\n📦 [1/5] 加载模型...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    feature_names = model_data['feature_names']
    models = model_data.get('models', [])
    weights_raw = model_data.get('regime_weights', {})

    if 'all' in weights_raw:
        weights = weights_raw['all']
    else:
        weights = weights_raw if weights_raw else {k: 1.0 / len(models) for k in range(len(models))}

    print(f"   模型版本: {model_data.get('model_version')}")
    print(f"   训练时间: {model_data.get('trained_at', 'unknown')}")
    print(f"   特征数: {len(feature_names)} | 子模型数: {len(models)}")
    print(f"   模型大小: {os.path.getsize(model_path) / 1024 / 1024:.1f}MB")

    # === 2. 获取股票列表 ===
    print(f"\n📋 [2/5] 获取股票列表...")
    conn = sqlite3.connect(DB_PATH)
    name_map = get_stock_names(conn)

    if date_str is None:
        date_str = get_latest_date(conn)
    target_date = pd.to_datetime(date_str)
    print(f"   预测日期: {date_str}")

    stock_list = get_stock_list(conn)
    print(f"   总股票数: {len(stock_list)}")

    # === 3. 分批计算特征 ===
    print(f"\n🔧 [3/5] 分批计算特征 (每批 {batch_size} 只)...")

    from strategy.features import FeaturePipeline
    pipeline = FeaturePipeline({
        'label': '日线', 'horizon': 3, 'db_table': 'kline_daily',
        'min_history': 120, 'purged_gap': 3, 'north_shift_days': 1
    })

    all_results = {}
    batches = [stock_list[i:i + batch_size] for i in range(0, len(stock_list), batch_size)]

    for batch_idx, batch_symbols in enumerate(batches):
        batch_start = time.time()

        df = load_kline_batch(conn, batch_symbols, target_date)
        if len(df) == 0:
            print(f"   [{batch_idx + 1}/{len(batches)}] 📭 批次无数据, 跳过")
            continue

        batch_results = compute_features_batch(df, conn, pipeline)
        all_results.update(batch_results)

        batch_elapsed = time.time() - batch_start
        actual_stocks = len(df['symbol'].unique())
        print(f"   [{batch_idx + 1}/{len(batches)}] "
              f"✅ {actual_stocks}只加载 → {len(batch_results)}只有效, "
              f"耗时 {batch_elapsed:.1f}s, "
              f"累计 {len(all_results)} 只")

        if dry_run and batch_idx >= 2:
            print(f"\n   🛑 dry-run 模式, 已测试 3 批, 退出")
            break

    conn.close()

    if len(all_results) == 0:
        print("❌ 没有可用样本")
        return None

    # === 4. 计算截面排名 + 预测 ===
    print(f"\n📊 [4/5] 计算截面排名...")
    compute_cross_sectional_ranks(all_results)

    print(f"🔮 [5/5] 模型预测...")
    pred_syms = []
    pred_rows = []
    pred_dates = []

    for sym, (date_obj, feat_row) in all_results.items():
        pred_syms.append(sym)
        pred_rows.append(feat_row)
        pred_dates.append(date_obj)

    feat_df = pd.DataFrame(pred_rows).fillna(0).replace([np.inf, -np.inf], 0)

    from strategy.features import rename_features_for_model
    feat_df = rename_features_for_model(feat_df, feature_names)
    X = feat_df.values

    actual_date = max(pred_dates) if pred_dates else target_date
    print(f"   实际预测日: {actual_date.date() if hasattr(actual_date, 'date') else actual_date}")
    print(f"   样本数: {len(X)}")

    # 集成预测
    preds = np.zeros(len(X))
    for i, model in enumerate(models):
        keys = list(weights.keys())
        key = keys[i] if i < len(keys) else i
        w = weights.get(key, 0)
        if w == 0 and isinstance(key, str):
            w = list(weights.values())[i] if i < len(weights) else 0
        if w > 0:
            preds += model.predict(X) * w

    w_sum = sum(weights.values())
    if w_sum > 0 and abs(w_sum - 1.0) > 0.01:
        preds = preds / w_sum

    sort_idx = np.argsort(-preds)

    # 计算分位数，用于信号划分
    p10 = np.percentile(preds, 10)
    p30 = np.percentile(preds, 30)
    p70 = np.percentile(preds, 70)
    p90 = np.percentile(preds, 90)

    # 给每只股票打信号
    signals = []
    signal_labels = []
    for s in preds:
        label, code = score_to_signal(s, p10, p30, p70, p90)
        signals.append(code)
        signal_labels.append(label)

    # === 信号统计 ===
    strong_buy = sum(1 for s in signals if s == 'strong_buy')
    buy = sum(1 for s in signals if s == 'buy')
    hold = sum(1 for s in signals if s == 'hold')
    sell = sum(1 for s in signals if s == 'sell')
    strong_sell = sum(1 for s in signals if s == 'strong_sell')

    print(f"\n{'='*70}")
    print(f"📊 信号分布 | 全市场 {len(preds)} 只股票")
    print(f"{'='*70}")
    print(f"  🟢 强烈买入 (>p90, score>{p90:+.4f}): {strong_buy} 只")
    print(f"  🟢 买入     (p70-p90, {p70:+.4f}~{p90:+.4f}): {buy} 只")
    print(f"  🟡 持有     (p30-p70, {p30:+.4f}~{p70:+.4f}): {hold} 只")
    print(f"  🔴 卖出     (p10-p30, {p10:+.4f}~{p30:+.4f}): {sell} 只")
    print(f"  🔴 强烈卖出 (<p10, score<{p10:+.4f}): {strong_sell} 只")

    # === 输出：买入建议 ===
    print(f"\n{'='*70}")
    print(f"📈 买入建议 (强烈买入 + 买入)")
    print(f"{'='*70}")
    print(f"{'排名':>4} {'代码':>10} {'名称':>15} {'得分':>8} {'信号':>12}")
    buy_count = 0
    for rank, idx in enumerate(sort_idx, 1):
        if signals[idx] in ('strong_buy', 'buy'):
            buy_count += 1
            sym = pred_syms[idx]
            name = name_map.get(sym, '未知')
            print(f"{rank:>4} {sym:>10} {name:>15} {preds[idx]:>+8.4f} {signal_labels[idx]:>12}")
            if buy_count >= top_n:
                break

    # === 输出：卖出建议 ===
    print(f"\n{'='*70}")
    print(f"📉 卖出建议 (强烈卖出 + 卖出)")
    print(f"{'='*70}")
    print(f"{'排名':>4} {'代码':>10} {'名称':>15} {'得分':>8} {'信号':>12}")
    sell_count = 0
    for rank, idx in enumerate(sort_idx[::-1], 1):
        rev_idx = sort_idx[-rank]
        if signals[rev_idx] in ('strong_sell', 'sell'):
            sell_count += 1
            sym = pred_syms[rev_idx]
            name = name_map.get(sym, '未知')
            print(f"{rank:>4} {sym:>10} {name:>15} {preds[rev_idx]:>+8.4f} {signal_labels[rev_idx]:>12}")
            if sell_count >= top_n:
                break

    date_str_out = (actual_date.strftime('%Y%m%d')
                    if hasattr(actual_date, 'strftime')
                    else str(actual_date)[:10])
    out_path = os.path.join(MODEL_DIR, f'prediction_{date_str_out}.csv')
    result_df = pd.DataFrame({
        'rank': np.arange(1, len(preds) + 1),
        'symbol': [pred_syms[i] for i in sort_idx],
        'name': [name_map.get(pred_syms[i], '未知') for i in sort_idx],
        'score': preds[sort_idx],
        'signal': [signal_labels[i] for i in sort_idx],
        'signal_code': [signals[i] for i in sort_idx],
    })
    result_df.to_csv(out_path, index=False)
    print(f"\n💾 完整结果保存到 {out_path} (含信号列)")
    print(f"   共 {len(preds)} 只股票打分")

    elapsed = time.time() - t_start
    print(f"\n⏱️ 总耗时: {elapsed:.1f}s ({elapsed / 60:.1f}min)")

    return result_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='分批预测 - 内存友好版')
    parser.add_argument('--date', type=str, default=None, help='预测日期 YYYY-MM-DD')
    parser.add_argument('--top', type=int, default=15, help='显示 Top/Bottom 数量')
    parser.add_argument('--batch', type=int, default=180, help='每批股票数 (默认180, 3批跑完)')
    parser.add_argument('--dry-run', action='store_true', help='只测试前3批')
    parser.add_argument('--start-memory', action='store_true', help='启用内存追踪')
    args = parser.parse_args()

    model_path = os.path.join(MODEL_DIR, 'model.pkl')
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        sys.exit(1)

    if args.start_memory:
        tracemalloc.start()

    result = predict_batched(
        model_path,
        date_str=args.date,
        top_n=args.top,
        batch_size=args.batch,
        dry_run=args.dry_run,
    )

    if args.start_memory and result is not None:
        current, peak = tracemalloc.get_traced_memory()
        print(f"\n💾 内存追踪:")
        print(f"   当前: {format_memory(current)}")
        print(f"   峰值: {format_memory(peak)}")
        tracemalloc.stop()