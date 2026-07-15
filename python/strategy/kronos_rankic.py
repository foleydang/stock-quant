#!/usr/bin/env python3
"""Kronos 零样本横截面 rank-IC 诚实验证。

方法 (与 add_advisor 同款诚实标尺):
- 全 A 股票池 (stock_info 的 SH/SZ), as-of 每个评估日, 只用 date<=eval 的 K 线做输入。
- Kronos-small 零样本 batch 预测未来 20 交易日, 取预测 20 日收益 pred_ret。
- 真实 20 交易日收益 realized_ret = close[eval+20td]/close[eval]-1 (取自 kline_daily)。
- 每个评估日算 Spearman(pred_ret, realized_ret) = 当日 rank-IC; 汇总 mean/std/t。

诚实边界:
- Kronos 2025-08 发布, 训练截止 <= 2025 年中, 评估日取 2026 年 → 对预训练权重为样本外。
- 未复权价; 单票绝对收益零样本不可信, 但横截面"排名"才是 edge 所在 —— 本脚本只信排名。
- rank-IC ~ 0 → 无 edge, 别浪费时间微调; 稳定 >0.03 → 有弱 edge, 值得下一步。
"""
import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
KRONOS_DIR = os.path.join(REPO, "third_party", "Kronos")
DB_PATH = os.path.join(ROOT, "data", "stock_data.db")
sys.path.insert(0, KRONOS_DIR)

TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
MODEL = "NeoQuasar/Kronos-small"


def trading_calendar(conn):
    # 只用 A 股日历: kline_daily 混入了 .HK(交易日不同), 会污染评估日选取
    rows = conn.execute(
        "SELECT DISTINCT date FROM kline_daily "
        "WHERE symbol LIKE '%.SZ' OR symbol LIKE '%.SH' ORDER BY date"
    ).fetchall()
    return [r[0] for r in rows]


def universe(conn, min_bars):
    rows = conn.execute(
        "SELECT s.symbol FROM stock_info s WHERE s.market IN ('SH','SZ')"
    ).fetchall()
    syms = [r[0] for r in rows]
    out = []
    for s in syms:
        n = conn.execute("SELECT COUNT(*) FROM kline_daily WHERE symbol=?", (s,)).fetchone()[0]
        if n >= min_bars:
            out.append(s)
    return out


def build_date_panel(conn, syms, eval_date, lookback, horizon):
    """返回 (df_list, xts_list, yts_list, meta) —— meta 含 realized_ret; 只保留数据齐全的票。"""
    df_list, xts_list, yts_list, meta = [], [], [], []
    for s in syms:
        hist = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM kline_daily "
            "WHERE symbol=? AND date<=? ORDER BY date DESC LIMIT ?",
            conn, params=(s, eval_date, lookback),
        )
        if len(hist) < lookback:
            continue
        hist = hist.iloc[::-1].reset_index(drop=True)
        if hist["date"].iloc[-1] != eval_date:      # eval 当日必须有 bar (停牌则跳过)
            continue
        fut = conn.execute(
            "SELECT close FROM kline_daily WHERE symbol=? AND date>? ORDER BY date LIMIT ?",
            (s, eval_date, horizon),
        ).fetchall()
        if len(fut) < horizon:                        # 真实 20 日收益不足则跳过
            continue
        eval_close = float(hist["close"].iloc[-1])
        realized = float(fut[-1][0]) / eval_close - 1.0
        for c in ["open", "high", "low", "close", "volume"]:
            hist[c] = pd.to_numeric(hist[c], errors="coerce")
        if hist[["open", "high", "low", "close"]].isnull().values.any():
            continue
        hist["date"] = pd.to_datetime(hist["date"])
        df_list.append(hist[["open", "high", "low", "close", "volume"]])
        xts_list.append(hist["date"])
        yts_list.append(pd.Series(pd.bdate_range(
            start=hist["date"].iloc[-1] + pd.Timedelta(days=1), periods=horizon)))
        meta.append({"symbol": s, "eval_close": eval_close, "realized_ret": realized})
    return df_list, xts_list, yts_list, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--n-dates", type=int, default=8, help="评估日个数, 均匀取自可用区间")
    ap.add_argument("--gap", type=int, default=5, help="相邻评估日间隔(交易日)")
    ap.add_argument("--sample-count", type=int, default=1)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-symbols", type=int, default=0, help=">0 时截断票池(冒烟测试用)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    cal = trading_calendar(conn)
    syms = universe(conn, min_bars=args.lookback + args.horizon + 5)
    if args.max_symbols:
        syms = syms[:args.max_symbols]
    print(f"票池 {len(syms)} 只 (A股, >= {args.lookback}+{args.horizon} bars)")

    # 评估日锚点: 用票池真实数据新鲜度(多数票的最新 bar), 而非全历史末尾
    # (票池多数只更新到 T-N, 只有少数持仓票最新; 用 10 分位 max-date 作common last)
    per_last = sorted(
        conn.execute("SELECT MAX(date) FROM kline_daily WHERE symbol=?", (s,)).fetchone()[0]
        for s in syms)
    common_last = per_last[int(len(per_last) * 0.10)]
    anchor_idx = max(i for i, d in enumerate(cal) if d <= common_last) - args.horizon
    print(f"票池 common-last={common_last}, 评估锚点={cal[anchor_idx]}")
    eval_idxs = [anchor_idx - k * args.gap for k in range(args.n_dates)]
    eval_idxs = [i for i in eval_idxs if i - args.lookback >= 0]
    eval_dates = [cal[i] for i in sorted(eval_idxs)]
    print(f"评估日: {eval_dates}\n")

    from model import Kronos, KronosTokenizer, KronosPredictor
    print(f"加载 {MODEL}...")
    tok = KronosTokenizer.from_pretrained(TOKENIZER)
    mdl = Kronos.from_pretrained(MODEL)
    predictor = KronosPredictor(mdl, tok, device=args.device, max_context=512)
    print(f"设备: {predictor.device}\n")

    ics = []
    print(f"{'评估日':<12}{'样本数':>6}{'rank-IC':>10}{'IC(pearson)':>12}")
    print("-" * 42)
    for ed in eval_dates:
        df_list, xts, yts, meta = build_date_panel(conn, syms, ed, args.lookback, args.horizon)
        if len(meta) < 20:
            print(f"{ed:<12}{len(meta):>6}   样本不足, 跳过")
            continue
        preds = predictor.predict_batch(
            df_list=df_list, x_timestamp_list=xts, y_timestamp_list=yts,
            pred_len=args.horizon, T=args.T, top_p=args.top_p,
            sample_count=args.sample_count, verbose=False,
        )
        pred_ret = np.array([float(p["close"].to_numpy()[-1]) / m["eval_close"] - 1.0
                             for p, m in zip(preds, meta)])
        real_ret = np.array([m["realized_ret"] for m in meta])
        ric = spearmanr(pred_ret, real_ret).correlation
        pic = np.corrcoef(pred_ret, real_ret)[0, 1]
        ics.append(ric)
        print(f"{ed:<12}{len(meta):>6}{ric:>10.4f}{pic:>12.4f}")

    conn.close()
    if ics:
        ics = np.array(ics)
        t = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))) if len(ics) > 1 else float('nan')
        print("-" * 42)
        print(f"\nrank-IC: mean={ics.mean():.4f}  std={ics.std(ddof=1):.4f}  "
              f"n={len(ics)}  t={t:.2f}")
        print(f"对照: add_advisor 诚实 edge rank-IC≈0.053; 泄漏模型真实 edge≈0")
        print("⚠️ 零样本 + 未复权; rank-IC≈0 则无 edge, 不值得微调。")


if __name__ == "__main__":
    main()
