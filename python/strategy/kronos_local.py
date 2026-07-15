#!/usr/bin/env python3
"""Kronos 本地推理桥接 — Mac (MPS) 上对持仓做零样本/微调后的概率预测。

诚实边界:
- kline_daily 是未复权 (raw) 日线, 与 add_advisor / 纸面引擎同口径, 可比。
- 零样本 (未微调) 的单票择时是 Kronos 最弱的一环 (雪球实测"差距蛮大"),
  这里的输出仅供人工评估与喂入纸面引擎前瞻验证, 不直接当实盘信号。
- pred_len=20 对齐 add_advisor 的 horizon(20 交易日), 便于把信号接进纸面引擎。
"""
import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # python/
REPO = os.path.dirname(ROOT)                                               # 仓库根
KRONOS_DIR = os.path.join(REPO, "third_party", "Kronos")
DB_PATH = os.path.join(ROOT, "data", "stock_data.db")

sys.path.insert(0, KRONOS_DIR)  # 使 `from model import ...` 可用

TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
MODEL = "NeoQuasar/Kronos-small"
MAX_CONTEXT = 512

HOLDINGS = ["300124.SZ", "600048.SH", "3690.HK", "300015.SZ", "159792.SZ"]


def load_bars(symbol: str, lookback: int, conn: sqlite3.Connection) -> pd.DataFrame:
    """取最近 lookback 根未复权日线, 升序。kline_daily 无 amount 列 → 交给 Kronos 自动填充。"""
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM kline_daily "
        "WHERE symbol=? ORDER BY date DESC LIMIT ?",
        conn, params=(symbol, lookback),
    )
    if df.empty:
        return df
    df = df.iloc[::-1].reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def predict_symbol(predictor, symbol, lookback, pred_len, T, top_p, sample_count, conn):
    df = load_bars(symbol, lookback, conn)
    if len(df) < 30:
        return None
    x_df = df[["open", "high", "low", "close", "volume"]].copy()
    x_ts = df["date"]
    # 未来 pred_len 个交易日的时间戳 (仅用于时间特征, 用工作日近似)
    y_ts = pd.Series(pd.bdate_range(start=df["date"].iloc[-1] + pd.Timedelta(days=1), periods=pred_len))

    pred = predictor.predict(
        df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
        pred_len=pred_len, T=T, top_p=top_p, sample_count=sample_count, verbose=False,
    )
    last_close = float(df["close"].iloc[-1])
    path = pred["close"].to_numpy(dtype=float)
    horizon_close = float(path[-1])
    return {
        "symbol": symbol,
        "last_date": df["date"].iloc[-1].date().isoformat(),
        "last_close": last_close,
        "pred_close": horizon_close,
        "ret_h": horizon_close / last_close - 1.0,
        "max_close": float(path.max()),
        "min_close": float(path.min()),
        "n_bars": len(df),
    }


def main():
    ap = argparse.ArgumentParser(description="Kronos 本地持仓概率预测")
    ap.add_argument("--symbols", nargs="*", default=HOLDINGS, help="标的代码, 默认 5 只持仓")
    ap.add_argument("--lookback", type=int, default=400)
    ap.add_argument("--pred-len", type=int, default=20, help="预测交易日数, 默认对齐 add_advisor horizon=20")
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--sample-count", type=int, default=5, help=">1 多路径取均值, 曲线更稳")
    ap.add_argument("--device", default=None, help="mps/cpu/cuda:0; 默认自动")
    args = ap.parse_args()

    from model import Kronos, KronosTokenizer, KronosPredictor

    print(f"加载 {MODEL} + {TOKENIZER} (首次会从 HuggingFace 下载)...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER)
    model = Kronos.from_pretrained(MODEL)
    predictor = KronosPredictor(model, tokenizer, device=args.device, max_context=MAX_CONTEXT)
    print(f"设备: {predictor.device}\n")

    conn = sqlite3.connect(DB_PATH)
    print(f"{'代码':<12}{'基准日':<12}{'现价':>10}{'预测(20d)':>12}{'区间收益':>10}  路径高/低")
    print("-" * 78)
    for sym in args.symbols:
        r = predict_symbol(predictor, sym, args.lookback, args.pred_len,
                           args.T, args.top_p, args.sample_count, conn)
        if r is None:
            print(f"{sym:<12} 数据不足, 跳过")
            continue
        print(f"{r['symbol']:<12}{r['last_date']:<12}{r['last_close']:>10.3f}"
              f"{r['pred_close']:>12.3f}{r['ret_h']*100:>9.2f}%"
              f"  {r['max_close']:.2f}/{r['min_close']:.2f}  (n={r['n_bars']})")
    conn.close()
    print("\n⚠️ 零样本单票择时是 Kronos 最弱环节; 未复权价; 仅供评估, 勿直接实盘。")


if __name__ == "__main__":
    main()
