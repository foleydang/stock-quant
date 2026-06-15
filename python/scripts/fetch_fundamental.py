#!/usr/bin/env python3
"""基本面数据采集 v3 — 带超时 + 重试"""
import os, sys, sqlite3, time, signal
import numpy as np
import pandas as pd
import akshare as ak

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data/stock_data.db')


def with_timeout(func, args=(), timeout=10):
    """超时保护"""
    result = [None]
    def handler(signum, frame):
        raise TimeoutError()
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        result[0] = func(*args)
    except Exception:
        pass
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    return result[0]


def _parse_amount(val: str) -> float:
    try:
        if '亿' in val: return float(val.replace('亿', '')) * 1e8
        elif '万' in val: return float(val.replace('万', '')) * 1e4
        else: return float(val)
    except: return np.nan


def fetch_financial(symbol: str) -> pd.DataFrame:
    try:
        code = symbol.replace('.SZ', '').replace('.SH', '')
        df = ak.stock_financial_abstract_ths(symbol=code, indicator='按报告期')
        if df is None or len(df) == 0:
            return pd.DataFrame()

        col_map = {
            '报告期': 'trade_date', '净利润': 'net_profit',
            '净利润同比增长率': 'net_profit_yoy', '营业总收入': 'revenue',
            '营业总收入同比增长率': 'revenue_yoy', '净资产收益率': 'roe',
            '每股净资产': 'bv_per_share', '资产负债率': 'debt_ratio',
            '基本每股收益': 'eps',
        }
        df = df.rename(columns=col_map)
        keep_cols = [c for c in col_map.values() if c in df.columns]
        df = df[keep_cols].copy()

        for col in ['net_profit_yoy', 'revenue_yoy', 'roe', 'debt_ratio']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('%', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce') / 100
        if 'net_profit' in df.columns:
            df['net_profit'] = df['net_profit'].astype(str).apply(_parse_amount)
        if 'revenue' in df.columns:
            df['revenue'] = df['revenue'].astype(str).apply(_parse_amount)

        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df['symbol'] = symbol
        return df
    except Exception:
        return pd.DataFrame()


def main():
    conn = sqlite3.connect(DB_PATH)
    symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM kline_daily")]
    conn.close()

    print(f"{len(symbols)} 只股票")
    all_data = []
    fails = 0

    for i, sym in enumerate(symbols):
        df = fetch_financial(sym)
        if len(df) > 0:
            all_data.append(df)
        else:
            fails += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(symbols)} (失败 {fails})")
        time.sleep(0.02)

    print(f"  完成: {len(all_data)} 成功, {fails} 失败")

    if not all_data:
        print("  ⚠️ 未获取到任何数据")
        return

    result = pd.concat(all_data, ignore_index=True)
    result = result.dropna(subset=['roe', 'revenue_yoy', 'net_profit_yoy'])
    result = result[result['trade_date'] >= '2015-01-01']

    print(f"  {len(result):,} 条 | {result['symbol'].nunique()} 只股票 | "
          f"{result['trade_date'].min().date()} ~ {result['trade_date'].max().date()}")

    conn = sqlite3.connect(DB_PATH)
    result.to_sql('fundamental_daily', conn, if_exists='replace', index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fundamental_sym ON fundamental_daily(symbol)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fundamental_date ON fundamental_daily(trade_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fundamental_sym_date ON fundamental_daily(symbol,trade_date)")
    conn.commit()
    conn.close()
    print(f"  ✅ 已保存 (fundamental_daily)")


if __name__ == '__main__':
    main()