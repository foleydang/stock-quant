#!/usr/bin/env python3
"""全量回填 ETF + 港股成分股日线历史 (akshare, 不受新浪 800 根上限)。

新浪 datalen 上限 800 根 (~3.3年), 想要 2020 起的长历史必须走 akshare 的
fund_etf_hist_em (ETF) / stock_hk_daily (港股) 全量接口。

本脚本对 159792 + 3 个同类 ETF + 10 只 HSTECH 成分股做全量回填,
打印每个标的的最早可拉日期, 帮你确认 159792 本体上市日。

用法:  python backfill_etf_history.py            # 全量回填
        python backfill_etf_history.py --check    # 只查最早日期, 不写库
"""
import os, sys, time, sqlite3, argparse
import pandas as pd
import akshare as ak

ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT, 'data', 'stock_data.db')

ETF_SYMBOLS = {
    '159792.SZ': '159792',   # 港股通互联网ETF (主仓, 本体)
    '513050.SH': '513050',   # 中概互联网ETF (2020起, 长历史)
    '513330.SH': '513330',   # 恒生互联网ETF (2020起)
    '159607.SZ': '159607',   # 中概互联网ETF易方达
}
HK_COMPONENTS = ['00700','09988','03690','09618','01024',
                 '01810','09999','09888','02015','00981']


def fetch_etf_full(code):
    """akshare ETF 全量日线(前复权)。返回 df 或 None。"""
    for attempt in range(3):
        try:
            df = ak.fund_etf_hist_em(symbol=code, period='daily',
                                      start_date='20150101', end_date='20261231',
                                      adjust='qfq')
            if df is not None and len(df):
                return df
        except Exception as e:
            print(f'   {code} 尝试{attempt}失败: {str(e)[:80]}'); time.sleep(1)
    return None


def fetch_hk_full(code5):
    for attempt in range(3):
        try:
            df = ak.stock_hk_daily(symbol=code5, adjust='qfq')
            if df is not None and len(df):
                return df
        except Exception as e:
            print(f'   {code5}.HK 尝试{attempt}失败: {str(e)[:80]}'); time.sleep(1)
    return None


def upsert(conn, symbol, df, date_col='日期'):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
    df = df.drop_duplicates(date_col, keep='last').sort_values(date_col)
    n = 0
    for _, r in df.iterrows():
        conn.execute("""
            INSERT OR REPLACE INTO kline_daily(symbol,date,open,high,low,close,volume,updated_at)
            VALUES(?,?,?,?,?,?,?,?)
        """, (symbol, r[date_col], float(r['开盘']), float(r['最高']),
              float(r['最低']), float(r['收盘']), float(r['成交量']),
              pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')))
        n += 1
    conn.commit()
    return n, df[date_col].iloc[0], df[date_col].iloc[-1]


def upsert_hk(conn, symbol, df):
    """stock_hk_daily 列名: date,open,high,low,close,volume (英文)"""
    df = df.copy().reset_index()
    if 'date' not in df.columns:
        df.columns = ['date','open','high','low','close','volume'] + list(df.columns[6:])
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df = df.drop_duplicates('date', keep='last').sort_values('date')
    n = 0
    for _, r in df.iterrows():
        conn.execute("""
            INSERT OR REPLACE INTO kline_daily(symbol,date,open,high,low,close,volume,updated_at)
            VALUES(?,?,?,?,?,?,?,?)
        """, (symbol, r['date'], float(r['open']), float(r['high']),
              float(r['low']), float(r['close']), float(r['volume']),
              pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')))
        n += 1
    conn.commit()
    return n, df['date'].iloc[0], df['date'].iloc[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='只查最早日期不写库')
    args = ap.parse_args()
    conn = sqlite3.connect(DB_PATH)

    print("=" * 60)
    print("全量回填 ETF + 港股成分股日线 (akshare)")
    print("=" * 60)

    print("\n--- ETF ---")
    for sym, code in ETF_SYMBOLS.items():
        df = fetch_etf_full(code)
        if df is None:
            print(f"  {sym}: 拉取失败"); continue
        first, last = df['日期'].astype(str).iloc[0], df['日期'].astype(str).iloc[-1]
        print(f"  {sym}: {len(df)} 根 | {first} ~ {last}", end='')
        if not args.check:
            n, _, _ = upsert(conn, sym, df)
            print(f" | 写入 {n} 行")
        else:
            print()
        time.sleep(0.5)

    print("\n--- 港股成分股 ---")
    for code5 in HK_COMPONENTS:
        df = fetch_hk_full(code5)
        if df is None:
            print(f"  {code5}.HK: 拉取失败"); continue
        df = df.reset_index() if 'date' not in df.columns else df
        dates = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        print(f"  {code5}.HK: {len(df)} 根 | {dates.iloc[0]} ~ {dates.iloc[-1]}", end='')
        if not args.check:
            try:
                n, _, _ = upsert_hk(conn, f'{code5}.HK', df)
                print(f" | 写入 {n} 行")
            except Exception as e:
                print(f" | 写入失败 {str(e)[:80]}")
        else:
            print()
        time.sleep(0.5)

    conn.close()
    print("\n✅ 完成。159792 若最早日≈2023-03 则确认本体上市于2023, 2020历史只能用 513050/513330 代理。")


if __name__ == '__main__':
    main()
