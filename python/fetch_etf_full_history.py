#!/usr/bin/env python3
"""159792 港股通互联网ETF + 同类 ETF 全量 OHLCV 回填(日线 + 30分钟)。

设计为在【服务器】上跑(本机东财 push2his 被限流)。两个数据源:

  日线 (全量, 真实OHLCV): 腾讯 fqkline 翻页 — count=640/页, 按 end_date 向前翻
    本机/服务器都稳定不限流。新浪 datalen 硬上限 800, 拉不到 2021。

  30分钟 (全量, 真实OHLCV): akshare stock_zh_a_hist_min_em — 底层东财 push2his
    本机被限流, 服务器正常。159792 现有 30m 仅 2024-06 起(sina 增量攒的),
    本脚本补 2021-09 ~ 2024-06 缺口 + 滚到今天。

幂等: 每个 symbol 按 date 范围 DELETE 后 INSERT, 重复跑安全。

用法 (服务器):
  python fetch_etf_full_history.py                     # 默认: 日线+30min 全量
  python fetch_etf_full_history.py --daily-only          # 只日线(腾讯, 本机也能跑)
  python fetch_etf_full_history.py --min30-only          # 只30min(需东财, 服务器跑)
  python fetch_etf_full_history.py --start 2021-09-01    # 指定起始日
  python fetch_etf_full_history.py --symbols 159792 513050
"""
import os, sys, time, json, argparse, sqlite3
from datetime import datetime, timedelta
import requests
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT, 'data', 'stock_data.db')

DEFAULT_ETF = {
    '159792.SZ': ('sz159792', '0.159792'),   # 港股通互联网ETF (主仓, 2021-09-15成立)
    '513050.SH': ('sh513050', '1.513050'),   # 中概互联网ETF
    '513330.SH': ('sh513330', '1.513330'),   # 恒生互联网ETF
    '159607.SZ': ('sz159607', '0.159607'),   # 中概互联网ETF易方达
}
TENCENT_PAGE = 640   # 腾讯 fqkline 单页最大返回


# ---------------- 日线: 腾讯 fqkline 翻页 ----------------
def tencent_fetch_page(tencent_sym, start, end):
    """返回 list[[date,open,close,high,low,vol,{},amount]] 或 []"""
    u = (f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
         f'?param={tencent_sym},day,{start},{end},{TENCENT_PAGE},qfq')
    r = requests.get(u, headers={'User-Agent': 'Mozilla/5.0',
                                 'Referer': 'https://gu.qq.com/'}, timeout=25)
    j = r.json()
    s = (j.get('data') or {}).get(tencent_sym, {})
    return s.get('day') or []


def fetch_daily_full(tencent_sym, start_date, end_date='2026-12-31'):
    """腾讯 fqkline 翻页, 拉从 start_date 到 end_date 的全部日线。"""
    all_bars = {}
    end = end_date
    for _ in range(10):  # 最多 10 页 = 6400 根
        bars = tencent_fetch_page(tencent_sym, start_date, end)
        if not bars:
            break
        for b in bars:
            all_bars[b[0]] = b
        earliest = bars[0][0]
        if earliest <= start_date.replace('-', ''):
            break
        # 下一页 end = 最早日的前一天
        end = (datetime.strptime(earliest, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        time.sleep(0.3)
    return sorted(all_bars.values(), key=lambda x: x[0])


def upsert_daily(conn, symbol, bars):
    """bars: [[date,open,close,high,low,vol,{},amount], ...]"""
    if not bars:
        return 0
    dates = [b[0] for b in bars]
    conn.execute(f"DELETE FROM kline_daily WHERE symbol=? AND date IN "
                 f"({','.join('?'*len(dates))})", [symbol] + dates)
    n = 0
    for b in bars:
        d, o, c, h, l, v = b[0], float(b[1]), float(b[2]), float(b[3]), float(b[4]), float(b[5])
        conn.execute(
            "INSERT INTO kline_daily(symbol,date,open,high,low,close,volume,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (symbol, d, o, h, l, c, v,
             pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')))
        n += 1
    conn.commit()
    return n


# ---------------- 30分钟: akshare (东财) ----------------
def fetch_min30_full(code6, start_date, end_date):
    """akshare 30分钟全量。code6 如 '159792'。返回 DataFrame 或 None。
    需要 start/end 为 'YYYY-MM-DD HH:MM:SS' 形式。"""
    import akshare as ak
    import warnings; warnings.filterwarnings('ignore')
    for attempt in range(4):
        try:
            df = ak.stock_zh_a_hist_min_em(
                symbol=code6, period='30',
                start_date=f'{start_date} 09:30:00',
                end_date=f'{end_date} 15:00:00', adjust='qfq')
            if df is not None and len(df):
                return df
        except Exception as e:
            print(f'     {code6} 30min try{attempt}: {str(e)[:70]}')
            time.sleep(3)
    return None


def upsert_min30(conn, symbol, df):
    """df: akshare 30min, 列 ['时间','开盘','收盘','最高','最低','成交量',...]"""
    if df is None or df.empty:
        return 0
    # akshare 时间列名 '时间', 格式 '2026-07-21 10:00'
    df = df.copy()
    tcol = '时间' if '时间' in df.columns else df.columns[0]
    df[tcol] = df[tcol].astype(str)
    df = df.drop_duplicates(tcol, keep='last').sort_values(tcol)
    dates = df[tcol].tolist()
    # 删旧再插(幂等)
    conn.execute(f"DELETE FROM kline_30m WHERE symbol=? AND date IN "
                 f"({','.join('?'*len(dates))})", [symbol] + dates)
    n = 0
    for _, r in df.iterrows():
        conn.execute(
            "INSERT INTO kline_30m(symbol,date,open,high,low,close,volume,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (symbol, r[tcol], float(r['开盘']), float(r['最高']), float(r['最低']),
             float(r['收盘']), int(float(r['成交量'])),
             pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')))
        n += 1
    conn.commit()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--daily-only', action='store_true')
    ap.add_argument('--min30-only', action='store_true')
    ap.add_argument('--start', default='2021-09-01', help='起始日 YYYY-MM-DD')
    ap.add_argument('--end', default=pd.Timestamp.now().strftime('%Y-%m-%d'))
    ap.add_argument('--symbols', nargs='*', default=None, help='指定代码如 159792 513050')
    args = ap.parse_args()

    do_daily = not args.min30_only
    do_min30 = not args.daily_only

    sel = DEFAULT_ETF
    if args.symbols:
        sel = {k: v for k, v in DEFAULT_ETF.items()
               if any(s in k for s in args.symbols)}

    conn = sqlite3.connect(DB_PATH)
    print('=' * 64)
    print(f'ETF 全量回填  start={args.start} end={args.end}  '
          f'daily={do_daily} min30={do_min30}')
    print(f'DB: {DB_PATH}')
    print('=' * 64)

    for sym, (tsym, secid) in sel.items():
        code6 = sym.split('.')[0]
        print(f'\n--- {sym} ({tsym}) ---')
        # 日线
        if do_daily:
            bars = fetch_daily_full(tsym, args.start, args.end)
            if bars:
                n = upsert_daily(conn, sym, bars)
                print(f'  日线: 写{n}根 | {bars[0][0]}~{bars[-1][0]}')
            else:
                print('  日线: 无数据')
            time.sleep(0.4)
        # 30min
        if do_min30:
            df = fetch_min30_full(code6, args.start, args.end)
            if df is not None:
                n = upsert_min30(conn, sym, df)
                tcol = '时间' if '时间' in df.columns else df.columns[0]
                print(f'  30min: 写{n}根 | {df[tcol].iloc[0]}~{df[tcol].iloc[-1]}')
            else:
                print('  30min: 拉取失败(东财限流? 服务器上应正常)')

    # 汇总
    print('\n' + '=' * 64)
    print('汇总 (库内现有):')
    for sym in sel:
        d = conn.execute("SELECT MIN(date),MAX(date),COUNT(*) FROM kline_daily WHERE symbol=?",
                         (sym,)).fetchone()
        m = conn.execute("SELECT MIN(date),MAX(date),COUNT(*) FROM kline_30m WHERE symbol=?",
                         (sym,)).fetchone()
        print(f'  {sym}: 日线 {d[2]}根 {d[0]}~{d[1]} | 30min {m[2] if m else 0}根 '
              f'{m[0] if m else "-"}~{m[1] if m else "-"}')
    conn.close()
    print('\n✅ 完成')


if __name__ == '__main__':
    main()
