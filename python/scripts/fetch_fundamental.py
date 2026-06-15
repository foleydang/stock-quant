#!/usr/bin/env python3
"""
基本面数据采集 v1

数据源: akshare
输出: fundamental_daily 表 (PE/PB/PS/市值等)

字段:
  symbol, trade_date,
  pe_ttm, pb, ps_ttm, pcf_ttm,
  total_mv, circ_mv,  # 总市值/流通市值(亿)
  roe_ttm,  # 净资产收益率
"""

import os, sys, sqlite3, time, argparse
import numpy as np
import pandas as pd
import akshare as ak

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data/stock_data.db')

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kw): return iterable


def fetch_pe_pb(symbol: str) -> pd.DataFrame:
    """获取单只股票的 PE/PB 历史 (akshare)"""
    try:
        # akshare: stock_a_pe_and_pb → 返回 date, pe, pb
        code = symbol.replace('.SZ', '').replace('.SH', '')
        # 判断交易所
        if symbol.endswith('.SH'):
            raw = f"sh{code}"
        else:
            raw = f"sz{code}"
        df = ak.stock_a_pe_and_pb(symbol=raw)
        if df is None or len(df) == 0:
            return pd.DataFrame()
        df = df.rename(columns={'date': 'trade_date', 'pe': 'pe_ttm', 'pb': 'pb'})
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        return df[['trade_date', 'pe_ttm', 'pb']]
    except Exception:
        return pd.DataFrame()


def fetch_individual_info(symbol: str) -> dict:
    """获取股票基本信息: 总市值、流通市值、ROE"""
    try:
        code = symbol.replace('.SZ', '').replace('.SH', '')
        if symbol.endswith('.SH'):
            raw = f"sh{code}"
        else:
            raw = f"sz{code}"
        info = ak.stock_individual_info_em(symbol=raw)
        if info is None or len(info) == 0:
            return {}
        info_dict = dict(zip(info['item'].values, info['value'].values))
        return {
            'total_mv': float(info_dict.get('总市值', 0)) / 1e8,  # 转为亿
            'circ_mv': float(info_dict.get('流通市值', 0)) / 1e8,
        }
    except Exception:
        return {}


def fetch_all(symbols: list, start_date: str = '2015-01-01'):
    """批量获取所有股票的基本面数据"""
    print(f"📊 获取 {len(symbols)} 只股票基本面数据...")
    all_data = []

    for sym in tqdm(symbols, desc='   PE/PB', unit='stock'):
        df = fetch_pe_pb(sym)
        if len(df) == 0:
            continue
        df['symbol'] = sym
        df = df[df['trade_date'] >= start_date]
        all_data.append(df)
        time.sleep(0.05)  # 限速

    if not all_data:
        print("   ⚠️ 未获取到任何数据")
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    result = result.dropna(subset=['pe_ttm', 'pb'])
    # 过滤异常值
    result = result[(result['pe_ttm'] > 0) & (result['pe_ttm'] < 1000)]
    result = result[(result['pb'] > 0) & (result['pb'] < 100)]

    print(f"   {len(result):,} 条 | {result['symbol'].nunique()} 只股票 | "
          f"{result['trade_date'].min().date()} ~ {result['trade_date'].max().date()}")
    return result


def save_to_db(df: pd.DataFrame):
    """保存到 SQLite"""
    conn = sqlite3.connect(DB_PATH)
    df.to_sql('fundamental_daily', conn, if_exists='replace', index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fundamental_sym ON fundamental_daily(symbol)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fundamental_date ON fundamental_daily(trade_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fundamental_sym_date ON fundamental_daily(symbol,trade_date)")
    conn.commit()
    conn.close()
    print(f"   ✅ 已保存到 fundamental_daily ({len(df):,} 条)")


def main():
    conn = sqlite3.connect(DB_PATH)
    symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM kline_daily")]
    conn.close()

    print(f"   {len(symbols)} 只股票")
    df = fetch_all(symbols)
    if len(df) > 0:
        save_to_db(df)


if __name__ == '__main__':
    main()