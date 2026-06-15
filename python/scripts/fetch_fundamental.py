#!/usr/bin/env python3
"""
基本面数据采集 v2

数据源: akshare stock_financial_abstract_ths
输出: fundamental_daily 表

字段:
  symbol, trade_date (报告期),
  net_profit, net_profit_yoy,  # 净利润/同比
  revenue, revenue_yoy,        # 营收/同比
  roe,                         # 净资产收益率
  bv_per_share,                # 每股净资产
  debt_ratio,                  # 资产负债率
  eps                          # 每股收益
"""

import os, sys, sqlite3, time
import numpy as np
import pandas as pd
import akshare as ak

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data/stock_data.db')

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kw): return iterable


def fetch_financial(symbol: str) -> pd.DataFrame:
    """获取单只股票的财务摘要"""
    try:
        # 提取纯数字代码
        code = symbol.replace('.SZ', '').replace('.SH', '')
        df = ak.stock_financial_abstract_ths(symbol=code, indicator='按报告期')
        if df is None or len(df) == 0:
            return pd.DataFrame()

        # 重命名列
        col_map = {
            '报告期': 'trade_date',
            '净利润': 'net_profit',
            '净利润同比增长率': 'net_profit_yoy',
            '营业总收入': 'revenue',
            '营业总收入同比增长率': 'revenue_yoy',
            '净资产收益率': 'roe',
            '每股净资产': 'bv_per_share',
            '资产负债率': 'debt_ratio',
            '基本每股收益': 'eps',
        }
        df = df.rename(columns=col_map)
        keep_cols = [c for c in col_map.values() if c in df.columns]
        df = df[keep_cols].copy()

        # 清理数据: 百分比转数值
        for col in ['net_profit_yoy', 'revenue_yoy', 'roe', 'debt_ratio']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('%', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce') / 100

        # 净利润转数值 (去除"万"、"亿"等)
        if 'net_profit' in df.columns:
            df['net_profit'] = df['net_profit'].astype(str).apply(_parse_amount)
        if 'revenue' in df.columns:
            df['revenue'] = df['revenue'].astype(str).apply(_parse_amount)

        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df['symbol'] = symbol
        return df
    except Exception:
        return pd.DataFrame()


def _parse_amount(val: str) -> float:
    """解析金额字符串: '1.13亿' → 113000000, '4302.00万' → 43020000"""
    try:
        if '亿' in val:
            return float(val.replace('亿', '')) * 1e8
        elif '万' in val:
            return float(val.replace('万', '')) * 1e4
        else:
            return float(val)
    except Exception:
        return np.nan


def fetch_all(symbols: list):
    """批量获取所有股票的财务数据"""
    print(f"📊 获取 {len(symbols)} 只股票财务数据...")
    all_data = []

    for sym in tqdm(symbols, desc='   财务数据', unit='stock'):
        df = fetch_financial(sym)
        if len(df) > 0:
            all_data.append(df)
        time.sleep(0.1)  # 限速

    if not all_data:
        print("   ⚠️ 未获取到任何数据")
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    result = result.dropna(subset=['roe', 'revenue_yoy', 'net_profit_yoy'])
    result = result[result['trade_date'] >= '2015-01-01']

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