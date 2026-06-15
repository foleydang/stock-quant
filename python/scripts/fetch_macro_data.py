#!/usr/bin/env python3
"""
获取宏观数据并存入 SQLite (v2)
数据源: akshare
表: macro_daily

数据:
  - SHIBOR (隔夜/1周/1月/3月)
  - 中国国债收益率 (2Y/5Y/10Y/30Y) + 美国国债收益率
  - USD/CNY (在岸) + USD/CNH (离岸)
  - 沪深300指数 (收盘/成交量)
  - 中美利差 (10Y)
"""

import sys, os, sqlite3, argparse
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/stock_data.db')


def fetch_shibor() -> pd.DataFrame:
    """SHIBOR 利率"""
    import akshare as ak
    df = ak.macro_china_shibor_all()
    col_map = {}
    for c in df.columns:
        c_str = str(c)
        if '日期' in c_str or 'date' in c_str.lower(): col_map['date'] = c
        elif '隔夜' in c_str or 'o/n' in c_str.lower(): col_map['on'] = c
        elif '1周' in c_str or '1w' in c_str.lower(): col_map['1w'] = c
        elif '1月' in c_str or '1m' in c_str.lower(): col_map['1m'] = c
        elif '3月' in c_str or '3m' in c_str.lower(): col_map['3m'] = c

    result = pd.DataFrame()
    result['trade_date'] = pd.to_datetime(df[col_map['date']]).dt.strftime('%Y-%m-%d')
    for key, col_name in [('shibor_on', 'on'), ('shibor_1w', '1w'), ('shibor_1m', '1m'), ('shibor_3m', '3m')]:
        if col_name in col_map:
            result[key] = pd.to_numeric(df[col_map[col_name]], errors='coerce')
    result = result.dropna(subset=['shibor_on'])
    print(f"  ✅ SHIBOR: {len(result)} 条 ({result['trade_date'].iloc[0]} ~ {result['trade_date'].iloc[-1]})")
    return result


def fetch_bond_yields() -> pd.DataFrame:
    """中国+美国国债收益率"""
    import akshare as ak
    df = ak.bond_zh_us_rate()
    result = pd.DataFrame()
    result['trade_date'] = pd.to_datetime(df['日期']).dt.strftime('%Y-%m-%d')
    # 中国国债
    for cn, en in [('中国国债收益率2年', 'cn_2y'), ('中国国债收益率5年', 'cn_5y'),
                    ('中国国债收益率10年', 'cn_10y'), ('中国国债收益率30年', 'cn_30y')]:
        if cn in df.columns:
            result[en] = pd.to_numeric(df[cn], errors='coerce')
    # 美国国债
    for cn, en in [('美国国债收益率2年', 'us_2y'), ('美国国债收益率5年', 'us_5y'),
                    ('美国国债收益率10年', 'us_10y'), ('美国国债收益率30年', 'us_30y')]:
        if cn in df.columns:
            result[en] = pd.to_numeric(df[cn], errors='coerce')
    # 中美利差
    if 'cn_10y' in result.columns and 'us_10y' in result.columns:
        result['cn_us_spread'] = result['cn_10y'] - result['us_10y']
    result = result.dropna(subset=['cn_10y'])
    print(f"  ✅ 国债收益率: {len(result)} 条 ({result['trade_date'].iloc[0]} ~ {result['trade_date'].iloc[-1]})")
    return result


def fetch_usdcny() -> pd.DataFrame:
    """USD/CNY 汇率 (在岸 + 离岸)"""
    import akshare as ak
    result = None

    # 在岸 CNY (BOC中间价)
    try:
        df = ak.currency_boc_sina(symbol='美元', start_date='20150101', end_date='20260615')
        tmp = pd.DataFrame()
        tmp['trade_date'] = pd.to_datetime(df['日期']).dt.strftime('%Y-%m-%d')
        # 使用央行中间价, 缺失用汇卖价/100
        mid = pd.to_numeric(df['央行中间价'], errors='coerce')
        sell = pd.to_numeric(df['中行钞卖价/汇卖价'], errors='coerce') / 100
        tmp['usdcny'] = mid.fillna(sell)
        result = tmp.dropna(subset=['usdcny'])
        print(f"  ✅ USD/CNY(在岸): {len(result)} 条 ({result['trade_date'].iloc[0]} ~ {result['trade_date'].iloc[-1]})")
    except Exception as e:
        print(f"  ⚠️ USD/CNY(在岸) 失败: {e}")

    # 离岸 CNH
    try:
        df = ak.forex_hist_em(symbol='USDCNH')
        tmp = pd.DataFrame()
        tmp['trade_date'] = pd.to_datetime(df['日期']).dt.strftime('%Y-%m-%d')
        tmp['usdcnh'] = pd.to_numeric(df['最新价'], errors='coerce')
        tmp = tmp.dropna(subset=['usdcnh'])
        if result is not None:
            result = pd.merge(result, tmp, on='trade_date', how='outer')
        else:
            result = tmp
        print(f"  ✅ USD/CNH(离岸): {len(tmp)} 条 ({tmp['trade_date'].iloc[0]} ~ {tmp['trade_date'].iloc[-1]})")
    except Exception as e:
        print(f"  ⚠️ USD/CNH(离岸) 失败: {e}")

    return result if result is not None else pd.DataFrame()


def fetch_hs300() -> pd.DataFrame:
    """沪深300指数数据"""
    import akshare as ak
    df = ak.stock_zh_index_daily(symbol='sh000300')
    result = pd.DataFrame()
    result['trade_date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    result['hs300_close'] = pd.to_numeric(df['close'], errors='coerce')
    result['hs300_volume'] = pd.to_numeric(df['volume'], errors='coerce')
    result = result.dropna(subset=['hs300_close'])
    print(f"  ✅ 沪深300: {len(result)} 条 ({result['trade_date'].iloc[0]} ~ {result['trade_date'].iloc[-1]})")
    return result


def merge_and_save(dfs: list):
    """合并所有数据源并保存"""
    if not dfs:
        print("❌ 无数据")
        return

    merged = dfs[0]
    for df in dfs[1:]:
        if df is not None and len(df) > 0:
            merged = pd.merge(merged, df, on='trade_date', how='outer')

    merged = merged.sort_values('trade_date').reset_index(drop=True)
    # 前向填充 (低频宏观数据用最近值)
    merged = merged.ffill()

    print(f"\n  合并后: {len(merged)} 条, {len(merged.columns)} 列")

    conn = sqlite3.connect(DB_PATH)
    merged.to_sql('macro_daily', conn, if_exists='replace', index=False)
    # 验证
    count = conn.execute("SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM macro_daily").fetchone()
    conn.close()
    print(f"  💾 已保存 macro_daily: {count[0]} 条 ({count[1]} ~ {count[2]})")


def main():
    print("=" * 60)
    print("  宏观数据获取 v2")
    print("=" * 60)

    dfs = [
        fetch_hs300(),
        fetch_shibor(),
        fetch_bond_yields(),
        fetch_usdcny(),
    ]

    merge_and_save([d for d in dfs if d is not None and len(d) > 0])
    print("\n✅ 完成")


if __name__ == '__main__':
    main()