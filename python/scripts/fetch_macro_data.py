#!/usr/bin/env python3
"""
获取宏观数据并存入 SQLite
数据源: akshare (免费, 无需 API token)
表: macro_daily

用法:
  python scripts/fetch_macro_data.py              # 获取全部历史
  python scripts/fetch_macro_data.py --update     # 增量更新最近30天
  python scripts/fetch_macro_data.py --start 2020-01-01  # 指定起始日期
"""

import sys, os, sqlite3, argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/stock_data.db')


def fetch_shibor() -> pd.DataFrame:
    """获取 SHIBOR 利率 (隔夜/1周/1月/3月)"""
    try:
        import akshare as ak
        df = ak.macro_china_shibor_all()
        if df is None or len(df) == 0:
            print("  ⚠️ SHIBOR: 无数据")
            return pd.DataFrame()

        # 列名可能是中文或英文
        col_map = {}
        for c in df.columns:
            c_lower = str(c).lower()
            if '日期' in c or 'date' in c_lower:
                col_map['date'] = c
            elif '隔夜' in c or 'o/n' in c_lower or 'on' in c_lower:
                col_map['overnight'] = c
            elif '1周' in c or '1w' in c_lower:
                col_map['1w'] = c
            elif '1月' in c or '1m' in c_lower:
                col_map['1m'] = c
            elif '3月' in c or '3m' in c_lower:
                col_map['3m'] = c

        if 'date' not in col_map:
            print(f"  ⚠️ SHIBOR: 无法识别日期列, 列名: {list(df.columns)}")
            return pd.DataFrame()

        result = pd.DataFrame()
        result['trade_date'] = pd.to_datetime(df[col_map['date']])
        result['shibor_on'] = pd.to_numeric(df[col_map.get('overnight', col_map.get('date'))], errors='coerce')
        if '1w' in col_map:
            result['shibor_1w'] = pd.to_numeric(df[col_map['1w']], errors='coerce')
        if '1m' in col_map:
            result['shibor_1m'] = pd.to_numeric(df[col_map['1m']], errors='coerce')
        if '3m' in col_map:
            result['shibor_3m'] = pd.to_numeric(df[col_map['3m']], errors='coerce')

        result = result.dropna(subset=['shibor_on'])
        print(f"  ✅ SHIBOR: {len(result)} 条 ({result['trade_date'].min().date()} ~ {result['trade_date'].max().date()})")
        return result
    except ImportError:
        print("  ❌ akshare 未安装: pip install akshare")
        return pd.DataFrame()
    except Exception as e:
        print(f"  ❌ SHIBOR: {e}")
        return pd.DataFrame()


def fetch_bond_yield() -> pd.DataFrame:
    """获取中国国债收益率 (10年期)"""
    try:
        import akshare as ak
        df = ak.bond_china_yield()
        if df is None or len(df) == 0:
            print("  ⚠️ 国债收益率: 无数据")
            return pd.DataFrame()

        # 列名通常是 日期, 10年期, 等
        date_col = None
        y10_col = None
        for c in df.columns:
            c_str = str(c)
            if '日期' in c_str or 'date' in c_str.lower():
                date_col = c
            elif '10年' in c_str or '10Y' in c_str.upper() or '10y' in c_str.lower():
                y10_col = c

        if date_col is None or y10_col is None:
            print(f"  ⚠️ 国债收益率: 列名不匹配, 列名: {list(df.columns)[:10]}")
            return pd.DataFrame()

        result = pd.DataFrame()
        result['trade_date'] = pd.to_datetime(df[date_col])
        result['cn_10y_yield'] = pd.to_numeric(df[y10_col], errors='coerce')
        result = result.dropna(subset=['cn_10y_yield'])
        print(f"  ✅ 国债收益率: {len(result)} 条 ({result['trade_date'].min().date()} ~ {result['trade_date'].max().date()})")
        return result
    except ImportError:
        return pd.DataFrame()
    except Exception as e:
        print(f"  ❌ 国债收益率: {e}")
        return pd.DataFrame()


def fetch_usdcny() -> pd.DataFrame:
    """获取 USD/CNY 汇率"""
    try:
        import akshare as ak
        # 使用外汇即期报价
        df = ak.currency_boc_sina(symbol='美元')
        if df is None or len(df) == 0:
            print("  ⚠️ USD/CNY: 无数据")
            return pd.DataFrame()

        # currency_boc_sina 返回的列名: 日期, 中行钞买价, 中行钞卖价, 中行汇买价, 中行汇卖价
        date_col = None
        price_col = None
        for c in df.columns:
            c_str = str(c)
            if '日期' in c_str or 'date' in c_str.lower():
                date_col = c
            elif '中行汇卖价' in c_str or '汇卖价' in c_str:
                price_col = c

        if date_col is None:
            # 尝试第一列作为日期
            date_col = df.columns[0]
        if price_col is None:
            # 尝试最后一列作为价格
            price_col = df.columns[-1]

        result = pd.DataFrame()
        result['trade_date'] = pd.to_datetime(df[date_col])
        result['usdcny'] = pd.to_numeric(df[price_col], errors='coerce') / 100  # 分 → 元
        result = result.dropna(subset=['usdcny'])
        result = result[result['usdcny'] > 5]  # 过滤异常值

        print(f"  ✅ USD/CNY: {len(result)} 条 ({result['trade_date'].min().date()} ~ {result['trade_date'].max().date()})")
        return result
    except ImportError:
        return pd.DataFrame()
    except Exception as e:
        print(f"  ❌ USD/CNY: {e}")
        return pd.DataFrame()


def fetch_hs300_index() -> pd.DataFrame:
    """获取沪深300指数数据"""
    try:
        import akshare as ak
        df = ak.stock_zh_index_daily_em(symbol="sh000300")
        if df is None or len(df) == 0:
            # 尝试 sz399300
            df = ak.stock_zh_index_daily_em(symbol="sz399300")
        if df is None or len(df) == 0:
            print("  ⚠️ 沪深300: 无数据")
            return pd.DataFrame()

        result = pd.DataFrame()
        result['trade_date'] = pd.to_datetime(df['date'])
        result['hs300_close'] = pd.to_numeric(df['close'], errors='coerce')
        result['hs300_volume'] = pd.to_numeric(df['volume'], errors='coerce')
        # 计算涨跌比 (如果有涨跌家数)
        if 'up_count' in df.columns or '上涨家数' in [str(c) for c in df.columns]:
            # 这个API可能没有涨跌家数, 我们先跳过, 用现有的 hs300_daily 表
            pass
        result = result.dropna(subset=['hs300_close'])
        print(f"  ✅ 沪深300: {len(result)} 条 ({result['trade_date'].min().date()} ~ {result['trade_date'].max().date()})")
        return result
    except ImportError:
        return pd.DataFrame()
    except Exception as e:
        print(f"  ❌ 沪深300: {e}")
        return pd.DataFrame()


def build_macro_table(dfs: dict) -> pd.DataFrame:
    """合并所有宏观数据到一张表, 按日期对齐"""
    if not dfs:
        return pd.DataFrame()

    # 从 existing hs300_daily 表获取市场广度 (如果存在)
    try:
        conn = sqlite3.connect(DB_PATH)
        hs300 = pd.read_sql("SELECT trade_date, up_count, volume FROM hs300_daily ORDER BY trade_date", conn)
        conn.close()
        if len(hs300) > 0:
            hs300['trade_date'] = pd.to_datetime(hs300['trade_date'].astype(str))
            hs300['market_breadth'] = hs300['up_count'].astype(float) / 300
            hs300 = hs300[['trade_date', 'market_breadth']]
            dfs['breadth'] = hs300
            print(f"  ✅ 市场广度 (from hs300_daily): {len(hs300)} 条")
    except Exception:
        pass

    # 合并所有数据源
    merged = None
    for name, df in dfs.items():
        if df is None or len(df) == 0:
            continue
        if merged is None:
            merged = df.copy()
        else:
            merged = pd.merge(merged, df, on='trade_date', how='outer')

    if merged is None:
        return pd.DataFrame()

    merged = merged.sort_values('trade_date').reset_index(drop=True)
    # 前向填充缺失值 (宏观数据低频发布, 用最近值填充)
    merged = merged.ffill()
    print(f"\n  合并后: {len(merged)} 条, 列: {list(merged.columns)}")
    return merged


def save_to_db(df: pd.DataFrame):
    """保存到 SQLite 的 macro_daily 表"""
    if df is None or len(df) == 0:
        print("  ⚠️ 无数据可保存")
        return

    conn = sqlite3.connect(DB_PATH)

    # 确保 trade_date 是字符串格式
    df['trade_date'] = df['trade_date'].dt.strftime('%Y-%m-%d')

    # 创建表 (如果不存在)
    cols = [f"{c} REAL" if c != 'trade_date' else 'trade_date TEXT PRIMARY KEY' for c in df.columns]
    conn.execute(f"CREATE TABLE IF NOT EXISTS macro_daily ({', '.join(cols)})")

    # UPSERT
    placeholders = ', '.join(['?'] * len(df.columns))
    cols_str = ', '.join(df.columns)
    update_str = ', '.join([f"{c}=excluded.{c}" for c in df.columns if c != 'trade_date'])

    conn.executemany(
        f"INSERT OR REPLACE INTO macro_daily ({cols_str}) VALUES ({placeholders})",
        df.values.tolist()
    )
    conn.commit()

    # 验证
    count = conn.execute("SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM macro_daily").fetchone()
    conn.close()
    print(f"\n  💾 已保存: {count[0]} 条 ({count[1]} ~ {count[2]})")


def main():
    parser = argparse.ArgumentParser(description='获取宏观数据')
    parser.add_argument('--update', action='store_true', help='仅更新最近30天')
    parser.add_argument('--start', type=str, default=None, help='起始日期 YYYY-MM-DD')
    args = parser.parse_args()

    print("=" * 60)
    print("  宏观数据获取")
    print("=" * 60)

    # 1. 获取各数据源
    dfs = {}
    dfs['shibor'] = fetch_shibor()
    dfs['bond'] = fetch_bond_yield()
    dfs['usdcny'] = fetch_usdcny()
    dfs['hs300'] = fetch_hs300_index()

    # 2. 合并
    merged = build_macro_table(dfs)

    if len(merged) == 0:
        print("\n  ❌ 未获取到任何宏观数据, 请检查网络或 akshare 版本")
        return

    # 3. 过滤日期范围
    if args.update:
        cutoff = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        merged = merged[merged['trade_date'] >= cutoff]
        print(f"\n  增量模式: 保留 {cutoff} 之后的数据 ({len(merged)} 条)")

    if args.start:
        merged = merged[merged['trade_date'] >= args.start]
        print(f"\n  起始日期: {args.start} ({len(merged)} 条)")

    # 4. 保存
    save_to_db(merged)
    print("\n✅ 完成")


if __name__ == '__main__':
    main()