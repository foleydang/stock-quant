#!/usr/bin/env python3
"""
拉取情绪因子数据 → stock_data.db

数据源:
1. 龙虎榜 (stock_lhb_detail_em) — 2010-2026, 逐月拉取
2. 融资融券 (stock_margin_detail_sse/szse) — 月度采样
3. K线情绪指标 — 从已有 kline_daily 计算 (涨跌停/异常量/异常收益)

表结构:
- sentiment_lhb     : 龙虎榜日明细
- sentiment_margin   : 融资融券月采样
- sentiment_daily    : 每日情绪聚合 (含K线情绪指标)
"""

import os
import sys
import sqlite3
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/stock_data.db')


def create_tables(conn):
    """创建情绪数据表"""
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS sentiment_lhb (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        name TEXT,
        trade_date TEXT NOT NULL,
        close REAL,
        pct_chg REAL,
        lhb_net_buy REAL,
        lhb_buy REAL,
        lhb_sell REAL,
        lhb_amount REAL,
        total_amount REAL,
        net_buy_ratio REAL,
        turnover REAL,
        reason TEXT,
        ret_1d REAL,
        ret_2d REAL,
        ret_5d REAL,
        ret_10d REAL,
        UNIQUE(symbol, trade_date)
    );

    CREATE TABLE IF NOT EXISTS sentiment_margin (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        trade_date TEXT NOT NULL,
        margin_balance REAL,
        margin_buy REAL,
        margin_repay REAL,
        short_balance REAL,
        short_sell REAL,
        short_repay REAL,
        exchange TEXT,
        UNIQUE(symbol, trade_date, exchange)
    );

    CREATE TABLE IF NOT EXISTS sentiment_daily (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        trade_date TEXT NOT NULL,
        -- 龙虎榜特征
        lhb_flag INTEGER DEFAULT 0,
        lhb_net_buy REAL DEFAULT 0,
        lhb_net_buy_ratio REAL DEFAULT 0,
        lhb_ret_5d REAL DEFAULT 0,
        -- 融资融券特征
        margin_balance REAL DEFAULT 0,
        margin_balance_chg REAL DEFAULT 0,
        short_balance REAL DEFAULT 0,
        -- K线情绪特征
        is_limit_up INTEGER DEFAULT 0,
        is_limit_down INTEGER DEFAULT 0,
        vol_ratio_20 REAL DEFAULT 0,
        abnormal_ret REAL DEFAULT 0,
        consecutive_limit_up INTEGER DEFAULT 0,
        UNIQUE(symbol, trade_date)
    );

    CREATE INDEX IF NOT EXISTS idx_lhb_symbol ON sentiment_lhb(symbol);
    CREATE INDEX IF NOT EXISTS idx_lhb_date ON sentiment_lhb(trade_date);
    CREATE INDEX IF NOT EXISTS idx_margin_symbol ON sentiment_margin(symbol);
    CREATE INDEX IF NOT EXISTS idx_margin_date ON sentiment_margin(trade_date);
    CREATE INDEX IF NOT EXISTS idx_sentiment_symbol ON sentiment_daily(symbol);
    CREATE INDEX IF NOT EXISTS idx_sentiment_date ON sentiment_daily(trade_date);
    """)
    conn.commit()


def fetch_lhb_data(conn, start_year=2012, end_year=2026):
    """拉取龙虎榜数据 (按月)"""
    import akshare as ak

    total = 0
    failed_months = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # 跳过未来月份
            now = datetime.now()
            if year > now.year or (year == now.year and month > now.month):
                break

            last_day = 28
            if month in [1, 3, 5, 7, 8, 10, 12]:
                last_day = 31
            elif month in [4, 6, 9, 11]:
                last_day = 30
            else:
                last_day = 29 if year % 4 == 0 else 28

            start_date = f"{year}{month:02d}01"
            end_date = f"{year}{month:02d}{last_day}"

            try:
                df = ak.stock_lhb_detail_em(start_date=start_date, end_date=end_date)
                if len(df) == 0:
                    continue

                records = []
                for _, row in df.iterrows():
                    symbol = row.get('代码', '')
                    records.append((
                        symbol,
                        row.get('名称', ''),
                        row.get('上榜日', ''),
                        float(row.get('收盘价', 0) or 0),
                        float(row.get('涨跌幅', 0) or 0),
                        float(row.get('龙虎榜净买额', 0) or 0),
                        float(row.get('龙虎榜买入额', 0) or 0),
                        float(row.get('龙虎榜卖出额', 0) or 0),
                        float(row.get('龙虎榜成交额', 0) or 0),
                        float(row.get('市场总成交额', 0) or 0),
                        float(row.get('净买额占总成交比', 0) or 0),
                        float(row.get('换手率', 0) or 0),
                        str(row.get('上榜原因', '')),
                        float(row.get('上榜后1日', 0) or 0),
                        float(row.get('上榜后2日', 0) or 0),
                        float(row.get('上榜后5日', 0) or 0),
                        float(row.get('上榜后10日', 0) or 0),
                    ))

                conn.executemany(
                    """INSERT OR REPLACE INTO sentiment_lhb
                    (symbol, name, trade_date, close, pct_chg, lhb_net_buy, lhb_buy, lhb_sell,
                     lhb_amount, total_amount, net_buy_ratio, turnover, reason,
                     ret_1d, ret_2d, ret_5d, ret_10d)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    records
                )
                conn.commit()
                total += len(records)

            except Exception as e:
                failed_months.append(f"{start_date}-{end_date}: {str(e)[:60]}")
                continue

            time.sleep(0.1)  # 限速

    print(f"龙虎榜: {total} 条记录")
    if failed_months:
        print(f"  失败月份: {len(failed_months)}")
        for f in failed_months[:5]:
            print(f"    {f}")
    return total


def fetch_margin_data(conn, start_year=2015, end_year=2026):
    """
    融资融券月度采样 (每月1号, 取最近交易日)
    上交所 + 深交所
    """
    import akshare as ak

    total = 0
    failed = 0

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            now = datetime.now()
            if year > now.year or (year == now.year and month > now.month):
                break

            # 尝试每月15号左右 (月中数据比较稳定)
            target_date = f"{year}{month:02d}15"

            for exchange, fn in [('sse', ak.stock_margin_detail_sse), ('szse', ak.stock_margin_detail_szse)]:
                try:
                    # 尝试几个日期 (15号, 14号, 16号...)
                    for day_offset in [0, -1, 1, -2, 2]:
                        try_date = (datetime(year, month, 15) + timedelta(days=day_offset)).strftime('%Y%m%d')
                        df = fn(date=try_date)
                        if len(df) > 0:
                            records = []
                            for _, row in df.iterrows():
                                symbol = row.get('标的证券代码', '')
                                if not symbol:
                                    continue
                                records.append((
                                    symbol,
                                    try_date,
                                    float(row.get('融资余额', 0) or 0),
                                    float(row.get('融资买入额', 0) or 0),
                                    float(row.get('融资偿还额', 0) or 0),
                                    float(row.get('融券余量', 0) or 0),
                                    float(row.get('融券卖出量', 0) or 0),
                                    float(row.get('融券偿还量', 0) or 0),
                                    exchange,
                                ))
                            if records:
                                conn.executemany(
                                    """INSERT OR REPLACE INTO sentiment_margin
                                    (symbol, trade_date, margin_balance, margin_buy, margin_repay,
                                     short_balance, short_sell, short_repay, exchange)
                                    VALUES (?,?,?,?,?,?,?,?,?)""",
                                    records
                                )
                                conn.commit()
                                total += len(records)
                            break
                except Exception:
                    continue

            time.sleep(0.1)

    print(f"融资融券: {total} 条记录, 失败: {failed}")


def compute_kline_sentiment(conn):
    """从 kline_daily 计算情绪指标 (涨跌停/异常量/异常收益)"""
    df = pd.read_sql("SELECT symbol, date, close, preclose, pct_chg, volume FROM kline_daily", conn)
    if len(df) == 0:
        print("kline_daily 无数据, 跳过")
        return 0

    df['date'] = pd.to_datetime(df['date'], format='mixed').dt.strftime('%Y-%m-%d')
    df = df.sort_values(['symbol', 'date'])

    # 涨跌停
    df['is_limit_up'] = (df['pct_chg'] >= 9.5).astype(int)
    df['is_limit_down'] = (df['pct_chg'] <= -9.5).astype(int)

    # 异常成交量 (当日量 / 20日均量)
    df['vol_ma20'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df['vol_ratio_20'] = np.where(df['vol_ma20'] > 0, df['volume'] / df['vol_ma20'], 1.0)

    # 异常收益率 (z-score)
    df['ret_ma20'] = df.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df['ret_std20'] = df.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(20, min_periods=5).std())
    df['abnormal_ret'] = np.where(
        df['ret_std20'] > 0,
        (df['pct_chg'] - df['ret_ma20']) / df['ret_std20'],
        0.0
    )

    # 连续涨停天数
    df['consecutive_limit_up'] = df.groupby('symbol')['is_limit_up'].transform(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
    )
    df['consecutive_limit_up'] = df['consecutive_limit_up'].where(df['is_limit_up'] == 1, 0)

    records = df[['symbol', 'date', 'is_limit_up', 'is_limit_down', 'vol_ratio_20', 'abnormal_ret', 'consecutive_limit_up']].values.tolist()

    conn.execute("DELETE FROM sentiment_daily WHERE 1=1")
    conn.executemany(
        """INSERT INTO sentiment_daily
        (symbol, trade_date, is_limit_up, is_limit_down, vol_ratio_20, abnormal_ret, consecutive_limit_up)
        VALUES (?,?,?,?,?,?,?)""",
        records
    )
    conn.commit()

    print(f"K线情绪: {len(records)} 条记录")
    return len(records)


def merge_lhb_to_sentiment(conn):
    """将龙虎榜数据合并到 sentiment_daily"""
    lhb = pd.read_sql("SELECT symbol, trade_date, lhb_net_buy, net_buy_ratio, ret_5d FROM sentiment_lhb", conn)
    if len(lhb) == 0:
        print("无龙虎榜数据可合并")
        return

    lhb['lhb_flag'] = 1
    lhb = lhb.rename(columns={
        'lhb_net_buy': 'lhb_net_buy',
        'net_buy_ratio': 'lhb_net_buy_ratio',
        'ret_5d': 'lhb_ret_5d'
    })

    for _, row in lhb.iterrows():
        conn.execute(
            """UPDATE sentiment_daily SET
               lhb_flag=?, lhb_net_buy=?, lhb_net_buy_ratio=?, lhb_ret_5d=?
               WHERE symbol=? AND trade_date=?""",
            (row['lhb_flag'], row['lhb_net_buy'], row['lhb_net_buy_ratio'],
             row['lhb_ret_5d'], row['symbol'], row['trade_date'])
        )

    conn.commit()
    print(f"龙虎榜合并: {len(lhb)} 条")


def show_summary(conn):
    """显示数据概览"""
    tables = ['sentiment_lhb', 'sentiment_margin', 'sentiment_daily']
    for t in tables:
        cur = conn.execute(f"SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM {t}")
        row = cur.fetchone()
        if row and row[0]:
            print(f"  {t}: {row[0]:,} 条, {row[1]} ~ {row[2]}")
        else:
            print(f"  {t}: 空")

    # 样本特征
    cur = conn.execute("SELECT COUNT(*), COUNT(DISTINCT symbol) FROM sentiment_daily WHERE lhb_flag=1")
    row = cur.fetchone()
    print(f"  龙虎榜上榜: {row[0]:,} 条, {row[1]} 只股票")

    cur = conn.execute("SELECT COUNT(*), COUNT(DISTINCT symbol) FROM sentiment_daily WHERE is_limit_up=1")
    row = cur.fetchone()
    print(f"  涨停: {row[0]:,} 条, {row[1]} 只股票")


def main():
    print("=" * 60)
    print("情绪因子数据拉取")
    print(f"DB: {DB_PATH}")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)

    print("\n[1/4] 拉取龙虎榜数据 (2010-2026, 按月)...")
    fetch_lhb_data(conn)

    print("\n[2/4] 拉取融资融券数据 (2015-2026, 月采样)...")
    fetch_margin_data(conn)

    print("\n[3/4] 计算K线情绪指标 (涨跌停/异常量/异常收益)...")
    compute_kline_sentiment(conn)

    print("\n[4/4] 合并龙虎榜到每日情绪表...")
    merge_lhb_to_sentiment(conn)

    print("\n" + "=" * 60)
    print("数据概览")
    print("=" * 60)
    show_summary(conn)

    conn.close()
    print("\n✅ 完成!")


if __name__ == '__main__':
    main()