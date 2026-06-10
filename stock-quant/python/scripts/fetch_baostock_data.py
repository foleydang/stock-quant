#!/usr/bin/env python3
"""
BaoStock 数据拉取器 - 带限频和内存控制

逐只股票处理，每只之间sleep 3秒，避免OOM和限频。
每次只做一件事（Tushare/BaoStock/AkShare 限频限制）。

运行方式:
  python3 fetch_baostock_data.py --step sector    # 板块映射
  python3 fetch_baostock_data.py --step hs300      # 沪深300日线
  python3 fetch_baostock_data.py --step daily      # 个股日线（逐只）
  python3 fetch_baostock_data.py --step north       # 北向资金补全(akshare)
  python3 fetch_baostock_data.py --step all         # 按优先级依次执行
"""

import baostock as bs
import sqlite3
import time
import argparse
from datetime import datetime
from collections import Counter

DB_PATH = '/root/github/stock-quant/stock-quant/python/data/stock_data.db'
SLEEP_BETWEEN_STOCKS = 3  # 每只股票之间sleep秒数


def step_sector():
    """板块映射 - BaoStock query_stock_industry"""
    print("\n[sector] 拉取行业分类...")
    
    conn = sqlite3.connect(DB_PATH)
    lg = bs.login()
    print(f"  登录: {lg.error_msg}")
    
    rs = bs.query_stock_industry()
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    
    print(f"  BaoStock返回: {len(rows)} 条")
    if not rows:
        print("  ❌ 无数据")
        bs.logout()
        conn.close()
        return
    
    # 格式: (code, code_name, industry, industryClassification, ...)
    # code = sh.600036, industry = 银行
    count = 0
    for row in rows:
        try:
            code = row[0]       # sh.600036
            name = row[1]       # 招商银行
            industry = row[2]   # 银行 (申万L1)
            if not industry:
                industry = row[3] if len(row) > 3 else '其他'
            
            # 转换: sh.600036 → 600036.SH
            parts = code.split('.')
            symbol = f"{parts[1]}.{parts[0].upper()}"
            
            conn.execute(
                """INSERT OR REPLACE INTO stock_sector 
                (symbol, name, industry, sector_code, updated_at)
                VALUES (?, ?, ?, ?, ?)""",
                (symbol, name, industry, industry,
                 datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            )
            count += 1
        except Exception as e:
            pass
    
    conn.commit()
    bs.logout()
    
    # 统计
    others = conn.execute("SELECT COUNT(*) FROM stock_sector WHERE industry='其他' OR industry IS NULL").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM stock_sector").fetchone()[0]
    industries = conn.execute("SELECT COUNT(DISTINCT industry) FROM stock_sector WHERE industry != '其他'").fetchone()[0]
    print(f"  ✅ 写入: {count} 条, {industries}个行业, {others}只'其他'/{total}总")
    
    # 验证关键股票
    for sym in ['600036.SH', '000001.SZ', '600519.SH']:
        r = conn.execute("SELECT name, industry FROM stock_sector WHERE symbol=?", (sym,)).fetchone()
        print(f"    {sym}: {r}")
    
    conn.close()


def step_hs300():
    """沪深300日线 - BaoStock"""
    print("\n[hs300] 拉取沪深300日线...")
    
    conn = sqlite3.connect(DB_PATH)
    lg = bs.login()
    print(f"  登录: {lg.error_msg}")
    
    # 查已有数据范围
    try:
        max_date_raw = conn.execute("SELECT MAX(trade_date) FROM hs300_daily").fetchone()[0]
        # BaoStock用的YYYYMMDD格式，直接从那一天开始（已存在的会被REPLACE）
        start_date = max_date_raw if max_date_raw else '20230101'
    except:
        start_date = '20230101'
    
    rs = bs.query_history_k_data_plus(
        "sh.000300",
        "date,open,high,low,close,volume,amount,pctChg",
        start_date='2023-01-01', end_date='2026-06-10',
        frequency="d", adjustflag="3"
    )
    
    count = 0
    while rs.next():
        row = rs.get_row_data()
        # row: [date(YYYY-MM-DD), open, high, low, close, volume, amount, pctChg]
        date_raw = row[0]  # 2023-01-03
        trade_date = date_raw.replace('-', '')  # → 20230103
        try:
            conn.execute(
                """INSERT OR REPLACE INTO hs300_daily 
                (trade_date, open, close, high, low, volume, amount, pct_chg)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (trade_date, float(row[1]), float(row[4]), float(row[2]),
                 float(row[3]), float(row[5]), float(row[6]), float(row[7]))
            )
            count += 1
        except Exception as e:
            pass
    
    conn.commit()
    bs.logout()
    
    total = conn.execute("SELECT COUNT(*) FROM hs300_daily").fetchone()[0]
    range_data = conn.execute("SELECT MIN(trade_date), MAX(trade_date) FROM hs300_daily").fetchone()
    print(f"  ✅ 沪深300: {count}条新增, 共{total}条, {range_data[0]}~{range_data[1]}")
    
    # 验证最近3天
    cur = conn.execute("SELECT trade_date, close, pct_chg FROM hs300_daily ORDER BY trade_date DESC LIMIT 3")
    for r in cur.fetchall():
        print(f"    {r[0]}: close={r[1]}, pct_chg={r[2]}%")
    
    conn.close()


def step_daily(batch_size=20):
    """个股日线 - 逐只拉取，带sleep"""
    print(f"\n[daily] 拉取个股日线（每批{batch_size}只，每只sleep {SLEEP_BETWEEN_STOCKS}s）...")
    
    conn = sqlite3.connect(DB_PATH)
    
    # 获取需要拉日线的股票列表
    symbols_in_30m = set(r[0] for r in conn.execute("SELECT DISTINCT symbol FROM kline_30m").fetchall())
    # 排除港股和ETF
    symbols = [s for s in symbols_in_30m if not s.endswith('.HK') and not s.startswith(('51', '159', '510'))]
    
    # 查已有日线覆盖
    existing_daily_symbols = set(r[0] for r in conn.execute("SELECT DISTINCT symbol FROM kline_daily").fetchall())
    symbols_to_fetch = [s for s in symbols if s not in existing_daily_symbols]
    
    # 如果都已有，检查日期覆盖
    if not symbols_to_fetch:
        # 检查哪只股票日线不够（需要>=500天≈2年）
        for sym in symbols:
            cnt = conn.execute("SELECT COUNT(*) FROM kline_daily WHERE symbol=?", (sym,)).fetchone()[0]
            if cnt < 500:
                symbols_to_fetch.append(sym)
    
    print(f"  需拉取: {len(symbols_to_fetch)} 只, 本次批: {min(batch_size, len(symbols_to_fetch))} 只")
    
    lg = bs.login()
    print(f"  登录: {lg.error_msg}")
    
    total_count = 0
    batch = symbols_to_fetch[:batch_size]
    
    for i, symbol in enumerate(batch):
        # 转换: 600036.SH → sh.600036
        code, market = symbol.split('.')
        bs_code = f"{market[0].lower()}.{code}"
        
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume",
            start_date='20230101', end_date='20260630',
            frequency="d", adjustflag="3"
        )
        
        symbol_count = 0
        while rs.next():
            row = rs.get_row_data()
            # date是YYYY-MM-DD格式，转YYYYMMDD
            trade_date = row[0].replace('-', '')
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO kline_daily 
                    (symbol, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (symbol, trade_date, float(row[1]), float(row[2]),
                     float(row[3]), float(row[4]), float(row[5]))
                )
                symbol_count += 1
            except:
                pass
        
        conn.commit()
        total_count += symbol_count
        print(f"  [{i+1}/{len(batch)}] {symbol}: {symbol_count} 条")
        
        # 限频sleep
        time.sleep(SLEEP_BETWEEN_STOCKS)
    
    bs.logout()
    
    # 统计
    total_daily = conn.execute("SELECT COUNT(*) FROM kline_daily").fetchone()[0]
    days = conn.execute("SELECT COUNT(DISTINCT date) FROM kline_daily").fetchone()[0]
    print(f"  ✅ 本次新增: {total_count}条, kline_daily共: {total_daily}条, {days}天")
    print(f"  ⏳ 还剩 {len(symbols_to_fetch) - batch_size} 只待下次拉取")
    
    conn.close()


def step_north():
    """北向资金补全 - akshare（慢，需要分步）"""
    print("\n[north] 补全北向资金近期数据...")
    
    try:
        import akshare as ak
    except ImportError:
        print("  ❌ akshare未安装")
        return
    
    conn = sqlite3.connect(DB_PATH)
    
    # 查已有北向数据的最新有效日期
    max_valid = conn.execute(
        "SELECT MAX(trade_date) FROM north_flow WHERE total_net IS NOT NULL AND total_net != 0"
    ).fetchone()[0]
    print(f"  当前最新有效: {max_valid}")
    
    # akshare的历史数据有效到2024-08-16
    # 先拉沪股通
    print("  拉取沪股通...")
    try:
        df = ak.stock_hsgt_hist_em(symbol="沪股通")
        valid = df[df['当日成交净买额'].notna()]
        print(f"  沪股通有效数据: {len(valid)} 条, {valid['日期'].iloc[0]}~{valid['日期'].iloc[-1]}")
        
        count = 0
        for _, row in valid.iterrows():
            date = str(row['日期'])
            net = row['当日成交净买额']  # 亿元
            net_wan = net * 10000 if net else None
            
            conn.execute(
                """UPDATE north_flow SET north_net=?, updated_at=? WHERE trade_date=?""",
                (net_wan, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), date)
            )
            count += 1
        
        conn.commit()
        print(f"  ✅ 沪股通更新: {count} 条")
    except Exception as e:
        print(f"  ❌ 沪股通错误: {e}")
    
    time.sleep(10)
    
    # 深股通
    print("  拉取深股通...")
    try:
        df = ak.stock_hsgt_hist_em(symbol="深股通")
        valid = df[df['当日成交净买额'].notna()]
        print(f"  深股通有效数据: {len(valid)} 条")
        
        count = 0
        for _, row in valid.iterrows():
            date = str(row['日期'])
            net = row['当日成交净买额']  # 亿元
            net_wan = net * 10000 if net else None
            
            conn.execute(
                """UPDATE north_flow SET sz_net=?, total_net=COALESCE(north_net,0)+?, updated_at=? WHERE trade_date=?""",
                (net_wan, net_wan, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), date)
            )
            count += 1
        
        conn.commit()
        print(f"  ✅ 深股通更新: {count} 条")
    except Exception as e:
        print(f"  ❌ 深股通错误: {e}")
    
    # 最终统计
    valid_count = conn.execute("SELECT COUNT(*) FROM north_flow WHERE total_net IS NOT NULL AND total_net != 0").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM north_flow").fetchone()[0]
    print(f"  北向资金: {valid_count}/{total} 有效")
    
    conn.close()


def verify():
    """验证数据完整性"""
    print("\n[verify] 数据完整性检查...")
    
    conn = sqlite3.connect(DB_PATH)
    
    import sys, os
    sys.path.insert(0, '/root/github/stock-quant/stock-quant/python')
    sys.path.insert(0, '/root/github/stock-quant/stock-quant/python/strategy')
    from strategy.train_lgb_enhanced import MarketFeatureEngineer
    import pandas as pd
    
    # 1. 各表数据量
    for table in ['north_flow', 'stock_sector', 'hs300_daily', 'kline_daily']:
        try:
            cnt = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {cnt} 条")
        except:
            print(f"  {table}: ❌ 未创建")
    
    # 2. 板块映射覆盖率
    others = conn.execute("SELECT COUNT(*) FROM stock_sector WHERE industry='其他' OR industry IS NULL").fetchone()[0]
    total_sectors = conn.execute("SELECT COUNT(*) FROM stock_sector").fetchone()[0]
    symbols_30m = conn.execute("SELECT COUNT(DISTINCT symbol) FROM kline_30m").fetchone()[0]
    matched = conn.execute("""
        SELECT COUNT(DISTINCT k.symbol) FROM kline_30m k 
        JOIN stock_sector s ON k.symbol = s.symbol WHERE s.industry != '其他'
    """).fetchone()[0]
    print(f"  板块映射: {matched}/{symbols_30m} 只有行业 ({matched/symbols_30m*100:.0f}%)")
    
    # 3. hs300日期匹配率
    matching = conn.execute("""
        SELECT COUNT(DISTINCT SUBSTR(date,1,10)) FROM kline_30m 
        WHERE REPLACE(SUBSTR(date,1,10),'-','') IN (SELECT trade_date FROM hs300_daily)
    """).fetchone()[0]
    total_days = conn.execute("SELECT COUNT(DISTINCT SUBSTR(date,1,10)) FROM kline_30m").fetchone()[0]
    print(f"  沪深300日期匹配: {matching}/{total_days} ({matching/total_days*100:.0f}%)")
    
    # 4. 北向资金日期匹配率
    north_match = conn.execute("""
        SELECT COUNT(DISTINCT SUBSTR(date,1,10)) FROM kline_30m 
        WHERE SUBSTR(date,1,10) IN (
            SELECT trade_date FROM north_flow WHERE total_net IS NOT NULL AND total_net != 0
        )
    """).fetchone()[0]
    print(f"  北向资金日期匹配: {north_match}/{total_days} ({north_match/total_days*100:.0f}%)")
    
    # 5. 市场特征实际值测试
    df = pd.read_sql(
        "SELECT date, open, close, high, low, volume FROM kline_30m "
        "WHERE symbol='600036.SH' AND date >= '2026-06-01' ORDER BY date",
        conn
    )
    feats = MarketFeatureEngineer.calculate_market_features(df, symbol='600036.SH')
    print(f"\n  市场特征值(最近5条):")
    for col in feats.columns:
        nonzero = (feats[col] != 0).sum()
        last5_mean = feats[col].tail(5).mean()
        print(f"    {col}: {nonzero}/{len(feats)} 非零, recent_mean={last5_mean:.6f}")
    
    conn.close()


def main():
    parser = argparse.ArgumentParser(description='BaoStock数据拉取器')
    parser.add_argument('--step', choices=['sector', 'hs300', 'daily', 'north', 'all', 'verify'],
                        default='all', help='执行步骤')
    parser.add_argument('--batch', type=int, default=20, help='daily步骤每批股票数')
    args = parser.parse_args()
    
    print("=" * 50)
    print(f"BaoStock数据拉取 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Step: {args.step}")
    print("=" * 50)
    
    if args.step == 'sector':
        step_sector()
    elif args.step == 'hs300':
        step_hs300()
    elif args.step == 'daily':
        step_daily(args.batch)
    elif args.step == 'north':
        step_north()
    elif args.step == 'verify':
        verify()
    elif args.step == 'all':
        step_sector()
        time.sleep(5)
        step_hs300()
        time.sleep(5)
        step_daily(args.batch)
        time.sleep(5)
        verify()
    
    print(f"\n✅ 完成 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == '__main__':
    main()