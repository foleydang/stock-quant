#!/usr/bin/env python3
"""
增量数据拉取器 - 每小时执行一次，逐步补充缺失数据

策略：
1. Tushare限频1次/小时，每次拉一个指标的数据
2. 东方财富限频严重，仅作为备选
3. akshare可用但慢，每次只拉一个品种

由cron每小时执行，逐步补全：
- 第1小时: Tushare stock_basic (板块映射)
- 第2小时: Tushare index_daily (沪深300日线)
- 第3小时+: Tushare daily (按日期拉个股日线，补3年数据)

每次执行自动判断还需要什么数据，跳过已完成的步骤。
"""

import sqlite3
import sys
import os
import time
import requests
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'stock_data.db')
TUSHARE_TOKEN = '7a9014b18909e8cbce5109d7175f7b21ce37354eaff2371db0da2c58'


def get_status(conn):
    """检查当前数据完成状态"""
    status = {}
    
    # 板块映射
    try:
        total = conn.execute("SELECT COUNT(*) FROM stock_sector").fetchone()[0]
        others = conn.execute("SELECT COUNT(*) FROM stock_sector WHERE industry='其他'").fetchone()[0]
        status['sector'] = {'total': total, 'others': others, 'done': others < 20}
    except:
        status['sector'] = {'total': 0, 'others': 0, 'done': False}
    
    # 北向资金
    try:
        total = conn.execute("SELECT COUNT(*) FROM north_flow").fetchone()[0]
        valid = conn.execute("SELECT COUNT(*) FROM north_flow WHERE total_net IS NOT NULL AND total_net != 0").fetchone()[0]
        max_date = conn.execute("SELECT MAX(trade_date) FROM north_flow WHERE total_net IS NOT NULL AND total_net != 0").fetchone()[0]
        status['north_flow'] = {'total': total, 'valid': valid, 'max_date': max_date}
    except:
        status['north_flow'] = {'total': 0, 'valid': 0, 'max_date': None}
    
    # 沪深300日线
    try:
        total = conn.execute("SELECT COUNT(*) FROM hs300_daily").fetchone()[0]
        status['hs300'] = {'total': total, 'done': total > 500}
    except:
        status['hs300'] = {'total': 0, 'done': False}
    
    # kline_daily覆盖天数
    try:
        days = conn.execute("SELECT COUNT(DISTINCT date) FROM kline_daily WHERE date >= '2023-01-01'").fetchone()[0]
        status['daily_coverage'] = {'days': days, 'done': days > 600}
    except:
        status['daily_coverage'] = {'days': 0, 'done': False}
    
    return status


def step_sector_mapping(conn):
    """Step: 拉取板块映射（Tushare stock_basic）"""
    import tushare as ts
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
    
    try:
        df = pro.stock_basic(exchange='', list_status='L',
            fields='ts_code,symbol,name,industry,market,list_date')
    except Exception as e:
        if '频率超限' in str(e):
            print("Tushare限频，跳过")
            return False
        print(f"stock_basic错误: {e}")
        return False
    
    count = 0
    for _, row in df.iterrows():
        symbol = row['ts_code']
        name = row['name']
        industry = row.get('industry', '其他') or '其他'
        conn.execute(
            """INSERT OR REPLACE INTO stock_sector 
            (symbol, name, industry, sector_code, updated_at)
            VALUES (?, ?, ?, ?, ?)""",
            (symbol, name, industry, industry,
             datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        )
        count += 1
    
    conn.commit()
    print(f"板块映射更新: {count} 条")
    return True


def step_hs300_daily(conn):
    """Step: 拉沪深300日线（Tushare index_daily）"""
    import tushare as ts
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
    
    # 先看已有数据到哪里
    try:
        max_date = conn.execute("SELECT MAX(trade_date) FROM hs300_daily").fetchone()[0]
        start = max_date
    except:
        start = '20230101'
    
    try:
        df = pro.index_daily(ts_code='000300.SH', start_date=start, end_date='20260610')
    except Exception as e:
        if '频率超限' in str(e):
            print("Tushare限频，跳过")
            return False
        print(f"index_daily错误: {e}")
        return False
    
    if len(df) == 0:
        print("沪深300数据为空（可能需要更高权限）")
        return False
    
    count = 0
    for _, row in df.iterrows():
        conn.execute(
            """INSERT OR REPLACE INTO hs300_daily 
            (trade_date, open, close, high, low, volume, amount, pct_chg)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (row['trade_date'], row.get('open'), row.get('close'),
             row.get('high'), row.get('low'), row.get('vol'),
             row.get('amount'), row.get('pct_chg'))
        )
        count += 1
    
    conn.commit()
    print(f"沪深300日线更新: {count} 条")
    return True


def step_kline_daily(conn):
    """Step: 补充个股日线数据（Tushare daily，每天拉5个日期的数据）"""
    import tushare as ts
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
    
    # 找缺失的日期（从2023-01-01开始）
    import pandas as pd
    all_dates = pd.date_range('2023-01-01', '2026-06-10', freq='B').strftime('%Y%m%d').tolist()
    
    # 已有日期
    existing = set(r[0] for r in conn.execute("SELECT DISTINCT date FROM kline_daily").fetchall())
    
    # 缺失日期
    missing = [d for d in all_dates if d not in existing]
    if not missing:
        print("日线数据已完整")
        return True
    
    # 每次拉5天（避免限频）
    batch = missing[:5]
    print(f"日线数据缺失 {len(missing)} 天，本次拉 {len(batch)} 天")
    
    for date in batch:
        try:
            df = pro.daily(trade_date=date)
            if len(df) == 0:
                continue
            
            # 只写沪深300成分股
            symbols_in_db = set(r[0] for r in conn.execute("SELECT DISTINCT symbol FROM kline_30m").fetchall())
            df_filtered = df[df['ts_code'].isin(symbols_in_db)]
            
            for _, row in df_filtered.iterrows():
                conn.execute(
                    """INSERT OR IGNORE INTO kline_daily 
                    (symbol, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (row['ts_code'], row['trade_date'], row['open'],
                     row['high'], row['low'], row['close'], row['vol'])
                )
            
            conn.commit()
            print(f"  {date}: {len(df_filtered)} 条")
            
            # Tushare daily按日期查无频次限制（免费版200次/分钟）
            time.sleep(0.5)
            
        except Exception as e:
            if '频率超限' in str(e):
                print(f"  {date} 限频，停止")
                return False
            print(f"  {date} 错误: {e}")
            continue
    
    return True


def step_north_flow(conn):
    """Step: 用akshare补全北向资金近期数据"""
    try:
        import akshare as ak
    except ImportError:
        print("akshare未安装")
        return False
    
    # 查看已有北向数据的最新有效日期
    try:
        max_valid = conn.execute(
            "SELECT MAX(trade_date) FROM north_flow WHERE total_net IS NOT NULL AND total_net != 0"
        ).fetchone()[0]
    except:
        max_valid = None
    
    print(f"北向资金最新有效日期: {max_valid}")
    
    # 如果已有数据覆盖到近期，跳过
    if max_valid and max_valid >= '2025-12-31':
        print("北向数据已足够，跳过")
        return True
    
    try:
        # 拉沪股通+深股通的历史净流入
        for channel in ['沪股通', '深股通']:
            df = ak.stock_hsgt_hist_em(symbol=channel)
            valid = df[df['当日成交净买额'].notna()]
            print(f"  {channel}: {len(valid)} 条有效数据")
            
            for _, row in valid.iterrows():
                date = str(row['日期'])
                net = row['当日成交净买额']  # 亿元
                
                # 更新对应字段
                if channel == '沪股通':
                    net_wan = net * 10000 if net else None
                    conn.execute(
                        """UPDATE north_flow SET north_net=?, updated_at=? WHERE trade_date=?""",
                        (net_wan, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), date)
                    )
                elif channel == '深股通':
                    net_wan = net * 10000 if net else None
                    conn.execute(
                        """UPDATE north_flow SET sz_net=?, total_net=north_net+?, updated_at=? WHERE trade_date=?""",
                        (net_wan, net_wan, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), date)
                    )
        
        conn.commit()
        return True
    except Exception as e:
        print(f"akshare北向资金错误: {e}")
        return False


def main():
    conn = sqlite3.connect(DB_PATH)
    status = get_status(conn)
    
    print("当前数据状态:")
    for key, val in status.items():
        print(f"  {key}: {val}")
    
    # 按优先级执行缺失的步骤（每次只做一个Tushare步骤）
    tushare_done = False
    
    # 1. 板块映射（如果"其他"类太多）
    if not status['sector']['done']:
        print("\n--- 执行: 板块映射 ---")
        ok = step_sector_mapping(conn)
        if ok:
            return  # Tushare成功，本次结束
        # 限频失败，继续尝试其他非Tushare步骤
    
    # 2. 沪深300日线
    if not status['hs300']['done']:
        print("\n--- 执行: 沪深300日线 ---")
        ok = step_hs300_daily(conn)
        if ok:
            return
    
    # 3. 个股日线数据（也是Tushare，如果前面Tushare限频了，这里也大概率限频，但可以试试）
    if not status['daily_coverage']['done']:
        print("\n--- 执行: 补充日线数据 ---")
        ok = step_kline_daily(conn)
        if ok:
            return
    
    # 4. 北向资金补全（用akshare，不占Tushare额度）
    if not status['north_flow'].get('max_date') or status['north_flow']['max_date'] < '2025-12-31':
        print("\n--- 执行: 北向资金补全 ---")
        step_north_flow(conn)
        return
    
    print("\n所有数据步骤已完成！")
    
    # 最终统计
    new_status = get_status(conn)
    print("更新后状态:")
    for key, val in new_status.items():
        print(f"  {key}: {val}")
    
    conn.close()


if __name__ == '__main__':
    main()