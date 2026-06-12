#!/usr/bin/env python3
"""
每日数据增量更新 - 由cron每小时执行

任务优先级：
1. 板块映射更新（如果"其他"类 > 20只）
2. 沪深300日线更新
3. 个股日线补充（缺失 > 600天）
4. 北向资金补全
"""

import os
import sys
import sqlite3
import time
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PYTHON_DIR = os.path.join(PROJECT_ROOT, 'python')
DATA_DIR = os.path.join(PYTHON_DIR, 'data')
DB_PATH = os.path.join(DATA_DIR, 'stock_data.db')
AGENT_DIR = os.path.join(PROJECT_ROOT, 'agent')

sys.path.insert(0, PYTHON_DIR)
sys.path.insert(0, AGENT_DIR)

TUSHARE_TOKEN = '7a9014b18909e8cbce5109d7175f7b21ce37354eaff2371db0da2c58'


def get_status(conn):
    """检查当前数据完成状态"""
    status = {}
    
    try:
        total = conn.execute("SELECT COUNT(*) FROM stock_sector").fetchone()[0]
        others = conn.execute("SELECT COUNT(*) FROM stock_sector WHERE industry='其他' OR industry IS NULL").fetchone()[0]
        status['sector'] = {'total': total, 'others': others, 'done': others < 20}
    except:
        status['sector'] = {'total': 0, 'others': 0, 'done': False}
    
    try:
        total = conn.execute("SELECT COUNT(*) FROM north_flow").fetchone()[0]
        valid = conn.execute("SELECT COUNT(*) FROM north_flow WHERE total_net IS NOT NULL AND total_net != 0").fetchone()[0]
        status['north_flow'] = {'total': total, 'valid': valid}
    except:
        status['north_flow'] = {'total': 0, 'valid': 0}
    
    try:
        total = conn.execute("SELECT COUNT(*) FROM hs300_daily").fetchone()[0]
        status['hs300'] = {'total': total, 'done': total > 500}
    except:
        status['hs300'] = {'total': 0, 'done': False}
    
    try:
        days = conn.execute("SELECT COUNT(DISTINCT date) FROM kline_daily WHERE date >= '2023-01-01'").fetchone()[0]
        status['daily'] = {'days': days, 'done': days > 600}
    except:
        status['daily'] = {'days': 0, 'done': False}
    
    return status


def step_sector_mapping(conn):
    """Step: 板块映射（Tushare stock_basic）"""
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
    others = conn.execute("SELECT COUNT(*) FROM stock_sector WHERE industry='其他'").fetchone()[0]
    print(f"板块映射更新: {count} 条, '其他'类: {others} 只")
    return True


def step_hs300_daily(conn):
    """Step: 沪深300日线（Tushare index_daily）"""
    import tushare as ts
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
    
    try:
        # 查已有最新日期
        max_date = conn.execute("SELECT MAX(trade_date) FROM hs300_daily").fetchone()[0]
        start = max_date if max_date else '20230101'
    except:
        start = '20230101'
    
    try:
        df = pro.index_daily(ts_code='000300.SH', start_date=start, end_date='20260630')
    except Exception as e:
        if '频率超限' in str(e):
            print("Tushare限频，跳过")
            return False
        print(f"index_daily错误: {e}")
        return False
    
    if len(df) == 0:
        print("沪深300数据为空")
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
    """Step: 补充个股日线数据（Tushare daily）"""
    import tushare as ts
    import pandas as pd
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
    
    all_dates = pd.date_range('2023-01-01', datetime.now(), freq='B').strftime('%Y%m%d').tolist()
    existing = set(r[0] for r in conn.execute("SELECT DISTINCT date FROM kline_daily").fetchall())
    missing = [d for d in all_dates if d not in existing]
    
    if not missing:
        print("日线数据已完整")
        return True
    
    batch = missing[:5]
    print(f"日线缺失 {len(missing)} 天, 本次拉 {len(batch)} 天")
    
    symbols_in_db = set(r[0] for r in conn.execute("SELECT DISTINCT symbol FROM kline_30m").fetchall())
    
    for date in batch:
        try:
            df = pro.daily(trade_date=date)
            if len(df) == 0:
                continue
            
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
            time.sleep(0.5)
            
        except Exception as e:
            if '频率超限' in str(e):
                print(f"  {date} 限频，停止")
                return False
            print(f"  {date} 错误: {e}")
            continue
    
    return True


def step_north_flow(conn):
    """Step: 北向资金补全（akshare）"""
    try:
        import akshare as ak
    except ImportError:
        print("akshare未安装")
        return False
    
    try:
        for channel, update_field, value_field in [
            ('沪股通', 'north_net', '当日成交净买额'),
            ('深股通', 'sz_net', '当日成交净买额'),
        ]:
            df = ak.stock_hsgt_hist_em(symbol=channel)
            valid = df[df['当日成交净买额'].notna()]
            print(f"  {channel}: {len(valid)} 条有效数据")
            
            for _, row in valid.iterrows():
                date = str(row['日期'])
                net = row['当日成交净买额']  # 亿元
                net_wan = net * 10000 if net else None
                
                # 更新对应字段
                conn.execute(
                    f"""UPDATE north_flow SET {update_field}=?, updated_at=? WHERE trade_date=?""",
                    (net_wan, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), date)
                )
            
        conn.commit()
        return True
    except Exception as e:
        print(f"akshare北向资金错误: {e}")
        return False


def main():
    print(f"\n{'='*50}")
    print(f"增量数据更新 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}")
    
    conn = sqlite3.connect(DB_PATH)
    status = get_status(conn)
    
    print("当前状态:")
    for key, val in status.items():
        print(f"  {key}: {val}")
    
    # 按优先级执行
    if not status['sector']['done']:
        print("\n--- 1. 板块映射 ---")
        if step_sector_mapping(conn):
            conn.close()
            return
    
    if not status['hs300']['done']:
        print("\n--- 2. 沪深300日线 ---")
        if step_hs300_daily(conn):
            conn.close()
            return
    
    if not status['daily']['done']:
        print("\n--- 3. 日线数据补充 ---")
        step_kline_daily(conn)
        conn.close()
        return
    
    print("\n--- 4. 北向资金补全 ---")
    step_north_flow(conn)
    
    # 最终统计
    new_status = get_status(conn)
    print("\n更新后状态:")
    for key, val in new_status.items():
        print(f"  {key}: {val}")
    
    conn.close()


if __name__ == '__main__':
    main()