#!/usr/bin/env python3
"""
北向资金数据补全 - 东方财富API
每次拉一个数据源，由cron驱动
"""

import sqlite3
import os
import requests
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'stock_data.db')

API_URL = 'https://push2his.eastmoney.com/api/qt/kamt.kline/get'
HEADERS = {'User-Agent': 'Mozilla/5.0'}

def fetch_north_flow(secid):
    """拉取北向资金数据"""
    params = {
        'fields1': 'f1,f2,f3,f4',
        'fields2': 'f51,f52',
        'klt': '101',
        'lmt': '1000',
        'secid': secid,
        'ut': 'b2884a393a59ad64002292a3e90d46a5',
    }
    r = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    data = r.json()
    if data.get('rc') != 0 or not data.get('data'):
        print(f"  API返回异常: {data.get('msg', '无数据')}")
        return {}
    
    # 数据在 data.hk2sh 或 data.hk2sz 下
    records = data['data'].get('hk2sh') or data['data'].get('hk2sz') or []
    result = {}
    for line in records:
        parts = line.split(',')
        if len(parts) >= 2:
            date = parts[0]
            net = float(parts[1]) if parts[1] else 0  # 万元
            result[date] = net
    return result


def main():
    conn = sqlite3.connect(DB_PATH)
    
    # 确保表存在
    conn.execute('''CREATE TABLE IF NOT EXISTS north_flow (
        trade_date TEXT PRIMARY KEY,
        north_net REAL,
        north_buy REAL,
        north_cum REAL,
        sz_net REAL,
        sz_buy REAL,
        sz_cum REAL,
        total_net REAL,
        total_buy REAL,
        updated_at TEXT
    )''')
    conn.commit()
    
    print("拉取北向资金数据...")
    
    # 沪股通 (secid=1.000001)
    print("  沪股通...")
    sh_data = fetch_north_flow('1.000001')
    print(f"    获取 {len(sh_data)} 条")
    
    # 深股通 (secid=1.000002)
    print("  深股通...")
    sz_data = fetch_north_flow('1.000002')
    print(f"    获取 {len(sz_data)} 条")
    
    # 合并写入
    all_dates = set(sh_data.keys()) | set(sz_data.keys())
    count = 0
    for date in sorted(all_dates):
        sh_net = sh_data.get(date, 0)  # 万元
        sz_net = sz_data.get(date, 0)  # 万元
        total = sh_net + sz_net
        
        if total == 0:
            continue
        
        conn.execute(
            """INSERT OR REPLACE INTO north_flow 
            (trade_date, north_net, sz_net, total_net, updated_at)
            VALUES (?, ?, ?, ?, ?)""",
            (date, sh_net, sz_net, total,
             datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        )
        count += 1
    
    conn.commit()
    
    # 统计
    total = conn.execute("SELECT COUNT(*) FROM north_flow").fetchone()[0]
    valid = conn.execute("SELECT COUNT(*) FROM north_flow WHERE total_net IS NOT NULL AND total_net != 0").fetchone()[0]
    max_date = conn.execute("SELECT MAX(trade_date) FROM north_flow WHERE total_net != 0").fetchone()[0]
    
    print(f"\n北向资金更新完成: {count} 条")
    print(f"数据库总计: {total} 条, 有效: {valid} 条, 最新: {max_date}")
    
    conn.close()


if __name__ == '__main__':
    main()