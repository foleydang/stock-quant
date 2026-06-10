#!/usr/bin/env python3
"""Step 1: 拉取北向资金历史数据（东方财富免费API）

数据源: 东方财富 push2his API
存储: stock_data.db 新表 north_flow
数据格式: 日期,当日净流入(万元),当日买入(万元),累计净流入(万元)
"""

import requests
import sqlite3
from datetime import datetime

DB_PATH = '/root/github/stock-quant/stock-quant/python/data/stock_data.db'


def create_table(conn):
    """创建北向资金表"""
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


def fetch_north_flow():
    """从东方财富获取北向资金历史数据"""
    url = "https://push2his.eastmoney.com/api/qt/kamt.kline/get"
    params = {
        "fields1": "f1,f3,f5",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63",
        "klt": "101", "lmt": "0",
        "ut": "7eea3ed2ced24c2974d3210a0be1e25",
    }
    
    r = requests.get(url, params=params, timeout=30)
    data = r.json()["data"]
    
    # 数据格式: 日期,当日净流入(万元),当日买入(万元),累计净流入(万元)
    hk2sh = data.get("hk2sh", [])  # 港股→沪股通
    hk2sz = data.get("hk2sz", [])  # 港股→深股通
    
    if not hk2sh or not hk2sz:
        print("北向资金数据为空")
        return
    
    conn = sqlite3.connect(DB_PATH)
    create_table(conn)
    
    # 解析沪股通数据
    sh_data = {}
    for line in hk2sh:
        parts = line.split(",")
        if len(parts) >= 4:
            date = parts[0]
            try:
                net = float(parts[1]) if parts[1] else 0  # 当日净流入(万元)
                buy = float(parts[2]) if parts[2] else 0  # 当日买入(万元)
                cum = float(parts[3]) if parts[3] else 0  # 累计净流入(万元)
                # 过滤无效数据（净流入=0且买入=0的可能是占位数据）
                if net == 0 and buy == 0:
                    continue
                sh_data[date] = {'net': net, 'buy': buy, 'cum': cum}
            except (ValueError, IndexError):
                continue
    
    # 解析深股通数据
    sz_data = {}
    for line in hk2sz:
        parts = line.split(",")
        if len(parts) >= 4:
            date = parts[0]
            try:
                net = float(parts[1]) if parts[1] else 0
                buy = float(parts[2]) if parts[2] else 0
                cum = float(parts[3]) if parts[3] else 0
                if net == 0 and buy == 0:
                    continue
                sz_data[date] = {'net': net, 'buy': buy, 'cum': cum}
            except (ValueError, IndexError):
                continue
    
    # 合并写入DB
    count = 0
    for date in sorted(set(sh_data.keys()) & set(sz_data.keys())):
        sh = sh_data[date]
        sz = sz_data[date]
        total_net = sh['net'] + sz['net']
        total_buy = sh['buy'] + sz['buy']
        
        conn.execute(
            """INSERT OR REPLACE INTO north_flow 
            (trade_date, north_net, north_buy, north_cum, sz_net, sz_buy, sz_cum,
             total_net, total_buy, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (date, sh['net'], sh['buy'], sh['cum'],
             sz['net'], sz['buy'], sz['cum'],
             total_net, total_buy,
             datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        )
        count += 1
    
    conn.commit()
    conn.close()
    
    # 统计
    dates = sorted(set(sh_data.keys()) & set(sz_data.keys()))
    print(f"✅ 北向资金数据写入完成: {count} 条")
    print(f"  日期范围: {dates[0]} ~ {dates[-1]}")
    # 最近5天数据
    recent_dates = dates[-5:]
    for d in recent_dates:
        sh = sh_data[d]
        sz = sz_data[d]
        print(f"  {d}: 沪净流入{sh['net']}万, 深净流入{sz['net']}万, 合计{sh['net']+sz['net']}万")


if __name__ == '__main__':
    fetch_north_flow()