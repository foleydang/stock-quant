#!/usr/bin/env python3
import os
"""拉取北向资金历史数据（东方财富 datacenter API）

数据源: 东方财富 RPT_MUTUAL_DEAL_HISTORY
MUTUAL_TYPE: 002=沪股通, 006=深股通 (001/003/005全None, 004是合计)
⚠️ filter参数有bug，加了MUTUAL_TYPE条件后返回004(合计)数据
   解决方案：不加filter，一次性拉全部数据再按类型分类
存储: stock_data.db 表 north_flow
"""

import requests
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'stock_data.db')

# 东方财富 datacenter API 的有效类型
# 002=沪股通, 006=深股通 (001/003/005返回全None, 004=北向合计)
VALID_TYPES = {'002', '006'}
TYPE_NAMES = {'002': '沪股通', '006': '深股通', '004': '北向合计'}


def create_table(conn):
    """创建北向资金表"""
    conn.execute('''CREATE TABLE IF NOT EXISTS north_flow (
        trade_date TEXT PRIMARY KEY,
        north_net REAL,
        north_buy REAL,
        north_sell REAL,
        sz_net REAL,
        sz_buy REAL,
        sz_sell REAL,
        total_net REAL,
        total_buy REAL,
        total_sell REAL,
        updated_at TEXT
    )''')
    conn.commit()


def fetch_north_flow():
    """从东方财富获取北向资金历史数据

    不加MUTUAL_TYPE filter，拉取全部数据后按002(沪股通)/006(深股通)分类
    """

    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'http://data.eastmoney.com/hsgt/'
    }

    # 每个日期存不同类型的数据
    all_records = {}
    page = 1
    page_size = 500
    total_count = 0

    print("正在获取北向资金历史数据（不带filter，一次拉全部）...")

    while True:
        params = {
            "reportName": "RPT_MUTUAL_DEAL_HISTORY",
            "columns": "TRADE_DATE,NET_DEAL_AMT,BUY_AMT,SELL_AMT,MUTUAL_TYPE",
            "source": "WEB",
            "client": "WEB",
            # 不加filter！东方财富filter有bug
            "pageNumber": page,
            "pageSize": page_size,
            "sortTypes": "-1",
            "sortColumns": "TRADE_DATE"
        }

        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)
            if response.status_code != 200:
                print(f"  请求失败: HTTP {response.status_code}")
                break

            data = response.json()
            if not data.get('result') or not data['result'].get('data'):
                print(f"  第{page}页: 无数据，结束")
                break

            records = data['result']['data']
            page_valid = 0

            for r in records:
                trade_date = str(r.get('TRADE_DATE', '')).split(' ')[0]
                mutual_type = str(r.get('MUTUAL_TYPE', ''))
                net = r.get('NET_DEAL_AMT')
                buy = r.get('BUY_AMT')
                sell = r.get('SELL_AMT')

                if not trade_date or mutual_type not in VALID_TYPES:
                    continue

                # 跳过全空数据
                if net is None and buy is None and sell is None:
                    continue

                if trade_date not in all_records:
                    all_records[trade_date] = {}

                all_records[trade_date][mutual_type] = {
                    'net': net or 0,
                    'buy': buy or 0,
                    'sell': sell or 0,
                }
                page_valid += 1

            total_count += page_valid
            print(f"  第{page}页: {len(records)}条原始, {page_valid}条有效(002/006)")

            if len(records) < page_size:
                break

            # 拉尽可能多的历史数据（12页×500=6000条原始）
            if page >= 12:
                break

            page += 1

        except Exception as e:
            print(f"  异常: {e}")
            break

    # 写入数据库
    if not all_records:
        print("✗ 未获取到数据")
        return

    conn = sqlite3.connect(DB_PATH)
    create_table(conn)
    conn.execute("DELETE FROM north_flow")

    count = 0
    for date in sorted(all_records.keys()):
        d = all_records[date]

        sh = d.get('002', {'net': 0, 'buy': 0, 'sell': 0})
        sz = d.get('006', {'net': 0, 'buy': 0, 'sell': 0})

        total_net = sh['net'] + sz['net']
        total_buy = sh['buy'] + sz['buy']
        total_sell = sh['sell'] + sz['sell']

        conn.execute(
            """INSERT OR REPLACE INTO north_flow
            (trade_date, north_net, north_buy, north_sell,
             sz_net, sz_buy, sz_sell,
             total_net, total_buy, total_sell, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (date, sh['net'], sh['buy'], sh['sell'],
             sz['net'], sz['buy'], sz['sell'],
             total_net, total_buy, total_sell,
             datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        )
        count += 1

    conn.commit()
    conn.close()

    # 统计输出
    dates = sorted(all_records.keys())
    print(f"\n✅ 北向资金数据写入完成: {count} 条")
    print(f"  日期范围: {dates[0]} ~ {dates[-1]}")

    # 最近5天数据
    recent = dates[-5:]
    print(f"\n最近5个交易日:")
    for d in recent:
        rec = all_records[d]
        sh = rec.get('002', {'net': 0, 'buy': 0, 'sell': 0})
        sz = rec.get('006', {'net': 0, 'buy': 0, 'sell': 0})
        total = sh['net'] + sz['net']
        print(f"  {d}: 沪净流入{sh['net']:,.2f}万, 深净流入{sz['net']:,.2f}万, 合计{total:,.2f}万")


if __name__ == '__main__':
    fetch_north_flow()