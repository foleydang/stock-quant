#!/usr/bin/env python3
"""
从东方财富 API 获取股票名称并更新数据库
"""
import sqlite3
import requests
import time

db_path = 'data/stock_data.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 获取所有股票
cursor.execute("SELECT symbol FROM stock_info WHERE name IS NULL OR name = '' OR length(name) < 4")
symbols = [row[0] for row in cursor.fetchall()]

print(f"找到 {len(symbols)} 只股票需要更新名称")

updated = 0
for sym in symbols:
    code = sym.split('.')[0]
    suffix = sym.split('.')[1].lower()

    # 东方财富个股行情 API
    url = f"http://push2.eastmoney.com/api/qt/stock/get?fltt=2&fields=f58&secid={suffix.replace('sh','1').replace('sz','0')}.{code}"

    try:
        resp = requests.get(url, timeout=5)
        data = resp.json()

        if data.get('data') and data['data'].get('f58'):
            name = data['data']['f58']
            cursor.execute(
                "UPDATE stock_info SET name = ? WHERE symbol = ?",
                (name, sym)
            )
            if cursor.rowcount > 0:
                updated += 1
                print(f"  {sym}: {name}")

        time.sleep(0.1)

    except Exception as e:
        print(f"获取 {sym} 失败：{e}")

conn.commit()
print(f"\n完成！更新了 {updated} 只股票名称")

# 验证
cursor.execute("SELECT symbol, name FROM stock_info WHERE length(name) < 4 LIMIT 20")
print("\n剩余短名称股票:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]}")

conn.close()
