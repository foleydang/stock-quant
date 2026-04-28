import json
import sqlite3

# 从最近的监控日志读取持仓
with open('logs/monitor_20260428_153149.json') as f:
    data = json.load(f)

positions = data.get('positions', [])

conn = sqlite3.connect('data/trading.db')
cursor = conn.cursor()

for pos in positions:
    cursor.execute('''
        INSERT OR REPLACE INTO positions (symbol, stock_name, shares, cost_price, current_price)
        VALUES (?, ?, ?, ?, ?)
    ''', (pos['symbol'], pos['stock_name'], pos['shares'], pos['cost_price'], pos['current_price']))

conn.commit()
print(f"导入 {len(positions)} 条持仓数据")

# 显示结果
cursor.execute('SELECT * FROM positions')
for row in cursor.fetchall():
    print(f"  {row[1]}: {row[2]}股 @ ¥{row[3]:.3f}")

conn.close()
