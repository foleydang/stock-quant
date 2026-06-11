#!/usr/bin/env python3
"""补全30分钟K线历史数据(2014-2024)

tushare限频1次/分钟(stk_mins接口), 每只股票按月拉取
372只 × (10年×12月) = 44640次调用 ≈ 746小时(太慢!)

优化: 按年拉取(每只每年1次), 372只 × 10年 = 3720次 ≈ 62小时
再优化: tushare pro_bar可以一次性拉1年的分钟线
"""
import tushare as ts, sqlite3, time, sys, os
import pandas as pd
from datetime import datetime

TUSHARE_TOKEN = '7a9014b18909e8cbce5109d7175f7b21ce37354eaff2371db0da2c58'
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/stock_data.db')

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()
conn = sqlite3.connect(DB_PATH)

# 获取当前已有的股票列表
symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM kline_30m ORDER BY symbol").fetchall()]
print(f"需补数据股票: {len(symbols)}只")

# 检查每只股票当前数据范围
for sym in symbols[:5]:
    r = conn.execute("SELECT MIN(date), MAX(date), COUNT(*) FROM kline_30m WHERE symbol=?", (sym,)).fetchone()
    print(f"  {sym}: {r[2]}条 ({r[0]} ~ {r[1]})")

# 补数据: 2014-2024 (2025-2026已有)
total_new = 0
total_fail = 0
years = range(2014, 2025)  # 2025-2026已有

for i, sym in enumerate(symbols):
    for year in years:
        try:
            # tushare pro_bar可以拉1年的分钟线
            df = ts.pro_bar(ts_code=sym, freq='30min', 
                           start_date=f'{year}0101', end_date=f'{year}1231')
            if df is None or len(df) == 0:
                continue
            
            new = 0
            for _, row in df.iterrows():
                trade_time = str(row.get('trade_time', ''))
                if not trade_time: continue
                # 格式: 2022-02-28 15:00:00
                if not conn.execute("SELECT 1 FROM kline_30m WHERE symbol=? AND date=?", (sym, trade_time)).fetchone():
                    conn.execute("""INSERT OR IGNORE INTO kline_30m VALUES (?,?,?,?,?,?,?)""",
                        (sym, trade_time, float(row.get('open',0) or 0), 
                         float(row.get('close',0) or 0), float(row.get('high',0) or 0),
                         float(row.get('low',0) or 0), float(row.get('vol',0) or 0)))
                    new += 1
            
            conn.commit()
            total_new += new
            if i % 50 == 0 or new > 0:
                print(f"[{i+1}/{len(symbols)}] {sym} {year}: {len(df)}条(新增{new}) 总新增{total_new}")
            
        except Exception as e:
            total_fail += 1
            if total_fail <= 5:
                print(f"[{i+1}/{len(symbols)}] {sym} {year}错误: {e}")
        
        # tushare限频: 1次/分钟
        time.sleep(61)
    
    # 进度汇报
    if (i+1) % 20 == 0:
        f = conn.execute("SELECT COUNT(*) FROM kline_30m").fetchone()[0]
        print(f"\n=== 进度 {i+1}/{len(symbols)} ({(i+1)/len(symbols)*100:.1f}%) ===")
        print(f"总K线: {f}条, 新增: {total_new}, 失败: {total_fail}")

conn.commit()
f = conn.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM kline_30m").fetchone()
print(f"\n✅ K线数据: {f[0]}条 ({f[1]} ~ {f[2]})")
print(f"新增: {total_new}条, 失败: {total_fail}次")
conn.close()