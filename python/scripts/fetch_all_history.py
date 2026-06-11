#!/usr/bin/env python3
"""补全历史数据(大盘+30分钟K线) - Mac版本用akshare

不限频、速度快，只在Mac上跑（Mac网络没有ACL限制）
预计: 372只 × 3秒 ≈ 20分钟拉完3年K线数据
"""
import akshare as ak
import sqlite3
import time
import sys
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/stock_data.db')
conn = sqlite3.connect(DB_PATH)

# ====== 1. 补大盘数据 (沪深300日线 2014-2022) ======
print("=== 补大盘数据 (2014-2022) ===")
for year in range(2014, 2023):
    try:
        df = ak.stock_zh_index_daily_em(symbol="sh000300", start_date=f'{year}0101', end_date=f'{year}1231')
        if df is None or len(df) == 0:
            print(f"{year}: 无数据"); continue
        new = 0
        for _, row in df.iterrows():
            td = str(row.get('日期', row.get('date', '')))
            if not td: continue
            pct = float(row.get('涨跌幅', 0))
            if not conn.execute("SELECT 1 FROM hs300_daily WHERE trade_date=?", (td,)).fetchone():
                conn.execute("INSERT OR IGNORE INTO hs300_daily VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (td, float(row.get('开盘',0) or 0), float(row.get('收盘',0) or 0),
                     float(row.get('最高',0) or 0), float(row.get('最低',0) or 0),
                     float(row.get('成交量',0) or 0), 0, pct, 0, 0, 0, 0))
                new += 1
        conn.commit()
        print(f"{year}: {len(df)}条(新增{new})")
        time.sleep(0.5)
    except Exception as e:
        print(f"{year}错误: {e}")
        time.sleep(2)

f = conn.execute("SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM hs300_daily").fetchone()
print(f"✅ 大盘数据: {f[0]}条 ({f[1]} ~ {f[2]})")

# ====== 2. 补30分钟K线 (2022-2024) ======
print("\n=== 补30分钟K线 (2022-2024) ===")
symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM kline_30m ORDER BY symbol").fetchall()]
print(f"需补数据股票: {len(symbols)}只")

total_new = 0
total_fail = 0

# akshare的stock_zh_a_hist_min_em每次只能拉1个月
# 每只股票 × 36个月(2022-2024) = 372×36 = 13392次
# 但不限频，每次3秒 ≈ 40小时(太慢!)

# 更好的方案: 用新浪接口拉最近5001条(不限频)
# 然后用akshare拉更早的数据

for i, sym in enumerate(symbols):
    ts_code = sym  # 000001.SZ
    ak_code = sym.replace('.SZ','').replace('.SH','')  # 000001
    
    try:
        # 先用akshare拉2022-2024的30分钟数据
        for year_start, year_end in [('2022-01-01 09:30:00', '2022-06-30 15:00:00'),
                                      ('2022-07-01 09:30:00', '2022-12-31 15:00:00'),
                                      ('2023-01-01 09:30:00', '2023-06-30 15:00:00'),
                                      ('2023-07-01 09:30:00', '2023-12-31 15:00:00'),
                                      ('2024-01-01 09:30:00', '2024-06-30 15:00:00'),
                                      ('2024-07-01 09:30:00', '2024-12-31 15:00:00')]:
            try:
                df = ak.stock_zh_a_hist_min_em(symbol=ak_code, period='30',
                                                start_date=year_start, end_date=year_end)
                if df is None or len(df) == 0: continue
                
                new = 0
                for _, row in df.iterrows():
                    td = str(row.get('时间', ''))
                    if not td: continue
                    if not conn.execute("SELECT 1 FROM kline_30m WHERE symbol=? AND date=?", (sym, td)).fetchone():
                        conn.execute("INSERT OR IGNORE INTO kline_30m VALUES (?,?,?,?,?,?,?)",
                            (sym, td, float(row.get('开盘',0) or 0), float(row.get('收盘',0) or 0),
                             float(row.get('最高',0) or 0), float(row.get('最低',0) or 0),
                             float(row.get('成交量',0) or 0)))
                        new += 1
                total_new += new
                if new > 0:
                    conn.commit()
            except Exception as e:
                total_fail += 1
                if total_fail <= 10:
                    print(f"  {sym} {year_start[:7]}错误: {e}")
            time.sleep(0.3)  # akshare不限频但0.3秒间隔防卡
        
        if (i+1) % 20 == 0:
            f = conn.execute("SELECT COUNT(*) FROM kline_30m").fetchone()[0]
            print(f"[{i+1}/{len(symbols)}] 总K线:{f} 新增:{total_new} 失败:{total_fail}")
    
    except Exception as e:
        total_fail += 1
        print(f"{sym}整体错误: {e}")

conn.commit()
f = conn.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM kline_30m").fetchone()
print(f"\n✅ K线数据: {f[0]}条 ({f[1]} ~ {f[2]})")
print(f"新增: {total_new}, 失败: {total_fail}")
conn.close()