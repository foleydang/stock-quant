#!/usr/bin/env python3
"""Tushare 30min 数据回填 — 按分钟频率限制, 依次拉取 4 只 ETF"""

import tushare as ts, sqlite3, time, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data', 'stock_data.db')

ts.set_token('7a9014b18909e8cbce5109d7175f7b21ce37354eaff2371db0da2c58')
pro = ts.pro_api()

ETF_CONFIG = [
    ('159792.SZ', '20210901', '20250531', '159792 2021-09~2025-05 缺口'),
    ('159792.SZ', '20250601', '20260722', '159792 2025-06~today'),
    ('513050.SH', '20210901', '20260722', '513050 全量'),
    ('513330.SH', '20210901', '20260722', '513330 全量'),
    ('159607.SZ', '20211201', '20260722', '159607 全量'),
]

conn = sqlite3.connect(DB_PATH)
total = 0

for sym, start, end, label in ETF_CONFIG:
    print(f'[{label}] 等待 65s...', flush=True)
    time.sleep(65)
    
    try:
        df = pro.stk_mins(ts_code=sym, freq='30min', start_date=start, end_date=end)
        n = len(df) if df is not None else 0
        print(f'  返回 {n} 行', flush=True)
        
        if n == 0:
            continue
            
        # 写入
        ins = 0
        for _, r in df.iterrows():
            conn.execute(
                "INSERT OR REPLACE INTO kline_30m(symbol,date,open,high,low,close,volume,updated_at) "
                "VALUES(?,?,?,?,?,?,?,datetime('now'))",
                (sym, str(r['trade_time']), float(r['open']), float(r['high']),
                 float(r['low']), float(r['close']), int(r['vol']))
            )
            ins += 1
        conn.commit()
        total += ins
        print(f'  写入 {ins} 条, 累计 {total}', flush=True)
        
    except Exception as e:
        msg = str(e)
        if '超限' in msg or '频率' in msg:
            print(f'  ⚠️ 频率限制, 跳过', flush=True)
        else:
            print(f'  ❌ {msg[:100]}', flush=True)

# 汇总
print('\n=== 最终统计 ===')
for sym in ['159792.SZ','513050.SH','513330.SH','159607.SZ']:
    r = conn.execute("SELECT MIN(date),MAX(date),COUNT(*) FROM kline_30m WHERE symbol=?",
                     (sym,)).fetchone()
    print(f'{sym}: 30min {r[2]}根 {r[0] or "-"}~{r[1] or "-"}')

conn.close()
print(f'\n总计写入 {total} 条')