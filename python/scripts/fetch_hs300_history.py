#!/usr/bin/env python3
"""补全沪深300指数日线数据(2014-2022)"""
import tushare as ts, sqlite3, time

ts.set_token('7a9014b18909e8cbce5109d7175f7b21ce37354eaff2371db0da2c58')
pro = ts.pro_api()
conn = sqlite3.connect('/root/github/stock-quant/python/data/stock_data.db')

for year in range(2014, 2023):
    try:
        df = pro.index_daily(ts_code='399300.SZ', start_date=f'{year}0101', end_date=f'{year}1231')
        if df is None or len(df) == 0:
            print(f"{year}: 无数据"); continue
        new = 0
        for _, row in df.iterrows():
            td = str(row['trade_date'])
            if conn.execute("SELECT 1 FROM hs300_daily WHERE trade_date=?", (td,)).fetchone():
                continue
            conn.execute("""INSERT INTO hs300_daily VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?)""",
                (td, float(row.get('open',0) or 0), float(row.get('close',0) or 0),
                 float(row.get('high',0) or 0), float(row.get('low',0) or 0),
                 float(row.get('vol',0) or 0), float(row.get('amount',0) or 0),
                 float(row.get('pct_chg',0) or 0), 0, 0, 0, 0))
            new += 1
        conn.commit()
        print(f"{year}: {len(df)}条(新增{new})")
    except Exception as e:
        print(f"{year}错误: {e}")
    time.sleep(61)  # tushare限频1次/分钟

f = conn.execute("SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM hs300_daily").fetchone()
print(f"\n✅ 大盘数据: {f[0]}条 ({f[1]} ~ {f[2]})")
conn.close()