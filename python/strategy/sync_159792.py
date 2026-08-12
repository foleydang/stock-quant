#!/usr/bin/env python3
"""
为 159792 (港股通互联网ETF) 拉取历史数据:
1. 30分钟K线 - 新浪API (历史全量)
2. 日线K线 - Tushare (历史全量)
"""

import sys, os, time, sqlite3, requests, json
from datetime import datetime

# 目标数据库
DB_PATH = '/root/github/stock-quant/python/data/stock_data.db'
SYMBOL = '159792.SZ'
SINA_CODE = 'sz159792'

def main():
    conn = sqlite3.connect(DB_PATH)
    
    # ========== 1. 拉取30分钟K线 ==========
    print("=" * 60)
    print("📡 拉取 30分钟K线 (新浪API)")
    print("=" * 60)
    
    # 先查已有数据
    existing = conn.execute(
        "SELECT date FROM kline_30m WHERE symbol=? ORDER BY date", (SYMBOL,)
    ).fetchall()
    existing_dates = set(r[0] for r in existing)
    print(f"已有 {len(existing_dates)} 条 30分钟K线")
    
    # 从新浪API拉取历史数据 (最多800根)
    url = "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"
    params = {"symbol": SINA_CODE, "scale": "30", "datalen": 800}
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        print(f"❌ API请求失败: {r.status_code}")
        conn.close()
        return
    
    data = json.loads(r.text)
    if not isinstance(data, list) or len(data) == 0:
        print("❌ 无数据返回")
        conn.close()
        return
    
    print(f"API返回 {len(data)} 条 (从 {data[0]['day']} 到 {data[-1]['day']})")
    
    new_count = 0
    for row in data:
        trade_time = row.get('day', '')
        if trade_time and len(trade_time) == 16:
            trade_time += ':00'
        if not trade_time:
            continue
        
        if trade_time in existing_dates:
            continue
        
        conn.execute(
            "INSERT INTO kline_30m (symbol, date, open, close, high, low, volume) VALUES (?,?,?,?,?,?,?)",
            (SYMBOL, trade_time,
             float(row.get('open', 0) or 0),
             float(row.get('close', 0) or 0),
             float(row.get('high', 0) or 0),
             float(row.get('low', 0) or 0),
             float(row.get('volume', 0) or 0))
        )
        new_count += 1
    
    conn.commit()
    print(f"✅ 30分钟K线: 新增 {new_count} 条, 总计 {len(existing_dates) + new_count} 条")
    
    # ========== 2. 拉取日线K线 (Tushare) ==========
    print("\n" + "=" * 60)
    print("📡 拉取日线K线 (Tushare)")
    print("=" * 60)
    
    existing_daily = conn.execute(
        "SELECT date FROM kline_daily WHERE symbol=? ORDER BY date", (SYMBOL,)
    ).fetchall()
    existing_daily_dates = set(r[0] for r in existing_daily)
    print(f"已有 {len(existing_daily_dates)} 条日线")
    
    try:
        import tushare as ts
        TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN', '')
        if not TUSHARE_TOKEN:
            # 尝试从 .env 读取
            env_path = '/root/github/stock-quant/.env'
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        if line.startswith('TUSHARE_TOKEN='):
                            TUSHARE_TOKEN = line.strip().split('=', 1)[1].strip('"\'')
                            break
        
        if TUSHARE_TOKEN:
            ts.set_token(TUSHARE_TOKEN)
            pro = ts.pro_api()
            
            # 拉取最近2年的日线数据
            df = pro.daily(ts_code=SYMBOL, start_date='20240101', end_date=datetime.now().strftime('%Y%m%d'))
            if df is not None and len(df) > 0:
                print(f"Tushare返回 {len(df)} 条日线")
                daily_new = 0
                for _, row in df.iterrows():
                    td = row['trade_date']
                    date_fmt = f"{td[:4]}-{td[4:6]}-{td[6:8]}"
                    if date_fmt in existing_daily_dates:
                        continue
                    conn.execute(
                        "INSERT INTO kline_daily (symbol, date, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
                        (SYMBOL, date_fmt, row['open'], row['high'], row['low'], row['close'], row['vol'] * 100)
                    )
                    daily_new += 1
                conn.commit()
                print(f"✅ 日线K线: 新增 {daily_new} 条")
            else:
                print("⚠️ Tushare返回空数据")
        else:
            print("⚠️ 未找到 TUSHARE_TOKEN")
    except Exception as e:
        print(f"⚠️ Tushare拉取失败: {e}")
    
    # ========== 3. 验证数据 ==========
    print("\n" + "=" * 60)
    print("📊 数据验证")
    print("=" * 60)
    
    cnt_30m = conn.execute("SELECT COUNT(*) FROM kline_30m WHERE symbol=?", (SYMBOL,)).fetchone()[0]
    cnt_daily = conn.execute("SELECT COUNT(*) FROM kline_daily WHERE symbol=?", (SYMBOL,)).fetchone()[0]
    
    print(f"  kline_30m: {cnt_30m} 条")
    print(f"  kline_daily: {cnt_daily} 条")
    
    if cnt_30m > 0:
        latest = conn.execute(
            "SELECT date, open, close, high, low, volume FROM kline_30m WHERE symbol=? ORDER BY date DESC LIMIT 3",
            (SYMBOL,)
        ).fetchall()
        print(f"  30min 最新3条:")
        for row in latest:
            print(f"    {row[0]} | O:{row[1]} H:{row[2]:.4f} L:{row[3]:.4f} C:{row[4]:.4f} V:{row[5]:.0f}")
    
    if cnt_daily > 0:
        latest = conn.execute(
            "SELECT date, open, close, volume FROM kline_daily WHERE symbol=? ORDER BY date DESC LIMIT 3",
            (SYMBOL,)
        ).fetchall()
        print(f"  日线 最新3条:")
        for row in latest:
            print(f"    {row[0]} | O:{row[1]:.4f} C:{row[2]:.4f} V:{row[3]:.0f}")
    
    conn.close()
    print("\n✅ 数据同步完成!")


if __name__ == '__main__':
    main()