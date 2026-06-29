#!/usr/bin/env python3
"""更新30分钟K线 + qlib bin数据"""
import sys, os, time, sqlite3, json
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))
from config_loader import get_db_path

DB_PATH = get_db_path()
EASTMONEY_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get"

def get_secid(symbol):
    code = symbol.split('.')[0]
    return f"0.{code}" if symbol.endswith('.SZ') else f"1.{code}" if symbol.endswith('.SH') else None

def main():
    conn = sqlite3.connect(DB_PATH)
    symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM kline_30m ORDER BY symbol").fetchall()]
    print(f"共 {len(symbols)} 只股票", flush=True)
    
    total_new = 0
    total_updated = 0
    
    for i, sym in enumerate(symbols):
        if sym.endswith('.HK'):
            continue
        
        time.sleep(0.5)  # 避免请求过快被限流
        
        # 重试机制（东方财富API偶发性断连）
        for retry in range(2):
            try:
                r = requests.get(EASTMONEY_URL, params={
                    'secid': get_secid(sym),
                    'fields1': 'f1,f2,f3',
                    'fields2': 'f51,f52,f53,f54,f55,f56,f57',
                    'klt': '30', 'fqt': '1', 'end': '20260625', 'lmt': 200,
                }, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Referer': 'https://quote.eastmoney.com/',
                }, timeout=15)
                data = r.json()
                klines = data.get('data', {}).get('klines', [])
                break
            except Exception:
                if retry == 0:
                    time.sleep(1)
                    continue
                klines = []
        
        new = 0
        for line in klines:
            parts = line.split(',')
            if len(parts) < 6:
                continue
            dt, o, c, h, l, v = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            if not conn.execute("SELECT 1 FROM kline_30m WHERE symbol=? AND date=?", (sym, dt)).fetchone():
                conn.execute(
                    "INSERT OR IGNORE INTO kline_30m (symbol,date,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?)",
                    (sym, dt, o, h, l, c, v))
                new += 1
        
        if new > 0:
            conn.commit()
            total_new += new
            total_updated += 1
            print(f"  [{i+1}/{len(symbols)}] {sym}: +{new}条", flush=True)
        
        if i % 50 == 49:
            print(f"  进度: {i+1}/{len(symbols)}", flush=True)
    
    conn.close()
    print(f"\n✅ 30min数据: {total_updated}只更新, +{total_new}条", flush=True)
    
    if total_new > 0:
        print("🔄 重新生成 qlib bin...", flush=True)
        import subprocess
        script_dir = os.path.dirname(os.path.abspath(__file__))
        convert_script = os.path.join(os.path.dirname(script_dir), 'python', 'qlib_pipeline', 'convert_data.py')
        subprocess.run([sys.executable, convert_script, '--db', DB_PATH, '--freq', '30min'], check=True)
        print("✅ qlib bin 更新完成", flush=True)

if __name__ == '__main__':
    main()