#!/usr/bin/env python3
"""北向资金数据拉取 - 每日收盘后执行"""
import sys, os, sqlite3, time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config_loader import get_db_path
DB_PATH = get_db_path()

def fetch_north_flow():
    try:
        import akshare as ak
        df = ak.stock_hsgt_fund_flow_summary_em()
        if df is None or len(df) == 0:
            print("⚠️ 北向资金数据为空")
            return
        
        conn = sqlite3.connect(DB_PATH)
        today = datetime.now().strftime('%Y-%m-%d')
        
        # 确保表存在
        conn.execute('''CREATE TABLE IF NOT EXISTS north_flow (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT NOT NULL,
            market TEXT,
            direction TEXT,
            net_amount REAL,
            buy_amount REAL,
            sell_amount REAL,
            updated_at TEXT,
            UNIQUE(trade_date, market, direction)
        )''')
        
        count = 0
        for _, row in df.iterrows():
            trade_date = str(row.get('交易日', ''))
            if not trade_date:
                continue
            
            market = str(row.get('板块', ''))
            direction = str(row.get('资金方向', ''))
            net = float(row.get('成交净买额', 0) or 0)
            buy = float(row.get('买入成交额', 0) or 0)
            sell = float(row.get('卖出成交额', 0) or 0)
            
            conn.execute(
                """INSERT OR REPLACE INTO north_flow 
                (trade_date, market, direction, net_amount, buy_amount, sell_amount, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (trade_date, market, direction, net, buy, sell, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            )
            count += 1
        
        conn.commit()
        conn.close()
        print(f"✅ 北向资金: +{count}条, 最新日期 {trade_date}")
        
    except Exception as e:
        print(f"❌ 北向资金拉取失败: {e}")

if __name__ == '__main__':
    fetch_north_flow()