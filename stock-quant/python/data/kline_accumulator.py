"""分钟K线累积器 - 用实时行情构建自己的分钟K线数据库"""

import os
import sys
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import time

BASE_DIR = "/root/github/stock-quant/stock-quant/python"
DB_PATH = f"{BASE_DIR}/data/stock_data.db"

sys.path.insert(0, BASE_DIR)
from data.data_handler import DataHandler


class KlineAccumulator:
    """Accumulate real K-line data"""
    
    def __init__(self):
        self.db_path = DB_PATH
        self.data_handler = DataHandler()
    
    def accumulate_realtime(self, symbols):
        """Accumulate real 30min K-line data
        
        For A-shares: use Tushare stk_mins API (real minute bars)
        For HK/ETF: use Tencent realtime API (best available)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        count = 0
        
        for symbol in symbols:
            try:
                if symbol.endswith(".SZ") or symbol.endswith(".SH"):
                    # A-shares: real 30min bars from Tushare
                    df = self.data_handler.fetch_real_30min_kline(symbol, count=20)
                    if df is not None and len(df) > 0:

                        for _, row in df.iterrows():
                            cursor.execute(
                                "INSERT OR REPLACE INTO kline_30m (symbol, date, open, high, low, close, volume, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                (symbol, row["date"].strftime("%Y-%m-%d %H:%M:%S"),
                                 float(row["open"]), float(row["high"]), float(row["low"]),
                                 float(row["close"]), float(row["volume"]),
                                 datetime.now().isoformat())
                            )
                        count += 1
                        print(f"{symbol}: {len(df)} real 30min bars from Tushare")
                    else:
                        self._accumulate_from_realtime(symbol, cursor)
                        count += 1
                else:
                    # HK stocks: realtime approximation
                    self._accumulate_from_realtime(symbol, cursor)
                    count += 1
            except Exception as e:
                print(f"Error {symbol}: {e}")
        
        conn.commit()
        conn.close()
        print(f"Accumulated {count}/{len(symbols)} symbols")
        return count
    
    def _accumulate_from_realtime(self, symbol, cursor):
        """Fallback: realtime for HK stocks"""
        prices = self.data_handler.get_realtime_prices([symbol])
        if not prices or symbol not in prices:
            print(f"{symbol}: realtime fetch failed")
            return
        
        data = prices[symbol]
        now = datetime.now()
        kline_time = now.replace(minute=now.minute // 30 * 30, second=0, microsecond=0)
        
        open_price = data.get("open", data["price"])
        high_price = data.get("high", data["price"])
        low_price = data.get("low", data["price"])
        close_price = data["price"]
        volume = data.get("volume", 0)
        
        cursor.execute(
            "SELECT id FROM kline_30m WHERE symbol=? AND date=?",
            (symbol, kline_time.strftime("%Y-%m-%d %H:%M:%S"))
        )
        existing = cursor.fetchone()
        
        if existing:
            cursor.execute(
                "UPDATE kline_30m SET high=?, low=?, close=?, volume=?, updated_at=? WHERE symbol=? AND date=?",
                (high_price, low_price, close_price, volume, now.isoformat(),
                 symbol, kline_time.strftime("%Y-%m-%d %H:%M:%S"))
            )
        else:
            cursor.execute(
                "INSERT OR REPLACE INTO kline_30m (symbol, date, open, high, low, close, volume, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (symbol, kline_time.strftime("%Y-%m-%d %H:%M:%S"),
                 open_price, high_price, low_price, close_price, volume, now.isoformat())
            )
        print(f"{symbol}: realtime approx @ {kline_time.strftime("%H:%M")}")
    
    def get_kline_stats(self, symbol):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*), MIN(date), MAX(date) FROM kline_30m WHERE symbol=?",
            (symbol,)
        )
        row = cursor.fetchone()
        conn.close()
        if row and row[0]:
            return {"count": row[0], "min_date": row[1], "max_date": row[2]}
        return None

if __name__ == "__main__":
    acc = KlineAccumulator()
    symbols = ["300124.SZ", "600048.SH", "3690.HK", "300015.SZ", "159792.SZ"]
    acc.accumulate_realtime(symbols)
