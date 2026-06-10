#!/usr/bin/env python3
"""
分钟K线累积器 - 用实时行情构建自己的分钟K线数据库
"""

import os
import sys
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = "/root/github/stock-quant/python"
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
                    # HK stocks / ETF: realtime approximation
                    self._accumulate_from_realtime(symbol, cursor)
                    count += 1
            except Exception as e:
                print(f"Error {symbol}: {e}")
        
        conn.commit()
        conn.close()
        print(f"Accumulated {count}/{len(symbols)} symbols")
        return count
    
    def _accumulate_from_realtime(self, symbol, cursor):
        """Fallback: realtime for HK stocks/ETF
        
        港股和ETF没有真正的30分钟K线API，用实时价格模拟
        每个时间窗口记录一个K线点：
        - open = 该时间窗口的第一次价格（或用close近似）
        - close = 当前实时价格
        - high/low 用当日累计值（同一天多个时间窗口的high/low合并)
        - 注意: 0值异常处理,避免脏数据污染模型
        """
        prices = self.data_handler.get_realtime_prices([symbol])
        if not prices or symbol not in prices:
            print(f"{symbol}: realtime fetch failed")
            return
        
        data = prices[symbol]
        now = datetime.now()
        kline_time = now.replace(minute=now.minute // 30 * 30, second=0, microsecond=0)
        
        close_price = data["price"]
        # 港股/ETF盘中open/high/low可能为0（盘后或数据不完整），用close填充
        raw_open = data.get("open", 0)
        raw_high = data.get("high", 0)
        raw_low = data.get("low", 0)
        
        # 0值安全检查
        open_price = raw_open if raw_open > 0 else close_price
        high_price = raw_high if raw_high > 0 else close_price
        low_price = raw_low if raw_low > 0 else close_price
        # 防止成交量混入价格字段（港股API偶尔会把volume放到low字段）
        if low_price > high_price * 2:
            low_price = close_price
        # 确保 OHLC 逻辑正确: high >= close >= low
        high_price = max(high_price, close_price)
        low_price = min(low_price, close_price)
        
        volume = data.get("volume", 0)
        
        cursor.execute(
            "SELECT id, open, high, low, close FROM kline_30m WHERE symbol=? AND date=?",
            (symbol, kline_time.strftime("%Y-%m-%d %H:%M:%S"))
        )
        existing = cursor.fetchone()
        
        if existing:
            # 累积更新: 保留有效open, high取max, low取min, 更新close
            orig_id, orig_open, orig_high, orig_low, orig_close = existing
            
            # 保留有效的open（第一次写入的）
            new_open = orig_open if orig_open > 0 else open_price
            # high取历史最大
            new_high = max(orig_high, high_price) if orig_high > 0 else high_price
            # low取历史最小（排除异常值）
            new_low = min(orig_low, low_price) if orig_low > 0 and orig_low < orig_high else low_price
            # 更新close为最新价格
            new_close = close_price
            
            
            cursor.execute(
                "UPDATE kline_30m SET open=?, high=?, low=?, close=?, volume=?, updated_at=? WHERE symbol=? AND date=?",
                (new_open, new_high, new_low, close_price, volume, now.isoformat(),
                 symbol, kline_time.strftime("%Y-%m-%d %H:%M:%S"))
            )
        else:
            cursor.execute(
                "INSERT OR REPLACE INTO kline_30m (symbol, date, open, high, low, close, volume, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (symbol, kline_time.strftime("%Y-%m-%d %H:%M:%S"),
                 open_price, high_price, low_price, close_price, volume, now.isoformat())
            )
        print(f"{symbol}: realtime approx @ {kline_time.strftime('%H:%M')} o={open_price:.2f} h={high_price:.2f} l={low_price:.2f} c={close_price:.2f}")
    
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