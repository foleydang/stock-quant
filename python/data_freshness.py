#!/usr/bin/env python3
"""数据新鲜度监控 - 检查数据是否过期"""

import os
import sys
import sqlite3
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import get_db_path

DB_PATH = get_db_path()

def update_data_freshness(symbol: str, source: str, count: int):
    """更新数据新鲜度记录"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    now = datetime.now().isoformat()
    cursor.execute(
        "INSERT OR REPLACE INTO data_freshness (symbol, source, last_update, data_count, is_valid) VALUES (?, ?, ?, ?, 1)",
        (symbol, source, now, count)
    )
    conn.commit()
    conn.close()

def check_data_freshness(symbol: str, max_age_minutes: int = 60) -> bool:
    """检查数据是否新鲜"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT last_update FROM data_freshness WHERE symbol=?", (symbol,))
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        return False
    
    try:
        last_update = datetime.fromisoformat(row[0])
        age = datetime.now() - last_update
        return age < timedelta(minutes=max_age_minutes)
    except:
        return False

def get_all_freshness_status():
    """获取所有数据新鲜度状态"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT symbol, source, last_update, data_count, is_valid FROM data_freshness")
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        symbol, source, last_update, count, valid = row
        try:
            update_time = datetime.fromisoformat(last_update)
            age_minutes = (datetime.now() - update_time).total_seconds() / 60
        except:
            age_minutes = -1
        
        results.append({
            'symbol': symbol,
            'source': source,
            'last_update': last_update,
            'age_minutes': age_minutes,
            'data_count': count,
            'is_valid': valid,
            'is_fresh': age_minutes < 60 and age_minutes >= 0
        })
    
    return results

if __name__ == '__main__':
    # 测试
    status = get_all_freshness_status()
    for s in status[:5]:
        print(f"{s['symbol']}: {s['age_minutes']:.0f}分钟前更新, {s['data_count']}条数据")