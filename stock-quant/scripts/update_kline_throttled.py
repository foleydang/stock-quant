#!/usr/bin/env python3
"""
限频K线更新脚本 - 每分钟1只股票，避免Tushare API限频

使用方式：
1. 全量更新（首次/补数据）：python3 update_kline_throttled.py --full
2. 增量更新（日常）：python3 update_kline_throttled.py --incremental
3. 指定股票：python3 update_kline_throttled.py --symbols 600519.SH,000858.SZ

设计：
- 每分钟只调1次 Tushare API（限频1次/分钟）
- 优先更新持仓股 > 自选股 > 其余
- 进度日志写入文件，方便追踪
"""

import argparse
import sqlite3
import time
import logging
import sys
import os
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(PROJECT_ROOT, 'python')
AGENT_DIR = os.path.join(PROJECT_ROOT, 'agent')
DB_PATH = os.path.join(PYTHON_DIR, 'data', 'stock_data.db')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

sys.path.insert(0, PYTHON_DIR)
sys.path.insert(0, AGENT_DIR)

from config import TUSHARE_TOKEN

import tushare as ts
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'kline_update.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_priority_symbols():
    """获取需要更新的股票列表（按优先级排序）"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # P0: 持仓股
    cursor.execute("SELECT symbol FROM positions")
    position_symbols = [r[0] for r in cursor.fetchall()]
    
    # P1: 自选股
    from config import WATCHLIST
    watchlist_symbols = [w.get('symbol') for w in WATCHLIST]
    
    # P2: DB里已有数据的股票（需要更新到最新）
    cursor.execute("SELECT DISTINCT symbol FROM kline_daily")
    db_symbols = [r[0] for r in cursor.fetchall()]
    
    # P3: stock_info里但DB没有数据的
    cursor.execute("SELECT symbol FROM stock_info")
    info_symbols = [r[0] for r in cursor.fetchall()]
    
    conn.close()
    
    # 按优先级排序：持仓 > 自选 > DB已有 > 新增
    priority = []
    seen = set()
    
    for s in position_symbols + watchlist_symbols:
        if s not in seen and '.HK' not in s:  # 跳过港股（Tushare不支持）
            priority.append(('P0', s))
            seen.add(s)
    
    for s in db_symbols:
        if s not in seen and '.HK' not in s:
            priority.append(('P1', s))
            seen.add(s)
    
    for s in info_symbols:
        if s not in seen and '.HK' not in s:
            priority.append(('P2', s))
            seen.add(s)
    
    return priority


def update_one_stock(symbol: str, conn, cursor) -> bool:
    """更新单只股票的K线数据"""
    try:
        # 获取DB里该股票的最新日期
        cursor.execute("SELECT MAX(date) FROM kline_daily WHERE symbol=?", (symbol,))
        last_date_row = cursor.fetchone()
        last_date = last_date_row[0] if last_date_row and last_date_row[0] else '20200101'
        
        # 转换格式
        start_date = last_date.replace('-', '') if '-' in last_date else last_date
        
        end_date = datetime.now().strftime('%Y%m%d')
        
        # 如果start_date == end_date，已经是最新的
        if start_date >= end_date:
            logger.info(f"  - {symbol}: 已是最新 ({last_date})")
            return True
        
        # 调用Tushare API
        df = pro.daily(ts_code=symbol, start_date=start_date, end_date=end_date)
        
        if df is None or df.empty:
            logger.info(f"  - {symbol}: 无新数据")
            return True
        
        # 转换列名和格式
        df = df.rename(columns={'trade_date': 'date', 'ts_code': 'symbol', 'vol': 'volume'})
        df = df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']]
        df['date'] = df['date'].apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}")
        
        # 只插入新数据
        df_new = df[df['date'] > last_date]
        
        if not df_new.empty:
            df_new.to_sql('kline_daily', conn, if_exists='append', index=False, method='multi')
            conn.commit()
            logger.info(f"  ✓ {symbol}: +{len(df_new)} 条 (到 {df_new['date'].max()})")
        else:
            logger.info(f"  - {symbol}: 无新数据需插入")
        
        return True
        
    except Exception as e:
        logger.error(f"  ✗ {symbol}: {str(e)[:100]}")
        return False


def run_full_update():
    """全量更新 - 每分钟1只股票"""
    priority_list = get_priority_symbols()
    total = len(priority_list)
    logger.info(f"开始全量更新，共 {total} 只股票（预计耗时约 {total} 分钟）")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    success = 0
    failed = 0
    
    for i, (level, symbol) in enumerate(priority_list):
        logger.info(f"[{i+1}/{total}] [{level}] {symbol}")
        ok = update_one_stock(symbol, conn, cursor)
        if ok:
            success += 1
        else:
            failed += 1
        
        # 限频：每次调用后等60秒（Tushare 1次/分钟）
        if i < total - 1:  # 最后一只不用等
            logger.info(f"  等待60秒（限频）...")
            time.sleep(60)
    
    conn.close()
    logger.info(f"更新完成: ✓{success} ✗{failed} / 共{total}")


def run_incremental_update():
    """增量更新 - 只更新持仓+自选股"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    from config import WATCHLIST
    cursor.execute("SELECT symbol FROM positions")
    symbols = [r[0] for r in cursor.fetchall()] + [w.get('symbol') for w in WATCHLIST]
    symbols = [s for s in set(symbols) if '.HK' not in s]
    
    logger.info(f"增量更新 {len(symbols)} 只核心股票")
    
    success = 0
    for i, symbol in enumerate(symbols):
        logger.info(f"[{i+1}/{len(symbols)}] {symbol}")
        ok = update_one_stock(symbol, conn, cursor)
        if ok:
            success += 1
        
        if i < len(symbols) - 1:
            time.sleep(60)
    
    conn.close()
    logger.info(f"增量更新完成: ✓{success}")


def run_symbol_list(symbols: list):
    """更新指定股票列表"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for i, symbol in enumerate(symbols):
        update_one_stock(symbol, conn, cursor)
        if i < len(symbols) - 1:
            time.sleep(60)
    
    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='限频K线更新')
    parser.add_argument('--full', action='store_true', help='全量更新所有股票')
    parser.add_argument('--incremental', action='store_true', help='增量更新持仓+自选')
    parser.add_argument('--symbols', type=str, help='指定股票，逗号分隔')
    args = parser.parse_args()
    
    if args.full:
        run_full_update()
    elif args.incremental:
        run_incremental_update()
    elif args.symbols:
        run_symbol_list(args.symbols.split(','))
    else:
        parser.print_help()
