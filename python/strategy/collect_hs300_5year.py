#!/usr/bin/env python3
"""
沪深300成分股历史数据收集脚本
- 收集最近5年的30分钟级别数据
- 使用SQLite存储
- 支持断点续传
- 智能限流避免被封

用法:
    nohup python3 strategy/collect_hs300_5year.py > logs/collect_hs300_5year.log 2>&1 &
"""

import os
import sys
import time
import sqlite3
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("错误: akshare 未安装")
    sys.exit(1)


class HS300DataCollector:
    """沪深300数据收集器 - SQLite版本"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(
            os.path.dirname(__file__), '../data/stock_data.db'
        )
        self._init_database()

        # 限流参数
        self.base_delay = 3.0
        self.max_delay = 15.0
        self.batch_break = 30.0

        # 统计
        self.success_count = 0
        self.fail_count = 0
        self.skip_count = 0
        self.update_count = 0

    def _init_database(self):
        """初始化数据库"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 股票基本信息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_info (
                symbol TEXT PRIMARY KEY,
                code TEXT,
                name TEXT,
                market TEXT,
                update_time TEXT
            )
        ''')

        # 30分钟K线数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS kline_30m (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                created_at TEXT,
                UNIQUE(symbol, date)
            )
        ''')

        # 创建索引加速查询
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_kline_symbol
            ON kline_30m(symbol)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_kline_date
            ON kline_30m(date)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_kline_symbol_date
            ON kline_30m(symbol, date)
        ''')

        # 数据收集进度表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collect_progress (
                symbol TEXT PRIMARY KEY,
                last_date TEXT,
                total_records INTEGER,
                last_update TEXT,
                status TEXT
            )
        ''')

        conn.commit()
        conn.close()
        print(f"数据库初始化完成: {self.db_path}")

    def get_hs300_constituents(self) -> List[Dict]:
        """获取沪深300成分股列表"""
        print("获取沪深300成分股列表...")

        try:
            df = ak.index_stock_cons_weight_csindex(symbol="000300")
            constituents = []
            for _, row in df.iterrows():
                code = str(row.get('成分券代码', ''))
                name = row.get('成分券名称', '')
                if code:
                    if code.startswith('6'):
                        symbol = f"{code}.SH"
                        market = 'sh'
                    else:
                        symbol = f"{code}.SZ"
                        market = 'sz'
                    constituents.append({
                        'symbol': symbol,
                        'code': code,
                        'name': name,
                        'market': market
                    })

            # 保存到数据库
            self._save_stock_info(constituents)
            print(f"获取到 {len(constituents)} 只沪深300成分股")
            return constituents

        except Exception as e:
            print(f"获取成分股失败: {e}")
            return self._get_backup_constituents()

    def _get_backup_constituents(self) -> List[Dict]:
        """备用成分股列表"""
        stocks = [
            ('600519.SH', '贵州茅台'), ('000858.SZ', '五粮液'), ('601318.SH', '中国平安'),
            ('600036.SH', '招商银行'), ('601166.SH', '兴业银行'), ('600000.SH', '浦发银行'),
            ('601398.SH', '工商银行'), ('601288.SH', '农业银行'), ('600030.SH', '中信证券'),
            ('600276.SH', '恒瑞医药'), ('000333.SZ', '美的集团'), ('000651.SZ', '格力电器'),
            ('600887.SH', '伊利股份'), ('601888.SH', '中国中免'), ('600009.SH', '上海机场'),
            ('601012.SH', '隆基绿能'), ('600900.SH', '长江电力'), ('601818.SH', '光大银行'),
            ('601939.SH', '建设银行'), ('601988.SH', '中国银行'), ('600048.SH', '保利发展'),
            ('601628.SH', '中国人寿'), ('601601.SH', '中国太保'), ('601336.SH', '新华保险'),
            ('600585.SH', '海螺水泥'), ('600309.SH', '万华化学'), ('600346.SH', '恒力石化'),
            ('000002.SZ', '万科A'), ('000001.SZ', '平安银行'), ('002594.SZ', '比亚迪'),
            ('002475.SZ', '立讯精密'), ('002415.SZ', '海康威视'), ('300750.SZ', '宁德时代'),
            ('300015.SZ', '爱尔眼科'), ('300124.SZ', '汇川技术'),
        ]
        return [{'symbol': s[0], 'code': s[0][:6], 'name': s[1],
                 'market': 'sh' if s[0].startswith('6') else 'sz'} for s in stocks]

    def _save_stock_info(self, constituents: List[Dict]):
        """保存股票信息到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        for stock in constituents:
            cursor.execute('''
                INSERT OR REPLACE INTO stock_info (symbol, code, name, market, update_time)
                VALUES (?, ?, ?, ?, ?)
            ''', (stock['symbol'], stock['code'], stock['name'], stock['market'], now))

        conn.commit()
        conn.close()

    def _smart_delay(self, consecutive_fails: int = 0):
        """智能延时"""
        delay = self.base_delay + random.uniform(1, 3)
        if consecutive_fails > 0:
            delay += min(consecutive_fails * 2, self.max_delay - self.base_delay)
        time.sleep(delay)

    def _get_existing_data_range(self, symbol: str) -> tuple:
        """获取已有数据的时间范围"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT MIN(date), MAX(date), COUNT(*)
            FROM kline_30m WHERE symbol = ?
        ''', (symbol,))

        result = cursor.fetchone()
        conn.close()

        if result and result[2] > 0:
            return result[0], result[1], result[2]
        return None, None, 0

    def fetch_stock_data(self, symbol: str, code: str, market: str) -> Optional[pd.DataFrame]:
        """获取单只股票的30分钟数据"""
        try:
            self._smart_delay()

            sina_code = f"{market}{code}"
            df = ak.stock_zh_a_minute(symbol=sina_code, period='30')

            if df is not None and not df.empty:
                df = df.rename(columns={'day': 'date'})
                df['date'] = pd.to_datetime(df['date'], format='mixed')
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                df = df.sort_values('date').reset_index(drop=True)
                return df

        except Exception as e:
            print(f"    获取失败 {symbol}: {e}")

        return None

    def save_kline_data(self, symbol: str, df: pd.DataFrame):
        """保存K线数据到数据库"""
        if df is None or df.empty:
            return 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        records = []
        for _, row in df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d %H:%M:%S')
            records.append((
                symbol,
                date_str,
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                int(row['volume']),
                now
            ))

        # 批量插入，冲突则忽略
        cursor.executemany('''
            INSERT OR IGNORE INTO kline_30m
            (symbol, date, open, high, low, close, volume, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', records)

        inserted = cursor.rowcount
        conn.commit()
        conn.close()

        return inserted

    def update_progress(self, symbol: str, last_date: str, total_records: int, status: str):
        """更新收集进度"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        cursor.execute('''
            INSERT OR REPLACE INTO collect_progress
            (symbol, last_date, total_records, last_update, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (symbol, last_date, total_records, now, status))

        conn.commit()
        conn.close()

    def collect_all(self, constituents: List[Dict], batch_size: int = 10):
        """收集所有股票数据"""
        total = len(constituents)

        print(f"\n{'='*60}")
        print(f"开始收集沪深300数据 - 5年历史")
        print(f"数据库: {self.db_path}")
        print(f"总数: {total} 只股票")
        print(f"批次大小: {batch_size}")
        print(f"{'='*60}\n")

        for i in range(0, total, batch_size):
            batch = constituents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total + batch_size - 1) // batch_size

            print(f"\n[批次 {batch_num}/{total_batches}]")

            for stock in batch:
                symbol = stock['symbol']
                code = stock['code']
                name = stock['name']
                market = stock['market']

                # 检查已有数据
                min_date, max_date, count = self._get_existing_data_range(symbol)

                # 如果数据量足够（约5年 = 250天/年 * 8条/天 * 5年 = 10000条）
                if count >= 10000:
                    self.skip_count += 1
                    print(f"  [跳过] {name}({symbol}): 已有{count}条数据")
                    continue

                # 获取数据
                print(f"  [获取] {name}({symbol})...", end=" ", flush=True)
                df = self.fetch_stock_data(symbol, code, market)

                if df is not None and len(df) > 0:
                    # 保存到数据库
                    inserted = self.save_kline_data(symbol, df)

                    # 更新进度
                    new_count = count + inserted
                    new_max_date = df['date'].max().strftime('%Y-%m-%d %H:%M:%S')
                    self.update_progress(symbol, new_max_date, new_count, 'completed')

                    if inserted > 0:
                        self.update_count += 1
                        print(f"✓ 新增{inserted}条, 总计{new_count}条")
                    else:
                        self.success_count += 1
                        print(f"✓ 已有{count}条, 无新增")

                else:
                    self.fail_count += 1
                    self.update_progress(symbol, max_date or '', count, 'failed')
                    print(f"✗ 获取失败")

                # 短暂延时
                time.sleep(random.uniform(0.5, 1.5))

            # 批次统计
            print(f"\n批次完成 | 成功:{self.success_count} 更新:{self.update_count} 跳过:{self.skip_count} 失败:{self.fail_count}")

            # 批次间休息
            if batch_num < total_batches:
                print(f"休息 {self.batch_break} 秒...\n")
                time.sleep(self.batch_break)

        # 最终统计
        print(f"\n{'='*60}")
        print(f"收集完成!")
        print(f"成功: {self.success_count}, 更新: {self.update_count}")
        print(f"跳过: {self.skip_count}, 失败: {self.fail_count}")
        print(f"数据库: {self.db_path}")

        # 显示数据库统计
        self.show_stats()
        print(f"{'='*60}\n")

    def show_stats(self):
        """显示数据库统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 股票数量
        cursor.execute('SELECT COUNT(*) FROM stock_info')
        stock_count = cursor.fetchone()[0]

        # K线数据量
        cursor.execute('SELECT COUNT(*) FROM kline_30m')
        kline_count = cursor.fetchone()[0]

        # 数据时间范围
        cursor.execute('SELECT MIN(date), MAX(date) FROM kline_30m')
        min_date, max_date = cursor.fetchone()

        # 每只股票的数据量
        cursor.execute('''
            SELECT symbol, COUNT(*) as cnt
            FROM kline_30m
            GROUP BY symbol
            ORDER BY cnt DESC
            LIMIT 5
        ''')
        top_stocks = cursor.fetchall()

        conn.close()

        print(f"\n数据库统计:")
        print(f"  股票数量: {stock_count}")
        print(f"  K线数据: {kline_count:,} 条")
        print(f"  时间范围: {min_date} ~ {max_date}")
        print(f"  数据最多的股票:")
        for symbol, cnt in top_stocks:
            print(f"    {symbol}: {cnt:,} 条")


def main():
    print(f"\n{'='*60}")
    print(f"沪深300成分股数据收集 - 5年历史数据")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    collector = HS300DataCollector()

    # 获取成分股
    constituents = collector.get_hs300_constituents()
    if not constituents:
        print("无法获取成分股列表")
        return

    # 开始收集
    collector.collect_all(constituents, batch_size=10)

    print("\n提示: 可以使用以下命令查看数据:")
    print("  sqlite3 stock-quant/python/data/stock_data.db")
    print("  SELECT COUNT(*) FROM kline_30m;")


if __name__ == "__main__":
    main()