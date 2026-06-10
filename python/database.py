#!/usr/bin/env python3
"""
交易信号数据库模块
使用SQLite存储交易信号历史，支持前端可视化查询
"""

import os
import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from contextlib import contextmanager


class SignalDatabase:
    """交易信号数据库"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), 'data', 'trading.db')

        self.db_path = db_path
        # 确保data目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """初始化数据库表"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 交易信号表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    stock_name TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    price REAL NOT NULL,
                    signal TEXT NOT NULL,
                    score REAL NOT NULL,
                    reasons TEXT,
                    indicators TEXT,
                    stop_loss REAL,
                    take_profit REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at)')

            # 持仓快照表（每日记录账户状态）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    total_value REAL NOT NULL,
                    cash REAL NOT NULL,
                    positions TEXT,
                    daily_pnl REAL,
                    total_pnl REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 交易记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    stock_name TEXT NOT NULL,
                    action TEXT NOT NULL,
                    shares INTEGER NOT NULL,
                    price REAL NOT NULL,
                    amount REAL NOT NULL,
                    reason TEXT,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()

    def save_signal(self, signal: Dict) -> int:
        """
        保存交易信号

        Args:
            signal: 信号字典，包含 symbol, stock_name, price, signal, score, reasons, indicators 等

        Returns:
            插入的记录ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO signals (symbol, stock_name, timestamp, price, signal, score,
                                     reasons, indicators, stop_loss, take_profit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal.get('symbol'),
                signal.get('stock_name'),
                signal.get('timestamp', datetime.now().isoformat()),
                signal.get('price'),
                signal.get('signal'),
                signal.get('score'),
                json.dumps(signal.get('reasons', []), ensure_ascii=False),
                json.dumps(signal.get('indicators', {}), ensure_ascii=False),
                signal.get('stop_loss'),
                signal.get('take_profit')
            ))
            conn.commit()
            return cursor.lastrowid

    def save_signals_batch(self, signals: List[Dict]) -> int:
        """批量保存信号"""
        count = 0
        for signal in signals:
            self.save_signal(signal)
            count += 1
        return count

    def save_portfolio_snapshot(self, snapshot: Dict) -> int:
        """
        保存持仓快照

        Args:
            snapshot: 包含 total_value, cash, positions, daily_pnl, total_pnl
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO portfolio_snapshots (timestamp, total_value, cash, positions,
                                                  daily_pnl, total_pnl)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                snapshot.get('timestamp', datetime.now().isoformat()),
                snapshot.get('total_value'),
                snapshot.get('cash'),
                json.dumps(snapshot.get('positions', {}), ensure_ascii=False),
                snapshot.get('daily_pnl'),
                snapshot.get('total_pnl')
            ))
            conn.commit()
            return cursor.lastrowid

    def save_trade(self, trade: Dict) -> int:
        """
        保存交易记录

        Args:
            trade: 包含 symbol, stock_name, action, shares, price, reason
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trades (symbol, stock_name, action, shares, price, amount, reason, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.get('symbol'),
                trade.get('stock_name'),
                trade.get('action'),
                trade.get('shares'),
                trade.get('price'),
                trade.get('shares', 0) * trade.get('price', 0),
                trade.get('reason'),
                trade.get('timestamp', datetime.now().isoformat())
            ))
            conn.commit()
            return cursor.lastrowid

    def get_signals(self, symbol: str = None, days: int = 30, limit: int = 500) -> List[Dict]:
        """
        查询交易信号

        Args:
            symbol: 股票代码（可选）
            days: 查询最近N天
            limit: 最大返回数量

        Returns:
            信号列表
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            since = (datetime.now() - timedelta(days=days)).isoformat()

            if symbol:
                cursor.execute('''
                    SELECT * FROM signals
                    WHERE symbol = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (symbol, since, limit))
            else:
                cursor.execute('''
                    SELECT * FROM signals
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (since, limit))

            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]

    def get_latest_signals(self) -> List[Dict]:
        """获取每只股票的最新信号"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT s.* FROM signals s
                INNER JOIN (
                    SELECT symbol, MAX(timestamp) as max_ts
                    FROM signals
                    GROUP BY symbol
                ) latest ON s.symbol = latest.symbol AND s.timestamp = latest.max_ts
                ORDER BY s.timestamp DESC
            ''')
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]

    def get_portfolio_history(self, days: int = 30) -> List[Dict]:
        """获取持仓历史"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            since = (datetime.now() - timedelta(days=days)).isoformat()
            cursor.execute('''
                SELECT * FROM portfolio_snapshots
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            ''', (since,))
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]

    def get_trades(self, symbol: str = None, days: int = 30) -> List[Dict]:
        """获取交易记录"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            since = (datetime.now() - timedelta(days=days)).isoformat()

            if symbol:
                cursor.execute('''
                    SELECT * FROM trades
                    WHERE symbol = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                ''', (symbol, since))
            else:
                cursor.execute('''
                    SELECT * FROM trades
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                ''', (since,))

            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]

    def get_signal_stats(self, days: int = 30) -> Dict:
        """获取信号统计"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            since = (datetime.now() - timedelta(days=days)).isoformat()

            # 按信号类型统计
            cursor.execute('''
                SELECT signal, COUNT(*) as count
                FROM signals
                WHERE timestamp >= ?
                GROUP BY signal
            ''', (since,))
            signal_counts = {row['signal']: row['count'] for row in cursor.fetchall()}

            # 按股票统计
            cursor.execute('''
                SELECT symbol, stock_name, COUNT(*) as count,
                       AVG(score) as avg_score
                FROM signals
                WHERE timestamp >= ?
                GROUP BY symbol
                ORDER BY count DESC
            ''', (since,))
            stock_stats = [dict(row) for row in cursor.fetchall()]

            return {
                'signal_counts': signal_counts,
                'stock_stats': stock_stats,
                'period_days': days
            }

    def cleanup_old_records(self, days: int = 90):
        """清理超过N天的记录"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()

            cursor.execute('DELETE FROM signals WHERE created_at < ?', (cutoff,))
            signals_deleted = cursor.rowcount

            cursor.execute('DELETE FROM portfolio_snapshots WHERE created_at < ?', (cutoff,))
            snapshots_deleted = cursor.rowcount

            conn.commit()

            return {
                'signals_deleted': signals_deleted,
                'snapshots_deleted': snapshots_deleted
            }

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        """将数据库行转换为字典"""
        result = dict(row)
        # 解析JSON字段
        if 'reasons' in result and result['reasons']:
            result['reasons'] = json.loads(result['reasons'])
        if 'indicators' in result and result['indicators']:
            result['indicators'] = json.loads(result['indicators'])
        if 'positions' in result and result['positions']:
            result['positions'] = json.loads(result['positions'])
        return result

    def export_to_csv(self, table: str = 'signals', output_path: str = None) -> str:
        """导出数据到CSV"""
        import csv

        if output_path is None:
            output_path = os.path.join(os.path.dirname(__file__), 'data', f'{table}.csv')

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'SELECT * FROM {table} ORDER BY timestamp DESC LIMIT 1000')
            rows = cursor.fetchall()

            if rows:
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows([dict(row) for row in rows])

        return output_path


# 单例实例
_db_instance = None

def get_db() -> SignalDatabase:
    """获取数据库实例"""
    global _db_instance
    if _db_instance is None:
        _db_instance = SignalDatabase()
    return _db_instance


# 测试
if __name__ == "__main__":
    db = get_db()

    # 测试保存信号
    test_signal = {
        'symbol': '300015.SZ',
        'stock_name': '爱尔眼科',
        'price': 9.68,
        'signal': '买入',
        'score': 4.5,
        'reasons': ['RSI超卖', 'MACD金叉'],
        'indicators': {'rsi': 28.5, 'macd': 0.02}
    }
    db.save_signal(test_signal)

    # 查询信号
    signals = db.get_signals(days=1)
    print(f"查询到 {len(signals)} 条信号")

    # 获取统计
    stats = db.get_signal_stats(days=30)
    print(f"统计: {stats}")