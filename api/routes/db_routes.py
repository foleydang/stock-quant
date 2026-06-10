"""数据库查询路由"""

from flask import Blueprint, jsonify, request
import sqlite3
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))
from config_loader import get_db_path

db_bp = Blueprint('db', __name__)

@db_bp.route('/db/signals', methods=['GET'])
def db_get_signals():
    """获取信号记录"""
    try:
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM signals ORDER BY created_at DESC LIMIT 100')
        rows = cursor.fetchall()
        conn.close()
        
        return jsonify({'status': 'success', 'count': len(rows), 'data': rows})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@db_bp.route('/db/portfolio', methods=['GET'])
def db_get_portfolio():
    """获取账户汇总"""
    try:
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM account ORDER BY id DESC LIMIT 1')
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return jsonify({
                'status': 'success',
                'cash': float(row[1]),
                'total_value': float(row[2]),
                'profit': float(row[3])
            })
        return jsonify({'status': 'error', 'message': '无数据'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@db_bp.route('/db/trades', methods=['GET'])
def db_get_trades():
    """获取交易记录"""
    try:
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM trades ORDER BY trade_time DESC LIMIT 100')
        rows = cursor.fetchall()
        conn.close()
        
        trades = [{
            'id': r[0],
            'symbol': r[1],
            'name': r[2],
            'action': r[3],
            'shares': r[4],
            'price': float(r[5]),
            'amount': float(r[6]) if r[6] else 0,
            'profit': float(r[7]) if r[7] else 0,
            'reason': r[8] or '',
            'timestamp': r[12] or r[10]  # 优先用timestamp，fallback到trade_time
        } for r in rows]
        
        return jsonify({'status': 'success', 'trades': trades})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@db_bp.route('/db/stats', methods=['GET'])
def db_get_stats():
    """获取数据统计"""
    try:
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        
        # 统计各表数据量
        stats = {}
        
        cursor.execute('SELECT COUNT(*) FROM kline_30m')
        stats['kline_count'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT symbol) FROM kline_30m')
        stats['stock_count'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM positions')
        stats['position_count'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT MAX(date) FROM kline_30m')
        stats['latest_date'] = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({'status': 'success', 'stats': stats})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500
@db_bp.route("/db/stocks", methods=["GET"])
def get_all_stocks():
    """获取所有股票列表（带名称）"""
    try:
        import sqlite3
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        
        # 从kline_30m获取所有股票
        cursor.execute("SELECT DISTINCT symbol FROM kline_30m ORDER BY symbol")
        symbols = [r[0] for r in cursor.fetchall()]
        
        # 尝试获取名称
        stocks = []
        for sym in symbols:
            cursor.execute("SELECT name FROM stock_info WHERE symbol=?", (sym,))
            row = cursor.fetchone()
            name = row[0] if row and row[0] else sym
            stocks.append({"symbol": sym, "name": name})
        
        conn.close()
        return jsonify({"status": "success", "stocks": stocks, "count": len(stocks)})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500
