"""股票数据相关路由"""

from flask import Blueprint, jsonify, request
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))
from config_loader import get_db_path

stock_bp = Blueprint('stock', __name__)

@stock_bp.route('/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    """获取股票日线数据"""
    try:
        import pandas as pd
        import sqlite3
        
        conn = sqlite3.connect(get_db_path())
        df = pd.read_sql_query(
            'SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol=? ORDER BY date DESC LIMIT 100',
            conn, params=(symbol,)
        )
        # conn.close() moved to end
        
        if df.empty:
            return jsonify({'status': 'error', 'message': '无数据'}), 404
        
        data = []
        for _, row in df.iterrows():
            data.append({
                'date': row['date'],
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row['volume'])
            })
        
        return jsonify({'status': 'success', 'data': data[::-1]})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@stock_bp.route('/stock/<symbol>/<period>', methods=['GET'])
def get_stock_data_by_period(symbol, period):
    """获取不同周期的股票数据"""
    try:
        import sqlite3
        
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        
        if period == "30m":
            # 30分钟线数据（最近50条）
            cursor.execute(
                "SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol=? ORDER BY date DESC LIMIT 50",
                (symbol,)
            )
        elif period == "daily":
            # 日线：按日期聚合，生成OHLC
            cursor.execute(
                "SELECT substr(date, 1, 10) as day, MIN(open) as open, MAX(high) as high, MIN(low) as low, MAX(close) as close, SUM(volume) as volume FROM kline_30m WHERE symbol=? GROUP BY substr(date, 1, 10) ORDER BY day DESC LIMIT 365",
                (symbol,)
            )
        elif period == "weekly":
            # 周线：按周聚合
            cursor.execute(
                "SELECT substr(date, 1, 10) as day, MIN(open) as open, MAX(high) as high, MIN(low) as low, MAX(close) as close, SUM(volume) as volume FROM kline_30m WHERE symbol=? GROUP BY strftime('%Y-%W', date) ORDER BY day DESC LIMIT 52",
                (symbol,)
            )
        elif period == "monthly":
            # 月线：按月聚合
            cursor.execute(
                "SELECT substr(date, 1, 7) as month, MIN(open) as open, MAX(high) as high, MIN(low) as low, MAX(close) as close, SUM(volume) as volume FROM kline_30m WHERE symbol=? GROUP BY substr(date, 1, 7) ORDER BY month DESC LIMIT 24",
                (symbol,)
            )
        else:
            conn.close()
            return jsonify({"status": "error", "message": "不支持的周期"}), 400
        
        rows = cursor.fetchall()
        
        if period == "30m":
            data = [{
                "date": r[0],
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": int(r[5])
            } for r in rows]
        else:
            # daily/weekly/monthly 返回真正的OHLCV
            data = [{
                "date": r[0],
                "open": float(r[1]) if r[1] else 0,
                "high": float(r[2]) if r[2] else 0,
                "low": float(r[3]) if r[3] else 0,
                "close": float(r[4]) if r[4] else 0,
                "volume": int(r[5]) if r[5] else 0
            } for r in rows]
        
        conn.close()
        return jsonify({"status": "success", "data": data[::-1], "period": period})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@stock_bp.route('/stocks', methods=['GET'])
def get_stocks():
    """获取所有股票列表"""
    try:
        import sqlite3
        
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT symbol FROM kline_30m ORDER BY symbol')
        rows = cursor.fetchall()
        
        stocks = [{'symbol': r[0]} for r in rows]
        return jsonify({'status': 'success', 'stocks': stocks})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@stock_bp.route('/positions', methods=['GET'])
def get_positions():
    """获取持仓列表"""
    try:
        import sqlite3
        
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        cursor.execute('SELECT symbol, stock_name, shares, cost_price, current_price FROM positions')
        rows = cursor.fetchall()
        
        positions = [{
            'symbol': r[0],
            'name': r[1],
            'shares': r[2],
            'cost': float(r[3]),
            'current': float(r[4]),
            'profit': float(r[4]) - float(r[3]),
            'profit_pct': (float(r[4]) - float(r[3])) / float(r[3]) * 100
        } for r in rows]
        
        return jsonify({'status': 'success', 'positions': positions})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500
@stock_bp.route("/positions", methods=["POST"])
def add_position():
    """添加持仓"""
    try:
        import sqlite3
        data = request.get_json()
        
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO positions (symbol, stock_name, shares, cost_price, current_price) VALUES (?, ?, ?, ?, ?)",
            (data["symbol"], data["name"], data["shares"], data["cost"], data["cost"])
        )
        conn.commit()
        conn.close()
        
        return jsonify({"status": "success", "message": "添加成功"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@stock_bp.route("/positions/<symbol>", methods=["PUT"])
def update_position(symbol):
    """更新持仓"""
    try:
        import sqlite3
        data = request.get_json()
        
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE positions SET stock_name=?, shares=?, cost_price=? WHERE symbol=?",
            (data["name"], data["shares"], data["cost"], symbol)
        )
        conn.commit()
        conn.close()
        
        return jsonify({"status": "success", "message": "更新成功"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@stock_bp.route("/positions/<symbol>", methods=["DELETE"])
def delete_position(symbol):
    """删除持仓"""
    try:
        import sqlite3
        
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        cursor.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
        conn.commit()
        conn.close()
        
        return jsonify({"status": "success", "message": "删除成功"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@stock_bp.route("/kline/<symbol>", methods=["GET"])
def get_kline_data(symbol):
    """获取分钟线数据"""
    try:
        import sqlite3
        period = request.args.get("period", "30min")
        limit = int(request.args.get("limit", 50))
        
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        
        if period == "30min":
            cursor.execute(
                "SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol=? ORDER BY date DESC LIMIT ?",
                (symbol, limit)
            )
        else:
            cursor.execute(
                "SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol=? ORDER BY date DESC LIMIT ?",
                (symbol, limit)
            )
        
        rows = cursor.fetchall()
        
        data = [{
            "date": r[0],
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "volume": int(r[5])
        } for r in rows]
        
        return jsonify({"status": "success", "data": data[::-1], "period": period})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@stock_bp.route("/stocks/with_kline", methods=["GET"])
def get_stocks_with_kline():
    """获取有分钟线数据的股票列表"""
    try:
        import sqlite3
        
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        
        # 获取有足够分钟线数据的股票（>50条）
        cursor.execute("""
            SELECT symbol, COUNT(*) as cnt, MAX(date) as latest
            FROM kline_30m 
            GROUP BY symbol 
            HAVING cnt > 50
            ORDER BY latest DESC
        """)
        rows = cursor.fetchall()
        
        stocks = []
        for r in rows:
            symbol = r[0]
            cnt = r[1]
            latest = r[2]
            
            # 解析市场
            if symbol.endswith(".SZ"):
                market = "深圳"
            elif symbol.endswith(".SH"):
                market = "上海"
            elif symbol.endswith(".HK"):
                market = "港股"
            else:
                market = "其他"
            
            # 获取股票名称（优先positions，其次stock_info）
            name = symbol
            try:
                cursor.execute("SELECT stock_name FROM positions WHERE symbol=?", (symbol,))
                row_name = cursor.fetchone()
                if row_name and row_name[0]:
                    name = row_name[0]
                else:
                    # 从stock_info获取
                    cursor.execute("SELECT name FROM stock_info WHERE symbol=?", (symbol,))
                    row_info = cursor.fetchone()
                    if row_info and row_info[0]:
                        name = row_info[0]
            except Exception as e:
                pass
            
            stocks.append({
                "symbol": symbol,
                "name": name,
                "count": cnt,
                "latest_date": latest,
                "market": market,
                "has_intraday": True
            })
        
        conn.close()
        return jsonify({"status": "success", "stocks": stocks, "count": len(stocks)})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@stock_bp.route("/trades", methods=["GET"])
def get_trades():
    """获取交易记录"""
    try:
        import sqlite3
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades ORDER BY id DESC LIMIT 100")
        rows = cursor.fetchall()
        conn.close()
        
        trades = []
        for r in rows:
            trades.append({
                "id": r[0],
                "symbol": r[1],
                "name": r[2],
                "action": r[3],
                "shares": r[4],
                "price": float(r[5]),
                "amount": float(r[6]) if r[6] else 0,
                "profit": float(r[7]) if r[7] else 0,
                "reason": r[8] or '',
                "up_prob": float(r[9]) if r[9] else 0,
                "timestamp": r[12] or r[10]  # timestamp or trade_time
            })
        
        return jsonify({"status": "success", "trades": trades})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@stock_bp.route("/trade", methods=["POST"])
def add_trade():
    """添加交易记录"""
    try:
        import sqlite3
        data = request.json
        
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO trades (symbol, stock_name, action, shares, price, amount, timestamp) VALUES (?, ?, ?, ?, ?, ?, datetime(\"now\"))",
            (data.get("symbol"), data.get("stock_name", ""), data.get("action"), data.get("shares", 0), data.get("price", 0), data.get("amount", 0))
        )
        conn.commit()
        conn.close()
        
        return jsonify({"status": "success", "message": "添加成功"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500
