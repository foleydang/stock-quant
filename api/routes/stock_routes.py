"""股票数据路由"""
from flask import Blueprint, jsonify, request
import sqlite3
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))
from config_loader import get_db_path

stock_bp = Blueprint('stock', __name__)


@stock_bp.route('/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    """获取股票数据"""
    try:
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM stock_info WHERE symbol=?", (symbol,))
        row = cursor.fetchone()
        name = row[0] if row and row[0] else symbol
        
        cursor.execute("SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol=? ORDER BY date DESC LIMIT 50", (symbol,))
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return jsonify({'status': 'error', 'message': '无数据'}), 404
        
        data = []
        for r in rows:
            data.append({
                'date': r[0],
                'open': float(r[1]),
                'high': float(r[2]),
                'low': float(r[3]),
                'close': float(r[4]),
                'volume': int(r[5])
            })
        
        prices = [d['close'] for d in data]
        latest_price = prices[0]
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'name': name,
            'latestPrice': latest_price,
            'totalReturn': round((latest_price - prices[-1]) / prices[-1] * 100, 2),
            'maxPrice': max(prices),
            'minPrice': min(prices),
            'avgPrice': round(sum(prices) / len(prices), 2),
            'dataCount': len(data),
            'data': data
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@stock_bp.route('/stock/<symbol>/<period>', methods=['GET'])
def get_stock_data_by_period(symbol, period):
    """获取指定周期的股票数据 - 日线/周线/月线从30分钟数据动态聚合"""
    try:
        import sqlite3
        import pandas as pd
        
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM stock_info WHERE symbol=?", (symbol,))
        row = cursor.fetchone()
        name = row[0] if row and row[0] else symbol
        
        if period == '30m':
            # 30分钟线直接从表读
            limit = 50
            cursor.execute(f"SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol=? ORDER BY date DESC LIMIT {limit}", (symbol,))
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return jsonify({'status': 'error', 'message': '无数据'}), 404
            
            data = [{'date': r[0], 'open': float(r[1]), 'high': float(r[2]), 'low': float(r[3]), 'close': float(r[4]), 'volume': int(r[5])} for r in rows]
            data.reverse()
        else:
            # 日线/周线/月线: 从30分钟数据聚合
            cursor.execute("SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol=? ORDER BY date ASC", (symbol,))
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return jsonify({'status': 'error', 'message': '无数据'}), 404
            
            df = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date'], format='mixed')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(int)
            
            if period == 'daily':
                df_agg = df.groupby(df['date'].dt.date).agg(
                    open=('open', 'first'),
                    high=('high', 'max'),
                    low=('low', 'min'),
                    close=('close', 'last'),
                    volume=('volume', 'sum')
                ).reset_index()
                df_agg = df_agg.tail(100)
            elif period == 'weekly':
                df['week'] = df['date'].dt.to_period('W').apply(lambda x: x.start_time)
                df_agg = df.groupby('week').agg(
                    open=('open', 'first'),
                    high=('high', 'max'),
                    low=('low', 'min'),
                    close=('close', 'last'),
                    volume=('volume', 'sum')
                ).reset_index()
                df_agg = df_agg.tail(50)
            elif period == 'monthly':
                df['month'] = df['date'].dt.to_period('M').apply(lambda x: x.start_time)
                df_agg = df.groupby('month').agg(
                    open=('open', 'first'),
                    high=('high', 'max'),
                    low=('low', 'min'),
                    close=('close', 'last'),
                    volume=('volume', 'sum')
                ).reset_index()
                df_agg = df_agg.tail(24)
            else:
                return jsonify({'status': 'error', 'message': f'未知周期: {period}'}), 400
            
            data = []
            for _, r in df_agg.iterrows():
                date_col = 'date' if 'date' in df_agg.columns else 'week' if 'week' in df_agg.columns else 'month'
                data.append({
                    'date': str(r[date_col]),
                    'open': float(r['open']),
                    'high': float(r['high']),
                    'low': float(r['low']),
                    'close': float(r['close']),
                    'volume': int(r['volume'])
                })
        
        prices = [d['close'] for d in data]
        latest_price = prices[-1]
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'name': name,
            'latestPrice': latest_price,
            'totalReturn': round((latest_price - prices[0]) / prices[0] * 100, 2),
            'maxPrice': max(prices),
            'minPrice': min(prices),
            'avgPrice': round(sum(prices) / len(prices), 2),
            'dataCount': len(data),
            'data': data
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@stock_bp.route('/stocks', methods=['GET'])
def get_stocks():
    """获取所有股票列表"""
    try:
        import sqlite3
        
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, name FROM stock_info ORDER BY symbol")
        rows = cursor.fetchall()
        conn.close()
        
        stocks = [{'symbol': r[0], 'name': r[1]} for r in rows]
        return jsonify({'status': 'success', 'stocks': stocks})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@stock_bp.route('/positions', methods=['GET'])
def get_positions():
    """获取持仓"""
    try:
        import sqlite3
        
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        cursor.execute('SELECT symbol, stock_name, shares, cost_price, current_price FROM positions')
        rows = cursor.fetchall()
        conn.close()
        
        positions = [{
            'symbol': r[0],
            'name': r[1],
            'shares': int(r[2]),
            'cost': float(r[3]),
            'current': float(r[4])
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
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM stock_info WHERE symbol=?", (symbol,))
        row = cursor.fetchone()
        name = row[0] if row and row[0] else symbol
        
        cursor.execute("SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol=? ORDER BY date DESC LIMIT 50", (symbol,))
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return jsonify({"status": "error", "message": "无数据"}), 404
        
        data = []
        for r in rows:
            data.append({
                "date": r[0],
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": int(r[5])
            })
        
        return jsonify({
            "status": "success",
            "symbol": symbol,
            "name": name,
            "data": data
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@stock_bp.route("/stocks/with_kline", methods=["GET"])
def get_stocks_with_kline():
    """获取有K线数据的股票"""
    try:
        import sqlite3
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT s.symbol, s.name, COUNT(k.date) as cnt, MAX(k.close) as latest_price
            FROM stock_info s
            JOIN kline_30m k ON s.symbol = k.symbol
            GROUP BY s.symbol
            HAVING cnt > 10
            ORDER BY cnt DESC
            LIMIT 20
        """)
        rows = cursor.fetchall()
        conn.close()
        
        stocks = [{
            "symbol": r[0],
            "name": r[1],
            "dataCount": int(r[2]),
            "latestPrice": float(r[3])
        } for r in rows]
        
        return jsonify({
            "status": "success",
            "stocks": stocks
        })
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
        
        # 获取列名
        trades = []
        for r in rows:
            trades.append({
                "id": r[0],
                "symbol": r[1],
                "name": r[2],
                "action": r[3],
                "shares": r[4],
                "price": r[5],
                "amount": r[6],
                "reason": r[7] if len(r) > 7 else "",
                "timestamp": r[12] if len(r) > 12 else r[10] if len(r) > 10 else ""
            })
        
        return jsonify({"status": "success", "trades": trades})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@stock_bp.route("/trade", methods=["POST"])
def add_trade():
    """添加交易记录 + 同步更新持仓"""
    try:
        import sqlite3
        data = request.json
        symbol = data.get("symbol", "")
        stock_name = data.get("stock_name", "")
        action = data.get("action", "BUY")
        shares = int(data.get("shares", 0) or 0)
        price = float(data.get("price", 0) or 0)
        amount = float(data.get("amount", 0) or 0)
        # 如果前端没传amount，自动计算
        if amount == 0 and price > 0 and shares > 0:
            amount = price * shares
        reason = data.get("reason", "")

        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()

        # 1. 插入交易记录
        cursor.execute(
            "INSERT INTO trades (symbol, stock_name, action, shares, price, amount, reason, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            (symbol, stock_name, action, shares, price, amount, reason)
        )

        # 2. 同步更新持仓
        cursor.execute("SELECT shares, cost_price FROM positions WHERE symbol=?", (symbol,))
        pos = cursor.fetchone()

        if action == "BUY":
            if pos:
                # 已有持仓：加权平均成本价
                old_shares = int(pos[0])
                old_cost = float(pos[1])
                new_shares = old_shares + shares
                new_cost = (old_cost * old_shares + price * shares) / new_shares
                cursor.execute(
                    "UPDATE positions SET shares=?, cost_price=?, current_price=?, stock_name=? WHERE symbol=?",
                    (new_shares, new_cost, price, stock_name, symbol)
                )
            else:
                # 新建持仓
                cursor.execute(
                    "INSERT INTO positions (symbol, stock_name, shares, cost_price, current_price) VALUES (?, ?, ?, ?, ?)",
                    (symbol, stock_name, shares, price, price)
                )
        elif action == "SELL":
            if pos:
                old_shares = int(pos[0])
                new_shares = old_shares - shares
                if new_shares <= 0:
                    # 清仓：删除持仓
                    cursor.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
                else:
                    # 减仓：成本价不变，更新现价
                    cursor.execute(
                        "UPDATE positions SET shares=?, current_price=? WHERE symbol=?",
                        (new_shares, price, symbol)
                    )

        conn.commit()
        conn.close()

        return jsonify({"status": "success", "message": "交易记录成功，持仓已更新"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500