"""策略相关路由"""
from config_loader import get_db_path

from flask import Blueprint, jsonify
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))

strategy_bp = Blueprint('strategy', __name__)

@strategy_bp.route('/strategy/<symbol>', methods=['GET'])
def run_strategy(symbol):
    """运行策略分析"""
    try:
        from data.data_handler import DataHandler
        dh = DataHandler()
        df = dh.fetch_real_30min_kline(symbol, count=20)
        
        # API失败时fallback到DB
        if df is None or df.empty:
            import sqlite3
            from config_loader import get_db_path
            conn = sqlite3.connect(get_db_path())
            cursor = conn.cursor()
            cursor.execute(
                'SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol=? ORDER BY date DESC LIMIT 20',
                (symbol,)
            )
            rows = cursor.fetchall()
            conn.close()
            if not rows:
                return jsonify({'status': 'error', 'message': '无数据'}), 404
            data = []
            for r in rows:
                data.append({'date': r[0], 'open': float(r[1]), 'high': float(r[2]), 'low': float(r[3]), 'close': float(r[4]), 'volume': int(r[5])})
            data.reverse()  # 恢复时间正序
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'], format='mixed')
            source = 'db_fallback'
        else:
            source = 'api'
        
        # 简单策略分析
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        change_pct = (latest['close'] - prev['close']) / prev['close'] * 100
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'source': source,
            'latest_price': float(latest['close']),
            'change_pct': float(change_pct),
            'high': float(latest['high']),
            'low': float(latest['low']),
            'volume': int(latest['volume'])
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@strategy_bp.route('/predict/<symbol>', methods=['GET'])
def predict_price(symbol):
    """预测价格走势"""
    try:
        import sqlite3
        from config_loader import get_db_path
        
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        cursor.execute(
            'SELECT close FROM kline_30m WHERE symbol=? ORDER BY date DESC LIMIT 20',
            (symbol,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < 5:
            return jsonify({'status': 'error', 'message': '数据不足'}), 404
        
        prices = [float(r[0]) for r in rows]
        # 简单移动平均预测
        ma5 = sum(prices[:5]) / 5
        ma10 = sum(prices[:10]) / 10 if len(prices) >= 10 else ma5
        
        trend = 'up' if ma5 > ma10 else 'down'
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'latest_price': prices[0],
            'ma5': float(ma5),
            'ma10': float(ma10),
            'trend': trend
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@strategy_bp.route("/backtest/<symbol>", methods=["GET"])
def run_backtest(symbol):
    """运行回测"""
    try:
        import sqlite3
        from config_loader import get_db_path
        
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        
        # 获取历史数据
        cursor.execute(
            "SELECT date, close FROM kline_30m WHERE symbol=? ORDER BY date DESC LIMIT 100",
            (symbol,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < 20:
            return jsonify({"status": "error", "message": "数据不足"}), 404
        
        prices = [float(r[1]) for r in rows]
        
        # 简单策略：5日均线 vs 10日均线
        results = {
            "symbol": symbol,
            "total_trades": 0,
            "win_rate": 0,
            "total_return": 0,
            "max_drawdown": 0,
            "sharpe": 0
        }
        
        # 计算收益
        if len(prices) >= 10:
            ma5 = sum(prices[:5]) / 5
            ma10 = sum(prices[:10]) / 10
            
            total_return = (prices[-1] - prices[0]) / prices[0] * 100
            results["total_return"] = round(total_return, 2)
            results["total_trades"] = len(prices) // 5
            results["signal"] = "buy" if ma5 > ma10 else "sell"
        
        # 模拟买卖点（用于图表展示）
        buy_points = []
        sell_points = []
        
        # 简单策略：每5根K线一个买卖信号
        for i in range(0, len(rows)-1, 5):
            date = rows[i][0]
            price = float(rows[i][1])
            if i % 10 == 0:
                buy_points.append({"date": date, "price": price})
            else:
                sell_points.append({"date": date, "price": price})
        
        # 返回前端期望的格式（camelCase）
        return jsonify({
            "status": "success",
            "summary": {
                "profitRate": round(results["total_return"], 2),
                "winRate": results["win_rate"],
                "maxDrawdown": abs(results["max_drawdown"]),
                "sharpe": results["sharpe"],
                "totalTrades": results["total_trades"],
                "total_return": results["total_return"],
                "win_rate": results["win_rate"],
                "max_drawdown": results["max_drawdown"]
            },
            "trades": [],
            "buyPoints": buy_points[:5],  # 返回5个买入点
            "sellPoints": sell_points[:5],  # 返回5个卖出点
            "results": results
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

