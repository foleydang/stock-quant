"""策略相关路由"""
from config_loader import get_db_path

from flask import Blueprint, jsonify, request
import sys
import os
import time
import pandas as pd
from functools import lru_cache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))

strategy_bp = Blueprint('strategy', __name__)

# 简单缓存（5分钟过期）
_cache = {}
_cache_expire = {}

def get_cache(key, ttl=300):
    """获取缓存"""
    if key in _cache and time.time() - _cache_expire.get(key, 0) < ttl:
        return _cache[key]
    return None

def set_cache(key, value):
    """设置缓存"""
    _cache[key] = value
    _cache_expire[key] = time.time()

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



@strategy_bp.route("/select", methods=["GET"])
def select_stocks():
    """[已退役] 旧"选股"的 predicted_return 实为 (现价-历史均价)/历史均价, 与预测无关;
    "选股"仅按数据条数取前若干只, 非真实信号。诚实口径改用 add_advisor 横截面排名:
    GET /api/advisor/scan (?refresh=1 手动重算)。"""
    return jsonify({
        'status': 'error', 'retired': True,
        'message': '该选股接口已退役 (predicted_return 实为价格偏离历史均价, 并非模型预测)。'
                   '请改用 /api/advisor/scan 获取 add_advisor 的诚实横截面排名。',
        'redirect': '/api/advisor/scan',
    }), 410


@strategy_bp.route("/lgbm_backtest/<symbol>", methods=["GET"])
def lgbm_backtest(symbol):
    """[已退役] 依赖存在数据泄漏的 LGBMBacktesterOptimized(lgb_30m 血统), 结果不可信。
    诚实盈利口径改用 add_advisor 横截面回测: GET /api/advisor/backtest。"""
    return jsonify({
        'status': 'error', 'retired': True,
        'message': '该回测接口已退役 (底层 lgb 模型有数据泄漏)。'
                   '请改用 /api/advisor/backtest 获取诚实的横截面回测口径。',
        'redirect': '/api/advisor/backtest',
    }), 410


def _lgbm_backtest_legacy(symbol):
    """[dead code, 保留仅供参考] 原泄漏回测实现, 已不服务化。"""
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "python"))
        
        from lgbm_backtest import LGBMBacktesterOptimized
        import sqlite3
        
        # 获取股票名称
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM stock_info WHERE symbol=?", (symbol,))
        row_name = cursor.fetchone()
        name = row_name[0] if row_name and row_name[0] else symbol
        
        # 检查CSV是否已存在（回测引擎会自动查找）
        cursor.execute("SELECT COUNT(*) FROM kline_30m WHERE symbol=?", (symbol,))
        count = cursor.fetchone()[0]
        conn.close()
        
        if count < 60:
            return jsonify({"status": "error", "message": "数据不足"}), 404
        
        # 运行真实回测
        backtester = LGBMBacktesterOptimized(initial_capital=500000)  # 50万初始资金，适应高价股
        stocks = [{"symbol": symbol, "name": name}]
        # 支持日期范围；默认最近3个月避免超时
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        if not start_date:
            from datetime import date, timedelta
            start_date = (date.today() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        backtester.run_backtest(stocks, start_date=start_date, end_date=end_date)
        
        # 提取买卖点
        buy_points = []
        sell_points = []
        
        for trade in backtester.trades:
            point = {
                "date": str(trade.time),
                "price": float(trade.price)
            }
            if trade.trade_type == "buy":
                buy_points.append(point)
            else:
                sell_points.append(point)
        
        # 计算结果
        final_value = backtester.daily_values[-1].get("value", backtester.initial_capital) if backtester.daily_values else 500000
        total_return = (final_value - backtester.initial_capital) / backtester.initial_capital * 100
        win_count = sum(1 for t in (backtester.trades[-20:] if len(backtester.trades) > 20 else backtester.trades) if t.trade_type == "sell" and t.profit > 0)
        loss_count = sum(1 for t in (backtester.trades[-20:] if len(backtester.trades) > 20 else backtester.trades) if t.trade_type == "sell" and t.profit <= 0)
        win_rate = win_count / (win_count + loss_count) * 100 if (win_count + loss_count) > 0 else 0
        
        # 构造portfolioValues（市值曲线）
        portfolio_values = []
        for dv in backtester.daily_values:
            portfolio_values.append({
                "date": str(dv.get("time", "")),  # 回测引擎用time字段
                "price": 0,  # 暂不记录价格
                "portfolioValue": float(dv.get("value", 0)),
                "benchmarkReturn": 0  # 简化，暂不计算基准
            })
        
        # 构造predictions（预测概率分布）
        predictions = []
        for trade in backtester.trades:
            if trade.trade_type == "buy":
                # 提取up_prob（如果有）
                up_prob = getattr(trade, 'up_prob', 0.5)
                predictions.append({
                    "date": str(trade.time),
                    "up_prob": float(up_prob)
                })
        
        return jsonify({
            "status": "success",
            "summary": {
                "profitRate": round(total_return, 2),
                "winRate": round(win_rate, 1),
                "maxDrawdown": 0,
                "sharpe": 0,
                "tradeCount": len(backtester.trades),
                "totalTrades": len(backtester.trades),
                "tradeCount": len(backtester.trades),
                "total_return": round(total_return, 2),
                "win_rate": round(win_rate, 1),
                "totalProfit": sum(float(t.profit) for t in (backtester.trades[-20:] if len(backtester.trades) > 20 else backtester.trades) if t.trade_type == "sell"),
                "holdingShares": 0,
                "avgCost": 0,
                "finalStockValue": final_value,
                "benchmarkReturn": 0,
                "excessReturn": round(total_return, 2)
            },
            "portfolioValues": portfolio_values[-100:] if portfolio_values else [],
            "predictions": predictions,
            "trades": [
                {
                    "symbol": t.symbol,
                    "name": t.stock_name,
                    "type": t.trade_type,
                    "price": float(t.price),
                    "shares": t.shares,
                    "time": str(t.time),
                    "reason": t.reason,
                    "profit": float(t.profit),
                    "up_prob": float(getattr(t, 'up_prob', 0))
                } for t in (backtester.trades[-20:] if len(backtester.trades) > 20 else backtester.trades)
            ],
            "buyPoints": buy_points,
            "sellPoints": sell_points,
            "results": {
                "total_return": round(total_return, 2),
                "win_rate": round(win_rate, 1),
                "total_trades": len(backtester.trades),
                "final_value": final_value
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


