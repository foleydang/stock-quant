#!/usr/bin/env python3
import os
import sys
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
# 添加CORS支持，允许所有跨域请求
CORS(app)

# Python模型路径
PYTHON_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python/models/model_runner.py')

@app.route('/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    """获取股票数据"""
    try:
        print(f"Fetching data for symbol: {symbol}")
        print(f"Python model path: {PYTHON_MODEL_PATH}")
        
        if not os.path.exists(PYTHON_MODEL_PATH):
            print(f"Python model file not found: {PYTHON_MODEL_PATH}")
            return jsonify({"error": "Python model file not found"}), 500
        
        result = subprocess.run(
            ['python3', PYTHON_MODEL_PATH, 'fetch', symbol],
            capture_output=True,
            text=True
        )
        
        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                return jsonify(data)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                return jsonify({"error": "Failed to parse JSON response"}), 500
        else:
            return jsonify({"error": "Failed to fetch stock data", "stderr": result.stderr}), 500
    except Exception as e:
        print(f"Exception: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/train/<symbol>', methods=['GET'])
def train_model(symbol):
    """训练模型"""
    try:
        model_type = request.args.get('model_type', 'rf')
        
        # 执行Python脚本训练模型
        result = subprocess.run(
            ['python3', PYTHON_MODEL_PATH, 'train', symbol, model_type],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return jsonify(data)
        else:
            return jsonify({"error": "Failed to train model", "stderr": result.stderr}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/<symbol>', methods=['GET'])
def predict_price(symbol):
    """预测价格"""
    try:
        model_type = request.args.get('model_type', 'rf')
        
        # 执行Python脚本预测价格
        result = subprocess.run(
            ['python3', PYTHON_MODEL_PATH, 'predict', symbol, model_type],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return jsonify(data)
        else:
            return jsonify({"error": "Failed to predict price", "stderr": result.stderr}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def calc_rsi(prices, period=14):
    """计算RSI指标"""
    if len(prices) < period + 1:
        return 50
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_macd(prices, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    if len(prices) < slow:
        return 0, 0, 0
    
    def ema(data, period):
        if len(data) < period:
            return data[-1] if data else 0
        multiplier = 2 / (period + 1)
        ema_val = sum(data[:period]) / period
        for price in data[period:]:
            ema_val = (price - ema_val) * multiplier + ema_val
        return ema_val
    
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow
    
    if len(prices) >= slow + signal:
        macd_values = []
        for i in range(slow, len(prices) + 1):
            ef = ema(prices[:i], fast)
            es = ema(prices[:i], slow)
            macd_values.append(ef - es)
        signal_line = ema(macd_values, signal) if len(macd_values) >= signal else macd_line
    else:
        signal_line = macd_line
    
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

@app.route('/strategy/<symbol>', methods=['GET'])
def run_strategy(symbol):
    """执行交易策略 - 多指标综合策略"""
    try:
        from datetime import datetime, timedelta
        
        result = subprocess.run(
            ['python3', PYTHON_MODEL_PATH, 'fetch', symbol],
            capture_output=True,
            text=True
        )
        
        stock_data = []
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                if data.get('status') == 'success' and data.get('data'):
                    stock_data = data['data']
            except:
                pass
        
        if not stock_data or len(stock_data) < 30:
            return jsonify({"error": "无法获取足够的股票数据，需要至少30天数据"}), 500
        
        initial_cash = 100000.00
        cash = initial_cash
        holding_shares = 0
        avg_cost = 0
        max_portfolio_value = initial_cash
        
        strategy_results = []
        trades = []
        buy_points = []
        sell_points = []
        price_data = []
        
        ma5_values = []
        ma20_values = []
        rsi_values = []
        
        prices_history = []
        
        for i, day_data in enumerate(stock_data):
            date = day_data['date']
            price = round(day_data['close'], 2)
            prices_history.append(day_data['close'])
            
            price_data.append({
                "date": date,
                "price": price,
                "open": round(day_data['open'], 2),
                "high": round(day_data['high'], 2),
                "low": round(day_data['low'], 2)
            })
            
            ma5 = sum(prices_history[-5:]) / min(5, len(prices_history)) if prices_history else price
            ma20 = sum(prices_history[-20:]) / min(20, len(prices_history)) if prices_history else price
            ma5_values.append({"date": date, "value": round(ma5, 2)})
            ma20_values.append({"date": date, "value": round(ma20, 2)})
            
            rsi = calc_rsi(prices_history)
            rsi_values.append({"date": date, "value": round(rsi, 2)})
            
            macd_line, signal_line, histogram = calc_macd(prices_history)
            
            trade_type = 'HOLD'
            shares = 0
            amount = 0.00
            
            buy_signals = 0
            sell_signals = 0
            
            if i >= 14:
                if rsi < 35:
                    buy_signals += 2
                elif rsi < 45:
                    buy_signals += 1
                
                if rsi > 65:
                    sell_signals += 2
                elif rsi > 55:
                    sell_signals += 1
                
                if histogram > 0:
                    buy_signals += 1
                    if i > 0:
                        prev_macd, prev_signal, prev_hist = calc_macd(prices_history[:-1])
                        if prev_hist <= 0:
                            buy_signals += 1
                
                if histogram < 0:
                    sell_signals += 1
                    if i > 0:
                        prev_macd, prev_signal, prev_hist = calc_macd(prices_history[:-1])
                        if prev_hist >= 0:
                            sell_signals += 1
                
                if ma5 > ma20:
                    buy_signals += 1
                    if len(ma5_values) > 1:
                        prev_ma5 = ma5_values[-2]['value']
                        prev_ma20 = ma20_values[-2]['value']
                        if prev_ma5 <= prev_ma20:
                            buy_signals += 1
                
                if ma5 < ma20:
                    sell_signals += 1
                    if len(ma5_values) > 1:
                        prev_ma5 = ma5_values[-2]['value']
                        prev_ma20 = ma20_values[-2]['value']
                        if prev_ma5 >= prev_ma20:
                            sell_signals += 1
                
                if price < ma20 * 0.97:
                    buy_signals += 1
                
                if price > ma20 * 1.03:
                    sell_signals += 1
            
            if holding_shares > 0 and avg_cost > 0:
                profit_rate = (price - avg_cost) / avg_cost
                
                if profit_rate >= 0.05:
                    sell_signals += 2
                elif profit_rate >= 0.03:
                    sell_signals += 1
                
                if profit_rate <= -0.03:
                    sell_signals += 2
                elif profit_rate <= -0.02:
                    sell_signals += 1
            
            current_value = cash + holding_shares * price
            if current_value > max_portfolio_value:
                max_portfolio_value = current_value
            drawdown = (max_portfolio_value - current_value) / max_portfolio_value
            if drawdown > 0.08 and holding_shares > 0:
                sell_signals += 2
            
            should_buy = buy_signals >= 2 and holding_shares == 0
            should_sell = sell_signals >= 2 and holding_shares > 0
            
            if should_buy and cash >= price * 100:
                position_ratio = 0.3 if buy_signals >= 5 else 0.5 if buy_signals >= 4 else 0.7
                invest_amount = cash * position_ratio
                max_lots = int(invest_amount / (price * 100))
                lots = max(1, min(max_lots, 20))
                
                shares = lots * 100
                amount = round(price * shares, 2)
                
                if amount <= cash:
                    trade_type = 'BUY'
                    cash = round(cash - amount, 2)
                    total_cost = avg_cost * holding_shares + amount
                    holding_shares += shares
                    avg_cost = total_cost / holding_shares
                    buy_points.append({
                        "date": date,
                        "price": price,
                        "shares": shares,
                        "amount": amount,
                        "signal_strength": buy_signals
                    })
            
            elif should_sell and holding_shares >= 100:
                if sell_signals >= 5:
                    sell_ratio = 1.0
                elif sell_signals >= 4:
                    sell_ratio = 0.7
                else:
                    sell_ratio = 0.5
                
                lots = int((holding_shares * sell_ratio) // 100)
                lots = max(1, lots)
                shares = lots * 100
                
                if shares <= holding_shares:
                    amount = round(price * shares, 2)
                    trade_type = 'SELL'
                    cash = round(cash + amount, 2)
                    holding_shares -= shares
                    if holding_shares == 0:
                        avg_cost = 0
                    sell_points.append({
                        "date": date,
                        "price": price,
                        "shares": shares,
                        "amount": amount,
                        "signal_strength": sell_signals
                    })
            
            if trade_type != 'HOLD':
                trades.append({
                    "date": date,
                    "symbol": symbol,
                    "type": trade_type,
                    "price": price,
                    "shares": shares,
                    "amount": amount
                })
            
            stock_value = round(holding_shares * price, 2)
            portfolio_value = round(cash + stock_value, 2)
            
            strategy_results.append({
                "date": date,
                "price": price,
                "portfolioValue": portfolio_value,
                "cash": cash,
                "stockValue": stock_value,
                "holdingShares": holding_shares,
                "rsi": rsi_values[-1]['value'] if rsi_values else 50,
                "ma5": ma5_values[-1]['value'] if ma5_values else price,
                "ma20": ma20_values[-1]['value'] if ma20_values else price
            })
        
        final_price = stock_data[-1]['close'] if stock_data else 0
        final_stock_value = round(holding_shares * final_price, 2)
        final_portfolio_value = round(cash + final_stock_value, 2)
        
        total_buy = sum([bp['amount'] for bp in buy_points])
        total_sell = sum([sp['amount'] for sp in sell_points])
        win_trades = len([sp for sp in sell_points if any(bp['date'] < sp['date'] and sp['price'] > bp['price'] for bp in buy_points)])
        
        return jsonify({
            "status": "success",
            "strategyResults": strategy_results,
            "trades": trades,
            "priceData": price_data,
            "buyPoints": buy_points,
            "sellPoints": sell_points,
            "ma5": ma5_values,
            "ma20": ma20_values,
            "rsi": rsi_values,
            "finalPortfolio": {
                "cash": cash,
                "stockValue": final_stock_value,
                "totalValue": final_portfolio_value,
                "holdingShares": holding_shares,
                "avgCost": round(avg_cost, 2),
                "profit": round(final_portfolio_value - initial_cash, 2),
                "profitRate": round((final_portfolio_value - initial_cash) / initial_cash * 100, 2),
                "totalBuyAmount": round(total_buy, 2),
                "totalSellAmount": round(total_sell, 2),
                "tradeCount": len(trades),
                "winRate": round(win_trades / max(1, len(sell_points)) * 100, 2)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

HS300_STOCKS = [
    {"symbol": "000001.SZ", "name": "平安银行"},
    {"symbol": "000002.SZ", "name": "万科A"},
    {"symbol": "000063.SZ", "name": "中兴通讯"},
    {"symbol": "000333.SZ", "name": "美的集团"},
    {"symbol": "000651.SZ", "name": "格力电器"},
    {"symbol": "000858.SZ", "name": "五粮液"},
    {"symbol": "002415.SZ", "name": "海康威视"},
    {"symbol": "002594.SZ", "name": "比亚迪"},
    {"symbol": "300750.SZ", "name": "宁德时代"},
    {"symbol": "600000.SH", "name": "浦发银行"},
    {"symbol": "600036.SH", "name": "招商银行"},
    {"symbol": "600276.SH", "name": "恒瑞医药"},
    {"symbol": "600519.SH", "name": "贵州茅台"},
    {"symbol": "600887.SH", "name": "伊利股份"},
    {"symbol": "601318.SH", "name": "中国平安"},
    {"symbol": "601398.SH", "name": "工商银行"},
    {"symbol": "601888.SH", "name": "中国中免"},
    {"symbol": "600030.SH", "name": "中信证券"},
    {"symbol": "600050.SH", "name": "中国联通"},
    {"symbol": "601166.SH", "name": "兴业银行"},
]

@app.route('/stocks', methods=['GET'])
def get_stocks():
    """获取股票列表"""
    stocks = [{"value": s["symbol"], "label": s["name"]} for s in HS300_STOCKS]
    return jsonify({"stocks": stocks})

@app.route('/select', methods=['GET'])
def select_stocks():
    """基于预训练模型的智能选股"""
    try:
        import pickle
        import pandas as pd
        
        model_dir = os.path.join(os.path.dirname(__file__), '../python/models/pretrained')
        data_dir = os.path.join(os.path.dirname(__file__), '../python/data')
        
        results_path = os.path.join(model_dir, "training_results.csv")
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path)
            results_df = results_df.dropna(subset=['predicted_return'])
            results_df = results_df.sort_values('predicted_return', ascending=False)
            
            selected_stocks = []
            for rank, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
                selected_stocks.append({
                    "symbol": row['symbol'],
                    "name": row['name'],
                    "current_price": round(row['current_price'], 2),
                    "predicted_return": round(row['predicted_return'] * 100, 2),
                    "predicted_price": round(row['predicted_price'], 2),
                    "direction_accuracy": round(row['direction_accuracy'] * 100, 1),
                    "rank": rank,
                    "recommendation": "强烈买入" if row['predicted_return'] > 0.03 else "买入" if row['predicted_return'] > 0.01 else "观望"
                })
            
            return jsonify({
                "status": "success",
                "selected_stocks": selected_stocks,
                "model_type": "pretrained",
                "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        predictions = []
        for stock in HS300_STOCKS:
            symbol = stock['symbol']
            model_path = os.path.join(model_dir, f"{symbol}_model.pkl")
            data_path = os.path.join(data_dir, f"{symbol}_processed.csv")
            
            if not os.path.exists(model_path) or not os.path.exists(data_path):
                continue
            
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                df = pd.read_csv(data_path)
                df['date'] = pd.to_datetime(df['date'])
                
                df_copy = df.copy()
                df_copy['return'] = df_copy['close'].pct_change()
                df_copy['ma5'] = df_copy['close'].rolling(5).mean()
                df_copy['ma10'] = df_copy['close'].rolling(10).mean()
                df_copy['ma20'] = df_copy['close'].rolling(20).mean()
                df_copy['volatility'] = df_copy['return'].rolling(10).std()
                df_copy['volume_ma5'] = df_copy['volume'].rolling(5).mean()
                df_copy['price_position'] = (df_copy['close'] - df_copy['low']) / (df_copy['high'] - df_copy['low'] + 0.0001)
                
                delta = df_copy['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / (loss + 0.0001)
                df_copy['rsi'] = 100 - (100 / (1 + rs))
                df_copy = df_copy.dropna()
                
                if len(df_copy) < 20:
                    continue
                
                window = df_copy.iloc[-20:]
                feature = []
                feature.extend(window['return'].values[-10:])
                feature.append(window['ma5'].iloc[-1] / window['close'].iloc[-1])
                feature.append(window['ma10'].iloc[-1] / window['close'].iloc[-1])
                feature.append(window['ma20'].iloc[-1] / window['close'].iloc[-1])
                feature.append(window['volatility'].iloc[-1])
                feature.append(window['volume'].iloc[-1] / window['volume_ma5'].iloc[-1])
                feature.append(window['price_position'].iloc[-1])
                feature.append(window['rsi'].iloc[-1] / 100)
                
                import numpy as np
                feature = np.array(feature).reshape(1, -1)
                feature_scaled = model_data['scaler'].transform(feature)
                predicted_return = model_data['model'].predict(feature_scaled)[0]
                
                current_price = df['close'].iloc[-1]
                predicted_price = current_price * (1 + predicted_return)
                
                predictions.append({
                    "symbol": symbol,
                    "name": stock['name'],
                    "current_price": round(current_price, 2),
                    "predicted_return": round(predicted_return * 100, 2),
                    "predicted_price": round(predicted_price, 2),
                })
            except Exception as e:
                continue
        
        predictions.sort(key=lambda x: x['predicted_return'], reverse=True)
        
        selected_stocks = []
        for rank, pred in enumerate(predictions[:10], 1):
            pred['rank'] = rank
            pred['recommendation'] = "强烈买入" if pred['predicted_return'] > 3 else "买入" if pred['predicted_return'] > 1 else "观望"
            selected_stocks.append(pred)
        
        return jsonify({
            "status": "success",
            "selected_stocks": selected_stocks,
            "model_type": "realtime",
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_strategy/<symbol>', methods=['GET'])
def predict_strategy(symbol):
    """基于预测的交易策略"""
    try:
        import pickle
        import pandas as pd
        import numpy as np
        
        model_dir = os.path.join(os.path.dirname(__file__), '../python/models/pretrained')
        data_dir = os.path.join(os.path.dirname(__file__), '../python/data')
        
        model_path = os.path.join(model_dir, f"{symbol}_model.pkl")
        data_path = os.path.join(data_dir, f"{symbol}_processed.csv")
        
        if not os.path.exists(data_path):
            return jsonify({"error": f"股票数据不存在: {symbol}"}), 404
        
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        has_model = os.path.exists(model_path)
        model_data = None
        if has_model:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
        
        initial_cash = 100000.00
        cash = initial_cash
        holding_shares = 0
        avg_cost = 0
        
        strategy_results = []
        trades = []
        buy_points = []
        sell_points = []
        predictions_list = []
        
        df_features = df.copy()
        df_features['return'] = df_features['close'].pct_change()
        df_features['ma5'] = df_features['close'].rolling(5).mean()
        df_features['ma10'] = df_features['close'].rolling(10).mean()
        df_features['ma20'] = df_features['close'].rolling(20).mean()
        df_features['volatility'] = df_features['return'].rolling(10).std()
        df_features['volume_ma5'] = df_features['volume'].rolling(5).mean()
        df_features['price_position'] = (df_features['close'] - df_features['low']) / (df_features['high'] - df_features['low'] + 0.0001)
        
        delta = df_features['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 0.0001)
        df_features['rsi'] = 100 - (100 / (1 + rs))
        
        for i in range(30, len(df)):
            date = df.iloc[i]['date'].strftime('%Y-%m-%d')
            price = round(df.iloc[i]['close'], 2)
            
            predicted_return = 0
            if has_model and i >= 30:
                try:
                    window = df_features.iloc[i-20:i]
                    if len(window.dropna()) >= 15:
                        feature = []
                        feature.extend(window['return'].dropna().values[-10:])
                        if len(feature) < 10:
                            feature = [0] * (10 - len(feature)) + feature
                        feature.append(window['ma5'].iloc[-1] / window['close'].iloc[-1])
                        feature.append(window['ma10'].iloc[-1] / window['close'].iloc[-1])
                        feature.append(window['ma20'].iloc[-1] / window['close'].iloc[-1])
                        feature.append(window['volatility'].iloc[-1] if not pd.isna(window['volatility'].iloc[-1]) else 0)
                        feature.append(window['volume'].iloc[-1] / window['volume_ma5'].iloc[-1] if window['volume_ma5'].iloc[-1] > 0 else 1)
                        feature.append(window['price_position'].iloc[-1])
                        feature.append(window['rsi'].iloc[-1] / 100 if not pd.isna(window['rsi'].iloc[-1]) else 0.5)
                        
                        feature = np.array(feature).reshape(1, -1)
                        feature_scaled = model_data['scaler'].transform(feature)
                        predicted_return = model_data['model'].predict(feature_scaled)[0]
                except:
                    predicted_return = 0
            
            predicted_price = round(price * (1 + predicted_return), 2)
            predictions_list.append({
                "date": date,
                "predicted_return": round(predicted_return * 100, 2),
                "predicted_price": predicted_price
            })
            
            trade_type = 'HOLD'
            shares = 0
            amount = 0.00
            
            should_buy = predicted_return > 0.02 and holding_shares == 0
            should_sell = (predicted_return < -0.01 or (holding_shares > 0 and avg_cost > 0 and (price - avg_cost) / avg_cost > 0.05)) and holding_shares > 0
            
            if should_buy and cash >= price * 100:
                position_ratio = 0.5 if predicted_return > 0.03 else 0.3
                invest_amount = cash * position_ratio
                lots = max(1, int(invest_amount / (price * 100)))
                shares = lots * 100
                amount = round(price * shares, 2)
                
                if amount <= cash:
                    trade_type = 'BUY'
                    cash = round(cash - amount, 2)
                    total_cost = avg_cost * holding_shares + amount
                    holding_shares += shares
                    avg_cost = total_cost / holding_shares
                    buy_points.append({
                        "date": date,
                        "price": price,
                        "shares": shares,
                        "amount": amount,
                        "predicted_return": round(predicted_return * 100, 2)
                    })
                    trades.append({
                        "date": date,
                        "symbol": symbol,
                        "type": "BUY",
                        "price": price,
                        "shares": shares,
                        "amount": amount
                    })
            
            elif should_sell and holding_shares >= 100:
                lots = holding_shares // 100
                shares = lots * 100
                amount = round(price * shares, 2)
                trade_type = 'SELL'
                cash = round(cash + amount, 2)
                holding_shares -= shares
                if holding_shares == 0:
                    avg_cost = 0
                sell_points.append({
                    "date": date,
                    "price": price,
                    "shares": shares,
                    "amount": amount,
                    "predicted_return": round(predicted_return * 100, 2)
                })
                trades.append({
                    "date": date,
                    "symbol": symbol,
                    "type": "SELL",
                    "price": price,
                    "shares": shares,
                    "amount": amount
                })
            
            stock_value = round(holding_shares * price, 2)
            portfolio_value = round(cash + stock_value, 2)
            
            strategy_results.append({
                "date": date,
                "price": price,
                "predicted_price": predicted_price,
                "predicted_return": round(predicted_return * 100, 2),
                "portfolioValue": portfolio_value,
                "cash": cash,
                "stockValue": stock_value,
                "holdingShares": holding_shares
            })
        
        final_price = df['close'].iloc[-1]
        final_stock_value = round(holding_shares * final_price, 2)
        final_portfolio_value = round(cash + final_stock_value, 2)
        
        return jsonify({
            "status": "success",
            "strategyResults": strategy_results,
            "trades": trades,
            "buyPoints": buy_points,
            "sellPoints": sell_points,
            "predictions": predictions_list,
            "hasModel": has_model,
            "finalPortfolio": {
                "cash": cash,
                "stockValue": final_stock_value,
                "totalValue": final_portfolio_value,
                "holdingShares": holding_shares,
                "avgCost": round(avg_cost, 2),
                "profit": round(final_portfolio_value - initial_cash, 2),
                "profitRate": round((final_portfolio_value - initial_cash) / initial_cash * 100, 2),
                "tradeCount": len(trades)
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/portfolio_strategy', methods=['GET'])
def portfolio_strategy():
    """综合选股策略 - 最多持有5只股票"""
    try:
        import pickle
        import pandas as pd
        import numpy as np
        from datetime import datetime
        
        model_dir = os.path.join(os.path.dirname(__file__), '../python/models/pretrained')
        data_dir = os.path.join(os.path.dirname(__file__), '../python/data')
        
        # 1. 获取选股结果
        select_result = select_stocks()
        if select_result.status_code != 200:
            return jsonify({"error": "选股失败"}), 500
        
        select_data = select_result.get_json()
        if select_data.get('status') != 'success':
            return jsonify({"error": "选股失败"}), 500
        
        selected_stocks = select_data.get('selected_stocks', [])
        if not selected_stocks:
            return jsonify({"error": "未选出符合条件的股票"}), 500
        
        # 2. 初始化投资组合
        initial_cash = 100000.00
        cash = initial_cash
        max_holdings = 5
        current_holdings = []
        
        portfolio_history = []
        trades = []
        portfolio_value_curve = []
        
        # 3. 构建股票数据字典
        stock_data_dict = {}
        model_dict = {}
        
        for stock in selected_stocks:
            symbol = stock['symbol']
            data_path = os.path.join(data_dir, f"{symbol}_processed.csv")
            model_path = os.path.join(model_dir, f"{symbol}_model.pkl")
            
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                df['date'] = pd.to_datetime(df['date'])
                stock_data_dict[symbol] = df
                
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        model_dict[symbol] = pickle.load(f)
        
        if not stock_data_dict:
            return jsonify({"error": "股票数据加载失败"}), 500
        
        # 4. 获取所有股票的日期范围
        all_dates = set()
        for symbol, df in stock_data_dict.items():
            dates = df['date'].dt.strftime('%Y-%m-%d').tolist()
            all_dates.update(dates)
        
        all_dates = sorted(all_dates)
        start_date = all_dates[0]
        end_date = all_dates[-1]
        
        # 5. 按日期执行策略
        for date_str in all_dates:
            portfolio_value = cash
            date_holdings = []
            
            # 计算当前持仓价值
            for holding in current_holdings:
                symbol = holding['symbol']
                shares = holding['shares']
                avg_cost = holding['avg_cost']
                
                if symbol in stock_data_dict:
                    df = stock_data_dict[symbol]
                    day_data = df[df['date'].dt.strftime('%Y-%m-%d') == date_str]
                    if not day_data.empty:
                        current_price = day_data.iloc[0]['close']
                        stock_value = shares * current_price
                        portfolio_value += stock_value
                        date_holdings.append({
                            "symbol": symbol,
                            "name": next((s['name'] for s in selected_stocks if s['symbol'] == symbol), symbol),
                            "shares": shares,
                            "avg_cost": avg_cost,
                            "current_price": current_price,
                            "value": stock_value,
                            "profit": (current_price - avg_cost) * shares,
                            "profit_rate": (current_price - avg_cost) / avg_cost * 100 if avg_cost > 0 else 0
                        })
            
            # 记录价值曲线
            portfolio_value_curve.append({
                "date": date_str,
                "portfolio_value": round(portfolio_value, 2),
                "cash": round(cash, 2),
                "holding_count": len(current_holdings)
            })
            
            # 检查卖出信号
            sell_symbols = []
            for i, holding in enumerate(current_holdings):
                symbol = holding['symbol']
                shares = holding['shares']
                avg_cost = holding['avg_cost']
                
                if symbol in stock_data_dict and symbol in model_dict:
                    df = stock_data_dict[symbol]
                    day_data = df[df['date'].dt.strftime('%Y-%m-%d') == date_str]
                    if not day_data.empty:
                        current_price = day_data.iloc[0]['close']
                        
                        # 计算技术指标
                        df_features = df.copy()
                        df_features['return'] = df_features['close'].pct_change()
                        df_features['ma5'] = df_features['close'].rolling(5).mean()
                        df_features['ma10'] = df_features['close'].rolling(10).mean()
                        df_features['ma20'] = df_features['close'].rolling(20).mean()
                        df_features['volatility'] = df_features['return'].rolling(10).std()
                        df_features['volume_ma5'] = df_features['volume'].rolling(5).mean()
                        df_features['price_position'] = (df_features['close'] - df_features['low']) / (df_features['high'] - df_features['low'] + 0.0001)
                        
                        delta = df_features['close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                        rs = gain / (loss + 0.0001)
                        df_features['rsi'] = 100 - (100 / (1 + rs))
                        
                        # 生成预测
                        try:
                            date_idx = df[df['date'].dt.strftime('%Y-%m-%d') == date_str].index
                            if len(date_idx) > 0 and date_idx[0] >= 30:
                                i = date_idx[0]
                                window = df_features.iloc[i-20:i]
                                if len(window.dropna()) >= 15:
                                    feature = []
                                    feature.extend(window['return'].dropna().values[-10:])
                                    if len(feature) < 10:
                                        feature = [0] * (10 - len(feature)) + feature
                                    feature.append(window['ma5'].iloc[-1] / window['close'].iloc[-1])
                                    feature.append(window['ma10'].iloc[-1] / window['close'].iloc[-1])
                                    feature.append(window['ma20'].iloc[-1] / window['close'].iloc[-1])
                                    feature.append(window['volatility'].iloc[-1] if not pd.isna(window['volatility'].iloc[-1]) else 0)
                                    feature.append(window['volume'].iloc[-1] / window['volume_ma5'].iloc[-1] if window['volume_ma5'].iloc[-1] > 0 else 1)
                                    feature.append(window['price_position'].iloc[-1])
                                    feature.append(window['rsi'].iloc[-1] / 100 if not pd.isna(window['rsi'].iloc[-1]) else 0.5)
                                    
                                    feature = np.array(feature).reshape(1, -1)
                                    feature_scaled = model_dict[symbol]['scaler'].transform(feature)
                                    predicted_return = model_dict[symbol]['model'].predict(feature_scaled)[0]
                                    
                                    # 卖出信号
                                    profit_rate = (current_price - avg_cost) / avg_cost if avg_cost > 0 else 0
                                    
                                    if (predicted_return < -0.01 or 
                                        profit_rate > 0.05 or 
                                        profit_rate < -0.03):
                                        sell_symbols.append(symbol)
                        except:
                            pass
            
            # 执行卖出操作
            for symbol in sell_symbols:
                holding_idx = next((i for i, h in enumerate(current_holdings) if h['symbol'] == symbol), -1)
                if holding_idx >= 0:
                    holding = current_holdings[holding_idx]
                    shares = holding['shares']
                    
                    df = stock_data_dict[symbol]
                    day_data = df[df['date'].dt.strftime('%Y-%m-%d') == date_str]
                    if not day_data.empty:
                        current_price = day_data.iloc[0]['close']
                        sell_amount = current_price * shares
                        
                        cash += sell_amount
                        trades.append({
                            "date": date_str,
                            "symbol": symbol,
                            "type": "SELL",
                            "price": current_price,
                            "shares": shares,
                            "amount": sell_amount
                        })
                        
                        current_holdings.pop(holding_idx)
            
            # 执行买入操作（如果持仓不足5只）
            if len(current_holdings) < max_holdings and cash > 1000:
                # 按预测收益排序股票
                buy_candidates = []
                for stock in selected_stocks:
                    symbol = stock['symbol']
                    if symbol not in [h['symbol'] for h in current_holdings] and symbol in stock_data_dict:
                        # 注意：选股API返回的predicted_return已经是百分比值，需要转换为小数
                        predicted_return = stock.get('predicted_return', 0) / 100
                        buy_candidates.append((symbol, predicted_return, stock))
                
                # 按预测收益排序
                buy_candidates.sort(key=lambda x: x[1], reverse=True)
                
                # 买入前N只股票
                for symbol, predicted_return, stock_info in buy_candidates:
                    if len(current_holdings) >= max_holdings:
                        break
                    
                    # 降低买入阈值以确保能触发交易
                    if predicted_return > 0.01:
                        df = stock_data_dict[symbol]
                        day_data = df[df['date'].dt.strftime('%Y-%m-%d') == date_str]
                        if not day_data.empty:
                            current_price = day_data.iloc[0]['close']
                            
                            # 计算买入金额 - 简化计算，确保能买入
                            position_ratio = 0.5 if predicted_return > 0.02 else 0.3
                            invest_amount = cash * position_ratio
                            
                            if invest_amount >= current_price * 100:
                                shares = int(invest_amount / (current_price * 100)) * 100
                                if shares >= 100:
                                    buy_amount = current_price * shares
                                    
                                    if buy_amount <= cash:
                                        cash -= buy_amount
                                        current_holdings.append({
                                            "symbol": symbol,
                                            "shares": shares,
                                            "avg_cost": current_price,
                                            "buy_date": date_str
                                        })
                                        
                                        trades.append({
                                            "date": date_str,
                                            "symbol": symbol,
                                            "type": "BUY",
                                            "price": current_price,
                                            "shares": shares,
                                            "amount": buy_amount
                                        })
            
            # 记录投资组合历史
            portfolio_history.append({
                "date": date_str,
                "portfolio_value": round(portfolio_value, 2),
                "cash": round(cash, 2),
                "holdings": date_holdings,
                "holding_count": len(current_holdings)
            })
        
        # 6. 计算最终结果
        final_value = cash
        for holding in current_holdings:
            symbol = holding['symbol']
            shares = holding['shares']
            
            if symbol in stock_data_dict:
                df = stock_data_dict[symbol]
                last_price = df.iloc[-1]['close']
                final_value += shares * last_price
        
        total_profit = final_value - initial_cash
        total_profit_rate = (total_profit / initial_cash) * 100
        
        # 7. 生成价值曲线数据
        value_curve = []
        for item in portfolio_value_curve:
            value_curve.append({
                "date": item['date'],
                "value": item['portfolio_value'],
                "cash": item['cash'],
                "holdings": item['holding_count']
            })
        
        # 8. 生成买入卖出点
        buy_points = [t for t in trades if t['type'] == 'BUY']
        sell_points = [t for t in trades if t['type'] == 'SELL']
        
        return jsonify({
            "status": "success",
            "portfolio_history": portfolio_history,
            "trades": trades,
            "value_curve": value_curve,
            "buy_points": buy_points,
            "sell_points": sell_points,
            "final_portfolio": {
                "cash": round(cash, 2),
                "total_value": round(final_value, 2),
                "profit": round(total_profit, 2),
                "profit_rate": round(total_profit_rate, 2),
                "holding_count": len(current_holdings),
                "current_holdings": current_holdings
            },
            "selected_stocks": selected_stocks,
            "strategy_params": {
                "max_holdings": max_holdings,
                "initial_cash": initial_cash,
                "date_range": {
                    "start": start_date,
                    "end": end_date
                }
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)