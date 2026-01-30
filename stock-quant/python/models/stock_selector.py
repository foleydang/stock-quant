#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from data.data_handler import DataHandler
import pickle

class StockSelector:
    def __init__(self):
        self.data_handler = DataHandler()
        self.model_dir = os.path.join(os.path.dirname(__file__), '../models')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def train_selector_model(self, stock_symbols):
        """训练选股模型"""
        try:
            # 收集所有股票的数据
            all_features = []
            all_labels = []
            all_symbols = []
            
            for symbol in stock_symbols:
                # 获取股票数据
                df = self.data_handler.load_stock_data(symbol)
                if df is None:
                    df = self.data_handler.fetch_hs300_data(symbol)
                
                if df is not None and len(df) > 20:
                    # 计算特征
                    features = self.calculate_features(df)
                    if features is not None:
                        # 计算收益率作为标签
                        returns = df['close'].pct_change().dropna()
                        if len(returns) > 0:
                            # 使用平均收益率作为标签
                            avg_return = returns.mean()
                            all_features.append(features)
                            all_labels.append(avg_return)
                            all_symbols.append(symbol)
            
            if len(all_features) < 5:
                print("Not enough data to train selector model")
                return None
            
            # 转换为数组
            X = np.array(all_features)
            y = np.array(all_labels)
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test, symbols_train, symbols_test = train_test_split(
                X, y, all_symbols, test_size=0.2, random_state=42
            )
            
            # 训练模型
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # 评估模型
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
            print(f"Selector model evaluation:")
            print(f"MSE: {mse:.4f}")
            
            # 保存模型
            model_path = os.path.join(self.model_dir, 'stock_selector_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            print("Selector model trained and saved")
            return model
        except Exception as e:
            print(f"Error training selector model: {e}")
            return None
    
    def calculate_features(self, df):
        """计算股票特征"""
        try:
            features = []
            
            # 价格特征
            close_prices = df['close'].values
            features.append(np.mean(close_prices))
            features.append(np.std(close_prices))
            features.append(np.max(close_prices) - np.min(close_prices))
            
            # 收益率特征
            returns = df['close'].pct_change().dropna()
            if len(returns) > 0:
                features.append(np.mean(returns))
                features.append(np.std(returns))
                features.append(np.max(returns))
                features.append(np.min(returns))
            else:
                features.extend([0, 0, 0, 0])
            
            # 成交量特征
            volumes = df['volume'].values
            features.append(np.mean(volumes))
            features.append(np.std(volumes))
            
            # 动量特征
            if len(close_prices) >= 5:
                short_term = close_prices[-5:].mean()
                long_term = close_prices[-20:].mean()
                features.append(short_term / long_term - 1)
            else:
                features.append(0)
            
            return features
        except Exception as e:
            print(f"Error calculating features: {e}")
            return None
    
    def select_stocks(self, stock_symbols, top_n=5):
        """选择最优股票"""
        try:
            # 加载模型
            model_path = os.path.join(self.model_dir, 'stock_selector_model.pkl')
            if not os.path.exists(model_path):
                # 如果模型不存在，先训练
                model = self.train_selector_model(stock_symbols)
                if not model:
                    return self.select_stocks_fallback(stock_symbols, top_n)
            else:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            # 预测每只股票的收益率
            predictions = []
            valid_symbols = []
            
            for symbol in stock_symbols:
                # 获取股票数据
                df = self.data_handler.load_stock_data(symbol)
                if df is None:
                    df = self.data_handler.fetch_hs300_data(symbol)
                
                if df is not None and len(df) > 20:
                    # 计算特征
                    features = self.calculate_features(df)
                    if features is not None:
                        # 预测收益率
                        predicted_return = model.predict([features])[0]
                        predictions.append(predicted_return)
                        valid_symbols.append(symbol)
            
            if len(predictions) == 0:
                return self.select_stocks_fallback(stock_symbols, top_n)
            
            # 按预测收益率排序，选择前N只股票
            sorted_indices = np.argsort(predictions)[::-1]
            selected_symbols = [valid_symbols[i] for i in sorted_indices[:top_n]]
            selected_returns = [predictions[i] for i in sorted_indices[:top_n]]
            
            # 生成选股结果
            result = []
            for symbol, predicted_return in zip(selected_symbols, selected_returns):
                result.append({
                    'symbol': symbol,
                    'predicted_return': float(predicted_return),
                    'rank': result.__len__() + 1
                })
            
            print(f"Selected stocks: {selected_symbols}")
            return result
        except Exception as e:
            print(f"Error selecting stocks: {e}")
            return self.select_stocks_fallback(stock_symbols, top_n)
    
    def select_stocks_fallback(self, stock_symbols, top_n=5):
        """备选选股方法（当模型不可用时）"""
        try:
            # 简单地基于最近收益率选择股票
            returns = []
            valid_symbols = []
            
            for symbol in stock_symbols:
                # 获取股票数据
                df = self.data_handler.load_stock_data(symbol)
                if df is None:
                    df = self.data_handler.fetch_hs300_data(symbol)
                
                if df is not None and len(df) > 20:
                    # 计算最近收益率
                    recent_returns = df['close'].pct_change().tail(20)
                    avg_return = recent_returns.mean()
                    returns.append(avg_return)
                    valid_symbols.append(symbol)
            
            if len(returns) == 0:
                # 如果没有数据，随机选择
                import random
                selected = random.sample(stock_symbols, min(top_n, len(stock_symbols)))
                result = []
                for i, symbol in enumerate(selected):
                    result.append({
                        'symbol': symbol,
                        'predicted_return': 0,
                        'rank': i + 1
                    })
                return result
            
            # 按收益率排序
            sorted_indices = np.argsort(returns)[::-1]
            selected_symbols = [valid_symbols[i] for i in sorted_indices[:top_n]]
            selected_returns = [returns[i] for i in sorted_indices[:top_n]]
            
            # 生成选股结果
            result = []
            for symbol, avg_return in zip(selected_symbols, selected_returns):
                result.append({
                    'symbol': symbol,
                    'predicted_return': float(avg_return),
                    'rank': result.__len__() + 1
                })
            
            print(f"Selected stocks (fallback): {selected_symbols}")
            return result
        except Exception as e:
            print(f"Error in fallback selection: {e}")
            # 随机选择
            import random
            selected = random.sample(stock_symbols, min(top_n, len(stock_symbols)))
            result = []
            for i, symbol in enumerate(selected):
                result.append({
                    'symbol': symbol,
                    'predicted_return': 0,
                    'rank': i + 1
                })
            return result

if __name__ == "__main__":
    # 测试代码
    selector = StockSelector()
    stock_symbols = ['000001.SZ', '000002.SZ', '000008.SZ', '000009.SZ', '000010.SZ', '600000.SH', '600004.SH', '600006.SH', '600007.SH', '600008.SH']
    
    # 训练模型
    selector.train_selector_model(stock_symbols)
    
    # 选择股票
    result = selector.select_stocks(stock_symbols, 5)
    print("Selection result:")
    for item in result:
        print(f"Rank {item['rank']}: {item['symbol']} (predicted return: {item['predicted_return']:.4f})")
