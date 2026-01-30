#!/usr/bin/env python3
"""
预训练所有股票的预测模型
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class StockPredictor:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.model = None
        self.scaler = StandardScaler()
        
    def create_features(self, df):
        features = []
        labels = []
        dates = []
        
        df = df.copy()
        df['return'] = df['close'].pct_change()
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['volatility'] = df['return'].rolling(10).std()
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.0001)
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 0.0001)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        df = df.dropna()
        
        for i in range(self.window_size, len(df) - 5):
            window = df.iloc[i-self.window_size:i]
            
            feature = []
            feature.extend(window['return'].values[-10:])
            feature.append(window['ma5'].iloc[-1] / window['close'].iloc[-1])
            feature.append(window['ma10'].iloc[-1] / window['close'].iloc[-1])
            feature.append(window['ma20'].iloc[-1] / window['close'].iloc[-1])
            feature.append(window['volatility'].iloc[-1])
            feature.append(window['volume'].iloc[-1] / window['volume_ma5'].iloc[-1])
            feature.append(window['price_position'].iloc[-1])
            feature.append(window['rsi'].iloc[-1] / 100)
            
            features.append(feature)
            
            future_return = (df.iloc[i+5]['close'] - df.iloc[i]['close']) / df.iloc[i]['close']
            labels.append(future_return)
            dates.append(df.iloc[i]['date'])
        
        return np.array(features), np.array(labels), dates
    
    def train(self, df):
        features, labels, dates = self.create_features(df)
        
        if len(features) < 50:
            return None
        
        features_scaled = self.scaler.fit_transform(features)
        
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, test_size=0.2, shuffle=False
        )
        
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        direction_accuracy = np.mean((y_pred > 0) == (y_test > 0))
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def predict(self, df):
        if self.model is None:
            return None
        
        features, _, _ = self.create_features(df)
        if len(features) == 0:
            return None
        
        latest_feature = features[-1:] 
        latest_scaled = self.scaler.transform(latest_feature)
        
        predicted_return = self.model.predict(latest_scaled)[0]
        latest_price = df['close'].iloc[-1]
        predicted_price = latest_price * (1 + predicted_return)
        
        return {
            'current_price': latest_price,
            'predicted_return': predicted_return,
            'predicted_price': predicted_price
        }
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'window_size': self.window_size
            }, f)
    
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        predictor = cls(window_size=data['window_size'])
        predictor.model = data['model']
        predictor.scaler = data['scaler']
        return predictor

def train_all_models():
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    model_dir = os.path.join(os.path.dirname(__file__), '../models/pretrained')
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    stock_list_path = os.path.join(data_dir, "stock_list.csv")
    if not os.path.exists(stock_list_path):
        print("错误: 请先运行 download_data.py 下载股票数据")
        return
    
    stocks = pd.read_csv(stock_list_path)
    
    print("=" * 60)
    print("开始预训练股票预测模型")
    print("=" * 60)
    
    results = []
    
    for _, stock in stocks.iterrows():
        symbol = stock['symbol']
        name = stock['name']
        
        data_path = os.path.join(data_dir, f"{symbol}_processed.csv")
        if not os.path.exists(data_path):
            print(f"跳过 {name}({symbol}): 数据文件不存在")
            continue
        
        print(f"\n训练 {name}({symbol}) 模型...")
        
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        predictor = StockPredictor(window_size=20)
        metrics = predictor.train(df)
        
        if metrics is None:
            print(f"  失败: 数据不足")
            continue
        
        model_path = os.path.join(model_dir, f"{symbol}_model.pkl")
        predictor.save(model_path)
        
        prediction = predictor.predict(df)
        
        print(f"  训练样本: {metrics['train_samples']}, 测试样本: {metrics['test_samples']}")
        print(f"  MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.4f}")
        print(f"  方向准确率: {metrics['direction_accuracy']*100:.1f}%")
        if prediction:
            print(f"  当前价格: {prediction['current_price']:.2f}")
            print(f"  预测5日收益: {prediction['predicted_return']*100:.2f}%")
            print(f"  预测价格: {prediction['predicted_price']:.2f}")
        
        results.append({
            'symbol': symbol,
            'name': name,
            'mse': metrics['mse'],
            'mae': metrics['mae'],
            'r2': metrics['r2'],
            'direction_accuracy': metrics['direction_accuracy'],
            'current_price': prediction['current_price'] if prediction else None,
            'predicted_return': prediction['predicted_return'] if prediction else None,
            'predicted_price': prediction['predicted_price'] if prediction else None
        })
    
    results_df = pd.DataFrame(results)
    results_path = os.path.join(model_dir, "training_results.csv")
    results_df.to_csv(results_path, index=False)
    
    print("\n" + "=" * 60)
    print("模型训练完成!")
    print(f"训练结果已保存到: {results_path}")
    print("=" * 60)
    
    print("\n预测收益排名（前5名）:")
    top_stocks = results_df.nlargest(5, 'predicted_return')
    for _, row in top_stocks.iterrows():
        print(f"  {row['name']}({row['symbol']}): 预测收益 {row['predicted_return']*100:.2f}%")

if __name__ == "__main__":
    train_all_models()
