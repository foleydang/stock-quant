import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

class DataHandler:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), '../data')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def _convert_symbol_to_yahoo(self, symbol):
        if symbol.endswith('.SZ'):
            return symbol[:6] + '.SZ'
        elif symbol.endswith('.SH'):
            return symbol[:6] + '.SS'
        return symbol
    
    def fetch_stock_data(self, symbol, days=365):
        try:
            processed_file_path = os.path.join(self.data_dir, f"{symbol}_processed.csv")
            
            if os.path.exists(processed_file_path):
                file_mod_time = datetime.fromtimestamp(os.path.getmtime(processed_file_path))
                if datetime.now() - file_mod_time < timedelta(hours=12):
                    df = pd.read_csv(processed_file_path)
                    df['date'] = pd.to_datetime(df['date'])
                    return df
            
            yahoo_symbol = self._convert_symbol_to_yahoo(symbol)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                if os.path.exists(processed_file_path):
                    df = pd.read_csv(processed_file_path)
                    df['date'] = pd.to_datetime(df['date'])
                    return df
                return None
            
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            df = df.sort_values('date')
            df = df.reset_index(drop=True)
            
            df.to_csv(processed_file_path, index=False)
            
            return df
        except Exception as e:
            processed_file_path = os.path.join(self.data_dir, f"{symbol}_processed.csv")
            if os.path.exists(processed_file_path):
                df = pd.read_csv(processed_file_path)
                df['date'] = pd.to_datetime(df['date'])
                return df
            sys.stderr.write(f"Error fetching data for {symbol}: {e}\n")
            return None
    
    def fetch_hs300_data(self, symbol, days=365):
        return self.fetch_stock_data(symbol, days)
    
    def load_stock_data(self, symbol):
        try:
            file_path = os.path.join(self.data_dir, f"{symbol}_processed.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'])
                return df
            else:
                return self.fetch_stock_data(symbol)
        except Exception as e:
            sys.stderr.write(f"Error loading data: {e}\n")
            return None
    
    def preprocess_data(self, df, window_size=10):
        try:
            if df is None or len(df) < window_size + 1:
                return None
            
            df['return'] = df['close'].pct_change()
            df = df.dropna()
            
            features = []
            labels = []
            
            for i in range(window_size, len(df)):
                window = df.iloc[i-window_size:i]
                
                price_features = window[['close', 'open', 'high', 'low']].values.flatten()
                volume_features = window['volume'].values.flatten()
                return_features = window['return'].values.flatten()
                
                feature = np.concatenate([price_features, volume_features, return_features])
                features.append(feature)
                
                labels.append(df.iloc[i]['return'])
            
            features = np.array(features)
            labels = np.array(labels)
            
            mean = features.mean(axis=0)
            std = features.std(axis=0)
            
            std[std == 0] = 1
            
            features = (features - mean) / std
            
            return {
                'features': features,
                'labels': labels,
                'mean': mean,
                'std': std,
                'close_prices': df['close'].values
            }
        except Exception as e:
            sys.stderr.write(f"Error preprocessing data: {e}\n")
            return None
    
    def get_available_symbols(self):
        return [
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
        ]
