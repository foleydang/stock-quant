import os
import sys
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from data.data_handler import DataHandler

class ModelTrainer:
    def __init__(self):
        self.data_handler = DataHandler()
        self.model_dir = os.path.join(os.path.dirname(__file__), '../models')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def train_model(self, symbol, model_type='rf'):
        try:
            df = self.data_handler.load_stock_data(symbol)
            if df is None:
                df = self.data_handler.fetch_hs300_data(symbol)
                if df is None:
                    sys.stderr.write(f"Failed to get data for {symbol}\n")
                    return None
            
            preprocessed_data = self.data_handler.preprocess_data(df, 10)
            if preprocessed_data is None:
                sys.stderr.write(f"Failed to preprocess data for {symbol}\n")
                return None
            
            features = preprocessed_data['features']
            labels = preprocessed_data['labels']
            mean = preprocessed_data['mean']
            std = preprocessed_data['std']
            
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )
            
            if model_type == 'lr':
                model = LinearRegression()
            elif model_type == 'rf':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type == 'gb':
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            else:
                sys.stderr.write(f"Unknown model type: {model_type}\n")
                return None
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            model_path = os.path.join(self.model_dir, f"{symbol}_{model_type}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            params_path = os.path.join(self.model_dir, f"{symbol}_{model_type}_params.pkl")
            params = {
                'mean': mean,
                'std': std,
                'model_type': model_type
            }
            with open(params_path, 'wb') as f:
                pickle.dump(params, f)
            
            return {
                'model': model,
                'params': params,
                'metrics': {'mse': mse, 'r2': r2}
            }
        except Exception as e:
            sys.stderr.write(f"Error training model: {e}\n")
            return None
    
    def load_model(self, symbol, model_type='rf'):
        try:
            model_path = os.path.join(self.model_dir, f"{symbol}_{model_type}_model.pkl")
            params_path = os.path.join(self.model_dir, f"{symbol}_{model_type}_params.pkl")
            
            if os.path.exists(model_path) and os.path.exists(params_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                with open(params_path, 'rb') as f:
                    params = pickle.load(f)
                
                return {'model': model, 'params': params}
            else:
                return None
        except Exception as e:
            sys.stderr.write(f"Error loading model: {e}\n")
            return None
    
    def predict(self, symbol, model_type='rf'):
        try:
            model_data = self.load_model(symbol, model_type)
            if not model_data:
                model_data = self.train_model(symbol, model_type)
                if not model_data:
                    return None
            
            model = model_data['model']
            params = model_data['params']
            
            df = self.data_handler.load_stock_data(symbol)
            if df is None:
                df = self.data_handler.fetch_hs300_data(symbol)
                if df is None:
                    return None
            
            preprocessed_data = self.data_handler.preprocess_data(df, 10)
            if not preprocessed_data:
                return None
            
            latest_features = preprocessed_data['features'][-1:]
            
            prediction = model.predict(latest_features)[0]
            
            latest_price = preprocessed_data['close_prices'][-1]
            
            predicted_price = latest_price * (1 + prediction)
            
            return {
                'predicted_return': prediction,
                'predicted_price': predicted_price,
                'latest_price': latest_price
            }
        except Exception as e:
            sys.stderr.write(f"Error predicting: {e}\n")
            return None
    
    def batch_predict(self, symbols, model_type='rf'):
        try:
            predictions = {}
            for symbol in symbols:
                prediction = self.predict(symbol, model_type)
                if prediction:
                    predictions[symbol] = prediction
            return predictions
        except Exception as e:
            sys.stderr.write(f"Error in batch prediction: {e}\n")
            return {}

if __name__ == "__main__":
    trainer = ModelTrainer()
    
    trainer.train_model('000001.SZ')
    
    prediction = trainer.predict('000001.SZ')
    print(prediction)
