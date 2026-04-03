#!/usr/bin/env python3
import sys
import json
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_trainer import ModelTrainer

def main():
    if len(sys.argv) < 2:
        print("Usage: python model_runner.py <command> [symbol] [model_type]")
        print("Commands:")
        print("  fetch <symbol> - Fetch stock data")
        print("  train <symbol> [model_type] - Train model for a stock")
        print("  predict <symbol> [model_type] - Predict stock price")
        print("  batch_predict <symbol1> <symbol2> ... - Predict multiple stocks")
        print("  select <symbol1> <symbol2> ... - Select optimal stocks")
        return
    
    command = sys.argv[1]
    trainer = ModelTrainer()
    
    if command == "fetch":
        if len(sys.argv) < 3:
            json_output = json.dumps({
                "status": "failure",
                "error": "Symbol is required for fetch command"
            })
            sys.stdout.write(json_output)
            sys.stdout.flush()
            return
        
        symbol = sys.argv[2]
        
        from data.data_handler import DataHandler
        data_handler = DataHandler()
        
        data = data_handler.fetch_stock_data(symbol)
        
        if data is not None:
            try:
                data_list = []
                for _, row in data.iterrows():
                    data_list.append({
                        "date": row["date"].strftime("%Y-%m-%d"),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row["volume"])
                    })
                json_output = json.dumps({
                    "status": "success",
                    "data": data_list
                })
                sys.stdout.write(json_output)
                sys.stdout.flush()
            except Exception as e:
                json_output = json.dumps({
                    "status": "failure",
                    "error": f"Failed to process data: {str(e)}"
                })
                sys.stdout.write(json_output)
                sys.stdout.flush()
        else:
            json_output = json.dumps({
                "status": "failure",
                "error": "Failed to fetch stock data from Yahoo Finance"
            })
            sys.stdout.write(json_output)
            sys.stdout.flush()
        return
    
    if command == "train":
        if len(sys.argv) < 3:
            print("Error: Symbol is required for train command")
            return
        
        symbol = sys.argv[2]
        model_type = sys.argv[3] if len(sys.argv) > 3 else "rf"
        
        result = trainer.train_model(symbol, model_type)
        if result:
            print(json.dumps({
                "status": "success",
                "symbol": symbol,
                "model_type": model_type,
                "metrics": result["metrics"]
            }))
        else:
            print(json.dumps({
                "status": "failure",
                "symbol": symbol,
                "model_type": model_type,
                "error": "Failed to train model"
            }))
    
    elif command == "predict":
        if len(sys.argv) < 3:
            print("Error: Symbol is required for predict command")
            return
        
        symbol = sys.argv[2]
        model_type = sys.argv[3] if len(sys.argv) > 3 else "rf"
        
        prediction = trainer.predict(symbol, model_type)
        if prediction:
            print(json.dumps({
                "status": "success",
                "symbol": symbol,
                "model_type": model_type,
                "predicted_price": prediction["predicted_price"],
                "predicted_return": prediction["predicted_return"],
                "latest_price": prediction["latest_price"]
            }))
        else:
            print(json.dumps({
                "status": "failure",
                "symbol": symbol,
                "model_type": model_type,
                "error": "Failed to predict price"
            }))
    
    elif command == "batch_predict":
        if len(sys.argv) < 3:
            print("Error: At least one symbol is required for batch_predict command")
            return
        
        symbols = sys.argv[2:]
        model_type = "rf"
        
        predictions = trainer.batch_predict(symbols, model_type)
        print(json.dumps({
            "status": "success",
            "model_type": model_type,
            "predictions": predictions
        }))
    
    elif command == "select":
        if len(sys.argv) < 3:
            print("Error: At least one symbol is required for select command")
            return
        
        symbols = sys.argv[2:]
        top_n = 5
        
        # 导入选股模块
        from models.stock_selector import StockSelector
        selector = StockSelector()
        
        result = selector.select_stocks(symbols, top_n)
        print(json.dumps({
            "status": "success",
            "selected_stocks": result
        }))
    
    else:
        print(f"Error: Unknown command: {command}")

if __name__ == "__main__":
    main()