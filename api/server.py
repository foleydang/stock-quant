#!/usr/bin/env python3
"""量化监控API服务 - 简化版"""

from flask import Flask
from flask_cors import CORS
import os
import sys

# 添加python目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))

app = Flask(__name__)
CORS(app)

# 导入路由模块
from routes.stock_routes import stock_bp
from routes.strategy_routes import strategy_bp
from routes.db_routes import db_bp
from routes.forecast_routes import forecast_bp
from routes.calculator_routes import calculator_bp
from routes.advisor_routes import advisor_bp

# 注册蓝图
app.register_blueprint(stock_bp)
app.register_blueprint(strategy_bp)
app.register_blueprint(db_bp)
app.register_blueprint(forecast_bp)
app.register_blueprint(calculator_bp)
app.register_blueprint(advisor_bp)

# 健康检查
@app.route('/health', methods=['GET'])
def health_check():
    return {'status': 'ok', 'message': 'API服务正常运行'}

# 主页
@app.route('/', methods=['GET'])
def index():
    return {
        'name': '股票量化监控API',
        'version': '2.0',
        'endpoints': {
            'stock': '/stock/<symbol>',
            'positions': '/positions',
            'strategy': '/strategy/<symbol>',
            'predict': '/predict/<symbol>',
            'advisor': '/advisor/holdings',
            'db/stats': '/db/stats',
            'db/trades': '/db/trades'
        }
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)