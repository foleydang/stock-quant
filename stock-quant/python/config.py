"""
配置中心 - 统一管理所有配置
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API配置
TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN', '7a9014b18909e8cbce5109d7175f7b21ce37354eaff2371db0da2c58')
TENCENT_QUOTE_API = 'http://qt.gtimg.cn/q='

# 数据库配置
DB_PATH = os.path.join(os.path.dirname(__file__), 'data/stock_data.db')

# 邮件配置
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.qq.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '465'))
SMTP_USERNAME = os.getenv('SMTP_USERNAME', '')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')
EMAIL_RECEIVERS = os.getenv('EMAIL_RECEIVERS', '').split(',')

# 交易配置
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '100000'))
POSITION_PCT = float(os.getenv('POSITION_PCT', '0.15'))
STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', '0.05'))
TAKE_PROFIT_PCT = float(os.getenv('TAKE_PROFIT_PCT', '0.08'))

# 数据新鲜度阈值（小时）
DATA_FRESHNESS_HOURS = {
    'weekday': 24,      # 工作日
    'weekend': 72,      # 周末（允许周五数据）
}
