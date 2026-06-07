#!/usr/bin/env python3
"""
统一配置中心 - 所有模块共用这一个配置文件

包含：飞书Bot、LLM、数据库、邮件、交易策略、监控
"""

import os
import yaml
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv()

# 项目目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(PROJECT_ROOT, 'python')
API_DIR = os.path.join(PROJECT_ROOT, 'api')

# 加载 config.yaml
_config_path = os.path.join(PYTHON_DIR, 'config.yaml')
if os.path.exists(_config_path):
    with open(_config_path, 'r', encoding='utf-8') as f:
        _config = yaml.safe_load(f)
else:
    _config = {}

# ========== 飞书 Bot 配置 ==========

FEISHU_APP_ID = os.environ.get("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.environ.get("FEISHU_APP_SECRET", "")
FEISHU_VERIFICATION_TOKEN = os.environ.get("FEISHU_VERIFICATION_TOKEN", "")
FEISHU_ENCRYPT_KEY = os.environ.get("FEISHU_ENCRYPT_KEY", "")
FEISHU_TARGET_CHAT_ID = os.environ.get("FEISHU_TARGET_CHAT_ID", "")
FEISHU_TARGET_OPEN_ID = os.environ.get("FEISHU_TARGET_OPEN_ID", "")
BOT_PORT = int(os.environ.get("BOT_PORT", 8001))

# ========== 百炼 LLM 配置 ==========

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen-plus")

# ========== Tushare / 数据源配置 ==========

TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN', _config.get('tushare_token', ''))
TENCENT_QUOTE_API = 'http://qt.gtimg.cn/q='

# ========== 数据库配置 ==========

DB_PATH = _config.get('database', {}).get('path', os.path.join(PYTHON_DIR, 'data/stock_data.db'))

# ========== 邮件配置 ==========

SMTP_SERVER = os.environ.get('SMTP_SERVER', _config.get('email', {}).get('smtp_server', 'smtp.qq.com'))
SMTP_PORT = int(os.environ.get('SMTP_PORT', '465'))
SMTP_USERNAME = os.environ.get('SMTP_USERNAME', '')
SMTP_PASSWORD = os.environ.get('SMTP_PASSWORD', '')
EMAIL_RECEIVERS = os.environ.get('EMAIL_RECEIVERS', '').split(',')
EMAIL_CONFIG = _config.get('email', {})

# ========== 交易配置 ==========

INITIAL_CAPITAL = float(os.environ.get('INITIAL_CAPITAL', _config.get('account', {}).get('initial_capital', '500000')))
POSITION_PCT = float(os.environ.get('POSITION_PCT', _config.get('account', {}).get('position_pct', '0.30')))
STOP_LOSS_PCT = float(os.environ.get('STOP_LOSS_PCT', '0.05'))
TAKE_PROFIT_PCT = float(os.environ.get('TAKE_PROFIT_PCT', '0.08'))
AVAILABLE_CASH = float(_config.get('account', {}).get('available_cash', 150000))
TOTAL_INVESTMENT = float(_config.get('account', {}).get('total_investment', 1044962))

# ========== 策略参数 ==========

STRATEGY_PARAMS = _config.get('strategy', {})
ACCOUNT_CONFIG = _config.get('account', {})

# ========== 监控配置 ==========

WATCHLIST = _config.get('watchlist', [])

# ========== 数据新鲜度 ==========

DATA_FRESHNESS_HOURS = {
    'weekday': 24,
    'weekend': 72,
}


def get_feishu_config():
    """检查飞书配置是否完整"""
    return all([FEISHU_APP_ID, FEISHU_APP_SECRET])


def get_dashscope_config():
    """检查百炼配置是否完整"""
    return bool(DASHSCOPE_API_KEY)