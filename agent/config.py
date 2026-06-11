#!/usr/bin/env python3
"""
统一配置中心

所有模块共用这一个配置文件。从 .env 和 config.yaml 读取。
"""

import os
import yaml
from dotenv import load_dotenv

# 项目目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(PROJECT_ROOT, 'python')

load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# 加载 config.yaml
_config_path = os.path.join(PYTHON_DIR, 'config.yaml')
if os.path.exists(_config_path):
    with open(_config_path, 'r', encoding='utf-8') as f:
        _config = yaml.safe_load(f)
else:
    _config = {}

# ========== 飞书 Bot ==========

FEISHU_APP_ID = os.environ.get("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.environ.get("FEISHU_APP_SECRET", "")
FEISHU_VERIFICATION_TOKEN = os.environ.get("FEISHU_VERIFICATION_TOKEN", "")
FEISHU_ENCRYPT_KEY = os.environ.get("FEISHU_ENCRYPT_KEY", "")
FEISHU_TARGET_CHAT_ID = os.environ.get("FEISHU_TARGET_CHAT_ID", "")
FEISHU_TARGET_OPEN_ID = os.environ.get("FEISHU_TARGET_OPEN_ID", "")
BOT_PORT = int(os.environ.get("BOT_PORT", 8001))

# ========== 百炼 LLM ==========

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen-plus")

# ========== 数据源 ==========

TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN', _config.get('tushare_token', ''))
TENCENT_QUOTE_API = 'http://qt.gtimg.cn/q='

# ========== 数据库 ==========

DB_PATH = _config.get('database', {}).get('path', os.path.join(PROJECT_ROOT, 'python', 'data', 'stock_data.db'))

# ========== 邮件 ==========

SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.qq.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', '465'))
SMTP_USERNAME = os.environ.get('SMTP_USERNAME', '')
SMTP_PASSWORD = os.environ.get('SMTP_PASSWORD', '')
EMAIL_RECEIVERS = os.environ.get('EMAIL_RECEIVERS', '').split(',')

# ========== 交易 & 策略 ==========

ACCOUNT_CONFIG = _config.get('account', {})
STRATEGY_PARAMS = _config.get('strategy', {})
WATCHLIST = _config.get('watchlist', [])

def save_watchlist():
    """保存自选股到 config.yaml"""
    import yaml
    _config['watchlist'] = WATCHLIST
    with open(_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(_config, f, allow_unicode=True, default_flow_style=False)
    logger = __import__('logging').getLogger('feishu_bot')
    logger.info(f"✓ 自选股已保存到 config.yaml ({len(WATCHLIST)}只)")
AVAILABLE_CASH = ACCOUNT_CONFIG.get('available_cash', 150000)
TOTAL_INVESTMENT = ACCOUNT_CONFIG.get('total_investment', 1044962)