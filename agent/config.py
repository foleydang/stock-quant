#!/usr/bin/env python3
"""
统一配置中心

所有密钥从 .env 读取，业务配置从 config.yaml 读取。
路径支持相对路径（相对于项目根目录），跨机器直接复制 .env 即可。

项目结构:
  stock-quant/          ← PROJECT_ROOT
    .env                ← 唯一需要 scp 的配置文件（所有密钥）
    agent/              ← 飞书 Bot
    python/             ← 策略 + 数据
      config.yaml       ← 业务配置（非敏感，可提交 git）
      data/             ← 数据库
"""

import os
import yaml
from dotenv import load_dotenv

# 项目根目录 - agent/ 的父目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(PROJECT_ROOT, 'python')
DATA_DIR = os.path.join(PYTHON_DIR, 'data')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# 加载 .env（唯一需要跨机器复制的配置文件）
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))


def _env(key, default=""):
    """从 .env 读取，优先环境变量"""
    return os.environ.get(key, default)


def _resolve_path(path: str) -> str:
    """解析相对路径为绝对路径"""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


# 加载 config.yaml（非敏感业务配置）
_config_path = os.path.join(PYTHON_DIR, 'config.yaml')
if os.path.exists(_config_path):
    with open(_config_path, 'r', encoding='utf-8') as f:
        _config = yaml.safe_load(f)
else:
    _config = {}

# ========== 飞书 Bot ==========

FEISHU_APP_ID = _env("FEISHU_APP_ID")
FEISHU_APP_SECRET = _env("FEISHU_APP_SECRET")
FEISHU_VERIFICATION_TOKEN = _env("FEISHU_VERIFICATION_TOKEN")
FEISHU_ENCRYPT_KEY = _env("FEISHU_ENCRYPT_KEY")
FEISHU_TARGET_CHAT_ID = _env("FEISHU_TARGET_CHAT_ID")
FEISHU_TARGET_OPEN_ID = _env("FEISHU_TARGET_OPEN_ID")
BOT_PORT = int(_env("BOT_PORT", "8001"))

# ========== 百炼 LLM ==========

DASHSCOPE_API_KEY = _env("DASHSCOPE_API_KEY")
DASHSCOPE_MODEL = _env("DASHSCOPE_MODEL", "qwen-plus")

# ========== 数据源 ==========

TUSHARE_TOKEN = _env("TUSHARE_TOKEN")
TENCENT_QUOTE_API = 'http://qt.gtimg.cn/q='

# ========== 阿里云 OSS ==========

OSS_ACCESS_KEY_ID = _env("OSS_ACCESS_KEY_ID")
OSS_ACCESS_KEY_SECRET = _env("OSS_ACCESS_KEY_SECRET")
OSS_ENDPOINT = _env("OSS_ENDPOINT", "oss-cn-hangzhou.aliyuncs.com")
OSS_BUCKET = _env("OSS_BUCKET", "yanten-data")

# ========== 数据库 ==========

DB_PATH = _resolve_path(_config.get('database', {}).get('path', 'python/data/stock_data.db'))
KLINE_DB_PATH = _resolve_path(_config.get('database', {}).get('kline_path', 'python/data/stock_data.db'))
BACKUP_DIR = _resolve_path(_config.get('database', {}).get('backup_dir', 'python/data/backup'))

# ========== 邮件 ==========

SMTP_SERVER = _env("SMTP_SERVER", "smtp.qq.com")
SMTP_PORT = int(_env("SMTP_PORT", "465"))
SMTP_USERNAME = _env("SMTP_USERNAME")
SMTP_PASSWORD = _env("SMTP_PASSWORD")
EMAIL_RECEIVERS = _env("EMAIL_RECEIVERS", "").split(",")

# ========== 交易 & 策略 ==========

ACCOUNT_CONFIG = _config.get('account', {})
STRATEGY_PARAMS = _config.get('strategy', {})
WATCHLIST = _config.get('watchlist', [])

AVAILABLE_CASH = int(_env("AVAILABLE_CASH", "150000"))
TOTAL_INVESTMENT = int(_env("TOTAL_INVESTMENT", "1044962"))


def save_watchlist():
    """保存自选股到 config.yaml"""
    _config['watchlist'] = WATCHLIST
    with open(_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(_config, f, allow_unicode=True, default_flow_style=False)
    logger = __import__('logging').getLogger('feishu_bot')
    logger.info(f"✓ 自选股已保存到 config.yaml ({len(WATCHLIST)}只)")