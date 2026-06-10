"""统一配置加载模块"""

import os
import yaml

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

_config = None

def load_config():
    """加载配置"""
    global _config
    if _config is None:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                _config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Config not found: {CONFIG_FILE}")
    return _config

def get(key, default=None):
    """获取配置值，支持嵌套键"""
    config = load_config()
    keys = key.split(".")
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    return value

# 常用配置（延迟加载）
def get_base_dir():
    return get("base.dir")

def get_db_path():
    return get("database.path")

def get_available_cash():
    return get("account.available_cash", 150000)

def get_watchlist():
    return get("watchlist", [])

def get_strategy_params():
    return get("strategy", {})

def get_email_config():
    return get("email", {})

# 测试
if __name__ == "__main__":
    print(f"BASE_DIR: {get_base_dir()}")
    print(f"DB_PATH: {get_db_path()}")
    print(f"CASH: {get_available_cash()}")
    print(f"Watchlist: {get_watchlist()}")
    print(f"Strategy: {get_strategy_params()}")
