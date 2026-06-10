#!/usr/bin/env python3
"""统一日志管理模块 - 支持分级别、持久化"""

import os
import sys
import logging
import sqlite3
from datetime import datetime
from typing import Optional

# 从配置读取路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import get_base_dir, get_db_path

LOG_DIR = os.path.join(get_base_dir(), 'logs')
DB_PATH = get_db_path()

# 日志级别映射
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

class QuantLogger:
    """量化系统统一日志器"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_logger()
        return cls._instance
    
    def _init_logger(self):
        """初始化日志器"""
        # 确保目录存在
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # 创建日志器
        self.logger = logging.getLogger('quant_monitor')
        self.logger.setLevel(logging.INFO)
        
        # 文件日志 (每天一个文件)
        log_file = os.path.join(LOG_DIR, f"monitor_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # 控制台日志
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # 错误日志单独文件
        error_file = os.path.join(LOG_DIR, f"errors_{datetime.now().strftime('%Y%m%d')}.log")
        error_handler = logging.FileHandler(error_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
    
    def info(self, msg: str):
        """记录信息日志"""
        self.logger.info(msg)
        self._db_log('INFO', msg)
    
    def warning(self, msg: str):
        """记录警告日志"""
        self.logger.warning(msg)
        self._db_log('WARNING', msg)
    
    def error(self, msg: str, exc_info: Optional[Exception] = None):
        """记录错误日志"""
        self.logger.error(msg, exc_info=exc_info)
        self._db_log('ERROR', msg)
    
    def debug(self, msg: str):
        """记录调试日志"""
        self.logger.debug(msg)
    
    def critical(self, msg: str):
        """记录严重错误"""
        self.logger.critical(msg)
        self._db_log('CRITICAL', msg)
    
    def _db_log(self, level: str, message: str):
        """写入数据库日志表"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # 检查表是否存在
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='logs'")
            if cursor.fetchone() is None:
                # 创建日志表
                cursor.execute('''
                    CREATE TABLE logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        level TEXT NOT NULL,
                        message TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
            
            cursor.execute(
                "INSERT INTO logs (level, message) VALUES (?, ?)",
                (level, message[:500])  # 限制消息长度
            )
            conn.commit()
            conn.close()
        except Exception:
            pass  # 日志写入失败不抛异常

# 全局日志器
logger = QuantLogger()

# 便捷函数
def log_info(msg):
    logger.info(msg)

def log_warning(msg):
    logger.warning(msg)

def log_error(msg, exc=None):
    logger.error(msg, exc_info=exc)

def log_debug(msg):
    logger.debug(msg)

# 测试
if __name__ == '__main__':
    log_info("测试信息日志")
    log_warning("测试警告日志")
    log_error("测试错误日志")