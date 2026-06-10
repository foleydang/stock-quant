#!/usr/bin/env python3
"""
日志管理模块
- 每天一个日志文件 (monitor_YYYYMMDD.log)
- 自动清理超过15天的日志
"""

import os
import glob
import logging
from datetime import datetime, timedelta


class DailyLogger:
    """每日日志管理器"""

    def __init__(self, log_dir: str = None, prefix: str = "monitor", retention_days: int = 15):
        """
        初始化日志管理器

        Args:
            log_dir: 日志目录路径
            prefix: 日志文件前缀
            retention_days: 日志保留天数
        """
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(__file__), 'logs')

        self.log_dir = log_dir
        self.prefix = prefix
        self.retention_days = retention_days

        # 确保日志目录存在
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # 清理过期日志
        self._cleanup_old_logs()

        # 创建当日日志文件
        self.today_log_path = self._get_today_log_path()

    def _get_today_log_path(self) -> str:
        """获取当日日志文件路径"""
        today = datetime.now().strftime('%Y%m%d')
        return os.path.join(self.log_dir, f"{self.prefix}_{today}.log")

    def _cleanup_old_logs(self):
        """清理超过保留天数的日志"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        cutoff_str = cutoff_date.strftime('%Y%m%d')

        # 查找所有匹配的日志文件
        pattern = os.path.join(self.log_dir, f"{self.prefix}_*.log")
        log_files = glob.glob(pattern)

        for log_file in log_files:
            # 提取文件名中的日期
            basename = os.path.basename(log_file)
            try:
                # 格式: prefix_YYYYMMDD.log
                date_str = basename.replace(f"{self.prefix}_", "").replace(".log", "")
                if date_str < cutoff_str:
                    os.remove(log_file)
                    print(f"清理过期日志: {basename}")
            except Exception:
                pass

    def log(self, message: str, level: str = "INFO"):
        """
        写入日志

        Args:
            message: 日志消息
            level: 日志级别 (INFO, WARNING, ERROR)
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] [{level}] {message}\n"

        with open(self.today_log_path, 'a', encoding='utf-8') as f:
            f.write(log_line)

        # 同时输出到控制台
        print(message)

    def info(self, message: str):
        self.log(message, "INFO")

    def warning(self, message: str):
        self.log(message, "WARNING")

    def error(self, message: str):
        self.log(message, "ERROR")

    def get_logger(self) -> logging.Logger:
        """获取标准 logging.Logger 对象"""
        logger = logging.getLogger(self.prefix)
        logger.setLevel(logging.INFO)

        # 文件处理器
        file_handler = logging.FileHandler(self.today_log_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        return logger


def get_daily_logger(log_dir: str = None, prefix: str = "monitor", retention_days: int = 15) -> DailyLogger:
    """
    获取每日日志管理器

    Args:
        log_dir: 日志目录
        prefix: 文件前缀
        retention_days: 日志保留天数

    Returns:
        DailyLogger 实例
    """
    return DailyLogger(log_dir=log_dir, prefix=prefix, retention_days=retention_days)


# 测试
if __name__ == "__main__":
    logger = get_daily_logger()
    logger.info("测试日志写入")
    print(f"日志文件: {logger.today_log_path}")