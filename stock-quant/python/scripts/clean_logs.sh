#!/bin/bash
# 日志清理脚本 - 每周运行

LOG_DIR="/root/github/stock-quant/stock-quant/logs"

# 清理超过30天的日志
find "$LOG_DIR" -name "*.log" -mtime +30 -delete

# 清理超过7天的错误日志
find "$LOG_DIR" -name "errors_*.log" -mtime +7 -delete

# 清理超过15天的JSON文件
find "$LOG_DIR" -name "*.json" -mtime +15 -delete

# 输出清理结果
echo "[$(date)] Logs cleaned in $LOG_DIR"
ls -la "$LOG_DIR"/*.log | wc -l
