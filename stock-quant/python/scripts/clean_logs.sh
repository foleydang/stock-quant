#!/bin/bash
# 清理7天前的日志文件

LOG_DIR="/root/github/stock-quant/stock-quant/python/logs"
find "$LOG_DIR" -name "*.json" -mtime +7 -delete
find "$LOG_DIR" -name "*.log" -mtime +7 -delete

echo "$(date): 清理完成" >> "$LOG_DIR/cleanup.log"
