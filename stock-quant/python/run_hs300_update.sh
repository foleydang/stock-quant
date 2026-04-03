#!/bin/bash
# 沪深300增量更新脚本
# 保留历史数据，只插入新数据

cd /Users/foleydang/github/stock-quant/stock-quant/python

LOG_FILE="logs/hs300_update.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') - 开始增量更新" >> $LOG_FILE
/opt/homebrew/bin/python3 update_hs300.py >> $LOG_FILE 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') - 更新完成" >> $LOG_FILE