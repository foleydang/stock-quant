#!/bin/bash
# LGBM交易监控启动脚本
# 每30分钟运行一次

cd /Users/foleydang/github/stock-quant/stock-quant/python

# 加载环境变量
export $(cat .env | xargs)

# 执行监控
/opt/homebrew/bin/python3 scheduled_monitor.py --once

echo "监控完成: $(date)"