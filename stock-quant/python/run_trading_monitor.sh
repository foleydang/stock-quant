#!/bin/bash
# 交易监控启动脚本
# 每30分钟运行一次

cd /Users/foleydang/github/stock-quant/stock-quant/python

# 邮件配置
export SMTP_SERVER='smtp.qq.com'
export SMTP_PORT='465'
export SMTP_USERNAME='719312518@qq.com'
export SMTP_PASSWORD='bxmkrxlrulgxbcha'
export EMAIL_RECEIVERS='21725056@zju.edu.cn'

LOG_FILE="logs/monitor.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') - 开始监控" >> $LOG_FILE
/opt/homebrew/bin/python3 trading_monitor.py >> $LOG_FILE 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') - 监控完成" >> $LOG_FILE