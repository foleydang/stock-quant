#!/bin/bash
# LGBM定时交易监控脚本
# 每30分钟: 更新数据 -> 执行策略 -> 发送邮件

cd /Users/foleydang/github/stock-quant/stock-quant/python

# 邮件配置
export SMTP_SERVER='smtp.qq.com'
export SMTP_PORT='465'
export SMTP_USERNAME='719312518@qq.com'
export SMTP_PASSWORD='bxmkrxlrulgxbcha'
export EMAIL_RECEIVERS='21725056@zju.edu.cn'

LOG_FILE="logs/monitor.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') - 开始监控" >> $LOG_FILE

# 执行监控（更新数据 + 监控）
/opt/homebrew/bin/python3 full_monitor.py --monitor >> $LOG_FILE 2>&1

echo "$(date '+%Y-%m-%d %H:%M:%S') - 监控完成" >> $LOG_FILE