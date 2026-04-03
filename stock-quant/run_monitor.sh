#!/bin/bash
# 股票监控启动脚本

# 配置环境变量
export SMTP_SERVER='smtp.qq.com'
export SMTP_PORT='465'
export SMTP_USERNAME='719312518@qq.com'
export SMTP_PASSWORD='bxmkrxlrulgxbcha'
export EMAIL_RECEIVERS='21725056@zju.edu.cn'

# 进入项目目录
cd /Users/foleydang/github/stock-quant/stock-quant/python

# 使用 Homebrew 安装的 Python 3.14
PYTHON3="/opt/homebrew/opt/python@3.14/bin/python3.14"

# 运行监控
"$PYTHON3" strategy/email_monitor.py >> ../logs/email_monitor.log 2>&1
