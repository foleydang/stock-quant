#!/bin/bash
# 盘中任务：30分钟K线同步
set -e
cd /root/github/stock-quant/python
python3 strategy/data_sync.py --30min-only