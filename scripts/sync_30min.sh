#!/bin/bash
# 盘中任务：30分钟K线同步
set -e
cd /root/github/stock-quant/python
/root/miniconda3/bin/python strategy/data_sync.py --30min-only