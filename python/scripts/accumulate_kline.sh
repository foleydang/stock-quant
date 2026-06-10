#!/bin/bash
# 每30分钟累积实时K线

cd /root/github/stock-quant/python
/root/miniconda3/bin/python data/kline_accumulator.py >> logs/kline_accumulate.log 2>&1
