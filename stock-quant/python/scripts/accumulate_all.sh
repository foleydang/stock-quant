#!/bin/bash
# 统一的K线累积脚本 - 每30分钟运行一次

cd /root/github/stock-quant/stock-quant/python
/root/miniconda3/bin/python -c "
import sys
sys.path.insert(0, '.')
from data.kline_accumulator import KlineAccumulator

acc = KlineAccumulator()
symbols = ['300124.SZ', '600048.SH', '3690.HK', '300015.SZ', '159792.SZ', '9988.HK']
acc.accumulate_realtime(symbols)
" >> logs/kline_accumulate.log 2>&1
