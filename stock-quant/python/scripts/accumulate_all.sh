#!/bin/bash
# 统一的K线累积脚本 - 每30分钟运行一次
# 采集所有关注列表 + 沪深300核心股票

# 防止重复运行
LOCK_FILE="/tmp/stock-quant-locks/accumulate.lock"
mkdir -p /tmp/stock-quant-locks
exec 200>$LOCK_FILE
if ! flock -n 200; then
    echo "⚠️ accumulate 正在运行，跳过"
    exit 0
fi

cd /root/github/stock-quant/stock-quant/python
/root/miniconda3/bin/python -c "
import sys
sys.path.insert(0, '.')
from data.kline_accumulator import KlineAccumulator

acc = KlineAccumulator()
# Watchlist股票 + 港股 + ETF
watchlist = ['300015.SZ', '300124.SZ', '600048.SH', '600519.SH', '000001.SZ',
             '000333.SZ', '002594.SZ', '601318.SH', '600036.SH', '000858.SZ',
             '3690.HK', '0700.HK', '9988.HK', '159792.SZ']
acc.accumulate_realtime(watchlist)
" >> logs/kline_accumulate.log 2>&1
