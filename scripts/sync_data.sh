#!/bin/bash
# 从本地同步数据到服务器
# 在本地 Mac 上运行: scp -r stock-quant/python/data/*.csv server:/root/github/stock-quant/python/data/

SERVER="your_server_ip"
REMOTE_PATH="/root/github/stock-quant/python/data/"

echo "数据同步脚本"
echo "============="
echo ""
echo "在本地 Mac 执行:"
echo "  scp -r ~/github/stock-quant/stock-quant/python/data/*.csv $SERVER:$REMOTE_PATH"
echo ""
echo "或使用 rsync:"
echo "  rsync -avz ~/github/stock-quant/stock-quant/python/data/*.csv $SERVER:$REMOTE_PATH"
