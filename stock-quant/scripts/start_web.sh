#!/bin/bash
# Stock-quant 网站启动脚本 - 先清理旧进程再启动

PROJECT_DIR="/root/github/stock-quant/stock-quant"

# 清理旧的 vite 进程
pkill -f "vite" 2>/dev/null
pkill -f "esbuild" 2>/dev/null
sleep 2

# 启动后端
cd $PROJECT_DIR
nohup python3.11 api/server.py > logs/api.log 2>&1 &

# 启动前端
cd $PROJECT_DIR/frontend
nohup npm run dev > ../logs/frontend.log 2>&1 &

sleep 3
echo "服务已启动"
echo "后端: http://localhost:8000"
echo "前端: http://localhost:3000"
netstat -tlnp | grep -E "3000|8000"