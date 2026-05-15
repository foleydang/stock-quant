#!/bin/bash
# 服务健康检查脚本

services=("nginx" "stock-api" "stock-frontend" "wg-quick@wg0" "fail2ban")

for svc in "${services[@]}"; do
    if systemctl is-active --quiet "$svc"; then
        echo "✓ $svc: 运行中"
    else
        echo "✗ $svc: 未运行，尝试启动..."
        systemctl start "$svc"
    fi
done
