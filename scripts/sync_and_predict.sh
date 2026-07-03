#!/bin/bash
# 数据同步脚本（仅下载）
# 服务器端执行：从 OSS 下载最新数据

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEEKDAY=$(date +%u)

# 周末不执行
if [ "$WEEKDAY" -ge 6 ]; then
    echo "[$(date)] 周末, 跳过" >> /tmp/stock_sync.log
    exit 0
fi

echo "[$(date)] 🔄 从OSS同步数据..." >> /tmp/stock_sync.log
bash "$SCRIPT_DIR/download_from_oss.sh" >> /tmp/stock_sync.log 2>&1
echo "[$(date)] ✅ 同步完成" >> /tmp/stock_sync.log