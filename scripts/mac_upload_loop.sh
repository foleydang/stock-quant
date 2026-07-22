#!/bin/bash
# Mac 端：每30分钟上传数据到 OSS（盘中执行）
# 放在 Mac 的 crontab 中，交易日 9:30-15:00 每30分钟执行

set -e

WEEKDAY=$(date +%u)
if [ "$WEEKDAY" -ge 6 ]; then
    exit 0  # 周末跳过
fi

HOUR=$(date +%H)
MINUTE=$(date +%M)

# 只在交易时段执行 (9:30-15:00)
if [ "$HOUR" -lt 9 ] || [ "$HOUR" -gt 15 ] || ([ "$HOUR" -eq 9 ] && [ "$MINUTE" -lt 30 ]); then
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
echo "[$(date)] 📤 Mac 上传 30m 增量到 OSS..."
# Mac/香港机只产 kline_30m; 按月分片, 只重传当月, 不再整库覆盖
cd "$PROJECT_DIR/python" && python3 strategy/oss_incr.py upload --producer mac --days 2
echo "[$(date)] ✅ 上传完成"