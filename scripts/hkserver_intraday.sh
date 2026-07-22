#!/bin/bash
# hkserver 盘中任务: 每 30 分钟抓 30m K线 (新浪API)
# cron 建议: */30 9-15 * * 1-5  (交易时段, data_sync 内部还会再判断具体时间)
# hkserver 是真身库, 抓完不上传任何地方 (--no-upload)。
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PY="${PYTHON:-/root/miniconda3/bin/python}"

# 交易日过滤
[ "$(date +%u)" -ge 6 ] && exit 0

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

echo "=== $(date '+%F %T') hkserver 盘中 30m 同步 ==="
cd "$PROJECT_DIR/python" && $PY strategy/data_sync.py --30min-only --no-upload
echo "=== $(date '+%T') 完成 ==="
