#!/bin/bash
# hkserver 盘后任务: A股日线 + 港股/ETF + 预测
# cron 建议: 30 15 * * 1-5  (收盘后)
# hkserver 是真身库(全包): 抓数不上传, 预测直接读本地库。Mac 训练好的模型已回推到本机。
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PY="${PYTHON:-/root/miniconda3/bin/python}"

[ "$(date +%u)" -ge 6 ] && exit 0

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

echo "=== $(date '+%F %T') hkserver 盘后开始 ==="

echo "📡 1/3 A股日线 (Tushare)..."
cd "$PROJECT_DIR/python" && $PY strategy/data_sync.py --daily-only --no-upload

echo "📡 2/3 港股/ETF (yfinance)..."
$PY "$PROJECT_DIR/scripts/sync_hk_etf.py"

echo "📊 3/3 跑预测..."
cd "$PROJECT_DIR/python" && $PY strategy/predict_today_batched.py --batch 500

echo "=== $(date '+%T') hkserver 盘后完成 ==="
