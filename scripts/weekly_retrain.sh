#!/bin/bash
# ============================================================
# 周度模型滚动重训 — 每周日晚跑, 周一盘前用上新模型
#
# 当前包含:
#   1. 更新 ETF + 港股成分股 + 南向资金数据 (update_etf_data.py --hk)
#   2. 重训 159792 港股通互联网ETF 专用模型 (horizon=20)
#
# 可选(重, 默认关): 重训 HS300 增强模型 ~30min, 用 --with-hs300 开启
#   bash scripts/weekly_retrain.sh --with-hs300
#
# cron (已由 setup_cron.sh 安装): 0 23 * * 0
# ============================================================
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python3"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/retrain_$(date +%Y%m%d).log"; }

cd "$PROJECT_ROOT/python"

log "=== 周度重训开始 ==="

# 1. 刷新 ETF + 港股成分股 + 南向资金
log "[1/2] 更新 ETF + 港股 + 南向数据..."
PYTHONPATH="$PROJECT_ROOT/python" $PYTHON update_etf_data.py --hk \
    2>&1 | grep -E "bars|南向|错误|完成" | while read line; do log "   $line"; done

# 2. 重训 ETF159792 模型 (horizon=20, 含南向特征)
log "[2/2] 重训 159792 模型 (horizon=20)..."
PYTHONPATH="$PROJECT_ROOT/python" $PYTHON strategy/etf159792_model.py --horizon 20 \
    2>&1 | grep -E "horizon|样本外|方向准确率|决策净收益|可用|当前|现价|ML:|建议|南向|💾" \
    | while read line; do log "   $line"; done

# 可选: 重训 HS300 增强模型 (重, ~30min)
if [ "$1" = "--with-hs300" ]; then
    log "[额外] 重训 HS300 增强模型 (lgb_hs300_enhanced)..."
    PYTHONPATH="$PROJECT_ROOT/python" $PYTHON strategy/train_enhanced.py 2>&1 \
        | grep -E "训练|IC=|耗时|保存|✅" | while read line; do log "   $line"; done
fi

log "=== 周度重训完成 ==="
