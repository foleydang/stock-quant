#!/bin/bash
# Mac: 从 hkserver 拉最新真身库 -> 本地训练 -> 把模型产物回推 hkserver
#   数据流是 Mac 主动拉 (Mac 在 NAT 后无公网IP, hkserver 推不过来)。
#   sqlite3_rsync 走 SSH 只传变化的页, 对 live 库安全。
#
# 用法: bash scripts/mac_pull_and_train.sh [--quick|--30m-only|--daily-only|...]
#   (透传给 retrain_all_mac.py)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PY="${PYTHON:-python3}"

HK="${HK_HOST:-hkserver}"                                   # ~/.ssh/config 里的别名
HK_ROOT="${HK_ROOT:-/root/github/stock-quant}"              # hkserver 上的项目路径
REMOTE_DB="$HK_ROOT/python/data/stock_data.db"
LOCAL_DB="$PROJECT_DIR/python/data/stock_data.db"

echo "=== $(date '+%F %T') [1/3] 拉取最新真身库 (sqlite3_rsync) ==="
mkdir -p "$(dirname "$LOCAL_DB")"
sqlite3_rsync "$HK:$REMOTE_DB" "$LOCAL_DB"
echo "本地库: $(ls -lh "$LOCAL_DB" | awk '{print $5}')"

echo "=== $(date '+%F %T') [2/3] 本地训练 ==="
cd "$PROJECT_DIR/python" && $PY strategy/retrain_all_mac.py "$@"

echo "=== $(date '+%F %T') [3/3] 回推模型产物到 hkserver ==="
# 只推训练产物 (几十MB), 不是整个 9.6G models 目录
push() {
    local src="$1" dst="$2"
    [ -f "$src" ] && rsync -avz "$src" "$HK:$HK_ROOT/$dst" && echo "  ✓ $dst" || echo "  ⏭️ 跳过 $src (不存在)"
}
push "$PROJECT_DIR/python/models/lgb_30m/model.pkl"          "python/models/lgb_30m/"
push "$PROJECT_DIR/models/lgb_hs300_enhanced/model.pkl"      "models/lgb_hs300_enhanced/"
push "$PROJECT_DIR/python/data/lstm_embeddings.pkl"          "python/data/"

echo "=== $(date '+%T') 完成: 数据已训练, 模型已回推 hkserver ==="
