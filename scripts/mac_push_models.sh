#!/bin/bash
# Mac: 把训练产物回推到 hkserver (仅回推, 不训练)
#   训练前先 mac_pull.sh 拉数据, 训练另跑 retrain_all_mac.py
#
# 认证: 优先 SSH key。若用密码, 设 HK_PASS 环境变量。
#
# 用法: bash scripts/mac_push_models.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

HK="${HK_HOST:-hkserver}"
HK_ROOT="${HK_ROOT:-/root/github/stock-quant}"

# rsync -e 的 ssh 命令
if [ -n "$HK_PASS" ]; then
  command -v sshpass >/dev/null || { echo "❌ HK_PASS 已设但缺 sshpass"; exit 1; }
  RSYNC_E="sshpass -p $HK_PASS ssh -o PubkeyAuthentication=no -o PreferredAuthentications=password -o StrictHostKeyChecking=accept-new"
else
  RSYNC_E="ssh"
fi

echo "=== $(date '+%F %T') 回推模型产物到 hkserver ==="
# 只推训练产物 (几十MB), 不是整个 9.6G models 目录
push() {
    local src="$1" dst="$2"
    if [ -f "$src" ]; then
      rsync -avz -e "$RSYNC_E" "$src" "$HK:$HK_ROOT/$dst" && echo "  ✓ $dst"
    else
      echo "  ⏭️ 跳过 $src (不存在)"
    fi
}
push "$PROJECT_DIR/python/models/lgb_30m/model.pkl"          "python/models/lgb_30m/"
push "$PROJECT_DIR/models/lgb_hs300_enhanced/model.pkl"      "models/lgb_hs300_enhanced/"
push "$PROJECT_DIR/python/data/lstm_embeddings.pkl"          "python/data/"

echo "=== $(date '+%T') 回推完成 ==="
