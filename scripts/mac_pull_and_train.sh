#!/bin/bash
# Mac: 从 hkserver 拉最新真身库 -> 本地训练 -> 把模型产物回推 hkserver
#   数据流是 Mac 主动拉 (Mac 在 NAT 后无公网IP, hkserver 推不过来)。
#   sqlite3_rsync 走 SSH 只传变化的页, 对 live 库安全。
#
# 认证: 优先用 SSH key (配过 ssh-copy-id 即可)。若仍用密码,
#   设置环境变量 HK_PASS, 脚本自动用 sshpass 包装。
#   配好 key 后建议轮换并清掉密码。
#
# 用法: bash scripts/mac_pull_and_train.sh [--quick|--30m-only|--daily-only|...]
#   (透传给 retrain_all_mac.py)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PY="${PYTHON:-python3}"

HK="${HK_HOST:-hkserver}"
HK_ROOT="${HK_ROOT:-/root/github/stock-quant}"
REMOTE_DB="$HK_ROOT/python/data/stock_data.db"
LOCAL_DB="$PROJECT_DIR/python/data/stock_data.db"

# --- SSH 包装: 有 HK_PASS 就用 sshpass, 否则走默认 key 认证 ---
SSH_WRAP=""
if [ -n "$HK_PASS" ]; then
  command -v sshpass >/dev/null || { echo "❌ HK_PASS 已设但缺 sshpass: brew install sshpass (或改用 key 认证)"; exit 1; }
  SSH_WRAP="$PROJECT_DIR/.ssh_pw.sh"
  cat > "$SSH_WRAP" <<EOF
#!/bin/bash
exec sshpass -p '$HK_PASS' ssh -o PubkeyAuthentication=no -o PreferredAuthentications=password -o StrictHostKeyChecking=accept-new "\$@"
EOF
  chmod +x "$SSH_WRAP"
  SSH_ARG="--ssh $SSH_WRAP"
  SCP_WRAP="sshpass -p $HK_PASS scp -o PubkeyAuthentication=no -o PreferredAuthentications=password -o StrictHostKeyChecking=accept-new"
else
  SSH_ARG=""
  SCP_WRAP="scp"
fi

# --- 拉取前清掉远端可能挂着的 sqlite3_rsync 僵尸 (它会锁住真身库) ---
if [ -n "$SSH_WRAP" ]; then
  "$SSH_WRAP" "$HK" "pkill -9 -f 'sqlite3_rsync --origin' 2>/dev/null; true"
else
  ssh "$HK" "pkill -9 -f 'sqlite3_rsync --origin' 2>/dev/null; true"
fi

echo "=== $(date '+%F %T') [1/3] 拉取最新真身库 (sqlite3_rsync 增量) ==="
mkdir -p "$(dirname "$LOCAL_DB")"
# 锁冲突时重试 (远端正在写库是正常的)
for attempt in 1 2 3 4 5; do
  if sqlite3_rsync $SSH_ARG "$HK:$REMOTE_DB" "$LOCAL_DB"; then
    echo "本地库: $(ls -lh "$LOCAL_DB" | awk '{print $5}')"
    break
  fi
  echo "  ⚠️ 第 $attempt 次拉取失败 (可能是远端写库占锁), 等 15s 重试..."
  [ "$attempt" -eq 5 ] && { echo "❌ 拉取 5 次仍失败, 放弃"; exit 1; }
  sleep 15
done

echo "=== $(date '+%F %T') [2/3] 本地训练 ==="
cd "$PROJECT_DIR/python" && $PY strategy/retrain_all_mac.py "$@"

echo "=== $(date '+%F %T') [3/3] 回推模型产物到 hkserver ==="
# 只推训练产物 (几十MB), 不是整个 9.6G models 目录
push() {
    local src="$1" dst="$2"
    if [ -f "$src" ]; then
      rsync -avz -e "${SCP_WRAP/ scp/}" "$src" "$HK:$HK_ROOT/$dst" && echo "  ✓ $dst"
    else
      echo "  ⏭️ 跳过 $src (不存在)"
    fi
}
push "$PROJECT_DIR/python/models/lgb_30m/model.pkl"          "python/models/lgb_30m/"
push "$PROJECT_DIR/models/lgb_hs300_enhanced/model.pkl"      "models/lgb_hs300_enhanced/"
push "$PROJECT_DIR/python/data/lstm_embeddings.pkl"          "python/data/"

# 临时密码包装文件清理
[ -n "$HK_PASS" ] && rm -f "$SSH_WRAP"

echo "=== $(date '+%T') 完成: 数据已训练, 模型已回推 hkserver ==="
