#!/bin/bash
# Mac: 从 hkserver 拉最新真身库 (仅同步, 不训练)
#   训练另跑 retrain_all_mac.py; 训完回推另跑 mac_push_models.sh
#   数据流是 Mac 主动拉 (Mac 在 NAT 后无公网IP)。sqlite3_rsync 只传变化的页。
#
# 认证: 优先 SSH key (配过 ssh-copy-id)。若仍用密码, 设 HK_PASS 环境变量。
#
# 用法: bash scripts/mac_pull.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

HK="${HK_HOST:-hkserver}"
HK_ROOT="${HK_ROOT:-/root/github/stock-quant}"
REMOTE_DB="$HK_ROOT/python/data/stock_data.db"
LOCAL_DB="$PROJECT_DIR/python/data/stock_data.db"

# --- SSH 包装: 有 HK_PASS 就用 sshpass, 否则走默认 key ---
SSH_WRAP=""
if [ -n "$HK_PASS" ]; then
  command -v sshpass >/dev/null || { echo "❌ HK_PASS 已设但缺 sshpass: brew install sshpass (或改用 key 认证)"; exit 1; }
  SSH_WRAP="$(mktemp)"
  cat > "$SSH_WRAP" <<EOF
#!/bin/bash
exec sshpass -p '$HK_PASS' ssh -o PubkeyAuthentication=no -o PreferredAuthentications=password -o StrictHostKeyChecking=accept-new "\$@"
EOF
  chmod +x "$SSH_WRAP"
  SSH_ARG="--ssh $SSH_WRAP"
  CLEANUP="$SSH_WRAP"
else
  SSH_ARG=""
fi
trap '[ -n "$CLEANUP" ] && rm -f "$CLEANUP"' EXIT

# --- 拉取前清掉远端可能挂着的 sqlite3_rsync 僵尸 (它会锁住真身库) ---
# 用 set +e 包住: pkill 没匹配返回非0 / 偶发连接问题都不能让脚本提前死。
set +e
if [ -n "$SSH_WRAP" ]; then
  "$SSH_WRAP" "$HK" "pkill -9 -f 'sqlite3_rsync --origin' 2>/dev/null; true" >/dev/null 2>&1
else
  ssh "$HK" "pkill -9 -f 'sqlite3_rsync --origin' 2>/dev/null; true" >/dev/null 2>&1
fi
set -e

echo "=== $(date '+%F %T') 拉取最新真身库 (sqlite3_rsync 增量) ==="
mkdir -p "$(dirname "$LOCAL_DB")"
for attempt in 1 2 3 4 5; do
  if sqlite3_rsync $SSH_ARG "$HK:$REMOTE_DB" "$LOCAL_DB"; then
    echo "本地库: $(ls -lh "$LOCAL_DB" | awk '{print $5}')"
    # 打印新鲜度
    python3 -c "
import sqlite3
c=sqlite3.connect('$LOCAL_DB')
for t,col in [('kline_daily','date'),('kline_30m','date'),('daily_features','date'),('sentiment_daily','trade_date'),('macro_daily','trade_date'),('hs300_daily','trade_date')]:
    try:
        mx=c.execute(f'SELECT MAX({col}) FROM {t}').fetchone()[0]
        print(f'  {t}: 最新 {mx}')
    except Exception: pass
" 2>/dev/null || true
    echo "=== $(date '+%T') 拉取完成 ==="
    exit 0
  fi
  echo "  ⚠️ 第 $attempt 次拉取失败 (可能远端写库占锁), 等 15s 重试..."
  [ "$attempt" -eq 5 ] && { echo "❌ 拉取 5 次仍失败"; exit 1; }
  sleep 15
done
