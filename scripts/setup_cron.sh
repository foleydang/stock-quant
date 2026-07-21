#!/bin/bash
# ============================================================
# 安装定时任务 — 每日数据+预测流水线 + 周度模型重训
#
# 用法: bash scripts/setup_cron.sh        # 安装(合并已有 crontab)
#        bash scripts/setup_cron.sh --remove  # 移除本项目条目
#
# 会在当前用户的 crontab 里追加两条(若已存在则跳过):
#   - 每个交易日 18:00  跑 daily_pipeline.sh  (数据更新 + 预测 + ETF信号)
#   - 每周日   23:00  跑 weekly_retrain.sh    (重训 ETF159792 + 增强 HS300 模型)
#
# 在服务器上执行: bash scripts/setup_cron.sh
# ============================================================
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MARK="# stock-quant-auto"

CRON_FILE="$(mktemp)"
trap 'rm -f "$CRON_FILE"' EXIT

# 取现有 crontab(允许为空)
crontab -l 2>/dev/null > "$CRON_FILE" || true

if [ "$1" = "--remove" ]; then
    # 删除本项目的条目(以 MARK 开头行到下一空行)
    sed -i '' "/$MARK/,/^\$/d" "$CRON_FILE" 2>/dev/null || sed -i "/$MARK/,/^\$/d" "$CRON_FILE"
    crontab "$CRON_FILE"
    echo "✅ 已移除 stock-quant 定时任务"
    exit 0
fi

# 若已安装则跳过
if grep -q "$MARK" "$CRON_FILE" 2>/dev/null; then
    echo "ℹ️ 已存在 stock-quant 定时任务,跳过。如需重装先: bash scripts/setup_cron.sh --remove"
    exit 0
fi

cat >> "$CRON_FILE" <<EOF

$MARK — 不要手动改这几行,改 scripts/ 再跑 setup_cron.sh
# 交易日 18:00 数据+预测+ETF信号(周日不出,周一盘前有数据)
0 18 * * 1-5 cd "$PROJECT_ROOT" && bash scripts/daily_pipeline.sh >> logs/cron_$(date +\%Y\%m\%d).log 2>&1
# 周日 23:00 滚动重训(周一盘前用上新模型)
0 23 * * 0 cd "$PROJECT_ROOT" && bash scripts/weekly_retrain.sh >> logs/retrain_$(date +\%Y\%m\%d).log 2>&1
$MARK-end
EOF

crontab "$CRON_FILE"
echo "✅ 已安装定时任务:"
echo "   - 18:00 (周一~周五) daily_pipeline.sh"
echo "   - 23:00 (周日)       weekly_retrain.sh"
echo "   查看: crontab -l"
