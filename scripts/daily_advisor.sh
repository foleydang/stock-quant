#!/bin/bash
# 每日补仓顾问 — 加载已训模型, 对 positions 表里的持仓打分, 输出体检报告
# 部署: 服务器 crontab, 交易日数据更新(20:00)之后运行
#   30 20 * * 1-5 /root/github/stock-quant/scripts/daily_advisor.sh >> /root/github/stock-quant/logs/advisor_cron.log 2>&1
#
# 前置: python/models/add_advisor/model.pkl 必须存在(在 Mac 跑完整训练生成后 scp 上来)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PY_DIR="$PROJECT_DIR/python"
LOG_DIR="$PROJECT_DIR/logs"
REPORT_DIR="$PY_DIR/models/add_advisor"
mkdir -p "$LOG_DIR"

# 周末跳过
if [ "$(date +%u)" -ge 6 ]; then
    echo "[$(date)] 周末, 跳过"
    exit 0
fi

# 选择 python: 优先 python3.11 (兼容新版 lightgbm/sklearn 模型)
if command -v python3.11 &>/dev/null; then
    PY="python3.11"
elif [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
    PY="$PROJECT_DIR/.venv/bin/python"
elif [ -f "$PY_DIR/.venv/bin/python" ]; then
    PY="$PY_DIR/.venv/bin/python"
else
    PY="python3"
fi

echo "=================================================="
echo "[$(date)] 🩺 每日持仓体检开始"

cd "$PY_DIR"

if [ ! -f "$REPORT_DIR/model.pkl" ]; then
    echo "❌ 未找到 $REPORT_DIR/model.pkl — 请先在 Mac 训练并 scp 上来"
    exit 1
fi

$PY strategy/add_advisor_ml.py --score-only

# 归档一份带日期的报告
if [ -f "$REPORT_DIR/holdings_report.txt" ]; then
    cp "$REPORT_DIR/holdings_report.txt" "$REPORT_DIR/report_$(date +%Y%m%d).txt"
fi

echo "[$(date)] ✅ 持仓体检完成, 报告: $REPORT_DIR/holdings_report.txt"

# ================================================================
# 纸面交易引擎 (前瞻验证) — 冻结当日信号→D+1开盘成交→mark-to-market NAV
# 账户A 需先跑全票池扫描刷新 advisor_scan.json(predDate 必须==今日,
# 否则 freeze 会跳过账户A本日调仓, 防用隔夜陈旧信号)。
# 各步骤失败均非致命: 用 if 包裹绕过 set -e, 记日志后继续。
# ================================================================
echo "[$(date)] 📝 纸面交易推进"

if $PY strategy/scan_advisor.py --board all --limit 500; then
    echo "  ✅ 全票池扫描完成 (advisor_scan.json 已刷新)"
else
    echo "  ⚠️ 全票池扫描失败, 账户A本日将跳过调仓"
fi

PAPER_DB="$PY_DIR/data/paper.db"
if [ ! -f "$PAPER_DB" ]; then
    echo "  ⚠️ paper.db 不存在, 请先在主机跑一次: $PY strategy/paper_trading.py --init"
elif $PY strategy/paper_trading.py --advance; then
    echo "  ✅ 纸面账户推进完成"
else
    echo "  ⚠️ 纸面账户推进失败 (见上方日志)"
fi

echo "[$(date)] ✅ 全部完成"
