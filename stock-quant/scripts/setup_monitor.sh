#!/bin/bash
# 股票监控快速配置脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_PATH="$PROJECT_ROOT/python"
LOG_PATH="$PROJECT_ROOT/logs"
SCRIPTS_PATH="$PROJECT_ROOT/scripts"

# 创建日志目录
mkdir -p "$LOG_PATH"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║           股票监控系统 - 快速配置                         ║"
echo "╚══════════════════════════════════════════════════════════╝"

echo ""
echo "步骤 1/3: 配置邮件通知"
echo "─────────────────────────────────────────"
echo ""
echo "请选择邮箱服务商:"
echo "  1) QQ 邮箱 (推荐)"
echo "  2) 163 邮箱"
echo "  3) Gmail"
echo "  4) 其他"
echo "  5) 跳过，手动配置"
echo ""
read -p "请选择 [1-5]: " mail_choice

case $mail_choice in
    1) SMTP_SERVER="smtp.qq.com"; SMTP_PORT="465" ;;
    2) SMTP_SERVER="smtp.163.com"; SMTP_PORT="465" ;;
    3) SMTP_SERVER="smtp.gmail.com"; SMTP_PORT="465" ;;
    4) read -p "SMTP 服务器：" SMTP_SERVER; read -p "SMTP 端口：" SMTP_PORT ;;
    5) echo "跳过邮件配置"; SMTP_SERVER="" ;;
    *) echo "无效选择"; exit 1 ;;
esac

if [ -n "$SMTP_SERVER" ]; then
    read -p "发件人邮箱：" SMTP_USERNAME
    read -p "授权码：" SMTP_PASSWORD
    read -p "收件人邮箱：" EMAIL_RECEIVERS
fi

echo ""
echo "步骤 2/3: 配置定时任务"
echo "─────────────────────────────────────────"
echo ""
echo "请选择执行频率:"
echo "  1) 每 30 分钟 (交易时段)"
echo "  2) 每 15 分钟 (交易时段)"
echo "  3) 每小时"
echo "  4) 仅手动执行"
echo ""
read -p "请选择 [1-4]: " freq_choice

case $freq_choice in
    1) CRON_MONITOR="*/30 9-15 * * 1-5"; echo "已选择：每 30 分钟" ;;
    2) CRON_MONITOR="*/15 9-15 * * 1-5"; echo "已选择：每 15 分钟" ;;
    3) CRON_MONITOR="0 9-15 * * 1-5"; echo "已选择：每小时" ;;
    4) CRON_MONITOR=""; echo "已选择：仅手动执行" ;;
    *) echo "无效选择"; exit 1 ;;
esac

# 数据更新默认每晚 20:00
CRON_UPDATE="0 20 * * 1-5"

echo ""
echo "步骤 3/3: 保存配置"
echo "─────────────────────────────────────────"
echo ""

# 保存环境变量到 .env 文件
if [ -n "$SMTP_SERVER" ]; then
    echo "保存邮件配置到 .env 文件..."
    cat > "$PYTHON_PATH/.env" << EOF
# 邮件配置
SMTP_SERVER=$SMTP_SERVER
SMTP_PORT=$SMTP_PORT
SMTP_USERNAME=$SMTP_USERNAME
SMTP_PASSWORD=$SMTP_PASSWORD
EMAIL_RECEIVERS=$EMAIL_RECEIVERS
EOF
    echo "✓ 邮件配置已保存到 $PYTHON_PATH/.env"
fi

# 添加 cron 任务
if [ -n "$CRON_MONITOR" ]; then
    echo ""
    echo "配置 cron 任务..."

    # 清理旧任务
    crontab -l 2>/dev/null | grep -v "stock-quant" | crontab - 2>/dev/null || true

    # 添加新任务
    MONITOR_CMD="$CRON_MONITOR $SCRIPTS_PATH/start.sh monitor >> $LOG_PATH/cron.log 2>&1"
    UPDATE_CMD="$CRON_UPDATE $SCRIPTS_PATH/start.sh update >> $LOG_PATH/cron.log 2>&1"

    (crontab -l 2>/dev/null; echo "# 股票量化系统"; echo "$MONITOR_CMD"; echo "$UPDATE_CMD") | crontab -

    echo "✓ cron 任务已配置"
    echo "  运行 'crontab -l' 查看任务列表"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                    配置完成！                             ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "下一步操作:"
echo ""
echo "1. 测试邮件发送:"
echo "   $SCRIPTS_PATH/start.sh monitor"
echo ""
echo "2. 手动更新数据:"
echo "   $SCRIPTS_PATH/start.sh update"
echo ""
echo "3. 查看日志:"
echo "   tail -f $LOG_PATH/monitor.log"
echo ""
echo "4. 启动 API 服务:"
echo "   $SCRIPTS_PATH/start.sh server"
echo ""