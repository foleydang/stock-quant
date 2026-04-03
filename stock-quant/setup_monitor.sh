#!/bin/bash
# 股票监控快速配置脚本

echo "╔══════════════════════════════════════════════════════════╗"
echo "║           股票监控系统 - 快速配置                         ║"
echo "╚══════════════════════════════════════════════════════════╝"

# 项目路径
PROJECT_PATH="/Users/foleydang/github/stock-quant/stock-quant"
PYTHON_PATH="$PROJECT_PATH/python"
LOG_PATH="$PROJECT_PATH/logs"

# 创建日志目录
mkdir -p "$LOG_PATH"

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
    1)
        SMTP_SERVER="smtp.qq.com"
        SMTP_PORT="465"
        ;;
    2)
        SMTP_SERVER="smtp.163.com"
        SMTP_PORT="465"
        ;;
    3)
        SMTP_SERVER="smtp.gmail.com"
        SMTP_PORT="465"
        ;;
    4)
        read -p "SMTP 服务器：" SMTP_SERVER
        read -p "SMTP 端口：" SMTP_PORT
        ;;
    5)
        echo "跳过邮件配置，稍后请手动编辑 ~/.zshrc"
        SMTP_SERVER=""
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
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
    1)
        CRON_SCHEDULE="*/30 9-15 * * 1-5"
        echo "已选择：每 30 分钟"
        ;;
    2)
        CRON_SCHEDULE="*/15 9-15 * * 1-5"
        echo "已选择：每 15 分钟"
        ;;
    3)
        CRON_SCHEDULE="0 9-15 * * 1-5"
        echo "已选择：每小时"
        ;;
    4)
        CRON_SCHEDULE=""
        echo "已选择：仅手动执行"
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo ""
echo "步骤 3/3: 保存配置"
echo "─────────────────────────────────────────"
echo ""

# 保存环境变量
if [ -n "$SMTP_SERVER" ]; then
    echo "添加环境变量到 ~/.zshrc..."

    # 检查是否已存在配置
    if grep -q "SMTP_SERVER" ~/.zshrc 2>/dev/null; then
        echo "⚠️  检测到已有邮件配置，是否覆盖？"
        read -p "覆盖 [y/n]: " overwrite
        if [ "$overwrite" = "y" ]; then
            # 删除旧配置
            sed -i.bak '/export SMTP_/d' ~/.zshrc
        fi
    fi

    # 添加新配置
    cat >> ~/.zshrc << EOF

# 股票监控邮件配置 (自动添加)
export SMTP_SERVER='$SMTP_SERVER'
export SMTP_PORT='$SMTP_PORT'
export SMTP_USERNAME='$SMTP_USERNAME'
export SMTP_PASSWORD='$SMTP_PASSWORD'
export EMAIL_RECEIVERS='$EMAIL_RECEIVERS'
EOF

    echo "✓ 环境变量已添加到 ~/.zshrc"
    echo "  运行 'source ~/.zshrc' 使配置生效"
fi

# 添加 cron 任务
if [ -n "$CRON_SCHEDULE" ]; then
    echo ""
    echo "添加 cron 任务..."

    # 创建 cron 条目
    CRON_CMD="$CRON_SCHEDULE cd $PYTHON_PATH && /usr/bin/python3 strategy/email_monitor.py >> $LOG_PATH/email_monitor.log 2>&1"

    # 检查是否已存在
    if crontab -l 2>/dev/null | grep -q "email_monitor.py"; then
        echo "⚠️  检测到已有 cron 任务"
        crontab -l | grep -v "email_monitor.py" | crontab -
    fi

    # 添加新任务
    (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -
    echo "✓ cron 任务已添加"
    echo "  运行 'crontab -l' 查看任务列表"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                    配置完成！                             ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "下一步操作:"
echo ""
echo "1. 使环境变量生效:"
echo "   source ~/.zshrc"
echo ""
echo "2. 测试邮件发送:"
echo "   cd $PYTHON_PATH"
echo "   python3 strategy/email_notifier.py"
echo ""
echo "3. 手动运行监控:"
echo "   python3 strategy/email_monitor.py"
echo ""
echo "4. 查看日志:"
echo "   tail -f $LOG_PATH/email_monitor.log"
echo ""
echo "详细说明请查看：$PROJECT_PATH/EMAIL_SETUP.md"
echo ""
