#!/bin/bash
# 股票量化系统启动脚本

# 防止重复运行：用 flock 文件锁
LOCK_DIR="/tmp/stock-quant-locks"
mkdir -p "$LOCK_DIR"

# 使用 miniconda Python
PYTHON="/root/miniconda3/bin/python"
# 用法: ./start.sh [command]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_DIR="$PROJECT_ROOT/python"
LOG_DIR="$PROJECT_ROOT/logs"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 加载环境变量
if [ -f "$PYTHON_DIR/.env" ]; then
    export $(cat "$PYTHON_DIR/.env" | grep -v '^#' | xargs)
fi

# 帮助信息
show_help() {
    echo "股票量化系统启动脚本"
    echo ""
    echo "用法: ./start.sh [command]"
    echo ""
    echo "命令:"
    echo "  server      启动 API 服务器"
    echo "  monitor     运行交易监控 (一次)"
    echo "  full        运行完整监控流程"
    echo "  update      更新沪深300数据"
    echo "  strategy    运行策略回测"
    echo "  backtest    运行 LGBM 回测"
    echo "  report      生成报告"
    echo "  setup       配置邮件和定时任务"
    echo "  help        显示帮助信息"
    echo ""
}

# 启动 API 服务器
start_server() {
    echo "启动 API 服务器..."
    cd "$PROJECT_ROOT/api"
    $PYTHON server.py
}

# 运行交易监控
run_monitor() {
    LOCK_FILE="$LOCK_DIR/monitor.lock"
    exec 200>$LOCK_FILE
    if ! flock -n 200; then
        echo "⚠️ monitor 正在运行，跳过本次"
        return 0
    fi
    echo "运行交易监控..."
    cd "$PYTHON_DIR"
    LOG_FILE="$LOG_DIR/monitor.log"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - 开始监控" >> "$LOG_FILE"
    $PYTHON trading_monitor.py >> "$LOG_FILE" 2>&1
    echo "$(date '+%Y-%m-%d %H:%M:%S') - 监控完成" >> "$LOG_FILE"
}

# 运行完整监控
run_full() {
    echo "运行完整监控流程..."
    cd "$PYTHON_DIR"
    LOG_FILE="$LOG_DIR/monitor.log"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - 开始完整监控" >> "$LOG_FILE"
    $PYTHON full_monitor.py --monitor >> "$LOG_FILE" 2>&1
    echo "$(date '+%Y-%m-%d %H:%M:%S') - 完整监控完成" >> "$LOG_FILE"
}

# 更新数据
run_update() {
    LOCK_FILE="$LOCK_DIR/update.lock"
    exec 201>$LOCK_FILE
    if ! flock -n 201; then
        echo "⚠️ update 正在运行，跳过本次"
        return 0
    fi
    echo "更新沪深300数据..."
    cd "$PYTHON_DIR"
    LOG_FILE="$LOG_DIR/hs300_update.log"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - 开始增量更新" >> "$LOG_FILE"
    $PYTHON update_hs300.py >> "$LOG_FILE" 2>&1
    echo "$(date '+%Y-%m-%d %H:%M:%S') - 更新完成" >> "$LOG_FILE"
}

# 运行策略
run_strategy() {
    echo "运行策略..."
    cd "$PYTHON_DIR"
    $PYTHON run_full_strategy.py
}

# 运行回测
run_backtest() {
    echo "运行 LGBM 回测..."
    cd "$PYTHON_DIR"
    $PYTHON lgbm_backtest.py
}

# 生成报告
run_report() {
    echo "生成报告..."
    cd "$PYTHON_DIR"
    $PYTHON generate_report.py
}

# 运行配置脚本
run_setup() {
    cd "$SCRIPT_DIR"
    ./setup_monitor.sh
}

# 主入口
case "$1" in
    server)     start_server ;;
    monitor)    run_monitor ;;
    full)       run_full ;;
    update)     run_update ;;
    strategy)   run_strategy ;;
    backtest)   run_backtest ;;
    report)     run_report ;;
    setup)      run_setup ;;
    help|--help|-h|"") show_help ;;
    *)
        echo "未知命令: $1"
        show_help
        exit 1
        ;;
esac