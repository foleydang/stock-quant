#!/bin/bash
# 股票量化策略启动脚本
# 使用 Homebrew 安装的 Python 3.14

PYTHON3="/opt/homebrew/opt/python@3.14/bin/python3.14"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 检查 Python 版本
echo "Python 版本：$($PYTHON3 --version)"

# 切换到项目目录
cd "$PROJECT_DIR"

# 运行脚本（传入第一个参数）
if [ -n "$1" ]; then
    exec "$PYTHON3" "$@"
else
    echo "用法：./run.sh <脚本名> [参数]"
    echo "例如：./run.sh strategy/email_monitor.py"
fi
