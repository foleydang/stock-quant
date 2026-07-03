#!/bin/bash
# Mac 端数据准备脚本
# 从 OSS 下载数据库 + 提取 LSTM embeddings
# 用法: bash scripts/setup_mac_training.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$ROOT/python/data"

echo "=== Mac 端训练数据准备 ==="
echo "ROOT: $ROOT"
echo ""

# 1. 下载数据库
echo "📦 1. 下载数据库..."

mkdir -p "$DATA_DIR"

if [ -f "$DATA_DIR/stock_data.db" ]; then
    SIZE=$(ls -lh "$DATA_DIR/stock_data.db" | awk '{print $5}')
    echo "   数据库已存在 ($SIZE)"
    read -p "   是否重新下载? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   跳过下载"
    else
        bash "$SCRIPT_DIR/download_from_oss.sh"
    fi
else
    echo "   调用 download_from_oss.sh ..."
    if [ -f "$SCRIPT_DIR/download_from_oss.sh" ]; then
        bash "$SCRIPT_DIR/download_from_oss.sh"
    else
        echo "   ❌ download_from_oss.sh 不存在"
        echo "   请先安装 ossutil 并配置 .env:"
        echo "   brew install ossutil"
        exit 1
    fi
fi

# 2. 提取 LSTM embeddings
echo ""
echo "🔮 2. LSTM embeddings..."

if [ -f "$DATA_DIR/lstm_embeddings.pkl" ]; then
    echo "   已存在: $(ls -lh $DATA_DIR/lstm_embeddings.pkl | awk '{print $5}')"
else
    echo "   提取中... (M4 Pro 约 10分钟)"
    cd "$ROOT/python"
    python3 strategy/extract_lstm_embeddings_30m.py
    cd "$ROOT"
fi

# 3. 验证
echo ""
echo "✅ 准备完成:"
echo "   DB: $(ls -lh $DATA_DIR/stock_data.db 2>/dev/null | awk '{print $5}')"
echo "   LSTM: $(ls -lh $DATA_DIR/lstm_embeddings.pkl 2>/dev/null | awk '{print $5}')"

echo ""
echo "=== 开始训练 ==="
echo "  python3 python/strategy/retrain_all_mac.py --quick      # 快速验证"
echo "  python3 python/strategy/retrain_all_mac.py              # 完整训练"