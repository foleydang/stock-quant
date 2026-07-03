#!/bin/bash
# Mac 端数据准备脚本
# 从 OSS 下载数据库 + 从服务器同步 LSTM embeddings
# 用法: bash scripts/setup_mac_training.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$ROOT/data"

echo "=== Mac 端训练数据准备 ==="
echo "ROOT: $ROOT"
echo ""

# 1. 从 OSS 下载数据库
echo "📦 1. 从 OSS 下载数据库..."
mkdir -p "$DATA_DIR"

if [ -f "$DATA_DIR/stock_data.db" ]; then
    echo "   数据库已存在: $(ls -lh $DATA_DIR/stock_data.db | awk '{print $5}')"
    echo "   跳过下载 (如需更新请手动删除后重试)"
else
    # 需要 ossutil 或从服务器中转
    echo "   ⚠️ 请手动下载数据库，方式二选一:"
    echo ""
    echo "   方式A: 从服务器 SCP (推荐)"
    echo "   scp root@47.242.158.242:~/github/stock-quant/data/stock_data.db $DATA_DIR/"
    echo ""
    echo "   方式B: 从 OSS 下载 (需要 ossutil)"
    echo "   ossutil cp oss://yanten-data/stock-quant/stock_data.db $DATA_DIR/"
    echo ""
    exit 1
fi

# 2. 提取 LSTM embeddings
echo ""
echo "🔮 2. 提取 LSTM embeddings..."

if [ -f "$DATA_DIR/lstm_embeddings.pkl" ]; then
    echo "   embeddings 已存在: $(ls -lh $DATA_DIR/lstm_embeddings.pkl | awk '{print $5}')"
else
    echo "   提取中... (M4 Pro 约 10分钟)"
    cd "$ROOT/python"
    python3 strategy/extract_lstm_embeddings_30m.py
fi

# 3. 验证
echo ""
echo "✅ 3. 验证:"
echo "   数据库: $(ls -lh $DATA_DIR/stock_data.db 2>/dev/null | awk '{print $5}')"
echo "   embeddings: $(ls -lh $DATA_DIR/lstm_embeddings.pkl 2>/dev/null | awk '{print $5}')"

echo ""
echo "=== 准备完成，可以开始训练 ==="
echo "  python3 python/strategy/retrain_all_mac.py --quick      # 快速验证"
echo "  python3 python/strategy/retrain_all_mac.py              # 完整训练"
echo "  python3 python/strategy/retrain_all_mac.py --30m-only   # 只训30m"