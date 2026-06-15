#!/bin/bash
# 上传数据库文件到阿里云 OSS
# 服务器端执行（每晚数据更新后上传）

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/python/data"

# 加载 .env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

echo "📤 上传到 OSS: $(date)"

# stock_data.db (K线/行情数据)
DB_FILE="$DATA_DIR/stock_data.db"
if [ -f "$DB_FILE" ]; then
    echo "   stock_data.db: $(ls -lh $DB_FILE | awk '{print $5}')"
    ossutil cp "$DB_FILE" "oss://${OSS_BUCKET}/stock-quant/stock_data.db" \
        --update \
        --access-key-id="$OSS_ACCESS_KEY_ID" \
        --access-key-secret="$OSS_ACCESS_KEY_SECRET" \
        --endpoint="$OSS_ENDPOINT"
else
    echo "   ⚠️ stock_data.db 不存在，跳过"
fi

# trading.db (持仓/交易记录)
TRADING_DB="$DATA_DIR/trading.db"
if [ -f "$TRADING_DB" ]; then
    echo "   trading.db: $(ls -lh $TRADING_DB | awk '{print $5}')"
    ossutil cp "$TRADING_DB" "oss://${OSS_BUCKET}/stock-quant/trading.db" \
        --update \
        --access-key-id="$OSS_ACCESS_KEY_ID" \
        --access-key-secret="$OSS_ACCESS_KEY_SECRET" \
        --endpoint="$OSS_ENDPOINT"
else
    echo "   ⚠️ trading.db 不存在，跳过"
fi

echo "✅ 上传完成: $(date)"

# LSTM embeddings (如果本地有)
EMB_FILE="$DATA_DIR/lstm_embeddings.pkl"
if [ -f "$EMB_FILE" ]; then
    echo "   lstm_embeddings.pkl: $(ls -lh $EMB_FILE | awk '{print $5}')"
    ossutil cp "$EMB_FILE" "oss://${OSS_BUCKET}/stock-quant/lstm_embeddings.pkl" \
        --update \
        --access-key-id="$OSS_ACCESS_KEY_ID" \
        --access-key-secret="$OSS_ACCESS_KEY_SECRET" \
        --endpoint="$OSS_ENDPOINT"
fi