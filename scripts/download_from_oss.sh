#!/bin/bash
# 从阿里云 OSS 下载数据库文件到本地
# 两端通用：Mac / 服务器均可执行

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/python/data"

# 加载 .env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

mkdir -p "$DATA_DIR"

echo "📥 从 OSS 下载: $(date)"

# stock_data.db (K线/行情数据)
DB_FILE="$DATA_DIR/stock_data.db"
if [ -f "$DB_FILE" ]; then
    cp "$DB_FILE" "$DB_FILE.bak"
    echo "📦 已备份旧 stock_data.db"
fi
ossutil cp "oss://${OSS_BUCKET}/stock-quant/stock_data.db" "$DB_FILE" \
    --update \
    --access-key-id="$OSS_ACCESS_KEY_ID" \
    --access-key-secret="$OSS_ACCESS_KEY_SECRET" \
    --endpoint="$OSS_ENDPOINT"
echo "   stock_data.db: $(ls -lh $DB_FILE | awk '{print $5}')"

# trading.db (持仓/交易记录)
TRADING_DB="$DATA_DIR/trading.db"
if [ -f "$TRADING_DB" ]; then
    cp "$TRADING_DB" "$TRADING_DB.bak"
    echo "📦 已备份旧 trading.db"
fi
ossutil cp "oss://${OSS_BUCKET}/stock-quant/trading.db" "$TRADING_DB" \
    --update \
    --access-key-id="$OSS_ACCESS_KEY_ID" \
    --access-key-secret="$OSS_ACCESS_KEY_SECRET" \
    --endpoint="$OSS_ENDPOINT"
echo "   trading.db: $(ls -lh $TRADING_DB | awk '{print $5}')"

echo "✅ 下载完成: $(date)"