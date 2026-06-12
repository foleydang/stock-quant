#!/bin/bash
# 上传 stock_data.db 到阿里云 OSS
# 服务器端执行（每晚数据更新后上传）

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DB_FILE="$PROJECT_DIR/python/data/stock_data.db"

# 加载 .env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

if [ ! -f "$DB_FILE" ]; then
    echo "❌ $DB_FILE 不存在"
    exit 1
fi

echo "📤 上传到 OSS: $(date)"
echo "   文件大小: $(ls -lh $DB_FILE | awk '{print $5}')"

ossutil cp "$DB_FILE" "oss://${OSS_BUCKET}/stock-quant/stock_data.db" \
    --update \
    --access-key-id="$OSS_ACCESS_KEY_ID" \
    --access-key-secret="$OSS_ACCESS_KEY_SECRET" \
    --endpoint="$OSS_ENDPOINT"

echo "✅ 上传完成: $(date)"