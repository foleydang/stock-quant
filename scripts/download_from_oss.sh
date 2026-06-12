#!/bin/bash
# 从阿里云 OSS 下载 stock_data.db 到本地
# 两端通用：Mac / 服务器均可执行

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DB_FILE="$PROJECT_DIR/python/data/stock_data.db"
DATA_DIR="$(dirname "$DB_FILE")"

# 加载 .env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

mkdir -p "$DATA_DIR"

# 备份旧文件
if [ -f "$DB_FILE" ]; then
    cp "$DB_FILE" "$DB_FILE.bak"
    echo "📦 已备份旧数据库"
fi

echo "📥 从 OSS 下载: $(date)"

ossutil cp "oss://${OSS_BUCKET}/stock-quant/stock_data.db" "$DB_FILE" \
    --update \
    --access-key-id="$OSS_ACCESS_KEY_ID" \
    --access-key-secret="$OSS_ACCESS_KEY_SECRET" \
    --endpoint="$OSS_ENDPOINT"

echo "   文件大小: $(ls -lh $DB_FILE | awk '{print $5}')"
echo "✅ 下载完成: $(date)"