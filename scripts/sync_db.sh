#!/bin/bash
# 从阿里云 OSS 同步 DB 数据到本地
# 两端通用：Mac / 服务器均可执行

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DB="$PROJECT_DIR/python/data/stock_data.db"
DATA_DIR="$(dirname "$DB")"

# 加载 .env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

mkdir -p "$DATA_DIR"

echo "=== 检查 DB: $DB ==="

if [ ! -f "$DB" ]; then
    echo "❌ DB 文件不存在，正在从 OSS 下载..."
    ossutil cp "oss://${OSS_BUCKET}/stock-quant/stock_data.db" "$DB" \
        --update \
        --access-key-id="$OSS_ACCESS_KEY_ID" \
        --access-key-secret="$OSS_ACCESS_KEY_SECRET" \
        --endpoint="$OSS_ENDPOINT"
    echo "✅ 下载完成"
    exit 0
fi

DB_SIZE=$(ls -lh "$DB" | awk '{print $5}')
echo "DB 大小: $DB_SIZE"

python3 -c "
import sqlite3
conn = sqlite3.connect('$DB')
# 日线
r = conn.execute('SELECT COUNT(*), COUNT(DISTINCT symbol) FROM kline_daily').fetchone()
print(f'日线: {r[0]:,} 条, {r[1]} 只股票')
good = conn.execute(\"SELECT COUNT(*) FROM (SELECT symbol FROM kline_daily GROUP BY symbol HAVING COUNT(*) > 200)\").fetchone()[0]
print(f'  完整数据(>200条): {good} 只')
# 30分钟
r = conn.execute('SELECT COUNT(*), COUNT(DISTINCT symbol) FROM kline_30m').fetchone()
print(f'30m:  {r[0]:,} 条, {r[1]} 只股票')
good = conn.execute(\"SELECT COUNT(*) FROM (SELECT symbol FROM kline_30m GROUP BY symbol HAVING COUNT(*) > 200)\").fetchone()[0]
print(f'  完整数据(>200条): {good} 只')
conn.close()
"

DAILY_COUNT=$(python3 -c "
import sqlite3
conn = sqlite3.connect('$DB')
cnt = conn.execute(\"SELECT COUNT(*) FROM (SELECT symbol FROM kline_daily GROUP BY symbol HAVING COUNT(*) > 200)\").fetchone()[0]
conn.close()
print(cnt)
")

if [ "$DAILY_COUNT" -lt 100 ]; then
    echo ""
    echo "⚠️ 日线数据不完整 ($DAILY_COUNT 只)，正在从 OSS 下载..."
    cp "$DB" "$DB.bak"
    ossutil cp "oss://${OSS_BUCKET}/stock-quant/stock_data.db" "$DB" \
        --update \
        --access-key-id="$OSS_ACCESS_KEY_ID" \
        --access-key-secret="$OSS_ACCESS_KEY_SECRET" \
        --endpoint="$OSS_ENDPOINT"
    echo "✅ 下载完成"
else
    echo ""
    echo "✅ 数据完整，可以训练"
fi