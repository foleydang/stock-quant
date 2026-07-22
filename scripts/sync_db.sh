#!/bin/bash
# 从阿里云 OSS 同步 DB 数据到本地 (按交易日增量分片, 见 python/strategy/oss_incr.py)
# 依赖: pip install oss2 pyarrow pandas
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DB="$PROJECT_DIR/python/data/stock_data.db"
DATA_DIR="$(dirname "$DB")"
PY="${PYTHON:-python3}"

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

mkdir -p "$DATA_DIR"

echo "=== 检查 DB: $DB ==="

if [ -f "$DB" ]; then
    DB_SIZE=$(ls -lh "$DB" | awk '{print $5}')
    echo "DB 大小: $DB_SIZE"

    python3 -c "
import sqlite3
conn = sqlite3.connect('$DB')
r = conn.execute('SELECT COUNT(*), COUNT(DISTINCT symbol) FROM kline_daily').fetchone()
print(f'日线: {r[0]:,} 条, {r[1]} 只股票')
good = conn.execute(\"SELECT COUNT(*) FROM (SELECT symbol FROM kline_daily GROUP BY symbol HAVING COUNT(*) > 200)\").fetchone()[0]
print(f'  完整数据(>200条): {good} 只')
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
else
    echo "DB 文件不存在"
    DAILY_COUNT=0
fi

# 库为空/几乎为空 -> 先整库 bootstrap (一次性), 否则只拉增量
if [ ! -f "$DB" ] || [ "$DAILY_COUNT" -lt 100 ]; then
    echo ""
    echo "⬇️ 本地库为空, 整库 bootstrap..."
    (cd "$PROJECT_DIR/python" && $PY strategy/oss_incr.py download --full)
fi

echo ""
echo "⬇️ 拉取增量分片..."
(cd "$PROJECT_DIR/python" && $PY strategy/oss_incr.py download)
echo "✅ 增量同步完成"
echo ""

# 同步 LSTM embeddings (如果远程有, 与表增量无关, 单独处理)
EMB="$PROJECT_DIR/python/data/lstm_embeddings.pkl"
OSS_EMB_KEY="stock-quant/lstm_embeddings.pkl"
if ! python3 -c "
import oss2, os
endpoint = os.environ.get('OSS_ENDPOINT', 'https://oss-cn-hangzhou.aliyuncs.com')
bucket = oss2.Bucket(oss2.Auth(os.environ.get('OSS_ACCESS_KEY_ID',''), os.environ.get('OSS_ACCESS_KEY_SECRET','')), endpoint, os.environ.get('OSS_BUCKET',''))
print('exists' if bucket.object_exists('$OSS_EMB_KEY') else 'no')
" 2>/dev/null | grep -q exists; then
    echo "⏭️ LSTM embeddings 未上传到 OSS, 跳过"
elif [ ! -f "$EMB" ] || [ "$(stat -f%z "$EMB" 2>/dev/null || stat -c%s "$EMB" 2>/dev/null || echo 0)" -lt 1000000 ]; then
    echo "⬇️ 正在下载 LSTM embeddings..."
    python3 -c "
import oss2, os
endpoint = os.environ.get('OSS_ENDPOINT', 'https://oss-cn-hangzhou.aliyuncs.com')
bucket = oss2.Bucket(oss2.Auth(os.environ.get('OSS_ACCESS_KEY_ID',''), os.environ.get('OSS_ACCESS_KEY_SECRET','')), endpoint, os.environ.get('OSS_BUCKET',''))
bucket.get_object_to_file('$OSS_EMB_KEY', '$EMB')
print(f'✅ LSTM embeddings 下载完成 ({os.path.getsize(\"$EMB\")/1024/1024:.0f}MB)')
"
else
    EMB_SIZE=$(ls -lh "$EMB" 2>/dev/null | awk '{print $5}')
    echo "✅ LSTM embeddings 已存在 ($EMB_SIZE)"
fi
