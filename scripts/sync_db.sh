#!/bin/bash
# 从阿里云 OSS 同步 DB 数据到本地
# 依赖: pip install oss2
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DB="$PROJECT_DIR/python/data/stock_data.db"
DATA_DIR="$(dirname "$DB")"
OSS_KEY="stock-quant/stock_data.db"

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
    MACRO_COUNT=$(python3 -c "
import sqlite3
conn = sqlite3.connect('$DB')
try:
    cnt = conn.execute('SELECT COUNT(*) FROM macro_daily').fetchone()[0]
    conn.close()
    print(cnt)
except Exception:
    conn.close()
    print(0)
")
else
    echo "DB 文件不存在"
    DAILY_COUNT=0
fi

    echo "宏观数据: $MACRO_COUNT 条"
fi

if [ "$DAILY_COUNT" -lt 100 ] || [ "${MACRO_COUNT:-0}" -lt 100 ]; then
    echo ""
    echo "⬇️ 正在从 OSS 下载..."
    python3 -c "
import oss2, sys, os, time

endpoint = os.environ.get('OSS_ENDPOINT', 'https://oss-cn-hangzhou.aliyuncs.com')
bucket_name = os.environ.get('OSS_BUCKET', '')
ak = os.environ.get('OSS_ACCESS_KEY_ID', '')
sk = os.environ.get('OSS_ACCESS_KEY_SECRET', '')
if not bucket_name: sys.exit('❌ 缺少 OSS_BUCKET 环境变量')

auth = oss2.Auth(ak, sk)
bucket = oss2.Bucket(auth, endpoint, bucket_name)
key = os.environ.get('OSS_BUCKET_KEY', '$OSS_KEY')

# 获取文件大小
meta = bucket.get_object_meta(key)
total = meta.content_length
print(f'  文件: {key} ({total/1024/1024:.0f}MB)')

# 进度回调
last = [0, time.time()]
def progress(consumed, total_bytes):
    if total_bytes:
        pct = consumed * 100 // total_bytes
        mb = consumed / 1024 / 1024
        elapsed = time.time() - last[1]
        if elapsed > 0.3 or pct >= 100:  # 每秒更新~3次
            speed = (consumed - last[0]) / 1024 / 1024 / elapsed if elapsed > 0 else 0
            bar_len = 30
            filled = int(bar_len * consumed / total_bytes)
            bar = '█' * filled + '░' * (bar_len - filled)
            sys.stdout.write(f'\r  [{bar}] {pct:3d}% {mb:5.0f}MB' + (f' {speed:.1f}MB/s' if speed > 0.1 else '          '))
            sys.stdout.flush()
            last[0] = consumed
            last[1] = time.time()

bucket.get_object_to_file(key, '$DB', progress_callback=progress)
print(f'\n✅ 下载完成')
"
    echo "✅ 同步完成"
else
    echo ""
    echo "✅ 数据完整，可以训练"
fi