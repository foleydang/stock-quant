#!/bin/bash
# 从阿里云 OSS 同步 Qlib .bin 数据到本地
# 用法: sh scripts/sync_qlib_data.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
QLIB_BIN_DIR="$HOME/.qlib/qlib_data/cn_30min/bin"
QLIB_TGZ="$PROJECT_DIR/../qlib_cn_30min_bin.tar.gz"
OSS_KEY="stock-quant/qlib_cn_30min_bin.tar.gz"

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

echo "=== 检查 Qlib .bin 数据 ==="

# 检查是否已有数据
if [ -d "$QLIB_BIN_DIR" ] && [ "$(ls -1 "$QLIB_BIN_DIR" 2>/dev/null | wc -l)" -gt 0 ]; then
    BIN_COUNT=$(find "$QLIB_BIN_DIR" -type f | wc -l | tr -d ' ')
    BIN_SIZE=$(du -sh "$QLIB_BIN_DIR" | awk '{print $1}')
    echo "✅ .bin 数据已存在: $BIN_COUNT 个文件, $BIN_SIZE"
    echo "   路径: $QLIB_BIN_DIR"
    echo ""
    echo "跳过下载 (如需重新下载请先删除: rm -rf $QLIB_BIN_DIR)"
    exit 0
fi

echo "本地无 .bin 数据"

# 检查 OSS 上是否有
echo ""
echo "🔍 检查 OSS..."
REMOTE_EXISTS=$(python3 -c "
import oss2, os
endpoint = os.environ.get('OSS_ENDPOINT', 'https://oss-cn-hangzhou.aliyuncs.com')
bucket = oss2.Bucket(oss2.Auth(os.environ.get('OSS_ACCESS_KEY_ID',''), os.environ.get('OSS_ACCESS_KEY_SECRET','')), endpoint, os.environ.get('OSS_BUCKET',''))
print('exists' if bucket.object_exists('$OSS_KEY') else 'no')
" 2>/dev/null)

if [ "$REMOTE_EXISTS" != "exists" ]; then
    echo "⏭️ Qlib .bin 数据未上传到 OSS, 跳过"
    exit 0
fi

# 下载
echo ""
echo "⬇️ 正在从 OSS 下载..."
python3 -c "
import oss2, sys, os, time, tarfile

endpoint = os.environ.get('OSS_ENDPOINT', 'https://oss-cn-hangzhou.aliyuncs.com')
bucket_name = os.environ.get('OSS_BUCKET', '')
ak = os.environ.get('OSS_ACCESS_KEY_ID', '')
sk = os.environ.get('OSS_ACCESS_KEY_SECRET', '')
if not bucket_name: sys.exit('❌ 缺少 OSS_BUCKET 环境变量')

auth = oss2.Auth(ak, sk)
bucket = oss2.Bucket(auth, endpoint, bucket_name)
key = '$OSS_KEY'

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
        if elapsed > 0.3 or pct >= 100:
            speed = (consumed - last[0]) / 1024 / 1024 / elapsed if elapsed > 0 else 0
            bar_len = 30
            filled = int(bar_len * consumed / total_bytes)
            bar = '█' * filled + '░' * (bar_len - filled)
            sys.stdout.write(f'\r  [{bar}] {pct:3d}% {mb:5.0f}MB' + (f' {speed:.1f}MB/s' if speed > 0.1 else '          '))
            sys.stdout.flush()
            last[0] = consumed
            last[1] = time.time()

bucket.get_object_to_file(key, '$QLIB_TGZ', progress_callback=progress)
print(f'\n✅ 下载完成 ({total/1024/1024:.0f}MB)')
"

# 解压到 ~/.qlib/
echo ""
echo "📦 解压..."
mkdir -p "$HOME/.qlib/qlib_data/cn_30min"
tar xzf "$QLIB_TGZ" -C "$HOME/.qlib/qlib_data/cn_30min/"
echo "   解压完成"

# 清理压缩包
rm -f "$QLIB_TGZ"

# 验证
BIN_COUNT=$(find "$QLIB_BIN_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')
BIN_SIZE=$(du -sh "$QLIB_BIN_DIR" 2>/dev/null | awk '{print $1}')
echo ""
echo "✅ 同步完成"
echo "   文件数: $BIN_COUNT"
echo "   大小: $BIN_SIZE"
echo "   路径: $QLIB_BIN_DIR"