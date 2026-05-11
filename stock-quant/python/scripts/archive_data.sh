#!/bin/bash
# 归档3个月前的历史数据

DB_PATH="/root/github/stock-quant/stock-quant/python/data/stock_data.db"
ARCHIVE_DIR="/root/github/stock-quant/stock-quant/python/data/archive"

# 创建归档目录
mkdir -p "$ARCHIVE_DIR"

# 备份数据库
sqlite3 "$DB_PATH" ".backup '$ARCHIVE_DIR/backup_$(date +%Y%m%d).db'"

# 删除3个月前的数据（保留索引数据）
# sqlite3 "$DB_PATH" "DELETE FROM kline_30m WHERE date < date('now', '-3 months')"

echo "$(date): 归档完成" >> "$ARCHIVE_DIR/archive.log"
