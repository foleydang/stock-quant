#!/bin/bash
# 导出 kline_30m 数据为 CSV（压缩后约 100-200MB）
DB_PATH=${1:-"../data/stock_data.db"}
OUTPUT=${2:-"kline_30m.csv.gz"}

echo "导出 kline_30m 表..."
sqlite3 -csv -header "$DB_PATH" "SELECT symbol, date, open, high, low, close, volume FROM kline_30m ORDER BY symbol, date;" | gzip > "$OUTPUT"
echo "完成: $OUTPUT ($(du -h $OUTPUT | cut -f1))"
