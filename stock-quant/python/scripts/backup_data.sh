#!/bin/bash
# 数据库备份脚本 - 保护珍贵的历史数据

DB_PATH="/root/github/stock-quant/stock-quant/python/data/stock_data.db"
BACKUP_DIR="/root/github/stock-quant/stock-quant/python/data/backup"

# 创建备份目录
mkdir -p "$BACKUP_DIR"

# 生成备份文件名（带日期）
BACKUP_FILE="$BACKUP_DIR/stock_data_$(date +%Y%m%d).db"

# SQLite备份（不影响原数据库）
sqlite3 "$DB_PATH" ".backup '$BACKUP_FILE'"

# 计算备份文件大小
BACKUP_SIZE=$(ls -lh "$BACKUP_FILE" | awk '{print $5}')
DB_SIZE=$(ls -lh "$DB_PATH" | awk '{print $5}')

echo "$(date): 备份完成 - 原库 $DB_SIZE -> 备份 $BACKUP_SIZE" >> "$BACKUP_DIR/backup.log"

# 保留最近30天的备份（不删除历史数据）
find "$BACKUP_DIR" -name "*.db" -mtime +30 -delete

echo "$(date): 清理30天前的备份文件" >> "$BACKUP_DIR/backup.log"
