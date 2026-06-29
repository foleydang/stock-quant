#!/bin/bash
# ============================================================
# LGBM 分钟级择时模型 — 训练 + 部署一键脚本
# 
# 用法：
#   1. 导出数据：    bash export_data.sh
#   2. 拷贝到本地：  scp root@47.242.158.242:/root/github/stock-quant/training-package/kline_30m.csv.gz .
#   3. 本地训练：    bash train.sh
#   4. 上传模型：    bash deploy.sh
# ============================================================

set -e

# === 配置 ===
HORIZON=3         # 预测未来N根30分钟K线
SKIP_BARS=3       # 下采样间隔（跳过相邻bar，减少自相关）
POOL_SIZE=100     # 股票池大小（按成交量选Top N）
QUICK=0           # 0=生产模式(5 ensemble), 1=快速验证

# === 路径 ===
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
MODEL_DIR="${SCRIPT_DIR}/models"

# 如果数据是 CSV 格式，先导入 SQLite
DB_PATH="${DATA_DIR}/stock_data.db"
CSV_FILE="${SCRIPT_DIR}/kline_30m.csv.gz"

if [ ! -f "$DB_PATH" ] && [ -f "$CSV_FILE" ]; then
    echo "📦 导入 CSV 到 SQLite..."
    sqlite3 "$DB_PATH" <<SQL
CREATE TABLE IF NOT EXISTS kline_30m (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    date TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL
);
CREATE INDEX IF NOT EXISTS idx_kline_symbol ON kline_30m(symbol);
CREATE INDEX IF NOT EXISTS idx_kline_date ON kline_30m(datetime);
SQL
    zcat "$CSV_FILE" | sqlite3 -csv -separator ',' "$DB_PATH" ".import /dev/stdin kline_30m"
    echo "✅ 数据导入完成"
fi

# === 训练 ===
echo ""
echo "🚀 开始训练..."
echo "   horizon=${HORIZON}  skip=${SKIP_BARS}  pool=${POOL_SIZE}"
echo ""

cd "$(dirname "$SCRIPT_DIR")/python"

if [ "$QUICK" = "1" ]; then
    python3 strategy/intraday_train.py \
        --horizon "$HORIZON" \
        --skip "$SKIP_BARS" \
        --pool-size "$POOL_SIZE" \
        --quick \
        --db "$DB_PATH"
else
    python3 strategy/intraday_train.py \
        --horizon "$HORIZON" \
        --skip "$SKIP_BARS" \
        --pool-size "$POOL_SIZE" \
        --db "$DB_PATH"
fi

echo ""
echo "✅ 训练完成！模型保存在 models/lgb_hs300/model.pkl"