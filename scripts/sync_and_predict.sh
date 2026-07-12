#!/bin/bash
# 盘后任务：数据同步 + 预测 + OSS上传
# 由 cron 触发，不要依赖 subagent 去理解复杂指令
set -e

export TUSHARE_TOKEN="7a9014b18909e8cbce5109d7175f7b21ce37354eaff2371db0da2c58"
cd /root/github/stock-quant

echo "=== $(date '+%Y-%m-%d %H:%M:%S') 盘后同步开始 ==="

# 1. A股日线数据同步 (Tushare)
echo "📡 拉取A股日线 (Tushare)..."
cd python && /root/miniconda3/bin/python strategy/data_sync.py --daily-only && cd ..
echo ""

# 2. 港股+ETF数据同步 (yfinance, 需要 miniconda python)
echo "📡 拉取港股/ETF日线 (yfinance)..."
/root/miniconda3/bin/python scripts/sync_hk_etf.py
echo ""

# 3. 跑预测
echo "📊 跑预测..."
cd python && /root/miniconda3/bin/python strategy/predict_today_batched.py --batch 500 && cd ..
echo ""

# 4. 上传 OSS
echo "📤 上传 OSS..."
bash scripts/upload_to_oss.sh 2>&1 | tail -3
echo ""

echo "=== $(date '+%H:%M:%S') 盘后任务完成 ==="