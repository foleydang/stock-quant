#!/bin/bash
# ============================================================
# Mac 一键训练脚本
# 用法: bash scripts/mac_full_train.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/python/data"
MODEL_DIR="$PROJECT_DIR/python/models/lgb_hs300"

echo "============================================"
echo "🚀 百炼量化模型训练 - Mac版"
echo "============================================"

# 1. 加载 .env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
    echo "✅ .env 已加载"
else
    echo "❌ 缺少 .env 文件，请先创建"
    exit 1
fi

# 2. 下载数据库
echo ""
echo "📥 从 OSS 下载数据库..."
bash "$SCRIPT_DIR/download_from_oss.sh"

# 3. 安装依赖
echo ""
echo "📦 安装依赖..."
pip install catboost xgboost lightgbm scikit-learn --quiet 2>&1 | tail -1
echo "✅ 依赖就绪"

# 4. 训练
echo ""
echo "============================================"
echo "🎯 开始训练 (369只股票, 市场分治, 3模型集成)"
echo "============================================"
python3 "$PROJECT_DIR/python/strategy/train_enhanced.py" \
    --horizon 3 \
    --db "$DATA_DIR/stock_data.db"

# 5. 部署
echo ""
echo "📋 部署模型..."
mkdir -p "$MODEL_DIR"
cp "$PROJECT_DIR/models/lgb_hs300_enhanced/model.pkl" "$MODEL_DIR/model.pkl"

# 备份旧模型
if [ -f "$MODEL_DIR/model.pkl" ]; then
    BACKUP_DIR="$MODEL_DIR/backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    cp "$MODEL_DIR/model.pkl" "$BACKUP_DIR/"
    echo "  旧模型已备份: $BACKUP_DIR"
fi

echo "✅ 模型已部署到 $MODEL_DIR"

# 6. 测试
echo ""
echo "🧪 测试预测..."
python3 -c "
import pickle, json
with open('$MODEL_DIR/model.pkl', 'rb') as f:
    d = pickle.load(f)
print(f'  版本: {d.get(\"model_version\", \"?\")}')
print(f'  预测周期: h={d.get(\"horizon\", \"?\")}')
print(f'  特征数: {d.get(\"n_features\", len(d.get(\"feature_names\", [])))}')
print(f'  股票数: {d.get(\"n_stocks\", \"?\")}')
print(f'  样本数: {d.get(\"n_samples\", \"?\")}')
print(f'  市场分治: {d.get(\"market_regimes\", \"?\")}')
print(f'  训练时间: {d.get(\"trained_at\", \"?\")}')

cv = d.get('cv_scores', {})
if cv:
    if 'all' in cv:
        for k, v in cv['all'].items():
            print(f'  {k}: IC={v:.4f}')
    else:
        for regime, scores in cv.items():
            print(f'  [{regime}]')
            for k, v in scores.items():
                print(f'    {k}: IC={v:.4f}')
"

echo ""
echo "============================================"
echo "✅ 全部完成!"
echo "============================================"
echo "模型路径: $MODEL_DIR/model.pkl"
echo "提交代码: git add python/models/lgb_hs300/model.pkl && git commit && git push"