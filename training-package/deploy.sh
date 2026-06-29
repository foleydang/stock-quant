#!/bin/bash
# 部署新模型到服务器
# 用法: bash deploy.sh [服务器IP]

SERVER=${1:-"47.242.158.242"}
MODEL_FILE="models/lgb_hs300/model.pkl"
BACKUP_DIR="models/lgb_hs300/backup_$(date +%Y%m%d_%H%M)"

echo "📤 上传模型到 ${SERVER}..."
echo ""

# 1. 备份旧模型
echo "💾 备份旧模型..."
ssh root@${SERVER} "mkdir -p /root/github/stock-quant/python/${BACKUP_DIR} && cp /root/github/stock-quant/python/models/lgb_hs300/model.pkl /root/github/stock-quant/python/${BACKUP_DIR}/"

# 2. 上传新模型
echo "📤 上传新模型..."
scp "${MODEL_FILE}" root@${SERVER}:/root/github/stock-quant/python/models/lgb_hs300/model.pkl

# 3. 重启服务
echo "🔄 重启 stock-feishu-bot..."
ssh root@${SERVER} "systemctl restart stock-feishu-bot && sleep 3 && systemctl status stock-feishu-bot --no-pager | head -5"

# 4. 验证
echo ""
echo "🔍 验证模型..."
ssh root@${SERVER} "python3 -c '
import pickle, warnings
warnings.filterwarnings(\"ignore\")
with open(\"/root/github/stock-quant/python/models/lgb_hs300/model.pkl\", \"rb\") as f:
    m = pickle.load(f)
print(f\"  模型版本: {m.get(\"model_version\", \"?\")}\")
print(f\"  训练时间: {m.get(\"trained_at\", \"?\")}\")
print(f\"  CV Spearman: {m.get(\"cv_spearman\", \"?\"):.4f}\")
print(f\"  特征数: {m.get(\"n_features_selected\", \"?\")}\")
print(f\"  模型数: {len(m.get(\"models\", m.get(\"model\", [])))}\")
'"

echo ""
echo "✅ 部署完成！"