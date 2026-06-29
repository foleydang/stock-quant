# LGBM 分钟级择时模型训练包

## 文件说明

```
training-package/
├── README.md          # 本文件
├── train.sh           # 训练脚本（一键）
├── deploy.sh          # 部署脚本（上传模型+重启服务）
├── export_data.sh     # 数据导出脚本（服务器端运行）
├── kline_30m.csv.gz   # 训练数据（需从服务器导出）
├── data/              # 本地 SQLite 数据库目录
├── models/            # 模型输出目录
│   ├── lgb_hs300/     # 30分钟模型
│   └── lgb_daily/     # 日线模型（特征工程依赖）
└── strategy/          # 训练代码
    ├── intraday_train.py
    └── intraday_features.py
```

## 快速开始

### 第一步：导出数据

在服务器上运行：
```bash
cd /root/github/stock-quant/training-package
bash export_data.sh
# 生成 kline_30m.csv.gz（约 100-200MB）
```

### 第二步：下载到本地

```bash
scp root@47.242.158.242:/root/github/stock-quant/training-package/kline_30m.csv.gz .
```

### 第三步：本地训练

```bash
# 确保本地有 Python 3.10+ 和依赖
pip install lightgbm pandas numpy scikit-learn scipy joblib tqdm

# 训练（需要 8GB+ 内存）
bash train.sh
```

### 第四步：部署

```bash
bash deploy.sh
```

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--horizon` | 3 | 预测未来N根30分钟K线 |
| `--skip` | 3 | 下采样间隔 |
| `--pool-size` | 100 | 股票池大小（按成交量Top N） |
| `--quick` | - | 快速验证模式（1模型，少树） |

## 模型评估标准

| Spearman | 评级 |
|----------|------|
| < 0.05 | ❌ 无效 |
| 0.05-0.15 | ⚠️ 可用 |
| 0.15+ | ✅ 优秀 |

目标：训练到 Spearman > 0.15 再部署。

## 注意事项

1. 服务器内存只有 1.8GB，无法本地训练
2. 本地训练建议 8GB+ 内存，100只股票约需 4-6GB
3. 日线模型 `lgb_daily/model.pkl` 是特征工程依赖，不需要重新训练
4. 部署后会自动重启 stock-feishu-bot 服务