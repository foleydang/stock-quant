# OSS 数据同步方案

## 1. 开通阿里云 OSS

控制台：https://oss.console.aliyun.com/
- 选择与服务器同地域（如华东-杭州）
- 创建 Bucket：stock-quant-data
- 权限：私有（安全）

## 2. Mac 上配置上传脚本

```bash
# 安装 ossutil
brew install ossutil

# 配置 AccessKey
ossutil config -e oss-cn-hangzhou.aliyuncs.com -i YOUR_ACCESS_KEY_ID -k YOUR_ACCESS_KEY_SECRET

# 上传脚本
#!/bin/bash
# ~/github/stock-quant/stock-quant/scripts/upload_to_oss.sh

BUCKET="oss://stock-quant-data"
DATA_DIR="$HOME/github/stock-quant/stock-quant/python/data"

# 上传所有 CSV 文件
ossutil cp $DATA_DIR/*.csv $BUCKET/data/ -r --update

# 上传数据库文件（可选）
ossutil cp $DATA_DIR/stock_data.db $BUCKET/data/ --update

echo "上传完成：$(date)"
```

## 3. 服务器定时下载脚本

```bash
#!/bin/bash
# /root/github/stock-quant/stock-quant/scripts/download_from_oss.sh

BUCKET="oss://stock-quant-data"
DATA_DIR="/root/github/stock-quant/stock-quant/python/data"

# 下载所有数据文件
ossutil cp $BUCKET/data/*.csv $DATA_DIR/ -r --update
ossutil cp $BUCKET/data/stock_data.db $DATA_DIR/ --update

echo "下载完成：$(date)"
```

## 4. 配置定时任务

Mac crontab（每晚 21:00 上传）：
```bash
0 21 * * 1-5 ~/github/stock-quant/stock-quant/scripts/upload_to_oss.sh
```

服务器 crontab（每晚 22:00 下载）：
```bash
0 22 * * 1-5 /root/github/stock-quant/stock-quant/scripts/download_from_oss.sh
```

## 费用估算

| 项目 | 费用 |
|------|------|
| 存储（1GB以内） | 免费（40GB额度） |
| 上传流量 | 免费 |
| 同地域下载 | 免费 |
| **总计** | **¥0** |

