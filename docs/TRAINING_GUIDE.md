# Mac 本地训练指南

## 前置条件

- Mac M4 Pro (24GB+)
- Python 3.12+
- 已配置 OSS 环境变量 (`.env` 文件)

## 一次性准备

```bash
cd ~/github/stock-quant
git pull
bash scripts/sync_db.sh          # 从 OSS 下载 DB (~1.1GB)
pip install -r requirements.txt
```

## 快速验证（可选）

```bash
cd python
python strategy/train.py --model daily --quick
# 2 模型, 1000 树, ~5 分钟跑完, 确认环境没问题
```

## 生产训练

```bash
python strategy/train.py --model daily   # 日线模型, 预计 30-60 分钟
python strategy/train.py --model 30m    # 30m 模型, 预计 2-4 小时
```

产出：
```
models/lgb_daily/model.pkl   # 日线 5模型 ensemble (~80-150MB)
models/lgb_30m/model.pkl    # 30m 5模型 ensemble (~100-200MB)
```

## 上传到服务器

```bash
scp models/lgb_daily/model.pkl root@47.242.158.242:/root/github/stock-quant/python/models/lgb_daily/
scp models/lgb_30m/model.pkl root@47.242.158.242:/root/github/stock-quant/python/models/lgb_30m/
```

## 日常更新

```bash
cd ~/github/stock-quant
git pull
bash scripts/sync_db.sh       # 拉最新数据
cd python
python strategy/train.py --model daily
python strategy/train.py --model 30m
# 上传新模型...

## 技术规格

详细参数见 `docs/LGBM_MODEL_SPEC.md`