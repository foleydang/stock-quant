# LGBM 金融量化两层模型技术方案 v2

## 一、概述

基于 LightGBM 构建双层量化回归模型，用于 A 股择时选股。日线模型负责选股（α层），30分钟模型负责择时（γ层），通过 Bagging Ensemble 提升稳定性。

**v2 核心变更：目标从 3 分类改为回归。**
- 分类的致命问题：离散化丢失收益率排序信息，40%「震荡」类为纯噪声，导致 early stopping 极早触发（30 棵树就停）
- 回归的优势：保留连续收益率信息，支持排序选股，与业界标准（Qlib / 微软）一致

## 二、模型架构

### 双层设计

```
输入: K线数据 (OHLCV)
  │
  ├─ 日线模型 (α选股层)
  │   ├─ 数据: kline_daily (2014-2026, 373只股票)
  │   ├─ 目标: 预测5日后收益率 (回归, 连续值)
  │   ├─ 输出: 单只股票预期收益率
  │   └─ 用途: 按预测收益率排序选股
  │
  └─ 30分钟模型 (γ择时层)
      ├─ 数据: kline_30m (2020-2026, 372只股票)
      ├─ 目标: 预测3根K线后收益率 (回归, 连续值)
      ├─ 输出: 短期预期收益率
      └─ 用途: 买卖点判断
```

### 评估指标：IC（Information Coefficient）

```
不再使用 F1/Accuracy（分类指标），改用：

  1. Rank IC（主指标）
     = Spearman 相关系数(预测值, 实际收益率)
     越大越好，>0.05 有经济意义

  2. 分组回测（辅助指标）
     = 按预测值分5组，多头组收益 - 空头组收益
     验证模型排序能力
```

### Bagging Ensemble

```
5个独立LGBM模型并行训练
  seed_42  seed_123  seed_456  seed_789  seed_1024
     │        │        │        │        │
     └────────┴────────┴───┬────┴────────┘
                           │
                    预测均值 (5个模型预测的平均)
                           │
                    最终预测 (连续收益率)
```

## 三、训练参数

### LGBM 核心参数

| 参数 | 值 | 说明 |
|------|----|------|
| objective | regression | 回归，预测连续收益率 |
| metric | l2 | 均方误差 (MSE) |
| boosting_type | gbdt | 梯度提升树 |
| num_leaves | 255 | 叶子数拉满 |
| max_depth | 16 | 最大深度 |
| learning_rate | 0.01 | 小学习率 → 多树 → 大模型 |
| n_estimators | 20000 | 最大树数（early_stopping 自动停） |
| early_stopping_rounds | 200 | 耐心等收敛 |
| subsample | 0.75 | 行采样 |
| colsample_bytree | 0.65 | 列采样 |
| subsample_freq | 5 | 每5轮重采样 |
| reg_alpha | 0.05 | L1正则化 |
| reg_lambda | 0.05 | L2正则化 |
| min_child_samples | 30 | 叶子最小样本数 |
| min_split_gain | 0.005 | 最小分裂增益 |

### 并行策略

```
Mac M4 Pro (16核):
  - joblib.Parallel(n_jobs=5): 5个模型并行训练
  - 单模型内部: n_jobs=3
  - 总利用率: 5×3=15核，留1核给系统
```

### 5个独立种子

```python
SEEDS = [42, 123, 456, 789, 1024]
```

## 四、特征体系

### 特征类别

| 类别 | 数量 | 说明 |
|------|------|------|
| Price（价格） | ~120 | 收益率、波动率、均线、RSI、MACD、KDJ、布林带、ATR、ADX |
| Volume（成交量） | ~30 | 量比、OBV、量趋势、换手率 |
| Pattern（形态） | ~15 | K线形态、影线、跳空、突破 |
| Momentum（动量） | ~15 | 动量加速度、衰减、二阶变化 |
| CrossSection（截面） | ~25 | 行业内/全市场截面排名 |
| Interaction（交互） | ~15 | 量价共振、趋势+量能交叉 |
| Market（市场） | ~15 | 北向资金、大盘、板块 |
| Sentiment（情绪） | ~10 | 龙虎榜、融资融券、涨跌停 |
| **总计** | **~245** | |

### 关键新增特征

#### 情绪特征（SentimentFeatures）

从 `sentiment_daily` 表接入：

| 特征 | 含义 | 周期 |
|------|------|------|
| sent_limit_up | 是否涨停 | 当日 |
| sent_limit_down | 是否跌停 | 当日 |
| sent_consecutive_limit | 连续涨停天数 | 滚动 |
| sent_vol_ratio | 量比（相对20日均量） | 20日 |
| sent_abnormal_ret | 异常收益率 | 当日 |
| sent_margin_chg | 融资余额变化 | 日频 |
| sent_short_balance | 融券余额 | 日频 |
| sent_lhb_ret_5d | 龙虎榜上榜后5日收益 | 5日 |

#### 扩充市场特征

| 特征 | 含义 | 来源 |
|------|------|------|
| mkt_north_sh_net | 沪股通净买入 | north_flow |
| mkt_north_sz_net | 深股通净买入 | north_flow |
| mkt_north_buy_ratio | 北向买卖比 | north_flow |
| mkt_hs300_volume_chg | 沪深300量变化 | hs300_daily |
| mkt_breadth | 涨跌家数比 | hs300_daily |
| mkt_hs300_volatility | 大盘波动率 | hs300_daily |

### 特征处理

```
原始特征 (245)
  → 去冗余: 移除相关系数 > 0.95 的冗余特征
  → 最终特征: ~200
```

## 五、回归目标定义

### 目标：连续收益率

```
目标 = (future_close - current_close) / current_close

日线: future_close = 5日后的收盘价
30m:  future_close = 3根K线后的收盘价

不做离散化，不做分位数切分。
保留收益率的连续值，让模型学习排序关系。
```

### vs 分类方案（废弃）

| 对比维度 | 分类（v1） | 回归（v2） |
|----------|-----------|-----------|
| 目标 | 3 离散标签（涨/震荡/跌） | 连续收益率 |
| 损失函数 | multi_logloss | l2 (MSE) |
| 评估指标 | F1, Accuracy | IC, Rank IC, 分组回测 |
| 信息利用 | 离散化丢失排序信息 | 保留完整排序信息 |
| 选股能力 | 3 档无法排序 | 连续分数天然排序 |
| 业界标准 | ❌ | ✅ (Qlib, 微软) |

### 为什么 3 分类失败

```
v1 训练结果（2026-06-14）：
  - 验证 F1: 0.19~0.23（仅略高于随机 0.33）
  - 树数: 24~43 棵（early stopping 极早触发）
  - 模型大小: 13MB（vs 预期 125-200MB）
  - 根本原因: 40%「震荡」类为纯噪声，模型学不到有效信号
```

## 六、内存与精度

### 精度选择

全部特征矩阵使用 **float32** 而非 float64：

```
理由:
  1. LGBM 是决策树模型，分裂操作不依赖高精度浮点
  2. LGBM 内部构造 Dataset 时自动将 float64 转为 float32
  3. float32 精度（7位有效数字）对金融特征完全足够
  4. 内存砍半，30m 模型 5 并行从 36GB 降到 18GB

效果影响: 预测结果差异 < 0.0001，可忽略
```

### 内存占用

| | float64 | float32 |
|------|---------|---------|
| 日线 5并行 | 9GB | **4.2GB** |
| 30m 5并行 | 36GB | **18GB** |
| M4 Pro 24GB | ⚠️ 30m 会爆 | ✅ 双模型均可 |

## 七、数据与训练

### 数据源

| 数据表 | 行数 | 时间范围 | 用途 |
|--------|------|---------|------|
| kline_daily | 97万 | 2014-2026 | 日线模型 |
| kline_30m | 410万 | 2020-2026 | 30m模型 |
| hs300_daily | 3856 | - | 大盘特征 |
| north_flow | 2649 | - | 北向资金 |
| sentiment_daily | 97万 | - | 情绪特征 |
| stock_sector | 5855 | - | 行业分类 |

### 时序切分

```
train (80%) → val (10%) → test (10%)
严格按时间顺序，无未来信息泄露
```

### 训练流程

```
# 1. 拉取数据
scp server:/path/to/stock_data.db ./data/

# 2. 快速验证 (2模型, 1000树, ~5分钟)
python strategy/train.py --model daily --quick
python strategy/train.py --model 30m --quick

# 3. 生产训练 (5模型, 20000树, 并行)
python strategy/train.py --model daily   # 预计30-60分钟
python strategy/train.py --model 30m    # 预计2-3小时

# 4. 模型产出
models/lgb_daily/model.pkl
models/lgb_30m/model.pkl

# 5. 上传部署
scp models/lgb_*/model.pkl server:/path/to/models/
```

## 八、模型产出

### 文件结构

```
models/
├── lgb_daily/
│   ├── model.pkl       # 5模型ensemble
│   └── meta.json       # 训练元信息
└── lgb_30m/
    ├── model.pkl       # 5模型ensemble
    └── meta.json       # 训练元信息
```

### model.pkl 结构

```python
{
    'models': [LGBMRegressor × 5],  # 5个独立回归模型
    'feature_names': [...],          # 特征名列表
    'keep_features': [...],          # 最终保留特征
    'n_models': 5,
    'horizon': 5,                    # 预测周期 (日线=5, 30m=3)
    'model_type': 'regression',
    'train_date': '2026-06-14',
    'train_samples': 44710,
    'seeds': [42, 123, 456, 789, 1024],
    'n_trees_per_model': [5000, 6200, 5800, 7100, 5400],
    'test_ic': 0.055,               # Rank IC (Spearman)
    'test_mse': 0.0012,             # 均方误差
    'params': {...},
}
```

### 预估规模

| | 单模型 | 5模型集成 | 双层总计 |
|---|---|---|---|
| 特征数 | ~200 | ×5 | ×2 |
| 树数 | 5000-12000 | ×5 | ×2 |
| 大小 | 25-40MB | 125-200MB | 250-400MB |

## 九、推理部署

### 在线推理

```
触发: 定时任务 (盘前9:25 / 盘中每30分钟)
  → 加载模型 (一次, 常驻内存)
  → 自选股池 (10-20只):
      读最新K线 → 计算245个特征 → 5模型推理 → 预测均值
  → 按预测收益率排序: 最高买入, 最低卖出
  → 推送飞书

耗时: <1秒 (仅推理自选股)
内存: ~150MB (常驻)
```

### 推理输出

```
预测值含义: 预期未来N期收益率 (连续值)

选股逻辑:
  pred > 0.02  → 强烈买入信号
  pred > 0.01  → 买入信号
  pred ≈ 0     → 持有/观望
  pred < -0.01 → 卖出信号
  pred < -0.02 → 强烈卖出信号

实际阈值根据历史回测校准
```

### 适用场景

- 盘前按预测收益率排序，推送 top N 选股建议
- 盘中30分钟频率择时信号
- 持仓股票预期收益率监控

## 十、文件清单

```
python/
├── strategy/
│   ├── features.py      # 特征工程: Price/Volume/Pattern/Momentum/
│   │                      CrossSection/Interaction/Market/Sentiment
│   └── train.py         # 训练入口: 数据加载/特征计算/并行训练/评估/保存
└── models/
    ├── lgb_daily/       # 日线模型输出目录
    └── lgb_30m/         # 30m模型输出目录
```