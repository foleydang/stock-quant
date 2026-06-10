# 策略模块

## 1. LGBM增强策略 (train_lgb_enhanced.py)

### 特征工程
- 收益率特征：1/5/10/20期
- 均线特征：MA5/10/20/60/120
- 技术指标：RSI/MACD/KDJ/布林带
- 时间特征：小时/分钟/星期

### 模型训练
```python
from strategy.train_lgb_enhanced import train_model
model = train_model(symbol='300124.SZ', days=365)
```

### 预测
```python
up_prob = model.predict_proba(features)[:, 1]
```

## 2. 日内策略 (intraday_strategy.py)

多因子评分系统：
- 评分 >= 4: 强烈买入
- 评分 >= 2: 买入
- 评分 -1~1: 持有
- 评分 <= -2: 卖出
- 评分 <= -4: 强烈卖出
