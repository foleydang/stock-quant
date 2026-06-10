# 量化模型优化计划 - Phase 1：数据增强

## 目标

提升预测准确率，核心思路：**先补数据，再调模型**

## 当前问题

- 118个特征全是量价指标，无消息面/资金面数据
- 预测窗口太短(90分钟)、阈值太低(1%)
- 训练数据只有15个月30分钟线
- 模型本质上在做波动率模式识别，不是趋势预测

## Phase 1：北向资金 + 板块数据接入（本周完成）

### Step 1：拉取北向资金历史数据

- **数据源**: Tushare `moneyflow_hsgt` (沪股通+深股通+港股通)
- **字段**: north_money(北向净流入), north_buy(买入), north_sell(卖出)
- **频率**: 每日一条
- **历史范围**: 2023-01-01 ~ 今天（约3年）
- **存储**: 新表 `north_flow` in stock_data.db
- **脚本**: `python/scripts/fetch_north_flow.py`

### Step 2：拉取板块/行业数据

- **数据源**: Tushare `index_daily` + 东方财富行业分类
- **字段**: 所属板块当日涨跌幅、板块排名、行业轮动信号
- **方式**: 
  1. Tushare获取沪深300各行业指数日线（银行/医药/科技等）
  2. 建立 `stock_sector` 映射表（股票→行业板块）
  3. 计算个股相对于板块的超额收益
- **存储**: 新表 `sector_daily` + `stock_sector` in stock_data.db
- **脚本**: `python/scripts/fetch_sector_data.py`

### Step 3：在特征工程中融合新数据

- **新增特征**（约6-8个）:
  | 特征名 | 说明 |
  |--------|------|
  | `north_flow_ratio` | 当日北向净流入/总成交额 |
  | `north_flow_change` | 北向资金3日变化率 |
  | `north_flow_cum_5` | 5日累计北向净流入 |
  | `sector_change` | 所属板块当日涨跌幅 |
  | `sector_rank` | 个股在板块内的涨幅排名 |
  | `stock_vs_sector` | 个股超额收益(vs板块) |
  | `sector_momentum` | 板块3日动量 |
  | `sector_rotation` | 板块轮动信号(资金流入新板块) |

- **修改文件**: 
  - `strategy/train_lgb_enhanced.py` 的 `EnhancedFeatureEngineer.calculate_features()`
  - `lgbm_backtest.py` 的特征计算部分
- **关键**: 特征计算需要从DB查 north_flow 和 sector_daily 表，合并到30分钟K线DataFrame

### Step 4：每日自动更新

- **修改**: `scripts/update_daily_data.py` 
- **新增**: 北向资金 + 板块数据的每日增量更新
- **crontab**: 已有每晚20:00执行

### Step 5：重新训练模型

- 用新特征集重新训练 LGBM ensemble
- 调整参数:
  - `horizon`: 3 → 10（预测10根K线=1交易日）
  - `threshold`: 0.01 → 0.03（涨3%才算"大涨"）
  - 目标: binary → 暂保持binary，Phase 2再改三分类
- 对比旧模型 vs 新模型的:
  - 准确率变化
  - 特征重要性变化（新特征排位）
  - 回测收益率变化
- **脚本**: `python/strategy/train_lgb_enhanced.py` (已有，修改参数即可)

## 预期效果

| 指标 | 当前 | Phase 1目标 |
|------|------|------------|
| 特征数 | 118 | ~126 |
| 准确率 | 72.67% | >75% |
| 回测收益率 | 16.46% | 待验证 |
| 信号质量 | 噪音多 | 减少无效信号 |

## 后续Phase（不在本次执行）

- Phase 2: 三分类模型（大涨/震荡/大跌）+ 阈值3%
- Phase 3: 新闻情绪特征接入
- Phase 4: 3年日线数据扩展训练
- Phase 5: LGBM + GRU 模型融合

## 风险点

- Tushare北向资金API可能有频次限制（200次/分钟够用）
- 板块映射需要手动维护或从东方财富获取
- 30分钟线与日线数据合并需要按日期对齐（不能直接merge）
- 新特征可能初期重要性低，需要观察迭代