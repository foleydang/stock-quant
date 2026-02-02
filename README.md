# 股票量化交易系统

## 项目介绍

这是一个基于机器学习的股票量化交易系统，支持A股（沪深300成分股）的智能选股和交易策略执行。系统使用预训练的梯度提升回归模型预测股票5日收益率，结合技术指标（RSI、MACD、移动平均线等）生成交易信号，实现自动化的投资组合管理。

## 功能特性

- **智能选股**：基于预训练模型的预测收益对股票进行排序和推荐
- **综合交易策略**：结合技术指标和机器学习预测生成交易信号
- **投资组合管理**：控制同时持仓不超过5只股票
- **价值曲线生成**：提供完整的投资组合价值变化曲线，体现买入和卖出时间
- **风险控制**：内置止损（3%）和止盈（5%）策略
- **数据接口**：使用BaoStock API获取A股数据，免费无限制
- **RESTful API**：提供完整的HTTP接口，支持前端集成

## 环境要求

- Python 3.7+
- 依赖包：
  - Flask
  - Flask-CORS
  - pandas
  - numpy
  - scikit-learn
  - baostock

## 安装步骤

### 1. 克隆项目

```bash
git clone <repository-url>
cd stock-quant-new
```

### 2. 安装依赖

```bash
pip install -r python/requirements.txt
```

### 3. 下载股票数据

系统首次运行时会自动下载数据，也可以手动执行：

```bash
python python/scripts/download_data.py
```

### 4. 预训练模型

系统已包含预训练模型，位于 `python/models/pretrained/` 目录。如需重新训练：

```bash
python python/scripts/pretrain_models.py
```

## 使用方法

### 1. 启动API服务器

```bash
python api/server.py
```

服务器将运行在 `http://localhost:8000`

### 2. 执行综合选股策略

```bash
curl http://localhost:8000/portfolio_strategy
```

返回结果包含：
- 投资组合历史价值
- 完整的交易记录
- 价值曲线数据
- 买入和卖出点
- 最终收益和收益率

### 3. 查看选股结果

```bash
curl http://localhost:8000/select
```

返回按预测收益排序的股票列表，包含推荐等级。

### 4. 单个股票策略

```bash
curl http://localhost:8000/predict_strategy/600519.SH
```

返回指定股票的预测策略结果。

### 5. 获取股票数据

```bash
curl http://localhost:8000/stock/600519.SH
```

返回股票的历史数据。

## API接口说明

### 1. `/portfolio_strategy` (GET)

**功能**：执行综合选股策略，控制持仓不超过5只股票

**返回数据**：
- `portfolio_history`：投资组合历史记录
- `trades`：交易记录
- `value_curve`：价值曲线
- `buy_points`：买入点
- `sell_points`：卖出点
- `final_portfolio`：最终投资组合状态
- `selected_stocks`：选股结果

### 2. `/select` (GET)

**功能**：基于预训练模型的智能选股

**返回数据**：
- `selected_stocks`：按预测收益排序的股票列表
- `model_type`：模型类型
- `update_time`：更新时间

### 3. `/predict_strategy/<symbol>` (GET)

**功能**：针对单个股票执行基于预测的交易策略

**参数**：
- `symbol`：股票代码，如 `600519.SH`

**返回数据**：
- `strategyResults`：策略执行结果
- `trades`：交易记录
- `predictions`：预测数据
- `finalPortfolio`：最终投资组合

### 4. `/stock/<symbol>` (GET)

**功能**：获取股票历史数据

**参数**：
- `symbol`：股票代码

**返回数据**：
- `data`：股票历史数据
- `status`：请求状态

### 5. `/stocks` (GET)

**功能**：获取支持的股票列表

**返回数据**：
- `stocks`：股票列表，包含代码和名称

## 项目结构

```
stock-quant-new/
├── api/                # API服务器
│   └── server.py       # Flask应用
├── python/
│   ├── data/           # 股票数据
│   ├── models/         # 模型文件
│   │   ├── pretrained/ # 预训练模型
│   │   ├── model_runner.py
│   │   └── stock_selector.py
│   ├── scripts/        # 脚本文件
│   │   ├── download_data.py     # 数据下载
│   │   └── pretrain_models.py   # 模型训练
│   └── requirements.txt
├── frontend/           # 前端代码
└── README.md
```

## 数据说明

- **数据来源**：BaoStock API（免费、无限制）
- **数据频率**：日线数据
- **数据范围**：近两年的历史数据
- **技术指标**：
  - RSI (14日)
  - MACD (12, 26, 9)
  - 移动平均线（MA5, MA10, MA20）
  - 波动率
  - 成交量指标

## 模型说明

- **模型类型**：梯度提升回归器 (Gradient Boosting Regressor)
- **预测目标**：股票5日收益率
- **特征维度**：技术指标和价格模式
- **训练方法**：滚动窗口训练，定期更新

## 风险提示

1. **投资风险**：量化策略不保证盈利，投资有风险
2. **模型风险**：机器学习模型预测存在误差
3. **数据风险**：历史数据不代表未来表现
4. **市场风险**：系统性风险可能导致策略失效

## 示例结果

### 综合策略示例

```json
{
  "final_portfolio": {
    "cash": 26551.67,
    "total_value": 121018.67,
    "profit": 21018.67,
    "profit_rate": 21.02,
    "holding_count": 5
  },
  "trades": [
    {
      "date": "2026-01-29",
      "symbol": "601318.SH",
      "type": "BUY",
      "price": 68.0,
      "shares": 100,
      "amount": 6800.0
    }
    // 更多交易记录...
  ]
}
```

### 选股结果示例

```json
{
  "selected_stocks": [
    {
      "symbol": "600887.SH",
      "name": "伊利股份",
      "predicted_return": 6.13,
      "recommendation": "强烈买入",
      "rank": 1
    },
    // 更多股票...
  ]
}
```

## 前端集成

系统提供完整的RESTful API，可与前端框架（如React、Vue等）集成。前端项目位于 `frontend/` 目录，使用Vite构建。

## 扩展建议

1. **模型优化**：尝试不同的机器学习算法和特征组合
2. **策略优化**：调整技术指标参数和交易阈值
3. **风险管理**：增加更多风险控制措施
4. **回测系统**：完善回测框架，支持多周期回测
5. **实时数据**：集成实时行情数据

## 联系方式

如有问题或建议，请联系项目维护者。

---

**免责声明**：本系统仅用于学习和研究目的，不构成任何投资建议。投资有风险，入市需谨慎。