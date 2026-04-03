# 股票量化交易系统

## 项目介绍

这是一个基于机器学习的股票量化交易系统，支持 A 股（沪深 300 成分股）的智能选股和交易策略执行。系统使用预训练的梯度提升回归模型预测股票 5 日收益率，结合技术指标（RSI、MACD、移动平均线等）生成交易信号，实现自动化的投资组合管理。

## 功能特性

- **智能选股**：基于预训练模型的预测收益对股票进行排序和推荐
- **综合交易策略**：结合技术指标和机器学习预测生成交易信号
- **30 分钟级别监控**：支持 30 分钟级别的实时监控股票并发送交易信号
- **多因子策略**：融合 RSI、MACD、布林带、均线系统、KDJ 等多维度指标
- **指定股票池**：支持自定义股票池（爱尔眼科、汇川技术、保利发展、美团等）
- **多渠道通知**：支持钉钉、企业微信、飞书、邮件等多种通知方式
- **投资组合管理**：控制同时持仓不超过 5 只股票
- **价值曲线生成**：提供完整的投资组合价值变化曲线，体现买入和卖出时间
- **风险控制**：内置止损（3%）和止盈（5%）策略
- **数据接口**：使用 yfinance API 获取股票数据，免费无限制
- **RESTful API**：提供完整的 HTTP 接口，支持前端集成

## 环境要求

- Python 3.7+
- 依赖包：
  - Flask
  - Flask-CORS
  - pandas
  - numpy
  - scikit-learn
  - yfinance
  - schedule
  - requests

## 快速开始

### 1. 配置邮件通知和定时任务

运行自动配置脚本：

```bash
cd /Users/foleydang/github/stock-quant/stock-quant
./setup_monitor.sh
```

脚本将引导您完成：
- 邮件 SMTP 配置（QQ 邮箱、163 邮箱等）
- 定时任务频率设置
- 环境变量保存

### 2. 手动配置邮件

编辑 `~/.zshrc` 添加：

```bash
export SMTP_SERVER='smtp.qq.com'
export SMTP_PORT='465'
export SMTP_USERNAME='your_qq@qq.com'
export SMTP_PASSWORD='your_auth_code'  # 授权码，不是密码
export EMAIL_RECEIVERS='21725056@zju.edu.cn'
```

使配置生效：

```bash
source ~/.zshrc
```

### 3. 设置定时任务

```bash
crontab -e
```

添加：

```cron
# 股票监控 - 每 30 分钟执行（交易日 9:00-15:00）
*/30 9-15 * * 1-5 cd /Users/foleydang/github/stock-quant/stock-quant/python && /usr/bin/python3 strategy/email_monitor.py >> logs/email_monitor.log 2>&1
```

### 4. 测试

```bash
# 测试邮件发送
python3 strategy/email_notifier.py

# 手动运行监控
python3 strategy/email_monitor.py
```

### 1. 克隆项目

```bash
git clone <repository-url>
cd stock-quant
```

### 2. 安装依赖

```bash
pip install -r python/requirements.txt
pip install schedule requests  # 新增依赖
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

### 1. 启动 API 服务器

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

### 6. 30 分钟级别策略（新功能）

```bash
# 执行 30 分钟级别多因子策略
curl http://localhost:8000/intraday_strategy

# 获取单个股票信号
curl http://localhost:8000/intraday_signal/300015.SZ

# 获取信号历史
curl http://localhost:8000/intraday_history?limit=50
```

### 7. 启动定时监控

```bash
# 每 30 分钟自动执行
cd python
python strategy/monitor_scheduler.py
```

## API 接口说明

### 原有接口

#### 1. `/portfolio_strategy` (GET)

**功能**：执行综合选股策略，控制持仓不超过 5 只股票

**返回数据**：
- `portfolio_history`：投资组合历史记录
- `trades`：交易记录
- `value_curve`：价值曲线
- `buy_points`：买入点
- `sell_points`：卖出点
- `final_portfolio`：最终投资组合状态
- `selected_stocks`：选股结果

#### 2. `/select` (GET)

**功能**：基于预训练模型的智能选股

**返回数据**：
- `selected_stocks`：按预测收益排序的股票列表
- `model_type`：模型类型
- `update_time`：更新时间

#### 3. `/predict_strategy/<symbol>` (GET)

**功能**：针对单个股票执行基于预测的交易策略

**参数**：
- `symbol`：股票代码，如 `600519.SH`

**返回数据**：
- `strategyResults`：策略执行结果
- `trades`：交易记录
- `predictions`：预测数据
- `finalPortfolio`：最终投资组合

#### 4. `/stock/<symbol>` (GET)

**功能**：获取股票历史数据

**参数**：
- `symbol`：股票代码

**返回数据**：
- `data`：股票历史数据
- `status`：请求状态

#### 5. `/stocks` (GET)

**功能**：获取支持的股票列表

**返回数据**：
- `stocks`：股票列表，包含代码和名称

### 新增接口（30 分钟级别策略）

#### 6. `/intraday_strategy` (GET)

**功能**：执行 30 分钟级别多因子交易策略

**参数**：
- `watchlist`（可选）：自定义股票池，逗号分隔

**返回数据**：
- `signals`：所有股票的交易信号
- `summary`：信号汇总（买入/卖出/持有数量）
- `latest_signals`：最新信号详情

**示例**：
```bash
curl http://localhost:8000/intraday_strategy
curl "http://localhost:8000/intraday_strategy?watchlist=300015.SZ,300124.SZ"
```

#### 7. `/intraday_signal/<symbol>` (GET)

**功能**：获取单个股票的 30 分钟级别交易信号

**参数**：
- `symbol`：股票代码

**返回数据**：
- `signal`：交易信号详情（包含 RSI、MACD、KDJ 等指标）

#### 8. `/intraday_history` (GET)

**功能**：获取历史交易信号

**参数**：
- `limit`（可选）：返回记录数量，默认 100

**返回数据**：
- `history`：信号历史列表
- `count`：记录数量

## 项目结构

```
stock-quant/
├── api/                    # API 服务器
│   └── server.py           # Flask 应用（已增强）
├── python/
│   ├── strategy/           # 30 分钟级别策略（新增）
│   │   ├── intraday_strategy.py   # 策略核心
│   │   ├── monitor_scheduler.py   # 定时调度器
│   │   ├── notifier.py            # 通知推送模块
│   │   └── config.ini             # 配置文件
│   ├── data/             # 股票数据
│   ├── models/           # 模型文件
│   │   ├── pretrained/   # 预训练模型
│   │   ├── model_runner.py
│   │   └── stock_selector.py
│   ├── scripts/          # 脚本文件
│   │   ├── download_data.py
│   │   └── pretrain_models.py
│   └── requirements.txt
├── logs/                 # 日志目录（自动创建）
├── frontend/             # 前端代码
├── README.md
└── STRATEGY_USAGE.md     # 策略使用说明
```

## 数据说明

- **数据来源**：yfinance API（免费、无限制）
- **数据频率**：日线数据（30 分钟级别策略需要实时数据源）
- **数据范围**：近两年的历史数据
- **技术指标**：
  - RSI (14 日)
  - MACD (12, 26, 9)
  - 移动平均线（MA5, MA10, MA20, MA60）
  - 布林带 (20, 2)
  - KDJ (9, 3, 3)
  - ATR (14)

## 模型说明

- **模型类型**：梯度提升回归器 (Gradient Boosting Regressor)
- **预测目标**：股票 5 日收益率
- **特征维度**：技术指标和价格模式
- **训练方法**：滚动窗口训练，定期更新

## 30 分钟级别策略说明

### 支持股票

默认股票池：
- 爱尔眼科 (300015.SZ)
- 汇川技术 (300124.SZ)
- 保利发展 (600048.SH)
- 美团 -W (3690.HK)

### 信号评分系统

| 评分 | 信号 | 含义 |
|------|------|------|
| ≥4   | 强烈买入 | 多个指标强烈看好 |
| ≥2   | 买入 | 多数指标看好 |
| -1~1 | 持有 | 指标中性 |
| ≤-2  | 卖出 | 多数指标看空 |
| ≤-4  | 强烈卖出 | 多个指标强烈看空 |

### 通知配置

通过环境变量配置通知渠道：

```bash
# 钉钉
export DINGTALK_WEBHOOK_URL="https://oapi.dingtalk.com/robot/send?access_token=xxx"
export DINGTALK_SECRET="xxx"

# 企业微信
export WECHAT_WEBHOOK_URL="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx"

# 飞书
export FEISHU_WEBHOOK_URL="https://open.feishu.cn/open-apis/bot/v2/hook/xxx"
export FEISHU_SECRET="xxx"

# 邮件
export EMAIL_SMTP_SERVER="smtp.example.com"
export EMAIL_SMTP_PORT="465"
export EMAIL_USERNAME="your_email@example.com"
export EMAIL_PASSWORD="your_password"
export EMAIL_RECEIVERS="receiver@example.com"
```

## 风险提示

1. **投资风险**：量化策略不保证盈利，投资有风险
2. **模型风险**：机器学习模型预测存在误差
3. **数据风险**：历史数据不代表未来表现
4. **市场风险**：系统性风险可能导致策略失效
5. **数据延迟**：免费数据源可能存在延迟，实盘建议使用专业数据服务

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
  ]
}
```

### 30 分钟级别策略示例

```json
{
  "status": "success",
  "timestamp": "2026-03-23T10:30:00",
  "signals": [
    {
      "symbol": "300015.SZ",
      "stock_name": "爱尔眼科",
      "price": 28.56,
      "signal": "买入",
      "score": 3,
      "reasons": ["RSI 超卖 (28.5)", "MACD 金叉", "短期均线向上"],
      "indicators": {
        "rsi": 28.5,
        "macd": 0.0012,
        "kdj_k": 35.2,
        "kdj_d": 28.4
      }
    }
  ],
  "summary": {
    "total": 4,
    "buy": 1,
    "sell": 0,
    "hold": 3
  }
}
```

## 前端集成

系统提供完整的 RESTful API，可与前端框架（如 React、Vue 等）集成。前端项目位于 `frontend/` 目录，使用 Vite 构建。

## 扩展建议

1. **模型优化**：尝试不同的机器学习算法和特征组合
2. **策略优化**：调整技术指标参数和交易阈值
3. **风险管理**：增加更多风险控制措施
4. **回测系统**：完善回测框架，支持多周期回测
5. **实时数据**：集成 Level2 或实时行情数据源
6. **自动化交易**：对接券商 API 实现自动下单

## 联系我们

如有问题或建议，请联系项目维护者。

---

**免责声明**：本系统仅用于学习和研究目的，不构成任何投资建议。投资有风险，入市需谨慎。
