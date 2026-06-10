# Stock Quant 交易监控系统

基于多因子技术指标和机器学习的A股/港股实时交易监控系统。

## 目录

- [快速开始](#快速开始)
- [策略系统](#策略系统)
- [实盘监控](#实盘监控)
- [邮件配置](#邮件配置)
- [定时任务](#定时任务)
- [机器学习模型](#机器学习模型)
- [日志管理](#日志管理)
- [常见问题](#常见问题)

---

## 快速开始

### 环境要求

- Python 3.14+ (推荐使用 Homebrew 安装)
- 依赖包: `pandas`, `numpy`, `lightgbm`, `scikit-learn`, `akshare`

```bash
# 安装依赖
pip install pandas numpy lightgbm scikit-learn akshare requests --break-system-packages
```

### 运行监控

```bash
cd /Users/foleydang/github/stock-quant/stock-quant/python
python trading_monitor.py
```

---

## 策略系统

### 监控股票池

| 代码 | 名称 | 市场 |
|------|------|------|
| 300015.SZ | 爱尔眼科 | A股 |
| 300124.SZ | 汇川技术 | A股 |
| 600048.SH | 保利发展 | A股 |
| 600519.SH | 贵州茅台 | A股 |
| 3690.HK | 美团-W | 港股 |
| 0700.HK | 腾讯控股 | 港股 |
| 9988.HK | 阿里巴巴-W | 港股 |

修改股票池请编辑 `strategy/intraday_strategy.py` 中的 `WATCHLIST_STOCKS`。

### 技术指标体系

#### 核心指标 (±2分权重)
| 指标 | 参数 | 作用 |
|------|------|------|
| RSI | 14周期 | 超买超卖判断 |
| MACD | 12,26,9 | 趋势动能 |
| ADX | 14 | 趋势强度过滤 |
| MFI | 14 | 资金流量超买超卖 |

#### 辅助指标 (±1分权重)
| 指标 | 参数 | 作用 |
|------|------|------|
| KDJ | 9,3,3 | 短期买卖点 |
| OBV | - | 成交量能量潮 |
| VWAP | 20 | 成交量加权均价 |
| 布林带 | 20,2 | 波动区间 |

### 信号评分系统

| 评分 | 信号 | 操作建议 |
|------|------|----------|
| ≥ 4 | 强烈买入 | 积极建仓 |
| ≥ 2 | 买入 | 逢低介入 |
| -1 ~ 1 | 持有 | 持仓观望 |
| ≤ -2 | 卖出 | 逢高减持 |
| ≤ -4 | 强烈卖出 | 果断离场 |

### 买入门槛

- 正常市场: 评分 ≥ 4分
- 下跌趋势: 评分 ≥ 5分 (需更多反转确认)
- 趋势过滤: 20周期下跌 > 10% 禁止买入

### 止损止盈

- 止损: 5% 或 ATR × 3.0
- 止盈: 8% 或 ATR × 4.0

---

## 实盘监控

### trading_monitor.py

实盘交易监控系统，功能:
1. 跟踪持仓和资金 (保存到 `portfolio.json`)
2. 生成买入/卖出信号
3. 发送邮件通知 (包含具体操作建议)

### 运行方式

```bash
# 手动运行
python trading_monitor.py

# 测试模式 (不发送邮件)
python trading_monitor.py --no-email
```

### 持仓管理

持仓信息保存在 `portfolio.json`:

```json
{
  "cash": 100000,
  "positions": {
    "300015.SZ": {
      "symbol": "300015.SZ",
      "stock_name": "爱尔眼科",
      "shares": 1000,
      "cost_price": 9.68,
      "current_price": 9.70,
      "entry_date": "2026-03-30",
      "available": true
    }
  },
  "last_update": "2026-03-30T15:30:00"
}
```

### T+1规则

- 买入当天 `available=false`，不可卖出
- 次日自动变为 `available=true`

---

## 邮件配置

### SMTP 服务器配置

| 邮箱服务 | SMTP服务器 | SSL端口 |
|---------|-----------|--------|
| QQ邮箱 | smtp.qq.com | 465 |
| 163邮箱 | smtp.163.com | 465 |
| Gmail | smtp.gmail.com | 465 |

### 配置步骤

#### QQ邮箱 (推荐)

1. 登录 mail.qq.com → 设置 → 账户
2. 开启「IMAP/SMTP服务」
3. 生成授权码 (不是QQ密码)

#### 设置环境变量

```bash
# 方式1: ~/.zshrc
export SMTP_SERVER='smtp.qq.com'
export SMTP_PORT='465'
export SMTP_USERNAME='your_qq@qq.com'
export SMTP_PASSWORD='your_auth_code'
export EMAIL_RECEIVERS='recipient@email.com'

# 使配置生效
source ~/.zshrc
```

#### 方式2: 创建 .env 文件

复制 `.env.example` 到 `.env` 并填写配置。

### 测试邮件发送

```bash
python strategy/email_notifier.py
```

---

## 定时任务

### macOS cron

```bash
crontab -e
```

添加以下行 (每30分钟，交易时段9:00-15:00):

```cron
*/30 9-15 * * 1-5 cd /path/to/python && /opt/homebrew/bin/python3 trading_monitor.py >> logs/trading_monitor.log 2>&1
```

### macOS LaunchAgent

创建 `~/Library/LaunchAgents/com.stockquant.monitor.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.stockquant.monitor</string>
    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/bin/python3</string>
        <string>/path/to/python/trading_monitor.py</string>
    </array>
    <key>StartInterval</key>
    <integer>1800</integer>
    <key>StandardOutPath</key>
    <string>/path/to/logs/monitor.log</string>
    <key>StandardErrorPath</key>
    <string>/path/to/logs/monitor_error.log</string>
</dict>
</plist>
```

加载任务:

```bash
launchctl load ~/Library/LaunchAgents/com.stockquant.monitor.plist
```

---

## 机器学习模型

### 模型版本

| 模型 | 准确率 | 特征数 | 路径 |
|------|--------|--------|------|
| zz500_full_optimized | 63.38% | 61 | models/lgb_enhanced/ |

### 使用模型预测

```python
from strategy.lgb_predictor import LGBPredictor

predictor = LGBPredictor()
prediction = predictor.predict(symbol, df)
# 返回: {'direction': 'up/down/flat', 'confidence': 0.65}
```

### 训练新模型

```bash
# 收集数据
python strategy/collect_zz500_data.py

# 训练模型
python strategy/train_full_optimized.py
```

---

## 数据库

交易信号使用SQLite存储，便于前端可视化展示。

### 数据库位置

```
data/trading.db
```

### 数据表结构

| 表名 | 说明 |
|------|------|
| signals | 交易信号历史 |
| portfolio_snapshots | 持仓快照 |
| trades | 交易记录 |

### 使用方式

```python
from database import get_db

db = get_db()

# 查询最近30天的信号
signals = db.get_signals(days=30)

# 获取每只股票的最新信号
latest = db.get_latest_signals()

# 获取持仓历史
history = db.get_portfolio_history(days=30)

# 获取交易记录
trades = db.get_trades(days=30)
```

---

## API接口

启动API服务器:

```bash
cd ../api
python server.py
```

服务运行在 `http://localhost:8000`

### 信号查询接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/db/signals` | GET | 获取交易信号列表 |
| `/db/signals/latest` | GET | 获取每只股票最新信号 |
| `/db/signals/stats` | GET | 获取信号统计 |
| `/db/portfolio` | GET | 获取持仓历史 |
| `/db/trades` | GET | 获取交易记录 |
| `/db/export/<table>` | GET | 导出数据到CSV |

### 示例请求

```bash
# 获取最近7天的信号
curl "http://localhost:8000/db/signals?days=7"

# 获取指定股票的信号
curl "http://localhost:8000/db/signals?symbol=300015.SZ"

# 获取最新信号
curl "http://localhost:8000/db/signals/latest"

# 获取信号统计
curl "http://localhost:8000/db/signals/stats?days=30"
```

### 返回示例

```json
{
  "status": "success",
  "signals": [
    {
      "id": 1,
      "symbol": "300015.SZ",
      "stock_name": "爱尔眼科",
      "price": 9.68,
      "signal": "买入",
      "score": 4.5,
      "reasons": ["RSI超卖", "MACD金叉"],
      "timestamp": "2026-03-30T15:30:00"
    }
  ],
  "count": 1
}
```

---

## 日志管理

### 日志目录

所有日志保存在 `logs/` 目录:

```
logs/
├── monitor_20260330.log    # 当日监控日志
├── monitor_20260329.log    # 昨日日志
├── ...
```

### 日志格式

- 文件名: `monitor_YYYYMMDD.log`
- 每天一个文件
- 自动清理超过15天的日志

### 查看日志

```bash
# 实时查看
tail -f logs/monitor_$(date +%Y%m%d).log

# 查看历史
cat logs/monitor_20260330.log
```

---

## 常见问题

### Q: 邮件发送失败?

检查:
1. SMTP服务器和端口正确
2. 使用授权码而非登录密码
3. 网络允许SMTP连接

### Q: 无交易信号?

正常情况:
- 市场下跌趋势时买入门槛提高
- 当前股票池不满足技术指标条件
- 系统只在发现机会时发送通知

### Q: 数据获取失败?

- AKShare可能有API限制
- 等待几分钟后重试
- 检查网络连接

### Q: cron任务不执行?

检查:
1. 使用绝对路径
2. python路径: `which python3`
3. 查看系统日志: `grep cron /var/log/system.log`

---

## 文件结构

```
python/
├── trading_monitor.py      # 实盘监控主程序
├── database.py             # 数据库模块
├── logger.py               # 日志模块
├── portfolio.json          # 持仓数据
├── .env.example            # 邮件配置模板
├── README.md               # 本文档
├── data/
│   ├── data_handler.py     # 数据获取
│   ├── trading.db          # SQLite数据库
│   └── *.csv               # 股票数据缓存
├── strategy/
│   ├── intraday_strategy.py # 策略核心
│   ├── email_notifier.py   # 邮件通知
│   └── lgb_predictor.py    # ML预测器
├── models/
│   └── lgb_enhanced/       # 模型文件
└── logs/
    └── monitor_*.log       # 日志文件
```

---

更新时间: 2026-03-30