# 飞书 Bot 金融分析 Agent - 部署指南

## 🥔 系统架构

```
飞书服务器 (HTTPS POST)
    │
    ▼
Nginx (443)
    ├─ /feishu/event → Bot Server (FastAPI, :8001)
    ├─ /bot/health   → Bot 健康检查
    ├─ /api/*        → API Server (Flask/Gunicorn, :8000) ← 保留
    └─ /*            → 前端静态文件 ← 保留
```

## 现有服务（全部保留）

| 服务 | 端口 | 说明 | 状态 |
|------|------|------|------|
| Flask API | 8000 | 量化监控 API | ✅ 不变 |
| Nginx | 80/443 | 反向代理 + 静态文件 | ✅ 已更新路由 |
| Crontab 邮件监控 | - | 定时邮件推送 | ✅ 不变 |
| OpenClaw Gateway | 18789 | AI 助手 | ✅ 不变 |
| **Feishu Bot** | **8001** | **新增：飞书机器人** | ✅ 已部署 |

## 新增文件清单

```
agent/
├── config.py           # 配置（从 config.yaml + 环境变量读取）
├── bot_server.py       # FastAPI 主服务（事件回调 + 消息处理）
├── start_bot.py        # 启动脚本
├── intent_router.py    # 意图路由（关键词 → 动作）
├── action_executor.py  # 动作执行器（调用现有量化模块）
├── card_templates.py   # 飞书消息卡片模板
├── feishu_client.py    # 飞书消息发送 SDK
├── llm_client.py       # 百炼 DashScope LLM 客户端（Phase 2）
├── scheduler.py        # APScheduler 定时推送
├── .env                # 环境变量（需要配置飞书凭证）
└── .env.example        # 环境变量示例
```

## 飞书 Bot 配置步骤（你必须做的）

### 第 1 步：创建飞书自建应用

1. 打开 https://open.feishu.cn/app
2. 点击「创建企业自建应用」
3. 填写应用名称（如"小土豆量化助手"）和描述
4. 记录 **App ID** 和 **App Secret**

### 第 2 步：开启机器人能力

1. 在应用管理页 → 「添加应用能力」 → 选择「机器人」
2. 在「权限管理」中申请以下权限：
   - `im:message` - 获取与发送单聊、群组消息
   - `im:message:send_as_bot` - 以应用身份发消息
   - `im:chat` - 获取群组信息

### 第 3 步：配置事件订阅

1. 在「事件订阅」页面：
   - **请求地址**: `https://stock.yanten.top/feishu/event`
   - 点击验证，系统会发送 challenge 请求（已验证通过 ✅）
2. 添加事件：
   - `im.message.receive_v1` - 接收消息
3. 记录 **Verification Token** 和 **Encrypt Key**

### 第 4 步：配置环境变量

编辑 `/root/github/stock-quant/stock-quant/agent/.env`：

```bash
FEISHU_APP_ID=cli_xxxxxxxx          # 第1步获取
FEISHU_APP_SECRET=xxxxxxxxxxxxxxxx   # 第1步获取
FEISHU_VERIFICATION_TOKEN=xxxxxxxx    # 第3步获取
FEISHU_ENCRYPT_KEY=                   # 可选
FEISHU_TARGET_CHAT_ID=oc_xxxxxxxx    # 群聊ID（见下）
DASHSCOPE_API_KEY=                    # 百炼API（Phase 2）
BOT_PORT=8001
```

### 第 5 步：获取群聊 ID

1. 在飞书创建一个群聊（或使用现有群）
2. 把机器人添加到群聊
3. 获取 chat_id（方法：在群设置 → 群信息中查看，或通过 API 查询）

### 第 6 步：重启服务

```bash
systemctl restart stock-bot
systemctl status stock-bot
```

## 使用方式

在飞书群中 @机器人 或私聊机器人，发送以下指令：

| 指令 | 功能 |
|------|------|
| `持仓` / `仓位` | 查看持仓概览 + 做T建议 |
| `行情 茅台` / `价格 600036` | 查看个股行情 |
| `做T` | 查看做T操作建议 |
| `信号` | 查看最新交易信号 |
| `回测 茅台` | 运行LGBM回测 |
| `总结` / `日报` | 盘后总结 |
| `分析 茅台` | 综合分析（行情+指标+做T） |
| `自选 阿里巴巴 9988.HK` | 添加自选股 |
| `帮助` | 显示帮助信息 |

## 定时推送

配置 `FEISHU_TARGET_CHAT_ID` 后，以下推送会自动启用：

| 时间 | 内容 |
|------|------|
| 9:25（盘前） | 自选股行情 + 开盘提醒 |
| 9:30-14:30（盘中，每30分钟） | 重要信号 + 异动告警 |
| 15:05（盘后） | 持仓总结 + 操作建议 |

## 和现有系统的关系

- **邮件监控**：crontab 继续每30分钟发邮件，飞书推送是额外通知渠道
- **网站**：stock.yanten.top 前端 + API 完全不受影响
- **API**：Flask API 在 8000 端口继续服务，Bot 在 8001 端口独立运行
- **两者共享数据库和模型**：Bot 通过 import 调用现有 Python 模块

## Phase 2 升级计划

1. 配置 `DASHSCOPE_API_KEY` → 启用百炼 LLM
2. 自然语言理解替代关键词路由
3. 新闻情绪分析
4. 智能问答（持仓诊断、策略解释）
5. 飞书卡片按钮交互