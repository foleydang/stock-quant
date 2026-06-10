# 香港网关 + 国内服务器 部署指南

## 架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 东方财富API │────▶│ 香港服务器  │────▶│ 国内服务器  │
│   (被封锁)  │     │  (可访问)   │     │  (运行监控) │
└─────────────┘     └─────────────┘     └─────────────┘
```

## 第一步：香港服务器部署

### 1. 上传网关代码

将 `/tmp/hk_gateway_server.py` 上传到香港服务器：

```bash
# 在国内服务器执行
scp /tmp/hk_gateway_server.py root@香港服务器IP:/home/scripts/
```

### 2. 安装依赖

香港服务器：

```bash
pip3 install flask requests
```

### 3. 启动网关

```bash
# 测试运行
python3 /home/scripts/hk_gateway_server.py

# 后台运行
nohup python3 /home/scripts/hk_gateway_server.py > /home/logs/gateway.log 2>&1 &
```

### 4. 测试验证

```bash
# 香港服务器本地测试
curl http://localhost:5000/api/kline?symbol=000001&klt=30

# 国内服务器测试（确保防火墙开放 5000 端口）
curl http://香港服务器IP:5000/api/kline?symbol=000001&klt=30
```

### 5. 设置开机启动

```bash
# 创建 systemd 服务
cat > /etc/systemd/system/hk-gateway.service << 'SERVICE'
[Unit]
Description=HK Gateway for EastMoney API
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/scripts/hk_gateway_server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl enable hk-gateway
systemctl start hk-gateway
```

## 第二步：国内服务器配置

### 1. 修改 data_handler.py

编辑 `/root/github/stock-quant/python/data/data_handler.py`：

```python
# 找到这一行，修改为香港服务器实际 IP
HK_GATEWAY_URL = 'http://你的香港服务器IP:5000'
```

### 2. 测试数据获取

```bash
cd /root/github/stock-quant/python
python3.8 -c "
from data.data_handler import DataHandler
dh = DataHandler()
df = dh.fetch_stock_data('000001.SZ', force_refresh=True)
print(f'获取到 {len(df)} 条数据')
print(df.tail())
"
```

### 3. 运行监控

```bash
./scripts/start.sh monitor
```

## 第三步：定时任务调整

国内服务器 crontab 保持不变：

```bash
*/30 9-15 * * 1-5 /root/github/stock-quant/scripts/start.sh monitor
0 20 * * 1-5 /root/github/stock-quant/scripts/start.sh update
```

## 费用

| 项目 | 费用 |
|------|------|
| 香港服务器网关 | 免费（你已有服务器） |
| 国内→香港网络 | 阿里云内网免费（如果是同一账号） |
| **总计** | **¥0** |

## 故障排查

### 1. 香港网关无法访问

```bash
# 检查服务状态
systemctl status hk-gateway

# 检查端口
netstat -tlnp | grep 5000

# 查看日志
tail -f /home/logs/gateway.log
```

### 2. 国内服务器连接失败

```bash
# 检查网络连通性
curl -v http://香港服务器IP:5000/health

# 检查防火墙（香港服务器）
firewall-cmd --add-port=5000/tcp --permanent
firewall-cmd --reload
```

### 3. 数据获取失败

检查 data_handler.py 中的 HK_GATEWAY_URL 是否正确

