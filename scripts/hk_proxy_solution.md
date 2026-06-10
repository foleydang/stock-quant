# 香港服务器数据中转方案

## 架构

```
东方财富 API → 香港服务器 → 国内服务器
     ✅            ✅ 拉取         ✅ 运行监控
```

## 方案 A：香港服务器定时拉取 + 同步到国内

### 1. 香港服务器定时任务

```bash
# 香港服务器 crontab
# 每30分钟拉取数据，然后推送到国内

*/30 9-15 * * 1-5 /home/scripts/fetch_and_push.sh
```

### 2. fetch_and_push.sh

```bash
#!/bin/bash
# 香港服务器运行

# 1. 拉取东方财富数据
python3 /home/scripts/fetch_eastmoney.py

# 2. 推送到国内服务器（SSH）
rsync -avz /home/data/*.csv 国内服务器IP:/root/github/stock-quant/python/data/

# 或者推送到 OSS（两边都免费）
ossutil cp /home/data/*.csv oss://stock-quant-data/data/ -r
```

## 方案 B：国内服务器通过香港代理访问

### 1. 香港服务器搭建 SOCKS5 代理

```bash
# 香港服务器安装
yum install dante  # 或 apt install dante-server

# 配置 /etc/danted.conf
internal: 0.0.0.0 port = 1080
external: eth0
method: none
client pass {
    from: 国内服务器IP/32 to: 0.0.0.0/0
}

# 启动
systemctl start danted
```

### 2. 国内服务器通过代理访问

```python
import requests

proxies = {
    'http': 'socks5://香港服务器IP:1080',
    'https': 'socks5://香港服务器IP:1080'
}

r = requests.get('https://push2his.eastmoney.com/...', proxies=proxies)
```

## 方案 C：香港服务器 API 网关

### 香港服务器运行简单的转发服务

```python
# 香港服务器运行 Flask
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/api/kline')
def get_kline():
    symbol = request.args.get('symbol')
    klt = request.args.get('klt', '30')
    
    # 转发到东方财富
    r = requests.get(
        'https://push2his.eastmoney.com/api/qt/stock/kline/get',
        params={
            'secid': f'0.{symbol}',
            'klt': klt,
            'fqt': '1',
            'beg': '0',
            'end': '20500000',
            'fields1': 'f1,f2,f3,f4,f5,f6',
            'fields2': 'f51,f52,f53,f54,f55,f56'
        },
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    
    return jsonify(r.json())

app.run(host='0.0.0.0', port=5000)
```

### 国内服务器调用香港网关

```python
# 国内服务器
r = requests.get('http://香港服务器IP:5000/api/kline?symbol=000001&klt=30')
data = r.json()
```

## 推荐选择

| 方案 | 复杂度 | 依赖 | 实时性 |
|------|--------|------|--------|
| **A. 定时同步** | ⭐⭐ | SSH/OSS | 每30分钟 |
| **B. SOCKS代理** | ⭐⭐⭐ | 需要代理服务 | 实时 |
| **C. API网关** | ⭐⭐ | Flask服务 | 实时 |

最简单：方案 A（香港定时拉取 + rsync 推送国内）

