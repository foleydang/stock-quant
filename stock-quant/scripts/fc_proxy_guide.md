# 阿里云函数计算 - 数据代理方案

## 原理

函数计算 + 住宅代理 → 拉取东方财富数据 → 存到 OSS → 服务器下载

完全绕过服务器 IP 封锁问题

## 1. 开通函数计算

控制台：https://fc.console.aliyun.com/
- 选择同地域
- 创建服务：stock-data-proxy

## 2. Python 函数代码

```python
# 函数入口
import requests
import json
import oss2

def handler(event, context):
    """拉取东方财富分钟数据"""
    
    # 解析参数
    params = json.loads(event)
    symbols = params.get('symbols', ['600036'])
    
    # 配置住宅代理（需购买，约 ¥50/月）
    proxies = {
        'http': 'http://your-proxy:port',
        'https': 'http://your-proxy:port'
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 Chrome/120.0.0.0'
    }
    
    # 拉取数据
    results = {}
    for symbol in symbols:
        url = f'https://push2his.eastmoney.com/api/qt/stock/kline/get'
        params = {
            'secid': f'1.{symbol}',
            'klt': '30',
            'fqt': '1',
            'fields1': 'f1,f2,f3,f4,f5,f6',
            'fields2': 'f51,f52,f53,f54,f55,f56'
        }
        
        r = requests.get(url, params=params, headers=headers, proxies=proxies)
        data = r.json()
        
        if data.get('data'):
            results[symbol] = data['data']['klines']
    
    # 存到 OSS
    bucket = oss2.Bucket(oss2.Auth('ACCESS_KEY', 'SECRET_KEY'), 
                         'oss-cn-hangzhou.aliyuncs.com', 'stock-quant-data')
    bucket.put_object('minute_data.json', json.dumps(results))
    
    return {'status': 'ok', 'count': len(results)}
```

## 3. 服务器调用函数

```python
import requests

# 阿里云函数计算 HTTP 触发器
FC_URL = 'https://your-service.cn-hangzhou.fc.aliyuncs.com/stock-data-proxy'

def fetch_minute_data():
    r = requests.post(FC_URL, json={'symbols': ['600036', '000001']})
    return r.json()
```

## 4. 定时触发

函数计算支持定时触发器：
- 每交易日 9:00, 15:00 自动拉取
- 服务器按需读取

## 费用估算

| 项目 | 费用 |
|------|------|
| 函数调用 | 免费（100万次额度） |
| 函数执行 | 免费（400GB秒额度） |
| OSS 存储 | 免费 |
| 住宅代理 | ¥50-100/月（可选） |
| **总计** | ¥0-100/月 |

## 注意

住宅代理不是必须的：
- 函数计算出口 IP 可能不被封锁
- 先测试不带代理能否访问
- 如果被封锁再加代理

