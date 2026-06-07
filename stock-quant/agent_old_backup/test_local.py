#!/usr/bin/env python3
"""本地测试金融小助手"""
import os, sys, json, requests

# 加载 .env
env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
with open(env_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'python'))

from config import FEISHU_APP_ID, FEISHU_APP_SECRET, FEISHU_TARGET_OPEN_ID

print('=== 1. 配置检查 ===')
print(f'APP_ID: {FEISHU_APP_ID}')
print(f'APP_SECRET: {FEISHU_APP_SECRET}')
print(f'TARGET_OPEN_ID: {FEISHU_TARGET_OPEN_ID}')

# 测试飞书认证
print('\n=== 2. 飞书认证 ===')
url = 'https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal'
data = {'app_id': FEISHU_APP_ID, 'app_secret': FEISHU_APP_SECRET}
resp = requests.post(url, json=data)
result = resp.json()
print(f'code: {result.get("code")}')
print(f'msg: {result.get("msg")}')
if result.get('tenant_access_token'):
    token = result['tenant_access_token']
    print(f'token: {token[:20]}...')
else:
    print('❌ 认证失败！')
    sys.exit(1)

# 测试发消息
print('\n=== 3. 发送测试消息 ===')
send_url = 'https://open.feishu.cn/open-apis/im/v1/messages'
headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
payload = {
    'receive_id': FEISHU_TARGET_OPEN_ID,
    'msg_type': 'text',
    'content': json.dumps({"text": "🥔 金融小助手本地测试 - 如果看到这条消息说明飞书API正常"})
}
params = {'receive_id_type': 'open_id'}
resp2 = requests.post(send_url, headers=headers, json=payload, params=params)
result2 = resp2.json()
print(f'code: {result2.get("code")}')
print(f'msg: {result2.get("msg")}')
if result2.get('code') == 0:
    print('✅ 消息发送成功！去飞书看看有没有收到')
else:
    print(f'❌ 发送失败: {result2}')

# 测试消息处理
print('\n=== 4. 消息处理逻辑 ===')
from bot_server import process_message

for text in ['持仓', '行情 茅台', '信号', '总结', '帮助']:
    card = process_message(text)
    if card:
        print(f'  "{text}" → 卡片OK (type={card.get("msg_type","?")})')
    else:
        print(f'  "{text}" → ❌ 返回空')

print('\n=== 5. LLM 状态 ===')
from llm_client import is_available
print(f'LLM 可用: {is_available()}')