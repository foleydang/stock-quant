#!/usr/bin/env python3
"""
交易通知推送模块
支持多种通知渠道：
- 钉钉
- 企业微信
- 飞书
- 邮件
- 控制台
"""

import os
import json
import smtplib
import hashlib
import hmac
import base64
import time
from email.mime.text import MIMEText
from email.header import Header
from datetime import datetime
from typing import Dict, List, Optional, Callable
from urllib.parse import quote
import requests


class NotificationChannel:
    """通知渠道基类"""

    def send(self, title: str, content: str, **kwargs) -> bool:
        raise NotImplementedError


class DingTalkNotifier(NotificationChannel):
    """钉钉机器人通知"""

    def __init__(self, webhook_url: str, secret: str = None):
        """
        初始化钉钉通知

        Args:
            webhook_url: 钉钉机器人 webhook 地址
            secret: 加签密钥（可选）
        """
        self.webhook_url = webhook_url
        self.secret = secret

    def _generate_sign(self) -> str:
        """生成钉钉加签"""
        if not self.secret:
            return ""

        timestamp = str(round(time.time() * 1000))
        secret_enc = self.secret.encode('utf-8')
        string_to_sign = f'{timestamp}\n{self.secret}'
        string_to_sign_enc = string_to_sign.encode('utf-8')

        hmac_code = hmac.new(
            secret_enc,
            string_to_sign_enc,
            digestmod=hashlib.sha256
        ).digest()

        sign = quote(base64.b64encode(hmac_code))
        return f"&timestamp={timestamp}&sign={sign}"

    def send(self, title: str, content: str, **kwargs) -> bool:
        """发送钉钉消息"""
        try:
            url = self.webhook_url
            if self.secret:
                url += self._generate_sign()

            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "title": title,
                    "text": f"# {title}\n\n{content}"
                },
                "at": {
                    "isAtAll": kwargs.get('at_all', True)
                }
            }

            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, json=payload, headers=headers, timeout=10)

            result = response.json()
            return result.get('errcode') == 0

        except Exception as e:
            print(f"钉钉通知失败：{e}")
            return False


class WeChatNotifier(NotificationChannel):
    """企业微信机器人通知"""

    def __init__(self, webhook_url: str):
        """
        初始化企业微信通知

        Args:
            webhook_url: 企业微信机器人 webhook 地址
        """
        self.webhook_url = webhook_url

    def send(self, title: str, content: str, **kwargs) -> bool:
        """发送企业微信消息"""
        try:
            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "content": f"# {title}\n\n{content}"
                }
            }

            headers = {'Content-Type': 'application/json'}
            response = requests.post(self.webhook_url, json=payload, headers=headers, timeout=10)

            result = response.json()
            return result.get('errmsg') == 'success'

        except Exception as e:
            print(f"企业微信通知失败：{e}")
            return False


class FeishuNotifier(NotificationChannel):
    """飞书机器人通知"""

    def __init__(self, webhook_url: str, secret: str = None):
        """
        初始化飞书通知

        Args:
            webhook_url: 飞书机器人 webhook 地址
            secret: 加签密钥（可选）
        """
        self.webhook_url = webhook_url
        self.secret = secret

    def _generate_sign(self, timestamp: str) -> str:
        """生成飞书签名"""
        if not self.secret:
            return ""

        string_to_sign = f'{timestamp}\n{self.secret}'
        hmac_code = hmac.new(
            string_to_sign.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()

        return base64.b64encode(hmac_code).decode('utf-8')

    def send(self, title: str, content: str, **kwargs) -> bool:
        """发送飞书消息"""
        try:
            timestamp = str(int(time.time()))

            payload = {
                "msg_type": "interactive",
                "card": {
                    "header": {
                        "title": {
                            "tag": "plain_text",
                            "content": title
                        },
                        "template": kwargs.get('template', 'blue')
                    },
                    "elements": [
                        {
                            "tag": "markdown",
                            "content": content
                        }
                    ]
                }
            }

            headers = {
                'Content-Type': 'application/json',
                'X-Sign-Timestamp': timestamp
            }

            if self.secret:
                headers['X-Sign-SHA256'] = self._generate_sign(timestamp)

            response = requests.post(self.webhook_url, json=payload, headers=headers, timeout=10)
            result = response.json()

            return result.get('StatusCode') == 0 or result.get('code') == 0

        except Exception as e:
            print(f"飞书通知失败：{e}")
            return False


class EmailNotifier(NotificationChannel):
    """邮件通知"""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        receivers: List[str],
        use_ssl: bool = True
    ):
        """
        初始化邮件通知

        Args:
            smtp_server: SMTP 服务器地址
            smtp_port: SMTP 端口
            username: 用户名
            password: 密码/授权码
            receivers: 收件人列表
            use_ssl: 是否使用 SSL
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.receivers = receivers
        self.use_ssl = use_ssl

    def send(self, title: str, content: str, **kwargs) -> bool:
        """发送邮件"""
        try:
            msg = MIMEText(content, 'plain', 'utf-8')
            msg['From'] = Header(self.username, 'utf-8')
            msg['To'] = Header(', '.join(self.receivers), 'utf-8')
            msg['Subject'] = Header(title, 'utf-8')

            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            else:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)

            server.login(self.username, self.password)
            server.sendmail(self.username, self.receivers, msg.as_string())
            server.quit()

            return True

        except Exception as e:
            print(f"邮件通知失败：{e}")
            return False


class ConsoleNotifier(NotificationChannel):
    """控制台通知"""

    def send(self, title: str, content: str, **kwargs) -> bool:
        """输出到控制台"""
        print("\n" + "=" * 60)
        print(f"📢 {title}")
        print("=" * 60)
        print(content)
        print("=" * 60 + "\n")
        return True


class NotificationManager:
    """通知管理器"""

    def __init__(self):
        """初始化通知管理器"""
        self.channels: Dict[str, NotificationChannel] = {}
        self.enabled = True

    def add_channel(self, name: str, channel: NotificationChannel):
        """
        添加通知渠道

        Args:
            name: 渠道名称
            channel: 通知渠道实例
        """
        self.channels[name] = channel
        print(f"已添加通知渠道：{name}")

    def remove_channel(self, name: str):
        """移除通知渠道"""
        if name in self.channels:
            del self.channels[name]

    def send(
        self,
        title: str,
        content: str,
        channels: List[str] = None,
        **kwargs
    ) -> Dict[str, bool]:
        """
        发送通知

        Args:
            title: 消息标题
            content: 消息内容
            channels: 指定渠道列表，None 表示所有渠道
            **kwargs: 额外参数

        Returns:
            各渠道发送结果
        """
        if not self.enabled:
            return {}

        results = {}
        target_channels = channels or list(self.channels.keys())

        for name in target_channels:
            if name in self.channels:
                try:
                    success = self.channels[name].send(title, content, **kwargs)
                    results[name] = success
                except Exception as e:
                    print(f"通知渠道 {name} 失败：{e}")
                    results[name] = False

        return results

    def enable(self):
        """启用通知"""
        self.enabled = True

    def disable(self):
        """禁用通知"""
        self.enabled = False


def create_notification_manager_from_env() -> NotificationManager:
    """
    从环境变量创建通知管理器

    支持的环境变量:
    - DINGTALK_WEBHOOK_URL: 钉钉 webhook 地址
    - DINGTALK_SECRET: 钉钉加签密钥
    - WECHAT_WEBHOOK_URL: 企业微信 webhook 地址
    - FEISHU_WEBHOOK_URL: 飞书 webhook 地址
    - FEISHU_SECRET: 飞书加签密钥
    - EMAIL_SMTP_SERVER: SMTP 服务器
    - EMAIL_SMTP_PORT: SMTP 端口
    - EMAIL_USERNAME: 邮箱用户名
    - EMAIL_PASSWORD: 邮箱密码/授权码
    - EMAIL_RECEIVERS: 收件人列表 (逗号分隔)
    """
    manager = NotificationManager()

    # 钉钉
    dingtalk_url = os.environ.get('DINGTALK_WEBHOOK_URL')
    if dingtalk_url:
        dingtalk_secret = os.environ.get('DINGTALK_SECRET')
        manager.add_channel(
            'dingtalk',
            DingTalkNotifier(dingtalk_url, dingtalk_secret)
        )

    # 企业微信
    wechat_url = os.environ.get('WECHAT_WEBHOOK_URL')
    if wechat_url:
        manager.add_channel(
            'wechat',
            WeChatNotifier(wechat_url)
        )

    # 飞书
    feishu_url = os.environ.get('FEISHU_WEBHOOK_URL')
    if feishu_url:
        feishu_secret = os.environ.get('FEISHU_SECRET')
        manager.add_channel(
            'feishu',
            FeishuNotifier(feishu_url, feishu_secret)
        )

    # 邮件
    email_server = os.environ.get('EMAIL_SMTP_SERVER')
    if email_server:
        try:
            manager.add_channel(
                'email',
                EmailNotifier(
                    smtp_server=email_server,
                    smtp_port=int(os.environ.get('EMAIL_SMTP_PORT', 465)),
                    username=os.environ.get('EMAIL_USERNAME', ''),
                    password=os.environ.get('EMAIL_PASSWORD', ''),
                    receivers=os.environ.get('EMAIL_RECEIVERS', '').split(',')
                )
            )
        except Exception as e:
            print(f"邮件配置错误：{e}")

    # 至少添加控制台通知
    if not manager.channels:
        manager.add_channel('console', ConsoleNotifier())
        print("未配置通知渠道，使用控制台输出")

    return manager


def format_trading_signal(signal: Dict) -> str:
    """
    格式化交易信号为消息内容

    Args:
        signal: 交易信号字典

    Returns:
        格式化后的消息内容
    """
    reasons = "\n".join([f"• {r}" for r in signal.get('reasons', [])])

    content = f"""
📈 股票：{signal.get('stock_name', signal.get('symbol', ''))} ({signal.get('symbol', '')})
💰 价格：{signal.get('price', 0):.2f}
📊 信号：{signal.get('signal', '')}
⭐ 评分：{signal.get('score', 0)}

🔍 触发原因:
{reasons}

📉 技术指标:
• RSI: {signal.get('indicators', {}).get('rsi', 0):.2f}
• MACD: {signal.get('indicators', {}).get('macd', 0):.4f}
• KDJ: K={signal.get('indicators', {}).get('kdj_k', 0):.2f}, D={signal.get('indicators', {}).get('kdj_d', 0):.2f}

⏰ 时间：{signal.get('timestamp', datetime.now().isoformat())}
"""
    return content


# 使用示例
if __name__ == "__main__":
    # 创建通知管理器
    manager = create_notification_manager_from_env()

    # 测试消息
    test_signal = {
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
        },
        "timestamp": datetime.now().isoformat()
    }

    title = f"交易信号：{test_signal['stock_name']} - {test_signal['signal']}"
    content = format_trading_signal(test_signal)

    # 发送通知
    results = manager.send(title, content)
    print(f"发送结果：{results}")
