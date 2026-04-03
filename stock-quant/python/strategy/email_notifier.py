#!/usr/bin/env python3
"""
邮件通知模块
发送交易信号到指定邮箱
"""

import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from datetime import datetime
from typing import Dict, List, Optional


class EmailNotifier:
    """邮件通知器"""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        receivers: List[str],
        use_ssl: bool = True,
        use_tls: bool = True
    ):
        """
        初始化邮件通知器

        Args:
            smtp_server: SMTP 服务器地址，如 'smtp.gmail.com'
            smtp_port: SMTP 端口，如 465 (SSL) 或 587 (TLS)
            username: 发件人邮箱
            password: 邮箱密码或授权码
            receivers: 收件人列表
            use_ssl: 是否使用 SSL
            use_tls: 是否使用 TLS
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.receivers = receivers
        self.use_ssl = use_ssl
        self.use_tls = use_tls

    def send_trading_signal(self, signal: Dict, stock_name: str = None) -> bool:
        """
        发送交易信号邮件

        Args:
            signal: 交易信号字典
            stock_name: 股票名称

        Returns:
            发送是否成功
        """
        # 构建邮件内容
        subject = f"交易信号：{stock_name or signal.get('symbol', '')} - {signal.get('signal', '')}"

        # HTML 内容
        html_content = self._build_html_email(signal, stock_name)

        # 纯文本内容
        text_content = self._build_text_email(signal, stock_name)

        return self.send(subject, text_content, html_content)

    def send_daily_summary(self, signals: List[Dict], date: str = None) -> bool:
        """
        发送每日汇总邮件

        Args:
            signals: 信号列表
            date: 日期

        Returns:
            发送是否成功
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        # 获取当前时间（包含时分）
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

        # 统计
        buy_count = len([s for s in signals if "买入" in s.get('signal', '')])
        sell_count = len([s for s in signals if "卖出" in s.get('signal', '')])
        hold_count = len([s for s in signals if s.get('signal') == '持有'])

        # 汇总信号类型
        signal_summary = []
        if buy_count > 0:
            signal_summary.append(f"{buy_count}只买入")
        if sell_count > 0:
            signal_summary.append(f"{sell_count}只卖出")

        subject = f"交易信号提醒 - {timestamp} - {', '.join(signal_summary)}信号"

        # 统计
        buy_count = len([s for s in signals if "买入" in s.get('signal', '')])
        sell_count = len([s for s in signals if "卖出" in s.get('signal', '')])
        hold_count = len([s for s in signals if s.get('signal') == '持有'])

        html_content = f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .buy {{ color: red; }}
        .sell {{ color: green; }}
        .hold {{ color: gray; }}
    </style>
</head>
<body>
    <h2>交易信号提醒</h2>
    <p>时间：{timestamp}</p>
    <p>汇总：{', '.join(signal_summary)}</p>

    <h3>详细信号</h3>
    <table>
        <tr>
            <th>股票</th>
            <th>信号</th>
            <th>价格</th>
            <th>评分</th>
            <th>原因</th>
        </tr>
        {''.join([self._signal_to_row(s) for s in signals])}
    </table>
</body>
</html>
"""

        text_content = f"交易信号提醒 - {timestamp}\n\n汇总：{', '.join(signal_summary) if signal_summary else '无操作信号'}"

        return self.send(subject, text_content, html_content)

    def send(self, subject: str, text_content: str, html_content: str = None) -> bool:
        """
        发送邮件

        Args:
            subject: 邮件主题
            text_content: 纯文本内容
            html_content: HTML 内容（可选）

        Returns:
            发送是否成功
        """
        try:
            # 创建邮件
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.username
            msg['To'] = ', '.join(self.receivers)

            # 添加纯文本部分
            part1 = MIMEText(text_content, 'plain', 'utf-8')
            msg.attach(part1)

            # 添加 HTML 部分
            if html_content:
                part2 = MIMEText(html_content, 'html', 'utf-8')
                msg.attach(part2)

            # 发送邮件 - 重试 3 次
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"  发送邮件... (尝试 {attempt + 1}/{max_retries})")

                    # 使用 SSL 连接
                    context = ssl.create_default_context()
                    context.check_hostname = False
                    context.verify_mode = ssl.CERT_NONE

                    server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context, timeout=30)
                    server.set_debuglevel(0)

                    server.login(self.username, self.password)
                    server.sendmail(self.username, self.receivers, msg.as_string())
                    server.quit()

                    print(f"✓ 邮件已发送：{subject}")
                    return True

                except (smtplib.SMTPServerDisconnected, smtplib.SMTPSenderRefused) as e:
                    print(f"  重试中... {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2)

        except Exception as e:
            print(f"✗ 邮件发送失败：{e}")
            return False

    def _build_html_email(self, signal: Dict, stock_name: str = None) -> str:
        """构建 HTML 邮件"""
        symbol = signal.get('symbol', '')
        signal_type = signal.get('signal', '')
        price = signal.get('price', 0)
        score = signal.get('score', 0)
        reasons = signal.get('reasons', [])
        indicators = signal.get('indicators', {})
        timestamp = signal.get('timestamp', datetime.now().isoformat())

        # 信号颜色
        if '买入' in signal_type:
            signal_color = 'red'
            signal_bg = '#ffe6e6'
        elif '卖出' in signal_type:
            signal_color = 'green'
            signal_bg = '#e6ffe6'
        else:
            signal_color = 'gray'
            signal_bg = '#f0f0f0'

        reasons_html = '<br>'.join([f"• {r}" for r in reasons])

    def _fmt_bb(self, value) -> str:
        """格式化布林带数据"""
        if value is None or value == 'N/A':
            return 'N/A'
        try:
            return f"{float(value):.2f}"
        except (ValueError, TypeError):
            return 'N/A'

        html = f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }}
        .signal {{ display: inline-block; padding: 10px 20px; border-radius: 5px; font-weight: bold; margin: 10px 0; }}
        .info-box {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; }}
        .indicator {{ display: inline-block; margin: 5px 10px; padding: 5px 10px; background: #e9ecef; border-radius: 3px; }}
        .footer {{ color: #666; font-size: 12px; margin-top: 20px; border-top: 1px solid #ddd; padding-top: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 交易信号通知</h1>
            <p>{stock_name or symbol} ({symbol})</p>
        </div>

        <div class="signal" style="background: {signal_bg}; color: {signal_color};">
            {signal_type}
        </div>

        <div class="info-box">
            <p><strong>当前价格：</strong>{price:.2f}</p>
            <p><strong>信号评分：</strong>{score}</p>
            <p><strong>时间：</strong>{timestamp}</p>
        </div>

        <h3>触发原因</h3>
        <p>{reasons_html}</p>

        <h3>技术指标</h3>
        <div>
            <span class="indicator">RSI: {indicators.get('rsi', 'N/A')}</span>
            <span class="indicator">MACD: {indicators.get('macd', 'N/A')}</span>
            <span class="indicator">KDJ: K={indicators.get('kdj_k', 'N/A')}</span>
        </div>
        <div>
            <span class="indicator">ADX: {indicators.get('adx', 'N/A')}</span>
            <span class="indicator">MFI: {indicators.get('mfi', 'N/A')}</span>
            <span class="indicator">ATR: {indicators.get('atr', 'N/A')}</span>
        </div>
        <div>
            <span class="indicator">MA5: {indicators.get('ma5', 'N/A')}</span>
            <span class="indicator">MA20: {indicators.get('ma20', 'N/A')}</span>
            <span class="indicator">布林带：{self._fmt_bb(indicators.get('lower_bb'))} - {self._fmt_bb(indicators.get('upper_bb'))}</span>
        </div>
        <div>
            <span class="indicator">VWAP: {self._fmt_bb(indicators.get('vwap'))}</span>
            <span class="indicator">止损：{self._fmt_bb(signal.get('stop_loss'))}</span>
            <span class="indicator">止盈：{self._fmt_bb(signal.get('take_profit'))}</span>
        </div>

        <div class="footer">
            <p>此邮件由量化交易系统自动发送，仅供参考，不构成投资建议。</p>
            <p>投资有风险，入市需谨慎。</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def _build_text_email(self, signal: Dict, stock_name: str = None) -> str:
        """构建纯文本邮件"""
        symbol = signal.get('symbol', '')
        signal_type = signal.get('signal', '')
        price = signal.get('price', 0)
        score = signal.get('score', 0)
        reasons = signal.get('reasons', [])
        indicators = signal.get('indicators', {})
        timestamp = signal.get('timestamp', datetime.now().isoformat())

        text = f"""
交易信号通知
{'='*50}
股票：{stock_name or symbol} ({symbol})
信号：{signal_type}
价格：{price:.2f}
评分：{score}
时间：{timestamp}

触发原因:
{chr(10).join(['• ' + r for r in reasons])}

技术指标:
RSI: {indicators.get('rsi', 'N/A')}
MACD: {indicators.get('macd', 'N/A')}
KDJ: K={indicators.get('kdj_k', 'N/A')}, D={indicators.get('kdj_d', 'N/A')}
ADX: {indicators.get('adx', 'N/A')} (趋势强度)
MFI: {indicators.get('mfi', 'N/A')} (资金流量)
ATR: {indicators.get('atr', 'N/A')} (平均波幅)
MA5: {indicators.get('ma5', 'N/A')}, MA20: {indicators.get('ma20', 'N/A')}
VWAP: {indicators.get('vwap', 'N/A')}
止损位：{signal.get('stop_loss', 'N/A')}
止盈位：{signal.get('take_profit', 'N/A')}

---
此邮件由量化交易系统自动发送，仅供参考，不构成投资建议。
投资有风险，入市需谨慎。
"""
        return text

    def _signal_to_row(self, signal: Dict) -> str:
        """将信号转换为表格行"""
        symbol = signal.get('symbol', '')
        name = signal.get('stock_name', '')
        signal_type = signal.get('signal', '')
        # 优先使用 current_price，其次用 price
        price = signal.get('current_price', signal.get('price', 0))
        score = signal.get('score', 0)
        reasons = ', '.join(signal.get('reasons', [])[:3])  # 只显示前3个原因

        # 信号样式
        if '买入' in signal_type:
            signal_class = 'buy'
        elif '卖出' in signal_type:
            signal_class = 'sell'
        else:
            signal_class = 'hold'

        return f"""
        <tr>
            <td>{name} ({symbol})</td>
            <td class="{signal_class}">{signal_type}</td>
            <td>{price:.2f}</td>
            <td>{score}</td>
            <td>{reasons}</td>
        </tr>
        """


def create_email_notifier_from_env() -> Optional[EmailNotifier]:
    """
    从环境变量创建邮件通知器

    支持的环境变量:
    - SMTP_SERVER: SMTP 服务器，如 'smtp.gmail.com'
    - SMTP_PORT: SMTP 端口，如 465 或 587
    - SMTP_USERNAME: 发件人邮箱
    - SMTP_PASSWORD: 邮箱密码或授权码
    - EMAIL_RECEIVERS: 收件人列表 (逗号分隔)
    """
    smtp_server = os.environ.get('SMTP_SERVER')
    smtp_port = os.environ.get('SMTP_PORT')
    smtp_username = os.environ.get('SMTP_USERNAME')
    smtp_password = os.environ.get('SMTP_PASSWORD')
    email_receivers = os.environ.get('EMAIL_RECEIVERS')

    if not all([smtp_server, smtp_port, smtp_username, smtp_password, email_receivers]):
        print("邮件配置不完整，请设置以下环境变量:")
        print("  SMTP_SERVER: SMTP 服务器地址")
        print("  SMTP_PORT: SMTP 端口")
        print("  SMTP_USERNAME: 发件人邮箱")
        print("  SMTP_PASSWORD: 邮箱密码/授权码")
        print("  EMAIL_RECEIVERS: 收件人列表 (逗号分隔)")
        return None

    return EmailNotifier(
        smtp_server=smtp_server,
        smtp_port=int(smtp_port),
        username=smtp_username,
        password=smtp_password,
        receivers=email_receivers.split(',')
    )


# 测试
if __name__ == "__main__":
    # 从环境变量创建通知器
    notifier = create_email_notifier_from_env()

    if notifier:
        # 测试信号
        test_signal = {
            "symbol": "300015.SZ",
            "stock_name": "爱尔眼科",
            "price": 9.66,
            "signal": "买入",
            "score": 3,
            "reasons": ["RSI 超卖 (28.5)", "MACD 金叉", "短期均线向上"],
            "indicators": {
                "rsi": 28.5,
                "macd": 0.0012,
                "kdj_k": 35.2,
                "kdj_d": 28.4,
                "ma5": 9.8,
                "ma20": 10.2,
                "lower_bb": 9.5,
                "upper_bb": 10.5
            }
        }

        # 发送测试邮件
        notifier.send_trading_signal(test_signal)
    else:
        print("\n无法创建邮件通知器，请配置环境变量")
        print("\n示例配置:")
        print("  export SMTP_SERVER='smtp.gmail.com'")
        print("  export SMTP_PORT='465'")
        print("  export SMTP_USERNAME='your_email@gmail.com'")
        print("  export SMTP_PASSWORD='your_app_password'")
        print("  export EMAIL_RECEIVERS='21725056@zju.edu.cn'")
