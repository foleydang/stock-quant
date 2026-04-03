#!/usr/bin/env python3
"""测试邮件发送"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 测试配置（需要用户修改）
SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.qq.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', '465'))
SMTP_USERNAME = os.environ.get('SMTP_USERNAME', '')
SMTP_PASSWORD = os.environ.get('SMTP_PASSWORD', '')
EMAIL_RECEIVERS = os.environ.get('EMAIL_RECEIVERS', '21725056@zju.edu.cn').split(',')

if not SMTP_USERNAME or not SMTP_PASSWORD:
    print("=" * 60)
    print("测试邮件发送")
    print("=" * 60)
    print()
    print("请先配置环境变量:")
    print()
    print("  export SMTP_SERVER='smtp.qq.com'")
    print("  export SMTP_PORT='465'")
    print("  export SMTP_USERNAME='your_qq@qq.com'")
    print("  export SMTP_PASSWORD='your_auth_code'")
    print("  export EMAIL_RECEIVERS='21725056@zju.edu.cn'")
    print()
    print("或者运行配置脚本:")
    print("  cd /Users/foleydang/github/stock-quant/stock-quant")
    print("  ./setup_monitor.sh")
    print()
    sys.exit(0)

from strategy.email_notifier import EmailNotifier

notifier = EmailNotifier(
    smtp_server=SMTP_SERVER,
    smtp_port=SMTP_PORT,
    username=SMTP_USERNAME,
    password=SMTP_PASSWORD,
    receivers=EMAIL_RECEIVERS
)

# 测试邮件
test_signal = {
    "symbol": "300015.SZ",
    "stock_name": "爱尔眼科",
    "price": 9.65,
    "signal": "买入",
    "score": 2,
    "reasons": ["今日涨跌幅：-3.31%", "RSI 超卖"],
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

print("发送测试邮件...")
success = notifier.send_trading_signal(test_signal)

if success:
    print("✓ 测试邮件已发送，请查收！")
else:
    print("✗ 邮件发送失败，请检查配置")
