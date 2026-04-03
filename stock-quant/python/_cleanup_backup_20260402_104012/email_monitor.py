#!/usr/bin/env python3
"""
股票监控脚本 - 30 分钟级别
功能:
1. 使用 AKShare 获取 30 分钟数据
2. 执行多因子策略分析
3. 发送邮件通知

使用方法:
    python email_monitor.py
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_handler import DataHandler
from strategy.intraday_strategy import IntradayStrategy, SignalType
from strategy.email_notifier import EmailNotifier, create_email_notifier_from_env


# 配置的股票池
WATCHLIST_STOCKS = [
    {'symbol': '300015.SZ', 'name': '爱尔眼科'},
    {'symbol': '300124.SZ', 'name': '汇川技术'},
    {'symbol': '600048.SH', 'name': '保利发展'},
    {'symbol': '3690.HK', 'name': '美团-W'},
    {'symbol': '9988.HK', 'name': '阿里巴巴-W'},
    {'symbol': '0700.HK', 'name': '腾讯控股'},
    {'symbol': '600519.SH', 'name': '贵州茅台'},
]


class EmailMonitor:
    """邮件监控器 - 使用新的 DataHandler"""

    def __init__(self, email_notifier: EmailNotifier = None):
        """
        初始化监控器

        Args:
            email_notifier: 邮件通知器实例
        """
        # 使用新的 DataHandler（强制刷新数据）
        self.data_handler = DataHandler(force_refresh=True)
        self.strategy = IntradayStrategy(watchlist=[], notify_enabled=False)
        self.email_notifier = email_notifier

        # 策略参数
        self.notify_on_signal = ['强烈买入', '买入', '强烈卖出', '卖出']

    def run(self, send_email: bool = True) -> dict:
        """
        执行监控任务

        Args:
            send_email: 是否发送邮件

        Returns:
            监控结果
        """
        print("=" * 60)
        print(f"股票监控任务 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("数据源: AKShare (30 分钟级别)")
        print("=" * 60)

        # 1. 获取数据并执行策略分析
        signals = []

        for i, stock in enumerate(WATCHLIST_STOCKS):
            symbol = stock['symbol']
            name = stock['name']

            print(f"\n[{i+1}/{len(WATCHLIST_STOCKS)}] {name} ({symbol})...")

            # 获取 30 分钟数据（强制刷新）
            df = self.data_handler.fetch_stock_data(symbol, force_refresh=True)

            if df is None or len(df) < 60:
                print(f"  ⚠️ 数据不足 ({len(df) if df is not None else 0} 条)")
                continue

            # 最新价格
            latest_price = df['close'].iloc[-1]
            latest_time = df['date'].iloc[-1]
            print(f"  ✓ 价格: {latest_price:.2f} | 时间: {latest_time}")

            # 生成信号
            signal = self.strategy.generate_signal(symbol, df)

            if signal:
                signal['stock_name'] = name
                signal['current_price'] = latest_price
                signals.append(signal)

                # 打印信号
                emoji = "🟢" if "买入" in signal['signal'] else "🔴" if "卖出" in signal['signal'] else "⚪"
                print(f"  {emoji} 信号: {signal['signal']} (评分: {signal['score']})")

        # 2. 发送邮件通知
        if send_email and self.email_notifier and signals:
            print("\n" + "=" * 60)
            print("发送邮件通知...")

            # 筛选需要通知的信号
            notify_signals = [s for s in signals if s['signal'] in self.notify_on_signal]

            if notify_signals:
                print(f"  发现 {len(notify_signals)} 个重要信号，发送汇总邮件...")
                self.email_notifier.send_daily_summary(signals)
            else:
                print("  无重要信号，跳过邮件发送")

        # 3. 保存结果
        result = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'signals': signals,
            'summary': {
                'total': len(signals),
                'buy': len([s for s in signals if '买入' in s['signal']]),
                'sell': len([s for s in signals if '卖出' in s['signal']]),
                'hold': len([s for s in signals if s['signal'] == '持有'])
            }
        }

        # 保存到文件
        self._save_result(result)

        # 打印汇总
        print("\n" + "=" * 60)
        print("监控汇总")
        print("=" * 60)
        print(f"总计：{result['summary']['total']} 只股票")
        print(f"买入：{result['summary']['buy']} | 卖出：{result['summary']['sell']} | 持有：{result['summary']['hold']}")

        for s in signals:
            emoji = "🟢" if "买入" in s['signal'] else "🔴" if "卖出" in s['signal'] else "⚪"
            price = s.get('current_price', s.get('price', 0))
            print(f"  {emoji} {s['stock_name']}: {s['signal']} ({price:.2f})")

        return result

    def _save_result(self, result: dict):
        """保存结果到文件"""
        log_dir = os.path.join(os.path.dirname(__file__), '../logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 保存 JSON
        date_str = datetime.now().strftime('%Y%m%d')
        json_file = os.path.join(log_dir, f'monitor_{date_str}.json')

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # 保存日志
        log_file = os.path.join(log_dir, f'monitor_{date_str}.log')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Timestamp: {result['timestamp']}\n")
            f.write(f"Signals: {result['summary']}\n")
            for s in result['signals']:
                price = s.get('current_price', s.get('price', 0))
                f.write(f"  {s['stock_name']}: {s['signal']} ({price:.2f})\n")


def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════╗
║           股票监控系统 - 30 分钟级别                         ║
║           数据源: AKShare (实时刷新)                        ║
╚══════════════════════════════════════════════════════════╝
    """)

    # 创建邮件通知器
    email_notifier = create_email_notifier_from_env()

    if email_notifier:
        print("✓ 邮件通知已配置")
    else:
        print("⚠ 邮件通知未配置，将不发送邮件")

    # 创建监控器
    monitor = EmailMonitor(email_notifier)

    # 执行监控
    send_email = email_notifier is not None
    result = monitor.run(send_email=send_email)

    return result


if __name__ == "__main__":
    main()