#!/usr/bin/env python3
"""
定时任务监控脚本
每 30 分钟自动运行交易策略并发送通知

使用方法:
    python monitor_scheduler.py

或后台运行:
    nohup python monitor_scheduler.py > monitor.log 2>&1 &
"""

import os
import sys
import time
import signal
import schedule
from datetime import datetime
from typing import Callable, List

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.intraday_strategy import IntradayStrategy, WATCHLIST_STOCKS, webhook_callback


class SchedulerMonitor:
    """策略监控调度器"""

    def __init__(self, interval_minutes: int = 30):
        """
        初始化调度器

        Args:
            interval_minutes: 检查间隔（分钟）
        """
        self.interval_minutes = interval_minutes
        # 初始化时创建策略实例，但每次运行时会重新创建以强制刷新数据
        self.watchlist = WATCHLIST_STOCKS

        # 运行标志
        self.running = False

        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """处理停止信号"""
        print(f"\n收到信号 {signum}，正在停止...")
        self.running = False

    def run_strategy(self):
        """执行一次策略检查 - 每次都强制刷新数据"""
        print(f"\n{'='*60}")
        print(f"执行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"数据刷新模式：强制刷新（获取实时数据）")
        print(f"{'='*60}")

        try:
            # 每次运行都创建新的策略实例，确保强制刷新数据
            strategy = IntradayStrategy(
                watchlist=self.watchlist,
                notify_enabled=True,
                force_refresh=True  # 强制获取实时数据
            )

            signals = strategy.check_all_stocks()

            # 汇总统计
            buy_signals = [s for s in signals if "买入" in s['signal']]
            sell_signals = [s for s in signals if "卖出" in s['signal']]
            hold_signals = [s for s in signals if s['signal'] == '持有']

            print(f"\n统计：买入={len(buy_signals)}, 卖出={len(sell_signals)}, 持有={len(hold_signals)}")

            # 写入 JSON 日志
            self._write_json_log(signals)

            return signals

        except Exception as e:
            sys.stderr.write(f"策略执行失败：{e}\n")
            import traceback
            traceback.print_exc()
            return []

    def _write_json_log(self, signals: List):
        """写入 JSON 格式的日志"""
        log_dir = os.path.join(os.path.dirname(__file__), '../logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(log_dir, f"monitor_{datetime.now().strftime('%Y%m%d')}.json")

        log_data = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "signals": signals,
            "summary": {
                "total": len(signals),
                "buy": sum(1 for s in signals if "买入" in s['signal']),
                "sell": sum(1 for s in signals if "卖出" in s['signal']),
                "hold": sum(1 for s in signals if s['signal'] == '持有')
            }
        }

        with open(log_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(log_data, f, ensure_ascii=False, indent=2)

    def start(self):
        """启动调度器"""
        self.running = True

        print("=" * 60)
        print("30 分钟级别交易策略监控")
        print("=" * 60)
        print(f"监控股票：{len(WATCHLIST_STOCKS)} 只")
        for stock in WATCHLIST_STOCKS:
            print(f"  • {stock['name']} ({stock['symbol']})")
        print(f"检查间隔：{self.interval_minutes} 分钟")
        print(f"启动时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # 立即执行一次
        self.run_strategy()

        # 设置定时任务
        schedule.every(self.interval_minutes).minutes.do(self.run_strategy)

        # 运行循环
        while self.running:
            schedule.run_pending()
            time.sleep(1)

        print("调度器已停止")

    def run_once(self):
        """只执行一次"""
        return self.run_strategy()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='30 分钟级别交易策略监控')
    parser.add_argument(
        '--interval',
        type=int,
        default=30,
        help='检查间隔（分钟），默认 30'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='只执行一次，不启动定时任务'
    )
    parser.add_argument(
        '--watchlist',
        type=str,
        nargs='+',
        help='指定股票代码列表'
    )

    args = parser.parse_args()

    monitor = SchedulerMonitor(interval_minutes=args.interval)

    if args.once:
        monitor.run_once()
    else:
        monitor.start()


if __name__ == "__main__":
    main()
