#!/usr/bin/env python3
"""运行策略分析 - 使用 30 分钟级别实时数据"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy.intraday_strategy import IntradayStrategy, WATCHLIST_STOCKS


def main():
    print("=" * 60)
    print("30 分钟级别交易策略分析")
    print("=" * 60)
    print("数据源: akshare (30 分钟级别)")
    print("限流: 接口层内部自动处理")
    print("=" * 60)

    # 创建策略实例
    strategy = IntradayStrategy(
        watchlist=WATCHLIST_STOCKS,
        notify_enabled=False,
        force_refresh=True
    )

    # 检查所有股票
    signals = strategy.check_all_stocks()

    # 汇总
    print("\n" + "=" * 60)
    print("信号汇总")
    print("=" * 60)

    buy_count = sum(1 for s in signals if "买入" in s['signal'])
    sell_count = sum(1 for s in signals if "卖出" in s['signal'])
    hold_count = sum(1 for s in signals if s['signal'] == '持有')

    print(f"买入: {buy_count}, 持有: {hold_count}, 卖出: {sell_count}\n")

    for s in signals:
        emoji = "🔴" if "卖出" in s['signal'] else "🟢" if "买入" in s['signal'] else "⚪"
        print(f"{emoji} {s['stock_name']}: {s['signal']} | 价格:{s['price']:.2f} | RSI:{s['indicators']['rsi']:.1f} | 评分:{s['score']}")
        # 显示详细原因
        if 'reasons' in s and s['reasons']:
            for reason in s['reasons'][:5]:  # 只显示前5个原因
                print(f"   └─ {reason}")


if __name__ == "__main__":
    main()