#!/usr/bin/env python3
"""
30 分钟级别多因子交易策略测试（使用本地数据）
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.intraday_strategy import IntradayStrategy, TechnicalIndicators


def test_with_local_data():
    """使用本地数据测试策略"""

    # 读取本地数据
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    test_symbols = [
        ('300015.SZ', '爱尔眼科'),
        ('300124.SZ', '汇川技术'),
        ('600048.SH', '保利发展'),
        ('000001.SZ', '平安银行'),  # 使用已有数据的股票
        ('600519.SH', '贵州茅台'),
    ]

    print("=" * 60)
    print("30 分钟级别多因子交易策略 - 本地数据测试")
    print("=" * 60)

    strategy = IntradayStrategy(watchlist=[], notify_enabled=True)

    results = []

    for symbol, name in test_symbols:
        data_file = os.path.join(data_dir, f"{symbol}_processed.csv")

        if not os.path.exists(data_file):
            print(f"⚠️  数据文件不存在：{symbol}")
            continue

        try:
            df = pd.read_csv(data_file)

            if len(df) < 60:
                print(f"⚠️  数据不足：{symbol} (只有 {len(df)} 行)")
                continue

            print(f"\n分析股票：{name} ({symbol}) - {len(df)} 条数据")

            # 生成信号
            signal = strategy.generate_signal(symbol, df)

            if signal:
                signal['stock_name'] = name
                results.append(signal)

                # 打印信号
                emoji = "🟢" if "买入" in signal['signal'] else "🔴" if "卖出" in signal['signal'] else "⚪"
                print(f"  {emoji} 信号：{signal['signal']}")
                print(f"     价格：{signal['price']:.2f}")
                print(f"     评分：{signal['score']}")
                print(f"     原因：{', '.join(signal['reasons'][:3])}")
                print(f"     RSI: {signal['indicators']['rsi']:.2f}")
                print(f"     MACD: {signal['indicators']['macd']:.4f}")

                # 如果不是持有，发送通知
                if signal['signal'] != '持有':
                    strategy.send_notification(signal)

        except Exception as e:
            print(f"⚠️  分析失败：{symbol} - {e}")

    # 汇总
    print("\n" + "=" * 60)
    print("信号汇总")
    print("=" * 60)

    if results:
        buy_count = len([s for s in results if "买入" in s['signal']])
        sell_count = len([s for s in results if "卖出" in s['signal']])
        hold_count = len([s for s in results if s['signal'] == '持有'])

        print(f"总计：{len(results)} 只股票")
        print(f"买入：{buy_count} | 卖出：{sell_count} | 持有：{hold_count}")

        print("\n详细信号:")
        for r in results:
            emoji = "🟢" if "买入" in r['signal'] else "🔴" if "卖出" in r['signal'] else "⚪"
            print(f"  {emoji} {r['stock_name']}: {r['signal']} (评分：{r['score']})")
    else:
        print("暂无信号")

    print("=" * 60)

    return results


def test_indicators():
    """测试技术指标计算"""
    print("\n" + "=" * 60)
    print("技术指标计算测试")
    print("=" * 60)

    # 生成测试数据
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(100)) + 100
    high = prices + np.random.rand(100) * 2
    low = prices - np.random.rand(100) * 2

    # 测试 RSI
    rsi = TechnicalIndicators.calculate_rsi(prices, 14)
    print(f"RSI(14) 最后值：{rsi[-1]:.2f}")

    # 测试 MACD
    macd = TechnicalIndicators.calculate_macd(prices)
    print(f"MACD 最后值：{macd['macd'][-1]:.4f}")
    print(f"Signal 最后值：{macd['signal'][-1]:.4f}")

    # 测试布林带
    bb = TechnicalIndicators.calculate_bollinger_bands(prices, 20, 2.0)
    print(f"布林带：[{bb['lower'][-1]:.2f}, {bb['mid'][-1]:.2f}, {bb['upper'][-1]:.2f}]")

    # 测试 KDJ
    kdj = TechnicalIndicators.calculate_kdj(high, low, prices)
    print(f"KDJ: K={kdj['k'][-1]:.2f}, D={kdj['d'][-1]:.2f}, J={kdj['j'][-1]:.2f}")

    print("=" * 60)


if __name__ == "__main__":
    # 测试技术指标
    test_indicators()

    # 使用本地数据测试策略
    test_with_local_data()
