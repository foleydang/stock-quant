#!/usr/bin/env python3
"""
策略报告生成器 - 生成详细的交易报告
"""
import os
import sys
import json
import sqlite3
from datetime import datetime
from typing import List, Dict

BASE_DIR = '/Users/foleydang/github/stock-quant/stock-quant/python'
DB_PATH = f'{BASE_DIR}/data/stock_data.db'
LOGS_DIR = f'{BASE_DIR}/logs'

def generate_report():
    """生成策略报告"""
    print("=" * 70)
    print("LGBM 量化交易策略报告")
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 1. 数据统计
    print("\n【数据统计】")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(DISTINCT symbol) FROM kline_30m')
    stock_count = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM kline_30m')
    kline_count = cursor.fetchone()[0]

    cursor.execute('SELECT MIN(date), MAX(date) FROM kline_30m')
    min_date, max_date = cursor.fetchone()

    print(f"  股票数量: {stock_count} 只")
    print(f"  K线数据: {kline_count:,} 条")
    print(f"  时间范围: {min_date[:10]} ~ {max_date[:10]}")

    conn.close()

    # 2. 读取最新策略结果
    print("\n【策略回测结果】")
    strategy_files = sorted([f for f in os.listdir(LOGS_DIR) if f.startswith('strategy_')], reverse=True)

    results = []
    if strategy_files:
        latest_file = os.path.join(LOGS_DIR, strategy_files[0])
        with open(latest_file, 'r') as f:
            data = json.load(f)

        # 支持两种JSON格式：直接列表或嵌套在selected_stocks下
        if isinstance(data, list):
            results = data
        elif isinstance(data, dict) and 'selected_stocks' in data:
            results = data['selected_stocks']

        if results:
            print(f"\n  Top 10 选股结果:")
            print(f"  {'排名':<6}{'股票代码':<15}{'收益率':<12}{'胜率':<12}{'交易次数':<10}")
            print("  " + "-" * 55)

            for i, r in enumerate(results[:10]):
                # 获取交易次数：优先使用 trade_count，否则检查 trades 是数字还是列表
                trade_count = r.get('trade_count')
                if trade_count is None:
                    trades_field = r.get('trades', 0)
                    if isinstance(trades_field, list):
                        trade_count = len(trades_field)
                    else:
                        trade_count = trades_field
                print(f"  #{i+1:<5}{r['symbol']:<15}{r['profit_rate']:>8.2f}%    {r['win_rate']:>6.1f}%      {trade_count:>5}次")

            # 3. 汇总统计
            total_profit = sum(r['profit_rate'] for r in results[:5]) / min(5, len(results))
            avg_win_rate = sum(r['win_rate'] for r in results[:5]) / min(5, len(results))

            print(f"\n  前5只股票平均收益率: {total_profit:.2f}%")
            print(f"  前5只股票平均胜率: {avg_win_rate:.1f}%")

    # 4. 策略参数
    print("\n【策略参数】")
    print("  初始资金: ¥100,000")
    print("  建仓比例: 90% (一次性建仓)")
    print("  止盈: 10%")
    print("  止损: 8%")
    print("  买入阈值: 预测上涨概率 > 60%")
    print("  卖出阈值: 预测上涨概率 < 40%")
    print("  T+1周期: 16个30分钟K线")

    # 5. 持仓建议
    print("\n【持仓建议】")
    if results:
        best = results[0]
        print(f"  推荐股票: {best['symbol']}")
        print(f"  预期收益: {best['profit_rate']:.2f}%")
        print(f"  风险提示: 胜率 {best['win_rate']:.1f}%")

        if best['profit_rate'] > 30:
            print("  操作建议: 强烈买入")
        elif best['profit_rate'] > 10:
            print("  操作建议: 买入")
        else:
            print("  操作建议: 观望")
    else:
        print("  无策略结果")

    print("\n" + "=" * 70)
    print("报告生成完成")
    print("=" * 70)

    # 保存报告
    report_path = os.path.join(LOGS_DIR, f"report_{datetime.now().strftime('%Y%m%d')}.txt")
    return report_path

if __name__ == "__main__":
    generate_report()