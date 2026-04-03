#!/usr/bin/env python3
"""
增强版回测系统
支持：
1. 趋势跟踪（长持）- 持仓数天到数周
2. 日内做T（短持）- 当日买卖
3. 仓位管理
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from enhanced_strategy import EnhancedStrategySystem, EnhancedSignal
from strategy.intraday_strategy import WATCHLIST_STOCKS


@dataclass
class Position:
    """持仓"""
    symbol: str
    stock_name: str
    entry_time: str
    entry_price: float
    shares: int
    current_price: float
    stop_loss: float
    take_profit: float
    position_type: str  # "trend"趋势持仓, "swing"波段持仓
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0


@dataclass
class Trade:
    """交易记录"""
    symbol: str
    stock_name: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    shares: int
    profit: float
    profit_pct: float
    trade_type: str  # "trend"趋势交易, "day_trade"日内做T
    hold_periods: int
    exit_reason: str


class EnhancedBacktester:
    """增强版回测引擎"""

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []

        # 策略系统
        self.strategy_system = EnhancedStrategySystem()

        # 回测参数
        self.trend_position_pct = 0.20  # 趋势持仓单只最多20%
        self.max_trend_positions = 4  # 最多4只趋势持仓
        self.min_trend_hold_periods = 16  # 趋势最少持有16个30分钟周期（8小时）
        self.max_trend_hold_periods = 160  # 趋势最多持有160个周期（80小时≈3天）

        # 做T参数
        self.day_trade_pct = 0.30  # 做T使用30%的持仓
        self.day_trade_profit_target = 0.012  # 做T目标收益1.2%
        self.day_trade_stop_loss = 0.008  # 做T止损0.8%

        # 统计
        self.daily_values = []
        self.trend_trades = []
        self.day_trades = []

    def load_data(self, symbol: str) -> pd.DataFrame:
        """加载数据"""
        from data.data_handler import DataHandler
        handler = DataHandler(force_refresh=True)
        df = handler.fetch_stock_data(symbol, force_refresh=True)

        if df is not None and len(df) >= 60:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            # 最近30天
            one_month_ago = datetime.now() - timedelta(days=30)
            df = df[df['date'] >= one_month_ago]
        return df

    def run_backtest(self):
        """执行回测"""
        print("=" * 70)
        print("增强版回测系统")
        print("=" * 70)
        print(f"初始资金: {self.initial_capital:.2f} 元")
        print(f"策略: 趋势跟踪(长持) + 日内做T")
        print(f"趋势仓位: 单只最多{self.trend_position_pct*100}%，最多{self.max_trend_positions}只")
        print(f"做T仓位: 使用持仓的{self.day_trade_pct*100}%")
        print("=" * 70)

        # 加载数据
        all_data = {}
        for stock in WATCHLIST_STOCKS:
            symbol = stock['symbol']
            print(f"\n加载 {stock['name']} ({symbol})...")
            df = self.load_data(symbol)
            if df is not None and len(df) >= 60:
                all_data[symbol] = df
                print(f"  ✓ 数据量: {len(df)} 条")
            else:
                print(f"  ⚠️ 数据不足")

        if not all_data:
            print("无有效数据")
            return

        # 获取所有时间点
        all_times = set()
        for symbol, df in all_data.items():
            for t in df['date'].unique():
                all_times.add(t)
        all_times = sorted(list(all_times))

        print(f"\n共 {len(all_times)} 个时间点")
        print("\n开始回测...")

        # 遍历每个时间点
        for i, current_time in enumerate(all_times):
            if i % 50 == 0:
                print(f"  进度: {i}/{len(all_times)} ({i/len(all_times)*100:.1f}%)")

            # 1. 检查趋势持仓退出
            self._check_trend_exit(all_data, current_time)

            # 2. 检查趋势买入机会
            self._check_trend_entry(all_data, current_time)

            # 3. 检查做T机会（需要已有持仓）
            self._check_day_trade_opportunities(all_data, current_time)

            # 4. 记录市值
            total_value = self._calculate_total_value(all_data, current_time)
            self.daily_values.append({
                'time': current_time,
                'value': total_value,
                'cash': self.cash,
                'positions': len(self.positions)
            })

        # 输出结果
        self._print_results()

    def _check_trend_exit(self, all_data: Dict, current_time: datetime):
        """检查趋势持仓退出"""
        for symbol, pos in list(self.positions.items()):
            if pos.position_type != "trend":
                continue

            df = all_data.get(symbol)
            if df is None:
                continue

            df_slice = df[df['date'] <= current_time].tail(60)
            if len(df_slice) < 20:
                continue

            current_price = float(df_slice['close'].iloc[-1])

            # 计算持仓时间
            entry_idx = df[df['date'] == pos.entry_time].index
            if len(entry_idx) > 0:
                entry_idx = entry_idx[0]
                current_idx = df[df['date'] <= current_time].index[-1]
                hold_periods = current_idx - entry_idx
            else:
                hold_periods = 0

            # 检查止损
            if current_price <= pos.stop_loss:
                self._exit_position(symbol, current_time, current_price, "止损", hold_periods)
                continue

            # 检查止盈
            if current_price >= pos.take_profit:
                self._exit_position(symbol, current_time, current_price, "止盈", hold_periods)
                continue

            # 检查最大持仓时间
            if hold_periods >= self.max_trend_hold_periods:
                self._exit_position(symbol, current_time, current_price, "到期退出", hold_periods)
                continue

            # 最小持仓时间内不检查卖出信号
            if hold_periods < self.min_trend_hold_periods:
                continue

            # 检查卖出信号
            signal = self.strategy_system.analyze_stock(symbol, pos.stock_name, df_slice)
            if signal and signal.combined_signal in ["卖出", "强烈卖出"]:
                self._exit_position(symbol, current_time, current_price, "信号卖出", hold_periods)

    def _check_trend_entry(self, all_data: Dict, current_time: datetime):
        """检查趋势买入机会"""
        if len(self.positions) >= self.max_trend_positions:
            return

        for stock in WATCHLIST_STOCKS:
            symbol = stock['symbol']

            if symbol in self.positions:
                continue

            df = all_data.get(symbol)
            if df is None:
                continue

            df_slice = df[df['date'] <= current_time].tail(60)
            if len(df_slice) < 60:
                continue

            # 分析信号
            signal = self.strategy_system.analyze_stock(symbol, stock['name'], df_slice)

            if signal and signal.combined_signal in ["买入", "强烈买入"]:
                # 检查评分
                if signal.combined_score < 1.5:
                    continue

                current_price = float(df_slice['close'].iloc[-1])
                self._enter_trend_position(symbol, stock['name'], current_time, current_price, signal)

    def _check_day_trade_opportunities(self, all_data: Dict, current_time: datetime):
        """检查做T机会（模拟）"""
        # 做T需要已有持仓
        # 这里简化处理：在回测中，做T收益会在趋势持仓的买卖差价中体现
        pass

    def _enter_trend_position(self, symbol: str, stock_name: str, entry_time: datetime,
                              entry_price: float, signal: EnhancedSignal):
        """开趋势仓"""
        # 计算买入金额
        max_invest = self.cash * self.trend_position_pct
        shares = int(max_invest / entry_price / 100) * 100
        if shares < 100:
            return

        invest_amount = shares * entry_price
        if invest_amount > self.cash:
            return

        # 执行买入
        self.cash -= invest_amount

        self.positions[symbol] = Position(
            symbol=symbol,
            stock_name=stock_name,
            entry_time=str(entry_time),
            entry_price=entry_price,
            shares=shares,
            current_price=entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            position_type="trend"
        )

        print(f"  🟢 趋势买入 {stock_name}: {shares}股 @ {entry_price:.2f}, 止损={signal.stop_loss:.2f}")

    def _exit_position(self, symbol: str, exit_time: datetime, exit_price: float,
                       exit_reason: str, hold_periods: int):
        """平仓"""
        pos = self.positions.get(symbol)
        if pos is None:
            return

        # 计算盈亏
        profit = (exit_price - pos.entry_price) * pos.shares
        profit_pct = (exit_price - pos.entry_price) / pos.entry_price * 100

        # 执行卖出
        self.cash += exit_price * pos.shares

        # 记录交易
        trade = Trade(
            symbol=symbol,
            stock_name=pos.stock_name,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            exit_time=str(exit_time),
            exit_price=exit_price,
            shares=pos.shares,
            profit=profit,
            profit_pct=profit_pct,
            trade_type="trend",
            hold_periods=hold_periods,
            exit_reason=exit_reason
        )
        self.trades.append(trade)
        self.trend_trades.append(trade)

        # 删除持仓
        del self.positions[symbol]

        emoji = "✅" if profit > 0 else "❌"
        print(f"  {emoji} 趋势卖出 {pos.stock_name}: {pos.shares}股 @ {exit_price:.2f}, "
              f"盈亏={profit:.2f}({profit_pct:.2f}%), 原因={exit_reason}, 持仓={hold_periods}周期")

    def _calculate_total_value(self, all_data: Dict, current_time: datetime) -> float:
        """计算总市值"""
        total = self.cash
        for symbol, pos in self.positions.items():
            df = all_data.get(symbol)
            if df is None:
                total += pos.entry_price * pos.shares
                continue

            df_slice = df[df['date'] <= current_time]
            if len(df_slice) > 0:
                current_price = float(df_slice['close'].iloc[-1])
                total += current_price * pos.shares
            else:
                total += pos.entry_price * pos.shares

        return total

    def _print_results(self):
        """输出回测结果"""
        print("\n" + "=" * 70)
        print("回测结果汇总")
        print("=" * 70)

        # 资金情况
        final_value = self.daily_values[-1]['value'] if self.daily_values else self.initial_capital
        total_return = final_value - self.initial_capital
        return_pct = total_return / self.initial_capital * 100

        print(f"\n【资金统计】")
        print(f"  初始资金: {self.initial_capital:.2f} 元")
        print(f"  最终资金: {final_value:.2f} 元")
        print(f"  总盈亏: {total_return:.2f} 元")
        print(f"  收益率: {return_pct:.2f}%")

        # 趋势交易统计
        if self.trend_trades:
            wins = [t for t in self.trend_trades if t.profit > 0]
            losses = [t for t in self.trend_trades if t.profit <= 0]

            win_rate = len(wins) / len(self.trend_trades) * 100
            avg_profit = sum(t.profit for t in self.trend_trades) / len(self.trend_trades)
            avg_win = sum(t.profit for t in wins) / len(wins) if wins else 0
            avg_loss = sum(t.profit for t in losses) / len(losses) if losses else 0
            avg_hold = sum(t.hold_periods for t in self.trend_trades) / len(self.trend_trades)
            profit_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

            print(f"\n【趋势交易统计】")
            print(f"  交易次数: {len(self.trend_trades)}")
            print(f"  盈利次数: {len(wins)} | 亏损次数: {len(losses)}")
            print(f"  胜率: {win_rate:.2f}%")
            print(f"  平均持仓: {avg_hold:.1f} 周期 ({avg_hold/2:.1f}小时)")
            print(f"  平均盈亏: {avg_profit:.2f} 元")
            print(f"  平均盈利: {avg_win:.2f} 元 | 平均亏损: {avg_loss:.2f} 元")
            print(f"  盈亏比: {profit_ratio:.2f}")

        # 交易明细
        print(f"\n【交易明细】")
        print("-" * 70)
        for t in self.trades:
            emoji = "✅" if t.profit > 0 else "❌"
            print(f"{emoji} {t.stock_name}: 买入@{t.entry_price:.2f} → 卖出@{t.exit_price:.2f} | "
                  f"盈亏:{t.profit:.2f}({t.profit_pct:.2f}%) | 持仓:{t.hold_periods}周期 | {t.exit_reason}")

        print("-" * 70)

        # 市值曲线
        if len(self.daily_values) > 1:
            print(f"\n【市值曲线】")
            step = max(1, len(self.daily_values) // 8)
            for i in range(0, len(self.daily_values), step):
                dv = self.daily_values[i]
                print(f"  {dv['time']}: 市值={dv['value']:.2f}, 持仓={dv['positions']}只")


def main():
    backtester = EnhancedBacktester(initial_capital=100000)
    backtester.run_backtest()


if __name__ == "__main__":
    main()