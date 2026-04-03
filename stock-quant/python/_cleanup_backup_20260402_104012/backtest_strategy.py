#!/usr/bin/env python3
"""
交易策略回测脚本
- 模拟10万资金操作
- 回测最近1个月的盈利情况
- 计算胜率、收益率等统计指标
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from dataclasses import dataclass
from strategy.intraday_strategy import IntradayStrategy, SignalType, WATCHLIST_STOCKS, TechnicalIndicators


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
    signal_type: str
    reason: str
    exit_reason: str


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


class Backtester:
    """回测引擎"""

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.strategy = IntradayStrategy(watchlist=WATCHLIST_STOCKS, notify_enabled=False, force_refresh=False)

        # 回测参数
        self.position_size_pct = 0.15  # 每只股票最多占用15%资金
        self.max_positions = 5  # 最大同时持有5只股票
        self.min_hold_periods = 2  # 最少持有2个30分钟周期
        self.max_hold_periods = 48  # 最多持有48个周期（24小时）
        self.min_buy_score = 2.0  # 买入信号最低评分（提高门槛减少低质量买入）

        # 统计
        self.daily_values = []
        self.win_count = 0
        self.loss_count = 0

    def load_historical_data(self, symbol: str) -> pd.DataFrame:
        """加载历史数据"""
        from data.data_handler import DataHandler
        handler = DataHandler(force_refresh=True)
        df = handler.fetch_stock_data(symbol, force_refresh=True)

        if df is None or df.empty:
            return None

        # 确保日期格式正确
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # 只取最近1个月的数据
        one_month_ago = datetime.now() - timedelta(days=30)
        df = df[df['date'] >= one_month_ago]

        return df

    def run_backtest(self):
        """执行回测"""
        print("=" * 60)
        print("交易策略回测")
        print("=" * 60)
        print(f"初始资金: {self.initial_capital:.2f} 元")
        print(f"回测周期: 最近 30 天")
        print(f"数据周期: 30 分钟级别")
        print(f"每只股票仓位: {self.position_size_pct*100}%")
        print("=" * 60)

        # 加载所有股票数据
        all_data = {}
        for stock in WATCHLIST_STOCKS:
            symbol = stock['symbol']
            print(f"\n加载 {stock['name']} ({symbol}) 数据...")
            df = self.load_historical_data(symbol)
            if df is not None and len(df) >= 60:
                all_data[symbol] = df
                print(f"  ✓ 数据量: {len(df)} 条, 时间范围: {df['date'].min()} ~ {df['date'].max()}")
            else:
                print(f"  ⚠️ 数据不足")

        if not all_data:
            print("无有效数据，回测终止")
            return

        # 获取所有时间点
        all_times = set()
        for symbol, df in all_data.items():
            for t in df['date'].unique():
                all_times.add(t)

        all_times = sorted(list(all_times))
        print(f"\n共 {len(all_times)} 个时间点")

        # 遍历每个时间点
        print("\n开始回测...")
        for i, current_time in enumerate(all_times):
            if i % 100 == 0:
                print(f"  进度: {i}/{len(all_times)} ({i/len(all_times)*100:.1f}%)")

            # 1. 检查现有持仓是否需要卖出
            self._check_exit_signals(all_data, current_time)

            # 2. 检查是否有新的买入机会
            self._check_entry_signals(all_data, current_time)

            # 3. 记录当日市值
            total_value = self._calculate_total_value(all_data, current_time)
            self.daily_values.append({
                'time': current_time,
                'value': total_value,
                'cash': self.cash,
                'positions': len(self.positions)
            })

        # 输出回测结果
        self._print_results()

    def _check_exit_signals(self, all_data: Dict, current_time: datetime):
        """检查卖出信号"""
        for symbol, pos in list(self.positions.items()):
            df = all_data.get(symbol)
            if df is None:
                continue

            # 获取当前时间的数据
            df_slice = df[df['date'] <= current_time].tail(60)
            if len(df_slice) < 60:
                continue

            current_price = float(df_slice['close'].iloc[-1])

            # 检查止损止盈
            if current_price <= pos.stop_loss:
                self._exit_position(symbol, current_time, current_price, "止损")
                continue
            elif current_price >= pos.take_profit:
                self._exit_position(symbol, current_time, current_price, "止盈")
                continue

            # 检查持仓时间
            entry_idx = df[df['date'] == pos.entry_time].index
            if len(entry_idx) > 0:
                entry_idx = entry_idx[0]
                current_idx = df[df['date'] <= current_time].index[-1]
                hold_periods = current_idx - entry_idx

                # 最长持有时间
                if hold_periods >= self.max_hold_periods:
                    self._exit_position(symbol, current_time, current_price, "超时卖出")
                    continue

            # 生成卖出信号
            signal = self.strategy.generate_signal(symbol, df_slice)
            if signal and signal['signal'] in ['卖出', '强烈卖出']:
                self._exit_position(symbol, current_time, current_price, "信号卖出")

    def _check_entry_signals(self, all_data: Dict, current_time: datetime):
        """检查买入信号"""
        if len(self.positions) >= self.max_positions:
            return

        for stock in WATCHLIST_STOCKS:
            symbol = stock['symbol']

            # 已持有则跳过
            if symbol in self.positions:
                continue

            df = all_data.get(symbol)
            if df is None:
                continue

            # 获取当前时间的数据
            df_slice = df[df['date'] <= current_time].tail(60)
            if len(df_slice) < 60:
                continue

            # 获取趋势信息（避免在下跌趋势中买入）
            close = df_slice['close'].values
            trend_20 = (close[-1] - close[-20]) / close[-20] * 100 if len(close) >= 20 else 0

            # 生成买入信号
            signal = self.strategy.generate_signal(symbol, df_slice)
            if signal and signal['signal'] in ['买入', '强烈买入']:
                # 检查评分是否达到最低门槛
                if signal['score'] < self.min_buy_score:
                    continue  # 评分太低，跳过买入

                # 在下跌趋势中，只接受"强烈买入"信号（评分>=4）
                if trend_20 < -3 and signal['signal'] == '买入':
                    continue  # 下跌趋势中跳过普通买入信号

                current_price = float(df_slice['close'].iloc[-1])
                self._enter_position(symbol, stock['name'], current_time, current_price, signal)

    def _enter_position(self, symbol: str, stock_name: str, entry_time: datetime,
                        entry_price: float, signal: Dict):
        """买入"""
        # 计算买入金额
        max_invest = self.cash * self.position_size_pct
        shares = int(max_invest / entry_price / 100) * 100  # 按手买入
        if shares < 100:
            return

        invest_amount = shares * entry_price
        if invest_amount > self.cash:
            return

        # 计算止损止盈
        atr = signal['indicators'].get('atr', entry_price * 0.02)
        stop_loss = signal.get('stop_loss', entry_price - atr * 2)
        take_profit = signal.get('take_profit', entry_price + atr * 3)

        # 执行买入
        self.cash -= invest_amount
        self.positions[symbol] = Position(
            symbol=symbol,
            stock_name=stock_name,
            entry_time=str(entry_time),
            entry_price=entry_price,
            shares=shares,
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        print(f"  🟢 买入 {stock_name} ({symbol}): {shares}股 @ {entry_price:.2f}, 止损={stop_loss:.2f}, 止盈={take_profit:.2f}")

    def _exit_position(self, symbol: str, exit_time: datetime, exit_price: float, exit_reason: str):
        """卖出"""
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
            signal_type="买入",
            reason="买入信号",
            exit_reason=exit_reason
        )
        self.trades.append(trade)

        # 统计胜负
        if profit > 0:
            self.win_count += 1
        else:
            self.loss_count += 1

        # 删除持仓
        del self.positions[symbol]

        emoji = "✅" if profit > 0 else "❌"
        print(f"  {emoji} 卖出 {pos.stock_name}: {pos.shares}股 @ {exit_price:.2f}, 盈亏={profit:.2f}({profit_pct:.2f}%), 原因={exit_reason}")

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
        print("\n" + "=" * 60)
        print("回测结果汇总")
        print("=" * 60)

        # 资金情况
        final_value = self.daily_values[-1]['value'] if self.daily_values else self.initial_capital
        total_return = final_value - self.initial_capital
        return_pct = total_return / self.initial_capital * 100

        print(f"\n【资金统计】")
        print(f"  初始资金: {self.initial_capital:.2f} 元")
        print(f"  最终资金: {final_value:.2f} 元")
        print(f"  总盈亏: {total_return:.2f} 元")
        print(f"  收益率: {return_pct:.2f}%")

        # 交易统计
        total_trades = len(self.trades)
        win_rate = self.win_count / total_trades * 100 if total_trades > 0 else 0

        print(f"\n【交易统计】")
        print(f"  总交易次数: {total_trades}")
        print(f"  盈利次数: {self.win_count}")
        print(f"  亏损次数: {self.loss_count}")
        print(f"  胜率: {win_rate:.2f}%")

        if total_trades > 0:
            avg_profit = sum(t.profit for t in self.trades) / total_trades
            avg_win = sum(t.profit for t in self.trades if t.profit > 0) / self.win_count if self.win_count > 0 else 0
            avg_loss = sum(t.profit for t in self.trades if t.profit <= 0) / self.loss_count if self.loss_count > 0 else 0

            print(f"  平均盈亏: {avg_profit:.2f} 元")
            print(f"  平均盈利: {avg_win:.2f} 元")
            print(f"  平均亏损: {avg_loss:.2f} 元")

            # 盈亏比
            profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            print(f"  盈亏比: {profit_loss_ratio:.2f}")

        # 交易详情
        print(f"\n【交易明细】")
        print("-" * 60)
        for t in self.trades:
            emoji = "✅" if t.profit > 0 else "❌"
            print(f"{emoji} {t.stock_name}: 买入@{t.entry_price:.2f} → 卖出@{t.exit_price:.2f} | "
                  f"盈亏:{t.profit:.2f}({t.profit_pct:.2f}%) | 原因:{t.exit_reason}")

        print("-" * 60)

        # 每日市值曲线
        if len(self.daily_values) > 1:
            print(f"\n【市值曲线】")
            # 取关键时间点
            step = max(1, len(self.daily_values) // 10)
            for i in range(0, len(self.daily_values), step):
                dv = self.daily_values[i]
                print(f"  {dv['time']}: 市值={dv['value']:.2f}, 持仓={dv['positions']}只")


def main():
    backtester = Backtester(initial_capital=100000)
    backtester.run_backtest()


if __name__ == "__main__":
    main()