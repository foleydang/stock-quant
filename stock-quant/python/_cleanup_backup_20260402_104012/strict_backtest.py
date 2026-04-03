#!/usr/bin/env python3
"""
严谨版回测系统
约束条件：
1. 卖出股数 <= 持有股数
2. 资金余额 >= 0
3. 操作以100股（1手）为单位
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from strategy.intraday_strategy import WATCHLIST_STOCKS, TechnicalIndicators
from data.data_handler import DataHandler


@dataclass
class Position:
    """持仓"""
    symbol: str
    stock_name: str
    shares: int  # 持有股数
    available_shares: int  # 可卖股数（T+1）
    cost_price: float  # 成本价
    current_price: float  # 当前价
    stop_loss: float
    take_profit: float
    entry_time: str
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0


@dataclass
class Trade:
    """交易记录"""
    symbol: str
    stock_name: str
    trade_type: str  # "buy", "sell", "t_buy", "t_sell"
    price: float
    shares: int
    amount: float
    time: str
    reason: str
    profit: float = 0.0  # 仅卖出时有


class StrictBacktester:
    """严谨版回测引擎"""

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_values = []

        # 参数
        self.position_pct = 0.15  # 单只最多15%资金
        self.max_positions = 5
        self.min_hold_periods = 4  # 最少持有4个周期（2小时）
        self.max_hold_periods = 48  # 最多持有48个周期（24小时）
        self.stop_loss_atr_mult = 3.0
        self.take_profit_atr_mult = 4.0

        # 统计
        self.total_buy_amount = 0.0
        self.total_sell_amount = 0.0

    def load_data(self, symbol: str) -> pd.DataFrame:
        """加载数据"""
        handler = DataHandler(force_refresh=True)
        df = handler.fetch_stock_data(symbol, force_refresh=True)
        if df is not None and len(df) >= 60:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            one_month_ago = datetime.now() - timedelta(days=30)
            df = df[df['date'] >= one_month_ago]
        return df

    def run_backtest(self):
        """执行回测"""
        print("=" * 70)
        print("严谨版回测系统")
        print("=" * 70)
        print(f"初始资金: {self.initial_capital:.2f} 元")
        print(f"约束条件:")
        print(f"  - 卖出股数 <= 持有股数")
        print(f"  - 资金余额 >= 0")
        print(f"  - 操作以100股（1手）为单位")
        print(f"  - 单只仓位 <= {self.position_pct*100}%")
        print(f"  - 最多持仓 {self.max_positions} 只")
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
        print("\n开始回测...\n")

        # 遍历每个时间点
        for i, current_time in enumerate(all_times):
            if i % 50 == 0:
                total_value = self._calculate_total_value(all_data, current_time)
                print(f"[{i}/{len(all_times)}] 市值: {total_value:.2f}, 现金: {self.cash:.2f}, 持仓: {len(self.positions)}只")

            # 1. 更新可卖股数（T+1）
            self._update_available_shares(current_time)

            # 2. 检查卖出信号
            self._check_sell_signals(all_data, current_time)

            # 3. 检查买入信号
            self._check_buy_signals(all_data, current_time)

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

    def _update_available_shares(self, current_time: datetime):
        """更新可卖股数（T+1规则）"""
        # 简化处理：买入后下一个周期可卖
        for symbol, pos in self.positions.items():
            # 这里简化，实际应该按日期判断
            if pos.available_shares < pos.shares:
                pos.available_shares = pos.shares

    def _check_sell_signals(self, all_data: Dict, current_time: datetime):
        """检查卖出信号"""
        for symbol, pos in list(self.positions.items()):
            df = all_data.get(symbol)
            if df is None:
                continue

            df_slice = df[df['date'] <= current_time].tail(60)
            if len(df_slice) < 20:
                continue

            current_price = float(df_slice['close'].iloc[-1])
            pos.current_price = current_price
            pos.unrealized_pnl = (current_price - pos.cost_price) * pos.shares
            pos.unrealized_pnl_pct = (current_price - pos.cost_price) / pos.cost_price * 100

            # 计算持仓周期
            entry_idx = df[df['date'].astype(str) == pos.entry_time].index
            if len(entry_idx) > 0:
                entry_idx = entry_idx[0]
                current_idx = df[df['date'] <= current_time].index[-1]
                hold_periods = current_idx - entry_idx
            else:
                hold_periods = 999

            sell_reason = None

            # 止损
            if current_price <= pos.stop_loss:
                sell_reason = "止损"
            # 止盈
            elif current_price >= pos.take_profit:
                sell_reason = "止盈"
            # 到期
            elif hold_periods >= self.max_hold_periods:
                sell_reason = "到期"
            # 信号卖出（最少持有时间后）
            elif hold_periods >= self.min_hold_periods:
                signal = self._generate_signal(symbol, pos.stock_name, df_slice)
                if signal in ["卖出", "强烈卖出"]:
                    sell_reason = "信号卖出"

            if sell_reason:
                self._sell_position(symbol, current_time, current_price, sell_reason)

    def _check_buy_signals(self, all_data: Dict, current_time: datetime):
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

            df_slice = df[df['date'] <= current_time].tail(60)
            if len(df_slice) < 60:
                continue

            # 生成买入信号
            signal, score = self._generate_buy_signal(symbol, stock['name'], df_slice)

            if signal in ["买入", "强烈买入"] and score >= 1.5:
                current_price = float(df_slice['close'].iloc[-1])
                self._buy_stock(symbol, stock['name'], current_time, current_price, df_slice)

    def _generate_signal(self, symbol: str, stock_name: str, df: pd.DataFrame) -> str:
        """生成信号"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        score = 0.0

        # RSI
        rsi = TechnicalIndicators.calculate_rsi(close, 14)[-1]
        if rsi > 70:
            score -= 2
        elif rsi > 60:
            score -= 1
        elif rsi < 30:
            score += 2
        elif rsi < 40:
            score += 1

        # MACD
        macd_data = TechnicalIndicators.calculate_macd(close)
        if len(macd_data['macd']) >= 2:
            macd = macd_data['macd'][-1]
            signal_line = macd_data['signal'][-1]
            if macd < signal_line and macd_data['histogram'][-1] < macd_data['histogram'][-2]:
                score -= 1.5
            elif macd > signal_line and macd_data['histogram'][-1] > macd_data['histogram'][-2]:
                score += 1.5

        # 均线
        ma5 = np.mean(close[-5:])
        ma10 = np.mean(close[-10:])
        ma20 = np.mean(close[-20:])
        if ma5 < ma10 < ma20:
            score -= 2
        elif ma5 > ma10 > ma20:
            score += 2

        # 趋势
        trend = (close[-1] - close[-20]) / close[-20] * 100
        if trend < -5 and rsi < 35:
            score += 2  # 超卖反弹

        if score >= 3:
            return "强烈买入"
        elif score >= 1:
            return "买入"
        elif score >= -1:
            return "持有"
        elif score >= -3:
            return "卖出"
        else:
            return "强烈卖出"

    def _generate_buy_signal(self, symbol: str, stock_name: str, df: pd.DataFrame) -> tuple:
        """生成买入信号，返回 (信号, 评分)"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        score = 0.0
        reasons = []

        # 趋势过滤
        trend_20 = (close[-1] - close[-20]) / close[-20] * 100
        if trend_20 < -8:
            return "持有", 0.0

        # RSI
        rsi = TechnicalIndicators.calculate_rsi(close, 14)[-1]
        if rsi < 25:
            score += 3
            reasons.append("RSI超卖")
        elif rsi < 35:
            score += 2
            reasons.append("RSI偏低")
        elif rsi > 60:
            score -= 1

        # MACD
        macd_data = TechnicalIndicators.calculate_macd(close)
        if len(macd_data['macd']) >= 2:
            if macd_data['macd'][-1] > macd_data['signal'][-1]:
                score += 1.5
                reasons.append("MACD金叉")

        # 均线
        ma5 = np.mean(close[-5:])
        ma10 = np.mean(close[-10:])
        ma20 = np.mean(close[-20:])
        if ma5 > ma10 > ma20:
            score += 2
            reasons.append("均线多头")
        elif ma5 > ma10:
            score += 1

        # KDJ
        kdj = TechnicalIndicators.calculate_kdj(high, low, close)
        k = kdj['k'][-1]
        j = kdj['j'][-1]
        if k < 20 or j < 10:
            score += 2
            reasons.append("KDJ超卖")

        if score >= 4:
            return "强烈买入", score
        elif score >= 2:
            return "买入", score
        else:
            return "持有", score

    def _buy_stock(self, symbol: str, stock_name: str, entry_time: datetime,
                   entry_price: float, df: pd.DataFrame):
        """买入股票（严谨版）"""
        # 计算可用资金
        available_cash = self.cash
        if available_cash <= 0:
            return

        # 计算最大买入金额
        max_invest = min(available_cash, self.initial_capital * self.position_pct)

        # 计算买入股数（必须是100的整数倍）
        shares = int(max_invest / entry_price / 100) * 100
        if shares < 100:
            return

        # 计算实际需要金额
        actual_amount = shares * entry_price

        # 检查资金是否足够
        if actual_amount > available_cash:
            # 减少股数
            shares = int(available_cash / entry_price / 100) * 100
            if shares < 100:
                return
            actual_amount = shares * entry_price

        # 再次检查资金
        if actual_amount > self.cash:
            return

        # 计算止损止盈
        atr = TechnicalIndicators.calculate_atr(df['high'].values, df['low'].values, df['close'].values, 14)
        atr_val = atr[-1] if len(atr) > 0 else entry_price * 0.02
        stop_loss = entry_price - atr_val * self.stop_loss_atr_mult
        take_profit = entry_price + atr_val * self.take_profit_atr_mult

        # 执行买入
        self.cash -= actual_amount
        self.total_buy_amount += actual_amount

        self.positions[symbol] = Position(
            symbol=symbol,
            stock_name=stock_name,
            shares=shares,
            available_shares=0,  # T+1，当日不可卖
            cost_price=entry_price,
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=str(entry_time)
        )

        # 记录交易
        self.trades.append(Trade(
            symbol=symbol,
            stock_name=stock_name,
            trade_type="buy",
            price=entry_price,
            shares=shares,
            amount=actual_amount,
            time=str(entry_time),
            reason="买入开仓"
        ))

        print(f"  🟢 买入 {stock_name}: {shares}股 @ {entry_price:.2f}, 金额={actual_amount:.2f}, 止损={stop_loss:.2f}")

    def _sell_position(self, symbol: str, sell_time: datetime, sell_price: float, reason: str):
        """卖出持仓（严谨版）"""
        pos = self.positions.get(symbol)
        if pos is None:
            return

        # 检查可卖股数
        available = pos.available_shares
        if available <= 0:
            return

        # 卖出数量（以100股为单位，但不能超过可卖数量）
        sell_shares = (available // 100) * 100
        if sell_shares <= 0:
            sell_shares = available  # 不足100股也全卖

        # 确保不超过持有数量
        sell_shares = min(sell_shares, pos.shares)
        if sell_shares <= 0:
            return

        # 计算卖出金额
        sell_amount = sell_shares * sell_price

        # 计算盈亏
        profit = (sell_price - pos.cost_price) * sell_shares
        profit_pct = (sell_price - pos.cost_price) / pos.cost_price * 100

        # 执行卖出
        self.cash += sell_amount
        self.total_sell_amount += sell_amount

        # 更新持仓
        pos.shares -= sell_shares
        pos.available_shares -= sell_shares

        # 记录交易
        self.trades.append(Trade(
            symbol=symbol,
            stock_name=pos.stock_name,
            trade_type="sell",
            price=sell_price,
            shares=sell_shares,
            amount=sell_amount,
            time=str(sell_time),
            reason=reason,
            profit=profit
        ))

        emoji = "✅" if profit > 0 else "❌"
        print(f"  {emoji} 卖出 {pos.stock_name}: {sell_shares}股 @ {sell_price:.2f}, 盈亏={profit:.2f}({profit_pct:.2f}%), 原因={reason}")

        # 如果清仓，删除持仓
        if pos.shares <= 0:
            del self.positions[symbol]

    def _calculate_total_value(self, all_data: Dict, current_time: datetime) -> float:
        """计算总市值"""
        total = self.cash
        for symbol, pos in self.positions.items():
            df = all_data.get(symbol)
            if df is None:
                total += pos.cost_price * pos.shares
                continue

            df_slice = df[df['date'] <= current_time]
            if len(df_slice) > 0:
                current_price = float(df_slice['close'].iloc[-1])
                total += current_price * pos.shares
            else:
                total += pos.cost_price * pos.shares

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
        print(f"  现金余额: {self.cash:.2f} 元")
        print(f"  总盈亏: {total_return:.2f} 元")
        print(f"  收益率: {return_pct:.2f}%")
        print(f"  总买入金额: {self.total_buy_amount:.2f} 元")
        print(f"  总卖出金额: {self.total_sell_amount:.2f} 元")

        # 交易统计
        buy_trades = [t for t in self.trades if t.trade_type == "buy"]
        sell_trades = [t for t in self.trades if t.trade_type == "sell"]

        wins = [t for t in sell_trades if t.profit > 0]
        losses = [t for t in sell_trades if t.profit <= 0]

        if sell_trades:
            win_rate = len(wins) / len(sell_trades) * 100
            total_profit = sum(t.profit for t in sell_trades)
            avg_profit = total_profit / len(sell_trades)
            avg_win = sum(t.profit for t in wins) / len(wins) if wins else 0
            avg_loss = sum(t.profit for t in losses) / len(losses) if losses else 0
            profit_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

            print(f"\n【交易统计】")
            print(f"  买入次数: {len(buy_trades)}")
            print(f"  卖出次数: {len(sell_trades)}")
            print(f"  盈利次数: {len(wins)} | 亏损次数: {len(losses)}")
            print(f"  胜率: {win_rate:.2f}%")
            print(f"  总盈亏: {total_profit:.2f} 元")
            print(f"  平均盈利: {avg_win:.2f} 元 | 平均亏损: {avg_loss:.2f} 元")
            print(f"  盈亏比: {profit_ratio:.2f}")

        # 交易明细
        print(f"\n【交易明细】")
        print("-" * 70)
        for t in self.trades:
            if t.trade_type == "buy":
                print(f"🟢 买入 {t.stock_name}: {t.shares}股 @ {t.price:.2f}, 金额={t.amount:.2f}")
            else:
                emoji = "✅" if t.profit > 0 else "❌"
                print(f"{emoji} 卖出 {t.stock_name}: {t.shares}股 @ {t.price:.2f}, 盈亏={t.profit:.2f}, 原因={t.reason}")

        print("-" * 70)

        # 当前持仓
        if self.positions:
            print(f"\n【当前持仓】")
            for symbol, pos in self.positions.items():
                pnl = (pos.current_price - pos.cost_price) * pos.shares
                pnl_pct = (pos.current_price - pos.cost_price) / pos.cost_price * 100
                print(f"  {pos.stock_name}: {pos.shares}股 @ {pos.cost_price:.2f}, 当前价={pos.current_price:.2f}, 浮盈={pnl:.2f}({pnl_pct:.2f}%)")


def main():
    backtester = StrictBacktester(initial_capital=100000)
    backtester.run_backtest()


if __name__ == "__main__":
    main()