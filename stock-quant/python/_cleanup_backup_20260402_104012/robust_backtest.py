#!/usr/bin/env python3
"""
稳健版回测系统
约束条件：
1. T+1规则：买入后至少持有1天（16个30分钟周期）
2. 不反复抄底：下跌趋势中需明确反转信号
3. 提高买入门槛：评分>=3分才买入
4. 趋势反转确认：RSI底背离+MACD金叉+价格站上均线
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass

from strategy.intraday_strategy import WATCHLIST_STOCKS, TechnicalIndicators
from data.data_handler import DataHandler


@dataclass
class Position:
    """持仓"""
    symbol: str
    stock_name: str
    shares: int
    cost_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    entry_time: str
    entry_idx: int  # 买入时的索引，用于计算持仓周期
    available: bool = False  # 是否可卖（T+1）


@dataclass
class Trade:
    """交易记录"""
    symbol: str
    stock_name: str
    trade_type: str
    price: float
    shares: int
    amount: float
    time: str
    reason: str
    profit: float = 0.0
    hold_periods: int = 0


class RobustBacktester:
    """稳健版回测引擎"""

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_values = []

        # 参数
        self.position_pct = 0.15
        self.max_positions = 5
        self.min_hold_periods = 32  # 最少持有32个周期（2天）
        self.max_hold_periods = 96  # 最多持有96个周期（6天）
        self.stop_loss_atr_mult = 3.5  # 放宽止损
        self.take_profit_atr_mult = 5.0  # 提高止盈

        # 买入门槛
        self.min_buy_score = 4.0  # 提高买入门槛到4分

    def load_data(self, symbol: str) -> pd.DataFrame:
        """加载数据（优先使用缓存）"""
        # 首先检查缓存文件
        cache_path = os.path.join(os.path.dirname(__file__), f'data/{symbol}_30m.csv')
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            # 只取最近30天数据
            one_month_ago = datetime.now() - timedelta(days=30)
            df = df[df['date'] >= one_month_ago].reset_index(drop=True)
            return df

        # 缓存不存在时才尝试获取新数据
        handler = DataHandler(force_refresh=False)
        df = handler.fetch_stock_data(symbol, force_refresh=False)
        if df is not None and len(df) >= 60:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            one_month_ago = datetime.now() - timedelta(days=30)
            df = df[df['date'] >= one_month_ago].reset_index(drop=True)
        return df

    def run_backtest(self):
        """执行回测"""
        print("=" * 70)
        print("稳健版回测系统（T+1规则）")
        print("=" * 70)
        print(f"初始资金: {self.initial_capital:.2f} 元")
        print(f"约束条件:")
        print(f"  - T+1规则：买入后至少持有1天")
        print(f"  - 不反复抄底：下跌趋势需反转确认")
        print(f"  - 买入门槛：评分 >= {self.min_buy_score}分")
        print(f"  - 操作以100股为单位")
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

        # 获取所有时间点（按日期分组）
        all_dates = set()
        for symbol, df in all_data.items():
            for t in df['date'].unique():
                all_dates.add(t)
        all_times = sorted(list(all_dates))

        print(f"\n共 {len(all_times)} 个时间点")
        print("\n开始回测...\n")

        # 遍历每个时间点
        for i, current_time in enumerate(all_times):
            if i % 30 == 0:
                total_value = self._calculate_total_value(all_data, i)
                print(f"[{i}/{len(all_times)}] {current_time.strftime('%m-%d %H:%M')} | "
                      f"市值:{total_value:.0f} | 现金:{self.cash:.0f} | 持仓:{len(self.positions)}只")

            # 1. 更新T+1可用状态（买入后下一交易日可卖）
            self._update_t1_availability(all_data, i, current_time)

            # 2. 检查卖出信号（满足T+1条件）
            self._check_sell_signals(all_data, i, current_time)

            # 3. 检查买入信号（严格条件）
            self._check_buy_signals(all_data, i, current_time)

            # 4. 记录市值
            total_value = self._calculate_total_value(all_data, i)
            self.daily_values.append({
                'time': current_time,
                'value': total_value,
                'cash': self.cash,
                'positions': len(self.positions)
            })

        # 输出结果
        self._print_results()

    def _update_t1_availability(self, all_data: Dict, current_idx: int, current_time: datetime):
        """更新T+1可用状态"""
        for symbol, pos in self.positions.items():
            if pos.available:
                continue

            # 计算持有周期数
            periods_held = current_idx - pos.entry_idx

            # 持有超过16个周期（1天）后可卖
            if periods_held >= self.min_hold_periods:
                pos.available = True

    def _check_sell_signals(self, all_data: Dict, current_idx: int, current_time: datetime):
        """检查卖出信号"""
        for symbol, pos in list(self.positions.items()):
            # T+1检查
            if not pos.available:
                continue

            df = all_data.get(symbol)
            if df is None:
                continue

            # 获取当前索引的数据切片
            df_slice = df.iloc[:current_idx+1].tail(60)
            if len(df_slice) < 20:
                continue

            current_price = float(df_slice['close'].iloc[-1])
            pos.current_price = current_price

            # 计算持有周期
            hold_periods = current_idx - pos.entry_idx

            sell_reason = None

            # 计算亏损比例
            loss_pct = (current_price - pos.cost_price) / pos.cost_price

            # 止损（亏损5%以上，给予更多空间）
            if loss_pct <= -0.05:
                sell_reason = "止损"

            # 止盈
            elif current_price >= pos.take_profit:
                sell_reason = "止盈"

            # 最大持仓时间
            elif hold_periods >= self.max_hold_periods:
                sell_reason = "到期"

            # 信号卖出（持有至少1天后）
            elif hold_periods >= self.min_hold_periods:
                signal = self._generate_sell_signal(df_slice)
                if signal:
                    sell_reason = signal

            if sell_reason:
                self._sell_position(symbol, current_time, current_price, sell_reason, hold_periods)

    def _generate_sell_signal(self, df: pd.DataFrame) -> str:
        """生成卖出信号"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        sell_score = 0

        # RSI超买
        rsi = TechnicalIndicators.calculate_rsi(close, 14)[-1]
        if rsi > 75:
            sell_score += 2
        elif rsi > 65:
            sell_score += 1

        # MACD死叉
        macd_data = TechnicalIndicators.calculate_macd(close)
        if len(macd_data['macd']) >= 2:
            if macd_data['macd'][-1] < macd_data['signal'][-1]:
                if macd_data['histogram'][-1] < macd_data['histogram'][-2]:
                    sell_score += 2

        # 均线转空
        ma5 = np.mean(close[-5:])
        ma10 = np.mean(close[-10:])
        ma20 = np.mean(close[-20:])
        if ma5 < ma10 and ma10 < ma20:
            sell_score += 2

        # 跌破重要支撑
        if close[-1] < ma20:
            sell_score += 1

        return "信号卖出" if sell_score >= 3 else None

    def _check_buy_signals(self, all_data: Dict, current_idx: int, current_time: datetime):
        """检查买入信号（严格条件）"""
        if len(self.positions) >= self.max_positions:
            return

        for stock in WATCHLIST_STOCKS:
            symbol = stock['symbol']

            if symbol in self.positions:
                continue

            df = all_data.get(symbol)
            if df is None:
                continue

            df_slice = df.iloc[:current_idx+1].tail(60)
            if len(df_slice) < 60:
                continue

            # 生成买入信号（严格）
            signal, score, reasons = self._generate_buy_signal_strict(df_slice)

            if signal and score >= self.min_buy_score:
                current_price = float(df_slice['close'].iloc[-1])
                self._buy_stock(symbol, stock['name'], current_time, current_price,
                               df_slice, current_idx, reasons)

    def _generate_buy_signal_strict(self, df: pd.DataFrame) -> Tuple[str, float, List[str]]:
        """
        生成严格买入信号
        要求：明确趋势反转，且不在强下跌趋势中买入
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        reasons = []
        score = 0.0

        # === 1. 趋势过滤（最重要）===
        trend_20 = (close[-1] - close[-20]) / close[-20] * 100
        trend_60 = (close[-1] - close[-40]) / close[-40] * 100 if len(close) >= 40 else 0

        # 均线方向
        ma5 = np.mean(close[-5:])
        ma10 = np.mean(close[-10:])
        ma20 = np.mean(close[-20:])
        ma5_prev = np.mean(close[-10:-5])
        ma10_prev = np.mean(close[-20:-10])

        ma_trend_up = ma5 > ma10 and ma10 > ma20
        ma_trend_down = ma5 < ma10 and ma10 < ma20

        # 强下跌趋势中完全禁止买入
        if trend_20 < -10 or trend_60 < -15:
            return None, 0, ["强下跌趋势，禁止买入"]

        # 均线空头排列时，必须有强烈反转信号
        if ma_trend_down and trend_20 < -5:
            return None, 0, ["均线空头排列，等待趋势反转"]

        is_downtrend = trend_20 < -3 or trend_60 < -5 or ma_trend_down

        # === 2. RSI超卖 ===
        rsi = TechnicalIndicators.calculate_rsi(close, 14)[-1]
        rsi_prev = TechnicalIndicators.calculate_rsi(close[:-1], 14)[-1] if len(close) > 15 else rsi

        if rsi < 20:
            score += 3
            reasons.append(f"RSI极度超卖({rsi:.1f})")
        elif rsi < 30:
            score += 2
            reasons.append(f"RSI超卖({rsi:.1f})")
        elif rsi < 40:
            score += 1
            reasons.append(f"RSI偏低({rsi:.1f})")

        # RSI底背离（价格新低但RSI不新低）
        if len(close) >= 20:
            price_low1 = min(close[-20:-10])
            price_low2 = min(close[-10:])
            rsi_arr = TechnicalIndicators.calculate_rsi(close, 14)
            if len(rsi_arr) >= 20:
                rsi_low1 = min(rsi_arr[-20:-10]) if len(rsi_arr) >= 20 else 50
                rsi_low2 = min(rsi_arr[-10:]) if len(rsi_arr) >= 10 else 50
                if price_low2 < price_low1 and rsi_low2 > rsi_low1:
                    score += 3
                    reasons.append("RSI底背离")

        # === 3. MACD金叉 ===
        macd_data = TechnicalIndicators.calculate_macd(close)
        if len(macd_data['macd']) >= 3:
            macd = macd_data['macd']
            signal_line = macd_data['signal']
            histogram = macd_data['histogram']

            # 金叉
            if macd[-1] > signal_line[-1] and macd[-2] <= signal_line[-2]:
                score += 2
                reasons.append("MACD金叉")
            # MACD底背离
            elif macd[-1] > 0 and macd[-3] < 0:
                score += 1
                reasons.append("MACD转正")

        # === 4. 均线系统 ===
        ma5 = np.mean(close[-5:])
        ma10 = np.mean(close[-10:])
        ma20 = np.mean(close[-20:])

        # 价格站上均线（反转确认）
        if close[-1] > ma5 and close[-2] <= ma5:
            score += 1.5
            reasons.append("价格站上MA5")
        if close[-1] > ma10:
            score += 1
            reasons.append("价格站上MA10")

        # 均线金叉
        if ma5 > ma10 and np.mean(close[-6:-1]) <= np.mean(close[-11:-1]):
            score += 2
            reasons.append("MA5上穿MA10")

        # === 5. KDJ ===
        kdj = TechnicalIndicators.calculate_kdj(high, low, close)
        k = kdj['k'][-1]
        d = kdj['d'][-1]
        j = kdj['j'][-1]

        if j < 10:
            score += 2
            reasons.append(f"KDJ超卖(J={j:.1f})")
        if k > d and len(kdj['k']) >= 2 and kdj['k'][-2] <= kdj['d'][-2]:
            score += 1.5
            reasons.append("KDJ金叉")

        # === 6. 成交量确认 ===
        vol_5 = np.mean(volume[-5:])
        vol_20 = np.mean(volume[-20:])
        if vol_5 > vol_20 * 1.3:
            score += 1
            reasons.append("放量确认")

        # === 7. 下跌趋势中的额外要求 ===
        if is_downtrend:
            # 下跌趋势中需要更强的反转信号
            reversal_signals = 0

            # RSI底背离
            if "RSI底背离" in reasons:
                reversal_signals += 2  # 权重更高

            # MACD金叉
            if "MACD金叉" in reasons:
                reversal_signals += 1

            # 均线金叉
            if "MA5上穿MA10" in reasons:
                reversal_signals += 1

            # 价格站上均线
            if "价格站上MA10" in reasons:
                reversal_signals += 1

            # 下跌趋势中至少需要3个反转信号，且评分要更高
            if reversal_signals < 3:
                return None, 0, [f"下跌趋势({trend_20:.1f}%)需更多反转确认"]

            # 下跌趋势中买入门槛提高到5分
            if score < 5:
                return None, score, [f"下跌趋势需评分>=5分，当前{score:.1f}分"]

        # === 确定信号 ===
        # 非下跌趋势：评分>=4分买入
        # 下跌趋势：前面已验证，这里直接返回
        if score >= 5:
            return "强烈买入", score, reasons
        elif score >= self.min_buy_score and not is_downtrend:
            return "买入", score, reasons
        else:
            return None, score, reasons

    def _buy_stock(self, symbol: str, stock_name: str, entry_time: datetime,
                   entry_price: float, df: pd.DataFrame, entry_idx: int, reasons: List[str]):
        """买入股票"""
        # 计算买入金额
        available_cash = self.cash
        if available_cash <= 0:
            return

        max_invest = min(available_cash * 0.9, self.initial_capital * self.position_pct)
        shares = int(max_invest / entry_price / 100) * 100
        if shares < 100:
            return

        actual_amount = shares * entry_price
        if actual_amount > self.cash:
            shares = int(self.cash / entry_price / 100) * 100
            if shares < 100:
                return
            actual_amount = shares * entry_price

        if actual_amount > self.cash:
            return

        # 止损止盈
        atr = TechnicalIndicators.calculate_atr(df['high'].values, df['low'].values, df['close'].values, 14)
        atr_val = atr[-1] if len(atr) > 0 else entry_price * 0.025
        stop_loss = entry_price - atr_val * self.stop_loss_atr_mult
        take_profit = entry_price + atr_val * self.take_profit_atr_mult

        # 执行买入
        self.cash -= actual_amount

        self.positions[symbol] = Position(
            symbol=symbol,
            stock_name=stock_name,
            shares=shares,
            cost_price=entry_price,
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=str(entry_time),
            entry_idx=entry_idx,
            available=False  # T+1，当日不可卖
        )

        self.trades.append(Trade(
            symbol=symbol,
            stock_name=stock_name,
            trade_type="buy",
            price=entry_price,
            shares=shares,
            amount=actual_amount,
            time=str(entry_time),
            reason=", ".join(reasons[:3])
        ))

        print(f"  🟢 买入 {stock_name}: {shares}股 @ {entry_price:.2f} | 止损:{stop_loss:.2f} | 止盈:{take_profit:.2f}")
        print(f"      原因: {', '.join(reasons[:3])}")

    def _sell_position(self, symbol: str, sell_time: datetime, sell_price: float,
                       reason: str, hold_periods: int):
        """卖出持仓"""
        pos = self.positions.get(symbol)
        if pos is None or not pos.available:
            return

        sell_shares = pos.shares
        sell_amount = sell_shares * sell_price
        profit = (sell_price - pos.cost_price) * sell_shares
        profit_pct = (sell_price - pos.cost_price) / pos.cost_price * 100

        self.cash += sell_amount

        self.trades.append(Trade(
            symbol=symbol,
            stock_name=pos.stock_name,
            trade_type="sell",
            price=sell_price,
            shares=sell_shares,
            amount=sell_amount,
            time=str(sell_time),
            reason=reason,
            profit=profit,
            hold_periods=hold_periods
        ))

        del self.positions[symbol]

        hold_hours = hold_periods * 0.5
        emoji = "✅" if profit > 0 else "❌"
        print(f"  {emoji} 卖出 {pos.stock_name}: {sell_shares}股 @ {sell_price:.2f} | "
              f"盈亏:{profit:.0f}({profit_pct:.1f}%) | 持有:{hold_hours:.1f}小时 | {reason}")

    def _calculate_total_value(self, all_data: Dict, current_idx: int) -> float:
        """计算总市值"""
        total = self.cash
        for symbol, pos in self.positions.items():
            df = all_data.get(symbol)
            if df is not None and current_idx < len(df):
                current_price = float(df['close'].iloc[current_idx])
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
        print(f"  最终市值: {final_value:.2f} 元")
        print(f"  现金余额: {self.cash:.2f} 元")
        print(f"  总盈亏: {total_return:.2f} 元")
        print(f"  收益率: {return_pct:.2f}%")

        # 交易统计
        buy_trades = [t for t in self.trades if t.trade_type == "buy"]
        sell_trades = [t for t in self.trades if t.trade_type == "sell"]

        wins = [t for t in sell_trades if t.profit > 0]
        losses = [t for t in sell_trades if t.profit <= 0]

        if sell_trades:
            win_rate = len(wins) / len(sell_trades) * 100
            total_profit = sum(t.profit for t in sell_trades)
            avg_win = sum(t.profit for t in wins) / len(wins) if wins else 0
            avg_loss = sum(t.profit for t in losses) / len(losses) if losses else 0
            profit_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            avg_hold = sum(t.hold_periods for t in sell_trades) / len(sell_trades) * 0.5

            print(f"\n【交易统计】")
            print(f"  买入次数: {len(buy_trades)}")
            print(f"  卖出次数: {len(sell_trades)}")
            print(f"  盈利次数: {len(wins)} | 亏损次数: {len(losses)}")
            print(f"  胜率: {win_rate:.1f}%")
            print(f"  总盈亏: {total_profit:.2f} 元")
            print(f"  平均盈利: {avg_win:.2f} 元 | 平均亏损: {avg_loss:.2f} 元")
            print(f"  盈亏比: {profit_ratio:.2f}")
            print(f"  平均持仓: {avg_hold:.1f} 小时")

        # 交易明细
        print(f"\n【交易明细】")
        print("-" * 70)
        for t in self.trades:
            if t.trade_type == "buy":
                print(f"🟢 买入 {t.stock_name}: {t.shares}股 @ {t.price:.2f}")
                print(f"   原因: {t.reason}")
            else:
                emoji = "✅" if t.profit > 0 else "❌"
                print(f"{emoji} 卖出 {t.stock_name}: {t.shares}股 @ {t.price:.2f} | "
                      f"盈亏:{t.profit:.0f} | 持有:{t.hold_periods*0.5:.1f}h | {t.reason}")

        print("-" * 70)

        # 当前持仓
        if self.positions:
            print(f"\n【当前持仓】")
            for symbol, pos in self.positions.items():
                pnl = (pos.current_price - pos.cost_price) * pos.shares
                pnl_pct = (pos.current_price - pos.cost_price) / pos.cost_price * 100
                status = "可卖" if pos.available else "T+1锁定"
                print(f"  {pos.stock_name}: {pos.shares}股 @ {pos.cost_price:.2f} | "
                      f"浮盈:{pnl:.0f}({pnl_pct:.1f}%) | {status}")


def main():
    backtester = RobustBacktester(initial_capital=100000)
    backtester.run_backtest()


if __name__ == "__main__":
    main()