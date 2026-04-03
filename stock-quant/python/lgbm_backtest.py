#!/usr/bin/env python3
"""
基于 LGBM 模型的回测系统
使用训练好的 LightGBM 模型进行买卖决策
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# 导入增强版特征工程
from strategy.train_lgb_enhanced import EnhancedFeatureEngineer


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
    entry_idx: int
    available: bool = False


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


class LGBMBacktester:
    """基于 LGBM 模型的回测引擎"""

    def __init__(self, initial_capital: float = 100000, model_path: str = None):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_values = []

        # 加载模型
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'models/lgb_hs300/model.pkl')

        self.model_data = self._load_model(model_path)
        self.model = self.model_data.get('model') if self.model_data else None
        self.feature_names = self.model_data.get('feature_names', []) if self.model_data else []

        # 参数
        self.position_pct = 0.15
        self.max_positions = 5
        self.min_hold_periods = 16  # T+1
        self.max_hold_periods = 64  # 最多持有2天
        self.stop_loss_pct = 0.04  # 4% 止损
        self.take_profit_pct = 0.06  # 6% 止盈

        # 模型预测阈值
        self.buy_threshold = 0.52  # 预测上涨概率 > 52% 才买入

    def _load_model(self, model_path: str) -> Optional[Dict]:
        """加载模型"""
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return None

        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"加载模型失败: {e}")
            return None

    def _get_model_prediction(self, df: pd.DataFrame) -> Tuple[float, str]:
        """获取模型预测"""
        if self.model is None:
            return 0.5, "模型未加载"

        try:
            # 使用增强版特征工程
            features = EnhancedFeatureEngineer.calculate_features(df)
            last_row = features.iloc[-1]

            # 检查是否有 NaN
            if last_row.isna().any():
                return 0.5, "特征含NaN"

            # 预测概率
            prob = self.model.predict_proba([last_row.values])[0]
            up_prob = prob[1] if len(prob) > 1 else prob[0]

            return up_prob, f"上涨概率:{up_prob:.1%}"

        except Exception as e:
            return 0.5, f"预测错误:{e}"

    def load_data(self, symbol: str) -> pd.DataFrame:
        """加载数据"""
        cache_path = os.path.join(os.path.dirname(__file__), f'data/{symbol}_30m.csv')
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            # 不过滤数据，使用全部历史数据计算特征
            return df
        return None

    def run_backtest(self, stocks: List[Dict]):
        """执行回测"""
        print("=" * 70)
        print("LGBM 模型回测系统")
        print("=" * 70)
        print(f"初始资金: {self.initial_capital:.2f} 元")
        print(f"模型准确率: {self.model_data.get('cv_accuracy', 0):.2%}" if self.model_data else "模型未加载")
        print(f"买入阈值: 上涨概率 > {self.buy_threshold:.0%}")
        print("=" * 70)

        # 加载数据
        all_data = {}
        for stock in stocks:
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
        all_dates = set()
        for symbol, df in all_data.items():
            for t in df['date'].unique():
                all_dates.add(t)
        all_times = sorted(list(all_dates))

        print(f"\n共 {len(all_times)} 个时间点")
        print("\n开始回测...\n")

        # 遍历时间点
        for i, current_time in enumerate(all_times):
            if i % 50 == 0:
                total_value = self._calculate_total_value(all_data, i)
                print(f"[{i}/{len(all_times)}] {current_time.strftime('%m-%d %H:%M')} | "
                      f"市值:{total_value:.0f} | 现金:{self.cash:.0f} | 持仓:{len(self.positions)}只")

            # 更新T+1
            self._update_availability(i)

            # 检查卖出
            self._check_sell(all_data, i, current_time)

            # 检查买入（使用模型预测）
            self._check_buy_ml(all_data, i, current_time, stocks)

            # 记录市值
            total_value = self._calculate_total_value(all_data, i)
            self.daily_values.append({
                'time': current_time,
                'value': total_value,
                'cash': self.cash,
                'positions': len(self.positions)
            })

        self._print_results()

    def _update_availability(self, current_idx: int):
        """更新T+1可用状态"""
        for symbol, pos in self.positions.items():
            if not pos.available:
                periods_held = current_idx - pos.entry_idx
                if periods_held >= self.min_hold_periods:
                    pos.available = True

    def _check_sell(self, all_data: Dict, current_idx: int, current_time: datetime):
        """检查卖出"""
        for symbol, pos in list(self.positions.items()):
            if not pos.available:
                continue

            df = all_data.get(symbol)
            if df is None or current_idx >= len(df):
                continue

            current_price = float(df['close'].iloc[current_idx])
            pos.current_price = current_price

            hold_periods = current_idx - pos.entry_idx
            sell_reason = None

            loss_pct = (current_price - pos.cost_price) / pos.cost_price

            # 止损
            if loss_pct <= -self.stop_loss_pct:
                sell_reason = "止损"
            # 止盈
            elif current_price >= pos.take_profit:
                sell_reason = "止盈"
            # 到期
            elif hold_periods >= self.max_hold_periods:
                sell_reason = "到期"
            # 模型预测下跌
            elif hold_periods >= self.min_hold_periods:
                df_slice = df.iloc[:current_idx+1].tail(60)
                if len(df_slice) >= 60:
                    up_prob, _ = self._get_model_prediction(df_slice)
                    if up_prob < 0.45:  # 预测下跌概率 > 55%
                        sell_reason = f"模型看跌({up_prob:.0%})"

            if sell_reason:
                self._sell_position(symbol, current_time, current_price, sell_reason, hold_periods)

    def _check_buy_ml(self, all_data: Dict, current_idx: int, current_time: datetime, stocks: List[Dict]):
        """检查买入（使用模型预测）"""
        if len(self.positions) >= self.max_positions:
            return

        for stock in stocks:
            symbol = stock['symbol']
            if symbol in self.positions:
                continue

            df = all_data.get(symbol)
            if df is None or current_idx < 150:  # 需要足够历史数据计算特征
                continue

            # 使用到当前时间点的所有数据计算特征
            df_slice = df.iloc[:current_idx+1]
            if len(df_slice) < 150:
                continue

            # 模型预测
            up_prob, reason = self._get_model_prediction(df_slice)

            if up_prob > self.buy_threshold:
                current_price = float(df_slice['close'].iloc[-1])
                self._buy_stock(symbol, stock['name'], current_time, current_price, current_idx, reason)

    def _buy_stock(self, symbol: str, stock_name: str, entry_time: datetime,
                   entry_price: float, entry_idx: int, reason: str):
        """买入"""
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

        stop_loss = entry_price * (1 - self.stop_loss_pct)
        take_profit = entry_price * (1 + self.take_profit_pct)

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
            available=False
        )

        self.trades.append(Trade(
            symbol=symbol,
            stock_name=stock_name,
            trade_type="buy",
            price=entry_price,
            shares=shares,
            amount=actual_amount,
            time=str(entry_time),
            reason=reason
        ))

        print(f"  🟢 买入 {stock_name}: {shares}股 @ {entry_price:.2f} | {reason}")

    def _sell_position(self, symbol: str, sell_time: datetime, sell_price: float,
                       reason: str, hold_periods: int):
        """卖出"""
        pos = self.positions.get(symbol)
        if pos is None or not pos.available:
            return

        sell_amount = pos.shares * sell_price
        profit = (sell_price - pos.cost_price) * pos.shares

        self.cash += sell_amount

        self.trades.append(Trade(
            symbol=symbol,
            stock_name=pos.stock_name,
            trade_type="sell",
            price=sell_price,
            shares=pos.shares,
            amount=sell_amount,
            time=str(sell_time),
            reason=reason,
            profit=profit,
            hold_periods=hold_periods
        ))

        del self.positions[symbol]

        emoji = "✅" if profit > 0 else "❌"
        print(f"  {emoji} 卖出 {pos.stock_name}: {pos.shares}股 @ {sell_price:.2f} | "
              f"盈亏:{profit:.0f} | {reason}")

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
        """输出结果"""
        print("\n" + "=" * 70)
        print("回测结果汇总")
        print("=" * 70)

        final_value = self.daily_values[-1]['value'] if self.daily_values else self.initial_capital
        total_return = final_value - self.initial_capital
        return_pct = total_return / self.initial_capital * 100

        print(f"\n【资金统计】")
        print(f"  初始资金: {self.initial_capital:.2f} 元")
        print(f"  最终市值: {final_value:.2f} 元")
        print(f"  总盈亏: {total_return:.2f} 元")
        print(f"  收益率: {return_pct:.2f}%")

        buy_trades = [t for t in self.trades if t.trade_type == "buy"]
        sell_trades = [t for t in self.trades if t.trade_type == "sell"]

        wins = [t for t in sell_trades if t.profit > 0]
        losses = [t for t in sell_trades if t.profit <= 0]

        if sell_trades:
            win_rate = len(wins) / len(sell_trades) * 100
            total_profit = sum(t.profit for t in sell_trades)
            avg_win = sum(t.profit for t in wins) / len(wins) if wins else 0
            avg_loss = sum(t.profit for t in losses) / len(losses) if losses else 0

            print(f"\n【交易统计】")
            print(f"  买入次数: {len(buy_trades)}")
            print(f"  卖出次数: {len(sell_trades)}")
            print(f"  盈利次数: {len(wins)} | 亏损次数: {len(losses)}")
            print(f"  胜率: {win_rate:.1f}%")
            print(f"  总盈亏: {total_profit:.2f} 元")

        print("-" * 70)


# 股票池
WATCHLIST = [
    {"symbol": "300015.SZ", "name": "爱尔眼科"},
    {"symbol": "300124.SZ", "name": "汇川技术"},
    {"symbol": "600048.SH", "name": "保利发展"},
    {"symbol": "600519.SH", "name": "贵州茅台"},
    {"symbol": "000001.SZ", "name": "平安银行"},
    {"symbol": "000333.SZ", "name": "美的集团"},
    {"symbol": "002594.SZ", "name": "比亚迪"},
    {"symbol": "601318.SH", "name": "中国平安"},
    {"symbol": "3690.HK", "name": "美团-W"},
    {"symbol": "0700.HK", "name": "腾讯控股"},
    {"symbol": "9988.HK", "name": "阿里巴巴-W"},
]


def main():
    backtester = LGBMBacktester(initial_capital=100000)
    backtester.run_backtest(WATCHLIST)


if __name__ == "__main__":
    main()