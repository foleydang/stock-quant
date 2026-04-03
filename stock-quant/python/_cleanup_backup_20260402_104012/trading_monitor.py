#!/usr/bin/env python3
"""
实盘交易监控系统
功能：
1. 跟踪持仓和资金
2. 生成买入/卖出信号
3. 发送邮件通知（包含具体操作建议）
日志：每日一个日志文件，自动清理超过15天的日志
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_handler import DataHandler
from strategy.intraday_strategy import WATCHLIST_STOCKS, TechnicalIndicators
from strategy.email_notifier import EmailNotifier, create_email_notifier_from_env
from logger import get_daily_logger
from database import get_db
import numpy as np


@dataclass
class Position:
    """持仓"""
    symbol: str
    stock_name: str
    shares: int  # 持有股数
    cost_price: float  # 成本价
    current_price: float  # 当前价
    entry_date: str  # 买入日期
    available: bool = True  # 是否可卖

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def profit(self) -> float:
        return (self.current_price - self.cost_price) * self.shares

    @property
    def profit_pct(self) -> float:
        return (self.current_price - self.cost_price) / self.cost_price * 100


@dataclass
class TradeSignal:
    """交易信号"""
    symbol: str
    stock_name: str
    action: str  # "买入", "卖出", "持有"
    shares: int  # 建议操作股数
    price: float  # 建议价格
    reason: str  # 操作原因
    stop_loss: float  # 止损价
    take_profit: float  # 止盈价
    score: float  # 信号评分


class PortfolioManager:
    """投资组合管理器"""

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}

        # 策略参数
        self.position_pct = 0.15  # 单只最多15%
        self.max_positions = 5
        self.stop_loss_pct = 0.05  # 止损5%
        self.take_profit_pct = 0.08  # 止盈8%

        # 数据处理器
        self.data_handler = DataHandler(force_refresh=True)

        # 持仓文件路径
        self.portfolio_file = os.path.join(os.path.dirname(__file__), 'portfolio.json')
        self.load_portfolio()

    def load_portfolio(self):
        """从文件加载持仓"""
        if os.path.exists(self.portfolio_file):
            try:
                with open(self.portfolio_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cash = data.get('cash', self.initial_capital)
                    positions_data = data.get('positions', {})
                    for symbol, pos_data in positions_data.items():
                        self.positions[symbol] = Position(**pos_data)
                print(f"✓ 加载持仓: 现金 {self.cash:.2f}, 持仓 {len(self.positions)} 只")
            except Exception as e:
                print(f"⚠ 加载持仓失败: {e}")

    def save_portfolio(self):
        """保存持仓到文件"""
        data = {
            'cash': self.cash,
            'positions': {symbol: asdict(pos) for symbol, pos in self.positions.items()},
            'last_update': datetime.now().isoformat()
        }
        with open(self.portfolio_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def update_prices(self):
        """更新所有持仓的当前价格"""
        for symbol, pos in self.positions.items():
            df = self.data_handler.fetch_stock_data(symbol, force_refresh=True)
            if df is not None and len(df) > 0:
                pos.current_price = float(df['close'].iloc[-1])

    def get_total_value(self) -> float:
        """计算总市值"""
        return self.cash + sum(pos.market_value for pos in self.positions.values())

    def get_total_profit(self) -> float:
        """计算总盈亏"""
        return self.get_total_value() - self.initial_capital

    def execute_buy(self, signal: TradeSignal) -> bool:
        """执行买入"""
        if signal.shares <= 0:
            return False

        amount = signal.shares * signal.price
        if amount > self.cash:
            # 调整买入数量
            signal.shares = int(self.cash / signal.price / 100) * 100
            if signal.shares < 100:
                return False
            amount = signal.shares * signal.price

        if signal.symbol in self.positions:
            # 加仓：调整成本价
            pos = self.positions[signal.symbol]
            total_shares = pos.shares + signal.shares
            total_cost = pos.cost_price * pos.shares + signal.price * signal.shares
            pos.shares = total_shares
            pos.cost_price = total_cost / total_shares
            pos.current_price = signal.price
        else:
            # 新建仓位
            self.positions[signal.symbol] = Position(
                symbol=signal.symbol,
                stock_name=signal.stock_name,
                shares=signal.shares,
                cost_price=signal.price,
                current_price=signal.price,
                entry_date=datetime.now().strftime('%Y-%m-%d'),
                available=False  # T+1
            )

        self.cash -= amount
        self.save_portfolio()
        return True

    def execute_sell(self, signal: TradeSignal) -> bool:
        """执行卖出"""
        if signal.symbol not in self.positions:
            return False

        pos = self.positions[signal.symbol]
        if not pos.available or signal.shares > pos.shares:
            signal.shares = min(signal.shares, pos.shares)

        if signal.shares <= 0:
            return False

        amount = signal.shares * signal.price
        self.cash += amount

        pos.shares -= signal.shares
        if pos.shares <= 0:
            del self.positions[signal.symbol]

        self.save_portfolio()
        return True


class TradingMonitor:
    """交易监控器"""

    def __init__(self, initial_capital: float = 100000):
        self.portfolio = PortfolioManager(initial_capital)
        self.email_notifier = create_email_notifier_from_env()
        self.data_handler = DataHandler(force_refresh=True)

        # 日志管理器（每日日志，15天自动清理）
        self.logger = get_daily_logger(prefix="monitor", retention_days=15)

        # 信号参数
        self.min_buy_score = 4.0
        self.sell_threshold = -0.03  # 亏损3%考虑卖出

    def run(self):
        """执行监控"""
        header = "=" * 70
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self.logger.info(header)
        self.logger.info(f"实盘交易监控 - {timestamp}")
        self.logger.info(header)

        # 1. 更新持仓价格
        self.portfolio.update_prices()

        # 2. 生成交易信号
        signals = self._generate_signals()

        # 3. 保存信号到数据库
        self._save_signals_to_db(signals)

        # 4. 保存持仓快照
        self._save_portfolio_snapshot()

        # 5. 打印当前状态
        self._print_status()

        # 6. 发送邮件通知
        if signals:
            self._send_email(signals)
            self.logger.info(f"【信号汇总】发现 {len(signals)} 个交易信号，邮件已发送")
        else:
            self.logger.info("【信号汇总】当前无交易信号，继续监控")

        return signals

    def _save_signals_to_db(self, signals: List[TradeSignal]):
        """保存信号到数据库"""
        db = get_db()
        for signal in signals:
            db.save_signal({
                'symbol': signal.symbol,
                'stock_name': signal.stock_name,
                'price': signal.price,
                'signal': signal.action,
                'score': signal.score,
                'reasons': [signal.reason],
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'timestamp': datetime.now().isoformat()
            })

    def _save_portfolio_snapshot(self):
        """保存持仓快照到数据库"""
        db = get_db()
        positions_data = {}
        for symbol, pos in self.portfolio.positions.items():
            positions_data[symbol] = {
                'stock_name': pos.stock_name,
                'shares': pos.shares,
                'cost_price': pos.cost_price,
                'current_price': pos.current_price,
                'profit': pos.profit,
                'profit_pct': pos.profit_pct
            }

        db.save_portfolio_snapshot({
            'total_value': self.portfolio.get_total_value(),
            'cash': self.portfolio.cash,
            'positions': positions_data,
            'total_pnl': self.portfolio.get_total_profit()
        })

    def _generate_signals(self) -> List[TradeSignal]:
        """生成交易信号"""
        signals = []

        # 检查卖出信号（现有持仓）
        for symbol, pos in list(self.portfolio.positions.items()):
            self.logger.info(f"  检查卖出信号: {pos.stock_name}")
            df = self.data_handler.fetch_stock_data(symbol, force_refresh=True)
            if df is None or len(df) < 60:
                self.logger.warning(f"    数据获取失败或数据不足")
                continue

            current_price = float(df['close'].iloc[-1])
            pos.current_price = current_price

            # 止损检查
            loss_pct = (current_price - pos.cost_price) / pos.cost_price
            if loss_pct <= -self.portfolio.stop_loss_pct:
                signals.append(TradeSignal(
                    symbol=symbol,
                    stock_name=pos.stock_name,
                    action="卖出",
                    shares=pos.shares,
                    price=current_price,
                    reason=f"触发止损(亏损{abs(loss_pct)*100:.1f}%)",
                    stop_loss=pos.cost_price * (1 - self.portfolio.stop_loss_pct),
                    take_profit=pos.cost_price * (1 + self.portfolio.take_profit_pct),
                    score=-5
                ))
                continue

            # 止盈检查
            profit_pct = (current_price - pos.cost_price) / pos.cost_price
            if profit_pct >= self.portfolio.take_profit_pct:
                signals.append(TradeSignal(
                    symbol=symbol,
                    stock_name=pos.stock_name,
                    action="卖出",
                    shares=pos.shares,
                    price=current_price,
                    reason=f"触发止盈(盈利{profit_pct*100:.1f}%)",
                    stop_loss=pos.cost_price * (1 - self.portfolio.stop_loss_pct),
                    take_profit=pos.cost_price * (1 + self.portfolio.take_profit_pct),
                    score=5
                ))
                continue

            # 信号卖出
            sell_signal = self._check_sell_signal(df)
            if sell_signal:
                signals.append(TradeSignal(
                    symbol=symbol,
                    stock_name=pos.stock_name,
                    action="卖出",
                    shares=pos.shares,
                    price=current_price,
                    reason=sell_signal,
                    stop_loss=pos.cost_price * (1 - self.portfolio.stop_loss_pct),
                    take_profit=pos.cost_price * (1 + self.portfolio.take_profit_pct),
                    score=-2
                ))

        # 检查买入信号（有空位时）
        if len(self.portfolio.positions) < self.portfolio.max_positions:
            self.logger.info("【信号检测】检查买入机会...")
            for stock in WATCHLIST_STOCKS:
                symbol = stock['symbol']
                self.logger.info(f"  检查 {stock['name']}({symbol})...")
                if symbol in self.portfolio.positions:
                    self.logger.info("    - 已持有，跳过")
                    continue

                df = self.data_handler.fetch_stock_data(symbol, force_refresh=True)
                if df is None or len(df) < 60:
                    self.logger.warning("    数据获取失败或数据不足")
                    continue

                buy_signal = self._check_buy_signal(df, stock['name'])
                if buy_signal:
                    self.logger.info(f"    发现买入信号: {buy_signal['reason']} (评分:{buy_signal['score']:.1f})")
                    current_price = float(df['close'].iloc[-1])
                    shares = self._calculate_buy_shares(current_price)

                    if shares >= 100:
                        signals.append(TradeSignal(
                            symbol=symbol,
                            stock_name=stock['name'],
                            action="买入",
                            shares=shares,
                            price=current_price,
                            reason=buy_signal['reason'],
                            stop_loss=current_price * 0.95,
                            take_profit=current_price * 1.08,
                            score=buy_signal['score']
                        ))
                else:
                    self.logger.info("    - 无买入信号")

        return signals

    def _check_sell_signal(self, df) -> Optional[str]:
        """检查卖出信号"""
        close = df['close'].values

        sell_score = 0
        reasons = []

        # RSI超买
        rsi = TechnicalIndicators.calculate_rsi(close, 14)[-1]
        if rsi > 75:
            sell_score += 2
            reasons.append("RSI超买")
        elif rsi > 65:
            sell_score += 1

        # MACD死叉
        macd_data = TechnicalIndicators.calculate_macd(close)
        if len(macd_data['macd']) >= 2:
            if macd_data['macd'][-1] < macd_data['signal'][-1]:
                if macd_data['histogram'][-1] < macd_data['histogram'][-2]:
                    sell_score += 2
                    reasons.append("MACD死叉")

        # 均线转空
        ma5 = np.mean(close[-5:])
        ma10 = np.mean(close[-10:])
        ma20 = np.mean(close[-20:])
        if ma5 < ma10 and ma10 < ma20:
            sell_score += 2
            reasons.append("均线空头")

        if sell_score >= 3:
            return ", ".join(reasons)
        return None

    def _check_buy_signal(self, df, stock_name: str) -> Optional[Dict]:
        """检查买入信号"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        score = 0.0
        reasons = []

        # 趋势过滤
        trend_20 = (close[-1] - close[-20]) / close[-20] * 100
        if trend_20 < -10:
            return None

        ma5 = np.mean(close[-5:])
        ma10 = np.mean(close[-10:])
        ma20 = np.mean(close[-20:])

        # 均线空头排列时禁止买入
        if ma5 < ma10 and ma10 < ma20 and trend_20 < -5:
            return None

        is_downtrend = trend_20 < -3 or (ma5 < ma10 and ma10 < ma20)

        # RSI
        rsi = TechnicalIndicators.calculate_rsi(close, 14)[-1]
        if rsi < 20:
            score += 3
            reasons.append(f"RSI极度超卖({rsi:.1f})")
        elif rsi < 30:
            score += 2
            reasons.append(f"RSI超卖({rsi:.1f})")
        elif rsi < 40:
            score += 1
            reasons.append(f"RSI偏低({rsi:.1f})")

        # RSI底背离
        if len(close) >= 20:
            price_low1 = min(close[-20:-10])
            price_low2 = min(close[-10:])
            rsi_arr = TechnicalIndicators.calculate_rsi(close, 14)
            if len(rsi_arr) >= 20:
                rsi_low1 = min(rsi_arr[-20:-10])
                rsi_low2 = min(rsi_arr[-10:])
                if price_low2 < price_low1 and rsi_low2 > rsi_low1:
                    score += 3
                    reasons.append("RSI底背离")

        # MACD金叉
        macd_data = TechnicalIndicators.calculate_macd(close)
        if len(macd_data['macd']) >= 2:
            if macd_data['macd'][-1] > macd_data['signal'][-1] and macd_data['macd'][-2] <= macd_data['signal'][-2]:
                score += 2
                reasons.append("MACD金叉")

        # 均线
        if close[-1] > ma5:
            score += 1
            reasons.append("价格站上MA5")
        if close[-1] > ma10:
            score += 1
            reasons.append("价格站上MA10")
        if ma5 > ma10:
            score += 1.5
            reasons.append("MA5>MA10")

        # KDJ
        kdj = TechnicalIndicators.calculate_kdj(high, low, close)
        j = kdj['j'][-1]
        if j < 10:
            score += 2
            reasons.append(f"KDJ超卖(J={j:.1f})")
        if kdj['k'][-1] > kdj['d'][-1] and len(kdj['k']) >= 2 and kdj['k'][-2] <= kdj['d'][-2]:
            score += 1.5
            reasons.append("KDJ金叉")

        # 放量
        vol_5 = np.mean(volume[-5:])
        vol_20 = np.mean(volume[-20:])
        if vol_5 > vol_20 * 1.3:
            score += 1
            reasons.append("放量确认")

        # 下跌趋势需要更高分数
        min_score = 5.0 if is_downtrend else self.min_buy_score

        if score >= min_score:
            return {
                'score': score,
                'reason': ", ".join(reasons[:3])
            }
        return None

    def _calculate_buy_shares(self, price: float) -> int:
        """计算买入股数"""
        max_invest = min(
            self.portfolio.cash * 0.95,
            self.portfolio.initial_capital * self.portfolio.position_pct
        )
        shares = int(max_invest / price / 100) * 100
        return shares

    def _print_status(self):
        """打印当前状态"""
        self.logger.info("【账户状态】")
        self.logger.info(f"  总市值: {self.portfolio.get_total_value():.2f} 元")
        self.logger.info(f"  现金: {self.portfolio.cash:.2f} 元")
        self.logger.info(f"  持仓市值: {sum(p.market_value for p in self.portfolio.positions.values()):.2f} 元")
        self.logger.info(f"  总盈亏: {self.portfolio.get_total_profit():.2f} 元")

        if self.portfolio.positions:
            self.logger.info("【当前持仓】")
            for pos in self.portfolio.positions.values():
                status = "盈利" if pos.profit > 0 else "亏损"
                self.logger.info(f"  [{status}] {pos.stock_name}: {pos.shares}股 @ {pos.cost_price:.2f} -> {pos.current_price:.2f} | "
                      f"盈亏:{pos.profit:.0f}({pos.profit_pct:.1f}%)")

    def _send_email(self, signals: List[TradeSignal]):
        """发送邮件通知"""
        if not self.email_notifier:
            self.logger.warning("邮件未配置")
            return

        subject = f"【交易信号】{datetime.now().strftime('%m-%d %H:%M')} - 发现{len(signals)}个信号"

        # 构建邮件内容
        content = self._build_email_content(signals)

        # 发送邮件
        self.email_notifier.send_email(subject, content)
        self.logger.info(f"邮件已发送: {subject}")

    def _build_email_content(self, signals: List[TradeSignal]) -> str:
        """构建邮件内容"""
        lines = []
        lines.append("=" * 60)
        lines.append(f"交易信号通知 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)

        # 账户状态
        lines.append("\n【账户状态】")
        lines.append(f"总市值: {self.portfolio.get_total_value():,.2f} 元")
        lines.append(f"现金余额: {self.portfolio.cash:,.2f} 元")
        lines.append(f"持仓市值: {sum(p.market_value for p in self.portfolio.positions.values()):,.2f} 元")
        lines.append(f"总盈亏: {self.portfolio.get_total_profit():,.2f} 元")

        # 当前持仓
        if self.portfolio.positions:
            lines.append("\n【当前持仓】")
            lines.append("-" * 60)
            for pos in self.portfolio.positions.values():
                emoji = "✅" if pos.profit > 0 else "❌"
                lines.append(f"{emoji} {pos.stock_name}({pos.symbol})")
                lines.append(f"   持股: {pos.shares}股 | 成本: {pos.cost_price:.2f} | 现价: {pos.current_price:.2f}")
                lines.append(f"   盈亏: {pos.profit:.0f}元 ({pos.profit_pct:.1f}%)")
                lines.append(f"   止损: {pos.cost_price*0.95:.2f} | 止盈: {pos.cost_price*1.08:.2f}")
        else:
            lines.append("\n【当前持仓】空仓")

        # 交易信号
        lines.append("\n" + "=" * 60)
        lines.append("【交易信号】")
        lines.append("=" * 60)

        buy_signals = [s for s in signals if s.action == "买入"]
        sell_signals = [s for s in signals if s.action == "卖出"]

        if sell_signals:
            lines.append("\n🔴 卖出信号:")
            lines.append("-" * 60)
            for s in sell_signals:
                lines.append(f"股票: {s.stock_name}({s.symbol})")
                lines.append(f"操作: 卖出 {s.shares}股 @ 约{s.price:.2f}")
                lines.append(f"原因: {s.reason}")
                lines.append("")

        if buy_signals:
            lines.append("\n🟢 买入信号:")
            lines.append("-" * 60)
            for s in buy_signals:
                lines.append(f"股票: {s.stock_name}({s.symbol})")
                lines.append(f"操作: 买入 {s.shares}股 @ 约{s.price:.2f}")
                lines.append(f"原因: {s.reason}")
                lines.append(f"止损: {s.stop_loss:.2f} | 止盈: {s.take_profit:.2f}")
                lines.append(f"信号评分: {s.score:.1f}")
                lines.append("")

        # 操作提示
        lines.append("\n" + "=" * 60)
        lines.append("【操作提示】")
        lines.append("-" * 60)
        lines.append("1. 请根据上述信号在交易软件中手动操作")
        lines.append("2. 操作完成后，请回复邮件告知成交情况")
        lines.append("3. 系统将在下次监控时更新持仓")
        lines.append("4. 止损止盈仅供参考，请根据实际情况调整")
        lines.append("=" * 60)

        return "\n".join(lines)


def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════╗
║           实盘交易监控系统                                   ║
║           初始资金: 10万元                                   ║
╚══════════════════════════════════════════════════════════╝
    """)

    monitor = TradingMonitor(initial_capital=100000)
    monitor.run()


if __name__ == "__main__":
    main()