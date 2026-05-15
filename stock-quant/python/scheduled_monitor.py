#!/usr/bin/env python3
"""
定时交易监控脚本
功能:
1. 使用LGBM模型预测最佳买卖点
2. 10万资金持仓管理
3. 邮件通知交易信号
4. 定时执行(每30分钟)

使用方法:
    python scheduled_monitor.py --once       # 执行一次
    python scheduled_monitor.py              # 后台定时运行
"""

import os
import sys
import json
import pickle
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# 路径配置（动态获取）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'data/stock_data.db')
MODEL_PATH = os.path.join(BASE_DIR, 'models/lgb_hs300/model.pkl')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

sys.path.insert(0, BASE_DIR)

# 邮件配置
try:
    from strategy.email_notifier import EmailNotifier, create_email_notifier_from_env
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

# 特征工程
try:
    from strategy.train_lgb_enhanced import EnhancedFeatureEngineer
    FEATURE_ENGINEER_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEER_AVAILABLE = False


@dataclass
class TradeSignal:
    """交易信号"""
    symbol: str
    stock_name: str
    action: str  # "买入", "卖出", "持有"
    shares: int
    price: float
    reason: str
    up_prob: float  # LGBM预测上涨概率
    stop_loss: float
    take_profit: float
    score: float


@dataclass
class Position:
    """持仓"""
    symbol: str
    stock_name: str
    shares: int
    cost_price: float
    current_price: float
    entry_date: str
    available: bool = True

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def profit(self) -> float:
        return (self.current_price - self.cost_price) * self.shares

    @property
    def profit_pct(self) -> float:
        return (self.current_price - self.cost_price) / self.cost_price * 100


class ScheduledMonitor:
    """定时监控器 - LGBM策略 + 10万资金"""

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}

        # 策略参数
        self.position_pct = 0.20  # 单只最多20%
        self.max_positions = 5    # 最多持有5只
        self.stop_loss_pct = 0.08  # 止损8%
        self.take_profit_pct = 0.10  # 止盈10%
        self.buy_threshold = 0.60  # 买入阈值
        self.sell_threshold = 0.40  # 卖出阈值

        # 加载模型
        self.model = None
        self._load_model()

        # 加载持仓
        self.portfolio_file = os.path.join(BASE_DIR, 'portfolio.json')
        self._load_portfolio()

        # 邮件通知器
        self.email_notifier = create_email_notifier_from_env() if EMAIL_AVAILABLE else None

        # 股票名称映射
        self.stock_names = self._load_stock_names()

    def _load_model(self):
        """加载LGBM模型"""
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, 'rb') as f:
                    model_data = pickle.load(f)
                self.model = model_data.get('model')
                self.model_accuracy = model_data.get('cv_accuracy', 0)
                print(f"✓ 模型加载成功, 准确率: {self.model_accuracy:.2%}")
            except Exception as e:
                print(f"⚠ 模型加载失败: {e}")
        else:
            print(f"⚠ 模型文件不存在: {MODEL_PATH}")

    def _load_portfolio(self):
        """加载持仓"""
        if os.path.exists(self.portfolio_file):
            try:
                with open(self.portfolio_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cash = data.get('cash', self.initial_capital)
                    for symbol, pos_data in data.get('positions', {}).items():
                        self.positions[symbol] = Position(**pos_data)
                print(f"✓ 加载持仓: 现金 {self.cash:.2f}, 持仓 {len(self.positions)} 只")
            except Exception as e:
                print(f"⚠ 加载持仓失败: {e}")

    def _save_portfolio(self):
        """保存持仓"""
        data = {
            'cash': self.cash,
            'positions': {s: asdict(p) for s, p in self.positions.items()},
            'last_update': datetime.now().isoformat()
        }
        with open(self.portfolio_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_stock_names(self) -> Dict[str, str]:
        """加载股票名称"""
        names = {}
        if os.path.exists(DB_PATH):
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute('SELECT symbol, name FROM stock_info')
                for row in cursor.fetchall():
                    names[row[0]] = row[1]
                conn.close()
            except Exception as e:
                pass
        return names

    def get_total_value(self) -> float:
        """总市值"""
        return self.cash + sum(p.market_value for p in self.positions.values())

    def get_total_profit(self) -> float:
        """总盈亏"""
        return self.get_total_value() - self.initial_capital

    def run(self) -> List[TradeSignal]:
        """执行监控"""
        print("=" * 70)
        print(f"LGBM交易监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        signals = []

        # 1. 更新持仓价格
        self._update_positions_price()

        # 2. 检查卖出信号（现有持仓）
        sell_signals = self._check_sell_signals()
        signals.extend(sell_signals)

        # 3. 检查买入信号（有空位时）
        if len(self.positions) < self.max_positions:
            buy_signals = self._check_buy_signals()
            signals.extend(buy_signals)

        # 4. 打印状态
        self._print_status()

        # 5. 发送邮件
        if signals and self.email_notifier:
            self._send_email(signals)

        # 6. 保存结果
        self._save_result(signals)

        return signals

    def _update_positions_price(self):
        """更新持仓价格"""
        conn = sqlite3.connect(DB_PATH)
        for symbol, pos in self.positions.items():
            try:
                df = pd.read_sql_query(
                    'SELECT close FROM kline_30m WHERE symbol=? ORDER BY date DESC LIMIT 1',
                    conn, params=(symbol,)
                )
                if not df.empty:
                    pos.current_price = float(df['close'].iloc[0])
            except Exception as e:
                pass
        conn.close()

    def _check_sell_signals(self) -> List[TradeSignal]:
        """检查卖出信号"""
        signals = []
        conn = sqlite3.connect(DB_PATH)

        for symbol, pos in list(self.positions.items()):
            # 获取数据
            df = pd.read_sql_query(
                'SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol=? ORDER BY date',
                conn, params=(symbol,)
            )
            if len(df) < 200:
                continue

            current_price = float(df['close'].iloc[-1])
            pos.current_price = current_price

            # 计算盈亏
            profit_pct = (current_price - pos.cost_price) / pos.cost_price

            # 止损止盈检查
            reason = None
            score = 0

            if profit_pct <= -self.stop_loss_pct:
                reason = f"触发止损(亏损{abs(profit_pct)*100:.1f}%)"
                score = -5
            elif profit_pct >= self.take_profit_pct:
                reason = f"触发止盈(盈利{profit_pct*100:.1f}%)"
                score = 5
            else:
                # 模型预测检查
                up_prob = self._predict_up_prob(df)
                if up_prob is not None and up_prob < self.sell_threshold:
                    reason = f"模型看跌(上涨概率{up_prob:.0%})"
                    score = -2

            if reason:
                signals.append(TradeSignal(
                    symbol=symbol,
                    stock_name=pos.stock_name,
                    action="卖出",
                    shares=pos.shares,
                    price=current_price,
                    reason=reason,
                    up_prob=up_prob or 0,
                    stop_loss=pos.cost_price * (1 - self.stop_loss_pct),
                    take_profit=pos.cost_price * (1 + self.take_profit_pct),
                    score=score
                ))

        conn.close()
        return signals

    def _check_buy_signals(self) -> List[TradeSignal]:
        """检查买入信号"""
        signals = []
        conn = sqlite3.connect(DB_PATH)

        # 获取候选股票
        cursor = conn.cursor()
        cursor.execute('''
            SELECT symbol, COUNT(*) as cnt FROM kline_30m
            GROUP BY symbol HAVING cnt >= 500 ORDER BY cnt DESC LIMIT 30
        ''')
        candidates = [row[0] for row in cursor.fetchall()]

        for symbol in candidates:
            if symbol in self.positions:
                continue

            df = pd.read_sql_query(
                'SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol=? ORDER BY date',
                conn, params=(symbol,)
            )
            if len(df) < 200:
                continue

            current_price = float(df['close'].iloc[-1])
            up_prob = self._predict_up_prob(df)

            if up_prob is not None and up_prob >= self.buy_threshold:
                # 计算买入股数
                max_invest = min(
                    self.cash * 0.95,
                    self.initial_capital * self.position_pct
                )
                shares = int(max_invest / current_price / 100) * 100

                if shares >= 100:
                    stock_name = self.stock_names.get(symbol, symbol)
                    signals.append(TradeSignal(
                        symbol=symbol,
                        stock_name=stock_name,
                        action="买入",
                        shares=shares,
                        price=current_price,
                        reason=f"模型看涨(上涨概率{up_prob:.0%})",
                        up_prob=up_prob,
                        stop_loss=current_price * (1 - self.stop_loss_pct),
                        take_profit=current_price * (1 + self.take_profit_pct),
                        score=up_prob * 10
                    ))

        conn.close()
        return signals

    def _predict_up_prob(self, df: pd.DataFrame) -> Optional[float]:
        """预测上涨概率"""
        if not self.model or not FEATURE_ENGINEER_AVAILABLE:
            return None

        try:
            features = EnhancedFeatureEngineer.calculate_features(df)
            if features.iloc[-1].isna().any():
                return None
            up_prob = self.model.predict_proba([features.iloc[-1].values])[0][1]
            return up_prob
        except Exception as e:
            return None

    def _print_status(self):
        """打印状态"""
        print("\n【账户状态】")
        print(f"  总市值: ¥{self.get_total_value():,.0f}")
        print(f"  现金: ¥{self.cash:,.0f}")
        print(f"  持仓市值: ¥{sum(p.market_value for p in self.positions.values()):,.0f}")
        print(f"  总盈亏: ¥{self.get_total_profit():,.0f} ({self.get_total_profit()/self.initial_capital*100:.1f}%)")

        if self.positions:
            print("\n【当前持仓】")
            for pos in self.positions.values():
                status = "✅" if pos.profit > 0 else "❌"
                print(f"  {status} {pos.stock_name}: {pos.shares}股 @ ¥{pos.cost_price:.2f} → ¥{pos.current_price:.2f} | "
                      f"盈亏 ¥{pos.profit:.0f} ({pos.profit_pct:.1f}%)")

    def _send_email(self, signals: List[TradeSignal]):
        """发送邮件"""
        if not self.email_notifier:
            print("⚠ 邮件未配置")
            return

        subject = f"【交易信号】{datetime.now().strftime('%m-%d %H:%M')} - {len(signals)}个信号"

        content = self._build_email_content(signals)
        self.email_notifier.send_email(subject, content)
        print(f"✓ 邮件已发送: {subject}")

    def _build_email_content(self, signals: List[TradeSignal]) -> str:
        """构建邮件内容"""
        lines = []
        lines.append("=" * 60)
        lines.append(f"交易信号通知 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)

        # 账户状态
        lines.append("\n【账户状态】")
        lines.append(f"总市值: ¥{self.get_total_value():,.0f}")
        lines.append(f"现金: ¥{self.cash:,.0f}")
        lines.append(f"持仓市值: ¥{sum(p.market_value for p in self.positions.values()):,.0f}")
        lines.append(f"总盈亏: ¥{self.get_total_profit():,.0f}")

        # 持仓
        if self.positions:
            lines.append("\n【当前持仓】")
            for pos in self.positions.values():
                emoji = "✅" if pos.profit > 0 else "❌"
                lines.append(f"{emoji} {pos.stock_name}: {pos.shares}股 @ ¥{pos.cost_price:.2f} → ¥{pos.current_price:.2f}")
                lines.append(f"   盈亏: ¥{pos.profit:.0f} ({pos.profit_pct:.1f}%)")
        else:
            lines.append("\n【当前持仓】空仓")

        # 交易信号
        lines.append("\n" + "=" * 60)
        lines.append("【交易信号】")
        lines.append("=" * 60)

        sell_signals = [s for s in signals if s.action == "卖出"]
        buy_signals = [s for s in signals if s.action == "买入"]

        if sell_signals:
            lines.append("\n🔴 卖出信号:")
            for s in sell_signals:
                lines.append(f"  {s.stock_name}({s.symbol})")
                lines.append(f"    操作: 卖出 {s.shares}股 @ ¥{s.price:.2f}")
                lines.append(f"    原因: {s.reason}")
                lines.append(f"    止损: ¥{s.stop_loss:.2f} | 止盈: ¥{s.take_profit:.2f}")
                lines.append("")

        if buy_signals:
            lines.append("\n🟢 买入信号:")
            for s in buy_signals:
                lines.append(f"  {s.stock_name}({s.symbol})")
                lines.append(f"    操作: 买入 {s.shares}股 @ ¥{s.price:.2f}")
                lines.append(f"    原因: {s.reason}")
                lines.append(f"    止损: ¥{s.stop_loss:.2f} | 止盈: ¥{s.take_profit:.2f}")
                lines.append(f"    信号评分: {s.score:.1f}")
                lines.append("")

        lines.append("\n" + "=" * 60)
        lines.append("【操作提示】")
        lines.append("1. 请根据信号在交易软件中操作")
        lines.append("2. 止损8%, 止盈10%")
        lines.append("3. 买入阈值: 模型预测上涨概率 > 60%")
        lines.append("4. 卖出阈值: 模型预测上涨概率 < 40%")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _save_result(self, signals: List[TradeSignal]):
        """保存结果"""
        os.makedirs(LOGS_DIR, exist_ok=True)
        result = {
            'timestamp': datetime.now().isoformat(),
            'signals': [asdict(s) for s in signals],
            'portfolio': {
                'cash': self.cash,
                'positions': {s: asdict(p) for s, p in self.positions.items()},
                'total_value': self.get_total_value(),
                'total_profit': self.get_total_profit()
            }
        }
        result_file = os.path.join(LOGS_DIR, f'monitor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✓ 结果已保存: {result_file}")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='LGBM定时交易监控')
    parser.add_argument('--once', action='store_true', help='只执行一次')
    parser.add_argument('--capital', type=float, default=100000, help='初始资金')
    args = parser.parse_args()

    monitor = ScheduledMonitor(initial_capital=args.capital)

    if args.once:
        signals = monitor.run()
        print(f"\n共发现 {len(signals)} 个交易信号")
        return signals

    # 定时运行
    import schedule
    print("=" * 70)
    print("启动定时监控 (每30分钟)")
    print("=" * 70)

    monitor.run()  # 立即执行一次
    schedule.every(30).minutes.do(monitor.run)

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    import time
    main()