#!/usr/bin/env python3
"""
完整交易监控系统
功能:
1. 使用DataHandler更新30分钟数据到DB
2. 从一个月前开始回测模拟，写入持仓和交易记录
3. 定时监控并发送邮件通知

使用方法:
    python full_monitor.py --backtest      # 回测模拟(从一个月前)
    python full_monitor.py --update        # 仅更新数据
    python full_monitor.py --monitor       # 实时监控
    python full_monitor.py                 # 完整流程
"""

import os
import sys
import json
import pickle
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

BASE_DIR = '/Users/foleydang/github/stock-quant/stock-quant/python'
DB_PATH = f'{BASE_DIR}/data/stock_data.db'
MODEL_PATH = f'{BASE_DIR}/models/lgb_hs300/model.pkl'
LOGS_DIR = f'{BASE_DIR}/logs'

sys.path.insert(0, BASE_DIR)

# 数据处理器
from data.data_handler import DataHandler

# 邮件通知
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
class Position:
    """持仓"""
    symbol: str
    stock_name: str
    shares: int
    cost_price: float
    current_price: float
    entry_date: str
    entry_time: str = ""
    available: int = 1
    stop_loss: float = 0.0
    take_profit: float = 0.0

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def profit(self) -> float:
        return (self.current_price - self.cost_price) * self.shares

    @property
    def profit_pct(self) -> float:
        if self.cost_price == 0:
            return 0
        return (self.current_price - self.cost_price) / self.cost_price * 100


class FullMonitor:
    """完整监控系统"""

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.db_path = DB_PATH
        self.model_path = MODEL_PATH

        # 策略参数
        self.position_pct = 0.20  # 单只最多20%
        self.max_positions = 5    # 最多5只
        self.stop_loss_pct = 0.08  # 止损8%
        self.take_profit_pct = 0.10  # 止盈10%
        self.buy_threshold = 0.60
        self.sell_threshold = 0.40

        # 数据处理器
        self.data_handler = DataHandler(force_refresh=True)

        # 模型
        self.model = None
        self._load_model()

        # 邮件
        self.email_notifier = create_email_notifier_from_env() if EMAIL_AVAILABLE else None

        # 股票名称
        self.stock_names = {}

    # 关注股票列表（不持仓，仅监控价格）
        self.watchlist = [
            {'symbol': '9988.HK', 'name': '阿里巴巴-W'},
        ]

    def _load_model(self):
        """加载模型"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                self.model = model_data.get('model')
                self.model_accuracy = model_data.get('cv_accuracy', 0)
                print(f"✓ 模型加载成功, 准确率: {self.model_accuracy:.2%}")
            except Exception as e:
                print(f"⚠ 模型加载失败: {e}")

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接"""
        return sqlite3.connect(self.db_path)

    def _load_stock_names(self) -> Dict[str, str]:
        """加载股票名称"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT symbol, name FROM stock_info')
        names = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return names

    # ==================== 数据更新 ====================

    def update_data(self, symbols: List[str] = None) -> int:
        """
        使用DataHandler更新数据到DB

        Args:
            symbols: 要更新的股票列表，None则更新所有

        Returns:
            更新成功的股票数量
        """
        print("\n" + "=" * 70)
        print(f"数据更新 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        conn = self._get_conn()
        cursor = conn.cursor()

        # 获取要更新的股票
        if symbols is None:
            cursor.execute('SELECT symbol FROM stock_info')
            symbols = [row[0] for row in cursor.fetchall()]

        success_count = 0
        total = len(symbols)

        for i, symbol in enumerate(symbols[:50], 1):  # 每次最多更新50只
            print(f"  [{i}/{total}] {symbol}...", end=" ")

            try:
                # 使用DataHandler获取数据
                df = self.data_handler.fetch_stock_data(symbol, force_refresh=True)

                if df is None or len(df) < 50:
                    print("数据不足")
                    continue

                # 写入DB
                self._write_kline_to_db(conn, symbol, df)
                print(f"✓ {len(df)}条")
                success_count += 1

            except Exception as e:
                print(f"失败: {e}")

        conn.commit()
        conn.close()

        print(f"\n✓ 数据更新完成: {success_count}/{total} 只股票")
        return success_count

    def _write_kline_to_db(self, conn: sqlite3.Connection, symbol: str, df: pd.DataFrame):
        """写入K线数据到DB"""
        cursor = conn.cursor()

        # 删除旧数据
        cursor.execute('DELETE FROM kline_30m WHERE symbol = ?', (symbol,))

        # 插入新数据
        for _, row in df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(row['date'], pd.Timestamp) else str(row['date'])
            cursor.execute('''
                INSERT INTO kline_30m (symbol, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, date_str,
                float(row['open']), float(row['high']), float(row['low']),
                float(row['close']), float(row['volume'])
            ))

    # ==================== 持仓管理 ====================

    def get_current_positions(self) -> Dict[str, Position]:
        """从DB获取当前持仓"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM positions WHERE shares > 0')

        positions = {}
        for row in cursor.fetchall():
            pos = Position(
                symbol=row[0],
                stock_name=row[1],
                shares=row[2],
                cost_price=row[3],
                current_price=row[4],
                entry_date=row[5],
                entry_time=row[6] or "",
                available=row[7],
                stop_loss=row[8],
                take_profit=row[9]
            )
            positions[pos.symbol] = pos

        conn.close()
        return positions

    def get_current_cash(self) -> float:
        """从DB获取当前现金"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT cash FROM account ORDER BY id DESC LIMIT 1')
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else self.initial_capital

    def save_position(self, pos: Position):
        """保存持仓到DB"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO positions
            (symbol, stock_name, shares, cost_price, current_price, entry_date, entry_time, available, stop_loss, take_profit, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pos.symbol, pos.stock_name, pos.shares, pos.cost_price, pos.current_price,
            pos.entry_date, pos.entry_time, pos.available, pos.stop_loss, pos.take_profit,
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()

    def delete_position(self, symbol: str):
        """删除持仓"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM positions WHERE symbol = ?', (symbol,))
        conn.commit()
        conn.close()

    def save_trade(self, symbol: str, stock_name: str, action: str, shares: int,
                   price: float, profit: float = 0, reason: str = "", up_prob: float = 0,
                   simulated: int = 0):
        """保存交易记录"""
        conn = self._get_conn()
        cursor = conn.cursor()
        amount = shares * price
        cursor.execute('''
            INSERT INTO trades (symbol, stock_name, action, shares, price, amount, profit, reason, up_prob, trade_time, simulated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol, stock_name, action, shares, price, amount, profit, reason, up_prob,
            datetime.now().isoformat(), simulated
        ))
        conn.commit()
        conn.close()

    def save_account(self, cash: float, positions: Dict[str, Position]):
        """保存账户状态"""
        conn = self._get_conn()
        cursor = conn.cursor()
        total_value = cash + sum(p.market_value for p in positions.values())
        total_profit = total_value - self.initial_capital
        cursor.execute('''
            INSERT INTO account (cash, total_value, total_profit, positions_count, updated_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (cash, total_value, total_profit, len(positions), datetime.now().isoformat()))
        conn.commit()
        conn.close()

    # ==================== 回测模拟 ====================

    def backtest_simulate(self, start_date: str = None, days: int = 30) -> Dict:
        """
        回测模拟交易

        Args:
            start_date: 开始日期，默认一个月前
            days: 回测天数

        Returns:
            回测结果
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        end_date = datetime.now().strftime('%Y-%m-%d')

        print("\n" + "=" * 70)
        print(f"回测模拟 - {start_date} ~ {end_date}")
        print("=" * 70)

        # 清空持仓和交易记录（模拟数据）
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM positions')
        cursor.execute('DELETE FROM trades WHERE simulated = 1')
        cursor.execute('DELETE FROM account')
        conn.commit()
        conn.close()

        # 初始化账户
        cash = self.initial_capital
        positions: Dict[str, Position] = {}

        # 获取候选股票
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT symbol FROM kline_30m
            GROUP BY symbol HAVING COUNT(*) >= 500
            ORDER BY COUNT(*) DESC LIMIT 30
        ''')
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()

        self.stock_names = self._load_stock_names()

        # 回测
        total_trades = 0
        wins = 0

        # 模拟时间线（每30分钟一个点）
        conn = self._get_conn()
        for symbol in symbols:
            df = pd.read_sql_query(
                'SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol=? AND date >= ? ORDER BY date',
                conn, params=(symbol, start_date)
            )

            if len(df) < 100:
                continue

            df['date'] = pd.to_datetime(df['date'])

            # 每8个bar检查一次(约4小时)
            for i in range(100, len(df), 8):
                current_time = df['date'].iloc[i]
                current_price = float(df['close'].iloc[i])

                # 更新持仓价格
                for s, pos in positions.items():
                    pos.current_price = current_price if s == symbol else pos.current_price

                # 检查卖出（现有持仓）
                if symbol in positions:
                    pos = positions[symbol]
                    profit_pct = (current_price - pos.cost_price) / pos.cost_price

                    # 止损止盈
                    if profit_pct <= -self.stop_loss_pct or profit_pct >= self.take_profit_pct:
                        # 卖出
                        sell_amount = pos.shares * current_price
                        cash += sell_amount
                        profit = (current_price - pos.cost_price) * pos.shares

                        reason = f"止盈({profit_pct*100:.1f}%)" if profit_pct >= self.take_profit_pct else f"止损({profit_pct*100:.1f}%)"

                        self.save_trade(symbol, pos.stock_name, "卖出", pos.shares, current_price,
                                       profit, reason, 0, simulated=1)

                        del positions[symbol]
                        self.delete_position(symbol)
                        total_trades += 1
                        if profit > 0:
                            wins += 1
                        continue

                    # 模型预测卖出
                    up_prob = self._predict_up(df.iloc[:i+1])
                    if up_prob and up_prob < self.sell_threshold:
                        sell_amount = pos.shares * current_price
                        cash += sell_amount
                        profit = (current_price - pos.cost_price) * pos.shares

                        self.save_trade(symbol, pos.stock_name, "卖出", pos.shares, current_price,
                                       profit, f"模型看跌({up_prob:.0%})", up_prob, simulated=1)

                        del positions[symbol]
                        self.delete_position(symbol)
                        total_trades += 1
                        if profit > 0:
                            wins += 1

                # 检查买入
                elif len(positions) < self.max_positions and cash > 10000:
                    up_prob = self._predict_up(df.iloc[:i+1])

                    if up_prob and up_prob >= self.buy_threshold:
                        # 计算买入数量
                        max_invest = min(cash * 0.9, self.initial_capital * self.position_pct)
                        shares = int(max_invest / current_price / 100) * 100

                        if shares >= 100:
                            buy_amount = shares * current_price
                            if buy_amount <= cash:
                                cash -= buy_amount
                                stock_name = self.stock_names.get(symbol, symbol)

                                pos = Position(
                                    symbol=symbol,
                                    stock_name=stock_name,
                                    shares=shares,
                                    cost_price=current_price,
                                    current_price=current_price,
                                    entry_date=current_time.strftime('%Y-%m-%d'),
                                    entry_time=current_time.strftime('%H:%M'),
                                    stop_loss=current_price * (1 - self.stop_loss_pct),
                                    take_profit=current_price * (1 + self.take_profit_pct)
                                )
                                positions[symbol] = pos
                                self.save_position(pos)

                                self.save_trade(symbol, stock_name, "买入", shares, current_price,
                                               0, f"模型看涨({up_prob:.0%})", up_prob, simulated=1)
                                total_trades += 1

            # 保存账户状态
            self.save_account(cash, positions)

        conn.close()

        # 最终结果
        final_value = cash + sum(p.market_value for p in positions.values())
        final_profit = final_value - self.initial_capital
        win_rate = wins / max(total_trades, 1) * 100

        result = {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'final_profit': final_profit,
            'profit_pct': final_profit / self.initial_capital * 100,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'positions': {s: asdict(p) for s, p in positions.items()},
            'cash': cash
        }

        print(f"\n【回测结果】")
        print(f"  初始资金: ¥{self.initial_capital:,.0f}")
        print(f"  最终市值: ¥{final_value:,.0f}")
        print(f"  盈亏: ¥{final_profit:,.0f} ({result['profit_pct']:.1f}%)")
        print(f"  交易次数: {total_trades}次")
        print(f"  胜率: {win_rate:.1f}%")
        print(f"  当前持仓: {len(positions)}只")

        if positions:
            print(f"\n【持仓明细】")
            for pos in positions.values():
                print(f"  {pos.stock_name}: {pos.shares}股 @ ¥{pos.cost_price:.2f}")

        # 保存回测结果
        result_file = os.path.join(LOGS_DIR, f'backtest_{datetime.now().strftime("%Y%m%d")}.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✓ 回测结果已保存: {result_file}")

        return result

    def _predict_up(self, df: pd.DataFrame) -> Optional[float]:
        """预测上涨概率"""
        if not self.model or not FEATURE_ENGINEER_AVAILABLE:
            return None
        try:
            features = EnhancedFeatureEngineer.calculate_features(df)
            if features.iloc[-1].isna().any():
                return None
            return self.model.predict_proba([features.iloc[-1].values])[0][1]
        except:
            return None

    # ==================== 实时监控 ====================

    def monitor(self) -> List[Dict]:
        """
        实时监控并生成交易信号

        Returns:
            交易信号列表
        """
        print("\n" + "=" * 70)
        print(f"实时监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        signals = []

        # 获取当前状态
        cash = self.get_current_cash()
        positions = self.get_current_positions()

        print(f"\n【账户状态】")
        print(f"  现金: ¥{cash:,.0f}")
        print(f"  持仓: {len(positions)}只")

        conn = self._get_conn()
        self.stock_names = self._load_stock_names()

        # 更新持仓价格
        for symbol, pos in positions.items():
            df = pd.read_sql_query(
                'SELECT close FROM kline_30m WHERE symbol=? ORDER BY date DESC LIMIT 1',
                conn, params=(symbol,)
            )
            if not df.empty:
                pos.current_price = float(df['close'].iloc[0])
                self.save_position(pos)

        # 检查卖出信号
        for symbol, pos in positions.items():
            df = pd.read_sql_query(
                'SELECT * FROM kline_30m WHERE symbol=? ORDER BY date',
                conn, params=(symbol,)
            )
            if len(df) < 100:
                continue

            current_price = float(df['close'].iloc[-1])
            pos.current_price = current_price
            profit_pct = (current_price - pos.cost_price) / pos.cost_price

            reason = None
            up_prob = 0

            # 止损止盈
            if profit_pct <= -self.stop_loss_pct:
                reason = f"触发止损(亏损{abs(profit_pct)*100:.1f}%)"
            elif profit_pct >= self.take_profit_pct:
                reason = f"触发止盈(盈利{profit_pct*100:.1f}%)"
            else:
                up_prob = self._predict_up(df) or 0
                if up_prob < self.sell_threshold:
                    reason = f"模型看跌(上涨概率{up_prob:.0%})"

            if reason:
                signals.append({
                    'symbol': symbol,
                    'stock_name': pos.stock_name,
                    'action': '卖出',
                    'shares': pos.shares,
                    'price': current_price,
                    'reason': reason,
                    'up_prob': up_prob,
                    'profit': pos.profit,
                    'profit_pct': pos.profit_pct
                })

        # 检查买入信号
        if len(positions) < self.max_positions:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT symbol FROM kline_30m GROUP BY symbol
                HAVING COUNT(*) >= 500 ORDER BY COUNT(*) DESC LIMIT 30
            ''')
            candidates = [row[0] for row in cursor.fetchall()]

            for symbol in candidates:
                if symbol in positions:
                    continue

                df = pd.read_sql_query(
                    'SELECT * FROM kline_30m WHERE symbol=? ORDER BY date',
                    conn, params=(symbol,)
                )
                if len(df) < 100:
                    continue

                up_prob = self._predict_up(df)
                if up_prob and up_prob >= self.buy_threshold:
                    current_price = float(df['close'].iloc[-1])
                    max_invest = min(cash * 0.9, self.initial_capital * self.position_pct)
                    shares = int(max_invest / current_price / 100) * 100

                    if shares >= 100:
                        stock_name = self.stock_names.get(symbol, symbol)
                        signals.append({
                            'symbol': symbol,
                            'stock_name': stock_name,
                            'action': '买入',
                            'shares': shares,
                            'price': current_price,
                            'reason': f"模型看涨(上涨概率{up_prob:.0%})",
                            'up_prob': up_prob,
                            'stop_loss': current_price * (1 - self.stop_loss_pct),
                            'take_profit': current_price * (1 + self.take_profit_pct)
                        })

        # 打印信号
        if signals:
            print(f"\n【交易信号】发现 {len(signals)} 个信号")
            for s in signals:
                emoji = "🟢" if s['action'] == '买入' else "🔴"
                print(f"  {emoji} {s['stock_name']}: {s['action']} {s['shares']}股 @ ¥{s['price']:.2f}")
                print(f"     原因: {s['reason']}")
        else:
            print(f"\n【交易信号】无信号")

        # 获取关注股票价格
        watchlist_prices = {}
        for stock in self.watchlist:
            df = pd.read_sql_query(
                'SELECT close FROM kline_30m WHERE symbol=? ORDER BY date DESC LIMIT 1',
                conn, params=(stock['symbol'],)
            )
            if not df.empty:
                watchlist_prices[stock['symbol']] = float(df['close'].iloc[0])

        conn.close()

        # 打印关注股票
        if watchlist_prices:
            print(f"\n【关注股票】")
            for stock in self.watchlist:
                price = watchlist_prices.get(stock['symbol'])
                if price:
                    print(f"  👀 {stock['name']}: ¥{price:.2f}")

        # 发送邮件（即使无信号也发送持仓汇总）
        if self.email_notifier:
            self._send_email(signals, positions, cash, watchlist_prices)

        # 保存结果
        self._save_monitor_result(signals, positions, cash)

        return signals

    def _send_email(self, signals: List[Dict], positions: Dict[str, Position], cash: float, watchlist_prices: Dict[str, float] = None):
        """发送邮件（HTML表格格式）"""
        total_value = cash + sum(p.market_value for p in positions.values())
        total_profit = total_value - self.initial_capital

        if signals:
            subject = f"【交易信号】{datetime.now().strftime('%m-%d %H:%M')} - {len(signals)}个信号"
        else:
            subject = f"【持仓汇总】{datetime.now().strftime('%m-%d %H:%M')} - 盈亏¥{total_profit:,.0f}"

        # 构建HTML邮件
        html = self._build_html_email(signals, positions, cash, total_value, total_profit, watchlist_prices)

        # 纯文本备用
        text_content = self._build_text_email(signals, positions, cash, total_value, total_profit, watchlist_prices)

        self.email_notifier.send(subject, text_content, html)
        print("✓ 邮件已发送")

    def _build_html_email(self, signals, positions, cash, total_value, total_profit, watchlist_prices):
        """构建HTML邮件"""
        rows = []
        for pos in sorted(positions.values(), key=lambda x: x.profit, reverse=True):
            color = "green" if pos.profit > 0 else "red"
            rows.append(f"""
                <tr>
                    <td>{pos.stock_name}</td>
                    <td>{pos.symbol}</td>
                    <td style="text-align:right">{pos.shares:,}</td>
                    <td style="text-align:right">¥{pos.cost_price:.2f}</td>
                    <td style="text-align:right">¥{pos.current_price:.2f}</td>
                    <td style="text-align:right;color:{color}">¥{pos.profit:,.0f}</td>
                    <td style="text-align:right;color:{color}">{pos.profit_pct:+.1f}%</td>
                    <td style="text-align:right">¥{pos.cost_price*(1-self.stop_loss_pct):.2f}</td>
                    <td style="text-align:right">¥{pos.cost_price*(1+self.take_profit_pct):.2f}</td>
                </tr>
            """)

        # 关注股票
        watchlist_rows = ""
        if watchlist_prices:
            for symbol, price in watchlist_prices.items():
                name = next((w['name'] for w in self.watchlist if w['symbol'] == symbol), symbol)
                watchlist_rows += f"""
                <tr>
                    <td>{name}</td>
                    <td>{symbol}</td>
                    <td style="text-align:right">¥{price:.2f}</td>
                    <td style="text-align:center" colspan="4">关注中</td>
                </tr>
                """

        # 交易信号
        signal_rows = ""
        for s in signals:
            action_color = "red" if s['action'] == '买入' else "green"
            signal_rows += f"""
                <tr>
                    <td style="color:{action_color};font-weight:bold">{s['action']}</td>
                    <td>{s['stock_name']}</td>
                    <td style="text-align:right">{s['shares']:,}股</td>
                    <td style="text-align:right">¥{s['price']:.2f}</td>
                    <td colspan="2">{s['reason']}</td>
                </tr>
            """

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; padding: 20px; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 10px 10px 0 0; }}
        .header h1 {{ margin: 0; font-size: 24px; }}
        .header p {{ margin: 5px 0 0 0; opacity: 0.9; }}
        .content {{ padding: 20px; }}
        .summary {{ display: flex; justify-content: space-around; padding: 20px; background: #f8f9fa; border-radius: 8px; margin-bottom: 20px; }}
        .summary-item {{ text-align: center; }}
        .summary-item .value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .summary-item .label {{ font-size: 12px; color: #666; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th {{ background: #667eea; color: white; padding: 12px 8px; text-align: left; font-size: 13px; }}
        td {{ padding: 10px 8px; border-bottom: 1px solid #eee; font-size: 13px; }}
        tr:hover {{ background: #f8f9fa; }}
        .profit {{ color: green; }}
        .loss {{ color: red; }}
        .section-title {{ font-size: 16px; font-weight: bold; margin: 20px 0 10px 0; padding-bottom: 5px; border-bottom: 2px solid #667eea; }}
        .footer {{ padding: 15px; background: #f8f9fa; border-radius: 0 0 10px 10px; font-size: 12px; color: #666; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 LGBM交易监控</h1>
            <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="content">
            <div class="summary">
                <div class="summary-item">
                    <div class="value">¥{total_value:,.0f}</div>
                    <div class="label">总市值</div>
                </div>
                <div class="summary-item">
                    <div class="value">¥{cash:,.0f}</div>
                    <div class="label">现金</div>
                </div>
                <div class="summary-item">
                    <div class="value {'profit' if total_profit > 0 else 'loss'}">¥{total_profit:+,.0f}</div>
                    <div class="label">总盈亏</div>
                </div>
                <div class="summary-item">
                    <div class="value">{len(positions)}只</div>
                    <div class="label">持仓数</div>
                </div>
            </div>

            <div class="section-title">📈 持仓明细</div>
            <table>
                <tr>
                    <th>股票</th>
                    <th>代码</th>
                    <th>持股</th>
                    <th>成本</th>
                    <th>现价</th>
                    <th>盈亏</th>
                    <th>幅度</th>
                    <th>止损</th>
                    <th>止盈</th>
                </tr>
                {"".join(rows) if rows else "<tr><td colspan='9' style='text-align:center'>空仓</td></tr>"}
            </table>

            {"<div class='section-title'>👀 关注股票</div><table><tr><th>股票</th><th>代码</th><th>现价</th><th style='text-align:center' colspan='4'>状态</th></tr>" + watchlist_rows + "</table>" if watchlist_rows else ""}

            {"<div class='section-title'>🔔 交易信号</div><table><tr><th>操作</th><th>股票</th><th>数量</th><th>价格</th><th colspan='2'>原因</th></tr>" + signal_rows + "</table>" if signal_rows else "<div class='section-title'>🔔 交易信号</div><p style='color:#666'>当前无操作信号，继续持有</p>"}
        </div>

        <div class="footer">
            策略参数: 止损8% | 止盈10% | 买入>60%概率 | 卖出<40%概率<br>
            此邮件由系统自动发送，仅供参考，不构成投资建议
        </div>
    </div>
</body>
</html>
"""
        return html

    def _build_text_email(self, signals, positions, cash, total_value, total_profit, watchlist_prices):
        """构建纯文本邮件"""
        lines = []
        lines.append("=" * 70)
        lines.append(f"LGBM交易监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append(f"\n总市值: ¥{total_value:,.0f} | 现金: ¥{cash:,.0f} | 盈亏: ¥{total_profit:+,.0f}")
        lines.append("\n" + "-" * 70)
        lines.append("持仓明细:")
        for pos in positions.values():
            status = "✅" if pos.profit > 0 else "❌"
            lines.append(f"  {status} {pos.stock_name}({pos.symbol}): {pos.shares}股 @ ¥{pos.cost_price:.2f} → ¥{pos.current_price:.2f} | 盈亏 ¥{pos.profit:+,.0f} ({pos.profit_pct:+.1f}%)")
        if watchlist_prices:
            lines.append("\n关注股票:")
            for symbol, price in watchlist_prices.items():
                name = next((w['name'] for w in self.watchlist if w['symbol'] == symbol), symbol)
                lines.append(f"  👀 {name}({symbol}): ¥{price:.2f}")
        if signals:
            lines.append("\n交易信号:")
            for s in signals:
                lines.append(f"  {'🟢' if s['action']=='买入' else '🔴'} {s['action']}: {s['stock_name']} {s['shares']}股 @ ¥{s['price']:.2f} - {s['reason']}")
        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

    def _save_monitor_result(self, signals: List[Dict], positions: Dict[str, Position], cash: float):
        """保存监控结果"""
        os.makedirs(LOGS_DIR, exist_ok=True)
        result = {
            'timestamp': datetime.now().isoformat(),
            'signals': signals,
            'account': {
                'cash': cash,
                'positions': {s: asdict(p) for s, p in positions.items()},
                'total_value': cash + sum(p.market_value for p in positions.values())
            }
        }
        result_file = os.path.join(LOGS_DIR, f'monitor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✓ 结果已保存: {result_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='完整交易监控系统')
    parser.add_argument('--backtest', action='store_true', help='回测模拟(从一个月前)')
    parser.add_argument('--update', action='store_true', help='仅更新数据')
    parser.add_argument('--monitor', action='store_true', help='仅执行监控')
    parser.add_argument('--days', type=int, default=30, help='回测天数')
    parser.add_argument('--capital', type=float, default=100000, help='初始资金')
    args = parser.parse_args()

    monitor = FullMonitor(initial_capital=args.capital)

    if args.backtest:
        monitor.backtest_simulate(days=args.days)
    elif args.update:
        monitor.update_data()
    elif args.monitor:
        monitor.monitor()
    else:
        # 完整流程：更新数据 -> 回测 -> 监控
        monitor.update_data()
        monitor.backtest_simulate(days=args.days)
        monitor.monitor()


if __name__ == "__main__":
    main()