#!/usr/bin/env python3
"""
v8 回归模型回测 — 基于截面排序的策略

核心改进:
1. 分类→回归: 预测每只股票的未来收益率
2. 截面排序: 每30分钟对所有股票按预测收益率排序
3. 买Top N: 买入预测收益率最高的N只股票
4. 卖Bottom N: 卖出持仓中排名跌出Top N的股票
5. 交易成本: 含手续费(0.03%) + 滑点(0.1%)
6. T+1: 当天买入的股票次日才能卖出

业界对标: WorldQuant/Citadel 的截面多空策略
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'strategy'))

from train_lgb_enhanced import EnhancedFeatureEngineer
from train_lgb_v8_regression import prepare_training_data_regression

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ====== 交易成本 ======
COMMISSION_RATE = 0.0003   # 手续费 0.03%
SLIPPAGE_RATE = 0.001      # 滑点 0.1% (A股流动性尚可)
TOTAL_COST_PER_SIDE = COMMISSION_RATE + SLIPPAGE_RATE  # 0.13%
TOTAL_COST_ROUND_TRIP = TOTAL_COST_PER_SIDE * 2        # 0.26% 买卖双边


@dataclass
class Position:
    symbol: str
    stock_name: str
    entry_time: datetime
    entry_price: float
    shares: int
    available: bool = True  # T+1: 当天买入的不可卖


@dataclass
class Trade:
    symbol: str
    stock_name: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    shares: int
    return_pct: float
    return_net_pct: float  # 扣除成本后


class CrossSectionalBacktester:
    """截面排序回测"""

    def __init__(
        self,
        initial_capital: float = 500000,
        model_path: str = None,
        top_n: int = 5,
        max_position_pct: float = 0.20,
        min_hold_minutes: int = 30,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.top_n = top_n
        self.max_position_pct = max_position_pct
        self.min_hold_minutes = min_hold_minutes

        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_values: List[Dict] = []
        self.entry_dates: Dict[str, datetime] = {}  # 买入日期(T+1用)

        # 加载模型
        self.model = None
        self.feature_names = None
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

        # 特征缓存
        self.features_cache: Dict[str, pd.DataFrame] = {}
        self.time_index_map: Dict[str, Dict[datetime, int]] = {}

    def _load_model(self, model_path: str):
        """加载模型"""
        with open(model_path, 'rb') as f:
            self.model_data = pickle.load(f)

        self.model = self.model_data.get('model')
        self.feature_names = self.model_data.get('feature_names')
        model_type = self.model_data.get('model_type', 'classification')
        version = self.model_data.get('model_version', 'unknown')

        logger.info(f"模型加载: {version} ({model_type})")
        if model_type == 'regression':
            logger.info(f"  Spearman={self.model_data.get('cv_spearman', 0):.4f}")
        else:
            logger.info(f"  准确率={self.model_data.get('cv_accuracy', 0):.2%}")

    def load_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """从数据库加载股票数据"""
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/stock_data.db')
        if not os.path.exists(db_path):
            return None

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol=? ORDER BY date ASC',
            (symbol,)
        )
        rows = cursor.fetchall()
        conn.close()

        if rows:
            df = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            return df
        return None

    def preload_features(self, all_data: Dict[str, pd.DataFrame]):
        """预计算所有股票的特征"""
        logger.info("预计算特征...")
        for symbol, df in all_data.items():
            try:
                features = EnhancedFeatureEngineer.calculate_features(df)
                features = features.fillna(0)

                if self.feature_names:
                    missing = [c for c in self.feature_names if c not in features.columns]
                    for c in missing:
                        features[c] = 0
                    features = features[self.feature_names]

                self.features_cache[symbol] = features
                self.time_index_map[symbol] = {row['date']: idx for idx, row in df.iterrows()}
            except Exception as e:
                logger.warning(f"特征计算失败 {symbol}: {e}")
        logger.info(f"特征预计算完成，共 {len(self.features_cache)} 只股票")

    def _get_stock_idx(self, symbol: str, time: datetime) -> Optional[int]:
        """获取股票在指定时间的局部索引"""
        time_map = self.time_index_map.get(symbol)
        if time_map is None:
            return None
        return time_map.get(time)

    def _get_prediction(self, symbol: str, time: datetime) -> Optional[float]:
        """获取股票在指定时间的预测收益率"""
        idx = self._get_stock_idx(symbol, time)
        if idx is None:
            return None

        features = self.features_cache.get(symbol)
        if features is None or idx >= len(features):
            return None

        try:
            feat_row = features.iloc[idx].fillna(0).values
            if self.model_data.get('model_type') == 'regression':
                return float(self.model.predict([feat_row])[0])
            else:
                # 分类模型: 返回涨跌概率(0.5中心化)
                prob = float(self.model.predict_proba([feat_row])[0][1])
                return (prob - 0.5) * 2  # 映射到 [-1, 1]
        except Exception:
            return None

    def run_backtest(self, stocks: List[Dict], start_date: str = None, end_date: str = None):
        """执行截面排序回测"""
        logger.info("=" * 70)
        logger.info("v8 截面排序回测")
        logger.info("=" * 70)
        logger.info(f"初始资金: {self.initial_capital:,.0f} 元")
        logger.info(f"Top N: {self.top_n} 只")
        logger.info(f"最大仓位: {self.max_position_pct:.0%}")
        logger.info(f"交易成本(单边): {TOTAL_COST_PER_SIDE:.2%}")
        logger.info("=" * 70)

        # 加载数据
        all_data = {}
        for stock in stocks:
            symbol = stock['symbol']
            df = self.load_data(symbol)
            if df is not None and len(df) >= 150:
                if start_date:
                    start_dt = pd.to_datetime(start_date)
                    df = df[df['date'] >= start_dt]
                if end_date:
                    end_dt = pd.to_datetime(end_date)
                    df = df[df['date'] <= end_dt]
                if len(df) >= 50:
                    df = df.reset_index(drop=True)
                    all_data[symbol] = df
                    logger.info(f"  ✓ {stock['name']} ({symbol}): {len(df)} 条")

        if not all_data:
            logger.error("无有效数据")
            return

        # 预计算特征
        self.preload_features(all_data)

        # 获取所有时间点
        all_times = sorted(set(
            row['date'] for df in all_data.values() for row in df.to_dict('records')
        ))

        # 过滤掉数据不足的时间点
        min_stocks_needed = self.top_n * 2
        valid_times = []
        for t in all_times:
            available = sum(1 for s in all_data if self._get_stock_idx(s, t) is not None)
            if available >= min_stocks_needed:
                valid_times.append(t)

        logger.info(f"\n共 {len(valid_times)} 个有效时间点 (需≥{min_stocks_needed}只股票)")
        logger.info("开始回测...\n")

        # 遍历时间点
        for i, current_time in enumerate(valid_times):
            if i % 100 == 0:
                total_value = self._calc_total_value(all_data, current_time)
                logger.info(f"[{i}/{len(valid_times)}] {current_time.strftime('%m-%d %H:%M')} | "
                           f"市值:{total_value:,.0f} | 现金:{self.cash:,.0f} | 持仓:{len(self.positions)}只")

            self._update_availability(current_time)
            self._execute_sells(all_data, current_time)
            self._execute_buys(all_data, current_time, stocks)

            total_value = self._calc_total_value(all_data, current_time)
            self.daily_values.append({
                'time': current_time,
                'value': total_value,
                'cash': self.cash,
                'positions': len(self.positions)
            })

        self._print_results()

    def _update_availability(self, current_time: datetime):
        """更新T+1可用性"""
        for symbol, pos in self.positions.items():
            if not pos.available:
                entry_date = self.entry_dates.get(symbol)
                if entry_date and current_time.date() > entry_date.date():
                    pos.available = True

    def _execute_sells(self, all_data: Dict, current_time: datetime):
        """卖出逻辑: 持仓中预测收益率排名跌出Top N+Margin的股票"""
        margin = self.top_n  # 卖出宽松度: 跌出Top 2N才卖
        to_sell = []

        for symbol, pos in list(self.positions.items()):
            if not pos.available:
                continue

            # 检查最小持仓时间
            hold_minutes = (current_time - pos.entry_time).total_seconds() / 60
            if hold_minutes < self.min_hold_minutes:
                continue

            prediction = self._get_prediction(symbol, current_time)
            if prediction is None:
                continue

            # 获取当前所有股票的排名
            all_predictions = {}
            for s in all_data:
                if s != symbol:
                    pred = self._get_prediction(s, current_time)
                    if pred is not None:
                        all_predictions[s] = pred

            all_predictions[symbol] = prediction
            rankings = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
            rank = next(i for i, (s, _) in enumerate(rankings) if s == symbol)

            if rank >= self.top_n + margin:
                to_sell.append(symbol)

        for symbol in to_sell:
            df = all_data[symbol]
            idx = self._get_stock_idx(symbol, current_time)
            if idx is None:
                continue

            sell_price = float(df.iloc[idx]['close'])
            self._sell_position(symbol, current_time, sell_price)

    def _execute_buys(self, all_data: Dict, current_time: datetime, stocks: List[Dict]):
        """买入逻辑: 对所有股票预测收益率, 买Top N"""
        # 获取所有股票的预测收益率
        predictions = {}
        for symbol in all_data:
            # 跳过已持仓的
            if symbol in self.positions:
                continue
            pred = self._get_prediction(symbol, current_time)
            if pred is not None:
                predictions[symbol] = pred

        if len(predictions) < self.top_n:
            return

        # 按预测收益率排序
        rankings = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

        # 买Top N
        n_to_buy = self.top_n - len(self.positions)
        if n_to_buy <= 0:
            return

        stock_info = {s['symbol']: s for s in stocks}

        for symbol, pred_return in rankings[:n_to_buy]:
            df = all_data[symbol]
            idx = self._get_stock_idx(symbol, current_time)
            if idx is None:
                continue

            buy_price = float(df.iloc[idx]['close'])
            # 含滑点的实际买入价
            actual_price = buy_price * (1 + SLIPPAGE_RATE)

            # 仓位计算
            position_value = self.cash * self.max_position_pct
            shares = int(position_value / actual_price / 100) * 100  # 100股整数倍

            if shares < 100:
                continue

            cost = shares * actual_price * (1 + COMMISSION_RATE)
            if cost > self.cash:
                shares = int(self.cash / (actual_price * (1 + COMMISSION_RATE)) / 100) * 100
                if shares < 100:
                    continue
                cost = shares * actual_price * (1 + COMMISSION_RATE)

            name = stock_info.get(symbol, {}).get('name', symbol)
            self._buy_position(symbol, name, current_time, actual_price, shares, cost)

    def _buy_position(self, symbol: str, name: str, time: datetime, price: float, shares: int, cost: float):
        """买入开仓"""
        self.cash -= cost
        self.positions[symbol] = Position(
            symbol=symbol,
            stock_name=name,
            entry_time=time,
            entry_price=price,
            shares=shares,
            available=False,
        )
        self.entry_dates[symbol] = time

    def _sell_position(self, symbol: str, time: datetime, price: float):
        """卖出平仓"""
        pos = self.positions.pop(symbol)
        self.entry_dates.pop(symbol, None)

        actual_price = price * (1 - SLIPPAGE_RATE)
        revenue = pos.shares * actual_price * (1 - COMMISSION_RATE)
        self.cash += revenue

        gross_return = (actual_price - pos.entry_price) / pos.entry_price
        net_return = (revenue - pos.shares * pos.entry_price * (1 + COMMISSION_RATE)) / (pos.shares * pos.entry_price * (1 + COMMISSION_RATE))

        self.trades.append(Trade(
            symbol=symbol,
            stock_name=pos.stock_name,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            exit_time=time,
            exit_price=actual_price,
            shares=pos.shares,
            return_pct=gross_return * 100,
            return_net_pct=net_return * 100,
        ))

    def _calc_total_value(self, all_data: Dict, current_time: datetime) -> float:
        """计算总市值"""
        total = self.cash
        for symbol, pos in self.positions.items():
            df = all_data.get(symbol)
            if df is None:
                continue
            idx = self._get_stock_idx(symbol, current_time)
            if idx is None:
                continue
            price = float(df.iloc[idx]['close'])
            total += pos.shares * price
        return total

    def _print_results(self):
        """打印回测结果"""
        if not self.trades:
            logger.warning("无交易记录")
            return

        final_value = self.daily_values[-1]['value'] if self.daily_values else self.initial_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital

        returns = [t.return_net_pct for t in self.trades]
        win_rate = sum(1 for r in returns if r > 0) / len(returns) if returns else 0

        # 计算夏普比率
        if len(self.daily_values) > 1:
            daily_returns = []
            for i in range(1, len(self.daily_values)):
                r = (self.daily_values[i]['value'] - self.daily_values[i-1]['value']) / self.daily_values[i-1]['value']
                daily_returns.append(r)
            if daily_returns and np.std(daily_returns) > 0:
                sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252 * 8)  # 每天8根30分钟K线
            else:
                sharpe = 0
        else:
            sharpe = 0

        # 最大回撤
        if self.daily_values:
            values = [d['value'] for d in self.daily_values]
            peak = np.maximum.accumulate(values)
            drawdown = (values - peak) / peak
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0

        logger.info("\n" + "=" * 70)
        logger.info("回测结果")
        logger.info("=" * 70)
        logger.info(f"初始资金: {self.initial_capital:,.0f} 元")
        logger.info(f"最终资金: {final_value:,.0f} 元")
        logger.info(f"总收益率: {total_return:.2%}")
        logger.info(f"交易次数: {len(self.trades)}")
        logger.info(f"胜率: {win_rate:.1%}")
        logger.info(f"平均净收益: {np.mean(returns):.2f}%")
        logger.info(f"夏普比率: {sharpe:.2f}")
        logger.info(f"最大回撤: {max_drawdown:.2%}")
        logger.info(f"交易成本(双边): {TOTAL_COST_ROUND_TRIP:.2%}")

        # 盈亏分布
        profit_trades = [r for r in returns if r > 0]
        loss_trades = [r for r in returns if r < 0]
        if profit_trades:
            logger.info(f"平均盈利: {np.mean(profit_trades):.2f}%")
        if loss_trades:
            logger.info(f"平均亏损: {np.mean(loss_trades):.2f}%")
        if profit_trades and loss_trades:
            logger.info(f"盈亏比: {abs(np.mean(profit_trades) / np.mean(loss_trades)):.2f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='v8 截面排序回测')
    parser.add_argument('--model', type=str, default=None, help='模型路径')
    parser.add_argument('--capital', type=float, default=500000, help='初始资金')
    parser.add_argument('--top-n', type=int, default=5, help='持仓数量')
    parser.add_argument('--start', type=str, default=None, help='开始日期 YYYY-MM-DD')
    parser.add_argument('--end', type=str, default=None, help='结束日期 YYYY-MM-DD')
    parser.add_argument('--min-hold', type=int, default=30, help='最小持仓分钟数')
    args = parser.parse_args()

    # 默认模型路径
    model_path = args.model
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  'models/lgb_hs300/model.pkl')

    if not os.path.exists(model_path):
        logger.error(f"模型不存在: {model_path}")
        return

    # 获取股票列表
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/stock_data.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM kline_30m ORDER BY symbol")
    symbols = [row[0] for row in cursor.fetchall()]

    # 获取股票名称
    stocks = []
    for sym in symbols:
        cursor.execute("SELECT name FROM stock_info WHERE symbol=?", (sym,))
        row = cursor.fetchone()
        name = row[0] if row and row[0] else sym
        stocks.append({'symbol': sym, 'name': name})
    conn.close()

    logger.info(f"加载 {len(stocks)} 只股票")

    bt = CrossSectionalBacktester(
        initial_capital=args.capital,
        model_path=model_path,
        top_n=args.top_n,
        min_hold_minutes=args.min_hold,
    )

    bt.run_backtest(stocks, start_date=args.start, end_date=args.end)


if __name__ == '__main__':
    main()