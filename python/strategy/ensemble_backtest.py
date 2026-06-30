#!/usr/bin/env python3
"""
联合回测引擎 v1 — 日线选股 + 分钟级择时

架构:
  日线模型(α) → 每日盘后/盘前选出Top N股票池
  分钟级模型(γ) → 盘中实时打分, 触发买卖信号

回测流程:
  1. 每日盘前: 日线模型选出Top K股票
  2. 盘中: 每根30分钟K线, 分钟级模型对pool内股票打分
  3. 信号: 评分 > 买入阈值 → 买入; 评分 < 卖出阈值 → 卖出
  4. 持仓管理: 最多持有M只, 单只仓位上限
  5. 风控: 止损、止盈、最大回撤

评估指标:
  - 年化收益率
  - 夏普比率
  - 最大回撤
  - 胜率、盈亏比
  - 换手率
  - 超额收益 (vs 沪深300)

用法:
  python strategy/ensemble_backtest.py
  python strategy/ensemble_backtest.py --pool-size 50 --max-hold 5
  python strategy/ensemble_backtest.py --no-daily  # 纯分钟级 (无日线筛选)
"""

import sys, os, argparse, pickle, json, sqlite3, warnings
from collections import defaultdict

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kw): return iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategy.intraday_features import IntradayFeaturePipeline

# ============ 路径 ============
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data/stock_data.db')
DAILY_MODEL_PATH = os.path.join(ROOT, 'models/lgb_daily/model.pkl')
INTRADAY_MODEL_PATH = os.path.join(ROOT, 'models/lgb_intraday/model.pkl')
RESULT_DIR = os.path.join(ROOT, 'results')

# ============ 回测配置 ============
INITIAL_CAPITAL = 1_000_000   # 初始资金 100万
MAX_POSITIONS = 5             # 最多持仓数
POSITION_PCT = 0.20           # 单只仓位 20%
COMMISSION = 0.00025          # 手续费 万2.5
SLIPPAGE = 0.001              # 滑点 0.1%
STOP_LOSS = -0.05             # 止损 -5%
TAKE_PROFIT = 0.15            # 止盈 +15%
DAILY_POOL_SIZE = 50          # 日线模型选股数
BUY_THRESHOLD = 0.003         # 分钟级买入阈值 (预测收益 > 0.3%)
SELL_THRESHOLD = -0.003       # 分钟级卖出阈值 (预测收益 < -0.3%)
MIN_HOLD_BARS = 3             # 最小持仓K线数 (避免频繁交易)
MIN_CASH_RATIO = 0.05         # 最低现金比例


@dataclass
class Position:
    """持仓"""
    symbol: str
    buy_date: str           # 买入时间
    buy_price: float
    shares: int
    cost: float             # 总成本 (含手续费)
    hold_bars: int = 0      # 已持仓K线数
    max_profit: float = 0   # 最大浮盈


@dataclass
class Trade:
    """交易记录"""
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    reason: str  # 'signal', 'stop_loss', 'take_profit', 'end'


@dataclass
class BacktestResult:
    """回测结果"""
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Tuple[str, float]] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)

    # 统计
    total_return: float = 0
    annual_return: float = 0
    sharpe_ratio: float = 0
    max_drawdown: float = 0
    win_rate: float = 0
    profit_factor: float = 0
    total_trades: int = 0
    turnover: float = 0


class EnsembleBacktest:
    """联合回测引擎"""

    def __init__(self,
                 daily_model_path: str = None,
                 intraday_model_path: str = None,
                 db_path: str = DB_PATH,
                 initial_capital: float = INITIAL_CAPITAL,
                 max_positions: int = MAX_POSITIONS,
                 position_pct: float = POSITION_PCT,
                 commission: float = COMMISSION,
                 slippage: float = SLIPPAGE,
                 stop_loss: float = STOP_LOSS,
                 take_profit: float = TAKE_PROFIT,
                 daily_pool_size: int = DAILY_POOL_SIZE,
                 buy_threshold: float = BUY_THRESHOLD,
                 sell_threshold: float = SELL_THRESHOLD,
                 min_hold_bars: int = MIN_HOLD_BARS,
                 use_daily_model: bool = True):
        self.daily_model_path = daily_model_path
        self.intraday_model_path = intraday_model_path
        self.db_path = db_path
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.position_pct = position_pct
        self.commission = commission
        self.slippage = slippage
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.daily_pool_size = daily_pool_size
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_hold_bars = min_hold_bars
        self.use_daily_model = use_daily_model

        # 加载模型
        self.daily_model = None
        self.intraday_models = []  # Ensemble: list of LGBM models
        self.intraday_feature_names = None
        self._load_models()

        # 运行时状态
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[str, float]] = []
        self.daily_returns: List[float] = []
        self.daily_pool: List[str] = []  # 当日股票池

        # 数据缓存
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._feature_pipeline = IntradayFeaturePipeline()

    def _load_models(self):
        """加载日线和分钟级模型"""
        # 日线模型
        if self.use_daily_model and self.daily_model_path and os.path.exists(self.daily_model_path):
            try:
                with open(self.daily_model_path, 'rb') as f:
                    pkg = pickle.load(f)
                self.daily_model = pkg
                print(f"✅ 日线模型已加载: {pkg.get('train_date', 'unknown')} "
                      f"({pkg.get('n_models', 0)}模型, {pkg.get('n_features', 0)}特征)")
            except Exception as e:
                print(f"⚠️ 日线模型加载失败: {e}")

        # 分钟级模型 (支持单个模型和 Ensemble 两种格式)
        if self.intraday_model_path and os.path.exists(self.intraday_model_path):
            try:
                with open(self.intraday_model_path, 'rb') as f:
                    pkg = pickle.load(f)
                self.intraday_feature_names = pkg['feature_names']

                # 兼容新旧格式
                if 'models' in pkg:
                    self.intraday_models = pkg['models']
                    n_models = len(self.intraday_models)
                    n_trees = sum(m.best_iteration_ or 100 for m in self.intraday_models)
                elif 'model' in pkg:
                    self.intraday_models = [pkg['model']]
                    n_models = 1
                    n_trees = pkg.get('n_trees', 0)
                else:
                    raise ValueError("模型文件格式不正确")

                print(f"✅ 分钟级模型已加载: {pkg.get('train_date', 'unknown')} "
                      f"({n_models}模型, {n_trees}棵树, {pkg.get('n_features', len(self.intraday_feature_names))}特征)")
            except Exception as e:
                print(f"⚠️ 分钟级模型加载失败: {e}")
                raise

    def get_daily_pool(self, date_str: str) -> List[str]:
        """获取日线模型选出的股票池"""
        if not self.use_daily_model or self.daily_model is None:
            # 无日线模型: 用全部股票
            conn = sqlite3.connect(self.db_path)
            symbols = [r[0] for r in conn.execute(
                "SELECT DISTINCT symbol FROM kline_30m")]
            conn.close()
            return symbols[:self.daily_pool_size] if self.daily_pool_size > 0 else symbols

        # 简化版: 由于日线模型需要完整的特征计算, 这里用占位逻辑
        # 实际使用时需要: 1) 加载该日期的日线特征 2) 运行日线模型预测 3) 选Top N
        # 目前: 返回全部股票 (后续可补充完整日线预测)
        conn = sqlite3.connect(self.db_path)
        symbols = [r[0] for r in conn.execute(
            "SELECT DISTINCT symbol FROM kline_30m")]
        conn.close()
        return symbols[:self.daily_pool_size] if self.daily_pool_size > 0 else symbols

    def load_intraday_data(self, symbol: str) -> pd.DataFrame:
        """加载单只股票的分钟级数据"""
        if symbol in self._data_cache:
            return self._data_cache[symbol]

        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(
            f"SELECT * FROM kline_30m WHERE symbol=? ORDER BY date", conn, params=(symbol,))
        conn.close()

        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'], format='mixed')
            df = df.sort_values('date').reset_index(drop=True)
            df = df.drop_duplicates(subset=['date']).reset_index(drop=True)
            self._data_cache[symbol] = df

        return df

    def predict_intraday(self, symbol: str, timestamp: pd.Timestamp) -> Optional[float]:
        """分钟级模型预测"""
        df = self.load_intraday_data(symbol)
        if df is None or len(df) < 50:
            return None

        # 找到当前时间戳之前的数据
        mask = df['date'] <= timestamp
        if mask.sum() < 50:
            return None

        hist = df[mask].tail(200)  # 最近200根K线

        # 计算特征
        try:
            feats = self._feature_pipeline.compute_stock(hist, symbol)
            feats = feats.ffill().fillna(0)

            # 对齐到训练时的特征列
            if self.intraday_feature_names:
                feats = feats.reindex(columns=self.intraday_feature_names, fill_value=0)

            # 预测 (Ensemble平均)
            latest = feats.iloc[-1:].values.astype(np.float32)
            preds = [m.predict(latest)[0] for m in self.intraday_models]
            pred = np.mean(preds)
            return float(pred)
        except Exception:
            return None

    def run(self, start_date: str = None, end_date: str = None) -> BacktestResult:
        """运行回测

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        """
        print(f"\n{'='*60}")
        print(f" 联合回测引擎 — 日线选股 + 分钟级择时")
        print(f"{'='*60}")
        print(f"  初始资金: ¥{self.initial_capital:,.0f}")
        print(f"  最大持仓: {self.max_positions}只 | 单只仓位: {self.position_pct:.0%}")
        print(f"  手续费: {self.commission:.4%} | 滑点: {self.slippage:.1%}")
        print(f"  止损: {self.stop_loss:.1%} | 止盈: {self.take_profit:.1%}")
        print(f"  日线池: {self.daily_pool_size}只 | 买入阈值: {self.buy_threshold:.3%}")

        # 获取所有交易日
        conn = sqlite3.connect(self.db_path)
        all_dates = pd.read_sql(
            "SELECT DISTINCT DATE(date) as d FROM kline_30m ORDER BY d", conn)
        conn.close()

        if len(all_dates) == 0:
            print("❌ 无数据")
            return BacktestResult()

        dates = all_dates['d'].tolist()
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]

        print(f"  回测区间: {dates[0]} → {dates[-1]} ({len(dates)} 个交易日)")

        unique_dates = sorted(set(dates))
        prev_day = None
        daily_equity_start = self.initial_capital

        # 逐日回测
        for day_idx, day in enumerate(tqdm(unique_dates, desc='   回测进度', unit='day')):
            # 盘前: 更新股票池
            self.daily_pool = self.get_daily_pool(day)

            # 获取该日所有时间戳
            conn = sqlite3.connect(self.db_path)
            day_ts = pd.read_sql(
                "SELECT DISTINCT date FROM kline_30m "
                f"WHERE DATE(date)='{day}' ORDER BY date", conn)
            conn.close()

            if len(day_ts) == 0:
                continue

            timestamps = [pd.Timestamp(t) for t in day_ts['date'].tolist()]

            # 盘中: 逐根K线
            for ts in timestamps:
                self._process_bar(ts)

            # 日终统计
            total_value = self._total_value()
            self.equity_curve.append((day, total_value))

            if prev_day is not None:
                daily_ret = (total_value - daily_equity_start) / daily_equity_start
                self.daily_returns.append(daily_ret)

            daily_equity_start = total_value
            prev_day = day

            if (day_idx + 1) % 20 == 0:
                print(f"  进度: {day_idx+1}/{len(unique_dates)} | "
                      f"净值: ¥{total_value:,.0f} | "
                      f"持仓: {len(self.positions)}只 | "
                      f"累计交易: {len(self.trades)}笔")

        # 清仓
        final_ts = pd.Timestamp(f"{unique_dates[-1]} 15:00:00")
        for sym in list(self.positions.keys()):
            self._close_position(sym, final_ts, 'end')

        # 计算统计
        result = self._compute_stats()
        self._print_summary(result)
        return result

    def _process_bar(self, timestamp: pd.Timestamp):
        """处理单根K线"""
        # 1. 更新现有持仓
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]
            pos.hold_bars += 1

            df = self.load_intraday_data(sym)
            if df is None:
                continue

            current = df[df['date'] == timestamp]
            if len(current) == 0:
                continue

            current_price = float(current['close'].iloc[0])
            pnl_pct = (current_price - pos.buy_price) / pos.buy_price

            # 更新最大浮盈
            pos.max_profit = max(pos.max_profit, pnl_pct)

            # 止损
            if pnl_pct <= self.stop_loss:
                self._close_position(sym, timestamp, 'stop_loss', current_price)
                continue

            # 止盈
            if pnl_pct >= self.take_profit:
                self._close_position(sym, timestamp, 'take_profit', current_price)
                continue

            # 卖出信号 (分钟级模型)
            if pos.hold_bars >= self.min_hold_bars:
                pred = self.predict_intraday(sym, timestamp)
                if pred is not None and pred < self.sell_threshold:
                    self._close_position(sym, timestamp, 'signal', current_price)

        # 2. 检查买入机会
        if len(self.positions) >= self.max_positions:
            return

        available_cash = self.cash - self.initial_capital * MIN_CASH_RATIO
        if available_cash <= 0:
            return

        # 遍历股票池
        for sym in self.daily_pool:
            if sym in self.positions:
                continue
            if len(self.positions) >= self.max_positions:
                break

            pred = self.predict_intraday(sym, timestamp)
            if pred is None or pred <= self.buy_threshold:
                continue

            # 获取当前价格
            df = self.load_intraday_data(sym)
            current = df[df['date'] == timestamp]
            if len(current) == 0:
                continue

            current_price = float(current['close'].iloc[0])
            buy_price = current_price * (1 + self.slippage)

            # 计算买入数量 (100股整数倍)
            budget = min(available_cash, self.initial_capital * self.position_pct)
            shares = int(budget / buy_price / 100) * 100
            if shares < 100:
                continue

            cost = shares * buy_price * (1 + self.commission)
            if cost > self.cash:
                continue

            # 执行买入
            self.cash -= cost
            self.positions[sym] = Position(
                symbol=sym,
                buy_date=str(timestamp),
                buy_price=buy_price,
                shares=shares,
                cost=cost,
            )

    def _close_position(self, symbol: str, timestamp: pd.Timestamp,
                        reason: str, price: float = None):
        """平仓"""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        if price is None:
            df = self.load_intraday_data(symbol)
            current = df[df['date'] == timestamp]
            if len(current) == 0:
                return
            price = float(current['close'].iloc[0])

        sell_price = price * (1 - self.slippage)
        proceeds = pos.shares * sell_price * (1 - self.commission)
        pnl = proceeds - pos.cost
        pnl_pct = pnl / pos.cost

        self.cash += proceeds
        self.trades.append(Trade(
            symbol=symbol,
            entry_date=pos.buy_date,
            exit_date=str(timestamp),
            entry_price=pos.buy_price,
            exit_price=sell_price,
            shares=pos.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            reason=reason,
        ))
        del self.positions[symbol]

    def _total_value(self) -> float:
        """计算总资产"""
        value = self.cash
        for sym, pos in self.positions.items():
            df = self.load_intraday_data(sym)
            if df is not None and len(df) > 0:
                last_price = float(df['close'].iloc[-1])
                value += pos.shares * last_price
        return value

    def _compute_stats(self) -> BacktestResult:
        """计算统计指标"""
        result = BacktestResult()
        result.trades = self.trades
        result.equity_curve = self.equity_curve

        if len(self.equity_curve) == 0:
            return result

        # 净值曲线
        values = np.array([v for _, v in self.equity_curve])
        result.total_return = (values[-1] - self.initial_capital) / self.initial_capital

        # 年化收益率
        n_days = len(self.equity_curve)
        if n_days > 1:
            result.annual_return = (1 + result.total_return) ** (252 / n_days) - 1

        # 每日收益
        if len(self.daily_returns) > 0:
            daily_ret = np.array(self.daily_returns)
            result.sharpe_ratio = np.sqrt(252) * daily_ret.mean() / (daily_ret.std() + 1e-10)

        # 最大回撤
        peak = values[0]
        max_dd = 0
        for v in values:
            peak = max(peak, v)
            dd = (v - peak) / peak
            max_dd = min(max_dd, dd)
        result.max_drawdown = max_dd

        # 交易统计
        if len(self.trades) > 0:
            result.total_trades = len(self.trades)
            wins = [t for t in self.trades if t.pnl > 0]
            result.win_rate = len(wins) / len(self.trades)

            total_profit = sum(t.pnl for t in wins)
            total_loss = abs(sum(t.pnl for t in self.trades if t.pnl <= 0))
            result.profit_factor = total_profit / (total_loss + 1e-10)

            # 换手率
            total_trade_value = sum(t.shares * t.entry_price for t in self.trades)
            avg_portfolio = values.mean()
            result.turnover = total_trade_value / (avg_portfolio + 1e-10) if avg_portfolio > 0 else 0

        return result

    def _print_summary(self, result: BacktestResult):
        """打印回测摘要"""
        print(f"\n{'='*60}")
        print(f" 📊 回测结果")
        print(f"{'='*60}")
        print(f"  总收益率:    {result.total_return:+.2%}")
        print(f"  年化收益率:  {result.annual_return:+.2%}")
        print(f"  夏普比率:    {result.sharpe_ratio:.2f}")
        print(f"  最大回撤:    {result.max_drawdown:.2%}")
        print(f"  总交易数:    {result.total_trades}")
        print(f"  胜率:        {result.win_rate:.1%}")
        print(f"  盈亏比:      {result.profit_factor:.2f}")
        print(f"  换手率:      {result.turnover:.1f}x")

        if result.total_trades > 0:
            # 按原因统计
            reasons = defaultdict(lambda: {'count': 0, 'pnl': 0})
            for t in result.trades:
                reasons[t.reason]['count'] += 1
                reasons[t.reason]['pnl'] += t.pnl
            print(f"\n  交易原因分布:")
            for reason, stats in reasons.items():
                print(f"    {reason}: {stats['count']}笔, 盈亏 ¥{stats['pnl']:,.0f}")

            # 盈利分布
            pnl_pcts = [t.pnl_pct for t in result.trades]
            print(f"\n  盈亏分布:")
            print(f"    最大盈利: {max(pnl_pcts):+.2%}")
            print(f"    最大亏损: {min(pnl_pcts):+.2%}")
            print(f"    平均盈亏: {np.mean(pnl_pcts):+.2%}")
            print(f"    中位盈亏: {np.median(pnl_pcts):+.2%}")

        print(f"{'='*60}")

    def save_results(self, result: BacktestResult, output_dir: str = RESULT_DIR):
        """保存回测结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 交易记录
        trades_df = pd.DataFrame([{
            'symbol': t.symbol,
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'shares': t.shares,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'reason': t.reason,
        } for t in result.trades])
        trades_df.to_csv(os.path.join(output_dir, 'trades.csv'), index=False)

        # 净值曲线
        equity_df = pd.DataFrame(result.equity_curve, columns=['date', 'equity'])
        equity_df.to_csv(os.path.join(output_dir, 'equity.csv'), index=False)

        # 摘要
        summary = {
            'total_return': result.total_return,
            'annual_return': result.annual_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor,
            'total_trades': result.total_trades,
            'turnover': result.turnover,
            'backtest_date': datetime.now().isoformat(),
        }
        with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n💾 结果已保存到: {output_dir}/")
        print(f"   trades.csv  | equity.csv  | summary.json")


# ============ 主入口 ============
def main():
    parser = argparse.ArgumentParser(description='联合回测 — 日线选股 + 分钟级择时')
    parser.add_argument('--daily-model', type=str, default=DAILY_MODEL_PATH)
    parser.add_argument('--intraday-model', type=str, default=INTRADAY_MODEL_PATH)
    parser.add_argument('--db', type=str, default=DB_PATH)
    parser.add_argument('--capital', type=float, default=INITIAL_CAPITAL)
    parser.add_argument('--pool-size', type=int, default=DAILY_POOL_SIZE)
    parser.add_argument('--max-hold', type=int, default=MAX_POSITIONS)
    parser.add_argument('--position-pct', type=float, default=POSITION_PCT)
    parser.add_argument('--buy-threshold', type=float, default=BUY_THRESHOLD)
    parser.add_argument('--sell-threshold', type=float, default=SELL_THRESHOLD)
    parser.add_argument('--stop-loss', type=float, default=STOP_LOSS)
    parser.add_argument('--take-profit', type=float, default=TAKE_PROFIT)
    parser.add_argument('--no-daily', action='store_true', help='不使用日线模型')
    parser.add_argument('--start', type=str, default=None, help='开始日期')
    parser.add_argument('--end', type=str, default=None, help='结束日期')
    parser.add_argument('--output', type=str, default=RESULT_DIR)
    args = parser.parse_args()

    bt = EnsembleBacktest(
        daily_model_path=args.daily_model if not args.no_daily else None,
        intraday_model_path=args.intraday_model,
        db_path=args.db,
        initial_capital=args.capital,
        max_positions=args.max_hold,
        position_pct=args.position_pct,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        daily_pool_size=args.pool_size,
        use_daily_model=not args.no_daily,
    )

    result = bt.run(start_date=args.start, end_date=args.end)
    bt.save_results(result, args.output)


if __name__ == '__main__':
    main()