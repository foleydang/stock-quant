#!/usr/bin/env python3
"""
做T策略模块 - 基于30分钟K线给出具体日内操作建议

功能：
1. 分析支撑位和阻力位
2. 判断当前是否适合做T
3. 给出具体的买入价位、卖出价位、操作数量

策略：
- 正T（先买后卖）：在支撑位附近买入，反弹到阻力位卖出原有持仓
- 做T条件：日内波动足够大（>2%），且有明确支撑阻力
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TTradeSuggestion:
    """做T建议"""
    symbol: str
    stock_name: str
    action: str  # '适合做T', '观望', '不建议'
    current_price: float
    cost_price: float
    profit_pct: float

    # 具体操作建议
    buy_price: Optional[float] = None  # 建议买入价位
    sell_price: Optional[float] = None  # 建议卖出价位
    buy_shares: Optional[int] = None   # 建议买入数量
    sell_shares: Optional[int] = None  # 建议卖出数量

    # 技术分析
    support_price: Optional[float] = None  # 支撑位
    resistance_price: Optional[float] = None  # 阻力位
    intraday_range: Optional[float] = None  # 日内波动幅度
    trend: Optional[str] = None  # 'up', 'down', 'sideways'

    # 操作理由
    reason: str = ""

    # 风险提示
    risk_level: str = "low"  # 'low', 'medium', 'high'


class TStrategy:
    """做T策略"""

    def __init__(self):
        # 策略参数
        self.min_intraday_range = 2.0  # 最小日内波动（%）才考虑做T
        self.min_support_distance = 1.0  # 支撑位距离现价至少1%
        self.min_resistance_distance = 1.0  # 阻力位距离现价至少1%
        self.t_profit_target = 1.5  # 做T目标利润（%）
        self.max_t_ratio = 0.33  # 单次做T不超过持仓1/3

        # 深套参数（亏损>15%才建议做T）
        self.deep_loss_threshold = -15.0

    def analyze(self, df: pd.DataFrame, position: Dict) -> TTradeSuggestion:
        """
        分析单只股票，给出做T建议

        Args:
            df: 30分钟K线数据，包含 date, open, high, low, close, volume
            position: 持仓信息，包含 symbol, stock_name, shares, cost_price, current_price

        Returns:
            TTradeSuggestion 做T建议
        """
        if df is None or len(df) < 20:
            return self._no_suggestion(position, "数据不足")

        symbol = position['symbol']
        stock_name = position['stock_name']
        shares = position['shares']
        cost_price = position['cost_price']
        current_price = position.get('current_price', float(df['close'].iloc[-1]))

        profit_pct = (current_price - cost_price) / cost_price * 100

        # 获取今日数据
        today_df = self._get_today_data(df)
        if today_df is None or len(today_df) < 3:
            return self._no_suggestion(position, "今日数据不足")

        # 计算技术指标
        support, resistance = self._calc_support_resistance(df, today_df)
        intraday_range = self._calc_intraday_range(today_df)
        trend = self._calc_trend(today_df)

        # 判断是否适合做T
        if profit_pct > 0:
            # 盈利的股票不建议做T（容易做成反T亏更多）
            return self._no_suggestion(position, f"当前盈利{profit_pct:.1f}%，不建议做T")

        if intraday_range < self.min_intraday_range:
            return self._no_suggestion(position, f"日内波动仅{intraday_range:.1f}%，波动太小不适合做T")

        # 计算具体操作建议
        suggestion = self._calc_t_operation(
            position, current_price, support, resistance,
            intraday_range, trend, profit_pct
        )

        return suggestion

    def _get_today_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """获取今日数据"""
        df['date'] = pd.to_datetime(df['date'], format='mixed')
        today = datetime.now().date()
        today_df = df[df['date'].dt.date == today]
        if len(today_df) == 0:
            # 可能数据还没更新，取最新一天
            latest_date = df['date'].dt.date.max()
            today_df = df[df['date'].dt.date == latest_date]
        return today_df if len(today_df) >= 3 else None

    def _calc_support_resistance(self, df: pd.DataFrame, today_df: pd.DataFrame) -> Tuple[float, float]:
        """
        计算支撑位和阻力位

        方法：
        1. 今日最低价附近的支撑
        2. 今日最高价附近的阻力
        3. 结合近期（5天）的关键价位
        """
        # 今日关键价位
        today_low = float(today_df['low'].min())
        today_high = float(today_df['high'].max())

        # 近5天的关键价位
        recent_df = df.tail(80)  # 约5天的30分钟数据
        recent_low = float(recent_df['low'].min())
        recent_high = float(recent_df['high'].max())

        # 支撑位：取今日低点和近期低点的加权
        # 如果今日低点接近近期低点，说明是强支撑
        if today_low <= recent_low * 1.02:
            support = today_low  # 强支撑
        else:
            support = today_low * 0.995  # 稍微保守一点

        # 阻力位：取今日高点和近期高点的加权
        if today_high >= recent_high * 0.98:
            resistance = today_high  # 强阻力
        else:
            resistance = today_high * 1.005

        return support, resistance

    def _calc_intraday_range(self, today_df: pd.DataFrame) -> float:
        """计算日内波动幅度"""
        high = float(today_df['high'].max())
        low = float(today_df['low'].min())

        # 取非零的价格作为基准（港股/ETF开盘价可能为0）
        valid_prices = today_df['close'].dropna()
        valid_prices = valid_prices[valid_prices > 0]
        if len(valid_prices) == 0:
            return 0.0
        base_price = float(valid_prices.iloc[0])

        # 相对于基准价的波动
        if base_price == 0:
            return 0.0
        range_pct = (high - low) / base_price * 100
        return range_pct

    def _calc_trend(self, today_df: pd.DataFrame) -> str:
        """计算日内趋势"""
        if len(today_df) < 3:
            return 'sideways'

        prices = today_df['close'].values
        first_half_avg = np.mean(prices[:len(prices)//2])
        second_half_avg = np.mean(prices[len(prices)//2:])

        if second_half_avg > first_half_avg * 1.005:
            return 'up'
        elif second_half_avg < first_half_avg * 0.995:
            return 'down'
        else:
            return 'sideways'

    def _calc_t_operation(
        self, position: Dict, current_price: float,
        support: float, resistance: float,
        intraday_range: float, trend: str, profit_pct: float
    ) -> TTradeSuggestion:
        """计算具体的做T操作"""

        symbol = position['symbol']
        stock_name = position['stock_name']
        shares = position['shares']
        cost_price = position['cost_price']

        # 计算做T数量（不超过持仓1/3）
        t_shares = int(shares * self.max_t_ratio / 100) * 100
        if t_shares < 100:
            t_shares = 100  # 最少100股

        # 判断趋势和操作方向
        current_vs_support = (current_price - support) / current_price * 100
        current_vs_resistance = (resistance - current_price) / current_price * 100

        # 根据趋势和位置给出建议
        if trend == 'down' and current_vs_support < 1.5:
            # 下跌趋势，接近支撑位 -> 适合买入做T
            action = '适合做T'
            buy_price = support * 1.005  # 支撑位附近买入
            sell_price = current_price * (1 + self.t_profit_target/100)  # 反弹1.5%卖出
            reason = f"下跌接近支撑位¥{support:.2f}，可在此处买入，反弹{self.t_profit_target}%后卖出"

            risk_level = 'medium'

        elif trend == 'sideways' and current_vs_support < 1.0:
            # 震荡走势，接近支撑 -> 适合做T
            action = '适合做T'
            buy_price = support * 1.005
            sell_price = resistance * 0.995  # 接近阻力位卖出
            t_profit = (sell_price - buy_price) / buy_price * 100
            reason = f"震荡走势，支撑¥{support:.2f}买入，阻力¥{resistance:.2f}卖出，预期利润{t_profit:.1f}%"

            risk_level = 'low'

        elif trend == 'up' and current_vs_resistance < 1.0:
            # 上涨趋势，接近阻力 -> 不建议买入，但可以卖出部分
            action = '可减仓'
            buy_price = None
            sell_price = resistance * 0.995
            sell_shares = t_shares
            reason = f"上涨接近阻力位¥{resistance:.2f}，可卖出{t_shares}股锁定利润"

            risk_level = 'low'

        elif current_price > cost_price:
            # 现价高于成本价，可以考虑卖出降成本
            action = '可减仓'
            sell_price = current_price
            sell_shares = t_shares
            reason = f"现价¥{current_price:.2f}高于成本¥{cost_price:.2f}，卖出{t_shares}股降低持仓成本"

            risk_level = 'low'

        else:
            # 其他情况，给出观望建议
            action = '观望'
            buy_price = None
            sell_price = None
            reason = f"当前位置不适合做T，等待更好机会（支撑¥{support:.2f}，阻力¥{resistance:.2f}）"

            risk_level = 'medium'

        return TTradeSuggestion(
            symbol=symbol,
            stock_name=stock_name,
            action=action,
            current_price=current_price,
            cost_price=cost_price,
            profit_pct=profit_pct,
            buy_price=buy_price,
            sell_price=sell_price,
            buy_shares=t_shares if buy_price else None,
            sell_shares=t_shares if sell_price else None,
            support_price=support,
            resistance_price=resistance,
            intraday_range=intraday_range,
            trend=trend,
            reason=reason,
            risk_level=risk_level
        )

    def _no_suggestion(self, position: Dict, reason: str) -> TTradeSuggestion:
        """返回不建议操作的建议"""
        return TTradeSuggestion(
            symbol=position['symbol'],
            stock_name=position['stock_name'],
            action='不建议',
            current_price=position.get('current_price', 0),
            cost_price=position['cost_price'],
            profit_pct=0,
            reason=reason,
            risk_level='high'
        )

    def analyze_batch(self, data_dict: Dict[str, pd.DataFrame], positions: Dict[str, Dict]) -> List[TTradeSuggestion]:
        """批量分析多只股票"""
        suggestions = []
        for symbol, position in positions.items():
            df = data_dict.get(symbol)
            if df is not None:
                suggestion = self.analyze(df, position)
                suggestions.append(suggestion)
        return suggestions


def format_t_suggestion(s: TTradeSuggestion) -> str:
    """格式化做T建议为可读文本"""
    lines = []

    # 基本信息
    profit_str = f"{s.profit_pct:+.1f}%"
    lines.append(f"【{s.stock_name}】现价¥{s.current_price:.2f} 成本¥{s.cost_price:.2f} 浮亏{profit_str}")

    # 技术分析
    if s.support_price and s.resistance_price:
        lines.append(f"  支撑: ¥{s.support_price:.2f} | 阻力: ¥{s.resistance_price:.2f} | 波动: {s.intraday_range:.1f}% | 趋势: {s.trend}")

    # 操作建议
    action_emoji = {
        '适合做T': '🟢',
        '可减仓': '🔵',
        '观望': '⚠️',
        '不建议': '❌'
    }.get(s.action, '⚪')

    lines.append(f"  {action_emoji} {s.action}: {s.reason}")

    # 具体操作
    if s.buy_price and s.buy_shares:
        lines.append(f"  💰 建议买入: ¥{s.buy_price:.2f} × {s.buy_shares}股 = ¥{s.buy_price * s.buy_shares:,.0f}")
    if s.sell_price and s.sell_shares:
        lines.append(f"  💵 建议卖出: ¥{s.sell_price:.2f} × {s.sell_shares}股 = ¥{s.sell_price * s.sell_shares:,.0f}")

    return '\n'.join(lines)


def format_t_suggestions_batch(suggestions: List[TTradeSuggestion]) -> str:
    """批量格式化"""
    lines = ["=" * 60]
    lines.append(f"🔄 做T操作建议 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 60)

    # 分类显示
    can_t = [s for s in suggestions if s.action == '适合做T']
    can_reduce = [s for s in suggestions if s.action == '可减仓']
    watch = [s for s in suggestions if s.action == '观望']
    no_t = [s for s in suggestions if s.action == '不建议']

    if can_t:
        lines.append("\n🟢 【适合做T】")
        for s in can_t:
            lines.append(format_t_suggestion(s))

    if can_reduce:
        lines.append("\n🔵 【可减仓】")
        for s in can_reduce:
            lines.append(format_t_suggestion(s))

    if watch:
        lines.append("\n⚠️ 【观望等待】")
        for s in watch:
            lines.append(format_t_suggestion(s))

    if no_t:
        lines.append("\n❌ 【不建议做T】")
        for s in no_t:
            lines.append(f"  {s.stock_name}: {s.reason}")

    lines.append("\n" + "=" * 60)
    lines.append("提示: 做T需确保有足够现金，先买后卖(正T)，数量不超过持仓1/3")

    return '\n'.join(lines)


# 测试
if __name__ == "__main__":
    import sqlite3
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB_PATH = os.path.join(BASE_DIR, 'data/stock_data.db')

    # 获取测试数据
    conn = sqlite3.connect(DB_PATH)

    # 模拟持仓
    test_positions = {
        '600048.SH': {
            'symbol': '600048.SH',
            'stock_name': '保利发展',
            'shares': 2000,
            'cost_price': 10.50,
            'current_price': 8.30
        },
        '300015.SZ': {
            'symbol': '300015.SZ',
            'stock_name': '爱尔眼科',
            'shares': 500,
            'cost_price': 15.80,
            'current_price': 9.68
        }
    }

    strategy = TStrategy()

    for symbol, pos in test_positions.items():
        df = pd.read_sql_query(
            'SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol=? ORDER BY date',
            conn, params=(symbol,)
        )

        if len(df) > 50:
            suggestion = strategy.analyze(df, pos)
            print(format_t_suggestion(suggestion))
            print()

    conn.close()