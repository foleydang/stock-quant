#!/usr/bin/env python3
"""
增强版交易策略系统
包含：
1. 趋势跟踪策略（长持）- 日线级别，持仓数天到数周
2. 日内回转策略（做T）- 30分钟级别，当日买卖
3. 消息面分析 - 网络爬取新闻
4. 综合信号系统 - 多策略融合
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import requests
from bs4 import BeautifulSoup

from strategy.intraday_strategy import IntradayStrategy, SignalType, WATCHLIST_STOCKS, TechnicalIndicators
from data.data_handler import DataHandler


class StrategyType(Enum):
    """策略类型"""
    TREND = "趋势跟踪"  # 长持策略
    SWING = "波段操作"  # 中线策略
    DAY_TRADE = "日内做T"  # 日内回转


class TradeIntent(Enum):
    """交易意图"""
    OPEN_LONG = "开多仓"
    ADD_POSITION = "加仓"
    HOLD = "持有"
    REDUCE_POSITION = "减仓"
    CLOSE_LONG = "平多仓"
    DO_NOTHING = "观望"


@dataclass
class NewsItem:
    """新闻条目"""
    title: str
    source: str
    time: str
    sentiment: str  # positive, negative, neutral
    impact: str  # high, medium, low
    summary: str


@dataclass
class EnhancedSignal:
    """增强版交易信号"""
    symbol: str
    stock_name: str
    timestamp: str
    current_price: float

    # 策略信号
    trend_signal: str = "持有"  # 趋势策略信号
    trend_score: float = 0.0
    swing_signal: str = "持有"  # 波段策略信号
    swing_score: float = 0.0
    day_trade_signal: str = "观望"  # 日内做T信号
    day_trade_score: float = 0.0

    # 综合信号
    combined_signal: str = "持有"
    combined_score: float = 0.0
    confidence: float = 0.0

    # 消息面
    news_sentiment: str = "中性"
    news_impact: str = "低"
    news_summary: str = ""

    # 操作建议
    action: str = "观望"
    position_pct: float = 0.0  # 建议仓位比例
    stop_loss: float = 0.0
    take_profit: float = 0.0

    # 做T建议
    t_buy_price: float = 0.0  # 做T买入价
    t_sell_price: float = 0.0  # 做T卖出价

    # 详细原因
    reasons: List[str] = field(default_factory=list)


class NewsAnalyzer:
    """消息面分析器"""

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.cache = {}
        self.cache_time = {}

    def fetch_news(self, stock_name: str, stock_code: str) -> List[NewsItem]:
        """获取股票新闻"""
        news_list = []

        # 检查缓存（1小时有效）
        cache_key = f"{stock_name}_{stock_code}"
        if cache_key in self.cache:
            if datetime.now() - self.cache_time.get(cache_key, datetime.min) < timedelta(hours=1):
                return self.cache[cache_key]

        try:
            # 东方财富新闻接口
            if 'HK' in stock_code:
                # 港股
                code = stock_code.replace('.HK', '').zfill(5)
                url = f"https://searchapi.eastmoney.com/bussiness/web/QuotationLabelSearch?cb=jQuery&keyword={code}&type=14&pi=1&ps=10"
            else:
                # A股
                code = stock_code.split('.')[0]
                url = f"https://searchapi.eastmoney.com/bussiness/web/QuotationLabelSearch?cb=jQuery&keyword={code}&type=14&pi=1&ps=10"

            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                # 解析JSONP
                text = resp.text
                if 'jQuery(' in text:
                    json_str = text[text.index('(')+1:text.rindex(')')]
                    data = json.loads(json_str)

                    if data.get('Data'):
                        for item in data['Data'][:5]:
                            news = NewsItem(
                                title=item.get('Title', ''),
                                source=item.get('Source', ''),
                                time=item.get('ShowTime', ''),
                                sentiment=self._analyze_sentiment(item.get('Title', '')),
                                impact=self._analyze_impact(item.get('Title', '')),
                                summary=item.get('Content', '')[:100] if item.get('Content') else ''
                            )
                            news_list.append(news)
        except Exception as e:
            print(f"  获取新闻失败: {e}")

        # 缓存结果
        self.cache[cache_key] = news_list
        self.cache_time[cache_key] = datetime.now()

        return news_list

    def _analyze_sentiment(self, title: str) -> str:
        """分析新闻情绪"""
        positive_words = ['上涨', '利好', '增长', '突破', '创新高', '盈利', '增持', '收购',
                         '业绩大增', '涨停', '牛市', '反弹', '回暖', '超预期']
        negative_words = ['下跌', '利空', '亏损', '跌停', '减持', '处罚', '调查',
                         '业绩下滑', '暴跌', '熊市', '危机', '风险', '预警']

        title_lower = title.lower()
        pos_count = sum(1 for w in positive_words if w in title_lower)
        neg_count = sum(1 for w in negative_words if w in title_lower)

        if pos_count > neg_count:
            return "正面"
        elif neg_count > pos_count:
            return "负面"
        return "中性"

    def _analyze_impact(self, title: str) -> str:
        """分析新闻影响程度"""
        high_impact_words = ['业绩', '重组', '收购', '处罚', '调查', '停牌', '复牌',
                            '重大', '突破', '涨停', '跌停', '公告']
        medium_impact_words = ['上涨', '下跌', '增持', '减持', '分红', '预告']

        for word in high_impact_words:
            if word in title:
                return "高"
        for word in medium_impact_words:
            if word in title:
                return "中"
        return "低"

    def get_sentiment_score(self, news_list: List[NewsItem]) -> Tuple[str, str, str]:
        """获取综合情绪评分"""
        if not news_list:
            return "中性", "低", "暂无最新消息"

        # 统计情绪
        sentiment_counts = {"正面": 0, "负面": 0, "中性": 0}
        high_impact_count = 0

        for news in news_list:
            sentiment_counts[news.sentiment] += 1
            if news.impact == "高":
                high_impact_count += 1

        # 确定整体情绪
        if sentiment_counts["正面"] > sentiment_counts["负面"]:
            overall = "正面"
        elif sentiment_counts["负面"] > sentiment_counts["正面"]:
            overall = "负面"
        else:
            overall = "中性"

        # 影响程度
        impact = "高" if high_impact_count >= 2 else ("中" if high_impact_count >= 1 else "低")

        # 摘要
        summaries = [f"{n.title[:20]}..." for n in news_list[:3]]
        summary = " | ".join(summaries) if summaries else "暂无消息"

        return overall, impact, summary


class TrendStrategy:
    """趋势跟踪策略（日线级别，长持）"""

    def __init__(self):
        self.min_hold_days = 3  # 最少持有3天
        self.ma_fast = 5
        self.ma_mid = 20
        self.ma_slow = 60

    def analyze(self, df: pd.DataFrame) -> Tuple[str, float, List[str]]:
        """
        分析趋势信号
        返回: (信号, 评分, 原因列表)
        """
        if len(df) < 60:
            return "持有", 0.0, ["数据不足"]

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        reasons = []
        score = 0.0

        # 0. 趋势方向判断（最重要）
        ma5 = np.mean(close[-5:])
        ma10 = np.mean(close[-10:])
        ma20 = np.mean(close[-20:])
        ma60 = np.mean(close[-60:])

        # 计算趋势斜率
        trend_20 = (close[-1] - close[-20]) / close[-20] * 100
        trend_60 = (close[-1] - close[-60]) / close[-60] * 100 if len(close) >= 60 else 0

        # 强下跌趋势过滤
        if trend_20 < -8 or trend_60 < -15:
            return "持有", 0.0, ["强下跌趋势，观望"]

        # 1. 均线趋势
        # 多头排列
        if ma5 > ma10 > ma20 > ma60:
            score += 3
            reasons.append("均线多头排列(强趋势)")
        # 空头排列
        elif ma5 < ma10 < ma20 < ma60:
            score -= 3
            reasons.append("均线空头排列(弱趋势)")
        # 金叉
        elif ma5 > ma10 and ma5 > ma20:
            score += 1.5
            reasons.append("短期均线上穿")
        elif ma5 < ma10 and ma5 < ma20:
            score -= 1.5
            reasons.append("短期均线下穿")

        # 2. 趋势强度（ADX）
        adx_data = TechnicalIndicators.calculate_adx(high, low, close, 14)
        if len(adx_data['adx']) > 0:
            adx = adx_data['adx'][-1]
            plus_di = adx_data['plus_di'][-1]
            minus_di = adx_data['minus_di'][-1]

            if adx > 30:
                if plus_di > minus_di:
                    score += 2
                    reasons.append(f"强上涨趋势(ADX={adx:.1f})")
                else:
                    score -= 2
                    reasons.append(f"强下跌趋势(ADX={adx:.1f})")

        # 3. 价格位置（相对于均线）
        current_price = close[-1]
        if current_price > ma20:
            score += 0.5
            reasons.append("价格在20日线上方")
        else:
            score -= 0.5
            reasons.append("价格在20日线下方")

        # 4. 成交量趋势
        vol_5 = np.mean(volume[-5:])
        vol_20 = np.mean(volume[-20:])
        if vol_5 > vol_20 * 1.5:
            if close[-1] > close[-5]:
                score += 1
                reasons.append("放量上涨")
            else:
                score -= 0.5
                reasons.append("放量下跌")

        # 5. MACD趋势
        macd_data = TechnicalIndicators.calculate_macd(close)
        if len(macd_data['macd']) >= 2:
            macd = macd_data['macd'][-1]
            signal = macd_data['signal'][-1]
            histogram = macd_data['histogram']

            if macd > signal and histogram[-1] > histogram[-2]:
                score += 1.5
                reasons.append("MACD多头趋势")
            elif macd < signal and histogram[-1] < histogram[-2]:
                score -= 1.5
                reasons.append("MACD空头趋势")

        # 确定信号
        if score >= 4:
            signal = "强烈买入"
        elif score >= 2:
            signal = "买入"
        elif score >= -1:
            signal = "持有"
        elif score >= -3:
            signal = "卖出"
        else:
            signal = "强烈卖出"

        return signal, score, reasons


class SwingStrategy:
    """波段操作策略（中线，持仓数天）"""

    def __init__(self):
        self.rsi_oversold = 30
        self.rsi_overbought = 70

    def analyze(self, df: pd.DataFrame) -> Tuple[str, float, List[str]]:
        """分析波段信号"""
        if len(df) < 40:
            return "持有", 0.0, ["数据不足"]

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        reasons = []
        score = 0.0

        # 0. 趋势过滤
        trend_20 = (close[-1] - close[-20]) / close[-20] * 100

        # 强下跌趋势中，只在极度超卖时买入
        if trend_20 < -8:
            # 只在极度超卖时才考虑
            rsi = TechnicalIndicators.calculate_rsi(close, 14)[-1]
            if rsi > 25:
                return "持有", 0.0, [f"下跌趋势({trend_20:.1f}%)，等待更低位"]
            reasons.append(f"强下跌+RSI超卖，抄底机会")

        # 1. RSI超买超卖
        rsi = TechnicalIndicators.calculate_rsi(close, 14)[-1]

        if rsi < 25:
            score += 3
            reasons.append(f"RSI严重超卖({rsi:.1f})")
        elif rsi < self.rsi_oversold:
            score += 2
            reasons.append(f"RSI超卖({rsi:.1f})")
        elif rsi > 75:
            score -= 3
            reasons.append(f"RSI严重超买({rsi:.1f})")
        elif rsi > self.rsi_overbought:
            score -= 2
            reasons.append(f"RSI超买({rsi:.1f})")

        # 2. 布林带位置
        bb = TechnicalIndicators.calculate_bollinger_bands(close, 20, 2.0)
        current_price = close[-1]
        upper = bb['upper'][-1]
        lower = bb['lower'][-1]
        mid = bb['mid'][-1]

        if current_price < lower:
            score += 2
            reasons.append("跌破布林下轨(超卖)")
        elif current_price > upper:
            score -= 2
            reasons.append("突破布林上轨(超买)")
        elif current_price < mid:
            score -= 0.5
        else:
            score += 0.5

        # 3. KDJ信号
        kdj = TechnicalIndicators.calculate_kdj(high, low, close)
        k = kdj['k'][-1]
        d = kdj['d'][-1]
        j = kdj['j'][-1]

        if j < 10:
            score += 2
            reasons.append(f"KDJ超卖(J={j:.1f})")
        elif j > 100:
            score -= 2
            reasons.append(f"KDJ超买(J={j:.1f})")

        # 4. 趋势+超卖结合
        ma20 = np.mean(close[-20:])
        trend = (close[-1] - close[-20]) / close[-20] * 100

        # 下跌趋势中超卖是买入机会
        if trend < -5 and rsi < 35:
            score += 2
            reasons.append(f"下跌趋势超卖反弹机会({trend:.1f}%)")

        # 确定信号
        if score >= 3:
            signal = "买入"
        elif score >= 1:
            signal = "轻仓买入"
        elif score >= -1:
            signal = "持有"
        elif score >= -3:
            signal = "减仓"
        else:
            signal = "卖出"

        return signal, score, reasons


class DayTradeStrategy:
    """日内做T策略（30分钟级别）"""

    def __init__(self):
        self.t_profit_pct = 0.015  # 做T目标收益1.5%
        self.t_stop_pct = 0.01  # 做T止损1%

    def analyze(self, df: pd.DataFrame) -> Tuple[str, float, List[str], float, float]:
        """
        分析做T机会
        返回: (信号, 评分, 原因, 建议买入价, 建议卖出价)
        """
        if len(df) < 20:
            return "观望", 0.0, ["数据不足"], 0.0, 0.0

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        current_price = close[-1]
        reasons = []
        score = 0.0

        # 1. 日内支撑压力位
        day_high = max(high[-8:])  # 今日最高
        day_low = min(low[-8:])  # 今日最低
        day_range = day_high - day_low

        # 2. VWAP
        vwap = TechnicalIndicators.calculate_vwap(high, low, close, volume, 20)[-1]

        # 3. 判断位置
        position_in_range = (current_price - day_low) / day_range if day_range > 0 else 0.5

        # 4. RSI
        rsi = TechnicalIndicators.calculate_rsi(close, 14)[-1]

        # 做T逻辑
        t_buy_price = 0.0
        t_sell_price = 0.0

        # 在低位+超卖 = 买入做T
        if position_in_range < 0.3 and rsi < 40:
            score += 2
            t_buy_price = current_price
            t_sell_price = current_price * (1 + self.t_profit_pct)
            reasons.append(f"低位+超卖,可在{t_buy_price:.2f}买入,目标{t_sell_price:.2f}")

        # 在高位+超买 = 卖出做T
        elif position_in_range > 0.7 and rsi > 60:
            score -= 2
            t_sell_price = current_price
            t_buy_price = current_price * (1 - self.t_profit_pct)
            reasons.append(f"高位+超买,可在{t_sell_price:.2f}卖出,回买价{t_buy_price:.2f}")

        # 接近VWAP下方 = 买入机会
        elif current_price < vwap * 0.98 and rsi < 45:
            score += 1.5
            t_buy_price = current_price
            t_sell_price = vwap
            reasons.append(f"低于VWAP,可在{t_buy_price:.2f}买入,目标{t_sell_price:.2f}")

        # 接近VWAP上方 = 卖出机会
        elif current_price > vwap * 1.02 and rsi > 55:
            score -= 1.5
            t_sell_price = current_price
            t_buy_price = vwap
            reasons.append(f"高于VWAP,可在{t_sell_price:.2f}卖出,回买价{t_buy_price:.2f}")

        else:
            reasons.append("当前不适合做T,观望")

        # 5. 成交量确认
        vol_ratio = volume[-1] / np.mean(volume[-10:]) if np.mean(volume[-10:]) > 0 else 1
        if vol_ratio > 1.5:
            if score > 0:
                score += 0.5
                reasons.append("放量确认买入信号")
            elif score < 0:
                score -= 0.5
                reasons.append("放量确认卖出信号")

        # 确定信号
        if score >= 1.5:
            signal = "买入做T"
        elif score <= -1.5:
            signal = "卖出做T"
        else:
            signal = "观望"

        return signal, score, reasons, t_buy_price, t_sell_price


class EnhancedStrategySystem:
    """增强版策略系统"""

    def __init__(self):
        self.trend_strategy = TrendStrategy()
        self.swing_strategy = SwingStrategy()
        self.day_trade_strategy = DayTradeStrategy()
        self.news_analyzer = NewsAnalyzer()
        self.data_handler = DataHandler(force_refresh=True)

    def analyze_stock(self, symbol: str, stock_name: str, df: pd.DataFrame = None) -> EnhancedSignal:
        """综合分析单只股票"""
        if df is None:
            df = self.data_handler.fetch_stock_data(symbol, force_refresh=True)

        if df is None or len(df) < 60:
            return None

        current_price = df['close'].iloc[-1]

        signal = EnhancedSignal(
            symbol=symbol,
            stock_name=stock_name,
            timestamp=datetime.now().isoformat(),
            current_price=current_price
        )

        # 1. 趋势策略分析
        trend_signal, trend_score, trend_reasons = self.trend_strategy.analyze(df)
        signal.trend_signal = trend_signal
        signal.trend_score = trend_score

        # 2. 波段策略分析
        swing_signal, swing_score, swing_reasons = self.swing_strategy.analyze(df)
        signal.swing_signal = swing_signal
        signal.swing_score = swing_score

        # 3. 日内做T分析
        day_signal, day_score, day_reasons, t_buy, t_sell = self.day_trade_strategy.analyze(df)
        signal.day_trade_signal = day_signal
        signal.day_trade_score = day_score
        signal.t_buy_price = t_buy
        signal.t_sell_price = t_sell

        # 4. 消息面分析
        try:
            news_list = self.news_analyzer.fetch_news(stock_name, symbol)
            sentiment, impact, summary = self.news_analyzer.get_sentiment_score(news_list)
            signal.news_sentiment = sentiment
            signal.news_impact = impact
            signal.news_summary = summary
        except Exception as e:
            signal.news_sentiment = "中性"
            signal.news_impact = "低"
            signal.news_summary = f"获取消息失败: {e}"

        # 5. 综合评分
        # 趋势权重: 40%, 波段权重: 30%, 消息面权重: 30%
        combined_score = trend_score * 0.4 + swing_score * 0.3

        # 消息面调整
        if signal.news_sentiment == "正面":
            combined_score += 1 if signal.news_impact == "高" else 0.5
        elif signal.news_sentiment == "负面":
            combined_score -= 1 if signal.news_impact == "高" else 0.5

        signal.combined_score = combined_score

        # 计算置信度
        signal.confidence = min(1.0, abs(combined_score) / 5)

        # 6. 综合信号
        if combined_score >= 3:
            signal.combined_signal = "强烈买入"
            signal.action = "开多仓"
            signal.position_pct = min(0.8, 0.3 + combined_score * 0.1)
        elif combined_score >= 1.5:
            signal.combined_signal = "买入"
            signal.action = "建仓/加仓"
            signal.position_pct = 0.3
        elif combined_score >= -1:
            signal.combined_signal = "持有"
            signal.action = "持有观望"
            signal.position_pct = 0.2
        elif combined_score >= -3:
            signal.combined_signal = "卖出"
            signal.action = "减仓"
            signal.position_pct = 0.1
        else:
            signal.combined_signal = "强烈卖出"
            signal.action = "清仓"
            signal.position_pct = 0

        # 7. 止损止盈
        atr = TechnicalIndicators.calculate_atr(df['high'].values, df['low'].values, df['close'].values, 14)
        atr_val = atr[-1] if len(atr) > 0 else current_price * 0.02
        signal.stop_loss = current_price - atr_val * 3
        signal.take_profit = current_price + atr_val * 4

        # 8. 汇总原因
        signal.reasons = []
        signal.reasons.append(f"【趋势】{trend_signal}(评分:{trend_score:.1f}): {', '.join(trend_reasons[:2])}")
        signal.reasons.append(f"【波段】{swing_signal}(评分:{swing_score:.1f}): {', '.join(swing_reasons[:2])}")
        signal.reasons.append(f"【做T】{day_signal}(评分:{day_score:.1f}): {', '.join(day_reasons[:2])}")
        signal.reasons.append(f"【消息】{sentiment}({impact}影响): {summary[:50]}...")

        return signal

    def analyze_all(self, watchlist: List[Dict] = None) -> List[EnhancedSignal]:
        """分析所有股票"""
        if watchlist is None:
            watchlist = WATCHLIST_STOCKS

        signals = []
        print("=" * 70)
        print("增强版策略分析")
        print("=" * 70)

        for i, stock in enumerate(watchlist):
            symbol = stock['symbol']
            name = stock['name']

            print(f"\n[{i+1}/{len(watchlist)}] {name} ({symbol})...")

            signal = self.analyze_stock(symbol, name)
            if signal:
                signals.append(signal)
                self._print_signal(signal)

        self._print_summary(signals)
        return signals

    def _print_signal(self, signal: EnhancedSignal):
        """打印单只股票信号"""
        print(f"  当前价格: {signal.current_price:.2f}")
        print(f"  综合信号: {signal.combined_signal} (评分:{signal.combined_score:.1f}, 置信度:{signal.confidence:.0%})")
        print(f"  操作建议: {signal.action}, 建议仓位:{signal.position_pct:.0%}")
        print(f"  止损: {signal.stop_loss:.2f} | 止盈: {signal.take_profit:.2f}")

        if signal.t_buy_price > 0 or signal.t_sell_price > 0:
            print(f"  做T建议: 买入价{signal.t_buy_price:.2f} | 卖出价{signal.t_sell_price:.2f}")

        print(f"  消息面: {signal.news_sentiment}({signal.news_impact})")

    def _print_summary(self, signals: List[EnhancedSignal]):
        """打印汇总"""
        print("\n" + "=" * 70)
        print("信号汇总")
        print("=" * 70)

        for s in signals:
            emoji = "🟢" if "买入" in s.combined_signal else "🔴" if "卖出" in s.combined_signal else "⚪"
            print(f"{emoji} {s.stock_name}: {s.combined_signal} | 评分:{s.combined_score:.1f} | "
                  f"建议:{s.action} | 消息:{s.news_sentiment}")

        # 统计
        buy_count = sum(1 for s in signals if "买入" in s.combined_signal)
        sell_count = sum(1 for s in signals if "卖出" in s.combined_signal)
        hold_count = len(signals) - buy_count - sell_count

        print(f"\n买入: {buy_count} | 持有: {hold_count} | 卖出: {sell_count}")

        # 做T机会
        t_opportunities = [s for s in signals if "做T" in s.day_trade_signal]
        if t_opportunities:
            print(f"\n做T机会: {len(t_opportunities)} 只")
            for s in t_opportunities:
                print(f"  - {s.stock_name}: {s.day_trade_signal}")


def main():
    system = EnhancedStrategySystem()
    system.analyze_all()


if __name__ == "__main__":
    main()