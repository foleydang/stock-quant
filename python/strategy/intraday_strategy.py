#!/usr/bin/env python3
"""
30分钟级别多因子交易策略

功能：
1. 分析多个技术指标（RSI、MACD、KDJ、布林带、均线）
2. 生成综合交易信号和评分
3. 支持自定义股票池
4. 可选多渠道通知

策略：
- 信号评分系统：综合多个指标给出买入/卖出/持有建议
- 评分 >= 4: 强烈买入
- 评分 >= 2: 买入
- 评分 -1~1: 持有
- 评分 <= -2: 卖出
- 评分 <= -4: 强烈卖出
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.data_handler import DataHandler


# 默认股票池
WATCHLIST_STOCKS = [
    {'symbol': '300015.SZ', 'name': '爱尔眼科'},
    {'symbol': '300124.SZ', 'name': '汇川技术'},
    {'symbol': '600048.SH', 'name': '保利发展'},
    {'symbol': '3690.HK', 'name': '美团-W'},
]


@dataclass
class IntradaySignal:
    """30分钟级别交易信号"""
    symbol: str
    stock_name: str
    price: float
    signal: str  # '强烈买入', '买入', '持有', '卖出', '强烈卖出'
    score: int
    reasons: List[str]
    indicators: Dict
    timestamp: str


class IntradayStrategy:
    """30分钟级别多因子策略"""

    def __init__(self, watchlist: List[Dict] = None, notify_enabled: bool = False):
        """
        Args:
            watchlist: 股票池列表，格式 [{'symbol': 'xxx', 'name': 'xxx'}, ...]
            notify_enabled: 是否启用通知
        """
        self.watchlist = watchlist or WATCHLIST_STOCKS
        self.notify_enabled = notify_enabled
        self.data_handler = DataHandler()
        self.signals_history: List[IntradaySignal] = []

        # 策略参数
        self.rsi_oversold = 30  # RSI 超卖阈值
        self.rsi_overbought = 70  # RSI 超买阈值
        self.macd_signal_threshold = 0.001  # MACD 信号阈值

    def check_all_stocks(self) -> List[Dict]:
        """检查所有股票，返回信号列表"""
        signals = []

        for stock in self.watchlist:
            symbol = stock['symbol']
            name = stock.get('name', symbol)

            try:
                signal = self._analyze_stock(symbol, name)
                if signal:
                    signals.append(asdict(signal))
                    self.signals_history.append(signal)
            except Exception as e:
                print(f"分析 {symbol} 失败: {e}")
                continue

        return signals

    def get_latest_signals(self) -> List[Dict]:
        """获取最近的信号"""
        if not self.signals_history:
            return []

        # 返回最近5条
        recent = self.signals_history[-5:]
        return [asdict(s) for s in recent]

    def _analyze_stock(self, symbol: str, name: str) -> Optional[IntradaySignal]:
        """分析单只股票"""
        try:
            # 获取数据
            df = self.data_handler.fetch_stock_data(symbol, force_refresh=False)
            if df is None or len(df) < 50:
                return None

            # 确保数据格式正确
            df = self._prepare_data(df)

            # 计算技术指标
            indicators = self._calc_indicators(df)

            # 计算评分和信号
            score, signal, reasons = self._calc_signal(indicators)

            # 获取当前价格
            current_price = float(df['close'].iloc[-1])
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            return IntradaySignal(
                symbol=symbol,
                stock_name=name,
                price=current_price,
                signal=signal,
                score=score,
                reasons=reasons,
                indicators=indicators,
                timestamp=timestamp
            )

        except Exception as e:
            print(f"分析 {symbol} 异常: {e}")
            return None

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备数据格式"""
        df = df.copy()

        # 确保日期格式
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='mixed')
        df = df.sort_values('date')

        # 确保数值列
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['close'])
        return df

    def _calc_indicators(self, df: pd.DataFrame) -> Dict:
        """计算技术指标"""
        close = df['close']

        indicators = {}

        # RSI (14)
        indicators['rsi'] = self._calc_rsi(close, 14)

        # MACD
        macd, signal, hist = self._calc_macd(close)
        indicators['macd'] = macd
        indicators['macd_signal'] = signal
        indicators['macd_hist'] = hist

        # KDJ
        kdj_k, kdj_d, kdj_j = self._calc_kdj(df, 9, 3, 3)
        indicators['kdj_k'] = kdj_k
        indicators['kdj_d'] = kdj_d
        indicators['kdj_j'] = kdj_j

        # 均线
        indicators['ma5'] = close.rolling(5).mean().iloc[-1]
        indicators['ma10'] = close.rolling(10).mean().iloc[-1]
        indicators['ma20'] = close.rolling(20).mean().iloc[-1]

        # 布林带
        bb_mid, bb_upper, bb_lower = self._calc_bollinger(close, 20, 2)
        indicators['bb_mid'] = bb_mid
        indicators['bb_upper'] = bb_upper
        indicators['bb_lower'] = bb_lower

        return indicators

    def _calc_rsi(self, close: pd.Series, period: int = 14) -> float:
        """计算 RSI"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

    def _calc_macd(self, close: pd.Series, fast=12, slow=26, signal_period=9) -> tuple:
        """计算 MACD"""
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal_period).mean()
        hist = macd - signal_line

        return (
            float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0.0,
            float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0,
            float(hist.iloc[-1]) if not pd.isna(hist.iloc[-1]) else 0.0
        )

    def _calc_kdj(self, df: pd.DataFrame, n=9, m1=3, m2=3) -> tuple:
        """计算 KDJ"""
        low_n = df['low'].rolling(n).min()
        high_n = df['high'].rolling(n).max()

        rsv = (df['close'] - low_n) / (high_n - low_n) * 100
        rsv = rsv.fillna(50)

        k = rsv.ewm(span=m1).mean()
        d = k.ewm(span=m2).mean()
        j = 3 * k - 2 * d

        return (
            float(k.iloc[-1]) if not pd.isna(k.iloc[-1]) else 50.0,
            float(d.iloc[-1]) if not pd.isna(d.iloc[-1]) else 50.0,
            float(j.iloc[-1]) if not pd.isna(j.iloc[-1]) else 50.0
        )

    def _calc_bollinger(self, close: pd.Series, period=20, std_dev=2) -> tuple:
        """计算布林带"""
        mid = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = mid + std_dev * std
        lower = mid - std_dev * std

        return (
            float(mid.iloc[-1]) if not pd.isna(mid.iloc[-1]) else float(close.iloc[-1]),
            float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else float(close.iloc[-1]),
            float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else float(close.iloc[-1])
        )

    def _calc_signal(self, indicators: Dict) -> tuple:
        """计算综合信号"""
        score = 0
        reasons = []

        rsi = indicators.get('rsi', 50)
        macd_hist = indicators.get('macd_hist', 0)
        kdj_k = indicators.get('kdj_k', 50)
        kdj_d = indicators.get('kdj_d', 50)
        kdj_j = indicators.get('kdj_j', 50)

        ma5 = indicators.get('ma5', 0)
        ma10 = indicators.get('ma10', 0)
        ma20 = indicators.get('ma20', 0)

        bb_upper = indicators.get('bb_upper', 0)
        bb_lower = indicators.get('bb_lower', 0)
        price = indicators.get('price', ma5)

        # RSI 分析
        if rsi < self.rsi_oversold:
            score += 2
            reasons.append(f"RSI 超卖 ({rsi:.1f})")
        elif rsi < 40:
            score += 1
            reasons.append(f"RSI 偏低 ({rsi:.1f})")
        elif rsi > self.rsi_overbought:
            score -= 2
            reasons.append(f"RSI 超买 ({rsi:.1f})")
        elif rsi > 60:
            score -= 1
            reasons.append(f"RSI 偏高 ({rsi:.1f})")

        # MACD 分析
        if macd_hist > self.macd_signal_threshold:
            score += 1
            reasons.append("MACD 金叉")
        elif macd_hist < -self.macd_signal_threshold:
            score -= 1
            reasons.append("MACD 死叉")

        # KDJ 分析
        if kdj_j < 20:
            score += 1
            reasons.append(f"KDJ 超卖 (J={kdj_j:.1f})")
        elif kdj_j > 80:
            score -= 1
            reasons.append(f"KDJ 超买 (J={kdj_j:.1f})")

        if kdj_k > kdj_d and kdj_k < 80:
            score += 1
            reasons.append("KDJ 向上")
        elif kdj_k < kdj_d and kdj_k > 20:
            score -= 1
            reasons.append("KDJ 向下")

        # 均线分析
        if ma5 > ma10 and ma10 > ma20:
            score += 1
            reasons.append("均线多头排列")
        elif ma5 < ma10 and ma10 < ma20:
            score -= 1
            reasons.append("均线空头排列")

        # 布林带分析
        if price < bb_lower:
            score += 1
            reasons.append("跌破布林下轨")

        # 确定信号文字
        if score >= 4:
            signal = "强烈买入"
        elif score >= 2:
            signal = "买入"
        elif score <= -4:
            signal = "强烈卖出"
        elif score <= -2:
            signal = "卖出"
        else:
            signal = "持有"

        if not reasons:
            reasons.append("指标中性，持有观望")

        return score, signal, reasons


# 测试入口
if __name__ == "__main__":
    print("测试 IntradayStrategy...")

    strategy = IntradayStrategy()
    signals = strategy.check_all_stocks()

    for sig in signals:
        print(f"\n{sig['stock_name']} ({sig['symbol']})")
        print(f"  价格: {sig['price']:.2f}")
        print(f"  信号: {sig['signal']} (评分: {sig['score']})")
        print(f"  理由: {', '.join(sig['reasons'])}")
        print(f"  RSI: {sig['indicators']['rsi']:.1f}")
        print(f"  MACD: {sig['indicators']['macd_hist']:.4f}")