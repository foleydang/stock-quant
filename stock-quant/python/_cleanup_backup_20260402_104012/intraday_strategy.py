#!/usr/bin/env python3
"""
30 分钟级别多因子交易策略
支持：爱尔眼科，汇川技术，保利地产，美团等指定股票
功能：
- 多技术指标融合（RSI, MACD, 布林带，均线系统）
- 实时信号生成
- 通知推送
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional
import requests

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_handler import DataHandler
from strategy.notifier import (
    create_notification_manager_from_env,
    format_trading_signal,
    NotificationManager
)
from strategy.lgb_predictor import LGBPredictor


class SignalType(Enum):
    """交易信号类型"""
    STRONG_BUY = "强烈买入"
    BUY = "买入"
    HOLD = "持有"
    SELL = "卖出"
    STRONG_SELL = "强烈卖出"


# 配置的股票池
WATCHLIST_STOCKS = [
    {"symbol": "300015.SZ", "name": "爱尔眼科", "category": "A 股"},
    {"symbol": "300124.SZ", "name": "汇川技术", "category": "A 股"},
    {"symbol": "600048.SH", "name": "保利发展", "category": "A 股"},  # 保利地产已更名为保利发展
    {"symbol": "600519.SH", "name": "贵州茅台", "category": "A 股"},
    {"symbol": "3690.HK", "name": "美团 -W", "category": "港股"},
    {"symbol": "0700.HK", "name": "腾讯控股", "category": "港股"},
    {"symbol": "9988.HK", "name": "阿里巴巴 -W", "category": "港股"},
]


class TechnicalIndicators:
    """技术指标计算类"""

    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """计算 RSI 指标"""
        if len(prices) < period + 1:
            return np.array([50.0])

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gains = np.zeros_like(prices, dtype=float)
        avg_losses = np.zeros_like(prices, dtype=float)

        avg_gains[period - 1] = np.mean(gains[:period])
        avg_losses[period - 1] = np.mean(losses[:period])

        for i in range(period, len(prices)):
            avg_gains[i] = (avg_gains[i - 1] * (period - 1) + gains[i - 1]) / period
            avg_losses[i] = (avg_losses[i - 1] * (period - 1) + losses[i - 1]) / period

        rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses != 0)
        rsi = 100 - (100 / (1 + rs))

        return rsi[period - 1:]

    @staticmethod
    def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Dict:
        """
        计算 ADX 趋势强度指标
        ADX > 25: 强趋势
        ADX < 20: 震荡市场
        """
        if len(close) < period * 2 + 1:
            return {"adx": np.array([20]), "plus_di": np.array([25]), "minus_di": np.array([25])}

        # 计算 TR
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        tr = np.maximum(np.maximum(tr1, tr2), tr3)

        # 计算 DM+ 和 DM-
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(low)

        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]

            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        # 平滑处理
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
        plus_di = pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr * 100
        minus_di = pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr * 100

        # 计算 DX 和 ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        dx = dx.fillna(0)
        adx = dx.ewm(span=period, adjust=False).mean()

        valid_start = period * 2 - 1
        return {
            "adx": adx.values[valid_start:],
            "plus_di": plus_di.values[valid_start:],
            "minus_di": minus_di.values[valid_start:]
        }

    @staticmethod
    def calculate_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """
        计算 OBV 能量潮指标
        价格上涨时累加成交量，下跌时累减
        """
        if len(close) < 2:
            return np.array([0])

        obv = np.zeros(len(close))
        obv[0] = volume[0]

        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]

        return obv

    @staticmethod
    def calculate_mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                      volume: np.ndarray, period: int = 14) -> np.ndarray:
        """
        计算 MFI 资金流量指标 (0-100)
        MFI > 80: 超买
        MFI < 20: 超卖
        """
        if len(close) < period + 1:
            return np.array([50])

        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume

        mfi_values = []
        for i in range(period, len(close)):
            positive_flow = 0
            negative_flow = 0
            for j in range(i - period + 1, i + 1):
                if j == 0:
                    continue
                if typical_price[j] > typical_price[j-1]:
                    positive_flow += raw_money_flow[j]
                elif typical_price[j] < typical_price[j-1]:
                    negative_flow += raw_money_flow[j]

            if negative_flow == 0:
                mfi = 100
            else:
                money_ratio = positive_flow / negative_flow
                mfi = 100 - (100 / (1 + money_ratio))
            mfi_values.append(mfi)

        return np.array(mfi_values)

    @staticmethod
    def calculate_vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                       volume: np.ndarray, window: int = 20) -> np.ndarray:
        """
        计算 VWAP 成交量加权平均价
        """
        if len(close) < window:
            return close.copy()

        typical_price = (high + low + close) / 3
        vwap = pd.Series(typical_price * volume).rolling(window=window).sum() / \
               pd.Series(volume).rolling(window=window).sum()

        return vwap.ffill().values

    @staticmethod
    def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """计算 MACD 指标"""
        if len(prices) < slow + signal:
            return {"macd": np.array([0]), "signal": np.array([0]), "histogram": np.array([0])}

        ema_fast = pd.Series(prices).ewm(span=fast, adjust=False).mean()
        ema_slow = pd.Series(prices).ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            "macd": macd_line.values[slow - 1:],
            "signal": signal_line.values[slow - 1:],
            "histogram": histogram.values[slow - 1:]
        }

    @staticmethod
    def calculate_bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Dict:
        """计算布林带"""
        if len(prices) < period:
            mid = np.array([np.mean(prices)])
            return {"upper": mid + std_dev, "mid": mid, "lower": mid - std_dev}

        mid_band = pd.Series(prices).rolling(window=period).mean()
        std = pd.Series(prices).rolling(window=period).std()
        upper_band = mid_band + (std * std_dev)
        lower_band = mid_band - (std * std_dev)

        valid_start = period - 1
        return {
            "upper": upper_band.values[valid_start:],
            "mid": mid_band.values[valid_start:],
            "lower": lower_band.values[valid_start:]
        }

    @staticmethod
    def calculate_ma(prices: np.ndarray, periods: List[int] = [5, 10, 20, 60]) -> Dict:
        """计算多条移动平均线"""
        mas = {}
        for period in periods:
            if len(prices) >= period:
                ma = pd.Series(prices).rolling(window=period).mean()
                mas[f"MA{period}"] = ma.values[period - 1:]
            else:
                mas[f"MA{period}"] = np.array([np.mean(prices)])
        return mas

    @staticmethod
    def calculate_kdj(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                      n: int = 9, m1: int = 3, m2: int = 3) -> Dict:
        """计算 KDJ 指标"""
        if len(close) < n:
            return {"k": np.array([50]), "d": np.array([50]), "j": np.array([50])}

        lowest_low = pd.Series(low).rolling(window=n).min()
        highest_high = pd.Series(high).rolling(window=n).max()

        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        rsv = rsv.fillna(50)

        k = rsv.ewm(com=m1 - 1, adjust=False).mean()
        d = k.ewm(com=m2 - 1, adjust=False).mean()
        j = 3 * k - 2 * d

        valid_start = n - 1
        return {
            "k": k.values[valid_start:],
            "d": d.values[valid_start:],
            "j": j.values[valid_start:]
        }

    @staticmethod
    def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """计算 ATR（平均真实波幅）"""
        if len(close) < period + 1:
            return np.array([np.std(high - low)])

        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        tr = np.maximum(np.maximum(tr1, tr2), tr3)

        atr = pd.Series(tr).rolling(window=period).mean()
        return atr.values[period:]


class IntradayStrategy:
    """30 分钟级别日内交易策略"""

    def __init__(self, watchlist: List[Dict] = None, notify_enabled: bool = True, force_refresh: bool = True):
        """
        初始化策略

        Args:
            watchlist: 监控股票列表
            notify_enabled: 是否启用通知
            force_refresh: 是否每次都强制刷新数据（默认 True，获取实时数据）
        """
        self.data_handler = DataHandler(force_refresh=force_refresh)
        self.watchlist = watchlist or WATCHLIST_STOCKS
        self.notify_enabled = notify_enabled
        self.force_refresh = force_refresh
        self.signals_history = []

        # 策略参数 - 增强版
        self.params = {
            # RSI 参数
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            # MACD 参数
            "macd_cross_threshold": 0.001,
            # 布林带参数
            "bb_breakout_threshold": 0.02,
            # 均线参数
            "ma_short": 5,
            "ma_mid": 20,
            "ma_long": 60,
            # ADX 参数
            "adx_trend_threshold": 25,
            "adx_weak_threshold": 20,
            # MFI 参数
            "mfi_overbought": 80,
            "mfi_oversold": 20,
            # 动态止损止盈 (基于 ATR) - 放宽止损空间
            "atr_stop_loss_mult": 3.0,  # 从2.0增加到3.0
            "atr_take_profit_mult": 4.0,  # 从3.0增加到4.0
            # 基础止损止盈 - 放宽止损
            "stop_loss_pct": 0.05,  # 从0.03增加到0.05
            "take_profit_pct": 0.08  # 从0.05增加到0.08
        }

        # 通知管理器
        self.notification_manager = create_notification_manager_from_env()

        # 缓存每个股票的最新信号
        self.latest_signals = {}

        # 缓存持仓信息 (用于动态止损止盈计算)
        self.positions = {}

        # LightGBM 预测器（使用增强模型 - 交叉验证优化版）
        # 模型使用 50 只中证 500 成分股 3 年历史数据训练，交叉验证准确率 48.78%
        self.lgb_predictor = LGBPredictor(model_dir='./models/lgb_enhanced')

    def add_notify_callback(self, callback):
        """添加通知回调函数"""
        self.notify_callbacks.append(callback)

    def fetch_data(self, symbol: str, timeframe: str = "30m", force_refresh: bool = None) -> Optional[pd.DataFrame]:
        """
        获取股票数据（30分钟级别）

        注意：限流延时已在接口层（DataHandler）内部处理，调用层无需关心

        Args:
            symbol: 股票代码
            timeframe: 时间周期（默认 30 分钟）
            force_refresh: 是否强制刷新（默认使用实例设置）

        Returns:
            DataFrame 包含 OHLCV 数据
        """
        try:
            # 直接调用数据处理器，延时已在接口层内部处理
            refresh = force_refresh if force_refresh is not None else self.force_refresh
            df = self.data_handler.fetch_stock_data(symbol, force_refresh=refresh)

            if df is None or df.empty:
                return None

            return df

        except Exception as e:
            sys.stderr.write(f"Error fetching data for {symbol}: {e}\n")
            return None

    def resample_to_30min(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将数据重采样为 30 分钟 K 线
        """
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

        # 重采样
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }

        df_30m = df.resample('30T').apply(ohlc_dict)
        df_30m = df_30m.dropna()

        return df_30m

    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict:
        """计算所有技术指标（增强版）"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_price = df['open'].values
        volume = df['volume'].values

        indicators = {}

        # === 传统指标 ===
        # RSI
        indicators['rsi'] = TechnicalIndicators.calculate_rsi(close, 14)

        # MACD
        macd_data = TechnicalIndicators.calculate_macd(close)
        indicators.update(macd_data)

        # 布林带
        bb_data = TechnicalIndicators.calculate_bollinger_bands(close, 20, 2.0)
        indicators.update(bb_data)

        # 均线系统
        mas = TechnicalIndicators.calculate_ma(close, [5, 10, 20, 60])
        indicators.update(mas)

        # KDJ
        kdj_data = TechnicalIndicators.calculate_kdj(high, low, close)
        indicators.update(kdj_data)

        # ATR
        indicators['atr'] = TechnicalIndicators.calculate_atr(high, low, close, 14)

        # === 新增增强指标 ===
        # ADX 趋势强度
        adx_data = TechnicalIndicators.calculate_adx(high, low, close, 14)
        indicators.update(adx_data)

        # OBV 能量潮
        indicators['obv'] = TechnicalIndicators.calculate_obv(close, volume)

        # MFI 资金流量
        indicators['mfi'] = TechnicalIndicators.calculate_mfi(high, low, close, volume, 14)

        # VWAP 成交量加权平均价
        indicators['vwap'] = TechnicalIndicators.calculate_vwap(high, low, close, volume, 20)

        return indicators

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """
        生成交易信号

        基于多因子综合判断：
        1. RSI 超买超卖
        2. MACD 金叉死叉
        3. 布林带突破
        4. 均线多头/空头排列
        5. KDJ 金叉死叉
        """
        if df is None or len(df) < 40:
            return None

        indicators = self.calculate_all_indicators(df)
        current_price = df['close'].iloc[-1]
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        # 信号评分系统
        score = 0
        reasons = []

        # === 优化后的评分系统：超卖优先原则 + 多指标共振 ===
        # 核心逻辑：超卖状态下寻找反转机会，多指标共振时增强信号

        # 1. RSI 信号 - 核心指标，超卖优先
        rsi = indicators['rsi'][-1]
        is_extreme_oversold = rsi < 15  # 极度超卖阈值降低
        is_oversold = rsi < 25  # 超卖阈值降低（更激进抄底）
        is_near_oversold = rsi < 40  # 接近超卖阈值提高到40

        if is_extreme_oversold:
            score += 5  # 极度超卖 - 强烈抄底信号
            reasons.append(f"★RSI 极度超卖 ({rsi:.1f})★")
        elif is_oversold:
            score += 4  # 超卖提高权重
            reasons.append(f"★RSI 超卖 ({rsi:.1f})★")
        elif is_near_oversold:
            score += 2
            reasons.append(f"RSI 接近超卖 ({rsi:.1f})")
        elif rsi < 45:
            score += 0.5
            reasons.append(f"RSI 偏低 ({rsi:.1f})")
        elif rsi > self.params['rsi_overbought']:
            score -= 4
            reasons.append(f"★RSI 超买 ({rsi:.1f})★")
        elif rsi > 65:
            score -= 2
            reasons.append(f"RSI 偏高 ({rsi:.1f})")
        elif rsi > 55:
            score -= 0.5

        # 2. MACD 信号 + 底背离检测（核心反转信号）
        macd = indicators['macd']
        signal_line = indicators['signal']
        histogram = indicators['histogram']

        if len(macd) >= 5:
            # 金叉死叉
            if macd[-1] > signal_line[-1] and macd[-2] <= signal_line[-2]:
                if is_oversold or is_extreme_oversold:
                    score += 3  # 超卖+金叉 = 强反转信号
                    reasons.append("★MACD 金叉(超卖反转)★")
                else:
                    score += 2
                    reasons.append("MACD 金叉")
            elif macd[-1] < signal_line[-1] and macd[-2] >= signal_line[-2]:
                score -= 2
                reasons.append("MACD 死叉")

            # 底背离检测：价格创新低但MACD没创新低 = 强反转信号
            if len(histogram) >= 10 and len(close) >= 10:
                price_low1 = min(close[-10:-5])
                price_low2 = min(close[-5:])
                macd_low1 = min(macd[-10:-5])
                macd_low2 = min(macd[-5:])

                if price_low2 < price_low1 and macd_low2 > macd_low1:
                    # 价格创新低但MACD没创新低 = 底背离
                    score += 4  # 底背离是强烈买入信号
                    reasons.append("★MACD 底背离★ - 强反转信号")
                # 顶背离检测 - 超卖状态下降低权重
                elif price_low2 > price_low1 and macd_low2 < macd_low1:
                    if is_oversold or is_extreme_oversold or is_near_oversold:
                        # 超卖状态下顶背离不扣分（可能是底部震荡）
                        score -= 0.5
                        reasons.append("MACD 顶背离(超卖状态-忽略)")
                    else:
                        score -= 2  # 降低顶背离负面权重
                        reasons.append("MACD 顶背离")

            # 柱状图变化趋势
            if histogram[-1] > histogram[-2] and histogram[-1] > 0:
                score += 1
                reasons.append("MACD 红柱放大")
            elif histogram[-1] < histogram[-2] and histogram[-1] < 0:
                # 超卖状态下绿柱缩小是好事
                if is_oversold or is_extreme_oversold:
                    score += 1
                    reasons.append("MACD 绿柱缩小(超卖反弹)")
                else:
                    score -= 1
                    reasons.append("MACD 绿柱放大")

            # MACD 在零轴下方且开始向上 = 底部反转
            if macd[-1] < 0 and macd[-1] > macd[-3] and is_oversold:
                score += 2
                reasons.append("MACD 零轴下反弹(超卖)")

        # 3. 布林带信号 - 跌破下轨是强烈抄底信号
        current_price = float(current_price)
        upper_bb = float(indicators['upper'][-1])
        lower_bb = float(indicators['lower'][-1])
        mid_bb = float(indicators['mid'][-1])

        bb_position = (current_price - lower_bb) / (upper_bb - lower_bb)  # 布林带位置百分比

        if current_price < lower_bb * (1 - self.params['bb_breakout_threshold']):
            score += 3  # 跌破下轨提高权重 - 强烈抄底信号
            reasons.append(f"跌破下轨 ({current_price:.2f} < {lower_bb:.2f}) - 抄底信号")
        elif current_price < lower_bb:
            score += 2
            reasons.append(f"触及下轨 ({current_price:.2f} ≈ {lower_bb:.2f})")
        elif current_price > upper_bb * (1 + self.params['bb_breakout_threshold']):
            score -= 3
            reasons.append(f"突破上轨 ({current_price:.2f} > {upper_bb:.2f})")
        elif current_price > upper_bb:
            score -= 2
            reasons.append(f"触及上轨 ({current_price:.2f} ≈ {upper_bb:.2f})")
        elif current_price < mid_bb:
            score -= 0.5
        else:
            score += 0.5

        # 4. 均线系统信号 - 超卖状态下降低负面权重（避免追涨杀跌）
        ma5 = float(indicators.get('MA5', [current_price])[-1])
        ma10 = float(indicators.get('MA10', [current_price])[-1])
        ma20 = float(indicators.get('MA20', [current_price])[-1])
        ma60 = float(indicators.get('MA60', [current_price])[-1])

        # 多头排列
        if ma5 > ma10 > ma20 > ma60:
            score += 2
            reasons.append("均线多头排列")
        # 空头排列 - 超卖状态下视为抄底机会
        elif ma5 < ma10 < ma20 < ma60:
            if is_oversold or is_extreme_oversold:
                # 超卖+空头排列 = 强抄底机会，不扣分甚至加分
                score += 1
                reasons.append("★空头排列+超卖=抄底机会★")
            elif is_near_oversold:
                score -= 0.5  # 接近超卖只扣少量分
                reasons.append("均线空头排列(接近超卖)")
            else:
                score -= 2
                reasons.append("均线空头排列")
        # 短期均线关系
        elif ma5 > ma10:
            score += 1
            reasons.append("短期均线向上")
        elif ma5 < ma10:
            if is_oversold or is_extreme_oversold:
                pass  # 超卖状态下不扣分
            elif is_near_oversold:
                score -= 0.25
            else:
                score -= 1
                reasons.append("短期均线向下")

        # 5. KDJ 信号 - 增强超卖检测
        k = float(indicators['k'][-1]) if len(indicators['k']) > 0 else 50.0
        d = float(indicators['d'][-1]) if len(indicators['d']) > 0 else 50.0
        j = float(indicators['j'][-1]) if len(indicators['j']) > 0 else 50.0

        kdj_oversold = k < 20 or j < 10  # KDJ 超卖
        kdj_overbought = k > 80 or j > 100  # KDJ 超买

        if kdj_oversold:
            score += 3  # KDJ 超卖 = 强抄底信号
            reasons.append(f"★KDJ 超卖 (K={k:.1f}, J={j:.1f})★")
        elif k < 30:
            score += 1.5
            reasons.append(f"KDJ 偏低 (K={k:.1f})")

        if kdj_overbought:
            score -= 3
            reasons.append(f"★KDJ 超买 (K={k:.1f}, J={j:.1f})★")

        if len(indicators['k']) >= 2 and len(indicators['d']) >= 2:
            # 金叉在低位更有效
            if k > d and indicators['k'][-2] <= indicators['d'][-2]:
                if k < 50:  # 低位金叉
                    score += 2
                    reasons.append("★KDJ 低位金叉★")
                else:
                    score += 1
                    reasons.append("KDJ 金叉")
            # 死叉在高位更危险
            elif k < d and indicators['k'][-2] >= indicators['d'][-2]:
                if k > 50:  # 高位死叉
                    score -= 2
                    reasons.append("★KDJ 高位死叉★")
                else:
                    score -= 1
                    reasons.append("KDJ 死叉")

        # 6. ADX 趋势强度指标 - 超卖状态下调整权重
        adx = float(indicators['adx'][-1]) if len(indicators['adx']) > 0 else 20
        plus_di = float(indicators['plus_di'][-1]) if len(indicators['plus_di']) > 0 else 25
        minus_di = float(indicators['minus_di'][-1]) if len(indicators['minus_di']) > 0 else 25

        if adx > self.params['adx_trend_threshold']:
            # 强趋势市场
            if plus_di > minus_di + 5:
                score += 2
                reasons.append(f"ADX 强趋势向上 ({adx:.1f})")
            elif minus_di > plus_di + 5:
                # 强趋势向下时，超卖状态视为抄底机会
                if is_extreme_oversold:
                    score += 2  # 极度超卖+强趋势向下 = 抄底反弹机会
                    reasons.append(f"★ADX强趋势下+极度超卖=反弹机会 ({adx:.1f})★")
                elif is_oversold:
                    score += 1  # 超卖状态下视为机会
                    reasons.append(f"ADX 强趋势向下+超卖 ({adx:.1f})")
                elif is_near_oversold:
                    pass  # 接近超卖不扣分
                    reasons.append(f"ADX 强趋势向下+接近超卖 ({adx:.1f})")
                else:
                    score -= 2
                    reasons.append(f"ADX 强趋势向下 ({adx:.1f})")
        elif adx < self.params['adx_weak_threshold']:
            # 震荡市场，区间操作 - 提高权重
            if rsi < 40:
                score += 1.5  # 提高权重
                reasons.append(f"ADX 震荡市+RSI偏低 ({adx:.1f})")
            elif rsi > 60:
                score -= 1.5
                reasons.append(f"ADX 震荡市+RSI偏高 ({adx:.1f})")

        # 7. MFI 资金流量指标 - 提高超卖权重
        mfi = float(indicators['mfi'][-1]) if len(indicators['mfi']) > 0 else 50
        if mfi < self.params['mfi_oversold']:
            score += 3  # 提高权重
            reasons.append(f"MFI 超卖 ({mfi:.1f}) - 资金枯竭抄底信号")
        elif mfi < 30:
            score += 2
            reasons.append(f"MFI 偏低 ({mfi:.1f})")
        elif mfi > self.params['mfi_overbought']:
            score -= 3
            reasons.append(f"MFI 超买 ({mfi:.1f})")
        elif mfi > 70:
            score -= 1
            reasons.append(f"MFI 偏高 ({mfi:.1f})")

        # 8. OBV 能量潮确认 - 超卖状态下降低负面权重
        obv = indicators['obv']
        obv_current = float(obv[-1]) if len(obv) > 0 else 0
        obv_prev = float(obv[-10]) if len(obv) >= 10 else obv_current

        # 收盘价变化（使用已定义的 close 变量）
        close_current = float(close[-1])
        close_prev = float(close[-10]) if len(close) >= 10 else close_current

        obv_trend = obv_current - obv_prev
        price_trend = close_current - close_prev

        if obv_trend > 0 and price_trend <= 0:
            score += 2  # 提高权重 - 底部资金流入是强烈买入信号
            reasons.append("OBV 资金流入背离 - 底部吸筹信号")
        elif obv_trend < 0 and price_trend >= 0:
            score -= 1
            reasons.append("OBV 资金流出背离")
        elif obv_trend > 0:
            score += 1
            reasons.append("OBV 资金流入")
        elif obv_trend < 0:
            # 超卖状态下，资金流出不扣太多分（恐慌抛售是正常的）
            if is_oversold or is_extreme_oversold:
                score -= 0.25
                reasons.append("OBV 资金流出(超卖状态)")
            else:
                score -= 0.5
                reasons.append("OBV 资金流出")

        # 9. VWAP 成交量加权平均价 - 提高权重
        vwap = float(indicators['vwap'][-1]) if len(indicators['vwap']) > 0 else current_price
        vwap_deviation = (vwap - current_price) / vwap * 100

        if current_price < vwap * 0.95:  # 大幅低于VWAP
            score += 2
            reasons.append(f"价格大幅低于 VWAP ({vwap_deviation:.1f}%) - 抄底机会")
        elif current_price < vwap * 0.98:
            score += 1.5
            reasons.append(f"价格低于 VWAP ({vwap_deviation:.1f}%)")
        elif current_price > vwap * 1.05:  # 大幅高于VWAP
            score -= 2
            reasons.append(f"价格大幅高于 VWAP ({-vwap_deviation:.1f}%)")
        elif current_price > vwap * 1.02:
            score -= 1.5
            reasons.append(f"价格高于 VWAP ({-vwap_deviation:.1f}%)")

        # 10. LightGBM ML 预测 - 在超卖状态下提高权重
        try:
            ml_score = self.lgb_predictor.get_signal_score(df, symbol)
            if ml_score != 0:
                # 超卖状态下ML看涨预测权重更高
                ml_weight = 0.5
                if (is_oversold or is_extreme_oversold) and ml_score > 0:
                    ml_weight = 1.0  # 超卖+ML看涨 = 提高权重
                    reasons.append(f"ML 预测看涨 (超卖状态权重+{ml_score*ml_weight:.1f}分)")
                elif ml_score > 0:
                    reasons.append(f"ML 预测看涨 (+{ml_score*ml_weight:.1f}分)")
                else:
                    reasons.append(f"ML 预测看跌 ({ml_score*ml_weight:.1f}分)")
                score += ml_score * ml_weight
        except Exception as e:
            # ML 预测失败不影响其他指标
            pass

        # 11. 多指标共振检测 - 多个超卖信号同时出现时增强
        oversold_count = 0
        if is_oversold or is_extreme_oversold:
            oversold_count += 1
        if kdj_oversold:
            oversold_count += 1
        if mfi < 25:
            oversold_count += 1
        if current_price < lower_bb:
            oversold_count += 1
        if current_price < vwap * 0.97:
            oversold_count += 1

        if oversold_count >= 3:
            score += 3  # 多指标共振超卖
            reasons.append(f"★{oversold_count}个指标共振超卖★")

        # 12. 连续下跌 + 超卖 = 反转信号
        if len(close) >= 5:
            down_days = sum(1 for i in range(1, 6) if close[-i] < close[-i-1])
            if down_days >= 4 and (is_oversold or is_extreme_oversold):
                score += 3
                reasons.append(f"★连续{down_days}根阴线+超卖=反转信号★")
            elif down_days >= 3 and is_near_oversold:
                score += 1
                reasons.append(f"连续{down_days}根阴线+接近超卖")

        # 13. 量价关系 - 放量下跌后的缩量 = 底部信号
        if len(volume) >= 5:
            recent_vol = volume[-1]
            avg_vol_5 = np.mean(volume[-5:])
            avg_vol_10 = np.mean(volume[-10:]) if len(volume) >= 10 else avg_vol_5

            # 缩量下跌（量价背离）
            if close[-1] < close[-2] and recent_vol < avg_vol_5 * 0.6:
                if is_oversold or is_extreme_oversold:
                    score += 2
                    reasons.append("★缩量下跌(抛售枯竭)★")
            # 放量上涨
            elif close[-1] > close[-2] and recent_vol > avg_vol_5 * 1.5:
                if is_oversold or is_near_oversold:
                    score += 2
                    reasons.append("★放量上涨(底部启动)★")

        # 14. 趋势过滤 - 避免在强势下跌趋势中频繁买入（除非超卖）
        if len(close) >= 20:
            trend_20 = (close[-1] - close[-20]) / close[-20] * 100
            trend_5 = (close[-1] - close[-5]) / close[-5] * 100

            # 强势下跌趋势（20周期跌幅超过5%）
            if trend_20 < -5:
                if not (is_oversold or is_extreme_oversold or oversold_count >= 3):
                    score -= 2  # 非超卖状态下减少买入
                    reasons.append(f"20周期下跌趋势({trend_20:.1f}%)")
                elif is_extreme_oversold:
                    score += 1  # 极度超卖+强下跌=反弹机会
                    reasons.append(f"★强下跌+极度超卖=反弹机会({trend_20:.1f}%)★")
            # 近期强势反弹
            elif trend_5 > 3 and trend_20 < -2:
                score += 2  # 底部反弹信号
                reasons.append(f"★底部反弹信号(5周期+{trend_5:.1f}%)★")

        # 确定信号类型 - 优化阈值，更容易触发买入信号
        # 新阈值：降低买入门槛，让超卖机会更容易被捕捉
        if score >= 4:
            signal_type = SignalType.STRONG_BUY
        elif score >= 1:  # 大幅降低买入门槛
            signal_type = SignalType.BUY
        elif score >= -2:  # 扩大持有区间
            signal_type = SignalType.HOLD
        elif score >= -5:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.STRONG_SELL

        # 特殊处理：超卖状态下强制买入信号
        if is_extreme_oversold:  # RSI < 15 极度超卖
            signal_type = SignalType.STRONG_BUY
            reasons.append("★★★ 极度超卖强烈买入信号 ★★★")
        elif is_oversold and score >= -1:  # RSI < 25 超卖，评分>=-1就买入
            signal_type = SignalType.BUY
            reasons.append("★ 超卖买入信号 ★")
        elif oversold_count >= 3:  # 多指标共振超卖
            signal_type = SignalType.BUY
            reasons.append("★ 多指标共振买入信号 ★")

        signal = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "price": current_price,
            "signal": signal_type.value,
            "score": score,
            "reasons": reasons,
            "indicators": {
                "rsi": float(rsi),
                "macd": float(macd[-1]) if len(macd) > 0 else 0,
                "macd_signal": float(signal_line[-1]) if len(signal_line) > 0 else 0,
                "kdj_k": float(k),
                "kdj_d": float(d),
                "kdj_j": float(j),
                "adx": float(adx),
                "plus_di": float(plus_di),
                "minus_di": float(minus_di),
                "mfi": float(mfi),
                "obv": float(obv[-1]) if len(obv) > 0 else 0,
                "vwap": float(vwap),
                "upper_bb": upper_bb,
                "mid_bb": mid_bb,
                "lower_bb": lower_bb,
                "ma5": ma5,
                "ma10": ma10,
                "ma20": ma20,
                "ma60": ma60,
                "atr": float(indicators['atr'][-1]) if len(indicators['atr']) > 0 else 0
            }
        }

        # 计算动态止损止盈价位 (基于 ATR)
        atr = float(indicators['atr'][-1]) if len(indicators['atr']) > 0 else 0
        if atr > 0:
            signal['stop_loss'] = current_price - atr * self.params['atr_stop_loss_mult']
            signal['take_profit'] = current_price + atr * self.params['atr_take_profit_mult']
        else:
            signal['stop_loss'] = current_price * (1 - self.params['stop_loss_pct'])
            signal['take_profit'] = current_price * (1 + self.params['take_profit_pct'])

        # 添加 ML 预测结果
        try:
            ml_result = self.lgb_predictor.predict(df, symbol)
            if ml_result:
                signal['ml_prediction'] = ml_result['label']
                signal['ml_confidence'] = ml_result['confidence']
                signal['ml_probabilities'] = ml_result['probabilities']
        except Exception:
            pass

        return signal

    def send_notification(self, signal: Dict):
        """发送交易信号通知"""
        if not self.notify_enabled:
            return

        title = f"交易信号：{signal.get('stock_name', signal['symbol'])} - {signal['signal']}"
        content = format_trading_signal(signal)

        # 使用通知管理器发送
        results = self.notification_manager.send(title, content)

        # 保存到历史
        self.signals_history.append({
            "timestamp": datetime.now().isoformat(),
            "signal": signal
        })

        # 写入通知日志
        self._write_notification_log(signal, content)

    def _write_notification_log(self, signal: Dict, message: str):
        """写入通知日志文件"""
        log_dir = os.path.join(os.path.dirname(__file__), '../logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(log_dir, f"signals_{datetime.now().strftime('%Y%m%d')}.log")

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Timestamp: {signal['timestamp']}\n")
            f.write(f"Symbol: {signal['symbol']}\n")
            f.write(f"Signal: {signal['signal']}\n")
            f.write(f"Price: {signal['price']:.2f}\n")
            f.write(f"Score: {signal['score']}\n")
            f.write(f"Reasons: {', '.join(signal['reasons'])}\n")
            f.write(f"{'='*60}\n")

    def check_all_stocks(self) -> List[Dict]:
        """
        检查所有监控股票

        注意：限流延时已在接口层（DataHandler）内部处理，调用层无需关心
        """
        results = []
        total = len(self.watchlist)

        print(f"\n{'='*60}")
        print(f"开始获取 30 分钟级别数据 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"监控股票：{total} 只")
        print(f"{'='*60}\n")

        for i, stock in enumerate(self.watchlist):
            symbol = stock['symbol']
            name = stock['name']

            print(f"[{i+1}/{total}] {name} ({symbol})...")

            # 获取数据（延时在接口层内部处理）
            df = self.fetch_data(symbol, force_refresh=True)

            if df is None or len(df) < 60:
                print(f"  ⚠️  数据不足 ({len(df) if df is not None else 0} 条)")
                continue

            # 显示最新数据
            latest_price = df['close'].iloc[-1]
            latest_date = df['date'].iloc[-1]
            print(f"  ✓ 价格: {latest_price:.2f} | 时间: {latest_date}")

            signal = self.generate_signal(symbol, df)

            if signal:
                signal['stock_name'] = name
                self.latest_signals[symbol] = signal
                results.append(signal)

                # 显示信号
                emoji = "🔴" if "卖出" in signal['signal'] else "🟢" if "买入" in signal['signal'] else "⚪"
                print(f"  {emoji} 信号: {signal['signal']} (评分: {signal['score']})")

                # 发送通知
                if signal['signal'] != SignalType.HOLD.value:
                    self.send_notification(signal)

        return results

    def get_latest_signals(self) -> Dict:
        """获取最新的所有信号"""
        return self.latest_signals

    def get_signals_history(self, limit: int = 100) -> List[Dict]:
        """获取信号历史"""
        return self.signals_history[-limit:]


def main():
    """主函数"""
    print("=" * 60)
    print("30 分钟级别多因子交易策略 - 实时数据版")
    print("=" * 60)
    print(f"监控股票池：{len(WATCHLIST_STOCKS)} 只")
    for stock in WATCHLIST_STOCKS:
        print(f"  • {stock['name']} ({stock['symbol']})")
    print("=" * 60)

    # 创建策略实例 - 强制刷新获取实时数据
    strategy = IntradayStrategy(
        watchlist=WATCHLIST_STOCKS,
        notify_enabled=True,
        force_refresh=True  # 每次运行都获取最新数据
    )

    print(f"通知管理器：{list(strategy.notification_manager.channels.keys())}")
    print(f"数据刷新模式：强制刷新（获取实时数据）")
    print("=" * 60)

    # 检查所有股票
    signals = strategy.check_all_stocks()

    # 输出汇总
    print("\n" + "=" * 60)
    print("信号汇总")
    print("=" * 60)

    for signal in signals:
        emoji = "🔴" if "卖出" in signal['signal'] else "🟢" if "买入" in signal['signal'] else "⚪"
        print(f"{emoji} {signal['stock_name']} ({signal['symbol']}): {signal['signal']} (评分：{signal['score']})")

    if not signals:
        print("暂无信号")

    print("=" * 60)

    return signals


if __name__ == "__main__":
    main()
