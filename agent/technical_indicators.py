#!/usr/bin/env python3
"""
技术指标计算模块

从数据库读取历史K线数据，计算常用技术指标：
- 均线系统（MA5/10/20/60）
- MACD（DIF/DEA/柱状）
- RSI（14日）
- KDJ
- 布林带（20日）
- 量比
- 支撑压力位
- 动态异动阈值
"""

import logging
import math
import os
import sqlite3
import sys
from typing import Dict, List, Optional, Tuple

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.join(os.path.dirname(AGENT_DIR), 'python')
sys.path.insert(0, AGENT_DIR)
sys.path.insert(0, PYTHON_DIR)

from config import DB_PATH

logger = logging.getLogger("feishu_bot")


def _hk_to_eastmoney_secid(symbol: str) -> Optional[str]:
    """港股代码转东方财富 secid 格式
    0700.HK -> 116.00700
    9988.HK -> 116.09988
    """
    if not symbol.endswith('.HK'):
        return None
    code = symbol.replace('.HK', '')
    # 港股代码补齐5位
    padded = code.zfill(5)
    return f'116.{padded}'


def _fetch_hk_kline_from_eastmoney(symbol: str, days: int = 120) -> Optional[List[Dict]]:
    """从东方财富获取港股K线数据"""
    secid = _hk_to_eastmoney_secid(symbol)
    if not secid:
        return None

    try:
        import requests
        url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
        params = {
            'secid': secid,
            'fields1': 'f1,f2,f3,f4,f5,f6',
            'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
            'klt': '101',  # 日K
            'fqt': '1',    # 前复权
            'beg': '0',
            'end': '20500101',
            'lmt': str(days),
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        klines = data.get('data', {}).get('klines', [])
        if not klines:
            return None

        # 东方财富K线格式: date,open,close,high,low,volume,amount,amplitude,change_pct,change,turnover
        result = []
        for line in klines:
            parts = line.split(',')
            if len(parts) >= 6:
                result.append({
                    'date': parts[0],
                    'open': float(parts[1]),
                    'close': float(parts[2]),
                    'high': float(parts[3]),
                    'low': float(parts[4]),
                    'volume': float(parts[5]),
                })
        # 取最近days条
        result = result[-days:]
        return result
    except Exception as e:
        logger.warning(f"东方财富港股K线获取失败 {symbol}: {e}")
        return None


def _get_kline(symbol: str, days: int = 120) -> Optional[List[Dict]]:
    """获取K线数据（优先数据库，港股用东方财富，A股用 DataHandler）"""
    # 先从数据库取
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT date, open, high, low, close, volume "
        "FROM kline_daily WHERE symbol=? ORDER BY date DESC LIMIT ?",
        (symbol, days)
    )
    rows = cursor.fetchall()
    conn.close()

    if rows and len(rows) >= 30:
        rows.reverse()
        return [
            {'date': r[0], 'open': float(r[1] or 0), 'high': float(r[2] or 0),
             'low': float(r[3] or 0), 'close': float(r[4] or 0), 'volume': float(r[5] or 0)}
            for r in rows
        ]

    # 港股：从东方财富获取
    if symbol.endswith('.HK'):
        hk_data = _fetch_hk_kline_from_eastmoney(symbol, days)
        if hk_data and len(hk_data) >= 30:
            # 同步写入数据库供下次使用
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                for k in hk_data:
                    cursor.execute(
                        "INSERT OR IGNORE INTO kline_daily (symbol, date, open, high, low, close, volume) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (symbol, k['date'], k['open'], k['high'], k['low'], k['close'], k['volume'])
                    )
                conn.commit()
                conn.close()
                logger.info(f"港股K线同步入库 {symbol}: {len(hk_data)} 条")
            except Exception as e:
                logger.warning(f"港股K线入库失败 {symbol}: {e}")
            return hk_data

    # A股：用 DataHandler 从 Tushare 获取
    try:
        from data.data_handler import DataHandler
        dh = DataHandler(force_refresh=True)
        df = dh.fetch_stock_data(symbol, days=days)
        if df is None or df.empty:
            return None

        result = []
        for _, row in df.iterrows():
            result.append({
                'date': str(row.get('trade_date', row.get('date', ''))),
                'open': float(row.get('open', 0)),
                'high': float(row.get('high', 0)),
                'low': float(row.get('low', 0)),
                'close': float(row.get('close', 0)),
                'volume': float(row.get('vol', row.get('volume', 0))),
            })
        return result
    except Exception as e:
        logger.error(f"获取K线失败 {symbol}: {e}")
        return None


# ========== 均线 ==========

def calc_ma(closes: List[float], period: int) -> List[float]:
    """计算移动平均线"""
    if len(closes) < period:
        return []
    result = []
    for i in range(period - 1, len(closes)):
        result.append(sum(closes[i - period + 1:i + 1]) / period)
    return result


# ========== MACD ==========

def calc_ema(data: List[float], period: int) -> List[float]:
    """计算EMA"""
    if len(data) < period:
        return []
    ema = [sum(data[:period]) / period]
    multiplier = 2 / (period + 1)
    for price in data[period:]:
        ema.append((price - ema[-1]) * multiplier + ema[-1])
    return ema


def calc_macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
    """计算MACD"""
    ema_fast = calc_ema(closes, fast)
    ema_slow = calc_ema(closes, slow)
    if len(ema_fast) < slow - fast + signal:
        return {'dif': [], 'dea': [], 'macd_bar': []}

    # DIF = EMA_fast - EMA_slow
    offset = len(ema_fast) - len(ema_slow)
    dif = [ema_fast[i + offset] - ema_slow[i] for i in range(len(ema_slow))]
    dea = calc_ema(dif, signal)
    # 柱状 = 2*(DIF - DEA)
    offset2 = len(dif) - len(dea)
    macd_bar = [2 * (dif[i + offset2] - dea[i]) for i in range(len(dea))]
    return {'dif': dif, 'dea': dea, 'macd_bar': macd_bar}


# ========== RSI ==========

def calc_rsi(closes: List[float], period: int = 14) -> List[float]:
    """计算RSI"""
    if len(closes) < period + 1:
        return []
    gains = []
    losses = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsi_values = []

    if avg_loss == 0:
        rsi_values.append(100)
    else:
        rsi_values.append(100 - 100 / (1 + avg_gain / avg_loss))

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi_values.append(100)
        else:
            rsi_values.append(100 - 100 / (1 + avg_gain / avg_loss))

    return rsi_values


# ========== KDJ ==========

def calc_kdj(kline: List[Dict], period: int = 9) -> Dict:
    """计算KDJ"""
    if len(kline) < period:
        return {'k': [], 'd': [], 'j': []}

    k_values = []
    d_values = []
    j_values = []

    prev_k = 50
    prev_d = 50

    for i in range(period - 1, len(kline)):
        high_n = max(k['high'] for k in kline[i - period + 1:i + 1])
        low_n = min(k['low'] for k in kline[i - period + 1:i + 1])
        close = kline[i]['close']

        if high_n == low_n:
            rsv = 50
        else:
            rsv = (close - low_n) / (high_n - low_n) * 100

        k = 2 / 3 * prev_k + 1 / 3 * rsv
        d = 2 / 3 * prev_d + 1 / 3 * k
        j = 3 * k - 2 * d

        k_values.append(k)
        d_values.append(d)
        j_values.append(j)

        prev_k = k
        prev_d = d

    return {'k': k_values, 'd': d_values, 'j': j_values}


# ========== 布林带 ==========

def calc_boll(closes: List[float], period: int = 20, std_dev: float = 2) -> Dict:
    """计算布林带"""
    ma = calc_ma(closes, period)
    if len(ma) < 2:
        return {'upper': [], 'middle': [], 'lower': []}

    upper, middle, lower = [], [], []
    offset = period - 1

    for i in range(len(ma)):
        start = i
        end = i + period
        segment = closes[start:end]
        std = math.sqrt(sum((x - ma[i]) ** 2 for x in segment) / period)
        upper.append(ma[i] + std_dev * std)
        middle.append(ma[i])
        lower.append(ma[i] - std_dev * std)

    return {'upper': upper, 'middle': middle, 'lower': lower}


# ========== 量比 ==========

def calc_volume_ratio(kline: List[Dict], period: int = 5) -> float:
    """计算量比 = 当前成交量 / 过去N日平均成交量"""
    if len(kline) < period + 1:
        return 1.0
    recent_vol = kline[-1]['volume']
    avg_vol = sum(k['volume'] for k in kline[-period - 1:-1]) / period
    if avg_vol == 0:
        return 1.0
    return recent_vol / avg_vol


# ========== 支撑压力位 ==========

def calc_support_resistance(kline: List[Dict], levels: int = 3, cost_price: float = None,
                            is_etf: bool = False, is_hk: bool = False) -> Dict:
    """计算支撑位和压力位
    
    基于多源融合：
    1. 近期高低点（20日/60日极值，100%真实成交价位）
    2. 斐波那契回撤（38.2%/50%/61.8%，交易员最常用的技术位）
    3. 均线（MA5/10/20/60，机构和大户的关键参考）
    4. 整数关口（8.00/8.50等，心理价位）
    5. 持仓成本价（持仓者的心理关口）
    
    规则：支撑 < 当前价（差距>1%，ETF用0.3%），压力 > 当前价（差距>1%，ETF用0.3%）
    """
    if len(kline) < 20:
        return {'supports': [], 'resistances': []}

    recent = kline[-60:] if len(kline) >= 60 else kline
    closes = [k['close'] for k in recent]
    highs = [k['high'] for k in recent]
    lows = [k['low'] for k in recent]
    volumes = [k.get('volume', 0) for k in recent]
    current = closes[-1]
    
    # ETF波动小，用更小的最小间距（0.3% vs 1%），否则会过滤掉所有有效支撑压力位
    min_gap_pct = 0.003 if is_etf else 0.01
    support_candidates = []
    resistance_candidates = []

    # ====== 1. 近期高低点（最近20日和60日的极值，真实成交位） ======
    high_60d = max(highs)
    low_60d = min(lows)
    high_20d = max(highs[-20:]) if len(highs) >= 20 else high_60d
    low_20d = min(lows[-20:]) if len(lows) >= 20 else low_60d
    
    for label, val in [("60日高", high_60d), ("20日高", high_20d), ("60日低", low_60d), ("20日低", low_20d)]:
        if 0 < val < current and (current - val) / current >= min_gap_pct:
            support_candidates.append(val)
        elif val > current and (val - current) / current >= min_gap_pct:
            resistance_candidates.append(val)

    # ====== 2. 斐波那契回撤（交易员最常用的技术位） ======
    fib_range = high_60d - low_60d
    if fib_range > 0:
        # 上升趋势：从低点向上回撤 | 下跌趋势：从高点向下回撤
        # 两个方向都算，取靠近当前价的
        fib_levels_up = [low_60d + fib_range * r for r in [0.236, 0.382, 0.5, 0.618, 0.786]]
        fib_levels_down = [high_60d - fib_range * r for r in [0.236, 0.382, 0.5, 0.618, 0.786]]
        
        for f in fib_levels_up + fib_levels_down:
            f = round(f, 2)
            if f < current and (current - f) / current >= min_gap_pct:
                support_candidates.append(f)
            elif f > current and (f - current) / current >= min_gap_pct:
                resistance_candidates.append(f)

    # ====== 3. 均线（MA5/10/20/60，机构和趋势跟踪者的关键参考） ======
    for period in [5, 10, 20, 60]:
        ma_vals = calc_ma(closes, period)
        if ma_vals:
            ma = ma_vals[-1]
            if ma < current and (current - ma) / current >= min_gap_pct:
                support_candidates.append(ma)
            elif ma > current and (ma - current) / current >= min_gap_pct:
                resistance_candidates.append(ma)

    # ====== 4. 整数关口（心理价位，根据股价量级和品种决定间隔） ======
    if is_etf:
        # ETF价格通常较低（0.5~5元），波动小，需要更细的步长
        if current >= 10:
            step = 0.5
        elif current >= 5:
            step = 0.2
        elif current >= 1:
            step = 0.1
        else:
            step = 0.05
    else:
        # 个股（含港股）
        if current >= 100:
            step = 5.0
        elif current >= 50:
            step = 2.0
        elif current >= 10:
            step = 1.0
        elif current >= 5:
            step = 0.5
        else:
            step = 0.2
    
    for i in range(-3, 4):
        round_price = round(round(current / step) * step + i * step, 2)
        if round_price <= 0:
            continue
        if round_price < current and (current - round_price) / current >= min_gap_pct:
            support_candidates.append(round_price)
        elif round_price > current and (round_price - current) / current >= min_gap_pct:
            resistance_candidates.append(round_price)

    # ====== 5. 持仓成本价 ======
    if cost_price:
        if cost_price < current and (current - cost_price) / current >= min_gap_pct:
            support_candidates.append(cost_price)
        elif cost_price > current and (cost_price - current) / current >= min_gap_pct:
            resistance_candidates.append(cost_price)

    # ====== 去重合并（相近价位合并，取均值） ======
    # ETF用更小的去重阈值（0.3%），个股用0.5%
    dedup_threshold = 0.003 if is_etf else 0.005
    
    def dedup_close_prices(prices: list, threshold: float = None) -> list:
        """合并距离<threshold的相近价位"""
        if threshold is None:
            threshold = dedup_threshold
        if not prices:
            return []
        sorted_prices = sorted(set(prices))
        merged = []
        group = [sorted_prices[0]]
        # 根据价格量级决定小数位（ETF价格<1用3位，否则2位）
        decimal_places = 3 if is_etf else 2
        for p in sorted_prices[1:]:
            if (p - group[-1]) / group[-1] < threshold:
                group.append(p)
            else:
                merged.append(round(sum(group) / len(group), decimal_places))
                group = [p]
        merged.append(round(sum(group) / len(group), decimal_places))
        return merged

    supports = dedup_close_prices(support_candidates)
    resistances = dedup_close_prices(resistance_candidates)

    # ====== 排序：离当前价越近越优先 ======
    supports = sorted(supports, reverse=True)[:levels]  # 降序 = 离当前价最近的在前面
    resistances = sorted(resistances)[:levels]  # 升序 = 离当前价最近的在前面

    # ====== 兜底：如果仍然不够，用斐波那契补充 ======
    if len(supports) < levels:
        existing = set(supports)
        fib_s = [round(low_60d + fib_range * r, 2) for r in [0.236, 0.382, 0.5, 0.618, 0.786]]
        fib_s += [round(high_60d - fib_range * r, 2) for r in [0.236, 0.382, 0.5, 0.618, 0.786]]
        for f in sorted(set(fib_s), reverse=True):
            if f < current and (current - f) / current >= min_gap_pct and f not in existing and f > 0:
                supports.append(f)
                existing.add(f)
                if len(supports) >= levels:
                    break
        supports.sort(reverse=True)

    if len(resistances) < levels:
        existing = set(resistances)
        fib_r = [round(low_60d + fib_range * r, 2) for r in [0.236, 0.382, 0.5, 0.618, 0.786]]
        fib_r += [round(high_60d - fib_range * r, 2) for r in [0.236, 0.382, 0.5, 0.618, 0.786]]
        for f in sorted(set(fib_r)):
            if f > current and (f - current) / current >= min_gap_pct and f not in existing:
                resistances.append(f)
                existing.add(f)
                if len(resistances) >= levels:
                    break
        resistances.sort()

    # ====== 持仓成本价优先级最高 ======
    if cost_price:
        if cost_price < current:
            supports = [cost_price] + [s for s in supports if abs(s - cost_price) / cost_price >= 0.01]
            supports = supports[:levels]
        elif cost_price > current:
            resistances = [r for r in resistances if abs(r - cost_price) / cost_price >= 0.01]
            resistances.append(cost_price)
            resistances.sort()
            if cost_price not in resistances[-levels:]:
                resistances = resistances[-(levels-1):] + [cost_price] if len(resistances) >= levels else [cost_price] + resistances
            resistances = resistances[-levels:]

    return {
        'supports': supports[:levels],
        'resistances': resistances[:levels],
        'current': round(current, 2),
        'cost_price': cost_price
    }


# ========== 动态异动阈值 ==========

def calc_dynamic_threshold(kline: List[Dict], base_pct: float = 1.0) -> float:
    """根据历史波动率计算动态异动阈值

    公式: threshold = max(base_pct, 1.5σ)
    σ = 20日日收益率标准差
    """
    if len(kline) < 21:
        return base_pct

    recent = kline[-21:]
    returns = []
    for i in range(1, len(recent)):
        if recent[i - 1]['close'] > 0:
            returns.append((recent[i]['close'] - recent[i - 1]['close']) / recent[i - 1]['close'])

    if not returns:
        return base_pct

    avg = sum(returns) / len(returns)
    variance = sum((r - avg) ** 2 for r in returns) / len(returns)
    std = math.sqrt(variance) * 100  # 转为百分比

    # threshold = max(base_pct, 1.5σ)，上限5%
    threshold = max(base_pct, 1.5 * std)
    return min(threshold, 5.0)


# ========== 新高/新低 ==========

def check_new_high_low(kline: List[Dict], period: int = 20) -> Dict:
    """检查是否创N日新高/新低"""
    if len(kline) < period + 1:
        return {'new_high': False, 'new_low': False, 'period': period}

    recent_high = max(k['high'] for k in kline[-period - 1:-1])
    recent_low = min(k['low'] for k in kline[-period - 1:-1])
    current_high = kline[-1]['high']
    current_low = kline[-1]['low']

    return {
        'new_high': current_high > recent_high,
        'new_low': current_low < recent_low,
        'period': period,
        'prev_high': recent_high,
        'prev_low': recent_low,
    }


# ========== 综合技术分析 ==========

def get_technical_analysis(symbol: str) -> Dict:
    """获取个股完整技术分析"""
    kline = _get_kline(symbol, 120)
    if not kline or len(kline) < 30:
        return {'error': f'{symbol} 数据不足，需要至少30日K线'}

    closes = [k['close'] for k in kline]
    current = closes[-1]

    # 均线
    ma5 = calc_ma(closes, 5)
    ma10 = calc_ma(closes, 10)
    ma20 = calc_ma(closes, 20)
    ma60 = calc_ma(closes, 60) if len(closes) >= 60 else []

    # MACD
    macd = calc_macd(closes)

    # RSI
    rsi = calc_rsi(closes, 14)

    # KDJ
    kdj_data = calc_kdj(kline)

    # 布林带
    boll = calc_boll(closes)

    # 量比
    vol_ratio = calc_volume_ratio(kline)

    # 用实时行情覆盖当前价（盘中K线可能缺少今天的数据）
    rt_current = None
    rt_change_pct = None
    rt_data = None
    try:
        from data_fetcher import get_stock_data
        rt_data = get_stock_data(symbol)
        if 'error' not in rt_data:
            rt_current = rt_data['current_price']
            rt_change_pct = rt_data['change_pct']
    except Exception as e:
        logger.warning(f"实时行情获取失败 {symbol}: {e}")

    final_current = rt_current if rt_current is not None else current
    final_change_pct = rt_change_pct if rt_change_pct is not None else ((kline[-1]['close'] - kline[-2]['close']) / kline[-2]['close'] * 100 if len(kline) > 1 else 0)

    # 支撑压力位（基于K线 + 持仓成本价计算）
    # 先查持仓成本价
    cost_price = None
    try:
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT cost_price FROM positions WHERE symbol=?", (symbol,))
        row = cursor.fetchone()
        conn.close()
        if row and row[0] > 0:
            cost_price = float(row[0])
    except Exception:
        pass

    # 检测ETF和港股（用于支撑压力位计算参数调整）
    is_hk = symbol.endswith('.HK')
    is_etf = symbol.startswith('15') or symbol.startswith('51') or symbol.startswith('50')
    currency = 'HK$' if is_hk else '¥'

    sr = calc_support_resistance(kline, cost_price=cost_price, is_etf=is_etf, is_hk=is_hk)
    sr['current'] = final_current

    # 如果实时价明显高于K线收盘价（盘中缺今天数据），局部高点可能低于实时价
    # 需要重新校验：确保支撑位低于实时价，压力位高于实时价
    # 只在实时价 > K线收盘时触发（正常盘中场景），实时价低于收盘时通常是数据源问题（如港股）
    if rt_current is not None and rt_current > current and (rt_current - current) / current > 0.005:
        # 实时价和K线收盘价差距>0.5%，说明盘中数据缺失
        sr['supports'] = [s for s in sr.get('supports', []) if s < final_current]
        sr['resistances'] = [r for r in sr.get('resistances', []) if r > final_current]

        # 用实时数据补充压力位（今天的最高价等）
        if rt_data and rt_data.get('high') and rt_data['high'] > final_current:
            sr['resistances'] = sorted(set(sr['resistances'] + [rt_data['high']]))

        # 补充不足的位置
        closes_all = [k['close'] for k in kline]
        std = math.sqrt(sum((x - sum(closes_all) / len(closes_all)) ** 2 for x in closes_all) / len(closes_all))
        step = max(final_current * 0.02, 2 * std)

        for i in range(1, 4):
            s = final_current - i * step
            if s > 0 and len(sr['supports']) < 3:
                sr['supports'].append(round(s, 2))
        sr['supports'] = sorted(sr['supports'], reverse=True)[:3]

        for i in range(1, 4):
            r = final_current + i * step
            if len(sr['resistances']) < 3:
                sr['resistances'].append(round(r, 2))
        sr['resistances'] = sorted(sr['resistances'])[:3]

    # 动态阈值
    threshold = calc_dynamic_threshold(kline)

    # 新高/新低
    new_hl_20 = check_new_high_low(kline, 20)
    new_hl_60 = check_new_high_low(kline, 60) if len(kline) >= 61 else {'new_high': False, 'new_low': False}

    # 信号判断
    signals = []

    # 均线信号
    if ma5 and ma10:
        if ma5[-1] > ma10[-1] and ma5[-2] <= ma10[-2]:
            signals.append('5日均线金叉10日均线')
        elif ma5[-1] < ma10[-1] and ma5[-2] >= ma10[-2]:
            signals.append('5日均线死叉10日均线')

    if ma10 and ma20:
        if ma10[-1] > ma20[-1] and len(ma10) >= 2 and ma10[-2] <= ma20[-2]:
            signals.append('10日均线金叉20日均线')
        elif ma10[-1] < ma20[-1] and len(ma10) >= 2 and ma10[-2] >= ma20[-2]:
            signals.append('10日均线死叉20日均线')

    # 均线排列
    if ma5 and ma10 and ma20 and len(ma5) >= 2:
        if ma5[-1] > ma10[-1] > ma20[-1] and final_current > ma5[-1]:
            signals.append('多头排列（看涨）')
        elif ma5[-1] < ma10[-1] < ma20[-1] and final_current < ma5[-1]:
            signals.append('空头排列（看跌）')

    # MACD信号
    if macd['dif'] and macd['dea'] and len(macd['dif']) >= 2:
        if macd['dif'][-1] > macd['dea'][-1] and macd['dif'][-2] <= macd['dea'][-2]:
            signals.append('MACD金叉')
        elif macd['dif'][-1] < macd['dea'][-1] and macd['dif'][-2] >= macd['dea'][-2]:
            signals.append('MACD死叉')
        if macd['dif'][-1] > 0 and macd['dif'][-2] <= 0:
            signals.append('DIF突破零轴')
        elif macd['dif'][-1] < 0 and macd['dif'][-2] >= 0:
            signals.append('DIF跌破零轴')

    # RSI信号
    if rsi:
        rsi_val = rsi[-1]
        if rsi_val > 70:
            signals.append(f'RSI超买({rsi_val:.1f})')
        elif rsi_val < 30:
            signals.append(f'RSI超卖({rsi_val:.1f})')

    # KDJ信号
    if kdj_data['j']:
        j_val = kdj_data['j'][-1]
        if j_val > 100:
            signals.append(f'KDJ超买(J={j_val:.1f})')
        elif j_val < 0:
            signals.append(f'KDJ超卖(J={j_val:.1f})')
        # K/D金叉死叉
        if kdj_data['k'] and kdj_data['d'] and len(kdj_data['k']) >= 2:
            if kdj_data['k'][-1] > kdj_data['d'][-1] and kdj_data['k'][-2] <= kdj_data['d'][-2]:
                signals.append('KDJ金叉')
            elif kdj_data['k'][-1] < kdj_data['d'][-1] and kdj_data['k'][-2] >= kdj_data['d'][-2]:
                signals.append('KDJ死叉')

    # 布林带信号
    if boll['upper']:
        if final_current > boll['upper'][-1]:
            signals.append('突破布林带上轨')
        elif final_current < boll['lower'][-1]:
            signals.append('跌破布林带下轨')

    # 新高/新低
    if new_hl_20['new_high']:
        signals.append(f'创{new_hl_20["period"]}日新高')
    if new_hl_20['new_low']:
        signals.append(f'创{new_hl_20["period"]}日新低')
    if new_hl_60['new_high']:
        signals.append('创60日新高')
    if new_hl_60['new_low']:
        signals.append('创60日新低')

    # 量价信号（用实时涨跌幅判断方向，避免日线和盘中方向不一致）
    if vol_ratio > 2:
        if final_change_pct > 0:
            signals.append(f'放量上涨(量比{vol_ratio:.1f})')
        elif final_change_pct < 0:
            signals.append(f'放量下跌(量比{vol_ratio:.1f})')
        else:
            signals.append(f'放量平盘(量比{vol_ratio:.1f})')
    elif vol_ratio < 0.5:
        if final_change_pct < 0:
            signals.append('缩量回调（可能反弹）')
        elif final_change_pct > 0:
            signals.append('缩量上涨（动力不足）')
        else:
            signals.append('缩量平盘（方向不明）')

    # 接近支撑压力位（用正确的货币符号和价格格式）
    for s in sr.get('supports', []):
        if abs(final_current - s) / final_current < 0.02 and final_current > s:
            price_str = f'{s:.3f}' if is_etf else f'{s:.2f}'
            signals.append(f'接近支撑位{currency}{price_str}')
            break
    for r in sr.get('resistances', []):
        if abs(final_current - r) / final_current < 0.02 and final_current < r:
            price_str = f'{r:.3f}' if is_etf else f'{r:.2f}'
            signals.append(f'接近压力位{currency}{price_str}')
            break

    # 股名
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM stock_info WHERE symbol=?", (symbol,))
    row = cursor.fetchone()
    name = row[0] if row else symbol
    conn.close()

    # 信号分与总结（区分强弱信号，避免误判）
    # 强信号：明确的多空方向（趋势类信号，不含KDJ等短期指标）
    strong_bearish = ['空头排列', '均线死叉', 'MACD死叉', '跌破', '新低', '放量下跌', '缩量上涨', 'DIF跌破零轴']
    strong_bullish = ['多头排列', '均线金叉', 'MACD金叉', '突破', '新高', '放量上涨', 'DIF突破零轴']
    # 弱/中性信号：短期指标或警示类，不参与方向打分
    neutral_warning = ['超买', '超卖', '缩量回调', '缩量平盘', '接近支撑', '接近压力', 'KDJ', '死叉', '金叉']
    
    strong_b = sum(1 for s in signals if any(kw in s for kw in strong_bearish))
    strong_u = sum(1 for s in signals if any(kw in s for kw in strong_bullish))
    
    # 加权打分：强信号权重 2
    score = strong_u * 2 - strong_b * 2
    
    # 检查是否有超卖反弹信号（在偏空时要提示）
    has_oversold = any('超卖' in s for s in signals)
    has_overbought = any('超买' in s for s in signals)
    has_reversal = any(kw in ''.join(signals) for kw in ['金叉', '反弹'])
    has_warning = any('死叉' in s or '超买' in s for s in signals)
    
    if score < -1:
        if has_oversold or has_reversal:
            action_hint = '⚠️ 偏空但有超卖反弹信号，观望等待企稳'
        else:
            action_hint = '🔴 偏空，建议减仓或观望'
    elif score < 0:
        action_hint = '⚠️ 略偏空，谨慎持有'
    elif score > 1:
        if has_overbought or has_warning:
            action_hint = '⚠️ 偏多但有回调风险，注意止盈'
        else:
            action_hint = '🟢 偏多，可持有或逢低加仓'
    elif score > 0:
        action_hint = '🟡 略偏多，可继续观察'
    else:
        if has_oversold or has_overbought:
            action_hint = '⚪ 多空均衡，关注超买超卖信号'
        else:
            action_hint = '⚪ 多空均衡，震荡行情，观望为主'

    return {
        'symbol': symbol, 'name': name, 'current': final_current,
        'is_hk': is_hk, 'is_etf': is_etf, 'currency': currency,
        'ma5': ma5[-1] if ma5 else None, 'ma10': ma10[-1] if ma10 else None,
        'ma20': ma20[-1] if ma20 else None, 'ma60': ma60[-1] if ma60 else None,
        'macd_dif': macd['dif'][-1] if macd['dif'] else None,
        'macd_dea': macd['dea'][-1] if macd['dea'] else None,
        'macd_bar': macd['macd_bar'][-1] if macd['macd_bar'] else None,
        'rsi': rsi[-1] if rsi else None,
        'kdj_k': kdj_data['k'][-1] if kdj_data['k'] else None,
        'kdj_d': kdj_data['d'][-1] if kdj_data['d'] else None,
        'kdj_j': kdj_data['j'][-1] if kdj_data['j'] else None,
        'boll_upper': boll['upper'][-1] if boll['upper'] else None,
        'boll_middle': boll['middle'][-1] if boll['middle'] else None,
        'boll_lower': boll['lower'][-1] if boll['lower'] else None,
        'volume_ratio': vol_ratio,
        'supports': sr.get('supports', []),
        'resistances': sr.get('resistances', []),
        'dynamic_threshold': threshold,
        'signals': signals,
        'action_hint': action_hint,
        'change_pct': final_change_pct,
        'new_high_20': new_hl_20['new_high'],
        'new_low_20': new_hl_20['new_low'],
    }


# ========== 智能异动检测 ==========

def get_smart_alerts(symbols: Optional[List[str]] = None) -> List[Dict]:
    """智能异动检测 - 对持仓+自选股扫描

    返回: [{'symbol', 'name', 'type', 'details', 'signals'}, ...]
    """
    if symbols is None:
        # 合并持仓+自选
        try:
            pos_data = get_positions_data()
            symbols = [p['symbol'] for p in pos_data.get('positions', [])] if 'error' not in pos_data else []
        except Exception:
            symbols = []
        from config import WATCHLIST
        for w in WATCHLIST:
            if w.get('symbol') not in symbols:
                symbols.append(w.get('symbol'))

    alerts = []
    skipped = []  # 记录数据不足的股票
    for symbol in symbols:
        try:
            analysis = get_technical_analysis(symbol)
            if 'error' in analysis:
                skipped.append(f"{symbol}: {analysis['error']}")
                continue

            change_pct = analysis.get('change_pct', 0)
            threshold = analysis.get('dynamic_threshold', 1.5)
            vol_ratio = analysis.get('volume_ratio', 1.0)

            # 1. 涨跌幅超动态阈值
            if abs(change_pct) > threshold:
                alert_type = '大涨' if change_pct > 0 else '大跌'
                # 量价联合判断
                if vol_ratio > 1.5:
                    alert_type = f'放量{alert_type}'
                elif vol_ratio < 0.7:
                    alert_type = f'缩量{alert_type}'
                is_etf = symbol.startswith('15') or symbol.startswith('51') or symbol.startswith('50') or 'ETF' in analysis.get('name', '')
                alerts.append({
                    'symbol': symbol, 'name': analysis['name'],
                    'type': alert_type, 'change_pct': change_pct,
                    'volume_ratio': vol_ratio,
                    'details': f'{change_pct:+.2f}% 量比{vol_ratio:.1f} 阈值{threshold:.1f}%',
                    'signals': analysis.get('signals', []),
                    'is_etf': is_etf,
                })

            # 2. 重要技术信号（金叉/死叉/超买超卖）
            important_signals = [s for s in analysis.get('signals', [])
                                 if any(kw in s for kw in ['金叉', '死叉', '超买', '超卖', '突破', '跌破', '新高', '新低'])]
            if important_signals:
                is_etf = symbol.startswith('15') or symbol.startswith('51') or symbol.startswith('50') or 'ETF' in analysis.get('name', '')
                alerts.append({
                    'symbol': symbol, 'name': analysis['name'],
                    'type': '技术信号', 'change_pct': change_pct,
                    'details': ' | '.join(important_signals),
                    'signals': analysis.get('signals', []),
                    'is_etf': is_etf,
                })

            # 3. 接近支撑/压力位（合并同一只股票的多个价位，避免重复推送）
            is_etf = symbol.startswith('15') or symbol.startswith('51') or symbol.startswith('50') or 'ETF' in analysis.get('name', '')

            # 合并所有接近的支撑位
            near_supports = [s for s in analysis.get('supports', [])
                             if abs(analysis['current'] - s) / analysis['current'] < 0.015 and analysis['current'] > s]
            if near_supports:
                supports_str = '、'.join([f'¥{s:.2f}' for s in near_supports])
                alerts.append({
                    'symbol': symbol, 'name': analysis['name'],
                    'type': '接近支撑位', 'change_pct': change_pct,
                    'details': f'当前¥{analysis["current"]:.2f} 接近支撑 {supports_str}',
                    'signals': [],
                    'is_etf': is_etf,
                })

            # 合并所有接近的压力位
            near_resistances = [r for r in analysis.get('resistances', [])
                                if abs(analysis['current'] - r) / analysis['current'] < 0.015 and analysis['current'] < r]
            if near_resistances:
                resistances_str = '、'.join([f'¥{r:.2f}' for r in near_resistances])
                alerts.append({
                    'symbol': symbol, 'name': analysis['name'],
                    'type': '接近压力位', 'change_pct': change_pct,
                    'details': f'当前¥{analysis["current"]:.2f} 接近压力 {resistances_str}',
                    'signals': [],
                    'is_etf': is_etf,
                })
        except Exception as e:
            logger.warning(f"异动检测 {symbol} 失败: {e}")

    if skipped:
        logger.info(f"异动检测跳过（数据不足）: {skipped}")

    return alerts