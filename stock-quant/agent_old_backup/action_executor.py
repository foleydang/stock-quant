#!/usr/bin/env python3
"""
动作执行器 - 根据意图调用现有的量化模块，返回结构化数据

核心思路：不修改现有模块代码，直接 import 并调用
"""

import os
import sys
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# 添加路径（agent目录优先，python目录次之）
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.join(os.path.dirname(AGENT_DIR), 'python')
API_DIR = os.path.join(os.path.dirname(AGENT_DIR), 'api')
sys.path.insert(0, AGENT_DIR)
sys.path.insert(0, PYTHON_DIR)
sys.path.insert(0, API_DIR)

from config import DB_PATH, WATCHLIST, AVAILABLE_CASH, TOTAL_INVESTMENT, STRATEGY_PARAMS

logger = logging.getLogger(__name__)


# ========== 数据库操作 ==========

def _get_conn():
    return sqlite3.connect(DB_PATH)


# ========== 查持仓 ==========

def get_positions_data() -> Dict:
    """获取持仓概览数据"""
    try:
        conn = _get_conn()
        cursor = conn.cursor()

        # 持仓列表
        cursor.execute('SELECT symbol, stock_name, shares, cost_price, current_price FROM positions')
        rows = cursor.fetchall()

        positions = []
        total_value = 0
        total_cost = 0

        for row in rows:
            symbol, name, shares, cost, current = row
            shares = int(shares)
            cost = float(cost)
            current = float(current)

            mv = shares * current
            cv = shares * cost
            profit = mv - cv
            profit_pct = (current - cost) / cost * 100 if cost > 0 else 0

            total_value += mv
            total_cost += cv

            positions.append({
                'symbol': symbol,
                'stock_name': name,
                'shares': shares,
                'cost_price': cost,
                'current_price': current,
                'market_value': mv,
                'profit': profit,
                'profit_pct': profit_pct
            })

        conn.close()

        total_profit = total_value - total_cost
        profit_pct = (total_value - total_cost) / total_cost * 100 if total_cost > 0 else 0

        return {
            'total_value': total_value + AVAILABLE_CASH,
            'total_cost': total_cost,
            'total_profit': total_profit,
            'profit_pct': profit_pct,
            'available_cash': AVAILABLE_CASH,
            'positions': positions
        }
    except Exception as e:
        logger.error(f"获取持仓数据失败: {e}")
        return {'error': str(e), 'positions': []}


# ========== 查行情 ==========

def get_stock_data(symbol: str) -> Dict:
    """获取单只股票行情数据"""
    if not symbol:
        return {'error': '请提供股票代码，例如：行情 茅台 或 行情 600036'}

    try:
        from data.data_handler import DataHandler
        dh = DataHandler(force_refresh=False)
        df = dh.fetch_stock_data(symbol)

        if df is None or df.empty:
            return {'error': f'无法获取 {symbol} 的数据'}

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        change_pct = (latest['close'] - prev['close']) / prev['close'] * 100

        # 从数据库获取股票名称
        conn = _get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM stock_info WHERE symbol=?", (symbol,))
        row = cursor.fetchone()
        name = row[0] if row and row[0] else symbol
        conn.close()

        # 获取实时价格
        realtime = dh.get_realtime_prices([symbol])
        current_price = realtime.get(symbol, {}).get('price', float(latest['close']))
        realtime_change = realtime.get(symbol, {}).get('change_pct', change_pct)

        return {
            'symbol': symbol,
            'name': name,
            'current_price': current_price,
            'change_pct': realtime_change,
            'latest_close': float(latest['close']),
            'data_count': len(df),
            'data_freshness': '实时' if realtime else '数据库缓存'
        }
    except Exception as e:
        logger.error(f"获取行情失败 {symbol}: {e}")
        return {'error': f'获取行情失败: {e}'}


# ========== 做T建议 ==========

def get_t_suggestions() -> List[Dict]:
    """获取所有持仓的做T建议"""
    try:
        from trading_monitor import TradingMonitor
        monitor = TradingMonitor()
        return monitor.analyze_t_opportunities()
    except Exception as e:
        logger.error(f"获取做T建议失败: {e}")
        # 回退：手动计算简化版
        try:
            conn = _get_conn()
            positions = get_positions_data().get('positions', [])
            suggestions = []
            for pos in positions:
                symbol = pos['symbol']
                df = pd_read_kline(symbol, limit=50)
                if df is None or len(df) < 20:
                    continue

                high = df['high'].max()
                low = df['low'].min()
                current = pos['current_price']
                intraday_range = (high - low) / current * 100

                support = low * 1.01
                resistance = high * 0.99

                if intraday_range > 2.0:
                    action = '适合做T'
                    reason = f"日内波动{intraday_range:.1f}%，支撑¥{support:.2f}，阻力¥{resistance:.2f}"
                else:
                    action = '观望'
                    reason = f"日内波动{intraday_range:.1f}%不够大"

                suggestions.append({
                    'symbol': symbol,
                    'stock_name': pos['stock_name'],
                    'action': action,
                    'current_price': current,
                    'cost_price': pos['cost_price'],
                    'profit_pct': pos['profit_pct'],
                    'support_price': support,
                    'resistance_price': resistance,
                    'intraday_range': intraday_range,
                    'reason': reason
                })

            return suggestions
        except Exception as e2:
            logger.error(f"做T建议回退也失败: {e2}")
            return []


def pd_read_kline(symbol: str, limit: int = 50):
    """从数据库读取K线数据"""
    import pandas as pd
    try:
        conn = _get_conn()
        df = pd.read_sql_query(
            f'SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol=? ORDER BY date DESC LIMIT {limit}',
            conn, params=(symbol,)
        )
        conn.close()
        if df.empty:
            return None
        return df.iloc[::-1]  # 按时间正序
    except Exception:
        return None


# ========== 交易信号 ==========

def get_signals_data() -> Dict:
    """获取最新交易信号"""
    try:
        # 使用现有 monitor 的 analyze_positions
        from trading_monitor import TradingMonitor
        monitor = TradingMonitor()
        suggestions = monitor.analyze_positions()
        return {'signals': suggestions}
    except Exception as e:
        logger.error(f"获取信号失败: {e}")
        # 回退：简单计算
        try:
            positions = get_positions_data().get('positions', [])
            signals = []
            for pos in positions:
                profit_pct = pos['profit_pct']
                if profit_pct <= -20:
                    signal = '补仓'
                    reason = f"浮亏{profit_pct:.0f}%，建议关注补仓机会"
                elif profit_pct >= 15:
                    signal = '减仓'
                    reason = f"浮盈{profit_pct:.0f}%，可考虑减仓"
                elif profit_pct <= -15:
                    signal = '观望'
                    reason = f"浮亏{profit_pct:.0f}%，暂不建议操作"
                else:
                    signal = '持有'
                    reason = f"盈亏{profit_pct:.0f}%，持有观望"

                signals.append({
                    'symbol': pos['symbol'],
                    'stock_name': pos['stock_name'],
                    'current_price': pos['current_price'],
                    'signal': signal,
                    'up_prob': 0,  # 简化版无模型预测
                    'reason': reason,
                    'action': signal,
                    'profit_pct': profit_pct,
                    'shares': pos['shares'],
                    'cost_price': pos['cost_price']
                })

            return {'signals': signals}
        except Exception as e2:
            return {'error': str(e2), 'signals': []}


# ========== 回测 ==========

def run_backtest(symbol: str) -> Dict:
    """运行LGBM回测"""
    if not symbol:
        return {'error': '请提供股票代码，例如：回测 茅台 或 回测 600036'}

    try:
        from lgbm_backtest import LGBMBacktesterOptimized
        import sqlite3

        # 获取股票名称
        conn = _get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM stock_info WHERE symbol=?", (symbol,))
        row = cursor.fetchone()
        name = row[0] if row and row[0] else symbol
        conn.close()

        # 运行回测
        backtester = LGBMBacktesterOptimized(initial_capital=500000)
        stocks = [{"symbol": symbol, "name": name}]
        backtester.run_backtest(stocks)

        # 提取结果
        final_value = backtester.daily_values[-1].get("value", 500000) if backtester.daily_values else 500000
        total_return = (final_value - 500000) / 500000 * 100

        # 交易列表
        trades = []
        for t in backtester.trades[-10:]:  # 最近10笔
            trades.append({
                'type': t.trade_type,
                'price': float(t.price),
                'shares': t.shares,
                'time': str(t.time),
                'reason': t.reason,
                'profit': float(t.profit)
            })

        # 买卖点
        buy_points = [{"date": str(t.time), "price": float(t.price)} for t in backtester.trades if t.trade_type == "buy"][-5:]
        sell_points = [{"date": str(t.time), "price": float(t.price)} for t in backtester.trades if t.trade_type == "sell"][-5:]

        return {
            'symbol': symbol,
            'name': name,
            'summary': {
                'total_return': round(total_return, 2),
                'total_trades': len(backtester.trades),
                'final_value': final_value
            },
            'trades': trades,
            'buy_points': buy_points,
            'sell_points': sell_points
        }
    except Exception as e:
        logger.error(f"回测失败 {symbol}: {e}")
        return {'error': f'回测失败: {e}'}


# ========== 盘后总结 ==========

def get_daily_summary() -> Dict:
    """获取盘后总结数据"""
    try:
        positions_data = get_positions_data()
        signals_data = get_signals_data()

        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_value': positions_data.get('total_value', 0),
            'total_cost': positions_data.get('total_cost', TOTAL_INVESTMENT),
            'total_profit': positions_data.get('total_profit', 0),
            'profit_pct': positions_data.get('profit_pct', 0),
            'available_cash': positions_data.get('available_cash', AVAILABLE_CASH),
            'positions': positions_data.get('positions', []),
            'signals': signals_data.get('signals', [])
        }
    except Exception as e:
        logger.error(f"获取总结失败: {e}")
        return {'error': str(e)}


# ========== 综合分析 ==========

def analyze_stock(symbol: str) -> Dict:
    """综合分析单只股票"""
    if not symbol:
        return {'error': '请提供股票代码，例如：分析 茅台 或 分析 600036'}

    try:
        # 获取行情数据
        stock_data = get_stock_data(symbol)
        if 'error' in stock_data:
            return stock_data

        # 获取持仓信息（如果有）
        positions_data = get_positions_data()
        position = None
        for pos in positions_data.get('positions', []):
            if pos['symbol'] == symbol:
                position = pos
                break

        # 技术指标（从数据库读K线计算）
        import pandas as pd
        import numpy as np

        df = pd_read_kline(symbol, limit=100)
        indicators = {}
        if df is not None and len(df) >= 20:
            closes = df['close'].values
            # MA
            if len(closes) >= 5:
                indicators['MA5'] = round(float(np.mean(closes[-5:])), 2)
            if len(closes) >= 20:
                indicators['MA20'] = round(float(np.mean(closes[-20:])), 2)
            # RSI (14)
            if len(closes) >= 15:
                deltas = np.diff(closes[-15:])
                gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
                losses = np.mean(-deltas[deltas < 0]) if np.any(deltas < 0) else 0
                rs = gains / losses if losses != 0 else 100
                indicators['RSI'] = round(float(100 - 100/(1+rs)), 1)
            # 涨跌幅
            if len(closes) >= 2:
                indicators['日涨跌'] = round(float((closes[-1] - closes[-2]) / closes[-2] * 100), 2)
                indicators['5日涨跌'] = round(float((closes[-1] - closes[-5]) / closes[-5] * 100), 2) if len(closes) >= 5 else 0
                indicators['20日涨跌'] = round(float((closes[-1] - closes[-20]) / closes[-20] * 100), 2) if len(closes) >= 20 else 0

        result = {
            'symbol': symbol,
            'name': stock_data.get('name', symbol),
            'current_price': stock_data.get('current_price', 0),
            'change_pct': stock_data.get('change_pct', 0),
            'indicators': indicators,
            'position': position,
        }

        # 做T建议（如果有持仓）
        if position:
            try:
                from strategy.t_strategy import TStrategy
                t_strategy = TStrategy()
                kline_df = pd_read_kline(symbol, limit=50)
                if kline_df is not None and len(kline_df) >= 20:
                    suggestion = t_strategy.analyze(kline_df, {
                        'symbol': symbol,
                        'stock_name': position['stock_name'],
                        'shares': position['shares'],
                        'cost_price': position['cost_price'],
                        'current_price': position['current_price']
                    })
                    result['t_suggestion'] = {
                        'action': suggestion.action,
                        'buy_price': suggestion.buy_price,
                        'sell_price': suggestion.sell_price,
                        'buy_shares': suggestion.buy_shares,
                        'sell_shares': suggestion.sell_shares,
                        'support_price': suggestion.support_price,
                        'resistance_price': suggestion.resistance_price,
                        'reason': suggestion.reason
                    }
            except Exception as e:
                logger.warning(f"做T分析失败: {e}")

        return result
    except Exception as e:
        logger.error(f"综合分析失败: {e}")
        return {'error': str(e)}


# ========== 自选股管理 ==========

def manage_watchlist(action: str, symbol: str, name: str = '') -> Dict:
    """管理自选股"""
    if not symbol:
        return {'error': '请提供股票代码，例如：自选 阿里巴巴 9988.HK'}

    if action == 'add':
        # 检查是否已存在
        for w in WATCHLIST:
            if w['symbol'] == symbol:
                return {'message': f'{w["name"]}({symbol}) 已在自选列表中'}

        # 获取股票名称
        if not name:
            try:
                conn = _get_conn()
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM stock_info WHERE symbol=?", (symbol,))
                row = cursor.fetchone()
                name = row[0] if row and row[0] else symbol
                conn.close()
            except Exception:
                name = symbol

        WATCHLIST.append({'symbol': symbol, 'name': name})
        return {'message': f'✓ 已添加 {name}({symbol}) 到自选列表'}

    elif action == 'remove':
        for i, w in enumerate(WATCHLIST):
            if w['symbol'] == symbol:
                WATCHLIST.pop(i)
                return {'message': f'✓ 已从自选列表移除 {w["name"]}({symbol})'}
        return {'message': f'{symbol} 不在自选列表中'}

    return {'error': '未知操作'}

# ========== 大盘指数 ==========



def get_market_data() -> Dict:
    """获取主要大盘指数行情（腾讯财经API）"""
    import requests
    
    indices_config = [
        {"name": "上证指数", "code": "sh000001", "display_code": "000001.SH"},
        {"name": "深证成指", "code": "sz399001", "display_code": "399001.SZ"},
        {"name": "创业板指", "code": "sz399006", "display_code": "399006.SZ"},
        {"name": "沪深300", "code": "sh000300", "display_code": "000300.SH"},
        {"name": "恒生指数", "code": "r_hkHSI", "display_code": "HSI"},
    ]

    try:
        query_str = ",".join([c["code"] for c in indices_config])
        url = f"https://qt.gtimg.cn/q={query_str}"
        r = requests.get(url, timeout=15)
        lines = r.text.strip().split(";")

        indices = []
        for cfg in indices_config:
            for line in lines:
                if cfg["code"] in line:
                    parts = line.split('~')
                    if len(parts) > 32:
                        price = float(parts[3])
                        change_pct = float(parts[32])
                        change_amount = float(parts[31])
                        indices.append({
                            "name": cfg["name"],
                            "code": cfg["display_code"],
                            "price": price,
                            "change_pct": change_pct,
                            "change_amount": change_amount,
                        })
                    break

        avg_change = sum(i.get('change_pct', 0) for i in indices) / len(indices) if indices else 0
        if avg_change > 1:
            sentiment = "🔴 市场偏强，多数指数上涨"
        elif avg_change < -1:
            sentiment = "🟢 市场偏弱，多数指数下跌"
        else:
            sentiment = "⚪ 市场震荡，涨跌互现"

        return {"indices": indices, "sentiment": sentiment}
    except Exception as e:
        logger.error(f"获取大盘数据异常: {e}")
        return {"indices": [], "sentiment": f"获取失败: {str(e)[:30]}"}


def get_sector_data() -> Dict:
    """获取今日热门板块行情（腾讯财经API）"""
    import requests
    
    sector_codes = {
        "食品饮料": "sh000036", "银行": "sh000022", "房地产": "sh000021",
        "医药生物": "sh000020", "电子": "sh000018", "计算机": "sh000017",
        "机械设备": "sh000019", "有色金属": "sh000033", "化工": "sh000034",
        "汽车": "sh000029", "家用电器": "sh000030", "非银金融": "sh000028",
        "公用事业": "sh000027", "通信": "sh000026", "传媒": "sh000025",
        "交通运输": "sh000023", "农林牧渔": "sh000014", "钢铁": "sh000032",
        "建筑装饰": "sh000024",
    }

    try:
        query_str = ",".join(sector_codes.values())
        url = f"https://qt.gtimg.cn/q={query_str}"
        r = requests.get(url, timeout=15)
        lines = r.text.strip().split(";")

        sectors = []
        for name, code in sector_codes.items():
            for line in lines:
                if code in line:
                    parts = line.split('~')
                    if len(parts) > 32:
                        pct = float(parts[32])
                        price = float(parts[3])
                        # 找出该板块涨跌幅排名
                        sectors.append({"name": name, "change_pct": pct, "price": price, "lead_stock": ""})
                    break

        # 按涨跌幅排序（涨幅最大的排前面）
        sectors.sort(key=lambda s: s['change_pct'], reverse=True)
        return {"sectors": sectors[:10]}
    except Exception as e:
        logger.error(f"获取板块数据异常: {e}")
        return {"sectors": []}


def compare_stocks(symbols: list) -> Dict:
    """多只股票对比"""
    stocks = []
    for symbol in symbols[:5]:
        data = get_stock_data(symbol)
        if 'error' not in data:
            stocks.append(data)

    if not stocks:
        return {"error": "无法获取对比数据"}

    return {"stocks": stocks, "count": len(stocks)}
