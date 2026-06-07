#!/usr/bin/env python3
"""
数据获取层 - 所有外部数据获取的统一入口

职责：从数据库、Tushare、腾讯财经API获取数据，返回结构化dict。
不包含任何飞书/卡片/意图逻辑。
"""

import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.join(os.path.dirname(AGENT_DIR), 'python')
sys.path.insert(0, AGENT_DIR)
sys.path.insert(0, PYTHON_DIR)

from config import DB_PATH, TUSHARE_TOKEN, TENCENT_QUOTE_API, WATCHLIST, AVAILABLE_CASH, TOTAL_INVESTMENT

logger = logging.getLogger("feishu_bot")


# ========== 数据库连接 ==========

def _get_conn():
    return sqlite3.connect(DB_PATH)


# ========== 腾讯财经 API（实时行情） ==========

def _tencent_batch_query(codes: dict) -> dict:
    """批量查询腾讯财经API，返回 {name: {price, change_pct, change_amount}}"""
    import requests
    try:
        query_str = ",".join(codes.values())
        url = f"{TENCENT_QUOTE_API}{query_str}"
        r = requests.get(url, timeout=15)
        lines = r.text.strip().split(";")
        results = {}
        for name, code in codes.items():
            for line in lines:
                if code in line:
                    parts = line.split('~')
                    if len(parts) > 32:
                        results[name] = {
                            'price': float(parts[3]),
                            'change_pct': float(parts[32]),
                            'change_amount': float(parts[31]),
                        }
                    break
        return results
    except Exception as e:
        logger.error(f"腾讯财经API查询异常: {e}")
        return {}


# ========== 持仓数据 ==========

def get_positions_data() -> Dict:
    """获取持仓数据"""
    conn = _get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT symbol, shares, cost_price, name FROM positions ORDER BY current_value DESC")
        rows = cursor.fetchall()
    except Exception:
        conn.close()
        return {'error': '无法读取持仓数据'}

    positions = []
    total_value = 0
    total_cost = 0

    from data.data_handler import DataHandler
    dh = DataHandler(force_refresh=False)
    symbols = [r[0] for r in rows]
    realtime = dh.get_realtime_prices(symbols) if symbols else {}

    for symbol, shares, cost_price, name in rows:
        current_price = realtime.get(symbol, {}).get('price', cost_price)
        market_value = current_price * shares
        cost = cost_price * shares
        profit = market_value - cost
        profit_pct = (current_price - cost_price) / cost_price * 100 if cost_price > 0 else 0
        positions.append({
            'symbol': symbol, 'stock_name': name, 'shares': shares,
            'cost_price': cost_price, 'current_price': current_price,
            'market_value': market_value, 'profit': profit, 'profit_pct': profit_pct,
        })
        total_value += market_value
        total_cost += cost

    conn.close()

    available_cash = AVAILABLE_CASH
    total_profit = total_value - total_cost
    profit_pct = total_profit / total_cost * 100 if total_cost > 0 else 0

    return {
        'positions': positions, 'total_value': total_value + available_cash,
        'total_cost': total_cost + available_cash, 'total_profit': total_profit,
        'profit_pct': profit_pct, 'available_cash': available_cash,
    }


# ========== 个股行情 ==========

def get_stock_data(symbol: str) -> Dict:
    """获取单只股票行情"""
    if not symbol:
        return {'error': '请提供股票代码'}

    from data.data_handler import DataHandler
    dh = DataHandler(force_refresh=False)
    df = dh.fetch_stock_data(symbol)

    if df is None or df.empty:
        return {'error': f'无法获取 {symbol} 的数据'}

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    change_pct = (latest['close'] - prev['close']) / prev['close'] * 100

    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM stock_info WHERE symbol=?", (symbol,))
    row = cursor.fetchone()
    name = row[0] if row and row[0] else symbol
    conn.close()

    realtime = dh.get_realtime_prices([symbol])
    current_price = realtime.get(symbol, {}).get('price', float(latest['close']))
    realtime_change = realtime.get(symbol, {}).get('change_pct', change_pct)

    return {
        'symbol': symbol, 'name': name,
        'current_price': current_price, 'change_pct': realtime_change,
        'change_amount': realtime.get(symbol, {}).get('change', current_price - float(prev['close'])),
        'latest_close': float(latest['close']),
    }


# ========== 做T建议 ==========

def get_t_suggestions() -> List[Dict]:
    """获取做T建议"""
    try:
        from lgbm_backtest import LGBMBacktesterOptimized
        positions_data = get_positions_data()
        if 'error' in positions_data:
            return []

        suggestions = []
        for pos in positions_data['positions']:
            if pos.get('profit_pct', 0) > 5:
                bt = LGBMBacktesterOptimized()
                result = bt.run_backtest(pos['symbol'])
                if result and result.get('summary'):
                    suggestions.append({
                        'stock_name': pos['stock_name'], 'symbol': pos['symbol'],
                        'current_price': pos['current_price'],
                        'action': '适合做T' if result['summary']['winRate'] > 50 else '观望',
                        'reason': f"胜率{result['summary']['winRate']:.1f}%",
                    })
        return suggestions[:5]
    except Exception as e:
        logger.error(f"做T建议获取失败: {e}")
        return []


# ========== 交易信号 ==========

def get_signals_data() -> Dict:
    """获取交易信号"""
    try:
        from lgbm_backtest import LGBMBacktesterOptimized
        bt = LGBMBacktesterOptimized()
        signals = []
        for item in WATCHLIST[:5]:
            symbol = item.get('symbol')
            name = item.get('name', symbol)
            result = bt.run_backtest(symbol)
            if result and result.get('summary'):
                pred = result.get('predictions', [])
                last_pred = pred[-1] if pred else {}
                up_prob = last_pred.get('up_probability', 0)
                action = '买入' if up_prob > 0.52 else '卖出' if up_prob < 0.48 else '持有'
                rt = get_stock_data(symbol)
                signals.append({
                    'stock_name': name, 'symbol': symbol,
                    'current_price': rt.get('current_price', 0),
                    'signal': action, 'up_prob': up_prob,
                    'reason': f"上涨概率{up_prob:.1%}",
                })
        return {'signals': signals}
    except Exception as e:
        logger.error(f"信号获取失败: {e}")
        return {'signals': []}


# ========== 回测 ==========

def run_backtest(symbol: str) -> Dict:
    """运行LGBM回测"""
    try:
        from lgbm_backtest import LGBMBacktesterOptimized
        bt = LGBMBacktesterOptimized()
        result = bt.run_backtest(symbol)
        if not result:
            return {'error': f'{symbol} 回测失败'}
        conn = _get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM stock_info WHERE symbol=?", (symbol,))
        row = cursor.fetchone()
        result['symbol'] = symbol
        result['name'] = row[0] if row else symbol
        conn.close()
        return result
    except Exception as e:
        return {'error': f'回测异常: {str(e)[:50]}'}


# ========== 盘后总结 ==========

def get_daily_summary() -> Dict:
    """获取盘后总结"""
    positions_data = get_positions_data()
    return {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'total_value': positions_data.get('total_value', 0),
        'total_profit': positions_data.get('total_profit', 0),
        'profit_pct': positions_data.get('profit_pct', 0),
        'positions': positions_data.get('positions', []),
    }


# ========== 综合分析 ==========

def analyze_stock(symbol: str) -> Dict:
    """综合分析个股"""
    return get_stock_data(symbol)


# ========== 自选股管理 ==========

def manage_watchlist(action: str, symbol: str, name: str = '') -> Dict:
    """管理自选股"""
    global WATCHLIST
    if action == 'add':
        for w in WATCHLIST:
            if w.get('symbol') == symbol:
                return {'message': f'{name}({symbol}) 已在自选列表'}
        WATCHLIST.append({'symbol': symbol, 'name': name})
        return {'message': f'✓ 已添加 {name}({symbol}) 到自选列表'}
    elif action == 'remove':
        for i, w in enumerate(WATCHLIST):
            if w.get('symbol') == symbol:
                WATCHLIST.pop(i)
                return {'message': f'✓ 已移除 {name}({symbol})'}
        return {'message': f'{symbol} 不在自选列表'}
    return {'error': '未知操作'}


# ========== 大盘指数（腾讯财经API） ==========

def get_market_data() -> Dict:
    indices_config = [
        {"name": "上证指数", "code": "sh000001", "display_code": "000001.SH"},
        {"name": "深证成指", "code": "sz399001", "display_code": "399001.SZ"},
        {"name": "创业板指", "code": "sz399006", "display_code": "399006.SZ"},
        {"name": "沪深300", "code": "sh000300", "display_code": "000300.SH"},
        {"name": "恒生指数", "code": "r_hkHSI", "display_code": "HSI"},
    ]

    results = _tencent_batch_query({cfg["name"]: cfg["code"] for cfg in indices_config})
    indices = []
    for cfg in indices_config:
        r = results.get(cfg["name"])
        if r:
            indices.append({"name": cfg["name"], "code": cfg["display_code"], "price": r["price"], "change_pct": r["change_pct"], "change_amount": r["change_amount"]})

    avg = sum(i.get('change_pct', 0) for i in indices) / len(indices) if indices else 0
    sentiment = "🔴 市场偏强" if avg > 1 else "🟢 市场偏弱" if avg < -1 else "⚪ 市场震荡"
    return {"indices": indices, "sentiment": sentiment}


# ========== 行业板块（腾讯财经API） ==========

def get_sector_data() -> Dict:
    sector_codes = {
        "食品饮料": "sh000036", "银行": "sh000022", "房地产": "sh000021",
        "医药生物": "sh000020", "电子": "sh000018", "计算机": "sh000017",
        "机械设备": "sh000019", "有色金属": "sh000033", "化工": "sh000034",
        "汽车": "sh000029", "家用电器": "sh000030", "非银金融": "sh000028",
        "公用事业": "sh000027", "通信": "sh000026", "传媒": "sh000025",
        "交通运输": "sh000023", "农林牧渔": "sh000014", "钢铁": "sh000032",
    }

    results = _tencent_batch_query(sector_codes)
    sectors = [{"name": name, "change_pct": r["change_pct"], "price": r["price"], "lead_stock": ""} for name, r in results.items()]
    sectors.sort(key=lambda s: s['change_pct'], reverse=True)
    return {"sectors": sectors[:10]}


# ========== 多股对比 ==========

def compare_stocks(symbols: list) -> Dict:
    stocks = [get_stock_data(s) for s in symbols[:5]]
    stocks = [s for s in stocks if 'error' not in s]
    return {"stocks": stocks, "count": len(stocks)} if stocks else {"error": "无法获取对比数据"}