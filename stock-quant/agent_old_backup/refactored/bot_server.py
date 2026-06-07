#!/usr/bin/env python3
"""
业务逻辑层 - 消息处理核心

职责：意图路由 → 数据获取 → 卡片构建 → 返回结果
不包含任何 HTTP/飞书 SDK 相关代码。
"""

import json
import logging
import os
import sys

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.join(os.path.dirname(AGENT_DIR), 'python')
sys.path.insert(0, AGENT_DIR)
sys.path.insert(0, PYTHON_DIR)

from config import FEISHU_APP_ID, FEISHU_APP_SECRET, FEISHU_VERIFICATION_TOKEN, FEISHU_ENCRYPT_KEY, BOT_PORT
from intent_router import classify_intent
from data_fetcher import (
    get_positions_data, get_stock_data, get_t_suggestions,
    get_signals_data, run_backtest, get_daily_summary,
    analyze_stock, manage_watchlist, get_market_data,
    get_sector_data, compare_stocks,
)
from card_templates import (
    make_position_card, make_stock_card, make_signal_card,
    make_backtest_card, make_daily_summary_card, make_help_card,
    make_text_card, make_chat_card, make_market_card,
    make_sector_card, make_compare_card,
)
from llm_client import is_available, chat_response

logger = logging.getLogger("feishu_bot")


def process_message(text: str) -> dict:
    """核心处理逻辑：消息 → 意图 → 数据 → 卡片"""
    intent, params = classify_intent(text)
    logger.info(f"意图: {intent}, 参数: {params}")

    try:
        if intent == 'help':
            return make_help_card()
        elif intent == 'positions':
            data = get_positions_data()
            if 'error' in data:
                return make_text_card(f"获取持仓失败: {data['error']}")
            t_data = get_t_suggestions()
            return make_position_card(
                {'total_value': data['total_value'], 'total_cost': data['total_cost'],
                 'total_profit': data['total_profit'], 'profit_pct': data['profit_pct'],
                 'available_cash': data['available_cash']},
                data['positions'], t_data)
        elif intent == 'stock':
            symbol = params.get('symbol')
            if not symbol:
                return make_text_card("请提供股票代码，如：`行情 茅台`")
            data = get_stock_data(symbol)
            if 'error' in data:
                return make_text_card(f"获取行情失败: {data['error']}")
            return make_stock_card(data)
        elif intent == 't_strategy':
            t_data = get_t_suggestions()
            if not t_data:
                return make_text_card("当前没有持仓,无法给出做T建议")
            return make_signal_card([
                {'stock_name': t['stock_name'], 'symbol': t['symbol'],
                 'current_price': t['current_price'], 'signal': t.get('action', '观望'),
                 'up_prob': 0, 'reason': t.get('reason', '')}
                for t in t_data])
        elif intent == 'signals':
            data = get_signals_data()
            if not data.get('signals'):
                return make_text_card("当前没有新的交易信号")
            return make_signal_card(data['signals'])
        elif intent == 'backtest':
            symbol = params.get('symbol')
            if not symbol:
                return make_text_card("请提供股票代码,如：`回测 茅台`")
            data = run_backtest(symbol)
            if 'error' in data:
                return make_text_card(f"回测失败: {data['error']}")
            return make_backtest_card(data)
        elif intent == 'summary':
            data = get_daily_summary()
            return make_daily_summary_card(data, data.get('positions', []), data.get('signals', []))
        elif intent == 'analyze':
            symbol = params.get('symbol')
            if not symbol:
                return make_text_card("请提供股票代码,如：`分析 茅台`")
            data = analyze_stock(symbol)
            if 'error' in data:
                return make_text_card(f"分析失败: {data['error']}")
            return make_stock_card(data)
        elif intent == 'watchlist':
            action = params.get('action', 'add')
            symbol = params.get('symbol')
            name = params.get('name', '')
            result = manage_watchlist(action, symbol, name)
            return make_text_card(result.get('message', result.get('error', '操作完成')))
        elif intent == 'market':
            data = get_market_data()
            return make_market_card(data)
        elif intent == 'sector':
            data = get_sector_data()
            return make_sector_card(data)
        elif intent == 'compare':
            symbols = params.get('symbols', [])
            if len(symbols) < 2:
                return make_text_card("请提供至少2只股票，如：`对比 茅台 五粮液`")
            data = compare_stocks(symbols)
            if 'error' in data:
                return make_text_card(data['error'])
            return make_compare_card(data)
        elif intent == 'chat':
            if is_available():
                context = {}
                try:
                    pos = get_positions_data()
                    context['positions'] = pos.get('positions', [])[:3]
                except Exception:
                    pass
                reply = chat_response(text, context)
                return make_chat_card(reply)
            else:
                return make_help_card()
        else:
            return make_help_card()
    except Exception as e:
        logger.error(f"处理消息异常: {e}")
        return make_text_card(f"处理失败,请稍后重试")