#!/usr/bin/env python3
"""
飞书 Bot 业务逻辑模块

纯业务逻辑，不包含任何 HTTP/WebSocket 服务框架代码。
所有消息处理、意图路由、卡片构建都在这里。
HTTP 回调服务由 start_bot_http.py (Flask + gunicorn) 提供。

核心功能：
1. 解析用户消息 → 意图路由 → 动作执行 → 返回飞书卡片
2. 定时推送调度器启动
3. 验证飞书请求 token
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional

# 添加路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.join(BASE_DIR, 'python')
sys.path.insert(0, AGENT_DIR)
sys.path.insert(0, PYTHON_DIR)

from config import (
    FEISHU_APP_ID, FEISHU_APP_SECRET,
    FEISHU_VERIFICATION_TOKEN, FEISHU_ENCRYPT_KEY,
    BOT_PORT
)
from intent_router import classify_intent, extract_symbol
from action_executor import (
    get_positions_data, get_stock_data, get_t_suggestions,
    get_signals_data, run_backtest, get_daily_summary,
    analyze_stock, manage_watchlist, get_market_data,
    get_sector_data, compare_stocks
)
from card_templates import (
    make_position_card, make_stock_card, make_signal_card,
    make_backtest_card, make_daily_summary_card, make_help_card,
    make_text_card, make_chat_card, make_market_card, make_sector_card,
    make_stock_compare_card
)
from feishu_client import reply_card, reply_text, send_card, get_client
from llm_client import is_available, chat_response

# 日志配置
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, 'feishu_bot.log'), encoding='utf-8'),
    ]
)
logger = logging.getLogger("feishu_bot")


# ========== 验证 ==========

def verify_feishu_token(body_json: dict) -> bool:
    """验证飞书请求 token"""
    # challenge 验证请求直接通过
    if body_json.get("challenge"):
        return True

    if FEISHU_VERIFICATION_TOKEN:
        header = body_json.get("header", {})
        token = header.get("token", "") or body_json.get("token", "")
        if token and token != FEISHU_VERIFICATION_TOKEN:
            logger.warning(f"Token 验证失败")
            return False

    return True


# ========== 消息处理逻辑 ==========

def process_message(text: str) -> dict:
    """核心处理逻辑：消息 → 意图 → 执行 → 卡片"""
    # 1. 意图分类
    intent, params = classify_intent(text)
    logger.info(f"意图: {intent}, 参数: {params}")

    # 2. 执行动作 → 构建卡片
    try:
        if intent == 'help':
            return make_help_card()

        elif intent == 'positions':
            positions_data = get_positions_data()
            if 'error' in positions_data:
                return make_text_card(f"获取持仓失败: {positions_data['error']}")
            t_data = []
            try:
                t_data = get_t_suggestions()
            except Exception:
                pass
            return make_position_card(
                summary={
                    'total_value': positions_data['total_value'],
                    'total_cost': positions_data['total_cost'],
                    'total_profit': positions_data['total_profit'],
                    'profit_pct': positions_data['profit_pct'],
                    'available_cash': positions_data['available_cash'],
                },
                positions=positions_data['positions'],
                t_suggestions=t_data,
            )

        elif intent == 'stock':
            symbol = params.get('symbol')
            if not symbol:
                return make_text_card("请提供股票代码，例如：`行情 茅台` 或 `行情 600036`")
            stock_data = get_stock_data(symbol)
            if 'error' in stock_data:
                return make_text_card(f"获取行情失败: {stock_data['error']}")
            return make_stock_card(stock_data)

        elif intent == 't_strategy':
            t_data = get_t_suggestions()
            if not t_data:
                return make_text_card("当前没有持仓,无法给出做T建议")
            return make_signal_card([
                {
                    'stock_name': t.get('stock_name', ''),
                    'symbol': t.get('symbol', ''),
                    'current_price': t.get('current_price', 0),
                    'signal': t.get('action', '观望'),
                    'up_prob': 0,
                    'reason': t.get('reason', ''),
                }
                for t in t_data
            ])

        elif intent == 'signals':
            signals_data = get_signals_data()
            signals = signals_data.get('signals', [])
            if not signals:
                return make_text_card("当前没有新的交易信号")
            return make_signal_card(signals)

        elif intent == 'backtest':
            symbol = params.get('symbol')
            if not symbol:
                return make_text_card("请提供股票代码,例如：`回测 茅台` 或 `回测 600036`")
            backtest_data = run_backtest(symbol)
            if 'error' in backtest_data:
                return make_text_card(f"回测失败: {backtest_data['error']}")
            return make_backtest_card(backtest_data)

        elif intent == 'summary':
            summary_data = get_daily_summary()
            t_data = []
            try:
                t_data = get_t_suggestions()
            except Exception:
                pass
            return make_daily_summary_card(
                summary=summary_data,
                positions=summary_data.get('positions', []),
                signals=summary_data.get('signals', []),
                t_suggestions=t_data
            )

        elif intent == 'analyze':
            symbol = params.get('symbol')
            if not symbol:
                return make_text_card("请提供股票代码,例如：`分析 茅台` 或 `分析 600036`")
            analysis_data = analyze_stock(symbol)
            if 'error' in analysis_data:
                return make_text_card(f"分析失败: {analysis_data['error']}")
            return make_stock_card(analysis_data)

        elif intent == 'watchlist':
            action = params.get('action', 'add')
            symbol = params.get('symbol')
            name = params.get('name', '')
            result = manage_watchlist(action, symbol, name)
            return make_text_card(result.get('message', result.get('error', '操作完成')))

        elif intent == 'chat':
            if is_available():
                context = {}
                try:
                    positions_data = get_positions_data()
                    context['positions'] = positions_data.get('positions', [])[:3]
                except Exception:
                    pass
                llm_reply = chat_response(text, context)
                return make_chat_card(llm_reply)
            else:
                return make_help_card()

        elif intent == 'market':
            try:
                market_data = get_market_data()
                return make_market_card(market_data)
            except Exception as e:
                logger.error(f"获取大盘数据异常: {e}")
                return make_text_card(f"获取大盘数据失败: {str(e)[:50]}")

        elif intent == 'sector':
            try:
                sector_data = get_sector_data()
                return make_sector_card(sector_data)
            except Exception as e:
                logger.error(f"获取板块数据异常: {e}")
                return make_text_card(f"获取板块数据失败: {str(e)[:50]}")

        elif intent == 'compare':
            symbols = params.get('symbols', [])
            if len(symbols) < 2:
                return make_text_card("请提供至少2只股票进行对比，例如：`对比 茅台 五粮液`")
            try:
                compare_data = compare_stocks(symbols)
                return make_stock_compare_card(compare_data)
            except Exception as e:
                logger.error(f"对比异常: {e}")
                return make_text_card(f"对比失败: {str(e)[:50]}")

        else:
            return make_help_card()
    except Exception as e:
        logger.error(f"处理消息异常: {e}")
        return make_text_card(f"处理失败,请稍后重试。\n错误: {str(e)[:50]}")


# ========== 定时推送 ==========

def start_scheduler_if_configured():
    """如果飞书配置完整,启动定时推送"""
    from config import FEISHU_TARGET_CHAT_ID, FEISHU_TARGET_OPEN_ID
    target = FEISHU_TARGET_OPEN_ID or FEISHU_TARGET_CHAT_ID
    if FEISHU_APP_ID and FEISHU_APP_SECRET and target:
        try:
            from scheduler import start_scheduler
            start_scheduler()
            logger.info("✓ 定时推送已启动")
        except Exception as e:
            logger.warning(f"定时推送启动失败（不影响 Bot 功能）: {e}")
    else:
        logger.info("定时推送未启动（需要配置 FEISHU_TARGET_CHAT_ID）")