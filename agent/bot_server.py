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
from datetime import datetime

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
    get_money_flow, get_stock_deep_data, get_north_flow,
)
from technical_indicators import get_technical_analysis, get_smart_alerts
from advanced_analysis import get_smart_t_strategy, get_portfolio_risk, search_stock_news, get_action_recommendations, get_valuation_judge
from card_templates import (
    make_position_card, make_stock_card, make_signal_card,
    make_backtest_card, make_daily_summary_card, make_help_card,
    make_text_card, make_chat_card, make_market_card,
    make_sector_card, make_compare_card,
    make_technical_card, make_alert_card_v2,
    make_money_flow_card, make_deep_data_card, make_compare_deep_card,
    make_t_strategy_card, make_risk_card, make_recommend_card,
    make_news_card, make_valuation_card,
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
            t_data = get_smart_t_strategy()
            if not t_data:
                return make_text_card("当前没有持仓，无法给出做T建议")
            return make_t_strategy_card(t_data)
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
        elif intent == 'money_flow':
            symbol = params.get('symbol')
            if not symbol:
                return make_text_card("请提供股票代码，如：`资金 茅台`")
            data = get_money_flow(symbol)
            if 'error' in data:
                return make_text_card(f"资金流向获取失败: {data['error']}")
            return make_money_flow_card(data)
        elif intent == 'deep':
            symbol = params.get('symbol')
            if not symbol:
                return make_text_card("请提供股票代码，如：`深度 茅台`")
            data = get_stock_deep_data(symbol)
            if 'error' in data:
                return make_text_card(f"深度数据获取失败: {data['error']}")
            return make_deep_data_card(data)
        elif intent == 'north_flow':
            data = get_north_flow()
            if 'error' in data:
                return make_text_card(f"北向资金获取失败: {data['error']}")
            if not data.get('stocks'):
                return make_text_card("暂无北向资金数据")
            return make_compare_deep_card({'stocks': data['stocks'], 'count': len(data['stocks']), 'cheapest': ''})
        elif intent == 'technical':
            symbol = params.get('symbol')
            if not symbol:
                return make_text_card("请提供股票代码，如：`指标 茅台`")
            data = get_technical_analysis(symbol)
            if 'error' in data:
                return make_text_card(f"技术分析失败: {data['error']}")
            # 注入消息面
            try:
                from scheduler import _search_stock_news_brief
                name = data.get('name', symbol)
                news_data = _search_stock_news_brief(symbol, name)
                if news_data and news_data.get('headlines'):
                    headlines = news_data['headlines'][:3]
                    news_line = "**📰 消息面**\n"
                    for h in headlines:
                        title = h.get('title', '')
                        if title:
                            news_line += f"- {title}\n"
                    # 尝试LLM情绪分析
                    try:
                        from llm_client import analyze_news_sentiment
                        items = [{'title': h.get('title', ''), 'content': h.get('snippet', ''), 'time': ''} for h in headlines]
                        sentiment = analyze_news_sentiment(items)
                        s_label = sentiment.get('sentiment_label', '中性')
                        s_score = sentiment.get('score', 0.5)
                        s_summary = sentiment.get('summary', '')
                        s_color = 'red' if s_score > 0.5 else 'green' if s_score < 0.5 else 'default'
                        news_line += f"\n**情绪**: <font color='{s_color}'>{s_label}（{s_score:.1f}）</font> — {s_summary}"
                    except Exception:
                        pass
                    data['news_hint'] = news_line
            except Exception as e:
                logger.warning(f"消息面注入失败 {symbol}: {e}")
            # 注入LLM推断消息面（新闻搜不到时）
            if not data.get('news_hint'):
                try:
                    from llm_client import _call_dashscope_chat, is_available
                    import json as _json
                    if is_available():
                        tech_summary = ' | '.join(data.get('signals', [])[:5]) if data.get('signals') else '无重要信号'
                        today_str = datetime.now().strftime('%Y-%m-%d')
                        msgs = [
                            {"role": "system", "content": "你是金融分析助手，根据今日技术面推断消息面因素。只返回JSON: {\"summary\": \"一句话概括(标注：基于行情推断)\", \"score\": 0-1分数, \"sentiment_label\": \"偏利好/偏利空/中性\", \"factors\": [\"1-2个可能的消息面因素\"]}"},
                            {"role": "user", "content": f"股票: {data.get('name', '')}({symbol}) 当前¥{data.get('current', 0):.2f} 涨跌{data.get('change_pct', 0):.2f}% 信号: {tech_summary} 今日: {today_str}"}
                        ]
                        result = _call_dashscope_chat(msgs, max_tokens=150, temperature=0.3)
                        if result and '{' in result:
                            start = result.index('{')
                            end = result.rindex('}') + 1
                            llm_s = _json.loads(result[start:end])
                            s_color = 'red' if llm_s.get('score', 0.5) > 0.5 else 'green' if llm_s.get('score', 0.5) < 0.5 else 'default'
                            factors_text = '\n'.join([f"- [推断] {f}" for f in llm_s.get('factors', [])[:3]])
                            data['news_hint'] = f"**📰 消息面**\n{factors_text}\n**情绪**: <font color='{s_color}'>{llm_s.get('sentiment_label', '中性')}（{llm_s.get('score', 0.5):.1f}）</font> — {llm_s.get('summary', '')}"
                except Exception as e:
                    logger.warning(f"LLM消息面推断失败: {e}")
            # 注入LGBM预测
            try:
                from lgbm_backtest import LGBMBacktesterOptimized
                bt = LGBMBacktesterOptimized()
                result = bt.run_backtest(symbol)
                if result and result.get('predictions'):
                    last_pred = result['predictions'][-1]
                    up_prob = last_pred.get('up_probability', 0)
                    win_rate = result.get('summary', {}).get('winRate', 0)
                    signal = '看涨' if up_prob > 0.52 else '看跌' if up_prob < 0.48 else '中性'
                    data['lgbm'] = {'up_prob': up_prob, 'signal': signal, 'win_rate': win_rate}
            except Exception as e:
                logger.warning(f"LGBM预测注入失败 {symbol}: {e}")
            # 注入操作建议
            try:
                rec_data = get_action_recommendations(symbol)
                recs = rec_data.get('recommendations', [])
                if recs:
                    r = recs[0]
                    action = r.get('action', '持有')
                    confidence = r.get('confidence', '中')
                    reason = r.get('reason', '')
                    t_sugg = r.get('t_suggestion', {})
                    t_line = ''
                    if t_sugg and t_sugg.get('buy_price'):
                        t_line = f"\n做T: 买¥{t_sugg['buy_price']:.2f} 卖¥{t_sugg.get('sell_price', 0):.2f}"
                    data['action_hint'] = f"{action}（置信度{confidence}）— {reason}{t_line}"
            except Exception as e:
                logger.warning(f"操作建议注入失败 {symbol}: {e}")
            return make_technical_card(data)
        elif intent == 'alert':
            alerts = get_smart_alerts()
            if not alerts:
                return make_text_card("当前无异动，所有持仓和自选股正常")
            return make_alert_card_v2(alerts)
        elif intent == 'stop_alert':
            # 止损止盈 - 目前只返回提醒
            symbol = params.get('symbol')
            action = params.get('action')
            price = params.get('price')
            if not symbol:
                return make_text_card("请提供股票代码，如：`止损 茅台 1200`")
            data = get_stock_data(symbol)
            if 'error' in data:
                return make_text_card(f"获取行情失败: {data['error']}")
            current = data.get('current_price', 0)
            action_name = '止损' if action == 'stop_loss' else '止盈'
            if price:
                diff_pct = (current - price) / price * 100 if price > 0 else 0
                direction = '距离' + ('下方' if diff_pct > 0 else '上方')
                return make_text_card(f"**{data.get('name', symbol)}** 当前 ¥{current:.2f}\n{action_name}价 ¥{price:.2f} {direction} {abs(diff_pct):.1f}%\n\n⚠️ 止损止盈提醒功能开发中，暂不支持自动监控")
            else:
                return make_text_card(f"**{data.get('name', symbol)}** 当前 ¥{current:.2f}\n\n用法：`{action_name} {data.get('name', symbol)} 价格`\n例如：`止损 茅台 1200`")
        elif intent == 'compare':
            symbols = params.get('symbols', [])
            if len(symbols) < 2:
                return make_text_card("请提供至少2只股票，如：`对比 茅台 五粮液`")
            # 先尝试增强版对比（价格+估值+盈利）
            try:
                data = compare_stocks_deep(symbols)
                if 'error' not in data and data.get('count', 0) >= 2:
                    return make_compare_deep_card(data)
            except Exception:
                pass
            # 增强版失败则回退到基础对比
            data = compare_stocks(symbols)
            if 'error' in data:
                return make_text_card(data['error'])
            return make_compare_card(data)
        elif intent == 'risk':
            data = get_portfolio_risk()
            if 'error' in data:
                return make_text_card(f"风控评分失败: {data['error']}")
            return make_risk_card(data)
        elif intent == 'recommend':
            symbol = params.get('symbol')
            data = get_action_recommendations(symbol=symbol)
            if 'error' in data:
                return make_text_card(f"操作建议获取失败: {data['error']}")
            return make_recommend_card(data)
        elif intent == 'news':
            keyword = params.get('keyword')
            data = search_stock_news(keyword)
            return make_news_card(data)
        elif intent == 'valuation':
            symbol = params.get('symbol')
            if not symbol:
                return make_text_card("请提供股票代码，如：`估值 茅台`")
            data = get_valuation_judge(symbol)
            if 'error' in data:
                return make_text_card(f"估值分析失败: {data['error']}")
            return make_valuation_card(data)
        elif intent == 'chat':
            if is_available():
                context = {}
                try:
                    pos = get_positions_data()
                    context['positions'] = pos.get('positions', [])[:3]
                except Exception:
                    pass
                try:
                    mkt = get_market_data()
                    context['market'] = {'sentiment': mkt.get('sentiment'), 
                                         'indices': [{'name': i['name'], 'change_pct': i['change_pct']} for i in mkt.get('indices', [])]}
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