#!/usr/bin/env python3
"""
业务逻辑层 v2 - 消息处理核心

8类意图路由：
  stock_brief  轻量行情    茅台 / 茅台行情 / 多少钱
  stock_deep   深度分析    茅台分析 / 建议 / 补仓 / 怎么样
  stock_news   股票新闻    茅台新闻 / 消息 / 利好 / 利空
  portfolio    持仓        持仓 / 仓位 / 风控 / 信号
  market       大盘        大盘 / 指数 / 板块 / 北向
  compare      对比        茅台和五粮液对比 / 哪个好
  help         帮助        帮助 / 功能 / 回测 / 自选
  chat         闲聊        总结 / 日报 / 其他
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

from config import (
    FEISHU_APP_ID, FEISHU_APP_SECRET, FEISHU_VERIFICATION_TOKEN,
    FEISHU_ENCRYPT_KEY, BOT_PORT,
)
from intent_router import classify_intent, llm_classify
from data_fetcher import (
    get_positions_data, get_stock_data, get_t_suggestions,
    get_signals_data, run_backtest, get_daily_summary,
    analyze_stock, manage_watchlist, get_market_data,
    get_sector_data, compare_stocks,
    get_money_flow, get_stock_deep_data, get_north_flow,
)
from technical_indicators import get_technical_analysis, get_smart_alerts
from advanced_analysis import (
    get_smart_t_strategy, get_portfolio_risk, search_stock_news,
    get_action_recommendations, get_valuation_judge,
)
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

# ========== 股票深度分析处理 ==========

def _build_stock_deep(symbol: str) -> dict:
    """
    构建深度分析卡片：技术指标 + 消息面 + LGBM预测 + 操作建议
    复用原有 technical 意图的完整逻辑
    """
    data = get_technical_analysis(symbol)
    if 'error' in data:
        return make_text_card(f"技术分析失败: {data['error']}")

    name = data.get('name', symbol)

    # 注入消息面
    try:
        from scheduler import _search_stock_news_brief
        news_data = _search_stock_news_brief(symbol, name)
        if news_data and news_data.get('headlines'):
            headlines = news_data['headlines'][:5]
            news_line = "**📰 消息面（近3日）**\n"
            for h in headlines:
                title = h.get('title', '')
                time_str = h.get('time', '')
                if title:
                    time_suffix = f" _{time_str}_" if time_str else ""
                    news_line += f"- {title}{time_suffix}\n"
            try:
                from llm_client import analyze_news_sentiment
                items = [{'title': h.get('title', ''), 'snippet': h.get('snippet', ''),
                          'time': h.get('time', '')} for h in headlines]
                sentiment = analyze_news_sentiment(items)
                s_label = sentiment.get('sentiment_label', '中性')
                s_score = sentiment.get('score', 0.5)
                s_summary = sentiment.get('summary', '')
                s_factors = sentiment.get('factors', [])

                s_color = 'red' if s_score > 0.6 else 'green' if s_score < 0.4 else 'default'
                news_line += f"\n**综合判断**: <font color='{s_color}'>{s_label}（{s_score:.2f}）</font>"
                if s_summary:
                    news_line += f" — {s_summary}"
                if s_factors:
                    news_line += "\n" + "\n".join([f"- {f}" for f in s_factors[:3]])
            except Exception:
                pass
            data['news_hint'] = news_line
    except Exception as e:
        logger.warning(f"消息面注入失败 {symbol}: {e}")

    # 注入LLM推断消息面（新闻搜不到时）
    if not data.get('news_hint'):
        try:
            from llm_client import _call_dashscope_chat, is_available as _avail
            if _avail():
                tech_summary = ' | '.join(data.get('signals', [])[:5]) if data.get('signals') else '无重要信号'
                today_str = datetime.now().strftime('%Y-%m-%d')
                msgs = [
                    {"role": "system", "content": "你是金融分析助手，根据今日技术面推断消息面因素。只返回JSON: {\"summary\": \"一句话概括(标注：基于行情推断)\", \"score\": 0-1分数, \"sentiment_label\": \"偏利好/偏利空/中性\", \"factors\": [\"1-2个可能的消息面因素\"]}"},
                    {"role": "user", "content": f"股票: {name}({symbol}) 当前¥{data.get('current', 0):.2f} 涨跌{data.get('change_pct', 0):.2f}% 信号: {tech_summary} 今日: {today_str}"}
                ]
                result = _call_dashscope_chat(msgs, max_tokens=150, temperature=0.3)
                if result and '{' in result:
                    start = result.index('{')
                    end = result.rindex('}') + 1
                    llm_s = json.loads(result[start:end])
                    s_color = 'red' if llm_s.get('score', 0.5) > 0.6 else 'green' if llm_s.get('score', 0.5) < 0.4 else 'default'
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


# ========== 持仓全家桶处理 ==========

def _build_portfolio() -> dict:
    """构建持仓全家桶：持仓快照 + 风控评分 + 交易信号"""
    pos_data = get_positions_data()
    if 'error' in pos_data:
        return make_text_card(f"获取持仓失败: {pos_data['error']}")

    t_data = get_t_suggestions()
    signals = get_signals_data()

    return make_position_card(
        {'total_value': pos_data['total_value'],
         'total_cost': pos_data['total_cost'],
         'total_profit': pos_data['total_profit'],
         'profit_pct': pos_data['profit_pct'],
         'available_cash': pos_data['available_cash']},
        pos_data['positions'],
        t_data,
    )


# ========== 大盘全家桶处理 ==========

def _build_market() -> dict:
    """构建大盘全家桶：指数 + 板块 + 北向资金"""
    data = get_market_data()
    try:
        sector = get_sector_data()
        if sector and 'error' not in sector:
            data['sectors'] = sector.get('sectors', sector.get('hot_sectors', []))[:5]
    except Exception:
        pass
    try:
        north = get_north_flow()
        if north and 'error' not in north:
            data['north_flow'] = north.get('summary', '')
    except Exception:
        pass
    return make_market_card(data)


# ========== 主处理函数 ==========

def process_message(text: str) -> dict:
    """核心处理逻辑：消息 → 意图 → 数据 → 卡片"""
    intent, params = classify_intent(text)
    logger.info(f"意图: {intent}, 参数: {params}")

    try:
        # ===== 帮助 =====
        if intent == 'help':
            action = params.get('action', '')
            if action in ('add', 'remove'):
                symbol = params.get('symbol')
                name = params.get('name', '')
                result = manage_watchlist(action, symbol, name)
                return make_text_card(result.get('message', result.get('error', '操作完成')))
            return make_help_card()

        # ===== 回测（help的子意图） =====
        if intent == 'backtest':
            symbol = params.get('symbol')
            if not symbol:
                return make_text_card("请提供股票代码，如：`回测 茅台`")
            data = run_backtest(symbol)
            if 'error' in data:
                return make_text_card(f"回测失败: {data['error']}")
            return make_backtest_card(data)

        # ===== 轻量行情 =====
        if intent == 'stock_brief':
            symbol = params.get('symbol')
            if not symbol:
                return make_text_card("请提供股票代码，如：`行情 茅台`")
            data = get_stock_data(symbol)
            if 'error' in data:
                return make_text_card(f"获取行情失败: {data['error']}")
            return make_stock_card(data)

        # ===== 深度分析 =====
        if intent == 'stock_deep':
            symbol = params.get('symbol')
            if not symbol:
                return make_text_card("请提供股票代码，如：`分析 茅台`")
            return _build_stock_deep(symbol)

        # ===== 股票新闻 =====
        if intent == 'stock_news':
            keyword = params.get('keyword', params.get('symbol', ''))
            data = search_stock_news(keyword)
            if 'error' in data:
                return make_text_card(f"新闻获取失败: {data['error']}")
            return make_news_card(data)

        # ===== 持仓全家桶 =====
        if intent == 'portfolio':
            return _build_portfolio()

        # ===== 大盘全家桶 =====
        if intent == 'market':
            return _build_market()

        # ===== 对比 =====
        if intent == 'compare':
            symbols = params.get('symbols', [])
            if len(symbols) < 2:
                return make_text_card("请提供至少2只股票，如：`对比 茅台 五粮液`")
            try:
                data = compare_stocks_deep(symbols)
                if 'error' not in data and data.get('count', 0) >= 2:
                    return make_compare_deep_card(data)
            except Exception:
                pass
            data = compare_stocks(symbols)
            if 'error' in data:
                return make_text_card(data['error'])
            return make_compare_card(data)

        # ===== 闲聊 =====
        if intent == 'chat':
            if is_available():
                context = {}
                try:
                    pos = get_positions_data()
                    context['positions'] = pos.get('positions', [])[:3]
                except Exception:
                    pass
                try:
                    mkt = get_market_data()
                    context['market'] = {
                        'sentiment': mkt.get('sentiment'),
                        'indices': [{'name': i['name'], 'change_pct': i['change_pct']}
                                    for i in mkt.get('indices', [])]
                    }
                except Exception:
                    pass
                reply = chat_response(text, context)
                return make_chat_card(reply)
            else:
                return make_help_card()

        return make_help_card()

    except Exception as e:
        logger.error(f"处理消息异常: {e}")
        return make_text_card(f"处理失败,请稍后重试")


# ========== 兼容旧接口 ==========

def compare_stocks_deep(symbols):
    """增强版对比（价格+估值+盈利）"""
    stocks = []
    for sym in symbols:
        try:
            basic = get_stock_data(sym)
            if 'error' in basic:
                continue
            deep = get_stock_deep_data(sym)
            stock = {
                'symbol': sym,
                'name': basic.get('name', sym),
                'price': basic.get('current_price', 0),
                'change_pct': basic.get('change_pct', 0),
                'pe': deep.get('pe', 'N/A'),
                'pb': deep.get('pb', 'N/A'),
                'roe': deep.get('roe', 'N/A'),
                'market_cap': deep.get('total_mv', 'N/A'),
            }
            stocks.append(stock)
        except Exception as e:
            logger.warning(f"对比数据获取失败 {sym}: {e}")
    return {'stocks': stocks, 'count': len(stocks)}