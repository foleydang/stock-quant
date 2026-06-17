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
    FEISHU_ENCRYPT_KEY, BOT_PORT, DB_PATH,
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

# ========== 简单内存缓存（避免重复计算） ==========
import time as _time
_cache = {}  # {key: (value, expiry_time)}

def _cached(key, ttl_seconds=30):
    """缓存装饰器，ttl_seconds内重复调用直接返回缓存"""
    now = _time.time()
    if key in _cache:
        val, expiry = _cache[key]
        if now < expiry:
            return val
    return None

def _cache_set(key, value, ttl_seconds=30):
    _cache[key] = (value, _time.time() + ttl_seconds)

# ========== 股票深度分析处理 ==========

def _build_stock_deep(symbol: str) -> dict:
    """
    构建深度分析卡片：技术指标 + 消息面 + LGBM预测 + 操作建议
    并行执行以降低延迟，单个数据源失败不影响整体
    30秒内重复查询同一股票直接返回缓存
    """
    import concurrent.futures
    
    # 缓存检查
    cache_key = f"stock_deep:{symbol}"
    cached = _cached(cache_key, ttl_seconds=30)
    if cached:
        logger.info(f"缓存命中: {symbol}")
        return cached
    
    # 1. 技术分析（必须，串行先跑）
    data = get_technical_analysis(symbol)
    if 'error' in data:
        return make_text_card(f"技术分析失败: {data['error']}")

    name = data.get('name', symbol)

    # 2. 并行获取：消息面 + LGBM预测 + 操作建议
    def _fetch_news():
        """获取消息面 + LLM情绪分析"""
        try:
            from scheduler import _search_stock_news_brief
            news_data = _search_stock_news_brief(symbol, name)
            if not news_data or not news_data.get('headlines'):
                return None
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
            return news_line
        except Exception as e:
            logger.warning(f"消息面获取失败 {symbol}: {e}")
            return None

    def _fetch_lgbm():
        """获取LGBM预测"""
        try:
            from lgbm_backtest import LGBMBacktesterOptimized
            bt = LGBMBacktesterOptimized()
            result = bt.run_backtest(symbol)
            if result and result.get('predictions'):
                last_pred = result['predictions'][-1]
                up_prob = last_pred.get('up_probability', 0)
                win_rate = result.get('summary', {}).get('winRate', 0)
                signal = '看涨' if up_prob > 0.52 else '看跌' if up_prob < 0.48 else '中性'
                return {'up_prob': up_prob, 'signal': signal, 'win_rate': win_rate}
        except Exception as e:
            logger.warning(f"LGBM预测失败 {symbol}: {e}")
        return None

    def _fetch_recommendations():
        """获取操作建议"""
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
                return f"{action}（置信度{confidence}）— {reason}{t_line}"
        except Exception as e:
            logger.warning(f"操作建议获取失败 {symbol}: {e}")
        return None

    # 并行执行，超时10秒
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_news = executor.submit(_fetch_news)
        future_lgbm = executor.submit(_fetch_lgbm)
        future_recs = executor.submit(_fetch_recommendations)
        
        try:
            news_hint = future_news.result(timeout=10)
            if news_hint:
                data['news_hint'] = news_hint
        except Exception as e:
            logger.warning(f"消息面超时: {e}")
        
        try:
            lgbm_data = future_lgbm.result(timeout=8)
            if lgbm_data:
                data['lgbm'] = lgbm_data
        except Exception as e:
            logger.warning(f"LGBM超时: {e}")
        
        try:
            action_hint = future_recs.result(timeout=8)
            if action_hint:
                data['action_hint'] = action_hint
        except Exception as e:
            logger.warning(f"操作建议超时: {e}")

    # 3. 消息面兜底（新闻搜不到时，用行情数据快速生成摘要）
    if not data.get('news_hint'):
        sigs = data.get('signals', [])
        cp = data.get('change_pct', 0)
        parts = []
        if cp > 2: parts.append('今日大幅上涨')
        elif cp < -2: parts.append('今日大幅下跌')
        elif cp > 0: parts.append('今日小幅上涨')
        elif cp < 0: parts.append('今日小幅下跌')
        else: parts.append('今日平盘')
        if any('放量' in s for s in sigs): parts.append('成交量放大')
        elif any('缩量' in s for s in sigs): parts.append('缩量运行')
        if any('超卖' in s for s in sigs): parts.append('技术面超卖')
        elif any('超买' in s for s in sigs): parts.append('技术面超买')
        if any('空头' in s for s in sigs): parts.append('空头排列中')
        elif any('多头' in s for s in sigs): parts.append('多头排列中')
        if parts:
            data['news_hint'] = f"**📰 消息面**\n📊 基于行情推断：{'，'.join(parts)}。\n💡 暂无相关新闻，关注盘后公告"

    card = make_technical_card(data)
    _cache_set(cache_key, card, ttl_seconds=30)
    return card


def _build_evening_review() -> dict:
    """构建今日总结（复用晚间复盘逻辑）"""
    try:
        import sqlite3
        from technical_indicators import get_technical_analysis

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, stock_name, shares, cost_price, current_price FROM positions WHERE shares > 0")
        positions = cursor.fetchall()
        conn.close()

        lines = ["**📊 今日持仓总结**\n"]
        lines.append(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        total_profit = 0
        total_cost_val = 0
        total_cur_val = 0

        for sym, name, shares, cost, cur_price in positions:
            cost_val = cost * shares
            cur_val = cur_price * shares
            daily_pnl = cur_val - cost_val
            pnl_pct = (cur_price - cost) / cost * 100 if cost > 0 else 0
            total_profit += daily_pnl
            total_cost_val += cost_val
            total_cur_val += cur_val

            is_hk = sym.endswith('.HK')
            c = 'HK$' if is_hk else '¥'

            ta_info = ''
            try:
                ta = get_technical_analysis(sym)
                if 'error' not in ta and ta.get('action_hint'):
                    ta_info = f" | {ta['action_hint']}"
            except Exception:
                pass

            emoji = "🔴" if pnl_pct < -10 else "🟡" if pnl_pct < 0 else "🟢"
            lines.append(f"{emoji} **{name}** {c}{cur_price:.2f} "
                        f"({pnl_pct:+.1f}%) ¥{daily_pnl:+,.0f}{ta_info}\n")

        if total_cost_val > 0:
            total_pnl_pct = (total_cur_val - total_cost_val) / total_cost_val * 100
            emoji_t = "🔴" if total_pnl_pct < -5 else "🟡" if total_pnl_pct < 0 else "🟢"
            lines.append(f"\n**总市值**: ¥{total_cur_val:,.0f} | "
                        f"累计盈亏: {emoji_t} ¥{total_profit:+,.0f} ({total_pnl_pct:+.1f}%)\n")

        from card_templates import make_text_card
        return make_text_card("".join(lines))

    except Exception as e:
        return make_text_card(f"获取今日总结失败: {e}")


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

def process_message(text: str, user_id: str = None) -> dict:
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
            # 今日总结 / 复盘 → 触发晚间复盘逻辑
            if any(kw in text for kw in ['今日总结', '复盘', '日报', '今日表现', '今天怎么样']):
                return _build_evening_review()
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
                reply = chat_response(text, context, user_id=user_id)
                return make_chat_card(reply)
            else:
                return make_help_card()

        # ===== 非chat意图：也保存对话历史，让chat能记住上下文 =====
        if user_id and intent not in ('chat', 'help'):
            try:
                import sqlite3
                from config import DB_PATH
                uid = str(user_id)
                conn = sqlite3.connect(DB_PATH)
                # 保存用户查询
                conn.execute(
                    "INSERT INTO conversation_history (user_id, role, content) VALUES (?, ?, ?)",
                    (uid, "user", text)
                )
                # 保存简要摘要（股票/意图）
                summary = f"[系统] 用户查询了{intent}: {params.get('symbol', '')} - {params}"
                conn.execute(
                    "INSERT INTO conversation_history (user_id, role, content) VALUES (?, ?, ?)",
                    (uid, "assistant", summary)
                )
                # 清理旧记录
                conn.execute(
                    "DELETE FROM conversation_history WHERE user_id=? AND id NOT IN "
                    "(SELECT id FROM conversation_history WHERE user_id=? ORDER BY created_at DESC LIMIT 20)",
                    (uid, uid)
                )
                conn.commit()
                conn.close()
            except Exception as e:
                logger.warning(f"保存上下文失败: {e}")

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