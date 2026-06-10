#!/usr/bin/env python3
"""
定时调度器 - APScheduler + 飞书推送

功能：
1. 盘前提醒（9:25）- 行情 + 重要新闻 + 操作建议
2. 盘中监控（每30分钟）- 异动检测 + 新闻面 + LLM操作建议
3. 盘后总结（15:05）
4. 异动告警（触发式）

注意：不替代现有的 crontab 邥件监控，两者并行运行
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python')
sys.path.insert(0, AGENT_DIR)
sys.path.insert(0, PYTHON_DIR)

from config import DB_PATH, WATCHLIST, AVAILABLE_CASH, FEISHU_TARGET_CHAT_ID, FEISHU_TARGET_OPEN_ID
from data_fetcher import get_positions_data, get_signals_data, get_daily_summary, get_t_suggestions, get_stock_data
from card_templates import (
    make_position_card, make_daily_summary_card,
    make_signal_card, make_alert_card, make_text_card
)

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler()

# 飞书推送目标
FEISHU_TARGET_CHAT_ID = os.environ.get("FEISHU_TARGET_CHAT_ID", "")


def _search_stock_news_brief(symbol: str, name: str) -> Optional[dict]:
    """搜索个股相关新闻（优先百度，备用DuckDuckGo）"""
    # 方案1: 百度新闻搜索（中文效果最好）
    try:
        import requests, re
        short_name = name.replace('股份', '').replace('集团', '').replace('有限', '').replace('-W', '')
        url = 'https://news.baidu.com/ns'
        params = {'word': short_name, 'tn': 'news', 'ie': 'utf-8', 'rn': 8}
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        titles = re.findall(r'<h3[^>]*>.*?<a[^>]*>(.*?)</a>', r.text, re.DOTALL)
        titles = [re.sub(r'<[^>]+>', '', t).strip() for t in titles if t.strip()]
        # 过滤行情页面
        skip_patterns = ['最新价格', '行情_走势图', '净申购', '净赎回', '份额增长', '份额减少', '资金流向']
        headlines = []
        for t in titles[:8]:
            if t and len(t) > 15 and not any(p in t for p in skip_patterns):
                headlines.append({'title': t, 'snippet': '', 'url': ''})
        if headlines:
            return {'headlines': headlines[:5], 'keyword': name}
    except Exception as e:
        logger.warning(f"百度新闻搜索失败 {name}: {e}")

    # 方案2: DuckDuckGo（备用）
    try:
        from search import search
        short_name = name.replace('股份', '').replace('集团', '').replace('有限', '').replace('-W', '')
        results = search(f"{short_name} 利好 利空 最新消息", count=5)
        if not results:
            results = search(f"{short_name} 新闻 股价", count=5)
        if not results:
            return None
        headlines = []
        skip_patterns = ['最新价格', '行情_走势图', '股价行情_财报', '个股资金流向', '股票股价_股价行情']
        for r in results[:5]:
            title = r.get('title', '').strip()
            if title and len(title) > 15 and not any(p in title for p in skip_patterns):
                headlines.append({'title': title, 'snippet': r.get('snippet', '')[:80], 'url': r.get('url', '')})
        if not headlines:
            return None
        return {'headlines': headlines[:3], 'keyword': name}
    except Exception as e:
        logger.warning(f"DuckDuckGo搜索失败 {name}: {e}")
        return None


def _llm_analyze_alert(alert: dict, ta: dict, news: Optional[dict] = None, position_info: dict = None) -> str:
    """用 LLM 综合技术面+消息面，给出精准操作建议"""
    try:
        from llm_client import _call_dashscope_chat, is_available
        if not is_available():
            return _generate_action_hint(alert, ta, position_info)  # 退回规则引擎

        # 构建上下文
        context_parts = []
        is_etf = alert.get('is_etf', False)
        context_parts.append(f"股票: {alert.get('name', '')} ({alert.get('symbol', '')})")
        if is_etf:
            context_parts.append("类型: ETF基金")
        context_parts.append(f"异动类型: {alert['type']}")
        context_parts.append(f"详情: {alert.get('details', '')}")
        context_parts.append(f"涨跌幅: {ta.get('change_pct', 0):.2f}%")
        context_parts.append(f"量比: {ta.get('volume_ratio', 1.0):.1f}")
        context_parts.append(f"当前价: ¥{ta.get('current', 0):.2f}")

        if ta.get('signals'):
            context_parts.append(f"技术信号: {', '.join(ta.get('signals', [])[:5])}")

        if ta.get('supports'):
            context_parts.append(f"支撑位: {', '.join([f'¥{s:.2f}' for s in ta.get('supports', [])[:3]])}")
        if ta.get('resistances'):
            context_parts.append(f"压力位: {', '.join([f'¥{r:.2f}' for r in ta.get('resistances', [])[:3]])}")

        if ta.get('rsi'):
            context_parts.append(f"RSI: {ta.get('rsi', 0):.1f}")

        # 持仓信息
        is_holding = position_info and position_info.get('shares', 0) > 0
        if is_holding:
            profit_pct = position_info.get('profit_pct', 0)
            cost_price = position_info.get('cost_price', 0)
            shares = position_info.get('shares', 0)
            context_parts.append(f"持仓: {shares}股，成本价¥{cost_price:.2f}")
            context_parts.append(f"浮亏/浮盈: {profit_pct:.1f}%")
            if profit_pct <= -20:
                context_parts.append("⚠️ 深度浮亏，不建议止损割肉")
            elif profit_pct <= -5:
                context_parts.append("⚠️ 有浮亏，谨慎操作")

        # 新闻面
        if news and news.get('headlines'):
            context_parts.append("相关新闻:")
            for h in news['headlines'][:3]:
                context_parts.append(f"- {h['title']}")

        context = '\n'.join(context_parts)

        system_prompt = (
            "你是A股操盘助手，根据技术面+消息面给出精准操作建议。\n"
            "规则：\n"
            "1. 只给1-2条最关键的操作建议，简洁有力\n"
            "2. 必须明确说具体动作\n"
            "3. 如果有新闻驱动，说明是利好还是利空\n"
            "4. 每条建议不超过30字\n"
            "5. 用emoji标记：🟢看涨 🔴看跌 ⚠️风险 💡机会 🛡️防守 🎯目标\n"
        )
        if is_etf:
            system_prompt += (
                "6. ETF操作建议规则：\n"
                "   - 跌到支撑位/RSI超卖 → 逢低定投加仓、分批布局\n"
                "   - 涨到压力位/RSI超买 → 分批减仓、波段止盈\n"
                "   - 不建议止损清仓，ETF适合长期持有+波段操作\n"
                "   - 强调'定投'和'逢低布局'而非'买入/卖出'\n"
            )
        else:
            system_prompt += (
                "6. 个股操作建议：明确说'加仓/减仓/止损/持有/观望'\n"
                "7. 如果是持仓股且深度浮亏(>20%)，优先建议'持有观望'而非'止损'\n"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ]

        result = _call_dashscope_chat(messages, max_tokens=200, temperature=0.3)
        if result:
            return result.strip()
    except Exception as e:
        logger.warning(f"LLM分析失败: {e}")

    # 退回规则引擎
    return _generate_action_hint(alert, ta, position_info)


def _generate_action_hint(alert: dict, ta: dict, position_info: dict = None) -> str:
    """规则引擎生成操作建议（LLM不可用时的备用方案）"""
    is_etf = alert.get('is_etf', False) or 'ETF' in alert.get('name', '')
    action_hints = []
    alert_type = alert['type']
    signals = alert.get('signals', []) or ta.get('signals', [])
    supports = ta.get('supports', [])
    resistances = ta.get('resistances', [])
    current = ta.get('current', 0)
    change_pct = ta.get('change_pct', 0)

    # ===== 持仓感知 =====
    # 如果是持仓股，需要考虑浮亏/浮盈状态，避免与邗件监控建议矛盾
    is_holding = position_info and position_info.get('shares', 0) > 0
    profit_pct = position_info.get('profit_pct', 0) if is_holding else 0
    cost_price = position_info.get('cost_price', 0) if is_holding else 0

    if is_holding:
        # 持仓浮亏严重（>20%）：不再建议止损，与邗件逻辑对齐
        if profit_pct <= -20:
            if alert_type in ('大跌', '放量大跌', '缩量大跌'):
                action_hints.append(f'⚠️ 持仓浮亏{profit_pct:.0f}%，下跌中不建议止损割肉')
                if any('超卖' in s for s in signals):
                    action_hints.append('💡 RSI超卖，等反弹补仓机会')
                if supports and current < supports[0] * 1.02:
                    action_hints.append(f'🛡️ 接近支撑位¥{supports[0]:.2f}，观察企稳信号')
                if not action_hints:
                    action_hints.append(f'📉 浮亏{profit_pct:.0f}%，持有观望等反弹')
                return '\n'.join(action_hints)
        # 持仓浮亏（-5%~ -20%）：谨慎建议
        elif profit_pct < -5:
            if alert_type in ('大跌', '放量大跌'):
                action_hints.append(f'⚠️ 持仓浮亏{profit_pct:.0f}%，注意风险控制')
                if any('超卖' in s for s in signals):
                    action_hints.append('💡 RSI超卖，可能有反弹机会')
                if not any(kw in alert_type for kw in ('放量',)):
                    action_hints.append('📉 缩量下跌，观察支撑位')
                return '\n'.join(action_hints)

    # ===== ETF 专属逻辑 =====
    if is_etf:
        if alert_type in ('大跌', '放量大跌', '缩量大跌') and change_pct < 0:
            action_hints.append('💡 ETF逢低可定投加仓，分批布局')
            if supports and current < supports[0] * 1.02:
                action_hints.append(f'🛡️ 接近支撑位，长期持有者可加仓')
            if any('超卖' in s for s in signals):
                action_hints.append('📊 RSI超卖，定投窗口')
        elif alert_type in ('大涨', '放量大涨', '缩量大涨') and change_pct > 0:
            if resistances and current > resistances[0] * 0.98:
                action_hints.append('🎯 接近压力位，波段操作可分批止盈')
            if any('超买' in s for s in signals):
                action_hints.append('⚠️ RSI超买，短期可减仓做波段')
            if not action_hints:
                action_hints.append('📈 短期强势，持有观察')
        elif alert_type == '接近支撑位':
            action_hints.append('💡 ETF接近支撑位，逢低定投/加仓窗口')
            if any('超卖' in s or '金叉' in s for s in signals):
                action_hints.append('📊 配合超卖/金叉信号，反弹概率增大')
        elif alert_type == '接近压力位':
            action_hints.append('🚧 ETF接近压力位，波段操作可减仓')
        elif alert_type == '技术信号':
            important = [s for s in signals if any(kw in s for kw in ['金叉', '死叉', '超买', '超卖'])]
            for s in important[:2]:
                if '金叉' in s:
                    action_hints.append(f'🟢 {s} → 看涨，可加仓')
                elif '超卖' in s:
                    action_hints.append(f'💡 {s} → 定投窗口，逢低布局')
                elif '死叉' in s:
                    action_hints.append(f'⚠️ {s} → 短期偏弱，持有等待')
                elif '超买' in s:
                    action_hints.append(f'⚠️ {s} → 波段减仓')
            if not action_hints:
                action_hints.append('📊 ETF信号触发，可关注波段机会')
        if not action_hints:
            action_hints.append('📊 ETF异动，关注波段定投机会')
        return '\n'.join(action_hints)

    # ===== 个股逻辑 =====
    # 大涨类
    if alert_type in ('大涨', '放量大涨', '缩量大涨') and change_pct > 0:
        if resistances and current > resistances[0] * 0.98:
            action_hints.append(f'🚧 接近压力位¥{resistances[0]:.2f}，考虑分批止盈')
        if any('超买' in s for s in signals):
            action_hints.append('⚠️ RSI超买，短期有回调风险，可减仓1/3')
        if any('死叉' in s for s in signals):
            action_hints.append('🔴 出现死叉信号，注意趋势反转')
        if alert_type == '缩量大涨':
            action_hints.append('🔇 缩量上涨动力不足，观察是否放量突破')
        if not action_hints:
            action_hints.append('📈 短期强势，可持有观察，注意压力位')

    # 大跌类
    elif alert_type in ('大跌', '放量大跌', '缩量大跌') and change_pct < 0:
        if supports and current < supports[0] * 1.02:
            action_hints.append(f'🛡️ 接近支撑位¥{supports[0]:.2f}，若有效跌破需止损')
        if any('超卖' in s for s in signals):
            action_hints.append('💡 RSI超卖，可能有技术反弹机会')
        if alert_type == '放量大跌':
            action_hints.append('⚡ 放量下跌，资金出逃明显，优先止损')
        if not action_hints:
            action_hints.append('📉 短期弱势，关注支撑位能否企稳')

    # 接近支撑位
    elif alert_type == '接近支撑位':
        action_hints.append('🛡️ 若缩量企稳可小仓位加仓，放量跌破则观望')

    # 接近压力位
    elif alert_type == '接近压力位':
        action_hints.append('🚧 若放量突破可加仓，缩量受阻则减仓')

    # 技术信号
    elif alert_type == '技术信号':
        important = [s for s in signals if any(kw in s for kw in ['金叉', '死叉', '超买', '超卖', '突破', '跌破'])]
        for s in important[:2]:
            if '金叉' in s:
                action_hints.append(f'🟢 {s} → 看涨，关注买入时机')
            elif '死叉' in s:
                action_hints.append(f'🔴 {s} → 看跌，注意止损')
            elif '超买' in s:
                action_hints.append(f'⚠️ {s} → 短期过热，考虑减仓')
            elif '超卖' in s:
                action_hints.append(f'💡 {s} → 超跌，留意反弹')
        if not action_hints:
            action_hints.append('🔔 技术信号触发，建议查看详细分析')

    if not action_hints:
        action_hints.append('📊 异动触发，建议查看详细分析')

    return '\n'.join(action_hints)


def morning_alert():
    """盘前提醒 9:25 - 行情 + 重要新闻 + 操作建议"""
    logger.info("盘前提醒触发")

    # 获取自选股行情
    watchlist_prices = []
    for w in WATCHLIST:
        try:
            data = get_stock_data(w['symbol'])
            if 'error' not in data:
                watchlist_prices.append({
                    'name': w['name'],
                    'symbol': w['symbol'],
                    'price': data['current_price'],
                    'change_pct': data.get('change_pct', 0)
                })
        except Exception as e:
            logger.warning(f"获取自选行情失败 {w['symbol']}: {e}")

    # 搜索重要财经新闻
    news_headlines = []
    try:
        from search import search
        results = search("A股 股市 重要新闻 今日", count=5)
        for r in results[:5]:
            title = r.get('title', '').strip()
            if title and len(title) > 10:
                news_headlines.append(title)
    except Exception as e:
        logger.warning(f"盘前新闻搜索失败: {e}")

    # 构建消息
    lines = ["**☀️ 盘前提醒**\n"]
    lines.append(f"日期: {datetime.now().strftime('%Y-%m-%d')}\n\n")
    lines.append("**自选股行情:**\n")
    for wp in watchlist_prices:
        sign = "+" if wp['change_pct'] >= 0 else ""
        color = "green" if wp['change_pct'] >= 0 else "red"
        lines.append(f"- <font color='{color}'>{wp['name']} ¥{wp['price']:.2f} ({sign}{wp['change_pct']:.2f}%)</font>\n")

    if news_headlines:
        lines.append("\n---\n**📰 今日要闻**\n")
        for h in news_headlines[:5]:
            lines.append(f"- {h}\n")

    card = make_text_card("".join(lines))
    _push_card(card)


def intraday_check():
    """盘中智能监控（每30分钟）- 异动 + 新闻 + LLM操作建议"""
    logger.info("盘中智能监控触发")

    try:
        from technical_indicators import get_smart_alerts, get_technical_analysis
        from card_templates import make_alert_card_with_hint, make_technical_card
        
        alerts = get_smart_alerts()
        
        if alerts:
            for a in alerts[:6]:  # 控制推送数量
                # 获取技术分析
                ta = get_technical_analysis(a['symbol'])
                ta_ok = 'error' not in ta
                
                # 搜索相关新闻（异动时才有必要搜）
                news = None
                if a['type'] in ('大涨', '大跌', '放量大涨', '放量大跌'):
                    try:
                        news = _search_stock_news_brief(a['symbol'], a['name'])
                    except Exception as e:
                        logger.warning(f"异动新闻搜索失败 {a['name']}: {e}")
                
                # 查询持仓信息（用于建议与邗件逻辑对齐）
                position_info = None
                try:
                    import sqlite3
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("SELECT shares, cost_price, current_price FROM positions WHERE symbol=?", (a['symbol'],))
                    pos_row = cursor.fetchone()
                    conn.close()
                    if pos_row and pos_row[0] > 0:
                        position_info = {
                            'shares': int(pos_row[0]),
                            'cost_price': float(pos_row[1]),
                            'current_price': float(pos_row[2]),
                            'profit_pct': (float(pos_row[2]) - float(pos_row[1])) / float(pos_row[1]) * 100 if pos_row[1] > 0 else 0
                        }
                except Exception:
                    pass

                # LLM综合分析
                hint = _llm_analyze_alert(a, ta if ta_ok else {}, news, position_info)
                
                if a['type'] in ('大涨', '大跌', '放量大涨', '放量大跌', '缩量大涨', '缩量大跌', '接近支撑位', '接近压力位'):
                    # 异动卡片：附上新闻标题（如果有）
                    news_line = ""
                    if news and news.get('headlines'):
                        titles = [h['title'][:40] for h in news['headlines'][:2]]
                        news_line = "\n\n**📰 相关消息**\n" + '\n'.join([f"- {t}" for t in titles])
                    
                    card = make_alert_card_with_hint(
                        a['type'], a['symbol'], a['name'],
                        a['details'] + news_line, hint,
                        ta_data=ta if ta_ok else None
                    )
                    _push_card(card)
                elif a['type'] == '技术信号':
                    if ta_ok:
                        ta['action_hint'] = hint
                        if news and news.get('headlines'):
                            news_line = '\n'.join([f"- {h['title'][:40]}" for h in news['headlines'][:2]])
                            ta['news_hint'] = f"📰 相关消息\n{news_line}"
                        card = make_technical_card(ta)
                        _push_card(card)
    except Exception as e:
        logger.error(f"智能异动检测失败: {e}")


def daily_summary_push():
    """盘后总结推送 15:05"""
    logger.info("盘后总结推送触发")

    summary_data = get_daily_summary()
    t_data = get_t_suggestions()

    card = make_daily_summary_card(
        summary=summary_data,
        positions=summary_data.get('positions', []),
        signals=summary_data.get('signals', []),
        t_suggestions=t_data
    )
    _push_card(card)


def _push_card(card: dict):
    """推送卡片到飞书（优先私聊 open_id，其次群聊 chat_id）"""
    try:
        from feishu_client import send_card, send_card_to_user
        if FEISHU_TARGET_OPEN_ID:
            send_card_to_user(FEISHU_TARGET_OPEN_ID, card)
        elif FEISHU_TARGET_CHAT_ID:
            send_card(FEISHU_TARGET_CHAT_ID, card)
        else:
            logger.warning("FEISHU_TARGET_OPEN_ID / FEISHU_TARGET_CHAT_ID 未配置，无法推送")
    except Exception as e:
        logger.error(f"推送飞书卡片失败: {e}")


def setup_scheduler():
    """配置定时任务"""
    # 盘前提醒 9:25
    scheduler.add_job(
        morning_alert,
        CronTrigger(hour=9, minute=25, day_of_week='mon-fri'),
        id='morning_alert',
        name='盘前提醒',
        misfire_grace_time=60
    )

    # 盘中监控 每30分钟（避开午休11:30-13:00和盘前时段）
    # A股交易时段: 9:30-11:30, 13:00-15:00
    # 推送时间: 9:30, 10:00, 10:30, 11:00, 13:00, 13:30, 14:00, 14:30
    scheduler.add_job(
        intraday_check,
        CronTrigger(hour='9', minute='30', day_of_week='mon-fri'),
        id='intraday_check_0930',
        name='盘中监控(9:30)',
        misfire_grace_time=120
    )
    scheduler.add_job(
        intraday_check,
        CronTrigger(hour='10', minute='0,30', day_of_week='mon-fri'),
        id='intraday_check_10',
        name='盘中监控(10点)',
        misfire_grace_time=120
    )
    scheduler.add_job(
        intraday_check,
        CronTrigger(hour='11', minute='0', day_of_week='mon-fri'),
        id='intraday_check_1100',
        name='盘中监控(11:00)',
        misfire_grace_time=120
    )
    scheduler.add_job(
        intraday_check,
        CronTrigger(hour='13-14', minute='0,30', day_of_week='mon-fri'),
        id='intraday_check_pm',
        name='盘中监控(下午)',
        misfire_grace_time=120
    )

    # 盘后总结 15:05
    scheduler.add_job(
        daily_summary_push,
        CronTrigger(hour=15, minute=5, day_of_week='mon-fri'),
        id='daily_summary_push',
        name='盘后总结',
        misfire_grace_time=120
    )

    logger.info("定时任务已配置: 盘前9:25, 盘中9:30/10:00/10:30/11:00/13:00/13:30/14:00/14:30, 盘后15:05")


def start_scheduler():
    """启动调度器"""
    setup_scheduler()
    scheduler.start()
    logger.info("✓ APScheduler 已启动")


def stop_scheduler():
    """停止调度器"""
    scheduler.shutdown(wait=False)
    logger.info("APScheduler 已停止")