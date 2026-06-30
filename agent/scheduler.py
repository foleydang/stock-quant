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
import time
from datetime import datetime, date
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# ========== 交易日历（带缓存） ==========
_trading_calendar = None
_trading_calendar_date = None

def is_trading_day() -> bool:
    """判断今天是否为A股交易日（含节假日判断）"""
    global _trading_calendar, _trading_calendar_date
    today_str = date.today().strftime('%Y-%m-%d')
    
    # 缓存：同一天不重复拉取
    if _trading_calendar is not None and _trading_calendar_date == today_str:
        return today_str in _trading_calendar
    
    try:
        import akshare as ak
        df = ak.tool_trade_date_hist_sina()
        _trading_calendar = set(df['trade_date'].astype(str).values)
        _trading_calendar_date = today_str
        result = today_str in _trading_calendar
        if not result:
            logger.info(f"{today_str} 非交易日，跳过定时推送")
        return result
    except Exception as e:
        logger.warning(f"交易日历获取失败，回退到周一至周五判断: {e}")
        # 回退：周一至周五
        return date.today().weekday() < 5

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
    """搜索个股近7日相关新闻（Bing News RSS），带日期过滤"""
    from datetime import timedelta
    short_name = name.replace('股份', '').replace('集团', '').replace('有限', '').replace('-W', '')
    
    skip_patterns = ['最新价格', '行情_走势图', '股价行情_财报', '个股资金流向', '股票股价_股价行情',
                    '东方财富网', '同花顺财经', '英为财情', '五档盘口', '实时行情数据',
                    '法律意见书', '研究报告', 'F10', '最新行情', '净申购', '净赎回']

    # 方案1: Bing News RSS（最可靠，带日期）
    try:
        import requests, re
        url = 'https://www.bing.com/news/search'
        params = {'q': short_name, 'qft': 'interval="7"', 'format': 'rss'}
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        
        items = re.findall(r'<item>(.*?)</item>', r.text, re.DOTALL)
        headlines = []
        for item in items:
            title_m = re.search(r'<title>(.*?)</title>', item)
            desc_m = re.search(r'<description>(.*?)</description>', item)
            link_m = re.search(r'<link>(.*?)</link>', item)
            date_m = re.search(r'<pubDate>(.*?)</pubDate>', item)
            if not title_m:
                continue
            title = title_m.group(1).strip()
            if len(title) < 12 or any(p in title for p in skip_patterns):
                continue
            snippet = desc_m.group(1).strip()[:200] if desc_m else ''
            url = link_m.group(1).strip() if link_m else ''
            time_str = date_m.group(1)[:16] if date_m else ''
            headlines.append({'title': title, 'snippet': snippet, 'url': url, 'time': time_str})
        
        if headlines:
            return {'headlines': headlines[:5], 'keyword': name}
    except Exception as e:
        logger.warning(f"Bing新闻搜索失败 {name}: {e}")

    # 备用: DuckDuckGo
    try:
        from search import search
        from datetime import timedelta
        short_name = name.replace('股份', '').replace('集团', '').replace('有限', '').replace('-W', '')
        results = search(f"{short_name} 最新消息", count=8)
        if not results:
            return None
        headlines = []
        skip_patterns = ['最新价格', '行情_走势图', '股价行情_财报', '个股资金流向', '股票股价_股价行情',
                        '东方财富网', '同花顺财经', '英为财情', '五档盘口', '实时行情数据',
                        '法律意见书', '研究报告', 'F10', '最新行情']
        # 近7天日期列表，用于识别旧闻
        recent_days = [(datetime.now() - timedelta(days=i)).strftime('%m月%d日') for i in range(7)]
        old_days = [(datetime.now() - timedelta(days=i)).strftime('%m月%d日') for i in range(8, 15)]
        for r in results:
            title = r.get('title', '').strip()
            snippet = r.get('snippet', '')
            if not title or len(title) < 12:
                continue
            if any(p in title for p in skip_patterns):
                continue
            # 时间过滤：只有明确旧闻（>7天）才过滤
            combined = title + snippet
            has_date = any(d in combined for d in recent_days + old_days)
            if has_date:
                is_old = any(d in combined for d in old_days)
                if is_old:
                    continue
            # 无日期信息 → 保留
            headlines.append({'title': title, 'snippet': snippet[:120], 'url': r.get('url', '')})
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

        # 港股标识
        is_hk = alert.get('symbol', '').endswith('.HK') or ta.get('is_hk', False)
        currency = ta.get('currency', 'HK$' if is_hk else '¥')

        # 构建上下文
        context_parts = []
        is_etf = alert.get('is_etf', False)
        context_parts.append(f"股票: {alert.get('name', '')} ({alert.get('symbol', '')})")
        if is_etf:
            context_parts.append("类型: ETF基金")
        if is_hk:
            context_parts.append("类型: 港股（货币单位港币HK$，非人民币¥）")
        context_parts.append(f"异动类型: {alert['type']}")
        context_parts.append(f"详情: {alert.get('details', '')}")
        context_parts.append(f"涨跌幅: {ta.get('change_pct', 0):.2f}%")
        context_parts.append(f"量比: {ta.get('volume_ratio', 1.0):.1f}")
        context_parts.append(f"当前价: {currency}{ta.get('current', 0):.2f}")

        if ta.get('signals'):
            context_parts.append(f"技术信号: {', '.join(ta.get('signals', [])[:5])}")

        if ta.get('supports'):
            context_parts.append(f"支撑位: {', '.join([f'{currency}{s:.2f}' for s in ta.get('supports', [])[:3]])}")
        if ta.get('resistances'):
            context_parts.append(f"压力位: {', '.join([f'{currency}{r:.2f}' for r in ta.get('resistances', [])[:3]])}")

        if ta.get('rsi'):
            context_parts.append(f"RSI: {ta.get('rsi', 0):.1f}")

        # 持仓信息
        is_holding = position_info and position_info.get('shares', 0) > 0
        if is_holding:
            profit_pct = position_info.get('profit_pct', 0)
            cost_price = position_info.get('cost_price', 0)
            shares = position_info.get('shares', 0)
            context_parts.append(f"持仓: {shares}股，成本价{currency}{cost_price:.2f}")
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
            "你是操盘助手，根据技术面+消息面给出精准操作建议。\n"
            "规则：\n"
            "1. 只给1-2条最关键的操作建议，简洁有力\n"
            "2. 必须明确说具体动作\n"
            "3. 如果有新闻驱动，说明是利好还是利空\n"
            "4. 每条建议不超过30字\n"
            "5. 用emoji标记：🟢看涨 🔴看跌 ⚠️风险 💡机会 🛡️防守 🎯目标\n"
            "6. 建议必须与异动类型一致：\n"
            "   - 接近压力位 → 说压力位风险，不要说'近支撑'\n"
            "   - 接近支撑位 → 说支撑位机会，不要说'近压力'\n"
            "   - 大跌/放量大跌 → 说风险控制，不要盲目乐观\n"
            "   - 大涨/放量大涨 → 说追高风险，不要盲目看多\n"
        )
        if is_etf:
            system_prompt += (
                "6. ETF操作建议规则：\n"
                "   - 跌到支撑位/RSI超卖 → 逢低定投加仓、分批布局\n"
                "   - 涨到压力位/RSI超买 → 分批减仓、波段止盈\n"
                "   - 不建议止损清仓，ETF适合长期持有+波段操作\n"
                "   - 强调'定投'和'逢低布局'而非'买入/卖出'\n"
            )
        elif is_hk:
            system_prompt += (
                "6. 港股操作建议规则：\n"
                "   - 港股无涨跌幅限制，波动更大，注意风险控制\n"
                "   - 明确说'加仓/减仓/止损/持有/观望'\n"
                "   - 港股T+0可日内交易，但建议分批操作\n"
                "   - 价格单位是港币(HK$)，不是人民币(¥)\n"
                "   - 深度浮亏(>20%)优先建议'持有观望'而非'止损'\n"
            )
        else:
            system_prompt += (
                "6. A股个股操作建议：明确说'加仓/减仓/止损/持有/观望'\n"
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
    is_hk = alert.get('symbol', '').endswith('.HK') or ta.get('is_hk', False)
    currency = ta.get('currency', 'HK$' if is_hk else '¥')
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
                    action_hints.append(f'🛡️ 接近支撑位{currency}{supports[0]:.2f}，观察企稳信号')
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

    # ===== 港股个股逻辑 =====
    if is_hk and not is_etf:
        if alert_type in ('大跌', '放量大跌', '缩量大跌') and change_pct < 0:
            action_hints.append('🔴 港股大跌，注意无涨跌幅限制风险')
            if supports and current < supports[0] * 1.02:
                action_hints.append(f'🛡️ 接近支撑位{currency}{supports[0]:.2f}')
            if any('超卖' in s for s in signals):
                action_hints.append('💡 RSI超卖，可能有反弹机会')
            if not action_hints:
                action_hints.append('📉 港股弱势，持有观望')
            return '\n'.join(action_hints)
        elif alert_type in ('大涨', '放量大涨', '缩量大涨') and change_pct > 0:
            if resistances and current > resistances[0] * 0.98:
                action_hints.append(f'🚧 接近压力位{currency}{resistances[0]:.2f}')
            if any('超买' in s for s in signals):
                action_hints.append('⚠️ RSI超买，注意回调')
            if not action_hints:
                action_hints.append('📈 港股强势，可持有观察')
            return '\n'.join(action_hints)
        elif alert_type == '接近支撑位':
            action_hints.append(f'🛡️ 港股接近支撑位{currency}{supports[0]:.2f}，可关注企稳信号')
            return '\n'.join(action_hints)
        elif alert_type == '接近压力位':
            action_hints.append(f'🚧 港股接近压力位{currency}{resistances[0]:.2f}')
            return '\n'.join(action_hints)

    # ===== A股个股逻辑 =====
    # 大涨类
    if alert_type in ('大涨', '放量大涨', '缩量大涨') and change_pct > 0:
        if resistances and current > resistances[0] * 0.98:
            action_hints.append(f'🚧 接近压力位{currency}{resistances[0]:.2f}，考虑分批止盈')
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
            action_hints.append(f'🛡️ 接近支撑位{currency}{supports[0]:.2f}，若有效跌破需止损')
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
    """盘前提醒 9:25 - 持仓分析 + 关键位 + AI新闻摘要"""
    if not is_trading_day():
        return
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

    # 获取持仓股技术分析（支撑/压力位 + 信号）
    position_advice = []
    try:
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, stock_name, shares, cost_price FROM positions WHERE shares > 0")
        positions = cursor.fetchall()
        conn.close()
        if positions:
            from technical_indicators import get_technical_analysis
            for sym, name, shares, cost in positions:
                try:
                    ta = get_technical_analysis(sym)
                    if 'error' not in ta:
                        current = ta['current']
                        profit_pct = (current - cost) / cost * 100
                        is_hk = sym.endswith('.HK')
                        currency = 'HK$' if is_hk else '¥'
                        advice = {
                            'symbol': sym, 'name': name, 'shares': shares,
                            'cost': cost, 'current': current,
                            'profit_pct': profit_pct, 'currency': currency,
                            'supports': ta.get('supports', [])[:2],
                            'resistances': ta.get('resistances', [])[:2],
                            'action_hint': ta.get('action_hint', ''),
                            'near_support': any(abs(current - s) / current < 0.02 for s in ta.get('supports', [])),
                            'near_resistance': any(abs(current - r) / current < 0.02 for r in ta.get('resistances', [])),
                        }
                        position_advice.append(advice)
                except Exception as e:
                    logger.warning(f"持仓分析失败 {sym}: {e}")
    except Exception as e:
        logger.warning(f"持仓查询失败: {e}")

    # 搜索重要财经新闻
    news_headlines = []
    try:
        import requests as req
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        }
        r = req.get('https://api-one.wallstcn.com/apiv1/content/lives?channel=global-channel&limit=10',
                    headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data.get('code') == 20000 and 'data' in data:
                items = data['data'].get('items', [])
                for item in items[:6]:
                    title = item.get('title', '') or item.get('content', '')
                    if title:
                        import re as _re
                        title = _re.sub(r'<[^>]+>', '', title).strip()
                        if title and len(title) > 8:
                            news_headlines.append(title)
    except Exception as e:
        logger.warning(f"盘前新闻搜索失败: {e}")

    # AI 新闻摘要（用LLM提炼关键信息）
    news_summary = ''
    if news_headlines:
        try:
            from llm_client import _call_dashscope_chat, is_available as _avail
            if _avail():
                msgs = [
                    {"role": "system", "content": "你是财经助手，用3-4句话总结今日盘前要闻，每条不超过30字，用emoji标记方向。"},
                    {"role": "user", "content": "今日新闻:\n" + '\n'.join(news_headlines[:6])}
                ]
                result = _call_dashscope_chat(msgs, max_tokens=150, temperature=0.3)
                if result:
                    news_summary = result.strip()
        except Exception as e:
            logger.warning(f"AI新闻摘要失败: {e}")

    # 构建消息
    lines = ["**☀️ 盘前提醒**\n"]
    lines.append(f"日期: {datetime.now().strftime('%Y-%m-%d %A')}\n\n")

    # 持仓关注
    if position_advice:
        lines.append("**📊 持仓关注**\n")
        for pa in position_advice:
            sign = "+" if pa['profit_pct'] >= 0 else ""
            color = "green" if pa['profit_pct'] >= 0 else "red"
            emoji = "🟢" if pa['profit_pct'] >= 0 else "🔴" if pa['profit_pct'] < -10 else "🟡"
            lines.append(f"{emoji} {pa['name']} {pa['currency']}{pa['current']:.2f} "
                        f"({sign}{pa['profit_pct']:.1f}%)\n")
            # 关键位提示
            if pa['near_support'] and pa['supports']:
                lines.append(f"  ⚠️ 接近支撑 {pa['currency']}{pa['supports'][0]:.2f}，关注企稳信号\n")
            if pa['near_resistance'] and pa['resistances']:
                lines.append(f"  🎯 接近压力 {pa['currency']}{pa['resistances'][0]:.2f}，关注突破\n")
            if pa['action_hint']:
                lines.append(f"  💡 {pa['action_hint']}\n")
        lines.append("\n")

    # 自选股行情（精简）
    lines.append("**自选股行情:**\n")
    for wp in watchlist_prices:
        sign = "+" if wp['change_pct'] >= 0 else ""
        color = "green" if wp['change_pct'] >= 0 else "red"
        is_etf_or_low = 'ETF' in wp.get('name', '') or wp['price'] < 1.0
        price_fmt = f"{wp['price']:.3f}" if is_etf_or_low else f"{wp['price']:.2f}"
        lines.append(f"- <font color='{color}'>{wp['name']} ¥{price_fmt} ({sign}{wp['change_pct']:.2f}%)</font>\n")

    # AI 新闻摘要
    if news_summary:
        lines.append("\n---\n**📰 今日要闻**\n")
        lines.append(f"{news_summary}\n")
    elif news_headlines:
        lines.append("\n---\n**📰 今日要闻**\n")
        for h in news_headlines[:5]:
            lines.append(f"- {h}\n")

    card = make_text_card("".join(lines))
    _push_card(card)



# ========== 异动去重：同一股票同一方向，每天最多推3次，跌幅须加深1%+ ==========
_recent_alerts = {}  # {(symbol, alert_type): [(timestamp, change_pct), ...]}
_ALERT_MAX_PER_DAY = 3  # 同方向每天最多推3次
_ALERT_DEEPEN_THRESHOLD = 1.0  # 跌幅/涨幅须比上次加深1%以上才再推


def _cleanup_old_alerts():
    """跨天清理：新的一天重置所有记录，每天独立统计3次配额"""
    import time
    today = time.strftime('%Y%m%d')
    if not hasattr(_cleanup_old_alerts, '_date'):
        _cleanup_old_alerts._date = today
        return
    if _cleanup_old_alerts._date != today:
        _recent_alerts.clear()
        _cleanup_old_alerts._date = today
def _is_alert_duplicate(symbol: str, alert_type: str, change_pct: float) -> bool:
    """判断是否重复异动
    规则：
    1. 同方向每天最多推3次
    2. 跌幅/涨幅须比上次推送加深1%以上才再推
    """
    import time
    _cleanup_old_alerts()  # 先清理过期记录
    key = (symbol, alert_type)
    records = _recent_alerts.get(key, [])

    # 规则1：次数上限
    if len(records) >= _ALERT_MAX_PER_DAY:
        return True

    # 规则2：跌幅/涨幅必须加深
    if records:
        last_pct = records[-1][1]
        # 大跌类：跌幅须更深（更负）1%以上
        if '跌' in alert_type and change_pct > last_pct - _ALERT_DEEPEN_THRESHOLD:
            return True  # 跌幅没加深，不推
        # 大涨类：涨幅须更高1%以上
        if '涨' in alert_type and change_pct < last_pct + _ALERT_DEEPEN_THRESHOLD:
            return True  # 涨幅没加深，不推

    # 记录本次推送
    records.append((time.time(), change_pct))
    _recent_alerts[key] = records
    return False


def _log_alert(alert_type: str, total_loss_pct: float = None):
    """记录报警日志，用于冷却判断"""
    try:
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO alert_log (alert_type, total_loss_pct) VALUES (?, ?)",
            (alert_type, total_loss_pct)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"记录报警日志失败: {e}")


def _risk_alert_check():
    """持仓组合风险预警：单日亏损过大 / 多只同时大跌 / 跌破支撑
    
    冷却机制：同类型预警每天最多发1次，除非亏损恶化>2%
    """
    try:
        import sqlite3
        from datetime import datetime, timedelta
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, stock_name, shares, cost_price, current_price FROM positions WHERE shares > 0")
        positions = cursor.fetchall()
        if not positions or len(positions) < 2:
            conn.close()
            return

        # ===== 冷却检查：同类型预警今天发过没？ =====
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute(
            "SELECT alert_type, total_loss_pct, created_at FROM alert_log "
            "WHERE created_at >= ? ORDER BY created_at DESC",
            (today,)
        )
        recent_alerts = cursor.fetchall()
        conn.close()
        
        # 检查今天是否已发过风险预警
        today_risk_alert = None
        for alert_type, loss_pct, created_at in recent_alerts:
            if alert_type == 'risk':
                today_risk_alert = (loss_pct, created_at)
                break
        
        # 先算总亏损，用于判断是否需要更新
        total_loss = 0
        total_cost = 0
        for sym, name, shares, cost, cur in positions:
            cost_val = cost * shares
            total_cost += cost_val
            pnl_pct = (cur - cost) / cost * 100 if cost > 0 else 0
            total_loss += pnl_pct * cost_val
        total_loss_pct = total_loss / total_cost if total_cost > 0 else 0
        
        # 如果今天已发过且亏损没恶化>2%，跳过
        if today_risk_alert and total_loss_pct < -5:
            last_loss = today_risk_alert[0]
            if total_loss_pct > last_loss - 2:  # 亏损没恶化2%以上
                logger.info(f"风险预警已冷却: 今天已发过({last_loss:.1f}%), 当前({total_loss_pct:.1f}%), 差{total_loss_pct - last_loss:.1f}%")
                return
            logger.info(f"风险预警恶化: {last_loss:.1f}% → {total_loss_pct:.1f}%，重新推送")

        from technical_indicators import get_technical_analysis

        dropping_count = 0
        broken_supports = []

        for sym, name, shares, cost, cur in positions:

            # 检查是否大跌 >2%
            try:
                from data_fetcher import get_stock_data
                sd = get_stock_data(sym)
                if 'error' not in sd and sd.get('change_pct', 0) < -2:
                    dropping_count += 1
            except Exception:
                pass

            # 检查是否跌破关键支撑
            try:
                ta = get_technical_analysis(sym)
                if 'error' not in ta:
                    supports = ta.get('supports', [])
                    current = ta.get('current', 0)
                    if supports and current < supports[0] * 0.98:
                        is_hk = sym.endswith('.HK')
                        c = 'HK$' if is_hk else '¥'
                        broken_supports.append(f"{name} {c}{current:.2f} 跌破支撑 {c}{supports[0]:.2f}")
            except Exception:
                pass

        # 触发条件1：总持仓亏损 >5%
        if total_loss_pct < -5:
            from card_templates import make_text_card
            lines = ["**⚠️ 风险预警**\n"]
            lines.append(f"总持仓亏损 {total_loss_pct:.1f}%，超过5%警戒线\n")
            lines.append("\n**📋 持仓明细：**\n")
            # 按亏损从大到小排序
            stock_details = []
            for sym, name, shares, cost, cur in positions:
                pnl = (cur - cost) / cost * 100 if cost > 0 else 0
                cost_val = cost * shares
                stock_details.append((name, sym, pnl, cost_val, cur, cost))
            stock_details.sort(key=lambda x: x[2])
            
            for name, sym, pnl, cost_val, cur, cost in stock_details:
                emoji = "🟢" if pnl >= 0 else "🔴" if pnl < -15 else "🟡"
                is_hk = sym.endswith('.HK')
                c = 'HK$' if is_hk else '¥'
                lines.append(f"{emoji} {name} {c}{cur:.2f} | 成本{c}{cost:.2f} | {pnl:+.1f}%\n")
            
            lines.append(f"\n💡 建议：检查是否需要减仓或止损\n")
            _push_card(make_text_card("".join(lines)))
            # 记录报警日志，防止重复推送
            _log_alert('risk', total_loss_pct)
            logger.info(f"风险预警: 总持仓亏损 {total_loss_pct:.1f}%")
            return

        # 触发条件2：>2只持仓同时大跌
        if dropping_count >= 2:
            from card_templates import make_text_card
            lines = ["**⚠️ 联动下跌预警**\n"]
            lines.append(f"{dropping_count}只持仓同时大跌(>2%)，可能系统性风险\n")
            lines.append(f"建议：检查大盘走势，考虑减仓\n")
            _push_card(make_text_card("".join(lines)))
            _log_alert('drop', total_loss_pct)
            logger.info(f"联动下跌预警: {dropping_count}只")
            return

        # 触发条件3：持仓跌破支撑
        if broken_supports:
            from card_templates import make_text_card
            lines = ["**⚠️ 跌破支撑预警**\n"]
            for bs in broken_supports[:3]:
                lines.append(f"{bs}\n")
            lines.append("建议：关注是否有效跌破，考虑止损/减仓\n")
            _push_card(make_text_card("".join(lines)))
            _log_alert('break_support', total_loss_pct)
            logger.info(f"跌破支撑预警: {len(broken_supports)}只")

    except Exception as e:
        logger.warning(f"风险预警检查失败: {e}")


def intraday_alert_monitor():
    """盘中异动轮询 - 分时段频率 + 技术信号限频推送"""
    if not is_trading_day():
        return
    # ===== 频率控制：高波动时段每5分钟，其他时段每10分钟 =====
    now = datetime.now()
    m = now.minute
    h = now.hour
    # 早盘高波动 9:30-10:05、下午开盘 13:00-13:30 → 每5分钟
    is_high_freq = (
        (h == 9 and m >= 30) or (h == 10 and m <= 5) or
        (h == 13 and m <= 30)
    )
    # 非高波动时段：只在分钟为0的倍数时执行
    if not is_high_freq and m % 10 != 0:
        return
    # 9:30之前不执行（还没开盘）
    if h == 9 and m < 30:
        return
    # 15:00之后不执行
    if h >= 15:
        return

    logger.info(f"盘中异动轮询触发 ({h:02d}:{m:02d})")

    try:
        from technical_indicators import get_smart_alerts
        alerts = get_smart_alerts()
        if not alerts:
            return

        # 1. 涨跌异动（始终推送）
        move_alerts = [a for a in alerts if a['type'] in ('大涨', '大跌', '放量大涨', '放量大跌', '缩量大涨', '缩量大跌')]

        # 2. 技术信号（限频：相同symbol+type每天最多1次，避免频繁推送）
        tech_alerts = [a for a in alerts if a['type'] == '技术信号']
        # 3. 接近支撑/压力位（限频：每天最多1次）
        sr_alerts = [a for a in alerts if a['type'] in ('接近支撑位', '接近压力位')]

        # 合并去重（按symbol优先涨跌，再技术信号，再支撑压力）
        seen_symbols = set()
        merged = []
        for a in move_alerts:
            if a['symbol'] not in seen_symbols:
                merged.append(a)
                seen_symbols.add(a['symbol'])
        for a in tech_alerts:
            if a['symbol'] not in seen_symbols and len(a.get('details', '')) > 5:
                if _is_alert_duplicate(a['symbol'], '技术信号', 0):
                    continue
                merged.append(a)
                seen_symbols.add(a['symbol'])
        for a in sr_alerts:
            if a['symbol'] not in seen_symbols:
                if _is_alert_duplicate(a['symbol'], a['type'], 0):
                    continue
                merged.append(a)
                seen_symbols.add(a['symbol'])

        if not merged:
            return

        for a in merged[:5]:  # 每次最多推5条
            if _is_alert_duplicate(a['symbol'], a['type'], a.get('change_pct', 0)):
                logger.debug(f"异动去重: {a['name']} {a['type']} {a.get('change_pct', 0):.2f}% 已推送或跌幅未加深")
                continue

            try:
                from technical_indicators import get_technical_analysis
                ta = get_technical_analysis(a['symbol'])
                ta_ok = 'error' not in ta
            except Exception:
                ta_ok = False
                ta = {}

            # 搜索新闻（跌幅>3%才搜）
            news = None
            if abs(a.get('change_pct', 0)) > 3:
                try:
                    news = _search_stock_news_brief(a['symbol'], a['name'])
                except Exception:
                    pass

            # 查持仓
            position_info = None
            try:
                import sqlite3
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT shares, cost_price, current_price FROM positions WHERE symbol=?", (a['symbol'],))
                pos_row = cursor.fetchone()
                conn.close()
                if pos_row and pos_row[0] > 0:
                    position_info = {'shares': int(pos_row[0]), 'cost_price': float(pos_row[1]), 'current_price': float(pos_row[2]), 'profit_pct': (float(pos_row[2]) - float(pos_row[1])) / float(pos_row[1]) * 100 if pos_row[1] > 0 else 0}
            except Exception:
                pass

            hint = _llm_analyze_alert(a, ta if ta_ok else {}, news, position_info)

            news_line = ""
            if news and news.get('headlines'):
                titles = [h['title'][:40] for h in news['headlines'][:2]]
                news_line = "\n\n**📰 相关消息**\n" + '\n'.join([f"- {t}" for t in titles])

            from card_templates import make_alert_card_with_hint
            card = make_alert_card_with_hint(
                a['type'], a['symbol'], a['name'],
                a['details'] + news_line, hint,
                ta_data=ta if ta_ok else None
            )
            _push_card(card)
            logger.info(f"⚡ 异动推送: {a['name']} {a['type']} {a.get('change_pct', 0):.2f}%")

        # ===== 风险预警：持仓组合风险 =====
        _risk_alert_check()

    except Exception as e:
        logger.error(f"盘中异动轮询失败: {e}")

def intraday_check():
    """盘中开盘全量检查 - 异动 + 新闻 + LLM操作建议"""
    if not is_trading_day():
        return
    logger.info("盘中开盘监控触发")

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
    if not is_trading_day():
        return
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


def evening_review():
    """晚间复盘推送 18:00 - 今日总结 + 明日计划"""
    if not is_trading_day():
        return
    logger.info("晚间复盘推送触发")

    try:
        import sqlite3

        # 1. 持仓今日表现
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, stock_name, shares, cost_price, current_price FROM positions WHERE shares > 0")
        positions = cursor.fetchall()
        conn.close()

        lines = ["**🌙 晚间复盘**\n"]
        lines.append(f"日期: {datetime.now().strftime('%Y-%m-%d')}\n\n")

        total_profit = 0
        total_cost_val = 0
        total_cur_val = 0

        if positions:
            from technical_indicators import get_technical_analysis
            position_summary = []
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

                # 技术分析
                ta_info = ''
                try:
                    ta = get_technical_analysis(sym)
                    if 'error' not in ta:
                        if ta.get('action_hint'):
                            ta_info = f" | {ta['action_hint']}"
                except Exception:
                    pass

                emoji = "🔴" if pnl_pct < -10 else "🟡" if pnl_pct < 0 else "🟢"
                position_summary.append(
                    f"{emoji} **{name}** {c}{cur_price:.2f} "
                    f"({pnl_pct:+.1f}%) ¥{daily_pnl:+,.0f}{ta_info}"
                )

            total_pnl_pct = (total_cur_val - total_cost_val) / total_cost_val * 100 if total_cost_val > 0 else 0
            emoji_t = "🔴" if total_pnl_pct < -5 else "🟡" if total_pnl_pct < 0 else "🟢"
            lines.append(f"**📊 持仓总览**: {emoji_t} 总市值 ¥{total_cur_val:,.0f} "
                        f"累计盈亏 ¥{total_profit:+,.0f} ({total_pnl_pct:+.1f}%)\n\n")

            for ps in position_summary:
                lines.append(f"{ps}\n")
            lines.append("\n")
        else:
            lines.append("当前无持仓\n\n")

        # 2. AI 明日展望
        try:
            from llm_client import _call_dashscope_chat, is_available as _avail
            if _avail() and position_summary:
                context = "持仓: " + ', '.join([
                    f"{name}({pnl_pct:+.1f}%)"
                    for _, name, _, _, _, pnl_pct in position_summary
                ])
                msgs = [
                    {"role": "system", "content": "你是交易复盘助手。根据今日持仓表现，给出明天操作建议。3-4条，每条不超过30字，用emoji标记方向。"},
                    {"role": "user", "content": f"{context}\n请给出明日操作要点。"}
                ]
                result = _call_dashscope_chat(msgs, max_tokens=150, temperature=0.3)
                if result:
                    lines.append(f"---\n**💡 明日要点**\n{result.strip()}\n")
            elif not position_summary:
                lines.append("---\n**💡 明日要点**\n暂无持仓，关注自选股机会\n")
        except Exception as e:
            logger.warning(f"AI复盘失败: {e}")

        from card_templates import make_text_card
        card = make_text_card("".join(lines))
        _push_card(card)
        logger.info("晚间复盘推送完成")

    except Exception as e:
        logger.error(f"晚间复盘推送失败: {e}")


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


def save_30min_kline_data(symbols: list = None):
    """保存30分钟K线数据到数据库（东方财富API）
    
    在盘中监控时调用，确保 kline_30m 表有当日最新数据，
    后续 qlib 模型可直接使用。
    """
    import sqlite3
    import requests
    
    if symbols is None:
        # 默认：所有持仓股 + 自选股
        conn = sqlite3.connect(DB_PATH)
        symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM positions").fetchall()]
        for w in WATCHLIST:
            s = w.get('symbol', '')
            if s and s not in symbols:
                symbols.append(s)
        conn.close()
    
    if not symbols:
        return
    
    conn = sqlite3.connect(DB_PATH)
    new_total = 0
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://quote.eastmoney.com/',
    }
    
    for sym in symbols:
        if sym.endswith('.HK'):
            continue
        code = sym.split('.')[0]
        secid = f"0.{code}" if sym.endswith('.SZ') else f"1.{code}"
        
        try:
            r = requests.get(
                'http://push2his.eastmoney.com/api/qt/stock/kline/get',
                params={'secid': secid, 'fields1': 'f1,f2,f3', 'fields2': 'f51,f52,f53,f54,f55,f56,f57',
                        'klt': '30', 'fqt': '1', 'end': '20260625', 'lmt': 8},
                headers=headers, timeout=10
            )
            data = r.json()
            klines = data.get('data', {}).get('klines', [])
        except Exception:
            continue
        
        for line in klines:
            parts = line.split(',')
            if len(parts) < 6:
                continue
            dt, o, c, h, l, v = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            if not conn.execute("SELECT 1 FROM kline_30m WHERE symbol=? AND date=?", (sym, dt)).fetchone():
                conn.execute(
                    "INSERT OR IGNORE INTO kline_30m (symbol,date,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?)",
                    (sym, dt, o, h, l, c, v))
                new_total += 1
    
    if new_total > 0:
        conn.commit()
        logger.info(f"💾 30min K线数据已保存: +{new_total}条 ({len(symbols)}只股票)")
    conn.close()


def v8_intraday_push():
    """v8 模型盘中推送 — 每30分钟预测排名 + 持仓加减仓建议"""
    if not is_trading_day():
        return
    logger.info("v8 模型预测推送触发")

    try:
        from v8_predictor import get_predictor, format_feishu_message, TOP_N_CANDIDATES
        import sqlite3
        predictor = get_predictor()

        if not predictor.is_loaded():
            logger.info("v8 模型未就绪，跳过推送")
            return

        # 获取持仓
        positions = []
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT symbol, name, shares, cost_price, current_price FROM positions WHERE shares > 0")
            for row in cursor.fetchall():
                positions.append({
                    'symbol': row[0],
                    'name': row[1] or row[0],
                    'shares': int(row[2]),
                    'cost_price': float(row[3]),
                    'current_price': float(row[4]),
                    'profit_pct': (float(row[4]) - float(row[3])) / float(row[3]) * 100 if float(row[3]) > 0 else 0,
                })
            conn.close()
        except Exception as e:
            logger.warning(f"获取持仓失败: {e}")

        # 获取全市场排名
        rankings = predictor.predict_all()
        if not rankings:
            logger.info("无有效预测")
            return

        # 获取买入候选
        held_symbols = [p['symbol'] for p in positions]
        buy_candidates = predictor.get_buy_candidates(existing_positions=held_symbols)

        # 获取持仓建议
        position_advice = predictor.get_position_advice(positions) if positions else []

        # 构建飞书消息
        spearman = predictor.model_data.get('test_ic') if predictor.model_data else None
        thresholds = predictor._regime_info if hasattr(predictor, '_regime_info') else {}
        regime = thresholds.get('regime', '?')

        regime_emoji = {'bull': '🐂', 'bear': '🐻', 'sideways': '📊'}.get(regime, '📊')
        regime_cn = {'bull': '牛市', 'bear': '熊市', 'sideways': '震荡'}.get(regime, '?')

        lines = [f"**📊 v9 Ensemble — 盘中预测 (window=3)**\n"]
        lines.append(f"{regime_emoji} 大盘: **{regime_cn}** | "
                    f"趋势强度: {thresholds.get('trend_strength', 0):.2%} | "
                    f"阈值×{thresholds.get('adjustment', 1.0):.1f}\n")

        # 买入候选
        if buy_candidates:
            lines.append("**🔥 买入候选**")
            for r in buy_candidates[:TOP_N_CANDIDATES]:
                lines.append(f"  {r['rank']}. **{r['name']}** — 预期收益 {r['predicted_return']:.2%}")
            lines.append("")

        # 持仓建议
        if position_advice:
            lines.append("**💼 持仓操作建议**")
            for p in position_advice[:10]:
                profit = p.get('profit_pct', 0)
                profit_str = f" (累计{profit:+.1f}%)" if profit else ""
                lines.append(f"  {p['name']}: {p['signal_text']}{profit_str}")
            lines.append("")

        # 信号统计
        strong_buy = sum(1 for r in rankings if r['signal'] == 'strong_buy')
        buy = sum(1 for r in rankings if r['signal'] == 'buy')
        sell = sum(1 for r in rankings if r['signal'] in ('sell', 'strong_sell'))
        lines.append(f"📈 信号: 🔥{strong_buy}只看涨 📈{buy}只关注 📉{sell}只看跌")
        if spearman:
            lines.append(f"*Spearman: {spearman:.4f}*")

        from card_templates import make_text_card
        card = make_text_card(''.join(lines))
        _push_card(card)
        logger.info(f"v8预测推送完成: {len(buy_candidates)}候选, {len(position_advice)}持仓")

    except Exception as e:
        logger.error(f"v8预测推送失败: {e}")


def qlib_intraday_push():
    """qlib 30min模型预测推送 — 盘前9:25（用昨天收盘数据预测今天）"""
    if not is_trading_day():
        return
    logger.info("qlib 盘前预测推送触发（用昨日数据）")

    try:
        from qlib_light_predictor import predict_top_stocks, format_feishu_card
        result = predict_top_stocks(10, use_yesterday=True)
        if not result or not result.get('signals'):
            logger.info("qlib 无有效预测")
            return

        text = format_feishu_card(result)
        from card_templates import make_text_card
        card = make_text_card(text)
        _push_card(card)
        logger.info(f"qlib盘前预测推送完成: {len(result['signals'])}只")

    except Exception as e:
        logger.error(f"qlib预测推送失败: {e}")


def save_and_rebuild_qlib_data():
    """收盘后保存全部372只股票的30min数据，并重建qlib bin"""
    if not is_trading_day():
        return
    logger.info("📊 收盘后30min数据保存 + qlib重建...")
    try:
        import subprocess
        result = subprocess.run(
            ['/root/miniconda3/bin/python3', '/root/github/stock-quant/scripts/update_qlib_data.py'],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            # 提取关键信息
            for line in result.stdout.strip().split('\n'):
                if '✅' in line:
                    logger.info(line.strip())
        else:
            logger.error(f"30min数据更新失败: {result.stderr[-200:]}")
    except Exception as e:
        logger.error(f"30min数据更新异常: {e}")


def lgbm_signal_push():
    """LGBM 模型信号推送 — 盘前推送 + 盘中更新"""
    if not is_trading_day():
        return
    logger.info("LGBM 模型信号推送触发")

    try:
        signals_data = get_signals_data()
        signals = signals_data.get('signals', [])
        if not signals:
            logger.info("LGBM: 无有效信号")
            return

        buy_count = sum(1 for s in signals if '买入' in s.get('signal', '') or s.get('signal') == 'buy')
        sell_count = sum(1 for s in signals if '卖出' in s.get('signal', '') or s.get('signal') == 'sell')

        card = make_signal_card(signals)
        _push_card(card)
        logger.info(f"LGBM信号推送完成: {len(signals)}只 (买入{buy_count}/卖出{sell_count})")
    except Exception as e:
        logger.error(f"LGBM信号推送失败: {e}")


def setup_scheduler():
    """配置定时任务"""
    # 盘前提醒 9:25 —— 集合竞价阶段，提供今日关注点
    scheduler.add_job(
        morning_alert,
        CronTrigger(hour=9, minute=25, day_of_week='mon-fri'),
        id='morning_alert',
        name='盘前提醒(集合竞价)',
        misfire_grace_time=60
    )

    # 盘中异动轮询（每5分钟，交易时段）- 内部按分时段频率控制
    scheduler.add_job(
        intraday_alert_monitor,
        CronTrigger(hour='9-11,13-14', minute='*/5', day_of_week='mon-fri'),
        id='intraday_alert_monitor',
        name='盘中异动轮询',
        misfire_grace_time=120
    )

    # 盘中开盘推送 —— 9:30上午开盘 + 13:00下午开盘
    scheduler.add_job(
        intraday_check,
        CronTrigger(hour='9', minute='30', day_of_week='mon-fri'),
        id='intraday_check_0930',
        name='上午开盘监控(9:30)',
        misfire_grace_time=120
    )
    scheduler.add_job(
        intraday_check,
        CronTrigger(hour='13', minute='0', day_of_week='mon-fri'),
        id='intraday_check_1300',
        name='下午开盘监控(13:00)',
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

    # 晚间复盘 18:00
    scheduler.add_job(
        evening_review,
        CronTrigger(hour=18, minute=0, day_of_week='mon-fri'),
        id='evening_review',
        name='晚间复盘',
        misfire_grace_time=300
    )

    # v8 模型预测推送 — 精简为 10:00 / 14:30 / 15:00
    scheduler.add_job(
        v8_intraday_push,
        CronTrigger(hour=10, minute=0, day_of_week='mon-fri'),
        id='v8_intraday_10',
        name='v8模型预测(10:00)',
        misfire_grace_time=120
    )
    scheduler.add_job(
        v8_intraday_push,
        CronTrigger(hour=14, minute=30, day_of_week='mon-fri'),
        id='v8_intraday_1430',
        name='v8模型预测(14:30)',
        misfire_grace_time=120
    )
    scheduler.add_job(
        v8_intraday_push,
        CronTrigger(hour=15, minute=0, day_of_week='mon-fri'),
        id='v8_intraday_close',
        name='v8模型预测(收盘)',
        misfire_grace_time=120
    )

    # qlib 30min模型预测推送 — 盘前9:25（用昨天数据预测今天）
    scheduler.add_job(
        qlib_intraday_push,
        CronTrigger(hour=9, minute=25, day_of_week='mon-fri'),
        id='qlib_intraday_0925',
        name='qlib盘前预测(9:25)',
        misfire_grace_time=300
    )

    # LGBM 模型信号推送 — 盘前9:25 + 盘中14:00
    scheduler.add_job(
        lgbm_signal_push,
        CronTrigger(hour=9, minute=25, day_of_week='mon-fri'),
        id='lgbm_signal_0925',
        name='LGBM信号推送(盘前)',
        misfire_grace_time=120
    )
    scheduler.add_job(
        lgbm_signal_push,
        CronTrigger(hour=14, minute=0, day_of_week='mon-fri'),
        id='lgbm_signal_1400',
        name='LGBM信号推送(盘中)',
        misfire_grace_time=120
    )

    # 收盘后保存全部30min数据 + 重建qlib bin（15:10）
    scheduler.add_job(
        save_and_rebuild_qlib_data,
        CronTrigger(hour=15, minute=10, day_of_week='mon-fri'),
        id='save_qlib_data_1510',
        name='30min数据保存+qlib重建(15:10)',
        misfire_grace_time=600
    )

    logger.info("定时任务已配置: 盘前9:25(含qlib+LGBM预测), 开盘9:30/13:00, 盘后15:05, "
               "晚间18:00, 异动轮询(分时段频率), v8预测(10:00/14:30/15:00), "
               "LGBM信号(9:25/14:00), 30min数据保存(15:10)")


def start_scheduler():
    """启动调度器"""
    setup_scheduler()
    scheduler.start()
    logger.info("✓ APScheduler 已启动")


def stop_scheduler():
    """停止调度器"""
    scheduler.shutdown(wait=False)
    logger.info("APScheduler 已停止")