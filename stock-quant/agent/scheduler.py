#!/usr/bin/env python3
"""
定时调度器 - APScheduler + 飞书推送

功能：
1. 盘前提醒（9:25）
2. 盘中监控（每30分钟）
3. 盘后总结（15:05）
4. 异动告警（触发式）

注意：不替代现有的 crontab 邮件监控，两者并行运行
"""

import os
import sys
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

# 飞书推送目标（需要配置：可以推给个人或群聊）
# Phase 1: 推给某个 chat_id（飞书群或个人对话）
FEISHU_TARGET_CHAT_ID = os.environ.get("FEISHU_TARGET_CHAT_ID", "")


def morning_alert():
    """盘前提醒 9:25"""
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

    # 构建消息
    lines = ["**☀️ 盘前提醒**\n"]
    lines.append(f"日期: {datetime.now().strftime('%Y-%m-%d')}\n\n")
    lines.append("**自选股行情:**\n")
    for wp in watchlist_prices:
        sign = "+" if wp['change_pct'] >= 0 else ""
        color = "green" if wp['change_pct'] >= 0 else "red"
        lines.append(f"- <font color='{color}'>{wp['name']} ¥{wp['price']:.2f} ({sign}{wp['change_pct']:.2f}%)</font>\n")

    card = make_text_card("".join(lines))
    _push_card(card)


def intraday_check():
    """盘中智能监控（每30分钟）"""
    logger.info("盘中智能监控触发")

    # 智能异动检测（替代固定3%阈值）
    try:
        from technical_indicators import get_smart_alerts, get_technical_analysis
        from card_templates import make_technical_card
        
        alerts = get_smart_alerts()
        
        if alerts:
            for a in alerts[:8]:
                if a['type'] in ('大涨', '大跌', '放量大涨', '放量大跌', '缩量大涨', '缩量大跌', '接近支撑位', '接近压力位'):
                    card = make_alert_card(a['type'], a['symbol'], a['name'], a['details'])
                    _push_card(card)
                elif a['type'] == '技术信号':
                    ta = get_technical_analysis(a['symbol'])
                    if 'error' not in ta:
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

    # 盘中监控 每30分钟
    scheduler.add_job(
        intraday_check,
        CronTrigger(minute='*/30', hour='9-14', day_of_week='mon-fri'),
        id='intraday_check',
        name='盘中监控',
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

    logger.info("定时任务已配置: 盘前9:25, 盘中每30分钟, 盘后15:05")


def start_scheduler():
    """启动调度器"""
    setup_scheduler()
    scheduler.start()
    logger.info("✓ APScheduler 已启动")


def stop_scheduler():
    """停止调度器"""
    scheduler.shutdown(wait=False)
    logger.info("APScheduler 已停止")