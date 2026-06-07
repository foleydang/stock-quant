#!/usr/bin/env python3
"""
飞书消息卡片模板

所有卡片定义集中在这里，供 bot_server 和 scheduler 使用。
"""

from datetime import datetime


def make_position_card(summary, positions, t_suggestions=None):
    """持仓概览"""
    total_value = summary.get('total_value', 0)
    total_profit = summary.get('total_profit', 0)
    profit_pct = summary.get('profit_pct', 0)
    available_cash = summary.get('available_cash', 0)
    profit_color = "green" if total_profit < 0 else "red"
    profit_sign = "+" if total_profit >= 0 else ""

    pos_rows = ""
    for p in positions[:10]:
        p_pct = p.get('profit_pct', 0)
        p_sign = "+" if p_pct >= 0 else ""
        emoji = "🔴" if p_pct > 0 else "🟢" if p_pct < 0 else "⚪"
        pos_rows += f"| {emoji} {p['stock_name']} | {p['shares']}股 | ¥{p['cost_price']:.2f} | ¥{p['current_price']:.2f} | {p_sign}{p_pct:.1f}% |\n"

    t_section = ""
    if t_suggestions:
        t_section = "\n---\n**💡 做T建议**\n"
        for t in t_suggestions[:5]:
            emoji_map = {"适合做T": "🟢", "可减仓": "🔵", "观望": "⚠️", "不建议": "❌"}
            emoji = emoji_map.get(t.get('action', ''), '⚪')
            t_section += f"- {emoji} **{t['stock_name']}** {t.get('action', '')}: {t.get('reason', '')}\n"

    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "📊 持仓概览"}, "template": "blue" if total_profit >= 0 else "red"},
        "elements": [
            {"tag": "column_set", "flex_mode": "bisect", "background_style": "default",
             "columns": [
                 {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**总市值**\n¥{total_value:,.0f}"}]},
                 {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**浮动盈亏**\n<font color='{profit_color}'>{profit_sign}¥{total_profit:,.0f} ({profit_sign}{profit_pct:.1f}%)</font>"}]},
                 {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**可用现金**\n¥{available_cash:,.0f}"}]},
             ]},
            {"tag": "hr"},
            {"tag": "markdown", "content": f"| 股票 | 持仓 | 成本 | 现价 | 盈亏 |\n|---|---|---|---|---|\n{pos_rows}"},
            {"tag": "markdown", "content": t_section},
            {"tag": "note", "elements": [{"tag": "plain_text", "content": "⚠️ 仅供参考，不构成投资建议"}]}
        ]
    }


def make_stock_card(data):
    """个股行情"""
    name = data.get('name', data.get('stock_name', data.get('symbol', '')))
    symbol = data.get('symbol', '')
    price = data.get('current_price', data.get('price', 0))
    change_pct = data.get('change_pct', 0)
    change_amount = data.get('change_amount', data.get('change', 0))

    # A股红涨绿跌, 港股绿涨红跌
    if symbol and symbol.endswith('.HK'):
        color = "green" if change_pct > 0 else "red" if change_pct < 0 else "default"
    else:
        color = "red" if change_pct > 0 else "green" if change_pct < 0 else "default"
    sign = "+" if change_pct > 0 else ""
    amount_sign = "+" if change_amount > 0 else ""

    detail = ""
    high = data.get('high', 0)
    low = data.get('low', 0)
    open_p = data.get('open', data.get('open_price', 0))
    volume = data.get('volume', 0)
    if high and low:
        detail = f"最高 ¥{high:.2f} | 最低 ¥{low:.2f} | 开盘 ¥{open_p:.2f}"
        if volume:
            v = f"{volume/10000:.1f}万手" if volume > 10000 else f"{volume}手"
            detail += f" | 成交 {v}"

    indicators = data.get('indicators', {})
    ind_section = ""
    if indicators:
        items = [f"**{k}**: {v}" for k, v in indicators.items() if v and v != 'N/A']
        if items:
            ind_section = "\n---\n**技术指标** " + " | ".join(items[:6])

    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": f"📈 {name} ({symbol})"}, "template": color},
        "elements": [
            {"tag": "column_set", "flex_mode": "none", "background_style": "default",
             "columns": [
                 {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**当前价格**\n<font color='{color}'>¥{price:.2f}</font>"}]},
                 {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**涨跌幅**\n<font color='{color}'>{sign}{change_pct:.2f}%</font> ({amount_sign}¥{abs(change_amount):.2f})"}]},
             ]},
            {"tag": "markdown", "content": detail},
            {"tag": "markdown", "content": ind_section},
            {"tag": "note", "elements": [{"tag": "plain_text", "content": f"更新: {datetime.now().strftime('%H:%M')} | Tushare"}]}
        ]
    }


def make_market_card(market_data):
    """大盘指数 - column_set 布局"""
    indices = market_data.get('indices', [])
    if not indices:
        return make_text_card("暂无大盘数据")

    elements = []
    for idx in indices:
        name, code, price = idx['name'], idx['code'], idx.get('price', 0)
        pct = idx.get('change_pct', 0)
        sign = "+" if pct > 0 else ""
        if code.endswith('.SH') or code.endswith('.SZ'):
            color = "red" if pct > 0 else "green" if pct < 0 else "default"
        else:
            color = "green" if pct > 0 else "red" if pct < 0 else "default"
        price_s = f"{price:,.2f}" if price >= 10000 else f"{price:.2f}"
        elements.append({
            "tag": "column_set", "flex_mode": "none", "background_style": "grey",
            "columns": [
                {"tag": "column", "width": "weighted", "weight": 3, "elements": [{"tag": "markdown", "content": f"**{name}**"}]},
                {"tag": "column", "width": "weighted", "weight": 2, "elements": [{"tag": "markdown", "content": price_s}]},
                {"tag": "column", "width": "weighted", "weight": 2, "elements": [{"tag": "markdown", "content": f"<font color='{color}'>{sign}{pct:.2f}%</font>"}]},
            ]
        })

    sentiment = market_data.get('sentiment', '')
    if sentiment:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": f"**市场情绪**: {sentiment}"})
    elements.append({"tag": "note", "elements": [{"tag": "plain_text", "content": f"更新: {datetime.now().strftime('%H:%M')} | 腾讯财经"}]})

    return {"config": {"wide_screen_mode": True}, "header": {"title": {"tag": "plain_text", "content": "🌍 大盘指数"}, "template": "blue"}, "elements": elements}


def make_sector_card(sector_data):
    """热门板块 - column_set 布局"""
    sectors = sector_data.get('sectors', [])
    if not sectors:
        return make_text_card("暂无板块数据")

    elements = []
    for s in sectors[:10]:
        name, pct = s['name'], s.get('change_pct', 0)
        sign = "+" if pct > 0 else ""
        color = "red" if pct > 0 else "green" if pct < 0 else "default"
        elements.append({
            "tag": "column_set", "flex_mode": "none", "background_style": "grey",
            "columns": [
                {"tag": "column", "width": "weighted", "weight": 5, "elements": [{"tag": "markdown", "content": f"**{name}**"}]},
                {"tag": "column", "width": "weighted", "weight": 2, "elements": [{"tag": "markdown", "content": f"<font color='{color}'>{sign}{pct:.2f}%</font>"}]},
            ]
        })
    elements.append({"tag": "note", "elements": [{"tag": "plain_text", "content": f"更新: {datetime.now().strftime('%H:%M')} | 腾讯财经"}]})

    return {"config": {"wide_screen_mode": True}, "header": {"title": {"tag": "plain_text", "content": "🔥 今日热门板块"}, "template": "orange"}, "elements": elements}


def make_compare_card(compare_data):
    """多股对比 - column_set 布局"""
    stocks = compare_data.get('stocks', [])
    if not stocks:
        return make_text_card("暂无可对比的数据")

    elements = []
    for s in stocks:
        name, symbol, price, pct = s.get('name', ''), s.get('symbol', ''), s.get('current_price', 0), s.get('change_pct', 0)
        sign = "+" if pct > 0 else ""
        if symbol.endswith('.HK'):
            color = "green" if pct > 0 else "red" if pct < 0 else "default"
        else:
            color = "red" if pct > 0 else "green" if pct < 0 else "default"
        elements.append({
            "tag": "column_set", "flex_mode": "none", "background_style": "grey",
            "columns": [
                {"tag": "column", "width": "weighted", "weight": 4, "elements": [{"tag": "markdown", "content": f"**{name}** ({symbol})"}]},
                {"tag": "column", "width": "weighted", "weight": 2, "elements": [{"tag": "markdown", "content": f"¥{price:.2f}"}]},
                {"tag": "column", "width": "weighted", "weight": 2, "elements": [{"tag": "markdown", "content": f"<font color='{color}'>{sign}{pct:.2f}%</font>"}]},
            ]
        })

    best = max(stocks, key=lambda s: s.get('change_pct', 0))
    worst = min(stocks, key=lambda s: s.get('change_pct', 0))
    b_pct, w_pct = best.get('change_pct', 0), worst.get('change_pct', 0)
    b_sign, w_sign = "+" if b_pct > 0 else "", "+" if w_pct > 0 else ""
    elements.append({"tag": "hr"})
    elements.append({"tag": "markdown", "content": f"🏆 **最强**: {best.get('name', '')} {b_sign}{b_pct:.2f}%\n💀 **最弱**: {worst.get('name', '')} {w_sign}{w_pct:.2f}%"})
    elements.append({"tag": "note", "elements": [{"tag": "plain_text", "content": f"更新: {datetime.now().strftime('%H:%M')} | ⚠️ 不构成投资建议"}]})

    return {"config": {"wide_screen_mode": True}, "header": {"title": {"tag": "plain_text", "content": f"📊 股票对比 ({len(stocks)}只)"}, "template": "blue"}, "elements": elements}


def make_signal_card(signals):
    """交易信号"""
    if not signals:
        return make_text_card("当前没有新的交易信号")
    rows = ""
    for s in signals[:10]:
        sig = s.get('signal', '持有')
        emoji = {"买入": "🟢", "卖出": "🔴", "buy": "🟢", "sell": "🔴"}.get(sig, "⚪")
        prob = f"{s.get('up_prob', 0):.0%}" if s.get('up_prob') else "-"
        rows += f"| {emoji} {s.get('stock_name', '')} | ¥{s.get('current_price', 0):.2f} | {sig} | {prob} | {s.get('reason', '')[:30]} |\n"
    buy_count = sum(1 for s in signals if '买入' in s.get('signal', '') or s.get('signal') == 'buy')
    sell_count = sum(1 for s in signals if '卖出' in s.get('signal', '') or s.get('signal') == 'sell')
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "🔔 交易信号"}, "template": "blue"},
        "elements": [
            {"tag": "column_set", "flex_mode": "bisect", "background_style": "default",
             "columns": [
                 {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**买入**: {buy_count}只"}]},
                 {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**卖出**: {sell_count}只"}]},
             ]},
            {"tag": "hr"},
            {"tag": "markdown", "content": f"| 股票 | 现价 | 信号 | 概率 | 原因 |\n|---|---|---|---|---|\n{rows}"},
            {"tag": "note", "elements": [{"tag": "plain_text", "content": "⚠️ 仅供参考"}]}
        ]
    }


def make_help_card():
    """帮助"""
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "📘 金融小助手"}, "template": "blue"},
        "elements": [
            {"tag": "markdown", "content": "**🔍 查询** 行情/大盘/板块/对比/北向\n\n**📊 分析** 指标/异动/深度/资金/估值\n\n**🔄 策略** 做T(具体买卖价)/建议(综合操作)/风控(评分)\n\n**📰 信息** 新闻/总结\n\n**⚙️ 配置** 自选(添加)/止损\n\n**💬 闲聊** 直接发问题，AI回答"},
            {"tag": "note", "elements": [{"tag": "plain_text", "content": "波动率+支撑压力+量比+风控评分 | ⚠️ 不构成投资建议"}]}
        ]
    }


def make_chat_card(text):
    """AI对话"""
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "💬 金融小助手"}, "template": "turquoise"},
        "elements": [
            {"tag": "markdown", "content": text},
            {"tag": "note", "elements": [{"tag": "plain_text", "content": "AI生成 | 试试「帮助」查看更多功能"}]}
        ]
    }


def make_text_card(text):
    """简单文本（错误提示等）"""
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "金融小助手"}, "template": "default"},
        "elements": [{"tag": "markdown", "content": text}]
    }


def make_alert_card(alert_type, symbol, name, details):
    """异动告警"""
    template = {"大涨": "green", "大跌": "red", "放量": "orange", "异动": "violet"}.get(alert_type, "red")
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": f"🚨 {alert_type} - {name}"}, "template": template},
        "elements": [
            {"tag": "markdown", "content": f"**{name}** ({symbol})\n{details}"},
            {"tag": "note", "elements": [{"tag": "plain_text", "content": "实时监控告警"}]}
        ]
    }


def make_backtest_card(data):
    """回测结果"""
    summary = data.get('summary', {})
    total_return = summary.get('total_return', 0)
    win_rate = summary.get('win_rate', 0)
    total_trades = summary.get('total_trades', 0)
    name = data.get('name', data.get('symbol', ''))
    ret_color = "green" if total_return > 0 else "red"
    ret_sign = "+" if total_return > 0 else ""
    trades = data.get('trades', [])
    rows = ""
    for t in trades[:5]:
        t_type = t.get('type', '')
        emoji = "🟢" if t_type == "buy" else "🔴" if t_type == "sell" else "⚪"
        rows += f"| {emoji} {t_type} | ¥{t.get('price', 0):.2f} | {t.get('shares', 0)}股 | {t.get('time', '')[:10]} |\n"
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": f"📊 回测 - {name}"}, "template": ret_color},
        "elements": [
            {"tag": "column_set", "flex_mode": "bisect", "background_style": "default",
             "columns": [
                 {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**总收益**\n<font color='{ret_color}'>{ret_sign}{total_return:.2f}%</font>"}]},
                 {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**胜率**\n{win_rate:.1f}%"}]},
                 {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**交易数**\n{total_trades}笔"}]},
             ]},
            {"tag": "hr"},
            {"tag": "markdown", "content": f"| 操作 | 价格 | 数量 | 时间 |\n|---|---|---|---|\n{rows}"},
            {"tag": "note", "elements": [{"tag": "plain_text", "content": "LGBM | ⚠️ 回测不代表未来"}]}
        ]
    }


def make_daily_summary_card(summary, positions, signals, t_suggestions=None):
    """盘后总结"""
    date = summary.get('date', '')
    total_value = summary.get('total_value', 0)
    total_profit = summary.get('total_profit', 0)
    profit_pct = summary.get('profit_pct', 0)
    color = "green" if total_profit < 0 else "red"
    sign = "+" if total_profit >= 0 else ""
    highlight = ""
    if positions:
        best = max(positions, key=lambda p: p.get('profit_pct', 0))
        worst = min(positions, key=lambda p: p.get('profit_pct', 0))
        highlight = f"🏆 **最佳**: {best['stock_name']} +{best['profit_pct']:.1f}%\n💀 **最差**: {worst['stock_name']} {worst['profit_pct']:.1f}%"
    suggestions = ""
    for s in signals:
        if s.get('action') != '持有':
            suggestions += f"- **{s['stock_name']}**: {s['action']} - {s['reason']}\n"
    if not suggestions:
        suggestions = "- 今日无特别操作建议"
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": f"📝 盘后总结 - {date}"}, "template": "blue"},
        "elements": [
            {"tag": "column_set", "flex_mode": "bisect", "background_style": "default",
             "columns": [
                 {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**总市值**\n¥{total_value:,.0f}"}]},
                 {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**今日盈亏**\n<font color='{color}'>{sign}¥{total_profit:,.0f} ({sign}{profit_pct:.1f}%)</font>"}]},
             ]},
            {"tag": "hr"},
            {"tag": "markdown", "content": highlight},
            {"tag": "hr"},
            {"tag": "markdown", "content": f"**操作建议**\n{suggestions}"},
            {"tag": "note", "elements": [{"tag": "plain_text", "content": "⚠️ 仅供参考"}]}
        ]
    }


def make_technical_card(data: dict) -> dict:
    """技术分析卡片"""
    name = data.get('name', data.get('symbol', ''))
    symbol = data.get('symbol', '')
    current = data.get('current', 0)
    change_pct = data.get('change_pct', 0)
    color = "red" if change_pct > 0 else "green" if change_pct < 0 else "default"
    sign = "+" if change_pct > 0 else ""

    # 指标概览
    indicators = []
    ma5, ma10, ma20, ma60 = data.get('ma5'), data.get('ma10'), data.get('ma20'), data.get('ma60')
    if ma5: indicators.append(f"MA5 {ma5:.2f}")
    if ma10: indicators.append(f"MA10 {ma10:.2f}")
    if ma20: indicators.append(f"MA20 {ma20:.2f}")
    if ma60: indicators.append(f"MA60 {ma60:.2f}")

    rsi = data.get('rsi')
    if rsi:
        rsi_color = "red" if rsi > 70 else "green" if rsi < 30 else "default"
        indicators.append(f"RSI <font color='{rsi_color}'>{rsi:.1f}</font>")

    kdj_j = data.get('kdj_j')
    if kdj_j:
        j_color = "red" if kdj_j > 100 else "green" if kdj_j < 0 else "default"
        indicators.append(f"KDJ-J <font color='{j_color}'>{kdj_j:.1f}</font>")

    vol_ratio = data.get('volume_ratio')
    if vol_ratio:
        vol_color = "red" if vol_ratio > 2 else "default"
        indicators.append(f"量比 <font color='{vol_color}'>{vol_ratio:.1f}</font>")

    macd_dif = data.get('macd_dif')
    macd_dea = data.get('macd_dea')
    if macd_dif and macd_dea:
        macd_color = "red" if macd_dif > macd_dea else "green"
        indicators.append(f"MACD <font color='{macd_color}'>DIF {macd_dif:.2f}</font>")

    boll_upper, boll_lower = data.get('boll_upper'), data.get('boll_lower')
    if boll_upper and boll_lower:
        indicators.append(f"布林 上{boll_upper:.2f} 下{boll_lower:.2f}")

    threshold = data.get('dynamic_threshold')
    if threshold:
        indicators.append(f"异动阈值 {threshold:.1f}%")

    ind_line = " | ".join(indicators)

    # 信号列表
    signals = data.get('signals', [])
    signal_line = ""
    if signals:
        signal_line = "\n---\n**🔔 技术信号**\n"
        for s in signals:
            emoji_map = {"金叉": "🟢", "死叉": "🔴", "超买": "⚠️", "超卖": "💡", "多头": "📈", "空头": "📉", "新高": "🔥", "新低": "❄️", "放量": "⚡", "缩量": "🔇", "支撑": "🛡️", "压力": "🚧", "突破": "🎯", "跌破": "💥"}
            emoji = "🔔"
            for kw, e in emoji_map.items():
                if kw in s:
                    emoji = e
                    break
            signal_line += f"- {emoji} {s}\n"
    else:
        signal_line = "\n---\n*当前无重要技术信号*"

    # 支撑压力位
    sr_line = ""
    supports = data.get('supports', [])
    resistances = data.get('resistances', [])
    if supports or resistances:
        sr_line = "\n**关键价位**\n"
        if supports:
            sr_line += f"🛡️ 支撑: " + " | ".join([f"¥{s:.2f}" for s in supports]) + "\n"
        if resistances:
            sr_line += f"🚧 压力: " + " | ".join([f"¥{r:.2f}" for r in resistances]) + "\n"

    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": f"📊 {name} ({symbol}) 技术分析"}, "template": "blue"},
        "elements": [
            {"tag": "markdown", "content": f"**当前** <font color='{color}'>¥{current:.2f} ({sign}{change_pct:.2f}%)</font>"},
            {"tag": "markdown", "content": ind_line},
            {"tag": "markdown", "content": signal_line},
            {"tag": "markdown", "content": sr_line},
            {"tag": "note", "elements": [{"tag": "plain_text", "content": "⚠️ 仅供参考，不构成投资建议"}]}
        ]
    }


def make_alert_card_v2(alerts: list) -> dict:
    """智能异动卡片（多条告警合并）"""
    if not alerts:
        return make_text_card("当前无异动")

    # 按类型分组
    type_emoji = {'大涨': '🔴', '大跌': '🟢', '放量大涨': '🔴⚡', '放量大跌': '🟢⚡', '缩量大涨': '🔴🔇', '缩量大跌': '🟢🔇', '技术信号': '🔔', '接近支撑位': '🛡️', '接近压力位': '🚧'}

    lines = ""
    for a in alerts[:15]:
        emoji = type_emoji.get(a['type'], '⚠️')
        pct = a.get('change_pct', 0)
        sign = "+" if pct > 0 else ""
        lines += f"- {emoji} **{a['name']}** {a['type']} → {a['details']}\n"

    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": f"🚨 智能异动 ({len(alerts)}条)"}, "template": "red"},
        "elements": [
            {"tag": "markdown", "content": lines},
            {"tag": "note", "elements": [{"tag": "plain_text", "content": "动态阈值+量价联合+技术指标 | ⚠️ 仅供参考"}]}
        ]
    }

def make_money_flow_card(data: dict) -> dict:
    """资金流向卡片"""
    if 'error' in data:
        return make_text_card(data['error'])
    
    name = data.get('name', '')
    net_mf = data.get('net_mf_amount', 0)  # 万元
    lg_net = data.get('lg_net', 0)
    sm_net = data.get('sm_net', 0)
    date = data.get('date', '')
    
    # 判断主力方向
    direction = "净流入" if net_mf > 0 else "净流出"
    color = "red" if net_mf > 0 else "green"  # A股红涨绿跌
    abs_net = abs(net_mf)
    sign = "+" if net_mf > 0 else ""
    
    # 大单小单
    lg_dir = "流入" if lg_net > 0 else "流出"
    sm_dir = "流入" if sm_net > 0 else "流出"
    lg_sign = "+" if lg_net > 0 else ""
    sm_sign = "+" if sm_net > 0 else ""
    
    # 近5天趋势
    trend = data.get('trend', [])
    trend_line = ""
    if trend:
        trend_line = "\n---\n**近5日资金趋势**\n"
        for t in trend:
            v = t.get('net_mf', 0)
            s = "+" if v > 0 else ""
            d = t.get('date', '')
            trend_line += f"- {d}: {s}{v:.0f}万\n"
    
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": f"💰 {name} 资金流向"}, "template": color},
        "elements": [
            {"tag": "markdown", "content": f"**主力资金** <font color='{color}'>{sign}{abs_net:.0f}万（{direction}）</font>"},
            {"tag": "column_set", "flex_mode": "none", "background_style": "default",
             "columns": [
                 {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**大单**\n{lg_sign}{abs(lg_net):.0f}万（{lg_dir}）"}]},
                 {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**小单**\n{sm_sign}{abs(sm_net):.0f}万（{sm_dir}）"}]},
             ]},
            {"tag": "markdown", "content": trend_line},
            {"tag": "note", "elements": [{"tag": "plain_text", "content": f"数据日期: {date} | Tushare | ⚠️ 仅供参考"}]}
        ]
    }


def make_deep_data_card(data: dict) -> dict:
    """个股深度数据卡片"""
    if 'error' in data:
        return make_text_card(data['error'])
    
    name = data.get('name', '')
    symbol = data.get('symbol', '')
    current = data.get('current_price', 0)
    change_pct = data.get('change_pct', 0)
    color = "red" if change_pct > 0 else "green" if change_pct < 0 else "default"
    sign = "+" if change_pct > 0 else ""
    
    v = data.get('valuation', {})
    p = data.get('profit', {})
    
    # 估值行
    val_items = []
    pe = v.get('pe_ttm')
    if pe: val_items.append(f"PE {pe:.1f}")
    pb = v.get('pb')
    if pb: val_items.append(f"PB {pb:.2f}")
    mv = v.get('total_mv')
    if mv: val_items.append(f"市值 {mv/10000:.0f}亿")  # 万元→亿
    tr = v.get('turnover_rate')
    if tr: val_items.append(f"换手率 {tr:.2f}%")
    vr = v.get('volume_ratio')
    if vr: val_items.append(f"量比 {vr:.1f}")
    val_line = " | ".join(val_items) if val_items else "暂无估值数据"
    
    # 盈利行
    profit_items = []
    roe = p.get('roe')
    if roe: profit_items.append(f"ROE {roe:.2f}%")
    npm = p.get('netprofit_margin')
    if npm: profit_items.append(f"净利率 {npm:.2f}%")
    gpm = p.get('grossprofit_margin')
    if gpm: profit_items.append(f"毛利率 {gpm:.2f}%")
    or_yoy = p.get('or_yoy')
    if or_yoy: profit_items.append(f"营收增速 {or_yoy:.2f}%")
    np_yoy = p.get('netprofit_yoy')
    if np_yoy: profit_items.append(f"净利增速 {np_yoy:.2f}%")
    profit_line = " | ".join(profit_items) if profit_items else "暂无盈利数据"
    end_date = p.get('end_date', '')
    
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": f"🔍 {name} ({symbol}) 深度数据"}, "template": "blue"},
        "elements": [
            {"tag": "markdown", "content": f"**当前** <font color='{color}'>¥{current:.2f} ({sign}{change_pct:.2f}%)</font>"},
            {"tag": "hr"},
            {"tag": "markdown", "content": f"**估值指标**\n{val_line}"},
            {"tag": "markdown", "content": f"**盈利指标**（{end_date}）\n{profit_line}"},
            {"tag": "note", "elements": [{"tag": "plain_text", "content": "Tushare | ⚠️ 仅供参考"}]}
        ]
    }


def make_compare_deep_card(data: dict) -> dict:
    """增强版对比卡片"""
    if 'error' in data:
        return make_text_card(data['error'])
    
    stocks = data.get('stocks', [])
    count = data.get('count', 0)
    cheapest = data.get('cheapest', '')
    
    elements = []
    for s in stocks:
        name = s.get('name', '')
        symbol = s.get('symbol', '')
        current = s.get('current_price', 0)
        pct = s.get('change_pct', 0)
        sign = "+" if pct > 0 else ""
        color = "red" if pct > 0 else "green" if pct < 0 else "default"
        
        v = s.get('valuation', {})
        p = s.get('profit', {})
        pe = v.get('pe_ttm', 'N/A')
        pb = v.get('pb', 'N/A')
        roe = p.get('roe', 'N/A')
        
        pe_str = f"{pe:.1f}" if isinstance(pe, (int, float)) else pe
        pb_str = f"{pb:.2f}" if isinstance(pb, (int, float)) else pb
        roe_str = f"{roe:.2f}%" if isinstance(roe, (int, float)) else roe
        
        elements.append({
            "tag": "column_set", "flex_mode": "none", "background_style": "grey",
            "columns": [
                {"tag": "column", "width": "weighted", "weight": 3, "elements": [{"tag": "markdown", "content": f"**{name}** ({symbol})"}]},
                {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"¥{current:.2f}"}]},
                {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"<font color='{color}'>{sign}{pct:.2f}%</font>"}]},
                {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"PE {pe_str}"}]},
                {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"ROE {roe_str}"}]},
            ]
        })
    
    if cheapest:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": f"🏆 **估值最低**: {cheapest}"})
    
    elements.append({"tag": "note", "elements": [{"tag": "plain_text", "content": "价格+估值+盈利对比 | ⚠️ 仅供参考"}]})
    
    return {"config": {"wide_screen_mode": True}, "header": {"title": {"tag": "plain_text", "content": f"📊 深度对比 ({count}只)"}, "template": "blue"}, "elements": elements}




def make_t_strategy_card(suggestions: list) -> dict:
    """智能做T策略卡片 - 给出具体买卖价位"""
    if not suggestions:
        return make_text_card("暂无做T建议")
    
    lines = ""
    for s in suggestions[:5]:
        action = s.get('action', '观望')
        action_emoji = {"适合做T": "🔄", "考虑止盈": "🎯", "持有观察": "✅", "观望": "⚪", "减仓": "🔻"}
        emoji = action_emoji.get(action, "⚪")
        
        current = s.get('current_price', 0)
        profit_pct = s.get('profit_pct', 0)
        avg_range = s.get('avg_range', 0)
        vol_ratio = s.get('vol_ratio', 0)
        
        profit_sign = "+" if profit_pct >= 0 else ""
        profit_color = "red" if profit_pct > 0 else "green"
        
        lines += emoji + " **" + s['name'] + "** <font color='" + profit_color + "'>" + profit_sign + f"{profit_pct:.1f}" + "%</font>" + chr(10)
        lines += "  现价¥" + f"{current:.2f}" + " | 日均振幅" + f"{avg_range:.1f}" + "% | 量比" + f"{vol_ratio:.1f}" + chr(10)
        
        if s.get('buy_price') and s.get('sell_price'):
            lines += "  🔄 **低买** ¥" + f"{s['buy_price']:.2f}" + " | **高卖** ¥" + f"{s['sell_price']:.2f}" + chr(10)
            if s.get('t_shares') and s.get('expected_profit'):
                lines += "  💰 做T" + str(int(s['t_shares'])) + "股，预期收益¥" + f"{s['expected_profit']:.0f}" + chr(10)
            if s.get('cost_reduction'):
                lines += "  📉 可降成本" + f"{s['cost_reduction']:.2f}" + "%" + chr(10)
        
        if s.get('reason'):
            lines += "  📝 " + s['reason'] + chr(10)
        
        if s.get('risk_notes'):
            for note in s['risk_notes']:
                lines += "  ⚠️ " + note + chr(10)
        
        if s.get('key_signals'):
            for sig in s['key_signals'][:2]:
                lines += "  🔔 " + sig + chr(10)
        
        lines += chr(10)
    
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "🔄 智能做T策略"}, "template": "blue"},
        "elements": [
            {"tag": "markdown", "content": lines},
            {"tag": "note", "elements": [{"tag": "plain_text", "content": "基于波动率+支撑压力位+量比 | ⚠️ 不构成投资建议"}]}
        ]
    }


def make_risk_card(data: dict) -> dict:
    """风控评分卡片"""
    stocks = data.get('stocks', [])
    avg_score = data.get('avg_score', 0)
    portfolio_risk = data.get('portfolio_risk', '未知')
    
    NL = chr(10)
    lines = "**组合风险**: " + portfolio_risk + " | 平均评分: " + f"{avg_score:.0f}" + "/100" + NL + NL
    
    for s in stocks:
        level = s.get('risk_level', '未知')
        score = s.get('total_score', 0)
        profit_pct = s.get('profit_pct', 0)
        suggestion = s.get('suggestion', '')
        profit_sign = "+" if profit_pct >= 0 else ""
        
        lines += "**" + level + " " + s['name'] + "** | 评分" + str(score) + "/100 | " + profit_sign + f"{profit_pct:.1f}" + "%" + NL
        lines += "  技术" + str(s['tech_score']) + " | 资金" + str(s['fund_score']) + " | 基本面" + str(s['basic_score']) + " → " + suggestion + NL + NL
    
    template = "blue" if avg_score >= 60 else "orange" if avg_score >= 40 else "red"
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "🛡️ 风控评分 | " + portfolio_risk}, "template": template},
        "elements": [
            {"tag": "markdown", "content": lines},
            {"tag": "note", "elements": [{"tag": "plain_text", "content": "技术40+资金30+基本面30=100 | ⚠️ 仅供参考"}]}
        ]
    }


def make_recommend_card(data: dict) -> dict:
    """综合操作建议卡片"""
    recs = data.get('recommendations', [])
    sentiment = data.get('market_sentiment', '未知')
    
    NL = chr(10)
    lines = "**大盘情绪**: " + sentiment + NL + NL
    
    for r in recs:
        action = r.get('action', '持有')
        priority = r.get('priority', '✅')
        profit_pct = r.get('profit_pct', 0)
        profit_sign = "+" if profit_pct >= 0 else ""
        profit_color = "red" if profit_pct > 0 else "green"
        confidence = r.get('confidence', '中')
        
        lines += priority + " **" + r['name'] + "** → " + action + "（置信度" + confidence + "）" + NL
        lines += "  <font color='" + profit_color + "'>" + profit_sign + f"{profit_pct:.1f}" + "%</font> | " + r['reason'] + NL
        
        if r.get('price_target'):
            lines += "  🎯 目标价: ¥" + f"{r['price_target']:.2f}" + NL
        
        if r.get('key_signals'):
            for sig in r['key_signals'][:2]:
                lines += "  🔔 " + sig + NL
        
        lines += NL
    
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "💡 综合操作建议"}, "template": "blue"},
        "elements": [
            {"tag": "markdown", "content": lines},
            {"tag": "note", "elements": [{"tag": "plain_text", "content": "风控+技术+资金综合判断 | ⚠️ 不构成投资建议"}]}
        ]
    }


def make_news_card(data: dict) -> dict:
    """财经要闻卡片"""
    news = data.get('news', [])
    sentiment = data.get('sentiment', {})
    keyword = data.get('keyword', '')
    
    if not news:
        return make_text_card("暂无相关财经新闻")
    
    sentiment_text = sentiment.get('summary', '中性')
    sentiment_score = sentiment.get('score', 0.5)
    sentiment_emoji = "🟢" if sentiment_score > 0.6 else "🔴" if sentiment_score < 0.4 else "🟡"
    
    NL = chr(10)
    lines = "**新闻情绪**: " + sentiment_emoji + " " + sentiment_text + NL + NL
    
    for n in news[:8]:
        title = n.get('title', '')
        lines += "- 📰 **" + title + "**" + NL
        if n.get('snippet'):
            lines += "  " + n['snippet'][:80] + NL
    
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "📰 财经要闻 | " + keyword[:20]}, "template": "blue"},
        "elements": [
            {"tag": "markdown", "content": lines},
            {"tag": "note", "elements": [{"tag": "plain_text", "content": "DuckDuckGo搜索 + LLM情绪分析"}]}
        ]
    }


def make_valuation_card(data: dict) -> dict:
    """估值判断卡片"""
    if 'error' in data:
        return make_text_card(data['error'])
    
    name = data.get('name', '')
    symbol = data.get('symbol', '')
    current = data.get('current_price', 0)
    pe = data.get('pe')
    pb = data.get('pb')
    level = data.get('valuation_level', '未知')
    color_emoji = data.get('valuation_color', '🟡')
    total_mv = data.get('total_mv')
    turnover = data.get('turnover_rate')
    signals = data.get('signals', [])
    
    NL = chr(10)
    
    val_items = []
    if pe: val_items.append("PE " + f"{pe:.1f}")
    if pb: val_items.append("PB " + f"{pb:.2f}")
    if total_mv: val_items.append("市值" + f"{total_mv/10000:.0f}" + "亿")
    if turnover: val_items.append("换手率" + f"{turnover:.2f}" + "%")
    
    val_line = " | ".join(val_items)
    
    signal_line = ""
    if signals:
        signal_line = NL + "**技术信号**" + NL
        for s in signals:
            signal_line += "- 🔔 " + s + NL
    
    val_content = "**当前** ¥" + f"{current:.2f}" + NL + val_line + NL + NL + "**估值判断**: " + color_emoji + " " + level
    
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": color_emoji + " " + name + " 估值: " + level}, "template": "blue"},
        "elements": [
            {"tag": "markdown", "content": val_content},
            {"tag": "markdown", "content": signal_line},
            {"tag": "note", "elements": [{"tag": "plain_text", "content": "PE/PB对比行业 | ⚠️ 仅供参考"}]}
        ]
    }
