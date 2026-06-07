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
            {"tag": "column_set", "flex_mode": "bisect", "background_style": "default",
             "columns": [
                 {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**当前价格**\n<font color='{color}'>¥{price:.2f}</font>"}]},
                 {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**涨跌幅**\n<font color='{color}'>{sign}{change_pct:.2f}%</font>\n{amount_sign}¥{abs(change_amount):.2f}"}]},
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
        amt = idx.get('change_amount', 0)
        sign = "+" if pct > 0 else ""
        # A股红涨绿跌, 港股绿涨红跌
        if code.endswith('.SH') or code.endswith('.SZ'):
            color = "red" if pct > 0 else "green" if pct < 0 else "default"
        else:
            color = "green" if pct > 0 else "red" if pct < 0 else "default"
        price_s = f"{price:,.2f}" if price >= 10000 else f"{price:.2f}"
        elements.append({
            "tag": "column_set", "flex_mode": "bisect", "background_style": "grey",
            "columns": [
                {"tag": "column", "width": "weighted", "weight": 2, "elements": [{"tag": "markdown", "content": f"**{name}**"}]},
                {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": price_s}]},
                {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"<font color='{color}'>{sign}{pct:.2f}%</font>"}]},
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
            "tag": "column_set", "flex_mode": "bisect", "background_style": "grey",
            "columns": [
                {"tag": "column", "width": "weighted", "weight": 2, "elements": [{"tag": "markdown", "content": f"**{name}**"}]},
                {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"<font color='{color}'>{sign}{pct:.2f}%</font>"}]},
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
            "tag": "column_set", "flex_mode": "bisect", "background_style": "grey",
            "columns": [
                {"tag": "column", "width": "weighted", "weight": 2, "elements": [{"tag": "markdown", "content": f"**{name}** ({symbol})"}]},
                {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"¥{price:.2f}"}]},
                {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"<font color='{color}'>{sign}{pct:.2f}%</font>"}]},
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
        "header": {"title": {"tag": "plain_text", "content": "📘 金融小助手使用指南"}, "template": "blue"},
        "elements": [
            {"tag": "markdown", "content": "**🔍 查询类**\n- `持仓` → 查看持仓和做T建议\n- `行情 茅台` → 查看个股行情\n- `信号` → 交易信号\n- `做T` → 做T建议\n- `大盘` / `指数` → 大盘指数\n- `板块` → 热门板块\n\n**📊 分析类**\n- `回测 茅台` → LGBM回测\n- `分析 茅台` → 综合分析\n- `总结` → 盘后总结\n- `对比 茅台 五粮液` → 多股对比\n\n**⚙️ 配置类**\n- `自选 阿里` → 添加自选股\n- `帮助` → 显示此指南\n\n**💬 闲聊**\n- 直接发任何问题，AI回答"},
            {"tag": "note", "elements": [{"tag": "plain_text", "content": "LGBM + 百炼 | ⚠️ 不构成投资建议"}]}
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