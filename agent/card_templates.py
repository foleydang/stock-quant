#!/usr/bin/env python3
"""
飞书消息卡片模板

所有卡片定义集中在这里，供 bot_server 和 scheduler 使用。
"""

from datetime import datetime

def _fmt_price(value, is_etf=False):
    """价格格式化：ETF用3位小数(0.600)，个股用2位(74.52)"""
    if value is None:
        return "-"
    if is_etf or (0 < value < 1):
        return f"{value:.3f}"
    return f"{value:.2f}"


def _fmt_amount(value, is_etf=False):
    """涨跌额格式化：ETF用3位小数(0.003)，个股用2位(0.52)"""
    if value is None:
        return "-"
    if is_etf or (0 < abs(value) < 1):
        return f"{abs(value):.3f}"
    return f"{abs(value):.2f}"



def make_position_card(summary, positions, t_suggestions=None):
    """持仓概览 - column_set + table布局"""
    total_value = summary.get('total_value', 0)
    total_profit = summary.get('total_profit', 0)
    profit_pct = summary.get('profit_pct', 0)
    available_cash = summary.get('available_cash', 0)
    profit_color = "green" if total_profit < 0 else "red"
    profit_sign = "+" if total_profit >= 0 else ""

    # 持仓数据转 column_set 行（类似表格）
    pos_elements = []
    # 表头行
    pos_elements.append({"tag": "column_set", "flex_mode": "none", "background_style": "default",
        "columns": [
            {"tag": "column", "width": "weighted", "weight": 2, "elements": [{"tag": "markdown", "content": "**股票**"}]},
            {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": "**持仓**"}]},
            {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": "**成本**"}]},
            {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": "**现价**"}]},
            {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": "**盈亏**"}]},
        ]})
    for p in positions[:10]:
        p_pct = p.get('profit_pct', 0)
        p_sign = "+" if p_pct >= 0 else ""
        emoji = "🔴" if p_pct > 0 else "🟢" if p_pct < 0 else "⚪"
        profit_color = "red" if p_pct > 0 else "green" if p_pct < 0 else "default"
        pos_elements.append({"tag": "column_set", "flex_mode": "none", "background_style": "grey",
            "columns": [
                {"tag": "column", "width": "weighted", "weight": 2, "elements": [{"tag": "markdown", "content": f"{emoji} {p['stock_name']}"}]},
                {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"{p['shares']}股"}]},
                {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"¥{p['cost_price']:.2f}"}]},
                {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"¥{p['current_price']:.2f}"}]},
                {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"<font color='{profit_color}'>{p_sign}{p_pct:.1f}%</font>"}]},
            ]})

    t_section = ""
    if t_suggestions:
        t_section = "\n**💡 做T建议**\n"
        for t in t_suggestions[:5]:
            emoji_map = {"适合做T": "🟢", "可减仓": "🔵", "观望": "⚠️", "不建议": "❌"}
            emoji = emoji_map.get(t.get('action', ''), '⚪')
            t_section += f"- {emoji} **{t['stock_name']}** {t.get('action', '')}: {t.get('reason', '')}\n"

    elements = [
        {"tag": "column_set", "flex_mode": "bisect", "background_style": "default",
         "columns": [
             {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**总市值**\n¥{total_value:,.0f}"}]},
             {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**浮动盈亏**\n<font color='{profit_color}'>{profit_sign}¥{total_profit:,.0f} ({profit_sign}{profit_pct:.1f}%)</font>"}]},
             {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**可用现金**\n¥{available_cash:,.0f}"}]},
         ]},
        {"tag": "hr"},
    ]
    if pos_elements:
        elements.extend(pos_elements)
    if t_section:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": t_section})
    elements.append({"tag": "note", "elements": [{"tag": "plain_text", "content": "⚠️ 仅供参考，不构成投资建议"}]})

    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "📊 持仓概览"}, "template": "blue" if total_profit >= 0 else "red"},
        "elements": elements
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

    ind_elements = []
    indicators = data.get('indicators', {})
    if indicators:
        items = [f"**{k}**: {v}" for k, v in indicators.items() if v and v != 'N/A']
        if items:
            ind_section = "\n---\n**技术指标**\n"
            # 每行3个指标，column_set布局
            for i in range(0, len(items[:6]), 3):
                row_items = items[i:i+3]
                columns = []
                for item in row_items:
                    columns.append({"tag": "column", "width": "weighted", "weight": 1,
                                   "elements": [{"tag": "markdown", "content": item}]})
                ind_elements.append({"tag": "column_set", "flex_mode": "bisect",
                                    "background_style": "grey", "columns": columns})

    stock_elements = [
        {"tag": "column_set", "flex_mode": "none", "background_style": "default",
         "columns": [
             {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**当前价格**\n<font color='{color}'>¥{_fmt_price(price, 'ETF' in name)}</font>"}]},
             {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**涨跌幅**\n<font color='{color}'>{sign}{change_pct:.2f}%</font> ({amount_sign}¥{_fmt_amount(change_amount, 'ETF' in name)})"}]},
         ]},
    ]
    if detail:
        stock_elements.append({"tag": "markdown", "content": detail})
    stock_elements.extend(ind_elements)
    stock_elements.append({"tag": "note", "elements": [{"tag": "plain_text", "content": f"更新: {datetime.now().strftime('%H:%M')} | Tushare"}]})

    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": f"📈 {name} ({symbol})"}, "template": color},
        "elements": stock_elements
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
    """交易信号 - column_set行布局"""
    if not signals:
        return make_text_card("当前没有新的交易信号")

    buy_count = sum(1 for s in signals if '买入' in s.get('signal', '') or s.get('signal') == 'buy')
    sell_count = sum(1 for s in signals if '卖出' in s.get('signal', '') or s.get('signal') == 'sell')

    sig_elements = []
    # 表头行
    sig_elements.append({"tag": "column_set", "flex_mode": "none", "background_style": "default",
        "columns": [
            {"tag": "column", "width": "weighted", "weight": 2, "elements": [{"tag": "markdown", "content": "**股票**"}]},
            {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": "**现价**"}]},
            {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": "**信号**"}]},
            {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": "**概率**"}]},
            {"tag": "column", "width": "weighted", "weight": 2, "elements": [{"tag": "markdown", "content": "**原因**"}]},
        ]})
    for s in signals[:10]:
        sig = s.get('signal', '持有')
        emoji = {"买入": "🟢", "卖出": "🔴", "buy": "🟢", "sell": "🔴"}.get(sig, "⚪")
        prob = f"{s.get('up_prob', 0):.0%}" if s.get('up_prob') else "-"
        sig_elements.append({"tag": "column_set", "flex_mode": "none", "background_style": "grey",
            "columns": [
                {"tag": "column", "width": "weighted", "weight": 2, "elements": [{"tag": "markdown", "content": f"{emoji} {s.get('stock_name', '')}"}]},
                {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"¥{s.get('current_price', 0):.2f}"}]},
                {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": sig}]},
                {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": prob}]},
                {"tag": "column", "width": "weighted", "weight": 2, "elements": [{"tag": "markdown", "content": s.get('reason', '')[:30]}]},
            ]})

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
        ] + sig_elements + [
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
    """异动告警（旧版，保留兼容）"""
    template = {"大涨": "green", "大跌": "red", "放量": "orange", "异动": "violet"}.get(alert_type, "red")
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": f"🚨 {alert_type} - {name}"}, "template": template},
        "elements": [
            {"tag": "markdown", "content": f"**{name}** ({symbol})\n{details}"},
            {"tag": "note", "elements": [{"tag": "plain_text", "content": "实时监控告警"}]}
        ]
    }


def make_alert_card_with_hint(alert_type, symbol, name, details, action_hint, ta_data=None):
    """异动告警 + 操作建议（含个股技术指标详情）"""
    template = {
        "大涨": "green", "大跌": "red", "放量大涨": "orange", "放量大跌": "red",
        "缩量大涨": "green", "缩量大跌": "red",
        "接近支撑位": "blue", "接近压力位": "orange",
        "异动": "violet"
    }.get(alert_type, "red")

    elements = []

    # 标题行
    elements.append({"tag": "markdown", "content": f"**{name}** ({symbol})"})

    # 个股技术指标详情（column_set布局）
    if ta_data and 'error' not in ta_data:
        current = ta_data.get('current', 0)
        change_pct = ta_data.get('change_pct', 0)
        sign = '+' if change_pct > 0 else ''
        color = 'red' if change_pct > 0 else 'green' if change_pct < 0 else 'default'
        vol_ratio = ta_data.get('volume_ratio')
        rsi_val = ta_data.get('rsi')
        is_etf = 'ETF' in name or symbol.startswith('15') or symbol.startswith('51') or symbol.startswith('50')
        is_hk = ta_data.get('is_hk', False) or symbol.endswith('.HK')
        currency = 'HK$' if is_hk else '¥'

        ind_columns = [
            {"tag": "column", "width": "weighted", "weight": 1,
             "elements": [{"tag": "markdown", "content": f"**现价**\n<font color='{color}'>{currency}{_fmt_price(current, is_etf)} ({sign}{change_pct:.2f}%)</font>"}]},
        ]
        if vol_ratio:
            vol_color = 'red' if vol_ratio > 2 else 'default'
            ind_columns.append({"tag": "column", "width": "weighted", "weight": 1,
                "elements": [{"tag": "markdown", "content": f"**量比**\n<font color='{vol_color}'>{vol_ratio:.1f}</font>"}]})
        if rsi_val:
            rsi_color = 'red' if rsi_val > 70 else 'green' if rsi_val < 30 else 'default'
            ind_columns.append({"tag": "column", "width": "weighted", "weight": 1,
                "elements": [{"tag": "markdown", "content": f"**RSI**\n<font color='{rsi_color}'>{rsi_val:.1f}</font>"}]})
        ma5 = ta_data.get('ma5')
        ma20 = ta_data.get('ma20')
        if ma5 and ma20:
            ma_color = 'red' if current > ma5 else 'green'
            ind_columns.append({"tag": "column", "width": "weighted", "weight": 1,
                "elements": [{"tag": "markdown", "content": f"**MA5/MA20**\n<font color='{ma_color}'>{_fmt_price(ma5, is_etf)} / {_fmt_price(ma20, is_etf)}</font>"}]})

        elements.append({"tag": "column_set", "flex_mode": "bisect", "background_style": "grey", "columns": ind_columns})

        # 支撑压力位（飞书table）
        supports = ta_data.get('supports', [])
        resistances = ta_data.get('resistances', [])
        if supports or resistances:
            elements.append({"tag": "hr"})
            elements.append({"tag": "markdown", "content": "**关键价位**"})
            sr_elements = _build_sr_columns(supports, resistances, current, is_etf, is_hk)
            elements.extend(sr_elements)

    # 异动详情
    elements.append({"tag": "hr"})
    elements.append({"tag": "markdown", "content": details})

    # 操作建议
    if action_hint:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": "**💡 操作建议**\n" + action_hint})

    elements.append({"tag": "note", "elements": [{"tag": "plain_text", "content": "动态阈值+量价联合 | ⚠️ 不构成投资建议"}]})

    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": f"🚨 {alert_type} - {name}"}, "template": template},
        "elements": elements
    }


def make_backtest_card(data):
    """回测结果 - column_set行布局"""
    summary = data.get('summary', {})
    total_return = summary.get('total_return', 0)
    win_rate = summary.get('win_rate', 0)
    total_trades = summary.get('total_trades', 0)
    name = data.get('name', data.get('symbol', ''))
    ret_color = "green" if total_return > 0 else "red"
    ret_sign = "+" if total_return > 0 else ""
    trades = data.get('trades', [])

    trade_elements = []
    # 表头行
    trade_elements.append({"tag": "column_set", "flex_mode": "none", "background_style": "default",
        "columns": [
            {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": "**操作**"}]},
            {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": "**价格**"}]},
            {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": "**数量**"}]},
            {"tag": "column", "width": "weighted", "weight": 2, "elements": [{"tag": "markdown", "content": "**时间**"}]},
        ]})
    for t in trades[:5]:
        t_type = t.get('type', '')
        emoji = "🟢" if t_type == "buy" else "🔴" if t_type == "sell" else "⚪"
        trade_elements.append({"tag": "column_set", "flex_mode": "none", "background_style": "grey",
            "columns": [
                {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"{emoji} {t_type}"}]},
                {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"¥{t.get('price', 0):.2f}"}]},
                {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"{t.get('shares', 0)}股"}]},
                {"tag": "column", "width": "weighted", "weight": 2, "elements": [{"tag": "markdown", "content": t.get('time', '')[:10]}]},
            ]})

    elements = [
        {"tag": "column_set", "flex_mode": "bisect", "background_style": "default",
         "columns": [
             {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**总收益**\n<font color='{ret_color}'>{ret_sign}{total_return:.2f}%</font>"}]},
             {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**胜率**\n{win_rate:.1f}%"}]},
             {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**交易数**\n{total_trades}笔"}]},
         ]},
        {"tag": "hr"},
    ]
    if trade_elements:
        elements.extend(trade_elements)
    elements.append({"tag": "note", "elements": [{"tag": "plain_text", "content": "LGBM | ⚠️ 回测不代表未来"}]})

    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": f"📊 回测 - {name}"}, "template": ret_color},
        "elements": elements
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


def _build_indicator_grid(indicators: list, row_size: int = 5) -> list:
    """将指标列表转为飞书卡片 column_set 行（每行 row_size 个指标）
    flex_mode 用 none 让列自适应宽度，不强制分组"""
    rows = []
    for i in range(0, len(indicators), row_size):
        row_items = indicators[i:i+row_size]
        columns = []
        for item in row_items:
            columns.append({"tag": "column", "width": "weighted", "weight": 1,
                           "elements": [{"tag": "markdown", "content": item}]})
        rows.append({"tag": "column_set", "flex_mode": "none",
                    "background_style": "grey", "columns": columns})
    return rows


def _build_sr_columns(supports: list, resistances: list, current: float, is_etf: bool = False, is_hk: bool = False) -> list:
    """将支撑压力位转为紧凑布局——支撑一行，压力一行"""
    currency = 'HK$' if is_hk else '¥'
    elements = []
    # 支撑位：一行横排所有价位
    if supports:
        support_parts = []
        for s in supports:
            dist = (current - s) / current * 100 if current > 0 else 0
            support_parts.append(f"{currency}{_fmt_price(s, is_etf)} ({dist:.1f}%)")
        support_text = "🛡️ **支撑**: " + " | ".join(support_parts)
        elements.append({"tag": "markdown", "content": support_text})
    # 压力位：一行横排所有价位
    if resistances:
        resist_parts = []
        for r in resistances:
            dist = (r - current) / current * 100 if current > 0 else 0
            resist_parts.append(f"{currency}{_fmt_price(r, is_etf)} ({dist:.1f}%)")
        resist_text = "🚧 **压力**: " + " | ".join(resist_parts)
        elements.append({"tag": "markdown", "content": resist_text})
    return elements


def make_technical_card(data: dict) -> dict:
    """技术分析卡片 - column_set网格 + table布局"""
    name = data.get('name', data.get('symbol', ''))
    symbol = data.get('symbol', '')
    current = data.get('current', 0)
    change_pct = data.get('change_pct', 0)
    is_etf = 'ETF' in name or symbol.startswith('15') or symbol.startswith('51') or symbol.startswith('50')
    is_hk = data.get('is_hk', False) or symbol.endswith('.HK')
    currency = 'HK$' if is_hk else '¥'
    color = "red" if change_pct > 0 else "green" if change_pct < 0 else "default"
    sign = "+" if change_pct > 0 else ""

    # 指标概览（column_set网格）
    indicators = []
    ma5, ma10, ma20, ma60 = data.get('ma5'), data.get('ma10'), data.get('ma20'), data.get('ma60')
    if ma5: indicators.append(f"**MA5**\n{ma5:.2f}")
    if ma10: indicators.append(f"**MA10**\n{ma10:.2f}")
    if ma20: indicators.append(f"**MA20**\n{ma20:.2f}")
    if ma60: indicators.append(f"**MA60**\n{ma60:.2f}")

    rsi = data.get('rsi')
    if rsi:
        rsi_color = "red" if rsi > 70 else "green" if rsi < 30 else "default"
        indicators.append(f"**RSI**\n<font color='{rsi_color}'>{rsi:.1f}</font>")

    kdj_j = data.get('kdj_j')
    if kdj_j:
        j_color = "red" if kdj_j > 100 else "green" if kdj_j < 0 else "default"
        indicators.append(f"**KDJ-J**\n<font color='{j_color}'>{kdj_j:.1f}</font>")

    # 量比（0.0表示数据缺失，不显示）
    vol_ratio = data.get('volume_ratio')
    if vol_ratio and vol_ratio > 0.01:
        vol_color = "red" if vol_ratio > 2 else "default"
        indicators.append(f"**量比**\n<font color='{vol_color}'>{vol_ratio:.1f}</font>")

    macd_dif = data.get('macd_dif')
    macd_dea = data.get('macd_dea')
    if macd_dif and macd_dea:
        macd_color = "red" if macd_dif > macd_dea else "green"
        indicators.append(f"**MACD**\n<font color='{macd_color}'>DIF {macd_dif:.2f}</font>")

    boll_upper, boll_lower = data.get('boll_upper'), data.get('boll_lower')
    if boll_upper and boll_lower:
        indicators.append(f"**布林**\n上{_fmt_price(boll_upper, is_etf)} 下{_fmt_price(boll_lower, is_etf)}")

    threshold = data.get('dynamic_threshold')
    if threshold:
        indicators.append(f"**阈值**\n{threshold:.1f}%")

    # 构建元素列表
    elements = [
        {"tag": "markdown", "content": f"**当前** <font color='{color}'>{currency}{_fmt_price(current, is_etf)} ({sign}{change_pct:.2f}%)</font>"},
    ]

    # 指标网格（每行3个）
    if indicators:
        elements.extend(_build_indicator_grid(indicators, row_size=5))

    # 信号列表
    signals = data.get('signals', [])
    if signals:
        signal_line = "**🔔 技术信号**\n"
        for s in signals:
            emoji_map = {"金叉": "🟢", "死叉": "🔴", "超买": "⚠️", "超卖": "💡", "多头": "📈", "空头": "📉", "新高": "🔥", "新低": "❄️", "放量": "⚡", "缩量": "🔇", "支撑": "🛡️", "压力": "🚧", "突破": "🎯", "跌破": "💥"}
            emoji = "🔔"
            for kw, e in emoji_map.items():
                if kw in s:
                    emoji = e
                    break
            signal_line += f"- {emoji} {s}\n"
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": signal_line})
    else:
        elements.append({"tag": "markdown", "content": "*当前无重要技术信号*"})

    # 消息面（提前到支撑压力位之前，更显眼）
    news_hint = data.get('news_hint', '')
    if news_hint:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": news_hint})

    # 支撑压力位
    supports = data.get('supports', [])
    resistances = data.get('resistances', [])
    if supports or resistances:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": "**关键价位**"})
        sr_elements = _build_sr_columns(supports, resistances, current, is_etf, is_hk)
        elements.extend(sr_elements)

    # 操作建议
    action_hint = data.get('action_hint', '')
    if action_hint:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": "**💡 操作建议**\n" + action_hint})

    # LGBM AI预测
    lgbm = data.get('lgbm')
    if lgbm:
        up_prob = lgbm.get('up_prob', 0)
        signal = lgbm.get('signal', '')
        win_rate = lgbm.get('win_rate', 0)
        prob_color = 'red' if up_prob > 0.5 else 'green' if up_prob < 0.5 else 'default'
        
        # 一致性检查：AI预测 vs 技术信号
        is_bearish = any('空头' in s or '死叉' in s for s in signals)
        is_bullish = any('多头' in s or '金叉' in s for s in signals)
        conflict_note = ''
        if signal == '看涨' and is_bearish:
            conflict_note = '\n⚠️ 与技术面冲突，仅供参考'
        elif signal == '看跌' and is_bullish:
            conflict_note = '\n⚠️ 与技术面冲突，仅供参考'
        
        elements.append({"tag": "column_set", "flex_mode": "bisect", "background_style": "grey",
            "columns": [
                {"tag": "column", "width": "weighted", "weight": 1,
                 "elements": [{"tag": "markdown", "content": f"**🤖 AI预测**\n<font color='{prob_color}'>{signal} ({up_prob:.0%})</font>{conflict_note}"}]},
                {"tag": "column", "width": "weighted", "weight": 1,
                 "elements": [{"tag": "markdown", "content": f"**历史胜率**\n{win_rate:.1%}"}]},
            ]})

    elements.append({"tag": "note", "elements": [{"tag": "plain_text", "content": "技术面+消息面+LLM | ⚠️ 不构成投资建议"}]})

    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": f"📊 {name} ({symbol}) 技术分析"}, "template": "blue"},
        "elements": elements
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
    
    # 估值行（column_set布局）
    val_items = []
    pe = v.get('pe_ttm')
    if pe: val_items.append(f"**PE**\n{pe:.1f}")
    pb = v.get('pb')
    if pb: val_items.append(f"**PB**\n{pb:.2f}")
    mv = v.get('total_mv')
    if mv: val_items.append(f"**市值**\n{mv/10000:.0f}亿")  # 万元→亿
    tr = v.get('turnover_rate')
    if tr: val_items.append(f"**换手率**\n{tr:.2f}%")
    vr = v.get('volume_ratio')
    if vr: val_items.append(f"**量比**\n{vr:.1f}")
    val_grid = _build_indicator_grid(val_items, row_size=3) if val_items else [{"tag": "markdown", "content": "暂无估值数据"}]
    
    # 盈利行（column_set布局）
    profit_items = []
    roe = p.get('roe')
    if roe: profit_items.append(f"**ROE**\n{roe:.2f}%")
    npm = p.get('netprofit_margin')
    if npm: profit_items.append(f"**净利率**\n{npm:.2f}%")
    gpm = p.get('grossprofit_margin')
    if gpm: profit_items.append(f"**毛利率**\n{gpm:.2f}%")
    or_yoy = p.get('or_yoy')
    if or_yoy: profit_items.append(f"**营收增速**\n{or_yoy:.2f}%")
    np_yoy = p.get('netprofit_yoy')
    if np_yoy: profit_items.append(f"**净利增速**\n{np_yoy:.2f}%")
    profit_grid = _build_indicator_grid(profit_items, row_size=3) if profit_items else [{"tag": "markdown", "content": "暂无盈利数据"}]
    end_date = p.get('end_date', '')
    
    elements = [
        {"tag": "markdown", "content": f"**当前** <font color='{color}'>¥{_fmt_price(current, 'ETF' in name)} ({sign}{change_pct:.2f}%)</font>"},
        {"tag": "hr"},
        {"tag": "markdown", "content": "**估值指标**"},
    ]
    elements.extend(val_grid)
    elements.append({"tag": "hr"})
    elements.append({"tag": "markdown", "content": f"**盈利指标**（{end_date}）"})
    elements.extend(profit_grid)
    elements.append({"tag": "note", "elements": [{"tag": "plain_text", "content": "Tushare | ⚠️ 仅供参考"}]})

    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": f"🔍 {name} ({symbol}) 深度数据"}, "template": "blue"},
        "elements": elements
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
    """综合操作建议卡片 - 每只股票独立区块，column_set+table布局"""
    recs = data.get('recommendations', [])
    sentiment = data.get('market_sentiment', '未知')
    avg_market_pct = data.get('avg_market_pct', 0)
    market_sign = '+' if avg_market_pct > 0 else '' if avg_market_pct != 0 else ''
    market_color = 'red' if avg_market_pct > 0 else 'green' if avg_market_pct < 0 else 'default'

    # 大盘情绪header
    elements = [
        {"tag": "markdown", "content": f"**大盘情绪**: {sentiment}"},
    ]
    if avg_market_pct:
        elements.append({"tag": "column_set", "flex_mode": "bisect", "background_style": "default",
            "columns": [
                {"tag": "column", "width": "weighted", "weight": 1,
                 "elements": [{"tag": "markdown", "content": f"**大盘涨跌**\n<font color='{market_color}'>{market_sign}{avg_market_pct:.2f}%</font>"}]},
            ]})
    elements.append({"tag": "hr"})

    # 每只股票一个独立区块
    for r in recs:
        action = r.get('action', '持有')
        priority = r.get('priority', '✅')
        profit_pct = r.get('profit_pct', 0)
        profit_sign = '+' if profit_pct >= 0 else ''
        profit_color = 'red' if profit_pct > 0 else 'green'
        confidence = r.get('confidence', '中')
        is_etf = r.get('is_etf', False)
        name = r.get('name', '')
        symbol = r.get('symbol', '')
        current = r.get('current_price', 0)

        # 标题：优先级+股票名+操作+置信度
        elements.append({"tag": "markdown",
            "content": priority + " **" + name + "** → " + action + "（置信度" + confidence + "）"})

        # 关键数据 column_set: 盈亏 + 原因（自适应宽度）
        reason_col = [{"tag": "column", "width": "auto",
            "elements": [{"tag": "markdown",
                "content": f"<font color='{profit_color}'>{profit_sign}{profit_pct:.1f}%</font>  " + r.get('reason', '')}]}]
        # 做T价位
        if r.get('t_suggestion') and r['t_suggestion'].get('buy_price'):
            reason_col.append({"tag": "column", "width": "auto",
                "elements": [{"tag": "markdown",
                    "content": f"**做T**\n低买¥{_fmt_price(r['t_suggestion']['buy_price'], is_etf)} 高卖¥{_fmt_price(r['t_suggestion']['sell_price'], is_etf)}"}]})
        if r.get('price_target'):
            reason_col.append({"tag": "column", "width": "auto",
                "elements": [{"tag": "markdown",
                    "content": f"**目标价**\n¥{_fmt_price(r['price_target'], is_etf)}"}]})
        elements.append({"tag": "column_set", "flex_mode": "bisect", "background_style": "grey", "columns": reason_col})

        # 技术指标详情（column_set网格）
        td = r.get('tech_detail', {})
        if td:
            td_items = []
            if td.get('change_pct') is not None:
                chg_sign = '+' if td['change_pct'] >= 0 else ''
                chg_color = 'red' if td['change_pct'] > 0 else 'green' if td['change_pct'] < 0 else 'default'
                td_items.append(f"**涨跌**\n<font color='{chg_color}'>{chg_sign}{td['change_pct']:.2f}%</font>")
            if td.get('rsi') is not None:
                rsi_color = 'red' if td['rsi'] > 70 else 'green' if td['rsi'] < 30 else 'default'
                td_items.append(f"**RSI**\n<font color='{rsi_color}'>{td['rsi']:.1f}</font>")
            if td.get('volume_ratio') is not None:
                td_items.append(f"**量比**\n{td['volume_ratio']:.1f}")
            if td.get('macd_dif') is not None and td.get('macd_dea') is not None:
                macd_color = 'red' if td['macd_dif'] > td['macd_dea'] else 'green'
                td_items.append(f"**MACD**\n<font color='{macd_color}'>DIF {td['macd_dif']:.2f}</font>")
            if td.get('supports'):
                td_items.append(f"**支撑**\n" + '/'.join([_fmt_price(s, is_etf) for s in td['supports']]))
            if td.get('resistances'):
                td_items.append(f"**压力**\n" + '/'.join([_fmt_price(v, is_etf) for v in td['resistances']]))
            if td_items:
                elements.extend(_build_indicator_grid(td_items, row_size=3))

        # 技术信号
        if r.get('key_signals'):
            sig_text = ""
            for sig in r['key_signals'][:3]:
                sig_text += "🔔 " + sig + chr(10)
            if sig_text:
                elements.append({"tag": "markdown", "content": sig_text})

        # LGBM AI预测
        lgbm = r.get('lgbm')
        if lgbm:
            up_prob = lgbm.get('up_prob', 0)
            signal = lgbm.get('signal', '')
            win_rate = lgbm.get('win_rate', 0)
            prob_color = 'red' if up_prob > 0.5 else 'green' if up_prob < 0.5 else 'default'
            elements.append({"tag": "column_set", "flex_mode": "bisect", "background_style": "default",
                "columns": [
                    {"tag": "column", "width": "weighted", "weight": 1,
                     "elements": [{"tag": "markdown", "content": f"**AI预测**\n<font color='{prob_color}'>看涨{up_prob:.0%}</font> → {signal}"}]},
                    {"tag": "column", "width": "weighted", "weight": 1,
                     "elements": [{"tag": "markdown", "content": f"**胜率**\n{win_rate:.1f}%"}]},
                ]})

        # 新闻面
        news_list = r.get('news', [])
        if news_list:
            news_text = "**📰 消息面**" + chr(10)
            for n in news_list[:3]:
                news_text += "- " + n['title'] + chr(10)
            elements.append({"tag": "markdown", "content": news_text})
        news_sent = r.get('news_sentiment')
        if news_sent:
            ns_score = news_sent.get('score', 0.5)
            ns_emoji = "🟢" if ns_score > 0.6 else "🔴" if ns_score < 0.4 else "🟡"
            ns_label = news_sent.get('sentiment_label', '中性')
            ns_summary = news_sent.get('summary', '')
            ns_line = ns_emoji + " " + ns_label + "（" + str(round(ns_score, 2)) + "）"
            if ns_summary:
                ns_line += " — " + ns_summary
            elements.append({"tag": "markdown",
                "content": ns_line})

        # 每只股票之间加分隔线
        elements.append({"tag": "hr"})

    # 去掉最后一个多余的分隔线
    if elements and elements[-1]['tag'] == 'hr':
        elements.pop()

    elements.append({"tag": "note", "elements": [{"tag": "plain_text", "content": "技术+消息+AI预测+做T | ⚠️ 不构成投资建议"}]})

    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "💡 综合操作建议"}, "template": "blue"},
        "elements": elements
    }


def make_news_card(data: dict) -> dict:
    """财经要闻卡片"""
    news = data.get('news', [])
    sentiment = data.get('sentiment', {})
    keyword = data.get('keyword', '')
    
    if not news:
        return make_text_card("暂无相关财经新闻")
    
    sentiment_label = sentiment.get('sentiment_label', '中性')
    sentiment_score = sentiment.get('score', 0.5)
    sentiment_emoji = "🟢" if sentiment_score > 0.6 else "🔴" if sentiment_score < 0.4 else "🟡"
    sentiment_summary = sentiment.get('summary', '')
    
    NL = chr(10)
    lines = "**新闻情绪**: " + sentiment_emoji + " " + sentiment_label + "（" + str(round(sentiment_score, 2)) + "）" + NL
    if sentiment_summary:
        lines += sentiment_summary + NL + NL
    else:
        lines += NL
    
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
            {"tag": "note", "elements": [{"tag": "plain_text", "content": "Bing News RSS + LLM情绪分析（近7日）"}]}
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
    
    # 估值指标（column_set布局）
    val_items = []
    if pe: val_items.append(f"**PE**\n{pe:.1f}")
    if pb: val_items.append(f"**PB**\n{pb:.2f}")
    if total_mv: val_items.append(f"**市值**\n{total_mv/10000:.0f}亿")
    if turnover: val_items.append(f"**换手率**\n{turnover:.2f}%")
    
    val_grid = _build_indicator_grid(val_items, row_size=3) if val_items else []
    
    elements = [
        {"tag": "markdown", "content": f"**当前** ¥{current:.2f}\n\n**估值判断**: {color_emoji} {level}"},
    ]
    if val_grid:
        elements.extend(val_grid)
    if signals:
        signal_line = "**技术信号**\n"
        for s in signals:
            signal_line += "- 🔔 " + s + "\n"
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": signal_line})
    elements.append({"tag": "note", "elements": [{"tag": "plain_text", "content": "PE/PB对比行业 | ⚠️ 仅供参考"}]})

    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": color_emoji + " " + name + " 估值: " + level}, "template": "blue"},
        "elements": elements
    }
