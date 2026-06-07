#!/usr/bin/env python3
"""
飞书消息卡片模板
定义各种场景下的卡片 JSON 结构

卡片设计原则：
- header 用 emoji + 简短标题，template 用颜色区分类型
- 数据用 column_set 多列布局
- 表格用 markdown 展示
- footer 用 note 带免责声明
"""

from datetime import datetime


def make_position_card(summary: dict, positions: list, t_suggestions: list = None) -> dict:
    """持仓概览卡片"""
    total_value = summary.get('total_value', 0)
    total_cost = summary.get('total_cost', 0)
    total_profit = summary.get('total_profit', 0)
    profit_pct = summary.get('profit_pct', 0)
    available_cash = summary.get('available_cash', 0)

    profit_color = "red" if total_profit < 0 else "green"
    profit_sign = "+" if total_profit >= 0 else ""

    pos_rows = ""
    for p in positions[:10]:
        p_profit_pct = p.get('profit_pct', 0)
        p_sign = "+" if p_profit_pct >= 0 else ""
        p_color = "🔴" if p_profit_pct < 0 else "🟢"
        pos_rows += f"| {p_color} {p['stock_name']} | {p['shares']}股 | ¥{p['cost_price']:.2f} | ¥{p['current_price']:.2f} | {p_sign}{p_profit_pct:.1f}% |\n"

    t_section = ""
    if t_suggestions:
        t_section = "\n---\n**💡 做T操作建议**\n"
        for t in t_suggestions[:5]:
            emoji = {"适合做T": "🟢", "可减仓": "🔵", "观望": "⚠️", "不建议": "❌"}.get(t.get('action', ''), '⚪')
            t_section += f"- {emoji} **{t['stock_name']}** {t.get('action', '')}：{t.get('reason', '')}\n"
            if t.get('buy_price'):
                t_section += f"  买入 ¥{t['buy_price']:.2f} × {t.get('buy_shares', 0)}股\n"
            if t.get('sell_price'):
                t_section += f"  卖出 ¥{t['sell_price']:.2f} × {t.get('sell_shares', 0)}股\n"

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "📊 持仓概览"},
            "template": "blue" if total_profit >= 0 else "red"
        },
        "elements": [
            {
                "tag": "column_set",
                "flex_mode": "bisect",
                "background_style": "default",
                "columns": [
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 1,
                        "elements": [{"tag": "markdown", "content": f"**总市值**\n¥{total_value:,.0f}"}]
                    },
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 1,
                        "elements": [{"tag": "markdown", "content": f"**浮动盈亏**\n<font color='{profit_color}'>{profit_sign}¥{total_profit:,.0f} ({profit_sign}{profit_pct:.1f}%)</font>"}]
                    },
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 1,
                        "elements": [{"tag": "markdown", "content": f"**可用现金**\n¥{available_cash:,.0f}"}]
                    }
                ]
            },
            {"tag": "hr"},
            {"tag": "markdown", "content": f"| 股票 | 持仓 | 成本 | 现价 | 盈亏 |\n|---|---|---|---|---|\n{pos_rows}"},
            {"tag": "markdown", "content": t_section},
            {
                "tag": "note",
                "elements": [{"tag": "plain_text", "content": "⚠️ 以上数据仅供参考，不构成投资建议"}]
            }
        ]
    }


def make_stock_card(stock_data: dict) -> dict:
    """单只股票行情卡片"""
    symbol = stock_data.get('symbol', '')
    name = stock_data.get('name', stock_data.get('stock_name', symbol))
    price = stock_data.get('current_price', stock_data.get('price', 0))
    change_pct = stock_data.get('change_pct', stock_data.get('profit_pct', 0))
    change_amount = stock_data.get('change_amount', stock_data.get('change', 0))

    change_color = "green" if change_pct > 0 else "red" if change_pct < 0 else "default"
    # A股: 红涨绿跌; 港股/国际: 绿涨红跌
    if symbol and symbol.endswith('.HK'):
        # 港股用国际惯例: 绿涨红跌
        change_color = "green" if change_pct > 0 else "red" if change_pct < 0 else "default"
    else:
        # A股用中国惯例: 红涨绿跌
        change_color = "red" if change_pct > 0 else "green" if change_pct < 0 else "default"
    change_sign = "+" if change_pct > 0 else ""
    amount_sign = "+" if change_amount > 0 else ""

    # 详细数据
    volume = stock_data.get('volume', 0)
    turnover = stock_data.get('turnover', stock_data.get('amount', 0))
    high = stock_data.get('high', 0)
    low = stock_data.get('low', 0)
    open_price = stock_data.get('open', stock_data.get('open_price', 0))

    # 技术指标
    indicators = stock_data.get('indicators', {})
    ind_section = ""
    if indicators:
        ind_items = []
        for key, val in indicators.items():
            if val is not None and val != 'N/A':
                ind_items.append(f"**{key}**: {val}")
        if ind_items:
            ind_section = "\n---\n**技术指标**\n" + " | ".join(ind_items[:6])

    # 做T建议
    t_section = ""
    t_data = stock_data.get('t_suggestion')
    if t_data:
        t_section = f"\n---\n**做T建议**: {t_data.get('action', '观望')} - {t_data.get('reason', '')}\n"
        if t_data.get('buy_price'):
            t_section += f"买入 ¥{t_data['buy_price']:.2f} × {t_data.get('buy_shares', 0)}股\n"
        if t_data.get('sell_price'):
            t_section += f"卖出 ¥{t_data['sell_price']:.2f} × {t_data.get('sell_shares', 0)}股\n"

    # 行情明细行
    detail_line = ""
    if high and low:
        detail_line = f"\n最高 ¥{high:.2f} | 最低 ¥{low:.2f} | 开盘 ¥{open_price:.2f}"
    if volume:
        volume_str = f"{volume/10000:.1f}万手" if volume > 10000 else f"{volume}手"
        detail_line += f" | 成交量 {volume_str}"

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"📈 {name} ({symbol})"},
            "template": change_color
        },
        "elements": [
            {
                "tag": "column_set",
                "flex_mode": "bisect",
                "background_style": "default",
                "columns": [
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 1,
                        "elements": [{"tag": "markdown", "content": f"**当前价格**\n<font color='{change_color}'>¥{price:.2f}</font>"}]
                    },
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 1,
                        "elements": [{"tag": "markdown", "content": f"**涨跌幅**\n<font color='{change_color}'>{change_sign}{change_pct:.2f}%</font>\n{amount_sign}¥{abs(change_amount):.2f}"}]
                    }
                ]
            },
            {"tag": "markdown", "content": detail_line},
            {"tag": "markdown", "content": ind_section + t_section},
            {
                "tag": "note",
                "elements": [{"tag": "plain_text", "content": f"更新时间: {datetime.now().strftime('%H:%M')} | 数据来源: Tushare"}]
            }
        ]
    }


def make_signal_card(signals: list) -> dict:
    """交易信号卡片"""
    if not signals:
        return make_text_card("当前没有新的交易信号")

    buy_count = sum(1 for s in signals if '买入' in s.get('signal', '') or s.get('signal') == 'buy')
    sell_count = sum(1 for s in signals if '卖出' in s.get('signal', '') or s.get('signal') == 'sell')
    hold_count = len(signals) - buy_count - sell_count

    rows = ""
    for s in signals[:10]:
        signal_type = s.get('signal', '持有')
        signal_emoji = {"买入": "🟢", "卖出": "🔴", "持有": "⚪", "buy": "🟢", "sell": "🔴", "hold": "⚪"}.get(signal_type, "⚪")
        up_prob = s.get('up_prob', 0)
        prob_str = f"{up_prob:.0%}" if up_prob else "-"
        rows += f"| {signal_emoji} {s.get('stock_name', '')} | ¥{s.get('current_price', s.get('price', 0)):.2f} | {signal_type} | {prob_str} | {s.get('reason', '')[:30]} |\n"

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "🔔 交易信号"},
            "template": "blue"
        },
        "elements": [
            {
                "tag": "column_set",
                "flex_mode": "bisect",
                "background_style": "default",
                "columns": [
                    {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**买入信号**: {buy_count}只"}]},
                    {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**卖出信号**: {sell_count}只"}]},
                    {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**持有**: {hold_count}只"}]}
                ]
            },
            {"tag": "hr"},
            {"tag": "markdown", "content": f"| 股票 | 现价 | 信号 | 上涨概率 | 原因 |\n|---|---|---|---|---|\n{rows}"},
            {
                "tag": "note",
                "elements": [{"tag": "plain_text", "content": "⚠️ 信号仅供参考，不构成投资建议"}]
            }
        ]
    }


def make_backtest_card(backtest_data: dict) -> dict:
    """回测结果卡片"""
    summary = backtest_data.get('summary', {})
    total_return = summary.get('total_return', summary.get('profitRate', 0))
    win_rate = summary.get('win_rate', summary.get('winRate', 0))
    total_trades = summary.get('total_trades', summary.get('totalTrades', 0))
    symbol = backtest_data.get('symbol', '')
    name = backtest_data.get('name', symbol)
    initial_capital = summary.get('initial_capital', 500000)
    final_value = summary.get('final_value', summary.get('finalCapital', 0))

    return_color = "green" if total_return > 0 else "red"
    return_sign = "+" if total_return > 0 else ""

    trades = backtest_data.get('trades', [])
    trade_rows = ""
    for t in trades[:5]:
        t_type = t.get('type', t.get('trade_type', ''))
        t_emoji = "🟢" if t_type == "buy" else "🔴" if t_type == "sell" else "⚪"
        trade_rows += f"| {t_emoji} {t_type} | ¥{t.get('price', 0):.2f} | {t.get('shares', 0)}股 | {t.get('time', '')[:10]} | {t.get('reason', '')[:20]} |\n"

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"📊 回测结果 - {name}"},
            "template": return_color
        },
        "elements": [
            {
                "tag": "column_set",
                "flex_mode": "bisect",
                "background_style": "default",
                "columns": [
                    {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**总收益**\n<font color='{return_color}'>{return_sign}{total_return:.2f}%</font>"}]},
                    {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**胜率**\n{win_rate:.1f}%"}]},
                    {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**交易数**\n{total_trades}笔"}]}
                ]
            },
            {"tag": "markdown", "content": f"初始资金 ¥{initial_capital:,.0f} → 终值 ¥{final_value:,.0f}"},
            {"tag": "hr"},
            {"tag": "markdown", "content": f"| 操作 | 价格 | 数量 | 时间 | 原因 |\n|---|---|---|---|---|\n{trade_rows}"},
            {
                "tag": "note",
                "elements": [{"tag": "plain_text", "content": "基于LGBM模型 | ⚠️ 回测不代表未来"}]
            }
        ]
    }


def make_daily_summary_card(summary: dict, positions: list, signals: list, t_suggestions: list = None) -> dict:
    """每日总结卡片"""
    date = summary.get('date', '')
    total_value = summary.get('total_value', 0)
    total_profit = summary.get('total_profit', 0)
    profit_pct = summary.get('profit_pct', 0)

    profit_color = "red" if total_profit < 0 else "green"
    profit_sign = "+" if total_profit >= 0 else ""

    if positions:
        best = max(positions, key=lambda p: p.get('profit_pct', 0))
        worst = min(positions, key=lambda p: p.get('profit_pct', 0))
        highlight = f"🏆 **最佳**: {best['stock_name']} +{best['profit_pct']:.1f}%\n💀 **最差**: {worst['stock_name']} {worst['profit_pct']:.1f}%"
    else:
        highlight = ""

    suggestions_text = ""
    for s in signals:
        if s.get('action') != '持有':
            suggestions_text += f"- **{s['stock_name']}**: {s['action']} - {s['reason']}\n"
    if not suggestions_text:
        suggestions_text = "- 今日无特别操作建议，继续持有观望"

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"📝 盘后总结 - {date}"},
            "template": "blue"
        },
        "elements": [
            {
                "tag": "column_set",
                "flex_mode": "bisect",
                "background_style": "default",
                "columns": [
                    {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**总市值**\n¥{total_value:,.0f}"}]},
                    {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"**今日盈亏**\n<font color='{profit_color}'>{profit_sign}¥{total_profit:,.0f} ({profit_sign}{profit_pct:.1f}%)</font>"}]}
                ]
            },
            {"tag": "hr"},
            {"tag": "markdown", "content": highlight},
            {"tag": "hr"},
            {"tag": "markdown", "content": f"**操作建议**\n{suggestions_text}"},
            {
                "tag": "note",
                "elements": [{"tag": "plain_text", "content": "⚠️ 以上数据仅供参考，不构成投资建议"}]
            }
        ]
    }


def make_help_card() -> dict:
    """帮助卡片"""
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "📘 金融小助手使用指南"},
            "template": "blue"
        },
        "elements": [
            {
                "tag": "markdown",
                "content": """**🔍 查询类**
- `持仓` → 查看持仓概览和做T建议
- `行情 茅台` / `茅台怎么样` → 查看个股行情
- `信号` → 查看最新交易信号
- `做T` → 查看做T操作建议
- `大盘` / `指数` → 查看A股/港股主要指数
- `板块` / `热门板块` → 查看今日热门板块

**📊 分析类**
- `回测 茅台` → 运行LGBM回测
- `分析 茅台` → 综合分析个股
- `总结` / `日报` → 盘后总结
- `对比 茅台 五粮液` → 多股对比

**⚙️ 配置类**
- `自选 阿里巴巴` → 添加自选股
- `帮助` → 显示此帮助信息

**💬 闲聊**
- 直接发任何问题，我会用AI回答"""
            },
            {
                "tag": "note",
                "elements": [{"tag": "plain_text", "content": "Powered by LGBM + 百炼 | ⚠️ 不构成投资建议"}]
            }
        ]
    }


def make_chat_card(text: str) -> dict:
    """AI对话卡片 - 区别于普通文本卡片，有金融助手风格"""
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "💬 金融小助手"},
            "template": "turquoise"
        },
        "elements": [
            {"tag": "markdown", "content": text},
            {
                "tag": "note",
                "elements": [{"tag": "plain_text", "content": "AI生成，仅供参考 | 试试发送「帮助」查看更多功能"}]
            }
        ]
    }


def make_text_card(text: str) -> dict:
    """简单文本卡片（用于错误提示等）"""
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "金融小助手"},
            "template": "default"
        },
        "elements": [{"tag": "markdown", "content": text}]
    }




def make_market_card(market_data: dict) -> dict:
    """大盘指数卡片 - 用飞书原生 column_set 布局"""
    indices = market_data.get('indices', [])
    if not indices:
        return make_text_card("暂无大盘数据")

    # 每个指数一行，用 column_set 展示
    elements = []
    for idx in indices:
        name = idx.get('name', '')
        code = idx.get('code', '')
        price = idx.get('price', 0)
        change_pct = idx.get('change_pct', 0)
        change_amount = idx.get('change_amount', 0)
        sign = "+" if change_pct > 0 else ""
        
        # A股红涨绿跌，港股绿涨红跌
        if code and (code.endswith('.SH') or code.endswith('.SZ')):
            color = "red" if change_pct > 0 else "green" if change_pct < 0 else "default"
        else:
            color = "green" if change_pct > 0 else "red" if change_pct < 0 else "default"
        
        # 格式化价格（指数点位可能是几千点）
        price_str = f"{price:.2f}" if price < 10000 else f"{price:,.2f}"

        elements.append({
            "tag": "column_set",
            "flex_mode": "bisect",
            "background_style": "grey",
            "columns": [
                {
                    "tag": "column",
                    "width": "weighted",
                    "weight": 2,
                    "elements": [{"tag": "markdown", "content": f"**{name}**"}]
                },
                {
                    "tag": "column",
                    "width": "weighted",
                    "weight": 1,
                    "elements": [{"tag": "markdown", "content": f"{price_str}"}]
                },
                {
                    "tag": "column",
                    "width": "weighted",
                    "weight": 1,
                    "elements": [{"tag": "markdown", "content": f"<font color='{color}'>{sign}{change_pct:.2f}%</font>"}]
                },
            ]
        })

    # 市场情绪
    sentiment = market_data.get('sentiment', '')
    if sentiment:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": f"**市场情绪**: {sentiment}"})

    elements.append({
        "tag": "note",
        "elements": [{"tag": "plain_text", "content": f"更新: {datetime.now().strftime('%H:%M')} | 数据来源: 腾讯财经"}]
    })

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "🌍 大盘指数"},
            "template": "blue"
        },
        "elements": elements
    }


def make_sector_card(sector_data: dict) -> dict:
    """热门板块卡片 - 用飞书原生 column_set 布局"""
    sectors = sector_data.get('sectors', [])
    if not sectors:
        return make_text_card("暂无板块数据")

    elements = []
    for s in sectors[:10]:
        name = s.get('name', '')
        change_pct = s.get('change_pct', 0)
        sign = "+" if change_pct > 0 else ""
        # A股红涨绿跌
        color = "red" if change_pct > 0 else "green" if change_pct < 0 else "default"
        
        elements.append({
            "tag": "column_set",
            "flex_mode": "bisect",
            "background_style": "grey",
            "columns": [
                {
                    "tag": "column",
                    "width": "weighted",
                    "weight": 2,
                    "elements": [{"tag": "markdown", "content": f"**{name}**"}]
                },
                {
                    "tag": "column",
                    "width": "weighted",
                    "weight": 1,
                    "elements": [{"tag": "markdown", "content": f"<font color='{color}'>{sign}{change_pct:.2f}%</font>"}]
                },
            ]
        })

    elements.append({
        "tag": "note",
        "elements": [{"tag": "plain_text", "content": f"更新: {datetime.now().strftime('%H:%M')} | 数据来源: 腾讯财经"}]
    })

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "🔥 今日热门板块"},
            "template": "orange"
        },
        "elements": elements
    }

    rows = ""
    for s in stocks:
        name = s.get('name', '')
        symbol = s.get('symbol', '')
        price = s.get('current_price', 0)
        change_pct = s.get('change_pct', 0)
        sign = "+" if change_pct > 0 else ""
        color = "🔴" if change_pct > 0 else "🟢" if change_pct < 0 else "⚪"
        rows += f"| {color} {name} | {symbol} | ¥{price:.2f} | {sign}{change_pct:.2f}% |\n"

    # 找出最强和最弱
    best = max(stocks, key=lambda s: s.get('change_pct', 0))
    worst = min(stocks, key=lambda s: s.get('change_pct', 0))
    insight = f"🏆 **最强**: {best.get('name', '')} +{best.get('change_pct', 0):.2f}%\n💀 **最弱**: {worst.get('name', '')} {worst.get('change_pct', 0):.2f}%"

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"📊 股票对比 ({len(stocks)}只)"},
            "template": "blue"
        },
        "elements": [
            {"tag": "markdown", "content": f"| 股票 | 代码 | 现价 | 涨跌幅 |\n|---|---|---|---|\n{rows}"},
            {"tag": "hr"},
            {"tag": "markdown", "content": insight},
            {
                "tag": "note",
                "elements": [{"tag": "plain_text", "content": f"更新: {datetime.now().strftime('%H:%M')} | ⚠️ 不构成投资建议"}]
            }
        ]
    }


def make_stock_compare_card(compare_data: dict) -> dict:
    """多股对比卡片 - column_set 布局"""
    stocks = compare_data.get('stocks', [])
    if not stocks:
        return make_text_card("暂无可对比的数据")

    elements = []
    best = max(stocks, key=lambda s: s.get('change_pct', 0))
    worst = min(stocks, key=lambda s: s.get('change_pct', 0))

    for s in stocks:
        name = s.get('name', '')
        symbol = s.get('symbol', '')
        price = s.get('current_price', 0)
        change_pct = s.get('change_pct', 0)
        sign = '+' if change_pct > 0 else ''
        # 港股绿涨红跌, A股红涨绿跌
        if symbol and symbol.endswith('.HK'):
            color = 'green' if change_pct > 0 else 'red' if change_pct < 0 else 'default'
        else:
            color = 'red' if change_pct > 0 else 'green' if change_pct < 0 else 'default'
        
        elements.append({
            "tag": "column_set",
            "flex_mode": "bisect",
            "background_style": "grey",
            "columns": [
                {"tag": "column", "width": "weighted", "weight": 2, "elements": [{"tag": "markdown", "content": f"**{name}** ({symbol})"}]},
                {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"\u00a5{price:.2f}"}]},
                {"tag": "column", "width": "weighted", "weight": 1, "elements": [{"tag": "markdown", "content": f"<font color='{color}'>{sign}{change_pct:.2f}%</font>"}]},
            ]
        })

    elements.append({"tag": "hr"})
    best_pct = best.get('change_pct', 0)
    worst_pct = worst.get('change_pct', 0)
    best_sign = '+' if best_pct > 0 else ''
    worst_sign = '+' if worst_pct > 0 else ''
    elements.append({"tag": "markdown", "content": f"\U0001f3c6 **最强**: {best.get('name', '')} {best_sign}{best_pct:.2f}%\n\U0001f480 **最弱**: {worst.get('name', '')} {worst_sign}{worst_pct:.2f}%"})
    elements.append({"tag": "note", "elements": [{"tag": "plain_text", "content": f"\u66f4\u65b0: {datetime.now().strftime('%H:%M')} | \u26a0\ufe0f \u4e0d\u6784\u6210\u6295\u8d44\u5efa\u8bae"}]})

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"\U0001f4ca \u80a1\u7968\u5bf9\u6bd4 ({len(stocks)}\u53ea)"},
            "template": "blue"
        },
        "elements": elements
    }


def make_alert_card(alert_type: str, symbol: str, name: str, details: str) -> dict:
    """异动告警卡片"""
    template_map = {"大涨": "green", "大跌": "red", "放量": "orange", "异动": "violet"}

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"🚨 {alert_type}告警 - {name}"},
            "template": template_map.get(alert_type, "red")
        },
        "elements": [
            {"tag": "markdown", "content": f"**{name}** ({symbol})\n{details}"},
            {
                "tag": "note",
                "elements": [{"tag": "plain_text", "content": "实时监控告警"}]
            }
        ]
    }