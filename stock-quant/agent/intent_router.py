#!/usr/bin/env python3
"""
意图路由器 - 根据用户消息决定执行什么动作

关键词路由 + LLM 辅助
"""

import re
import os
import sys
from typing import Dict, Optional, Tuple

PYTHON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python')
sys.path.insert(0, PYTHON_DIR)

from config import DB_PATH


# ========== 股票代码映射 ==========

STOCK_NAME_MAP = {
    "茅台": "600519.SH",
    "贵州茅台": "600519.SH",
    "爱尔眼科": "300015.SZ",
    "招商银行": "600036.SH",
    "招行": "600036.SH",
    "平安": "601318.SH",
    "中国平安": "601318.SH",
    "五粮液": "000858.SZ",
    "宁德时代": "300750.SZ",
    "宁德": "300750.SZ",
    "比亚迪": "002594.SZ",
    "阿里": "9988.HK",
    "阿里巴巴": "9988.HK",
    "腾讯": "0700.HK",
    "腾讯控股": "0700.HK",
    "中芯国际": "00981.HK",
    "美团": "03690.HK",
    "恒瑞医药": "600276.SH",
    "恒瑞": "600276.SH",
    "中免": "601888.SH",
    "中国中免": "601888.SH",
    "海天味业": "603288.SH",
    "海天": "603288.SH",
    "保利发展": "600048.SH",
    "保利": "600048.SH",
    "汇川技术": "300124.SZ",
    "汇川": "300124.SZ",
    "中国船舶": "600150.SH",
    "伊利": "600887.SH",
    "伊利股份": "600887.SH",
    "中国海油": "600938.SH",
    "海油": "600938.SH",
    "美的": "000333.SZ",
    "美的集团": "000333.SZ",
    "格力": "000651.SZ",
    "格力电器": "000651.SZ",
    "万科": "000002.SZ",
    "万科A": "000002.SZ",
    "隆基": "601012.SH",
    "隆基绿能": "601012.SH",
    "紫金矿业": "601899.SH",
    "紫金": "601899.SH",
    "美团-W": "03690.HK",
    "阿里巴巴-W": "9988.HK",
    "港股通互联网ETF": "159792.SZ",
    "港股通ETF": "159792.SZ",
    "互联网ETF": "159792.SZ",
    "港股通互联网": "159792.SZ",
    "港股互联网": "159792.SZ",
}


def extract_symbol(text: str) -> Optional[str]:
    """从消息中提取股票代码"""
    # 1. 直接匹配标准代码格式
    patterns = [
        r'(\d{6}\.SZ)',
        r'(\d{6}\.SH)',
        r'(\d{4,5}\.HK)',
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1)

    # 2. 匹配纯数字代码（含ETF：15xxxx, 51xxxx, 50xxxx）
    num_match = re.search(r'(\d{6})', text)
    if num_match:
        code = num_match.group(1)
        if code.startswith(('0', '3', '1')):  # 0/3=深市个股, 1=深市ETF
            return f"{code}.SZ"
        elif code.startswith(('5', '6', '9')):  # 5=沪市ETF, 6/9=沪市个股
            return f"{code}.SH"

    # 3. 匹配港股简码
    hk_match = re.search(r'(\d{4,5})\.HK', text)
    if hk_match:
        return hk_match.group(0)

    # 4. 匹配中文简称
    for name, symbol in STOCK_NAME_MAP.items():
        if name in text:
            return symbol

    # 5. 查数据库
    try:
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        for word in text.split():
            if len(word) >= 2:
                cursor.execute("SELECT symbol FROM stock_info WHERE name LIKE ?", (f'%{word}%',))
                row = cursor.fetchone()
                if row:
                    conn.close()
                    return row[0]
        conn.close()
    except Exception:
        pass

    return None


def classify_intent(text: str) -> Tuple[str, Dict]:
    """分类用户意图"""
    text_lower = text.lower().strip()

    if any(k in text_lower for k in ['帮助', 'help', '你能做什么', '功能', '怎么用']):
        return 'help', {}

    if any(k in text_lower for k in ['持仓', '仓位', '持仓概览', '我的股票', '持有']):
        return 'positions', {}

    if any(k in text_lower for k in ['做t', 't操作', '做t建议', '日内', '做T']):
        return 't_strategy', {'symbol': extract_symbol(text)}

    # 风控评分
    if any(k in text_lower for k in ['风险', '风控', '评分', '危险', '安全']):
        return 'risk', {'symbol': extract_symbol(text)}

    # 综合操作建议
    if any(k in text_lower for k in ['建议', '操作建议', '该怎么办', '综合建议', '补仓', '加仓', '减仓', '清仓', '割肉', '跑吗', '买吗', '卖吗', '需要补', '需要加', '可以买', '该买', '该卖', '要不要', '怎么操作', '该怎么操作', '如何操作', '该怎么']):
        return 'recommend', {'symbol': extract_symbol(text)}

    # 财经要闻
    if any(k in text_lower for k in ['新闻', '要闻', '资讯']):
        keyword = text.replace('新闻', '').replace('要闻', '').replace('资讯', '').strip()
        return 'news', {'keyword': keyword}

    # 估值判断
    if text_lower.strip() in ['估值', '估值判断', '贵不贵', '便宜吗'] or (any(k in text_lower for k in ['估值', '贵不贵', '值不值']) and not any(k in text_lower for k in ['深度', '分析', 'pe', 'pb', 'roe', '基本面'])):
        symbol = extract_symbol(text)
        return 'valuation', {'symbol': symbol}

    if any(k in text_lower for k in ['回测', '测试策略', '策略测试']):
        return 'backtest', {'symbol': extract_symbol(text)}

    if any(k in text_lower for k in ['信号', '交易信号', '买卖信号']):
        return 'signals', {}

    if any(k in text_lower for k in ['总结', '日报', '盘后', '今日总结', '今日行情']):
        return 'summary', {}

    if any(k in text_lower for k in ['自选', '关注', '添加自选', '删除自选', '取消关注']):
        symbol = extract_symbol(text)
        action = 'add' if any(k in text_lower for k in ['添加', '加', '关注']) else 'remove'
        name_match = re.search(r'自选\s+(\S+)', text) or re.search(r'关注\s+(\S+)', text)
        return 'watchlist', {'action': action, 'symbol': symbol, 'name': name_match.group(1) if name_match else ''}

    if any(k in text_lower for k in ['分析', '综合分析', '诊断', '深度分析']):
        return 'analyze', {'symbol': extract_symbol(text)}

    # 资金流向
    if any(k in text_lower for k in ['资金', '资金流向', '主力资金', '大单', '小单', '流入', '流出']):
        return 'money_flow', {'symbol': extract_symbol(text)}

    # 深度数据
    if any(k in text_lower for k in ['深度', '深度分析', 'pe', 'pb', 'roe', '基本面', '财报']):
        return 'deep', {'symbol': extract_symbol(text)}

    # 北向资金
    if any(k in text_lower for k in ['北向', '北向资金', '沪股通', '深股通', '外资']):
        return 'north_flow', {}

    # 技术指标
    if any(k in text_lower for k in ['指标', '技术指标', '技术分析', '均线', 'macd', 'rsi', 'kdj', '布林', 'boll', '压力位', '支撑位']):
        return 'technical', {'symbol': extract_symbol(text)}

    # 异动检测
    if any(k in text_lower for k in ['异动', '异动监控', '预警', '告警', '警报', '提醒', '盯盘']):
        return 'alert', {}

    # 止损止盈
    if any(k in text_lower for k in ['止损', '止盈', '止损价', '止盈价']):
        symbol = extract_symbol(text)
        # 尝试提取价格
        price_match = re.search(r'(\d+\.?\d*)', text)
        price = float(price_match.group(1)) if price_match else None
        action = 'stop_loss' if '止损' in text_lower else 'take_profit'
        return 'stop_alert', {'symbol': symbol, 'action': action, 'price': price}

    # 行情/价格（带明确关键词才走stock）
    if any(k in text_lower for k in ['行情', '价格', '现价', '多少钱', '股价', '涨跌']):
        return 'stock', {'symbol': extract_symbol(text)}

    # ★★★ 以下必须在 stock 默认匹配之前 ★★★

    # 先检查是否包含已知股票/ETF名称（避免被大盘等规则误抢）
    # 只在文本较短（<=15字）且明确是股票名/代码时才抢占
    stock_symbol = extract_symbol(text)
    if stock_symbol:
        # 检查symbol是否确实匹配了文本中的内容（而不是数据库模糊匹配的误杀）
        is_explicit = False
        for name in STOCK_NAME_MAP:
            if name in text:
                is_explicit = True
                break
        # 或纯数字代码
        import re
        if re.search(r'\d{6}', text):
            is_explicit = True
        # 或ETF关键词
        if 'ETF' in text or 'etf' in text.lower():
            is_explicit = True

        if is_explicit and not any(k in text_lower for k in ['指标', '技术', '分析', '做t', '建议', '风控', '估值', '深度', '资金', '新闻', '自选', '止损', '异动', '板块', '北向', '对比', '信号', '总结', '持仓', '回测']):
            return 'stock', {'symbol': stock_symbol}

    # 大盘指数
    if any(k in text_lower for k in ['大盘', '指数', 'a股', '港股', '沪指', '深指', '创业板指', '恒生', '上证', '纳斯达克', '纳指', '道琼斯', '标普']):
        return 'market', {}

    # 热门板块
    if any(k in text_lower for k in ['板块', '热门板块', '行业', '概念', '题材', '领涨板块']):
        return 'sector', {}

    # 多股对比
    if any(k in text_lower for k in ['对比', '比较', 'vs', '哪个好', '哪个更强', '比比']):
        symbols = []
        names_found = []
        remaining = text
        while remaining:
            sym = extract_symbol(remaining)
            if sym and sym not in symbols:
                symbols.append(sym)
                found = False
                for name, s in STOCK_NAME_MAP.items():
                    if s == sym and name in remaining:
                        remaining = remaining.replace(name, '', 1)
                        names_found.append(name)
                        found = True
                        break
                if not found:
                    remaining = remaining.replace(sym, '', 1)
            else:
                break
        return 'compare', {'symbols': symbols, 'names': names_found}

    # 兜底：包含股票代码 → stock
    symbol = extract_symbol(text)
    if symbol:
        return 'stock', {'symbol': symbol}

    # 兜底：自由对话
    return 'chat', {'raw_text': text}
