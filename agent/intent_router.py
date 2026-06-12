#!/usr/bin/env python3
"""
意图路由器 v2 - 8类意图 + LLM 兜底

stock_brief  轻量行情    茅台 / 茅台行情 / 多少钱
stock_deep   深度分析    茅台分析 / 建议 / 补仓 / 怎么样
stock_news   股票新闻    茅台新闻 / 消息 / 利好 / 利空
portfolio    持仓        持仓 / 仓位 / 风控 / 信号
market       大盘        大盘 / 指数 / 板块 / 北向
compare      对比        茅台和五粮液对比 / 哪个好
help         帮助        帮助 / 功能 / 回测
chat         闲聊        总结 / 日报 / 其他
"""

import re
import os
import sys
import json
import logging
from typing import Dict, Optional, Tuple

PYTHON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python')
sys.path.insert(0, PYTHON_DIR)

from config import DB_PATH

logger = logging.getLogger("feishu_bot")

# ========== 股票代码映射 ==========

STOCK_NAME_MAP = {
    "茅台": "600519.SH", "贵州茅台": "600519.SH",
    "爱尔眼科": "300015.SZ",
    "招商银行": "600036.SH", "招行": "600036.SH",
    "平安": "601318.SH", "中国平安": "601318.SH",
    "五粮液": "000858.SZ",
    "宁德时代": "300750.SZ", "宁德": "300750.SZ",
    "比亚迪": "002594.SZ",
    "阿里": "9988.HK", "阿里巴巴": "9988.HK",
    "腾讯": "0700.HK", "腾讯控股": "0700.HK",
    "中芯国际": "00981.HK",
    "美团": "03690.HK", "美团-W": "03690.HK",
    "恒瑞医药": "600276.SH", "恒瑞": "600276.SH",
    "中免": "601888.SH", "中国中免": "601888.SH",
    "海天味业": "603288.SH", "海天": "603288.SH",
    "保利发展": "600048.SH", "保利": "600048.SH",
    "汇川技术": "300124.SZ", "汇川": "300124.SZ",
    "中国船舶": "600150.SH",
    "伊利": "600887.SH", "伊利股份": "600887.SH",
    "中国海油": "600938.SH", "海油": "600938.SH",
    "美的": "000333.SZ", "美的集团": "000333.SZ",
    "格力": "000651.SZ", "格力电器": "000651.SZ",
    "万科": "000002.SZ", "万科A": "000002.SZ",
    "隆基": "601012.SH", "隆基绿能": "601012.SH",
    "紫金矿业": "601899.SH", "紫金": "601899.SH",
    "阿里巴巴-W": "9988.HK",
    "港股通互联网ETF": "159792.SZ", "港股通ETF": "159792.SZ",
    "互联网ETF": "159792.SZ", "港股通互联网": "159792.SZ",
    "港股互联网": "159792.SZ",
}

# ========== 意图关键词 ==========

# 需要深度分析的问法（检测到这些 → stock_deep）
DEEP_KEYWORDS = [
    '分析', '深度', '诊断', '综合', '建议', '操作', '策略',
    '估值', '贵不贵', '值不值', '值不值买', '值不值得',
    '技术', '指标', '均线', 'macd', 'rsi', 'kdj', '布林', 'boll',
    '压力位', '支撑位', '技术分析', '技术面',
    '资金', '主力', '资金流向', '大单', '小单', '流入', '流出',
    '补仓', '加仓', '减仓', '清仓', '割肉', '跑吗', '该跑吗',
    '买吗', '卖吗', '可以买', '可以卖', '该买', '该卖', '要不要',
    '做t', '做T', '日内', '止损', '止盈',
    '怎么样', '如何', '怎么操作', '该怎么操作', '如何操作',
    '怎么买', '怎么卖', '好不好', '该不该', '值得',
    '基本面', 'pe', 'pb', 'roe', '财报', '财务',
    '消息面', '预测', '后市', '走势', '趋势',
    '目标价', '能涨', '能跌', '还会涨', '还会跌',
]

# 纯行情查询（检测到这些但无DEEP关键词 → stock_brief）
BRIEF_KEYWORDS = [
    '行情', '价格', '现价', '多少钱', '股价', '涨跌', '涨跌幅',
    '收盘', '开盘', '最高', '最低', '成交量', '成交额',
    '实时', '最新', '报价', '市值',
]

# 新闻关键词
NEWS_KEYWORDS = [
    '新闻', '消息', '资讯', '要闻', '利好', '利空',
    '异动', '预警', '告警', '盯盘',
]

# 持仓关键词
PORTFOLIO_KEYWORDS = [
    '持仓', '仓位', '持有', '我的股票', '我的持仓',
    '风控', '风险', '评分', '信号', '交易信号',
    '盈亏', '收益', '赚了', '亏了', '持仓概览',
]

# 大盘关键词
MARKET_KEYWORDS = [
    '大盘', '指数', '板块', '热门', '行业', '概念', '题材',
    '北向', '外资', '北向资金', '沪股通', '深股通',
    '港股', 'a股', '沪指', '深指', '创业板指', '恒生', '上证',
    '纳斯达克', '纳指', '道琼斯', '标普', '热搜',
    '领涨', '领跌', '涨幅榜', '跌幅榜',
]

# 对比关键词
COMPARE_KEYWORDS = [
    '对比', '比较', 'vs', '哪个好', '哪个更强', '哪个强', '比比',
    '比', '比一比', '比一下', '对比下', '比较下', '对比一下', '比较一下',
    'pk', '相比', '哪个更', '谁更', '孰优',
]

# 关系词（连接两个股票名，暗示对比意图）
RELATION_WORDS = ['和', '跟', '与', '同', '以及', '还有']

# 单独出现两字股票名也视为可能的对比倾向
# 例如："茅台五粮液" → 两个股票名连写 → compare

# 帮助关键词
HELP_KEYWORDS = [
    '帮助', 'help', '功能', '怎么用', '你能做什么',
    '回测', '测试策略', '策略测试',
    '自选', '关注', '添加自选', '删除自选', '取消关注',
]

# 闲聊关键词
CHAT_KEYWORDS = [
    '总结', '日报', '盘后', '今日总结', '今日行情',
    '你好', '谢谢', '再见', '早', '晚安',
]


def extract_symbol(text: str) -> Optional[str]:
    """从消息中提取股票代码"""
    # 1. 标准代码格式
    for p in [r'(\d{6}\.SZ)', r'(\d{6}\.SH)', r'(\d{4,5}\.HK)']:
        m = re.search(p, text)
        if m:
            return m.group(1)

    # 2. 纯数字代码
    num_match = re.search(r'(\d{6})', text)
    if num_match:
        code = num_match.group(1)
        if code.startswith(('0', '3', '1')):
            return f"{code}.SZ"
        elif code.startswith(('5', '6', '9')):
            return f"{code}.SH"

    # 3. 港股简码
    hk_match = re.search(r'(\d{4,5})\.HK', text)
    if hk_match:
        return hk_match.group(0)

    # 4. 中文简称
    for name, symbol in STOCK_NAME_MAP.items():
        if name in text:
            return symbol

    # 5. 数据库模糊查（词长度≥3才查，避免"港股""A股"等市场词误匹配）
    try:
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        for word in text.split():
            if len(word) >= 3:
                cursor.execute(
                    "SELECT symbol FROM stock_info WHERE name LIKE ?",
                    (f'%{word}%',)
                )
                row = cursor.fetchone()
                if row:
                    conn.close()
                    return row[0]
        conn.close()
    except Exception:
        pass

    return None


def _extract_multi_symbols(text: str) -> list:
    """从文本中提取所有股票代码（去重）"""
    symbols = []
    # 1. 标准代码格式
    for p in [r'(\d{6}\.SZ)', r'(\d{6}\.SH)', r'(\d{4,5}\.HK)']:
        for m in re.finditer(p, text):
            sym = m.group(1)
            if sym not in symbols:
                symbols.append(sym)
    # 2. 纯数字代码
    for m in re.finditer(r'\d{6}', text):
        code = m.group(0)
        if code.startswith(('0', '3', '1')):
            sym = f"{code}.SZ"
        elif code.startswith(('5', '6', '9')):
            sym = f"{code}.SH"
        else:
            continue
        if sym not in symbols:
            symbols.append(sym)
    # 3. 中文简称（按名称长度降序，避免短名误匹配长名的一部分）
    remaining = text
    for name, sym in sorted(STOCK_NAME_MAP.items(), key=lambda x: -len(x[0])):
        if name in remaining:
            symbols.append(sym)
            remaining = remaining.replace(name, '\n', 1)
    return symbols


def _has_any(text_lower: str, keywords: list) -> bool:
    """检查文本是否包含任一关键词"""
    return any(k in text_lower for k in keywords)


def classify_intent(text: str) -> Tuple[str, Dict]:
    """分类用户意图 → (intent, params)"""
    text_lower = text.lower().strip()

    # ===== 1. 帮助 =====
    if _has_any(text_lower, HELP_KEYWORDS):
        symbol = extract_symbol(text)
        if any(k in text_lower for k in ['回测', '测试策略', '策略测试']):
            return 'backtest', {'symbol': symbol}
        if any(k in text_lower for k in ['自选', '关注', '添加自选', '删除自选', '取消关注']):
            action = 'add' if any(k in text_lower for k in ['添加', '加', '关注', '自选']) else 'remove'
            name_match = re.search(r'自选\s+(\S+)', text) or re.search(r'关注\s+(\S+)', text)
            return 'help', {'action': action, 'symbol': symbol,
                           'name': name_match.group(1) if name_match else ''}
        return 'help', {}

    # ===== 2. 持仓 =====
    if _has_any(text_lower, PORTFOLIO_KEYWORDS):
        return 'portfolio', {}

    # ===== 3. 对比（先于个股判断，因为可能包含多个股票名） =====
    # 显式对比关键词
    if _has_any(text_lower, COMPARE_KEYWORDS):
        symbols = _extract_multi_symbols(text)
        return 'compare', {'symbols': symbols}

    # 关系词 + 多股票名（如：茅台和五粮液、茅台跟五粮液）
    if _has_any(text_lower, RELATION_WORDS):
        symbols = _extract_multi_symbols(text)
        if len(symbols) >= 2:
            return 'compare', {'symbols': symbols}

    # 多股票名连写（如：茅台五粮液）
    symbols = _extract_multi_symbols(text)
    if len(symbols) >= 2:
        # 剔除股票名后再检查深度/新闻关键词（避免"技术"在"汇川技术"中被误判）
        clean_text = text
        for name, sym in sorted(STOCK_NAME_MAP.items(), key=lambda x: -len(x[0])):
            if name in clean_text:
                clean_text = clean_text.replace(name, ' ', 1)
        if not _has_any(clean_text.lower(), DEEP_KEYWORDS + NEWS_KEYWORDS):
            return 'compare', {'symbols': symbols}

    # ===== 4. 个股相关（优先于大盘，避免"港股通互联网"被"港股"误抢） =====
    symbol = extract_symbol(text)
    if symbol:
        # 新闻 → stock_news
        if _has_any(text_lower, NEWS_KEYWORDS):
            return 'stock_news', {'symbol': symbol, 'keyword': text}
        # 深度分析 vs 轻量行情
        if _has_any(text_lower, DEEP_KEYWORDS):
            return 'stock_deep', {'symbol': symbol}
        return 'stock_brief', {'symbol': symbol}

    # ===== 5. 大盘 =====
    if _has_any(text_lower, MARKET_KEYWORDS):
        return 'market', {}

    # ===== 6. 新闻（无股票名） =====
    if _has_any(text_lower, NEWS_KEYWORDS):
        return 'market', {}

    # ===== 7. 闲聊（总结/日报等） =====
    if _has_any(text_lower, ['总结', '日报', '盘后', '今日总结', '今日行情']):
        return 'chat', {'raw_text': text}
        # 深度分析 vs 轻量行情
        if _has_any(text_lower, DEEP_KEYWORDS):
            return 'stock_deep', {'symbol': symbol}
        return 'stock_brief', {'symbol': symbol}

    # ===== 8. 兜底 =====
    if _has_any(text_lower, CHAT_KEYWORDS):
        return 'chat', {'raw_text': text}

    # 有内容但不是已知意图 → LLM 兜底
    if len(text_lower) >= 2:
        return 'chat', {'raw_text': text}

    return 'help', {}


def llm_classify(text: str, symbol: str = None) -> Tuple[str, Dict]:
    """
    LLM 兜底分类（当关键词匹配置信度低时调用）
    """
    try:
        from llm_client import _call_dashscope_chat, is_available
        if not is_available():
            return 'chat', {'raw_text': text}

        if symbol:
            prompt = f"""你是金融Bot意图分类器。用户发来一条消息，请判断意图，只能返回以下之一：
- stock_brief: 只是想看股价/行情（轻量）
- stock_deep: 想深度分析/操作建议/技术指标/估值（需要完整分析）
- stock_news: 想看新闻/消息面
- chat: 闲聊/其他

股票: {symbol}
用户消息: "{text}"

只返回JSON: {{"intent": "xxx", "reason": "一句话"}}"""
        else:
            prompt = f"""你是金融Bot意图分类器。判断用户意图，只能返回以下之一：
- portfolio: 持仓/仓位/风控相关
- market: 大盘/指数/板块/北向资金
- chat: 闲聊/其他

用户消息: "{text}"

只返回JSON: {{"intent": "xxx", "reason": "一句话"}}"""

        msgs = [
            {"role": "system", "content": "你是意图分类器，只返回JSON。"},
            {"role": "user", "content": prompt}
        ]
        result = _call_dashscope_chat(msgs, max_tokens=100, temperature=0.1)
        if result and '{' in result:
            start = result.index('{')
            end = result.rindex('}') + 1
            data = json.loads(result[start:end])
            intent = data.get('intent', 'chat')
            logger.info(f"LLM分类: {intent} ({data.get('reason', '')})")

            params = {'raw_text': text}
            if intent in ('stock_brief', 'stock_deep', 'stock_news'):
                s = symbol or extract_symbol(text)
                params['symbol'] = s
                if intent == 'stock_news':
                    params['keyword'] = text
            return intent, params
    except Exception as e:
        logger.warning(f"LLM分类失败: {e}")

    return 'chat', {'raw_text': text}