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

    # 2. 匹配纯数字代码
    num_match = re.search(r'(\d{6})', text)
    if num_match:
        code = num_match.group(1)
        if code.startswith(('0', '3')):
            return f"{code}.SZ"
        elif code.startswith(('6', '9')):
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
        return 't_strategy', {}

    if any(k in text_lower for k in ['回测', '测试策略', '策略测试']):
        return 'backtest', {'symbol': extract_symbol(text)}

    if any(k in text_lower for k in ['信号', '交易信号', '买卖信号', '操作建议']):
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

    # 行情/价格（带明确关键词才走stock）
    if any(k in text_lower for k in ['行情', '价格', '现价', '多少钱', '股价', '涨跌']):
        return 'stock', {'symbol': extract_symbol(text)}

    # ★★★ 以下必须在 stock 默认匹配之前 ★★★

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
