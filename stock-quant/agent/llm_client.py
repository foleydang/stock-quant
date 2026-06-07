#!/usr/bin/env python3
"""
LLM 客户端 - 百炼 DashScope API

使用 OpenAI 兼容接口（更快、支持流式）
"""

import json
import logging
import urllib.request
from typing import Dict, Optional

from config import DASHSCOPE_API_KEY, DASHSCOPE_MODEL

logger = logging.getLogger("feishu_bot")

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _call_dashscope_chat(messages: list, model: str = None, max_tokens: int = 1024, temperature: float = 0.7) -> Optional[str]:
    """调用 DashScope OpenAI兼容接口"""
    try:
        url = f"{DASHSCOPE_BASE_URL}/chat/completions"
        payload = {
            "model": model or DASHSCOPE_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json",
        })
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read().decode())
        return result['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"DashScope Chat API 调用异常: {e}")
        return None


def understand_intent(text: str) -> Dict:
    """用 LLM 理解用户意图"""
    system_content = (
        "你是金融助手的意图识别模块。判断用户意图并提取参数，只返回JSON。\n"
        "意图列表：\n"
        "- positions: 查看持仓\n"
        "- stock: 查看个股行情\n"
        "- t_strategy: 做T建议\n"
        "- backtest: 运行回测\n"
        "- signals: 交易信号\n"
        "- summary: 盘后总结\n"
        "- analyze: 综合分析\n"
        "- watchlist: 自选股管理\n"
        "- market: 大盘指数行情\n"
        "- sector: 热门板块\n"
        "- compare: 多股对比\n"
        "- help: 帮助\n"
        "- chat: 自由对话\n"
        '返回格式: {"intent": "意图", "symbol": "股票代码", "params": {"其他参数"}}'
    )
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": text}
    ]
    result = _call_dashscope_chat(messages, temperature=0.1)
    if result:
        try:
            if '{' in result:
                start = result.index('{')
                end = result.rindex('}') + 1
                return json.loads(result[start:end])
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"LLM 意图解析失败: {e}, raw: {result}")
    return {"intent": "chat", "symbol": None, "params": {"raw_text": text}}


def chat_response(text: str, context: dict = None) -> str:
    """自由对话 - 用 LLM 回答金融相关问题"""
    from datetime import datetime
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    context_str = ""
    if context:
        if 'positions' in context:
            context_str += "\n当前持仓: " + json.dumps(context['positions'][:5], ensure_ascii=False)
        if 'stock' in context:
            context_str += "\n当前行情: " + json.dumps(context['stock'], ensure_ascii=False)

    system_prompt = (
        "你是金融分析助手金融小助手，专业、简洁、有观点。\n"
        "核心原则：\n"
        "1. 回答简洁，不超过3段\n"
        "2. 对金融问题给出明确观点，不回避\n"
        "3. 非金融问题可以回答，但要提醒这不是你的专长\n"
        "4. 每次回复风格和内容都要不同，避免模板化\n"
        "5. 适当使用emoji让回复更生动\n"
        "6. 风险提醒用⚠️标注\n"
        f"当前时间：{now}\n"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{text}\n{context_str if context_str else ''}"}
    ]

    result = _call_dashscope_chat(messages, max_tokens=800, temperature=0.8)
    logger.info(f"Chat LLM 回复: {(result or 'fallback')[:200]}")
    return result or "抱歉，我暂时无法回答这个问题。试试发送「帮助」查看我能做什么"


def analyze_news_sentiment(news_list: list) -> Dict:
    """分析新闻情绪"""
    if not news_list:
        return {"sentiment": "无数据", "score": 0, "summary": "暂无新闻"}

    news_text = "\n".join([f"- [{n['time']}] {n['title']}: {n['content'][:100]}" for n in news_list[:10]])
    messages = [
        {"role": "system", "content": "分析财经新闻的市场情绪倾向。只返回JSON。"},
        {"role": "user", "content": news_text}
    ]
    result = _call_dashscope_chat(messages, temperature=0.3)
    if result:
        try:
            if '{' in result:
                start = result.index('{')
                end = result.rindex('}') + 1
                return json.loads(result[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
    return {"sentiment": "中性", "score": 0.5, "summary": "无法分析情绪"}


def is_available() -> bool:
    """检查百炼 API 是否可用"""
    return bool(DASHSCOPE_API_KEY)