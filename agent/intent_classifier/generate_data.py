#!/usr/bin/env python3
"""
训练数据生成器 - 用百炼 LLM 构造意图分类数据集

用途: 为 8 类意图各生成 N 条多样化问法
输出: training_data.jsonl
"""

import json
import os
import sys
import time
import random

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(AGENT_DIR)
sys.path.insert(0, AGENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'python'))

from llm_client import _call_dashscope_chat

INTENTS = {
    "stock_brief": {
        "desc": "轻量行情查询 - 只想看股价/涨跌/行情，不需要分析",
        "examples": [
            "茅台", "茅台行情", "茅台多少钱", "茅台股价",
            "看看茅台", "查一下茅台价格", "茅台现价", "茅台涨了没",
            "茅台今天怎么样", "茅台涨跌", "茅台最新价",
            "300124行情", "159792价格", "汇川技术多少钱",
        ],
    },
    "stock_deep": {
        "desc": "深度分析 - 想知道能不能买/卖/持有，需要技术面/消息面/估值/操作建议",
        "examples": [
            "茅台分析", "茅台建议", "茅台怎么样", "茅台怎么操作",
            "茅台能买吗", "茅台该卖了吗", "茅台补仓",
            "茅台值得吗", "茅台会不会涨", "茅台后市",
            "分析一下茅台", "茅台技术面", "茅台基本面",
            "茅台可以加仓吗", "茅台要减仓吗",
        ],
    },
    "stock_news": {
        "desc": "股票新闻/消息查询",
        "examples": [
            "茅台新闻", "茅台消息", "茅台利好",
            "茅台有什么消息", "茅台最近新闻", "茅台资讯",
            "茅台有利好吗", "茅台最近有利空吗",
            "汇川技术有什么新闻", "茅台消息面",
        ],
    },
    "portfolio": {
        "desc": "持仓/仓位/风控/交易信号查询",
        "examples": [
            "持仓", "持仓概览", "我的持仓", "我的仓位",
            "持仓盈亏", "风控评分", "风险评估",
            "我的收益", "赚了多少", "亏了多少",
            "看看持仓", "交易信号", "有什么信号",
            "仓位情况", "持仓情况",
        ],
    },
    "market": {
        "desc": "大盘/指数/板块/北向资金查询",
        "examples": [
            "大盘", "今天大盘怎么样", "指数",
            "今天什么板块涨", "热门板块", "领涨板块",
            "北向资金", "北向流入还是流出", "外资",
            "A股行情", "港股行情", "创业板",
            "今天哪个板块好", "现在什么板块强",
        ],
    },
    "compare": {
        "desc": "多只股票对比",
        "examples": [
            "茅台和五粮液对比", "茅台五粮液哪个好",
            "比较下茅台五粮液", "茅台跟五粮液",
            "对比爱尔眼科和伊利", "茅台五粮液vs",
            "茅台五粮液比一比", "茅台和五粮液谁更强",
            "茅台和五粮液怎么样", "茅台和伊利",
        ],
    },
    "help": {
        "desc": "帮助/功能/回测",
        "examples": [
            "帮助", "怎么用", "能做什么", "功能",
            "回测茅台", "测试策略", "自选茅台",
            "添加自选", "你有什么功能", "help",
        ],
    },
    "chat": {
        "desc": "闲聊/总结/日报/其他非股票查询",
        "examples": [
            "总结", "今日总结", "今天怎么样", "日报",
            "盘后总结", "你好", "谢谢", 
            "今天赚了吗", "推荐什么股票",
            "讲个笑话", "现在几点了",
        ],
    },
}

SYSTEM_PROMPT = """你是金融Bot训练数据生成器。请为每类意图生成多样化的用户问法。

要求：
1. 每类意图生成 {per_intent} 条训练样本
2. 包含以下变体：
   - 不同股票名（茅台、五粮液、比亚迪、宁德、腾讯、平安、招行、恒瑞、美的、格力、汇川技术、保利发展、爱尔眼科、港股通互联网ETF、美团、阿里等）
   - 不同口语化表达（"咋样"→"怎么样"、"买不买"→"能买吗"、"瞅瞅"→"看看"）
   - 可能的打字错误/简称（"毛台"→茅台、"五粮"→五粮液）
   - 带语气词的（"啊"、"呢"、"吧"、"嘛"）
   - 连写无空格的（"茅台五粮液哪个好"）
3. 只有 compare 意图需要包含2只以上股票，其他意图一般1只或0只
4. 多样性第一，不要重复

返回格式：
{
  "stock_brief": ["问法1", "问法2", ...],
  "stock_deep": ["问法1", "问法2", ...],
  ...
}"""


def generate_intent_data(intent: str, info: dict, per_intent: int = 80) -> list:
    """为单个意图生成数据"""
    all_samples = []
    existing = set(info['examples'])

    prompt = f"""意图: {intent}
描述: {info['desc']}
已有示例: {json.dumps(info['examples'][:10], ensure_ascii=False)}

请生成 {per_intent} 条新的、多样化的用户问法。要求口语化、自然、有真实用户会说的感觉。
包含问句、陈述句、带语气词的、简称、简写的各种形式。
只返回JSON数组，不要其他内容：["问法1", "问法2", ...]"""

    try:
        msgs = [
            {"role": "system", "content": "你是训练数据生成器，只返回JSON数组。"},
            {"role": "user", "content": prompt},
        ]
        result = _call_dashscope_chat(msgs, max_tokens=2000, temperature=0.9)
        if result and '[' in result:
            start = result.index('[')
            end = result.rindex(']') + 1
            samples = json.loads(result[start:end])
            new_samples = [s for s in samples if s not in existing]
            print(f"  {intent}: 生成 {len(new_samples)} 条")
            return new_samples
    except Exception as e:
        print(f"  {intent}: 生成失败 - {e}")
    
    return []


def main():
    per_intent = 80
    total_per_intent = per_intent + 20  # 多生成一些，去重用
    
    all_data = []
    
    for intent, info in INTENTS.items():
        # 加入已有示例
        for ex in info['examples']:
            all_data.append({"text": ex, "intent": intent})
        
        # LLM 生成新样本
        new_samples = generate_intent_data(intent, info, total_per_intent)
        for s in new_samples:
            all_data.append({"text": s, "intent": intent})
        
        time.sleep(0.5)  # 避免 API 限流
    
    # 保存
    output_dir = os.path.join(AGENT_DIR, 'intent_classifier')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'training_data.jsonl')
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 统计
    counts = {}
    for item in all_data:
        counts[item['intent']] = counts.get(item['intent'], 0) + 1
    
    print(f"\n训练数据: {len(all_data)} 条")
    for intent, count in sorted(counts.items()):
        print(f"  {intent}: {count}")
    print(f"保存到: {output_path}")


if __name__ == '__main__':
    main()