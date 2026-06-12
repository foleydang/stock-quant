#!/usr/bin/env python3
"""
训练数据扩充 - 用关键词路由自动标注更多数据
"""

import json, os, sys, random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(os.path.dirname(SCRIPT_DIR), '..', 'python'))

from intent_router import classify_intent

# ========== 多样化模板生成 ==========

# 股票名变体
STOCKS = ['茅台', '五粮液', '比亚迪', '宁德', '腾讯', '平安', '招行', '恒瑞', '美的', '格力', 
          '汇川技术', '保利发展', '爱尔眼科', '港股通互联网', '美团', '阿里', '伊利',
          '隆基', '万科', '海天', '中免', '紫金', '中国船舶', '中国海油']

# 各种口语化前后缀
PREFIXES = ['', '帮我看看', '查一下', '瞅瞅', '看看', '请问', '问一下', '想知道', '了解下']
SUFFIXES = ['', '啊', '呢', '吧', '嘛', '呗', '哈', '哦', '呀']
TYPO_VARIANTS = {
    '茅台': ['毛台', '矛台', '茅苔'], '五粮液': ['五粮', '5粮液'],
    '比亚迪': ['比压迪', 'BYD'], '宁德': ['宁得', '宁德时代'],
    '保利发展': ['保利', '保立'], '爱尔眼科': ['爱尔', '眼科'],
    '港股通互联网': ['港股通', '港股互联网', '互联ETF'],
}

# 意图扩展模板
INTENT_TEMPLATES = {
    'stock_brief': [
        "{stock}", "{stock}行情", "{stock}价格", "{stock}多少钱",
        "{stock}股价", "{stock}现价", "{prefix}{stock}{suffix}",
        "{stock}涨了没", "{stock}今天怎么样", "看看{stock}",
        "查一下{stock}价格", "{stock}最新价", "{stock}涨跌",
        "{stock}收盘价", "{stock}实时行情",
    ],
    'stock_deep': [
        "{stock}分析", "{stock}建议", "{stock}怎么样",
        "{stock}能买吗", "{stock}该卖吗", "{stock}补仓",
        "{stock}值得吗", "{stock}会不会涨", "{stock}后市",
        "分析一下{stock}", "{stock}技术面", "{stock}基本面",
        "{stock}可以加仓吗", "{stock}减仓", "{stock}怎么操作",
        "{stock}操作指南", "{stock}深度分析", "{stock}评估",
        "{stock}走势怎么样", "{stock}还能持有吗", "{stock}行情分析",
        "{stock}怎么看", "{stock}后市如何", "{stock}还能涨吗",
        "{stock}止损还是持有", "{stock}适合加仓吗",
        "{prefix}{stock}建议{suffix}", "给{stock}做个分析",
    ],
    'stock_news': [
        "{stock}新闻", "{stock}消息", "{stock}利好",
        "{stock}有什么消息", "{stock}最近新闻", "{stock}资讯",
        "{stock}有利好吗", "{stock}最近有利空吗", "{stock}消息面",
        "{prefix}{stock}新闻{suffix}", "查查{stock}新闻",
        "{stock}有什么大事", "{stock}出什么消息了",
    ],
    'portfolio': [
        "持仓", "持仓概览", "我的持仓", "我的仓位",
        "持仓盈亏", "风控评分", "风险评估", "我的收益",
        "赚了多少", "亏了多少", "看看持仓", "交易信号",
        "有什么信号", "仓位情况", "持仓情况", "持仓分析",
        "今天亏了多少", "今天赚了多少", "看看赚了多少",
        "{prefix}盈亏{suffix}", "我的股票怎么样",
    ],
    'market': [
        "大盘", "今天大盘怎么样", "指数", "板块",
        "今天什么板块涨", "热门板块", "领涨板块",
        "北向资金", "外资", "A股行情", "港股行情",
        "创业板", "今天哪个板块好", "现在什么板块强",
        "大盘分析", "市场怎么样", "今天行情",
        "{prefix}大盘{suffix}", "看看大盘",
    ],
    'compare': [
        "{s1}和{s2}对比", "{s1}{s2}哪个好", "比较下{s1}{s2}",
        "{s1}跟{s2}", "对比{s1}和{s2}", "{s1}{s2}谁更强",
        "{s1}{s2}比一比", "{s1}和{s2}怎么样", "{s1}和{s2}",
        "{s1}{s2}哪个更值得买", "{s1}和{s2}选哪个",
        "{s1}与{s2}", "{s1}VS{s2}", "vs {s1} {s2}",
        "比比{s1}和{s2}", "{s1}和{s2}对比下",
    ],
    'help': [
        "帮助", "怎么用", "能做什么", "功能", "help",
        "回测{s}", "测试策略", "自选{s}", "添加自选",
        "你有什么功能", "怎么回测", "怎么添加自选",
    ],
    'chat': [
        "总结", "今日总结", "日报", "盘后总结",
        "你好", "谢谢", "推荐什么", "讲个笑话",
        "今天行情总结", "每日总结", "复盘",
        "今天市场怎么样", "收盘总结",
    ],
}


def expand_templates():
    """用模板 + 随机替换生成多样本"""
    new_data = []
    existing = set()
    
    # 加载已有数据做去重
    data_path = os.path.join(SCRIPT_DIR, 'training_data.jsonl')
    if os.path.exists(data_path):
        with open(data_path) as f:
            for line in f:
                item = json.loads(line)
                existing.add(item['text'])
    
    for intent, templates in INTENT_TEMPLATES.items():
        for tmpl in templates:
            for _ in range(3):  # 每个模板生成3个变体
                s = tmpl
                # 替换股票名
                for stock_var in ['{stock}', '{s1}', '{s2}', '{s}']:
                    if stock_var in s:
                        base_stock = random.choice(STOCKS)
                        # 10%概率用打字错误
                        if random.random() < 0.1 and base_stock in TYPO_VARIANTS:
                            base_stock = random.choice(TYPO_VARIANTS[base_stock])
                        s = s.replace(stock_var, base_stock, 1)
                
                # 替换前缀后缀
                s = s.replace('{prefix}', random.choice(PREFIXES))
                s = s.replace('{suffix}', random.choice(SUFFIXES))
                
                # 随机加语气词
                if random.random() < 0.15 and '{suffix}' not in tmpl:
                    s += random.choice(SUFFIXES)
                
                if s not in existing:
                    existing.add(s)
                    new_data.append({"text": s, "intent": intent})
    
    # 用关键词路由验证标注
    verified = []
    for item in new_data:
        intent, _ = classify_intent(item['text'])
        # 如果路由一致，接受
        if intent == item['intent'] or (item['intent'] == 'help' and intent in ('help', 'backtest')):
            verified.append(item)
    
    print(f"模板生成: {len(new_data)}, 路由验证通过: {len(verified)}")
    return verified


def main():
    random.seed(42)
    
    # 1. 加载已有数据
    data_path = os.path.join(SCRIPT_DIR, 'training_data.jsonl')
    all_data = []
    if os.path.exists(data_path):
        with open(data_path) as f:
            for line in f:
                all_data.append(json.loads(line))
    print(f"已有数据: {len(all_data)} 条")
    
    # 2. 模板扩充
    new_data = expand_templates()
    all_data.extend(new_data)
    
    # 3. 去重 + 限制数量
    seen = set()
    unique = []
    for item in all_data:
        key = item['text']
        if key not in seen:
            seen.add(key)
            unique.append(item)
    
    # 每类最多 400 条，保证均衡
    from collections import Counter
    counts = Counter()
    balanced = []
    for item in unique:
        if counts[item['intent']] < 400:
            balanced.append(item)
            counts[item['intent']] += 1
    
    # 保存
    with open(data_path, 'w', encoding='utf-8') as f:
        for item in balanced:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n最终数据: {len(balanced)} 条")
    for intent, count in sorted(counts.items()):
        print(f"  {intent}: {count}")


if __name__ == '__main__':
    main()