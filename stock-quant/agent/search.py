#!/usr/bin/env python3
"""
搜索引擎 - 基于 DuckDuckGo HTML 版

DuckDuckGo HTML 版（html.duckduckgo.com）是最可靠的免费搜索接口：
- 不需要 API Key
- 不需要注册
- 反爬最弱（纯 HTML 页面）
- 支持中文搜索

用法：
  python search.py "AI金融助手功能"
  python search.py "股票异动监控 agent" --count 15
"""

import json
import re
import sys
import urllib.parse
from typing import List, Dict

import requests


def search(query: str, count: int = 10) -> List[Dict]:
    """搜索并返回结构化结果列表

    Args:
        query: 搜索关键词
        count: 结果数量（默认10，最大约30）

    Returns:
        [{"title": str, "url": str, "snippet": str}, ...]
    """
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        return [{"title": f"搜索失败: {e}", "url": "", "snippet": ""}]

    # 提取结果链接
    result_links = re.findall(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html
    )

    # 提取摘要
    snippets = re.findall(
        r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', html
    )

    results = []
    for i, (raw_url, raw_title) in enumerate(result_links[:count]):
        # 清理标题
        title = re.sub(r'<[^>]+>', '', raw_title).strip()

        # 解析 DDG 重定向 URL
        # DDG 链接格式: //duckduckgo.com/l/?uddg=https%3A%2F%2Freal-url&...
        url_match = re.search(r'uddg=([^&]+)', raw_url)
        if url_match:
            real_url = urllib.parse.unquote(url_match.group(1))
        elif raw_url.startswith("http"):
            real_url = raw_url
        else:
            real_url = ""

        # 清理摘要
        snippet = ""
        if i < len(snippets):
            snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip()[:200]

        if title and real_url:
            results.append({"title": title, "url": real_url, "snippet": snippet})

    return results


def search_and_format(query: str, count: int = 10) -> str:
    """搜索并格式化为可读文本"""
    results = search(query, count)
    if not results:
        return f"❌ 未找到 '{query}' 的搜索结果"

    output = f"🔍 搜索: {query}（{len(results)} 条结果）\n\n"
    for i, r in enumerate(results, 1):
        output += f"{i}. **{r['title']}**\n"
        output += f"   🔗 {r['url']}\n"
        if r['snippet']:
            output += f"   📝 {r['snippet']}\n"
        output += "\n"
    return output


def fetch_page(url: str, max_chars: int = 5000) -> str:
    """获取网页内容，提取正文文本"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        return f"获取失败: {e}"

    # 简单提取正文
    # 去掉 script/style
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
    html = re.sub(r'<[^>]+>', ' ', html)
    html = re.sub(r'\s+', ' ', html).strip()
    return html[:max_chars]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python search.py <查询关键词> [--count N]")
        print("示例: python search.py 'AI金融助手 功能'")
        sys.exit(1)

    query = sys.argv[1]
    count = 10
    for arg in sys.argv[2:]:
        if arg.startswith("--count"):
            count = int(arg.split("=")[1]) if "=" in arg else int(sys.argv[sys.argv.index(arg) + 1])

    print(search_and_format(query, count))