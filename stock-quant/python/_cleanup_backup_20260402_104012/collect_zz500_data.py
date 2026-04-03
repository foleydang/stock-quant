#!/usr/bin/env python3
"""
后台数据收集脚本
继续收集中证 500 成分股历史数据，以训练更强大的 ML 模型

用法：
    nohup python3 strategy/collect_zz500_data.py > logs/collect_zz500.log 2>&1 &
"""

import os
import sys
import time
import pickle
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 使用封装好的 DataHandler（内置随机延时，避免被封）
from data.data_handler import DataHandler

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("错误：akshare 未安装")
    sys.exit(1)


def fetch_zz500_constituents() -> List[Dict]:
    """获取中证 500 成分股列表"""
    try:
        # 尝试使用 akshare 接口（延时在 DataHandler 内部）
        # 获取成分股列表不需要频繁调用，直接用 akshare
        df = ak.index_stock_cons(symbol="000905")
        constituents = []
        for _, row in df.iterrows():
            code = row.get('品种代码', '')
            name = row.get('品种名称', '')
            # 转换为标准格式
            if code.startswith('6'):
                symbol = f"{code}.SH"
            elif code.startswith('0') or code.startswith('3'):
                symbol = f"{code}.SZ"
            else:
                symbol = code  # 其他情况保留原格式
            constituents.append({
                'symbol': symbol,
                'code': code,
                'name': name
            })
        print(f"获取到 {len(constituents)} 只中证 500 成分股")
        return constituents
    except Exception as e:
        print(f"获取成分股失败：{e}")
        # 使用备用方法：通过东方财富获取
        try:
            import urllib.request
            import ssl
            # 忽略SSL验证
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            url = "http://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=500&po=1&np=1&ut=b28f4f5d3b1b4b6&fltt=2&invt=2&fid=f3&fs=b:BK0500&fields=f12,f14"
            response = urllib.request.urlopen(url, context=ssl_context, timeout=30)
            data = response.read().decode('utf-8')
            import json
            json_data = json.loads(data)
            if json_data.get('data') and json_data['data'].get('diff'):
                constituents = []
                for item in json_data['data']['diff']:
                    code = item.get('f12', '')
                    name = item.get('f14', '')
                    # 转换为标准格式
                    if code.startswith('6'):
                        symbol = f"{code}.SH"
                    elif code.startswith('0') or code.startswith('3'):
                        symbol = f"{code}.SZ"
                    else:
                        symbol = code
                    constituents.append({
                        'symbol': symbol,
                        'code': code,
                        'name': name
                    })
                print(f"通过东方财富获取到 {len(constituents)} 只成分股")
                return constituents
        except Exception as e2:
            print(f"备用方法也失败：{e2}")
        return []


def fetch_stock_history(symbol: str, code: str, handler: DataHandler) -> Optional[pd.DataFrame]:
    """
    获取股票历史数据，使用封装好的 DataHandler（内置随机延时）

    Args:
        symbol: 股票代码（标准格式，如 '300015.SZ'）
        code: 6位股票代码
        handler: DataHandler 实例（内置限流和重试）
    """
    try:
        # 使用 DataHandler 获取数据（延时和重试在内部处理）
        df = handler.fetch_stock_data(symbol, force_refresh=True)

        if df is None or df.empty:
            return None

        df = df.sort_values('date').reset_index(drop=True)
        return df

    except Exception as e:
        print(f"获取失败 {symbol}: {e}")
        return None


def collect_batch(constituents: List[Dict], batch_size: int = 50,
                  cache_dir: str = None, delay: float = 2.0):
    """
    批量收集数据（使用 DataHandler，延时已在接口层内部处理）

    Args:
        constituents: 成分股列表
        batch_size: 每批处理数量
        cache_dir: 缓存目录
        delay: 请求间隔（秒）- 注意：DataHandler 内部已有限流，此参数仅作为额外缓冲
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), '../data/zz500_cache')

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # 创建 DataHandler 实例（内置随机延时）
    handler = DataHandler(force_refresh=True)

    # 过滤只保留A股（港股分钟数据暂时不可用）
    a_stocks = [s for s in constituents if s['symbol'].endswith('.SH') or s['symbol'].endswith('.SZ')]
    print(f"中证500共 {len(constituents)} 只，A股 {len(a_stocks)} 只（港股分钟数据暂不可用，跳过）")

    total = len(a_stocks)
    processed = 0
    success = 0
    failed = 0
    skipped = 0

    print(f"开始收集数据，共 {total} 只A股，每批 {batch_size} 只...")
    print(f"缓存目录：{cache_dir}")
    print(f"数据周期：30分钟")
    print(f"数据源：新浪财经接口（更稳定）")
    print("-" * 60)

    # 加载已完成的进度
    progress_file = os.path.join(cache_dir, 'progress.json')
    completed_symbols = set()
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                completed_symbols = set(progress.get('completed', []))
                print(f"已完成 {len(completed_symbols)} 只股票，将继续剩余下载")
        except:
            pass

    for i in range(0, total, batch_size):
        batch = a_stocks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size

        print(f"\n批次 {batch_num}/{total_batches}")

        for stock in batch:
            symbol = stock['symbol']
            code = stock.get('code', symbol[:6] if '.' in symbol else symbol)
            name = stock.get('name', '')

            # 检查是否已完成
            if symbol in completed_symbols:
                skipped += 1
                continue

            cache_file = os.path.join(cache_dir, f"{symbol}.pkl")

            # 检查已有缓存是否有效
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        df = pickle.load(f)
                    min_records = 100
                    if len(df) >= min_records:
                        processed += 1
                        success += 1
                        completed_symbols.add(symbol)
                        continue
                    else:
                        os.remove(cache_file)
                except:
                    pass

            # 获取数据（使用 DataHandler，新浪接口）
            idx = processed + skipped + 1
            print(f"[{idx}/{total}] {name} ({symbol})...", end=" ", flush=True)

            df = fetch_stock_history(symbol, code, handler)

            if df is not None and len(df) >= 100:
                with open(cache_file, 'wb') as f:
                    pickle.dump(df, f)
                print(f"✓ {len(df)}条")
                success += 1
                completed_symbols.add(symbol)
            else:
                print(f"✗ ({len(df) if df else 0}条)")
                failed += 1

            processed += 1

            # 保存进度
            with open(progress_file, 'w') as f:
                json.dump({'completed': list(completed_symbols)}, f)

            # 短暂缓冲
            time.sleep(delay)

            # 每10只休息一下
            if success > 0 and success % 10 == 0:
                print(f"  已成功 {success} 只，休息5秒...")
                time.sleep(5)

        # 批次间休息
        print(f"\n批次完成")
        print(f"累计：{success} 成功，{failed} 失败，{skipped} 跳过")
        time.sleep(10)

    print("\n" + "=" * 60)
    print(f"收集完成！总计：{success} 成功，{failed} 失败，{skipped} 跳过")
    print(f"缓存目录：{cache_dir}")
    print("=" * 60)

    return success, failed


def main():
    """主函数"""
    print("=" * 60)
    print("中证 500 成分股数据收集脚本 - 30 分钟级别")
    print("=" * 60)
    print("使用 DataHandler 接口（内置随机延时，避免被封禁）")

    if not AKSHARE_AVAILABLE:
        print("akshare 未安装，退出")
        return

    # 获取成分股
    constituents = fetch_zz500_constituents()
    if not constituents:
        print("无法获取成分股列表")
        return

    # 开始收集（增加延时到5秒）
    collect_batch(constituents, batch_size=10, delay=5.0)

    print("\n提示：可以使用以下命令重新训练模型:")
    print("  python3 strategy/train_lgb_30m_full.py")


if __name__ == "__main__":
    main()
