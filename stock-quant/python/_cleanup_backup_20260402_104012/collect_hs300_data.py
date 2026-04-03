#!/usr/bin/env python3
"""
沪深300成分股数据收集脚本
特点：
1. 更稳健的避让策略（长延时、随机化）
2. 断点续传
3. 仅使用 akshare 新浪接口

用法：
    nohup python3 strategy/collect_hs300_data.py > logs/collect_hs300.log 2>&1 &
"""

import os
import sys
import time
import pickle
import json
import random
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("错误：akshare 未安装")
    sys.exit(1)


class RobustDataCollector:
    """稳健的数据收集器"""

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), '../data/hs300_cache')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.progress_file = os.path.join(self.cache_dir, 'progress.json')

        # 避让参数
        self.base_delay = 8.0  # 基础延时8秒
        self.max_delay = 30.0  # 最大延时30秒
        self.batch_break = 60.0  # 批次间休息60秒

        # 统计
        self.success_count = 0
        self.fail_count = 0
        self.consecutive_fails = 0

    def get_hs300_constituents(self) -> List[Dict]:
        """获取沪深300成分股"""
        print("获取沪深300成分股列表...")

        try:
            df = ak.index_stock_cons_weight_csindex(symbol="000300")
            constituents = []
            for _, row in df.iterrows():
                code = str(row.get('成分券代码', ''))
                name = row.get('成分券名称', '')
                if code:
                    # 转换格式
                    if code.startswith('6'):
                        symbol = f"{code}.SH"
                    else:
                        symbol = f"{code}.SZ"
                    constituents.append({
                        'symbol': symbol,
                        'code': code,
                        'name': name
                    })
            print(f"获取到 {len(constituents)} 只沪深300成分股")
            return constituents
        except Exception as e:
            print(f"获取成分股失败: {e}")
            # 备用：手动获取常见沪深300成分股
            return self._get_backup_constituents()

    def _get_backup_constituents(self) -> List[Dict]:
        """备用成分股列表"""
        stocks = [
            ('600519.SH', '贵州茅台'), ('000858.SZ', '五粮液'), ('601318.SH', '中国平安'),
            ('600036.SH', '招商银行'), ('601166.SH', '兴业银行'), ('600000.SH', '浦发银行'),
            ('601398.SH', '工商银行'), ('601288.SH', '农业银行'), ('600030.SH', '中信证券'),
            ('600276.SH', '恒瑞医药'), ('000333.SZ', '美的集团'), ('000651.SZ', '格力电器'),
            ('600887.SH', '伊利股份'), ('601888.SH', '中国中免'), ('600009.SH', '上海机场'),
            ('601012.SH', '隆基绿能'), ('600900.SH', '长江电力'), ('601818.SH', '光大银行'),
            ('601939.SH', '建设银行'), ('601988.SH', '中国银行'), ('600048.SH', '保利发展'),
            ('601628.SH', '中国人寿'), ('601601.SH', '中国太保'), ('601336.SH', '新华保险'),
            ('600585.SH', '海螺水泥'), ('600309.SH', '万华化学'), ('600346.SH', '恒力石化'),
            ('000002.SZ', '万科A'), ('000001.SZ', '平安银行'), ('002594.SZ', '比亚迪'),
            ('002475.SZ', '立讯精密'), ('002415.SZ', '海康威视'), ('300750.SZ', '宁德时代'),
            ('300015.SZ', '爱尔眼科'), ('300124.SZ', '汇川技术'), ('600309.SH', '万华化学'),
        ]
        return [{'symbol': s[0], 'code': s[0][:6], 'name': s[1]} for s in stocks]

    def load_progress(self) -> set:
        """加载进度"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('completed', []))
            except:
                pass
        return set()

    def save_progress(self, completed: set):
        """保存进度"""
        with open(self.progress_file, 'w') as f:
            json.dump({'completed': list(completed), 'updated': datetime.now().isoformat()}, f)

    def smart_delay(self):
        """智能延时：根据连续失败次数动态调整"""
        # 基础延时 + 随机抖动
        delay = self.base_delay + random.uniform(2, 8)

        # 连续失败时增加延时
        if self.consecutive_fails > 0:
            extra = min(self.consecutive_fails * 5, self.max_delay - self.base_delay)
            delay += extra
            print(f"    [连续失败{self.consecutive_fails}次，延时增加到{delay:.1f}秒]")

        # 随机化，避免规律性
        delay += random.uniform(-2, 5)

        time.sleep(max(delay, 3))

    def fetch_stock_data(self, symbol: str, code: str) -> Optional[pd.DataFrame]:
        """获取单只股票数据"""
        try:
            # 智能延时
            self.smart_delay()

            # 新浪接口格式
            market = 'sh' if code.startswith('6') else 'sz'
            sina_code = f"{market}{code}"

            df = ak.stock_zh_a_minute(symbol=sina_code, period='30')

            if df is not None and not df.empty:
                df = df.rename(columns={'day': 'date'})
                df['date'] = pd.to_datetime(df['date'])
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                df = df.sort_values('date').reset_index(drop=True)

                self.consecutive_fails = 0  # 重置连续失败计数
                return df

        except Exception as e:
            self.consecutive_fails += 1
            # 不打印错误，避免日志过多

        return None

    def collect_all(self, constituents: List[Dict], batch_size: int = 5):
        """收集所有数据"""
        total = len(constituents)
        completed = self.load_progress()

        print(f"\n开始收集沪深300数据")
        print(f"总数: {total}, 已完成: {len(completed)}, 待收集: {total - len(completed)}")
        print(f"批次大小: {batch_size}, 基础延时: {self.base_delay}秒")
        print("-" * 60)

        for i in range(0, total, batch_size):
            batch = constituents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total + batch_size - 1) // batch_size

            print(f"\n{'='*60}")
            print(f"批次 {batch_num}/{total_batches}")

            for stock in batch:
                symbol = stock['symbol']
                code = stock['code']
                name = stock['name']

                # 跳过已完成的
                if symbol in completed:
                    print(f"  [跳过] {name} ({symbol}) - 已完成")
                    continue

                # 获取数据
                cache_file = os.path.join(self.cache_dir, f"{symbol}.pkl")

                # 检查缓存
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'rb') as f:
                            df = pickle.load(f)
                        if len(df) >= 500:  # 至少500条数据
                            completed.add(symbol)
                            self.save_progress(completed)
                            print(f"  [缓存] {name}: {len(df)}条")
                            continue
                    except:
                        pass

                # 下载
                print(f"  [下载] {name} ({symbol})...", end=" ", flush=True)
                df = self.fetch_stock_data(symbol, code)

                if df is not None and len(df) >= 100:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(df, f)
                    completed.add(symbol)
                    self.save_progress(completed)
                    self.success_count += 1
                    print(f"✓ {len(df)}条")
                else:
                    self.fail_count += 1
                    print(f"✗ 失败")

            # 批次间休息
            print(f"\n批次完成 | 成功: {self.success_count}, 失败: {self.fail_count}")
            if batch_num < total_batches:
                print(f"休息 {self.batch_break} 秒...")
                time.sleep(self.batch_break)

        print("\n" + "=" * 60)
        print(f"收集完成！成功: {self.success_count}, 失败: {self.fail_count}")
        print(f"缓存目录: {self.cache_dir}")
        print("=" * 60)


def main():
    print("=" * 60)
    print("沪深300成分股数据收集脚本 - 30分钟级别")
    print("=" * 60)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    collector = RobustDataCollector()

    # 获取成分股
    constituents = collector.get_hs300_constituents()
    if not constituents:
        print("无法获取成分股列表")
        return

    # 开始收集
    collector.collect_all(constituents, batch_size=5)

    print("\n提示: 训练模型请运行:")
    print("  python3 strategy/train_lgb_enhanced.py")


if __name__ == "__main__":
    main()