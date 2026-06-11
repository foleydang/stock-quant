#!/usr/bin/env python3
"""
Step 1 + Step 2: 数据增强 - 北向资金 + 板块映射 + 指数日线

由于各API限频严重，此脚本采用多源分步策略：
1. 北向资金: 东方财富API（已有历史数据，但近期缺失，需要用akshare补全）
2. 板块映射: Tushare stock_basic industry字段（每小时只能请求一次）
3. 行业指数日线: 东方财富或Tushare（等限频恢复后拉）

执行策略：
- 北向资金：先用东方财富kamt API拉全量（2016-2024有效），再用akshare补2024-至今
- 板块映射：等Tushare限频后拉stock_basic的industry字段
- 行业指数：东方财富批量拉（需要分批、带延迟）

DB_PATH: <project>/python/data/stock_data.db
"""

import sqlite3
import requests
import json
import time
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'stock_data.db')


def create_tables(conn):
    """创建所有新表"""
    # 北向资金表（已存在则跳过）
    conn.execute('''CREATE TABLE IF NOT EXISTS north_flow (
        trade_date TEXT PRIMARY KEY,
        north_net REAL,
        north_buy REAL,
        north_cum REAL,
        sz_net REAL,
        sz_buy REAL,
        sz_cum REAL,
        total_net REAL,
        total_buy REAL,
        updated_at TEXT
    )''')
    
    # 板块映射表
    conn.execute('''CREATE TABLE IF NOT EXISTS stock_sector (
        symbol TEXT PRIMARY KEY,
        name TEXT,
        industry TEXT,
        sector_code TEXT,
        updated_at TEXT
    )''')
    
    # 行业指数日线表
    conn.execute('''CREATE TABLE IF NOT EXISTS sector_index_daily (
        sector_code TEXT,
        trade_date TEXT,
        open REAL,
        close REAL,
        high REAL,
        low REAL,
        volume REAL,
        amount REAL,
        pct_chg REAL,
        PRIMARY KEY (sector_code, trade_date)
    )''')
    
    # 沪深300指数日线表（大盘情绪基准）
    conn.execute('''CREATE TABLE IF NOT EXISTS hs300_daily (
        trade_date TEXT PRIMARY KEY,
        open REAL,
        close REAL,
        high REAL,
        low REAL,
        volume REAL,
        amount REAL,
        pct_chg REAL
    )''')
    
    conn.commit()


def fetch_north_flow_em(conn):
    """从东方财富拉北向资金历史数据"""
    print("\n[1/4] 拉取北向资金数据...")
    url = "https://push2his.eastmoney.com/api/qt/kamt.kline/get"
    params = {
        "fields1": "f1,f3,f5",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63",
        "klt": "101", "lmt": "0",
        "ut": "7eea3ed2ced24c2974d3210a0be1e25",
    }
    
    try:
        r = requests.get(url, params=params, timeout=30)
        data = r.json()["data"]
    except Exception as e:
        print(f"  东方财富API失败: {e}")
        return
    
    hk2sh = data.get("hk2sh", [])
    hk2sz = data.get("hk2sz", [])
    
    # 解析：格式=日期,当日净流入(万元),当日买入(万元),累计净流入(万元)
    sh_data = {}
    for line in hk2sh:
        parts = line.split(",")
        if len(parts) >= 4:
            date = parts[0]
            try:
                net = float(parts[1]) if parts[1] else 0
                buy = float(parts[2]) if parts[2] else 0
                cum = float(parts[3]) if parts[3] else 0
                # 东方财富近期数据的net/buy全为0（占位数据），但cum有效
                sh_data[date] = {'net': net, 'buy': buy, 'cum': cum}
            except (ValueError, IndexError):
                continue
    
    sz_data = {}
    for line in hk2sz:
        parts = line.split(",")
        if len(parts) >= 4:
            date = parts[0]
            try:
                net = float(parts[1]) if parts[1] else 0
                buy = float(parts[2]) if parts[2] else 0
                cum = float(parts[3]) if parts[3] else 0
                sz_data[date] = {'net': net, 'buy': buy, 'cum': cum}
            except (ValueError, IndexError):
                continue
    
    # 写入DB - 对于net=0的记录，先用cum存储，后面用akshare补全
    count = 0
    valid_count = 0
    for date in sorted(set(sh_data.keys()) & set(sz_data.keys())):
        sh = sh_data[date]
        sz = sz_data[date]
        total_net = sh['net'] + sz['net']
        total_buy = sh['buy'] + sz['buy']
        
        # 如果net=0但有cum变化，用累计差反推
        if sh['net'] == 0 and sz['net'] == 0:
            # 找前一天的数据来反推
            prev_date = None
            all_dates = sorted(sh_data.keys())
            idx = all_dates.index(date) - 1
            if idx >= 0:
                prev_date = all_dates[idx]
                prev_sh_cum = sh_data[prev_date]['cum']
                prev_sz_cum = sz_data.get(prev_date, {}).get('cum', 0)
                sh_net = sh['cum'] - prev_sh_cum
                sz_net = sz['cum'] - prev_sz_cum
                # 如果累计值没变化（说明是占位数据），则标记为NULL
                if sh_net == 0 and sz_net == 0:
                    total_net = None
                else:
                    total_net = sh_net + sz_net
        
        conn.execute(
            """INSERT OR REPLACE INTO north_flow 
            (trade_date, north_net, north_buy, north_cum, sz_net, sz_buy, sz_cum,
             total_net, total_buy, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (date, sh['net'], sh['buy'], sh['cum'],
             sz['net'], sz['buy'], sz['cum'],
             total_net, total_buy,
             datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        )
        count += 1
        if total_net is not None and total_net != 0:
            valid_count += 1
    
    conn.commit()
    print(f"  ✅ 北向资金写入: {count} 条, 有效数据: {valid_count} 条")


def fetch_north_flow_akshare(conn):
    """用akshare补全近期的北向资金数据"""
    print("\n[2/4] 用akshare补全近期北向资金...")
    try:
        import akshare as ak
    except ImportError:
        print("  akshare未安装，跳过")
        return
    
    try:
        df = ak.stock_hsgt_hist_em(symbol="北向资金")
    except Exception as e:
        print(f"  akshare获取失败: {e}")
        return
    
    # akshare有2264条有效数据（2014-2024年8月）
    valid = df[df['当日成交净买额'].notna()]
    print(f"  akshare有效数据: {len(valid)} 条, 范围: {valid['日期'].iloc[0]} ~ {valid['日期'].iloc[-1]}")
    
    # 补全东方财富缺失的数据（akshare有效但东方财富net=0的日期）
    count = 0
    for _, row in valid.iterrows():
        date = str(row['日期'])
        net = row['当日成交净买额']  # 亿元
        
        # 检查DB中该日期是否total_net为NULL或0
        cursor = conn.execute("SELECT total_net FROM north_flow WHERE trade_date=?", (date,))
        existing = cursor.fetchone()
        
        if existing is None or existing[0] is None or existing[0] == 0:
            # akshare数据单位是亿元，东方财富单位是万元
            total_net_wan = net * 10000 if net is not None else None
            
            conn.execute(
                """INSERT OR REPLACE INTO north_flow 
                (trade_date, north_net, north_buy, north_cum, sz_net, sz_buy, sz_cum,
                 total_net, total_buy, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (date, None, None, None, None, None, None,
                 total_net_wan, None,
                 datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            )
            count += 1
    
    conn.commit()
    print(f"  ✅ akshare补全: {count} 条")


def fetch_sector_mapping(conn):
    """从Tushare拉取行业映射（需等限频恢复）"""
    print("\n[3/4] 拉取板块映射...")
    try:
        import tushare as ts
        ts.set_token('7a9014b18909e8cbce5109d7175f7b21ce37354eaff2371db0da2c58')
        pro = ts.pro_api()
        df = pro.stock_basic(exchange='', list_status='L',
            fields='ts_code,symbol,name,industry,market,list_date')
    except Exception as e:
        if '频率超限' in str(e):
            print(f"  ⚠️ Tushare限频(1次/小时)，请稍后再运行此脚本")
            print(f"  使用本地fallback映射...")
            _fallback_sector_mapping(conn)
            return
        else:
            print(f"  Tushare错误: {e}")
            _fallback_sector_mapping(conn)
            return
    
    # 写入映射表
    count = 0
    for _, row in df.iterrows():
        symbol = row['ts_code']  # 如 600036.SH
        name = row['name']
        industry = row.get('industry', '其他') or '其他'
        
        conn.execute(
            """INSERT OR REPLACE INTO stock_sector 
            (symbol, name, industry, sector_code, updated_at)
            VALUES (?, ?, ?, ?, ?)""",
            (symbol, name, industry, industry,
             datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        )
        count += 1
    
    conn.commit()
    print(f"  ✅ 板块映射写入: {count} 条, 行业数: {df['industry'].nunique()}")


def _fallback_sector_mapping(conn):
    """当Tushare限频时的本地fallback映射"""
    print("  使用关键词匹配fallback...")
    
    cursor = conn.execute("SELECT symbol, name FROM stock_info")
    stocks = cursor.fetchall()
    
    SW_L1 = {
        '银行': ['银行'],
        '非银金融': ['证券', '保险', '信托', '期货'],
        '医药生物': ['医药', '生物', '药', '医'],
        '食品饮料': ['茅台', '五粮液', '泸州', '洋河', '汾酒', '古井', '伊利', '双汇', '海天', '食品', '酒', '饮料', '奶', '啤'],
        '电子': ['电子', '芯片', '半导体', '立讯', '歌尔', '京东方', '韦尔', '兆易'],
        '计算机': ['软件', '计算机', '科大', '中科', '浪潮', '用友', '金山'],
        '传媒': ['传媒', '影视', '游戏', '广告', '出版'],
        '通信': ['通信', '中兴', '烽火'],
        '电力设备': ['宁德', '隆基', '通威', '阳光', '光伏', '锂', '风电', '电池', '电气'],
        '汽车': ['比亚迪', '长城', '长安', '上汽', '广汽', '汽车', '车', '福耀'],
        '家用电器': ['美的', '格力', '海尔', '家电', '老板'],
        '房地产': ['万科', '保利', '地产', '置地', '新城'],
        '建筑装饰': ['建筑', '装饰', '中铁', '中建', '水泥'],
        '国防军工': ['中航', '军工', '航发', '航天', '兵装', '船舶'],
        '公用事业': ['电力', '水务', '环保', '燃气', '核电'],
        '交通运输': ['航空', '航运', '港口', '高速', '物流', '快递'],
        '钢铁': ['宝钢', '钢铁', '鞍钢'],
        '煤炭': ['煤炭', '神华', '陕煤'],
        '石油石化': ['石油', '石化', '中海油'],
        '有色金属': ['铜', '铝', '锌', '镍', '锂矿', '黄金', '紫金'],
        '基础化工': ['化工', '聚氨酯'],
        '农林牧渔': ['农业', '牧原', '温氏', '饲料', '种业', '养猪'],
        '商贸零售': ['商业', '零售', '超市', '百货'],
        '综合金融': ['综合'],
        '机械设备': ['机械', '设备', '三一', '中联'],
        '纺织服饰': ['纺织', '服饰', '服装'],
        '轻工制造': ['造纸', '包装', '家具'],
    }
    
    count = 0
    for symbol, name in stocks:
        code = symbol.split('.')[0]
        
        if symbol.endswith('.HK'):
            industry = '港股'
        elif code.startswith(('51', '159', '510', '511', '515')):
            industry = 'ETF'
        else:
            industry = '其他'
            for sector, keywords in SW_L1.items():
                if any(kw in name for kw in keywords):
                    industry = sector
                    break
        
        conn.execute(
            """INSERT OR REPLACE INTO stock_sector 
            (symbol, name, industry, sector_code, updated_at)
            VALUES (?, ?, ?, ?, ?)""",
            (symbol, name, industry, industry,
             datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        )
        count += 1
    
    conn.commit()
    from collections import Counter
    industries = [r[0] for r in conn.execute("SELECT industry FROM stock_sector").fetchall()]
    print(f"  ✅ fallback映射写入: {count} 条")
    print(f"  ⚠️ '其他'类有 {Counter(industries)['其他']} 只，待Tushare恢复后需更新")


def fetch_hs300_daily(conn):
    """从东方财富拉沪深300指数日线"""
    print("\n[4/4] 拉取沪深300指数日线...")
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": "1.000300",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "1",
        "lmt": "0",
        "end": "20500101",
        "ut": "7eea3ed2ced24c2974d3210a0be1e25",
    }
    
    data = {}
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=30)
            data = r.json()
            break
        except Exception as e:
            print(f"  第{attempt+1}次请求失败，等待10秒...")
            time.sleep(10)
    
    if not data.get("data") or not data["data"].get("klines"):
        print("  ❌ 沪深300数据获取失败")
        return
    
    klines = data["data"]["klines"]
    # 格式: 日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
    count = 0
    for line in klines:
        parts = line.split(",")
        if len(parts) >= 11:
            date = parts[0]
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO hs300_daily 
                    (trade_date, open, close, high, low, volume, amount, pct_chg)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (date, float(parts[1]), float(parts[2]), float(parts[3]),
                     float(parts[4]), float(parts[5]), float(parts[6]),
                     float(parts[9]))  # 涨跌幅
                )
                count += 1
            except (ValueError, IndexError):
                continue
    
    conn.commit()
    print(f"  ✅ 沪深300日线写入: {count} 条")


def main():
    print("=" * 50)
    print("量化模型数据增强 - Step 1 & 2")
    print("=" * 50)
    
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)
    
    # Step 1: 北向资金
    fetch_north_flow_em(conn)
    fetch_north_flow_akshare(conn)
    
    # Step 2: 板块映射
    fetch_sector_mapping(conn)
    
    # Step 2b: 沪深300日线（大盘基准）
    fetch_hs300_daily(conn)
    
    # 统计
    print("\n" + "=" * 50)
    print("数据增强完成统计:")
    print("=" * 50)
    
    tables = ['north_flow', 'stock_sector', 'hs300_daily']
    for table in tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {count} 条")
        except Exception:
            print(f"  {table}: 未创建")
    
    conn.close()


if __name__ == '__main__':
    main()