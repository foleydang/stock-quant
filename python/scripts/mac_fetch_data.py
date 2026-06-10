#!/usr/bin/env python3
"""
Mac端数据拉取脚本 - 在你的24G Mac上运行

输出3个CSV文件到 ~/Downloads/：
1. north_flow.csv - 北向资金（沪股通+深股通，2024-08至今）
2. stock_sector.csv - 行业映射（BaoStock全量5529只）
3. hs300_daily.csv - 沪深300日线（备用，服务器已有）

使用方式：
  pip install akshare baostock
  python3 mac_fetch_data.py
"""

import os
import time
from datetime import datetime

OUTPUT_DIR = os.path.expanduser("~/Downloads")

# ========================================
# 1. 北向资金（akshare）
# ========================================
print("=" * 50)
print("1. 拉取北向资金数据（akshare）")
print("=" * 50)

import akshare as ak

north_records = []

for channel in ["沪股通", "深股通"]:
    print(f"\n拉取 {channel}...")
    try:
        df = ak.stock_hsgt_hist_em(symbol=channel)
        valid = df[df["当日成交净买额"].notna()]
        print(f"  {channel}: {len(valid)} 条有效数据")
        print(f"  范围: {valid['日期'].iloc[0]} ~ {valid['日期'].iloc[-1]}")
        
        for _, row in valid.iterrows():
            north_records.append({
                "trade_date": str(row["日期"]),
                "channel": channel,
                "net_billion": row["当日成交净买额"],  # 亿元
                "buy_billion": row.get("买入成交额", 0),  # 亿元
                "sell_billion": row.get("卖出成交额", 0),  # 亿元
                "cum_billion": row.get("历史累计净买额", 0),  # 亿元
                "inflow_billion": row.get("当日资金流入", 0),  # 亿元
            })
        
        time.sleep(10)  # akshare限频
    except Exception as e:
        print(f"  ❌ {channel} 错误: {e}")

# 合并沪股通+深股通 → 北向合计
import pandas as pd
north_df = pd.DataFrame(north_records)

# 按日期合并
sh = north_df[north_df["channel"] == "沪股通"][["trade_date", "net_billion", "buy_billion", "cum_billion"]].rename(
    columns={"net_billion": "north_net", "buy_billion": "north_buy", "cum_billion": "north_cum"}
)
sz = north_df[north_df["channel"] == "深股通"][["trade_date", "net_billion", "buy_billion", "cum_billion"]].rename(
    columns={"net_billion": "sz_net", "buy_billion": "sz_buy", "cum_billion": "sz_cum"}
)

merged = sh.merge(sz, on="trade_date", how="outer")
merged["total_net"] = merged["north_net"].fillna(0) + merged["sz_net"].fillna(0)
merged["total_buy"] = merged["north_buy"].fillna(0) + merged["sz_buy"].fillna(0)

# 转成万元（服务器DB的单位）
for col in ["north_net", "north_buy", "north_cum", "sz_net", "sz_buy", "sz_cum", "total_net", "total_buy"]:
    merged[col] = merged[col] * 10000

north_csv = os.path.join(OUTPUT_DIR, "north_flow.csv")
merged.to_csv(north_csv, index=False)
print(f"\n✅ north_flow.csv: {len(merged)} 条 → {north_csv}")
print(f"  日期范围: {merged['trade_date'].min()} ~ {merged['trade_date'].max()}")
print(f"  最近5天:")
print(merged.tail(5).to_string())

# ========================================
# 2. 行业映射（BaoStock）
# ========================================
print("\n" + "=" * 50)
print("2. 拉取行业映射数据（BaoStock）")
print("=" * 50)

import baostock as bs

lg = bs.login()
print(f"  登录: {lg.error_msg}")

rs = bs.query_stock_industry()
industry_rows = []
while rs.next():
    row = rs.get_row_data()
    industry_rows.append(row)

print(f"  BaoStock返回: {len(industry_rows)} 条")
print(f"  字段: {rs.fields}")

# 格式: (updateDate, code, code_name, industry, industryClassification)
# fields顺序: updateDate是第0列, code是第1列
sector_records = []
for row in industry_rows:
    # 正确的列索引
    code = row[1]        # sh.600036 (第1列)
    name = row[2]        # 招商银行 (第2列)
    industry = row[3] if len(row) > 3 and row[3] else "其他"  # 银行 (第3列)
    industryClassification = row[4] if len(row) > 4 else ""  # 申万L1 (第4列)
    
    # 如果row[3]为空，用row[4]
    if not industry or industry == "":
        industry = industryClassification if industryClassification else "其他"
    
    # 转换: sh.600036 → 600036.SH
    parts = code.split(".")
    if len(parts) >= 2:
        symbol = f"{parts[1]}.{parts[0].upper()}"
    else:
        symbol = code
    
    sector_records.append({
        "symbol": symbol,
        "name": name,
        "industry": industry
    })

bs.logout()

sector_df = pd.DataFrame(sector_records)
sector_csv = os.path.join(OUTPUT_DIR, "stock_sector.csv")
sector_df.to_csv(sector_csv, index=False)

from collections import Counter
ind_counts = Counter(sector_df["industry"])
print(f"\n✅ stock_sector.csv: {len(sector_df)} 条 → {sector_csv}")
print(f"  行业数: {len(ind_counts)}")
print(f"  '其他': {ind_counts.get('其他', 0)} 只")
print(f"  前5行业:")
for ind, cnt in ind_counts.most_common(5):
    print(f"    {ind}: {cnt}只")

# 验证几个关键股票
for sym in ["600036.SH", "000001.SZ", "601318.SH"]:
    r = sector_df[sector_df["symbol"] == sym]
    if len(r) > 0:
        print(f"    {sym}: {r.iloc[0]['name']} → {r.iloc[0]['industry']}")
    else:
        print(f"    {sym}: ❌ 未找到")

# ========================================
# 3. 沪深300日线（BaoStock，备用）
# ========================================
print("\n" + "=" * 50)
print("3. 沪深300日线（服务器已有，可选）")
print("=" * 50)

lg = bs.login()
rs = bs.query_history_k_data_plus(
    "sh.000300",
    "date,open,high,low,close,volume,amount,pctChg",
    start_date="2023-01-01", end_date="2026-06-10",
    frequency="d", adjustflag="3"
)

hs300_rows = []
while rs.next():
    row = rs.get_row_data()
    # date是YYYY-MM-DD，转YYYYMMDD
    trade_date = row[0].replace("-", "")
    hs300_rows.append({
        "trade_date": trade_date,
        "open": row[1], "high": row[2], "low": row[3],
        "close": row[4], "volume": row[5], "amount": row[6],
        "pct_chg": row[7]
    })

bs.logout()

hs300_df = pd.DataFrame(hs300_rows)
hs300_csv = os.path.join(OUTPUT_DIR, "hs300_daily.csv")
hs300_df.to_csv(hs300_csv, index=False)
print(f"✅ hs300_daily.csv: {len(hs300_df)} 条 → {hs300_csv}")

# ========================================
# ========================================
# 4. 市场资金流向（含北向资金近期数据）
# ========================================
print("\n" + "=" * 50)
print("4. 拉取市场资金流向（akshare）")
print("=" * 50)

try:
    df_flow = ak.stock_market_fund_flow()
    print(f"  market_fund_flow: {len(df_flow)} 条")
    print(f"  列: {df_flow.columns.tolist()[:10]}")
    print(f"  范围: {df_flow['日期'].iloc[0]} ~ {df_flow['日期'].iloc[-1]}")
    
    # 看有没有北向字段
    north_cols = [c for c in df_flow.columns if '北向' in c or '外资' in c or '沪股' in c or '深股' in c]
    print(f"  北向相关列: {north_cols}")
    
    flow_csv = os.path.join(OUTPUT_DIR, "market_fund_flow.csv")
    df_flow.to_csv(flow_csv, index=False)
    print(f"  ✅ market_fund_flow.csv: {len(df_flow)} 条 → {flow_csv}")
    print(f"  最近3天:")
    print(df_flow.tail(3).to_string())
    
    time.sleep(5)
except Exception as e:
    print(f"  ❌ market_fund_flow错误: {e}")
    print(f"  ⚠️ 这台Mac上应该能正常拉，服务器IP被限频才失败")
# ========================================
print("\n" + "=" * 50)
print("拉取完成！请将以下3个文件传到服务器：")
print("=" * 50)
print(f"  1. {north_csv}  ({len(merged)} 条)")
print(f"  2. {sector_csv}  ({len(sector_df)} 条)")
print(f"  3. {hs300_csv}  ({len(hs300_df)} 条)")
print("\n传输方式（任选）：")
print("  scp ~/Downloads/north_flow.csv ~/Downloads/stock_sector.csv ~/Downloads/hs300_daily.csv root@47.242.158.242:/root/github/stock-quant/python/data/")
print("  或用其他方式传上去")
print("\n服务器导入命令：")
print("  python3 /root/github/stock-quant/python/scripts/import_csv_data.py")