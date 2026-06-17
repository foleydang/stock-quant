#!/usr/bin/env python3
"""
Qlib 数据转换脚本: sqlite kline_30m → Qlib .bin 格式

用法:
  python qlib_pipeline/convert_data.py                          # 默认配置
  python qlib_pipeline/convert_data.py --db data/stock_data.db  # 指定数据库
  python qlib_pipeline/convert_data.py --freq 30min             # 指定频率

步骤:
  1. 从 sqlite 读取 kline_30m 表
  2. 导出为 Qlib 兼容的 CSV 格式
  3. 调用 dump_bin 转换为 .bin 格式
"""

import os, sys, argparse, sqlite3, time
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ============ 配置 ============
DB_PATH = os.path.join(ROOT, 'data', 'stock_data.db')
DB_TABLE = 'kline_30m'
FREQ = '30min'
QLIB_DATA_DIR = os.path.expanduser('~/.qlib/qlib_data/cn_30min')
CSV_DIR = os.path.join(QLIB_DATA_DIR, 'source')
BIN_DIR = os.path.join(QLIB_DATA_DIR, 'bin')


def export_to_csv(db_path: str, table: str, out_dir: str):
    """从 sqlite 导出为 Qlib CSV 格式"""
    os.makedirs(out_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    symbols = [r[0] for r in conn.execute(f"SELECT DISTINCT symbol FROM {table}")]
    print(f"📊 导出 {len(symbols)} 只股票 → {out_dir}")

    total_rows = 0
    for sym in symbols:
        df = pd.read_sql(
            f"SELECT * FROM {table} WHERE symbol=? ORDER BY date",
            conn, params=(sym,)
        )
        if len(df) == 0:
            continue

        # Qlib 要求的列: symbol, date, open, high, low, close, volume, factor
        df['symbol'] = sym
        df['date'] = pd.to_datetime(df['date'], format='mixed')
        df['factor'] = 1.0  # 不复权因子

        out_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'factor']
        df = df[out_cols]

        out_path = os.path.join(out_dir, f"{sym}.csv")
        df.to_csv(out_path, index=False)
        total_rows += len(df)

    conn.close()
    print(f"  ✅ 完成: {len(symbols)} 只股票, {total_rows:,} 行")
    return len(symbols), total_rows


def convert_to_bin(csv_dir: str, bin_dir: str, freq: str):
    """调用 Qlib dump_bin 转换为 .bin 格式"""
    from scripts.dump_bin import DumpBinAll

    os.makedirs(bin_dir, exist_ok=True)

    print(f"🔄 转换 CSV → .bin (freq={freq})")
    print(f"  源: {csv_dir}")
    print(f"  目标: {bin_dir}")

    d = DumpBinAll(
        csv_path=csv_dir,
        qlib_dir=bin_dir,
        freq=freq,
        date_field_name='date',
        symbol_field_name='symbol',
        include_fields=['open', 'high', 'low', 'close', 'volume', 'factor'],
    )
    d.dump()

    # 计算大小
    total_size = 0
    for root, dirs, files in os.walk(bin_dir):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))
    print(f"  ✅ .bin 数据大小: {total_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='Qlib 数据转换')
    parser.add_argument('--db', default=DB_PATH, help='sqlite 数据库路径')
    parser.add_argument('--table', default=DB_TABLE, help='表名')
    parser.add_argument('--freq', default=FREQ, help='K线频率')
    parser.add_argument('--csv-dir', default=CSV_DIR, help='CSV 输出目录')
    parser.add_argument('--bin-dir', default=BIN_DIR, help='.bin 输出目录')
    parser.add_argument('--skip-csv', action='store_true', help='跳过 CSV 导出 (已有CSV)')
    parser.add_argument('--skip-bin', action='store_true', help='跳过 bin 转换 (仅导出CSV)')
    args = parser.parse_args()

    t0 = time.time()
    print(f"{'='*60}")
    print(f" Qlib 数据转换: {args.table} → .bin (freq={args.freq})")
    print(f"{'='*60}")

    if not args.skip_csv:
        t1 = time.time()
        export_to_csv(args.db, args.table, args.csv_dir)
        print(f"  CSV导出耗时: {time.time()-t1:.0f}s")

    if not args.skip_bin:
        t1 = time.time()
        convert_to_bin(args.csv_dir, args.bin_dir, args.freq)
        print(f"  bin转换耗时: {time.time()-t1:.0f}s")

    print(f"\n{'='*60}")
    print(f" ✅ 总耗时: {time.time()-t0:.0f}s")
    print(f" 数据目录: {args.bin_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()