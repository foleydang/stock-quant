#!/usr/bin/env python3
"""日线数据转换: SQLite → Qlib Bin (单进程版，节省内存)"""
import os, sys, sqlite3, struct, shutil
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import get_db_path, get_base_dir

BASE_DIR = get_base_dir()
DB_PATH = get_db_path()
QLIB_DIR = os.path.expanduser('~/.qlib/qlib_data/cn_daily')

def convert():
    """单进程转换，内存友好"""
    conn = sqlite3.connect(DB_PATH)
    symbols = [r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM kline_daily ORDER BY symbol"
    ).fetchall()]

    print(f"转换 {len(symbols)} 只股票日线数据...")

    # 清理并创建目录
    if os.path.exists(QLIB_DIR):
        shutil.rmtree(QLIB_DIR)
    features_dir = os.path.join(QLIB_DIR, 'features')
    calendars_dir = os.path.join(QLIB_DIR, 'calendars')
    instruments_dir = os.path.join(QLIB_DIR, 'instruments')
    for d in [features_dir, calendars_dir, instruments_dir]:
        os.makedirs(d)

    all_dates = set()
    instrument_info = []
    success = 0

    for i, sym in enumerate(symbols):
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(symbols)}")

        df = pd.read_sql(
            "SELECT date, open, high, low, close, volume FROM kline_daily "
            "WHERE symbol=? ORDER BY date", conn, params=(sym,)
        )
        if len(df) < 120:
            continue

        # 标准化日期
        df['date'] = df['date'].apply(lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(str(d)) == 8 else str(d)[:10])
        df['date'] = pd.to_datetime(df['date'])

        # 收集所有日期
        all_dates.update(df['date'].dt.strftime('%Y-%m-%d').tolist())

        # 记录股票时间范围
        instrument_info.append(
            f"{sym.upper()}\t{df['date'].min().strftime('%Y-%m-%d')}\t{df['date'].max().strftime('%Y-%m-%d')}"
        )

        # 写入 features (二进制格式, float32, 字段名.频率.bin)
        sym_dir = os.path.join(features_dir, sym.lower())
        os.makedirs(sym_dir, exist_ok=True)

        arr_len = len(df)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            arr = df[col].fillna(0).values.astype(np.float32)
            bin_path = os.path.join(sym_dir, f"{col}.day.bin")
            with open(bin_path, 'wb') as f:
                f.write(arr.tobytes())

        # factor (复权因子, 无复权则全为1)
        factor_arr = np.ones(arr_len, dtype=np.float32)
        with open(os.path.join(sym_dir, 'factor.day.bin'), 'wb') as f:
            f.write(factor_arr.tobytes())

        success += 1

    conn.close()

    # 写入日历
    sorted_dates = sorted(all_dates)
    cal_path = os.path.join(calendars_dir, 'day.txt')
    with open(cal_path, 'w') as f:
        f.write('\n'.join(sorted_dates))

    # 写入 instruments
    inst_path = os.path.join(instruments_dir, 'all.txt')
    with open(inst_path, 'w') as f:
        f.write('\n'.join(instrument_info))

    print(f"✅ 完成: {success} 只股票, {len(sorted_dates)} 个交易日")
    print(f"   数据目录: {QLIB_DIR}")
    return success, len(sorted_dates)


def verify():
    """验证"""
    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri=QLIB_DIR, region=REG_CN)
    from qlib.data import D
    instruments = D.instruments(market='all')
    symbols = D.list_instruments(instruments)
    print(f"✅ 验证: {len(symbols)} 只股票, 频率 day")
    # 测试读取
    df = D.features(symbols[:3], ['$close', '$open'], 
                    start_time='2026-06-01', end_time='2026-06-18', freq='day')
    print(f"   测试数据: {df.shape}")
    return symbols


if __name__ == '__main__':
    convert()
    verify()
    print(f"\n✅ 日线数据转换完成!")
    print(f"下一步: python qlib_pipeline/train_daily.py")