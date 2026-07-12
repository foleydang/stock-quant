#!/usr/bin/env python3
"""
服务器端自给自足数据同步器
- 30分钟K线: 新浪API (实时, 盘中每30分钟拉取)
- 日线K线: Tushare (盘后拉取)
- 同步后上传到OSS备份

用法:
  python strategy/data_sync.py                # 自动判断时段
  python strategy/data_sync.py --force        # 强制全量拉取
  python strategy/data_sync.py --upload-only  # 仅上传到OSS
"""

import sys, os, time, sqlite3, requests, json
import subprocess
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DB_PATH = os.path.join(ROOT, 'data', 'stock_data.db')

# 从环境变量加载敏感配置
TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN', '')
OSS_BUCKET = os.environ.get('OSS_BUCKET', 'yanten-data')
OSS_ENDPOINT = os.environ.get('OSS_ENDPOINT', 'https://oss-cn-hangzhou.aliyuncs.com')
OSS_AK = os.environ.get('OSS_ACCESS_KEY_ID', '')
OSS_SK = os.environ.get('OSS_ACCESS_KEY_SECRET', '')


def is_trading_time():
    """判断是否在交易时段"""
    now = datetime.now()
    if now.weekday() >= 5:  # 周末
        return False
    t = now.hour * 100 + now.minute
    return 930 <= t <= 1505


def is_after_market():
    """判断是否盘后"""
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    return now.hour >= 15


def fetch_30min_kline(conn, symbols=None, force=False):
    """
    从新浪API拉取30分钟K线
    返回新增条数
    """
    if symbols is None:
        symbols = [r[0] for r in conn.execute(
            "SELECT DISTINCT symbol FROM kline_30m WHERE symbol LIKE '%.SZ' OR symbol LIKE '%.SH'"
        ).fetchall()]

    today_str = datetime.now().strftime('%Y-%m-%d')
    total_new = 0
    errors = 0

    for i, sym in enumerate(symbols):
        code = sym[:6]
        if sym.endswith('.SZ'):
            sina_code = f'sz{code}'
        elif sym.endswith('.SH'):
            sina_code = f'sh{code}'
        else:
            continue

        try:
            url = "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"
            params = {"symbol": sina_code, "scale": "30", "datalen": 40}
            r = requests.get(url, params=params, timeout=10)
            if r.status_code != 200:
                errors += 1
                continue

            data = json.loads(r.text)
            if not isinstance(data, list) or len(data) == 0:
                continue

            new = 0
            for row in data:
                trade_time = row.get('day', '')
                if trade_time and len(trade_time) == 16:
                    trade_time += ':00'
                if not trade_time:
                    continue

                # 检查是否已存在
                exists = conn.execute(
                    "SELECT 1 FROM kline_30m WHERE symbol=? AND date=?",
                    (sym, trade_time)
                ).fetchone()

                if not exists:
                    conn.execute(
                        "INSERT INTO kline_30m (symbol, date, open, close, high, low, volume) VALUES (?,?,?,?,?,?,?)",
                        (sym, trade_time,
                         float(row.get('open', 0) or 0),
                         float(row.get('close', 0) or 0),
                         float(row.get('high', 0) or 0),
                         float(row.get('low', 0) or 0),
                         float(row.get('volume', 0) or 0))
                    )
                    new += 1

            conn.commit()
            total_new += new
            if new > 0:
                print(f"  [{i+1}/{len(symbols)}] {sym}: +{new}条")

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [{i+1}] {sym} 错误: {e}")

        # 避免请求过快
        time.sleep(0.08)

    print(f"\n📊 30分钟K线: 新增 {total_new} 条, 错误 {errors} 只")
    return total_new


def fetch_daily_kline(conn, force=False):
    """从Tushare拉取日线数据"""
    try:
        import tushare as ts
        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()
    except ImportError:
        print("⚠️ tushare未安装, 跳过日线")
        return 0
    except Exception as e:
        print(f"⚠️ Tushare连接失败: {e}")
        return 0

    # 获取最新日期
    max_date = conn.execute("SELECT MAX(date) FROM kline_daily").fetchone()[0]
    if not max_date:
        max_date = '2026-06-01'
    
    today = datetime.now().strftime('%Y%m%d')
    print(f"日线最新: {max_date}, 拉取到: {today}")

    total_new = 0
    # 从最新日期后一天开始拉
    import pandas as pd
    start = pd.Timestamp(max_date) + pd.Timedelta(days=1)
    end = pd.Timestamp(today)
    dates = pd.date_range(start, end, freq='B').strftime('%Y%m%d').tolist()

    if not dates:
        print("日线数据已是最新")
        return 0

    print(f"需要拉取 {len(dates)} 个交易日: {dates[:3]}...")

    for date in dates:
        try:
            df = pro.daily(trade_date=date)
            if df is None or len(df) == 0:
                print(f"  {date}: 无数据")
                continue

            count = 0
            for _, row in df.iterrows():
                sym = row['ts_code']
                # Tushare返回YYYYMMDD，转为YYYY-MM-DD
                date_fmt = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
                conn.execute(
                    """INSERT OR IGNORE INTO kline_daily 
                    (symbol, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (sym, date_fmt, row['open'], row['high'],
                     row['low'], row['close'], row['vol'] * 100)
                )
                count += 1

            conn.commit()
            total_new += count
            print(f"  {date}: +{count} 条")
            time.sleep(0.5)

        except Exception as e:
            if '频率超限' in str(e) or '每分钟' in str(e):
                print(f"  {date}: Tushare限频, 等待60秒...")
                time.sleep(60)
                continue
            print(f"  {date}: {e}")
            continue

    return total_new


def upload_to_oss():
    """上传数据库到OSS"""
    script = os.path.join(ROOT, '..', 'scripts', 'upload_to_oss.sh')
    if os.path.exists(script):
        try:
            result = subprocess.run(['bash', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=300)
            if '上传完成' in result.stdout or 'Succeed' in result.stdout:
                print("✅ OSS上传成功")
                return True
            else:
                print(f"⚠️ OSS上传: {result.stdout[-200:]}")
                return False
        except subprocess.TimeoutExpired:
            print("⚠️ OSS上传超时")
            return False
    else:
        print("⚠️ upload_to_oss.sh 不存在")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--upload-only', action='store_true')
    parser.add_argument('--30min-only', action='store_true')
    parser.add_argument('--daily-only', action='store_true')
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 DB: {DB_PATH} ({os.path.getsize(DB_PATH)/1024/1024/1024:.1f}GB)")

    if args.upload_only:
        upload_to_oss()
        conn.close()
        return

    new_30min = 0
    new_daily = 0

    do_30min = getattr(args, '30min_only', False)
    do_daily = args.daily_only

    # 30分钟K线 (盘中或强制)
    if not do_daily and not args.upload_only:
        if args.force or do_30min or is_trading_time():
            print("\n📡 拉取30分钟K线 (新浪API)...")
            new_30min = fetch_30min_kline(conn, force=args.force)

    # 日线K线 (盘后/daily-only/强制)
    if not do_30min and not args.upload_only:
        if args.force or do_daily or is_after_market():
            print("\n📡 拉取日线K线 (Tushare)...")
            new_daily = fetch_daily_kline(conn)

    conn.close()

    # 有新增数据则上传OSS
    if new_30min > 0 or new_daily > 0:
        print(f"\n📤 有新增数据 (30min:{new_30min}, daily:{new_daily}), 上传OSS...")
        upload_to_oss()
    else:
        print("\n✅ 无新增数据, 跳过OSS上传")

    print(f"\n✅ 同步完成: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()