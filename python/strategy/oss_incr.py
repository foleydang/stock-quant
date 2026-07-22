#!/usr/bin/env python3
"""
OSS 增量/快照同步 (替换整库 blob 上传/下载)

设计: 训练/回测继续用本地 SQLite 全量副本; 机器之间只通过 OSS 交换 parquet:
大表按"月"分片(只重传当月), 小表整表快照。纯追加 + 幂等合并, 各写各表,
永不互相覆盖。走 oss2 (本机已装), 不依赖 ossutil。

OSS 布局:
  <bucket>/<prefix>/incr/<table>/<YYYY-MM>.parquet   # 增量: 大表按月分片(含该月所有股票)
  <bucket>/<prefix>/full/<table>.parquet             # 全量: 小表/维表整表单文件快照
  <bucket>/<prefix>/stock_data.db                    # --full 整库快照(bootstrap/备份)

用法:
  python strategy/oss_incr.py upload   --producer server|mac [--days N] [--dry-run] [--full]
  python strategy/oss_incr.py upload   [--tables a,b] [--days N] [--dry-run]
  python strategy/oss_incr.py download [--tables a,b] [--dry-run] [--full]
"""

import os
import sys
import argparse
import tempfile
from datetime import datetime

import pandas as pd

STRATEGY_DIR = os.path.dirname(os.path.abspath(__file__))
PY_ROOT = os.path.dirname(STRATEGY_DIR)
PROJECT_ROOT = os.path.dirname(PY_ROOT)
DB_PATH = os.path.join(PY_ROOT, 'data', 'stock_data.db')
ENV_PATH = os.path.join(PROJECT_ROOT, '.env')

OSS_PREFIX = os.environ.get('OSS_PREFIX', 'stock-quant')
REUPLOAD_RECENT_PERIODS = 2       # 最近 N 个分片周期(月)总是重传, 兜住当月增长/迟到的更新
PARQUET_COMPRESSION = 'zstd'

# 表配置。kind:
#   'ts'  = 每股票×每天的稠密大表, 按 period_expr(默认"月")拆片, 一个文件含该月所有股票。
#           选月而非日: 按日会产生上万个几十KB小文件(kline_daily 就 5599 个), list/bootstrap
#           都遭殃; 按月总文件数 ~760, 当月片才 0.2~1.6M, 每次同步只重传"当月"这一个文件。
#   'dim' = 全量快照单文件(下载整表替换): 维表 + 全量<3M 的小表(市场级"每天1行"序列、
#           季频 fundamental、月频 sentiment_margin) —— 拆片没意义, 整表快照最省最简单。
#   period_expr : 'ts' 用, 从行里算出分片周期键的 SQL 表达式 (substr(col,1,7)=按月)
#   id          : 导出时丢弃的自增列 (下载让目标库自动分配)
#   mode        : ignore(默认,价格不可变) / replace(可重算,如特征)
TABLES = {
    # --- A. 按月分片 (每股票×每天的大表) ---
    'kline_daily':        {'kind': 'ts', 'period_expr': 'substr(date,1,7)',       'id': 'id',  'mode': 'ignore'},
    'kline_30m':          {'kind': 'ts', 'period_expr': 'substr(date,1,7)',       'id': 'id',  'mode': 'ignore'},
    'daily_features':     {'kind': 'ts', 'period_expr': 'substr(date,1,7)',       'id': None,  'mode': 'replace'},
    'sentiment_daily':    {'kind': 'ts', 'period_expr': 'substr(trade_date,1,7)', 'id': 'id',  'mode': 'ignore'},
    'sentiment_lhb':      {'kind': 'ts', 'period_expr': 'substr(trade_date,1,7)', 'id': 'id',  'mode': 'ignore'},
    # --- B. 全量快照 (全量 <3M 的小表 + 维表) ---
    'sentiment_margin':   {'kind': 'dim', 'id': None},   # 月频, 全量 3M
    'macro_daily':        {'kind': 'dim', 'id': None},   # 1行/天 × 6531天
    'hs300_daily':        {'kind': 'dim', 'id': None},   # 1行/天 × 3048天
    'north_flow':         {'kind': 'dim', 'id': None},   # 1行/天
    'south_flow':         {'kind': 'dim', 'id': None},   # 1行/天
    'fundamental_daily':  {'kind': 'dim', 'id': None},   # 季频, 无唯一键, 日期带时分秒
    'sector_index_daily': {'kind': 'dim', 'id': None},   # 目前为空
    'stock_info':         {'kind': 'dim', 'id': None},
    'stock_sector':       {'kind': 'dim', 'id': None},
}

# 生产者分工: 避免多机抢同一张表的分片文件 (谁产谁传)。
PRODUCER_TABLES = {
    'mac': ['kline_30m'],                                        # Mac/香港机: 新浪 30m
    'server': [t for t in TABLES if t != 'kline_30m'],           # 服务器: 其余全部
}


# ---------------------------------------------------------------- env / oss

def _load_env():
    """未从环境注入 OSS_* 时, 从项目根 .env 兜底加载 (与 sync_db.sh 一致)。"""
    if os.environ.get('OSS_BUCKET') and os.environ.get('OSS_ACCESS_KEY_ID'):
        return
    if not os.path.exists(ENV_PATH):
        return
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _bucket():
    import oss2
    ak = os.environ.get('OSS_ACCESS_KEY_ID', '')
    sk = os.environ.get('OSS_ACCESS_KEY_SECRET', '')
    endpoint = os.environ.get('OSS_ENDPOINT', 'https://oss-cn-hangzhou.aliyuncs.com')
    name = os.environ.get('OSS_BUCKET', '')
    if not (ak and sk and name):
        sys.exit('❌ 缺少 OSS_BUCKET / OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET (检查 .env)')
    return oss2.Bucket(oss2.Auth(ak, sk), endpoint, name)


def _remote_periods(bucket, table):
    """列出某表在 OSS 上已有的分片周期 -> {period: etag}。"""
    import oss2
    prefix = f"{OSS_PREFIX}/incr/{table}/"
    out = {}
    for obj in oss2.ObjectIterator(bucket, prefix=prefix):
        base = obj.key[len(prefix):]
        if base.endswith('.parquet'):
            out[base[:-len('.parquet')]] = obj.etag
    return out


# ---------------------------------------------------------------- helpers

def _table_exists(conn, table):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _cols(conn, table, exclude=None):
    """返回列名列表, 排除 exclude (自增 id)。表不存在返回 None。"""
    rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    if not rows:
        return None
    return [r[1] for r in rows if r[1] != exclude]


def _rows(df):
    """DataFrame -> 供 executemany 的 python 原生元组; NaN->None, numpy 标量->python。"""
    df = df.where(pd.notna(df), None)
    out = []
    for rec in df.itertuples(index=False, name=None):
        out.append(tuple(v.item() if hasattr(v, 'item') else v for v in rec))
    return out


def _write_parquet(df, path):
    df.to_parquet(path, engine='pyarrow', compression=PARQUET_COMPRESSION, index=False)


def _insert(conn, table, df, verb):
    """verb: 'INSERT OR IGNORE' | 'INSERT OR REPLACE' | 'INSERT'"""
    if df is None or df.empty:
        return 0
    cols = list(df.columns)
    ph = ','.join('?' * len(cols))
    sql = f"{verb} INTO {table} ({','.join(cols)}) VALUES ({ph})"
    conn.executemany(sql, _rows(df))
    return len(df)


def _ensure_ledger(conn):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sync_ledger ("
        "tbl TEXT, part TEXT, etag TEXT, imported_at TEXT, "
        "PRIMARY KEY(tbl, part))"
    )


def _ledger_seen(conn, table, part, etag):
    row = conn.execute(
        "SELECT etag FROM sync_ledger WHERE tbl=? AND part=?", (table, part)
    ).fetchone()
    return row is not None and row[0] == etag


def _ledger_mark(conn, table, part, etag):
    conn.execute(
        "INSERT OR REPLACE INTO sync_ledger (tbl, part, etag, imported_at) VALUES (?,?,?,?)",
        (table, part, etag, datetime.now().isoformat(timespec='seconds')),
    )


# ---------------------------------------------------------------- upload

def _upload_ts(bucket, conn, table, cfg, days_limit, dry_run):
    cols = _cols(conn, table, exclude=cfg.get('id'))
    if cols is None:
        return 0
    period_expr = cfg['period_expr']
    local = [r[0] for r in conn.execute(
        f"SELECT DISTINCT {period_expr} AS p FROM {table} WHERE p IS NOT NULL ORDER BY p"
    )]
    if not local:
        print(f"  {table}: 本地无数据, 跳过")
        return 0
    if days_limit:
        local = local[-days_limit:]

    remote = _remote_periods(bucket, table)
    recent = set(local[-REUPLOAD_RECENT_PERIODS:]) if REUPLOAD_RECENT_PERIODS else set()
    planned = [p for p in local if p not in remote or p in recent]

    if not planned:
        print(f"  {table}: 已最新 (本地{len(local)}片/远端{len(remote)})")
        return 0

    sel = ', '.join(cols)
    n = 0
    with tempfile.TemporaryDirectory() as tmp:
        for p in planned:
            df = pd.read_sql_query(
                f"SELECT {sel} FROM {table} WHERE {period_expr}=?", conn, params=(p,)
            )
            if df.empty:
                continue
            key = f"{OSS_PREFIX}/incr/{table}/{p}.parquet"
            path = os.path.join(tmp, f"{table}_{p}.parquet")
            _write_parquet(df, path)
            if dry_run:
                print(f"  [DRY] {key}  {len(df)}行 {os.path.getsize(path)/1024:.0f}KB")
            else:
                bucket.put_object_from_file(key, path)
            os.remove(path)
            n += 1
    print(f"  {table}: {'将上传' if dry_run else '已上传'} {n} 个月度分片")
    return n


def _upload_dim(bucket, conn, table, cfg, dry_run):
    if not _table_exists(conn, table):
        return 0
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    key = f"{OSS_PREFIX}/full/{table}.parquet"
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"{table}.parquet")
        _write_parquet(df, path)
        size = os.path.getsize(path)
        if dry_run:
            print(f"  [DRY] {key}  {len(df)}行 {size/1024:.0f}KB (全量快照)")
        else:
            bucket.put_object_from_file(key, path)
            print(f"  {table}: 已上传全量快照 {len(df)}行 {size/1024:.0f}KB")
    return 1


def upload(tables=None, days=None, dry_run=False, full=False):
    import sqlite3
    _load_env()
    bucket = _bucket()
    print(f"📤 增量上传 OSS  bucket={os.environ.get('OSS_BUCKET')}  prefix={OSS_PREFIX}")
    conn = sqlite3.connect(DB_PATH)
    try:
        names = tables or list(TABLES.keys())
        for t in names:
            cfg = TABLES.get(t)
            if not cfg:
                print(f"  ⚠️ 未知表 {t}, 跳过")
                continue
            if cfg['kind'] == 'ts':
                _upload_ts(bucket, conn, t, cfg, days, dry_run)
            else:
                _upload_dim(bucket, conn, t, cfg, dry_run)
    finally:
        conn.close()

    if full and not dry_run:
        key = f"{OSS_PREFIX}/stock_data.db"
        print(f"📦 --full 整库快照 -> {key} ({os.path.getsize(DB_PATH)/1024/1024/1024:.1f}GB)")
        bucket.put_object_from_file(key, DB_PATH)
    print("✅ 上传完成")


# ---------------------------------------------------------------- download

def _download_ts(bucket, conn, table, cfg, dry_run):
    if not _table_exists(conn, table):
        print(f"  ⚠️ 本地缺表 {table} (先跑应用建表), 跳过")
        return 0
    remote = _remote_periods(bucket, table)
    mode = cfg.get('mode', 'ignore')
    verb = {'ignore': 'INSERT OR IGNORE', 'replace': 'INSERT OR REPLACE'}[mode]
    n = 0
    with tempfile.TemporaryDirectory() as tmp:
        for p, etag in sorted(remote.items()):
            if _ledger_seen(conn, table, p, etag):
                continue
            if dry_run:
                print(f"  [DRY] {table}/{p} 待下载合并 ({mode})")
                n += 1
                continue
            path = os.path.join(tmp, f"{table}_{p}.parquet")
            bucket.get_object_to_file(f"{OSS_PREFIX}/incr/{table}/{p}.parquet", path)
            df = pd.read_parquet(path)
            os.remove(path)
            _insert(conn, table, df, verb)
            _ledger_mark(conn, table, p, etag)
            conn.commit()
            n += 1
    if n:
        print(f"  {table}: {'待合并' if dry_run else '已合并'} {n} 个月度分片")
    return n


def _download_dim(bucket, conn, table, cfg, dry_run):
    if not _table_exists(conn, table):
        print(f"  ⚠️ 本地缺表 {table} (先跑应用建表), 跳过")
        return 0
    key = f"{OSS_PREFIX}/full/{table}.parquet"
    if not bucket.object_exists(key):
        return 0
    etag = bucket.head_object(key).etag
    if _ledger_seen(conn, table, '_snapshot', etag):
        return 0
    if dry_run:
        print(f"  [DRY] {table} 全量快照待替换")
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"{table}.parquet")
        bucket.get_object_to_file(key, path)
        df = pd.read_parquet(path)
    conn.execute(f"DELETE FROM {table}")
    _insert(conn, table, df, 'INSERT')
    _ledger_mark(conn, table, '_snapshot', etag)
    conn.commit()
    print(f"  {table}: 已替换全量快照 {len(df)}行")
    return 1


def download(tables=None, dry_run=False, full=False):
    import sqlite3
    _load_env()
    bucket = _bucket()

    if full:
        key = f"{OSS_PREFIX}/stock_data.db"
        print(f"📥 --full 整库 bootstrap <- {key}")
        if not dry_run:
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            bucket.get_object_to_file(key, DB_PATH)
            print("✅ 整库下载完成")
        return

    print(f"📥 增量下载 OSS  bucket={os.environ.get('OSS_BUCKET')}  prefix={OSS_PREFIX}")
    conn = sqlite3.connect(DB_PATH)
    try:
        _ensure_ledger(conn)
        names = tables or list(TABLES.keys())
        total = 0
        for t in names:
            cfg = TABLES.get(t)
            if not cfg:
                print(f"  ⚠️ 未知表 {t}, 跳过")
                continue
            if cfg['kind'] == 'ts':
                total += _download_ts(bucket, conn, t, cfg, dry_run)
            else:
                total += _download_dim(bucket, conn, t, cfg, dry_run)
        print(f"✅ 下载完成 (新增/更新 {total} 个分片)")
    finally:
        conn.close()


# ---------------------------------------------------------------- cli

def main():
    p = argparse.ArgumentParser(description='OSS 按月增量 / 全量快照同步')
    p.add_argument('action', choices=['upload', 'download'])
    p.add_argument('--producer', choices=list(PRODUCER_TABLES),
                   help='按生产者选表: mac=只 kline_30m / server=其余全部 (与 --tables 二选一)')
    p.add_argument('--tables', help='逗号分隔, 只处理这些表 (默认全部)')
    p.add_argument('--days', type=int, help='upload: 只扫描最近 N 个分片周期(月), 性能上限')
    p.add_argument('--dry-run', action='store_true', help='只打印计划, 不实际传输')
    p.add_argument('--full', action='store_true', help='整库快照 (bootstrap/备份)')
    args = p.parse_args()

    if args.producer:
        tables = PRODUCER_TABLES[args.producer]
    elif args.tables:
        tables = [t.strip() for t in args.tables.split(',')]
    else:
        tables = None
    if args.action == 'upload':
        upload(tables=tables, days=args.days, dry_run=args.dry_run, full=args.full)
    else:
        download(tables=tables, dry_run=args.dry_run, full=args.full)


if __name__ == '__main__':
    main()
