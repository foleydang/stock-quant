#!/usr/bin/env python3
"""Step 1: 特征计算 → 写入 SQLite (内存恒定)"""
import os, sys, sqlite3, gc, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import get_db_path, get_base_dir
import numpy as np, pandas as pd

HORIZON = 5

FEAT_COLS = [
    'ret_1d', 'ret_5d', 'ret_20d', 'ma5_ratio', 'ma10_ratio', 'ma20_ratio', 'ma60_ratio',
    'vol_5', 'vol_20', 'vol_ratio_5', 'vol_ratio_20', 'ampl', 'ampl_5', 'rsi_6', 'rsi_14',
    'macd', 'macd_sig', 'macd_hist', 'bb_upper', 'bb_lower', 'high_5', 'low_5',
    'ret_lag1', 'ret_lag2', 'ret_lag3', 'ret_lag5'
]

def compute(df):
    """单只股票特征计算"""
    df = df.sort_values('date').reset_index(drop=True)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    feats = pd.DataFrame(index=df.index)
    feats['ret_1d'] = c.pct_change(1)
    feats['ret_5d'] = c.pct_change(5)
    feats['ret_20d'] = c.pct_change(20)
    for w in [5,10,20,60]:
        feats[f'ma{w}_ratio'] = c / c.rolling(w).mean() - 1
    ret = feats['ret_1d']
    feats['vol_5'] = ret.rolling(5).std()
    feats['vol_20'] = ret.rolling(20).std()
    feats['vol_ratio_5'] = v / v.rolling(5).mean()
    feats['vol_ratio_20'] = v / v.rolling(20).mean()
    ampl = (h - l) / c.shift(1)
    feats['ampl'] = ampl
    feats['ampl_5'] = ampl.rolling(5).mean()
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    for w in [6,14]:
        avg_gain = gain.rolling(w).mean()
        avg_loss = loss.rolling(w).mean()
        feats[f'rsi_{w}'] = 100 - 100/(1 + avg_gain/(avg_loss + 1e-8))
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    feats['macd'] = ema12 - ema26
    feats['macd_sig'] = feats['macd'].ewm(span=9).mean()
    feats['macd_hist'] = feats['macd'] - feats['macd_sig']
    ma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    feats['bb_upper'] = (c - (ma20 + 2*std20)) / c
    feats['bb_lower'] = (c - (ma20 - 2*std20)) / c
    feats['high_5'] = c / h.rolling(5).max() - 1
    feats['low_5'] = c / l.rolling(5).min() - 1
    for lag in [1,2,3,5]:
        feats[f'ret_lag{lag}'] = ret.shift(lag)
    feats['label'] = c.shift(-HORIZON) / c - 1
    feats['date'] = df['date']
    feats['symbol'] = df['symbol']
    return feats.dropna()

def main():
    db_path = get_db_path()
    # Clean up stale journal files
    for suffix in ['-journal', '-wal', '-shm']:
        try:
            os.remove(db_path + suffix)
        except:
            pass
    
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")  # faster writes
    conn.execute("PRAGMA synchronous=NORMAL")

    # Drop old table
    conn.execute("DROP TABLE IF EXISTS daily_features")

    # Create table matching DataFrame column order
    col_defs = ",\n            ".join([f"{c} REAL NOT NULL" for c in FEAT_COLS])
    conn.execute(f"""
        CREATE TABLE daily_features (
            date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            label REAL NOT NULL,
            {col_defs},
            PRIMARY KEY(date, symbol)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_df_date ON daily_features(date)")

    # Read symbols from file (avoids slow DISTINCT query on 1.4GB DB)
    sym_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           'data', 'symbols.txt')
    if not os.path.exists(sym_file):
        # Fallback: create from qlib bin data
        qlib_dir = os.path.expanduser('~/.qlib/qlib_data/cn_daily/features')
        if os.path.exists(qlib_dir):
            symbols = sorted([f.replace('.sz', '.SZ') for f in os.listdir(qlib_dir)])
        else:
            symbols = [r[0] for r in conn.execute(
                "SELECT DISTINCT symbol FROM kline_daily ORDER BY symbol"
            ).fetchall()]
    else:
        with open(sym_file) as f:
            symbols = [line.strip() for line in f if line.strip()]
    print(f"Computing features for {len(symbols)} stocks...")

    # Build INSERT with correct column order
    all_cols = ['date', 'symbol', 'label'] + FEAT_COLS
    placeholders = ','.join(['?'] * len(all_cols))
    insert_sql = f"INSERT OR REPLACE INTO daily_features VALUES ({placeholders})"

    total = 0
    for i, sym in enumerate(symbols):
        if (i+1) % 50 == 0:
            print(f"  ... {i+1}/{len(symbols)}, {total} rows")
            gc.collect()

        df = pd.read_sql(
            "SELECT date, symbol, open, high, low, close, volume "
            "FROM kline_daily WHERE symbol=? ORDER BY date",
            conn, params=(sym,)
        )
        if len(df) < 200:
            continue

        feats = compute(df)
        if len(feats) == 0:
            continue

        # Insert with correct column order
        rows = [tuple(r) for r in feats[all_cols].values]
        conn.executemany(insert_sql, rows)
        total += len(feats)

    conn.commit()
    conn.close()
    print(f"\n✅ Done: {total} feature rows")

if __name__ == '__main__':
    main()