#!/bin/bash
# ============================================================
# 每日量化流水线 — 数据更新 + 集成预测 + 信号输出
#
# 用法:
#   ./scripts/daily_pipeline.sh           # 完整流水线
#   ./scripts/daily_pipeline.sh --predict-only  # 只预测(数据已更新)
#   ./scripts/daily_pipeline.sh --update-only   # 只更新数据
#
# 建议 crontab: 0 18 * * 1-5 cd /path/to/stock-quant && ./scripts/daily_pipeline.sh
# ============================================================

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python3"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/pipeline_$(date +%Y%m%d).log"; }

MODE="${1:-full}"

# ====== Step 1: 更新30分钟K线 (新浪API, 372只, ~60秒) ======
if [ "$MODE" != "--predict-only" ]; then
    log "📊 Step 1/6: 更新30分钟K线数据..."
    $PYTHON -c "
import sys, os, time, sqlite3, requests, json
sys.path.insert(0, 'agent')
from config import DB_PATH
conn = sqlite3.connect(DB_PATH)
symbols = [r[0] for r in conn.execute(
    \"SELECT DISTINCT symbol FROM kline_30m WHERE (symbol LIKE '%.SZ' OR symbol LIKE '%.SH')\"
).fetchall()]
total = 0
for i, sym in enumerate(symbols):
    code = sym[:6]
    sc = f'sz{code}' if sym.endswith('.SZ') else f'sh{code}'
    try:
        r = requests.get(
            'https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData',
            params={'symbol': sc, 'scale': '30', 'datalen': 30}, timeout=10)
        if r.status_code != 200: continue
        for row in json.loads(r.text):
            dt = row.get('day','')
            if dt and len(dt)==16: dt += ':00'
            if not dt: continue
            if not conn.execute('SELECT 1 FROM kline_30m WHERE symbol=? AND date=?',(sym,dt)).fetchone():
                conn.execute('INSERT OR IGNORE INTO kline_30m VALUES(?,?,?,?,?,?,?)',
                    (sym, dt, float(row.get('open',0) or 0), float(row.get('close',0) or 0),
                     float(row.get('high',0) or 0), float(row.get('low',0) or 0),
                     float(row.get('volume',0) or 0)))
                total += 1
    except: pass
    time.sleep(0.08)
conn.commit()
conn.close()
print(f'DONE: +{total} bars')
" 2>&1 | tail -1 | while read line; do log "   $line"; done
    log "   ✓ 30min数据更新完成"
fi

# ====== Step 2: 更新HS300指数 + 日线数据 (AKShare) ======
if [ "$MODE" != "--predict-only" ]; then
    log "📈 Step 2/6: AKShare更新HS300 + 日线数据..."
    PYTHONPATH="$PROJECT_ROOT/python" $PYTHON -c "
import akshare as ak, sqlite3, pandas as pd, sys, time
sys.path.insert(0, '$PROJECT_ROOT/agent')
from config import DB_PATH
conn = sqlite3.connect(DB_PATH)

# 2a. HS300 - AKShare日线 (精确)
df_hs = ak.stock_zh_index_daily(symbol='sh000300')
df_hs['date'] = pd.to_datetime(df_hs['date']).dt.strftime('%Y-%m-%d')
db_latest = conn.execute('SELECT MAX(trade_date) FROM hs300_daily').fetchone()[0]
df_new = df_hs[df_hs['date'] > db_latest] if db_latest else df_hs
hs_new = 0
for _, row in df_new.iterrows():
    conn.execute('DELETE FROM hs300_daily WHERE trade_date=?', (row['date'],))
    conn.execute('INSERT INTO hs300_daily(trade_date,open,close,high,low,volume,amount) VALUES(?,?,?,?,?,?,?)',
        (row['date'], float(row['open']), float(row['close']), float(row['high']),
         float(row['low']), float(row['volume']), 0))
    hs_new += 1

# 2b. 清理HS300中的NaN
conn.execute(\"DELETE FROM hs300_daily WHERE close IS NULL OR close = 'None'\")
conn.commit()

# 2c. 个股日线 - AKShare
symbols = [r[0] for r in conn.execute(
    \"SELECT DISTINCT symbol FROM kline_30m WHERE symbol LIKE '%.SZ' OR symbol LIKE '%.SH'\"
).fetchall()]
daily_ok = 0
daily_fail = 0
for sym in symbols:
    code = sym[:6]
    ak_sym = f'sz{code}' if sym.endswith('.SZ') else f'sh{code}'
    try:
        df = ak.stock_zh_a_daily(symbol=ak_sym, adjust='qfq')
        if df is None or df.empty: continue
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        db_max = conn.execute('SELECT MAX(date) FROM kline_daily WHERE symbol=?', (sym,)).fetchone()[0]
        df_n = df[df['date'] > db_max] if db_max else df
        for _, row in df_n.iterrows():
            conn.execute('INSERT OR REPLACE INTO kline_daily(symbol,date,open,high,low,close,volume) VALUES(?,?,?,?,?,?,?)',
                (sym, row['date'], float(row['open']), float(row['high']),
                 float(row['low']), float(row['close']), float(row['volume'])))
            daily_ok += 1
        daily_ok += 0
    except:
        daily_fail += 1
    time.sleep(0.08)
conn.commit()

# 验证
df_v = pd.read_sql('SELECT trade_date, close FROM hs300_daily ORDER BY trade_date', conn)
df_v['close'] = pd.to_numeric(df_v['close'], errors='coerce')
df_v['ma60'] = df_v['close'].rolling(60).mean()
ma60_val = df_v['ma60'].iloc[-1]
close_val = df_v['close'].iloc[-1]
print(f'DONE: HS300+{hs_new}条 日线+{daily_ok}条 fail={daily_fail} MA60={ma60_val:.0f} close={close_val:.0f}')
conn.close()
" 2>&1 | tail -1 | while read line; do log "   $line"; done
    log "   ✓ AKShare数据更新完成"
fi

# ====== Step 3: 更新 ETF + 港股成分股日线 (新浪+akshare) ======
if [ "$MODE" != "--predict-only" ] && [ "$MODE" != "--update-only-daily" ]; then
    log "📈 Step 3/6: 更新ETF + 港股成分股日线 (update_etf_data.py --hk)..."
    PYTHONPATH="$PROJECT_ROOT/python" $PYTHON "$PROJECT_ROOT/python/update_etf_data.py" --hk \
        2>&1 | grep -E "bars|错误|完成|无数据" | while read line; do log "   $line"; done
    log "   ✓ ETF/港股数据更新完成"
fi

# ====== Step 4: 运行集成预测 (regime-aware) ======
if [ "$MODE" != "--update-only" ] && [ "$MODE" != "--update-only-daily" ]; then
    log "🔮 Step 4/6: 集成预测 (regime-aware)..."
    PYTHONPATH="$PROJECT_ROOT/python" $PYTHON -m strategy.predict_today_batched \
        2>&1 | grep -E "(✅|📊|📈|📉|💾|⏱️|强烈买入|强烈卖出|市场状态)" | while read line; do
        log "   $line"
    done
    log "   ✓ 预测完成"
fi

# ====== Step 5: ETF 159792 专用模型信号 ======
if [ "$MODE" != "--update-only" ] && [ "$MODE" != "--update-only-daily" ]; then
    log "📊 Step 5/6: 159792 港股通互联网ETF 信号..."
    PYTHONPATH="$PROJECT_ROOT/python" $PYTHON "$PROJECT_ROOT/python/strategy/etf159792_model.py" --signal \
        2>&1 | grep -E "现价|规则分|ML:|建议|成本|汇总" | while read line; do log "   $line"; done
    log "   ✓ ETF信号完成"
fi

# ====== Step 4: 输出持仓信号 ======
if [ "$MODE" != "--update-only" ]; then
    log "📋 Step 6/6: 生成持仓建议..."
    TODAY=$(date +%Y%m%d)
    PRED_FILE="$PROJECT_ROOT/models/lgb_hs300_enhanced/prediction_${TODAY}.csv"
    
    if [ -f "$PRED_FILE" ]; then
        PYTHONPATH="$PROJECT_ROOT/python" $PYTHON -c "
import pandas as pd, sqlite3, sys
conn = sqlite3.connect('$PROJECT_ROOT/python/data/stock_data.db')

df = pd.read_csv('$PRED_FILE')

print('='*60)
print('📊 今日持仓信号 — Regime-Aware集成模型')
print('='*60)

# 显示Top买入/卖出
print()
print('🔥 Top-10 强烈买入:')
buy = df[df['signal'].str.contains('强烈买入', na=False)].head(10)
for _, r in buy.iterrows():
    print(f'  {r[\"symbol\"]:12s} {r[\"name\"]:12s} score={r[\"score\"]:+.4f} rank={int(r[\"rank\"]):>3}/341')

print()
print('💀 Top-10 强烈卖出:')
sell = df[df['signal'].str.contains('强烈卖出', na=False)].sort_values('score').head(10)
for _, r in sell.iterrows():
    print(f'  {r[\"symbol\"]:12s} {r[\"name\"]:12s} score={r[\"score\"]:+.4f} rank={int(r[\"rank\"]):>3}/341')

print()
print(f'共 {len(df)} 只股票打分')
conn.close()
" 2>&1 | while read line; do log "   $line"; done
    else
        log "   ⚠️ 预测文件未生成: $PRED_FILE"
    fi
fi

log "✅ 流水线完成 ($MODE)"
