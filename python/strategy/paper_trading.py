"""纸面交易引擎(Paper Trading Engine)

前瞻记账验证 add_advisor 信号是否真能赚钱(区别于有幸存者偏差的历史回测)。
两个独立账户,同一套成本模型(strategy/costs.py):

  账户 A(系统化 top-K): 每 20 交易日按当日冻结 scan 的横截面预测取 top20% 等权持有。
    基准 = 同一票池等权(收益指数,不做整手取整,与回测 universe 口径一致)。
  账户 B(建议跟随): 镜像用户真实 positions,每日按 add_advisor 逐票 verdict
    补/减/持 + 破位止损(自筹现金,不注资)。基准 = launch 日持仓原样买入持有。

诚实性铁律(见 plan sorted-prancing-piglet.md):
  1. 信号 as-of D 收盘冻结,成交按 D+1 开盘价。
  2. --rebuild 只回放 paper_signals,绝不重算模型(score_holding 读全历史无日期
     过滤,回算历史会前视泄漏)。故 freeze 只允许发生在最新交易日(=今天)。
  3. T+1: 仅约束卖出;当日买入次日才可卖(paper_position.lot_date + mark_available)。
  4. 价格用 kline_daily 原始未复权价(与回测同基准);持有期内除权除息会被识别为
     超跌幅(A股 ±10%/创业科创 ±20% 涨跌停以外的跳空)并在 paper_nav.ex_div_flag 标记,
     不静默吸收,也不做复权重构。
  5. 独立库 python/data/paper.db(WAL),不碰 account/account_history/trades。

CLI:
  --init [--capital N]   建库 + 两账户初始态(账户B镜像当前 positions)
  --advance [--date D]   推进到 D(默认=kline_daily 最新日): 冻结信号→T+1解锁→
                         执行到期成交→计算NAV。freeze 要求 D 是最新交易日(防泄漏)。
  --rebuild              清空成交/持仓/NAV,按已存 paper_signals 逐日重放(证明无重算)。
"""

import os
import sys
import json
import time
import sqlite3
import argparse
import datetime as dt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # -> python/
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'python'))

from strategy.costs import trade_cost, roundtrip_frac, is_etf  # noqa: E402

DB_PATH = os.path.join(ROOT, 'data', 'stock_data.db')
PAPER_DB = os.path.join(ROOT, 'data', 'paper.db')
SCAN_JSON = os.path.join(ROOT, 'data', 'advisor_scan.json')
FROZEN_DIR = os.path.join(ROOT, 'data', 'paper_scans')  # 带日期的 scan 冻结副本

# ---------- 配置 ----------
INIT_CAPITAL = 1_000_000       # 账户A初始资金
REBALANCE_DAYS = 20            # 账户A调仓周期(=HORIZON)
TOP_Q = 0.20                   # 横截面前 20% 做多
LOT = 100                      # A股一手
CAL_REF = '600000.SH'          # 交易日历参照(长期活跃)

# 账户B动作阈值(镜像 add_advisor _verdict 的口径)
B_ADD_PTP = 0.55               # 候选态 P(TP) 阈值 → 补
B_TRIM_REG = -0.01             # 预测20日收益 < 此 → 减
B_TRIM_PUP = 0.42              # 且上涨概率 < 此 → 减
B_ADD_FRAC = 0.20              # 补: 加仓额 = min(cash, 0.2×当前市值)
B_TRIM_FRAC = 0.25             # 减: 卖出 25% 可卖股数


def _limit_pct(symbol):
    """A股当日涨跌停幅度: 创业板300/301、科创688 为 20%,北交所 8/4 为 30%,余 10%。"""
    code = (symbol or '').split('.')[0]
    if code.startswith(('300', '301', '688')):
        return 0.20
    if code.startswith(('8', '4')):
        return 0.30
    return 0.10


# ============ DB ============
def _connect(path):
    c = sqlite3.connect(path)
    c.execute('PRAGMA journal_mode=WAL')
    c.execute('PRAGMA foreign_keys=ON')
    return c


def init_db(capital=INIT_CAPITAL, launch_date=None):
    """建 paper.db 并初始化两账户。账户B镜像 stock_data.db 的 positions。"""
    src = _connect(DB_PATH)
    launch_date = launch_date or _latest_bar(src)
    p = _connect(PAPER_DB)
    p.executescript("""
    CREATE TABLE IF NOT EXISTS paper_account(
        account TEXT PRIMARY KEY, cash REAL, init_capital REAL,
        launch_date TEXT, created_at TEXT);
    CREATE TABLE IF NOT EXISTS paper_position(
        id INTEGER PRIMARY KEY AUTOINCREMENT, account TEXT, symbol TEXT,
        shares REAL, cost REAL, available REAL, lot_date TEXT);
    CREATE TABLE IF NOT EXISTS paper_trade(
        id INTEGER PRIMARY KEY AUTOINCREMENT, account TEXT, date TEXT,
        symbol TEXT, side TEXT, shares REAL, price REAL, amount REAL,
        cost REAL, reason TEXT);
    CREATE TABLE IF NOT EXISTS paper_nav(
        account TEXT, date TEXT, nav REAL, benchmark REAL, cash REAL,
        ex_div_flag INTEGER DEFAULT 0, PRIMARY KEY(account, date));
    CREATE TABLE IF NOT EXISTS paper_signals(
        date TEXT, account TEXT, symbol TEXT, reg REAL, up_prob REAL,
        tp_prob REAL, candidate INTEGER, verdict TEXT, action TEXT,
        in_topk INTEGER, as_of_bar TEXT, PRIMARY KEY(date, account, symbol));
    CREATE TABLE IF NOT EXISTS paper_bench_a(
        base REAL, launch_date TEXT);
    """)
    # 清空重建账户态(保留表结构)
    for t in ('paper_account', 'paper_position', 'paper_trade', 'paper_nav', 'paper_bench_a'):
        p.execute(f'DELETE FROM {t}')

    now = time.strftime('%Y-%m-%d %H:%M')
    # 账户A: 全现金
    p.execute("INSERT INTO paper_account VALUES(?,?,?,?,?)",
              ('A', float(capital), float(capital), launch_date, now))
    # 账户A基准: 收益指数, base=capital
    p.execute("INSERT INTO paper_bench_a VALUES(?,?)", (float(capital), launch_date))

    # 账户B: 镜像真实持仓(现金=0,自筹再平衡)
    holdings = src.execute(
        "SELECT symbol, shares, cost_price FROM positions").fetchall()
    p.execute("INSERT INTO paper_account VALUES(?,?,?,?,?)",
              ('B', 0.0, 0.0, launch_date, now))
    b_init = 0.0
    for sym, shares, cost in holdings:
        px = _close_on(src, sym, launch_date)[0]
        if px is None:
            continue
        b_init += shares * px
        # 持仓(可交易)与基准(固定买入持有)各存一份
        p.execute("INSERT INTO paper_position(account,symbol,shares,cost,available,lot_date)"
                  " VALUES('B',?,?,?,?,?)", (sym, shares, cost, shares, launch_date))
        p.execute("INSERT INTO paper_position(account,symbol,shares,cost,available,lot_date)"
                  " VALUES('B_bench',?,?,?,?,?)", (sym, shares, cost, shares, launch_date))
    p.execute("UPDATE paper_account SET init_capital=? WHERE account='B'", (b_init,))
    p.commit()
    src.close(); p.close()
    print(f"✅ paper.db 初始化: launch={launch_date}  账户A={capital:,.0f}  "
          f"账户B镜像持仓市值≈{b_init:,.0f}")


# ============ 行情辅助 ============
def _latest_bar(conn):
    return conn.execute("SELECT MAX(date) FROM kline_daily").fetchone()[0]


def _trading_days(conn, start, end):
    """[start,end] 之间的交易日(用参照票 CAL_REF,回退全表 DISTINCT)。"""
    rows = conn.execute(
        "SELECT DISTINCT date FROM kline_daily WHERE symbol=? AND date>=? AND date<=? "
        "ORDER BY date", (CAL_REF, start, end)).fetchall()
    if not rows:
        rows = conn.execute(
            "SELECT DISTINCT date FROM kline_daily WHERE date>=? AND date<=? ORDER BY date",
            (start, end)).fetchall()
    return [r[0] for r in rows]


def _next_open(conn, symbol, date):
    """D+1(及以后首个交易日)的开盘价与其日期。停牌/退市 → (None,None)。"""
    r = conn.execute(
        "SELECT date, open FROM kline_daily WHERE symbol=? AND date>? "
        "ORDER BY date LIMIT 1", (symbol, date)).fetchone()
    if not r or r[1] is None:
        return None, None
    return float(r[1]), r[0]


def _close_on(conn, symbol, date):
    """date 当日收盘;若停牌(无当日 bar)取 ≤date 最近收盘。返回 (close, actual_date)。"""
    r = conn.execute(
        "SELECT date, close FROM kline_daily WHERE symbol=? AND date<=? "
        "ORDER BY date DESC LIMIT 1", (symbol, date)).fetchone()
    if not r or r[1] is None:
        return None, None
    return float(r[1]), r[0]


def _prev_close(conn, symbol, date):
    r = conn.execute(
        "SELECT close FROM kline_daily WHERE symbol=? AND date<? "
        "ORDER BY date DESC LIMIT 1", (symbol, date)).fetchone()
    return float(r[0]) if r and r[0] is not None else None


# ============ 信号冻结 ============
def freeze_signals(p, src, date, model_adv=None):
    """冻结当日信号到 paper_signals。要求 date 是最新交易日(防 score_holding 前视)。

    账户A: 从 advisor_scan.json 读全票池打分(校验 predDate==date),标记 top-K。
    账户B: 用模型对真实持仓逐票打分(model_adv 已加载则用之,否则跳过B冻结)。
    """
    latest = _latest_bar(src)
    if date != latest:
        raise RuntimeError(
            f"freeze 只允许在最新交易日({latest})进行,拒绝为历史日 {date} 重算(防泄漏)。"
            f" 历史请用 --rebuild 回放已冻结信号。")

    ymd = date.replace('-', '')
    # ---- 账户A: scan ----
    scan = None
    if os.path.exists(SCAN_JSON):
        with open(SCAN_JSON) as f:
            scan = json.load(f)
    if scan and scan.get('predDate') == ymd:
        items = []
        for lst in scan.get('signals', {}).values():
            items.extend(lst)
        items.sort(key=lambda s: s['score'], reverse=True)
        n = len(items)
        k = max(1, int(n * TOP_Q))
        os.makedirs(FROZEN_DIR, exist_ok=True)
        with open(os.path.join(FROZEN_DIR, f'scan_{ymd}.json'), 'w') as f:
            json.dump(scan, f, ensure_ascii=False)
        for i, s in enumerate(items):
            p.execute(
                "INSERT OR REPLACE INTO paper_signals"
                "(date,account,symbol,reg,up_prob,tp_prob,candidate,verdict,action,in_topk,as_of_bar)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (date, 'A', s['symbol'], s['score'], s.get('upProb'), s.get('tpProb'),
                 int(bool(s.get('candidate'))), s.get('signal', ''), 'topk' if i < k else '',
                 1 if i < k else 0, date))
        print(f"  🅰 冻结 scan: {n} 只, top-K={k}")
    else:
        pd_seen = scan.get('predDate') if scan else None
        print(f"  ⚠️ scan 缺失或 predDate({pd_seen})≠{ymd},账户A本日不冻结(调仓日会跳过)")

    # ---- 账户B: 逐持仓打分 ----
    if model_adv is not None:
        from strategy.add_advisor_ml import score_holding, _verdict
        data = model_adv['data']; pipe = model_adv['pipeline']
        a2 = data.get('a2_usable', False); a3 = data.get('a3_usable', False)
        holds = src.execute("SELECT symbol FROM positions").fetchall()
        for (sym,) in holds:
            try:
                s = score_holding(src, pipe, sym, data['feat_names'],
                                  data['reg'], data['clf_s'], data['clf_tb'])
            except Exception:
                s = None
            if s is None:
                continue
            act = _b_action(s, a2, a3)
            p.execute(
                "INSERT OR REPLACE INTO paper_signals"
                "(date,account,symbol,reg,up_prob,tp_prob,candidate,verdict,action,in_topk,as_of_bar)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (date, 'B', sym, s['reg'], s['pup'], s['ptp'], int(bool(s['cand'])),
                 _verdict(s, a2, a3), act, 0, s['date']))
        print(f"  🅱 冻结持仓打分: {len(holds)} 只")
    p.commit()


def _b_action(s, a2_ok, a3_ok):
    """由打分派生账户B机器动作: add / trim / hold(镜像 _verdict 阈值)。"""
    if s['cand'] and a3_ok and s['ptp'] >= B_ADD_PTP and s['reg'] > 0:
        return 'add'
    if s['reg'] < B_TRIM_REG and s['pup'] < B_TRIM_PUP:
        return 'trim'
    return 'hold'


# ============ T+1 ============
def mark_available(p, date):
    """把 lot_date < date 的持仓标记为可卖(T+1 解锁)。"""
    p.execute("UPDATE paper_position SET available=shares WHERE lot_date<? AND account IN('A','B')",
              (date,))
    p.commit()


# ============ 账户A: 系统化 top-K ============
def _acct_positions(p, account):
    rows = p.execute(
        "SELECT symbol, SUM(shares), SUM(available) FROM paper_position "
        "WHERE account=? GROUP BY symbol HAVING SUM(shares)>0", (account,)).fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


def _period_index(src, launch, date):
    """从 launch 到 date 的交易日序号(launch=0)。用于判断 20 日调仓节奏。"""
    days = _trading_days(src, launch, date)
    return len(days) - 1 if days else 0


def advance_account_a(p, src, date):
    """账户A: 仅在调仓日(距 launch 为 REBALANCE_DAYS 整数倍)按冻结 top-K 等权调仓,
    成交价 = date 的次日开盘。非调仓日不动仓。"""
    row = p.execute("SELECT cash, launch_date FROM paper_account WHERE account='A'").fetchone()
    cash, launch = row[0], row[1]
    idx = _period_index(src, launch, date)
    if idx == 0 or idx % REBALANCE_DAYS != 0:
        return  # 非调仓日
    # 目标 top-K(当日冻结)
    tk = [r[0] for r in p.execute(
        "SELECT symbol FROM paper_signals WHERE date=? AND account='A' AND in_topk=1",
        (date,)).fetchall()]
    if not tk:
        print(f"  🅰 {date} 调仓日但无冻结 top-K,跳过")
        return
    # 1) 全部清仓(D+1 开盘卖出)
    pos = _acct_positions(p, 'A')
    for sym, (shares, avail) in pos.items():
        px, fdate = _next_open(src, sym, date)
        if px is None:
            continue  # 停牌无法成交,保留
        amt = shares * px
        fee = trade_cost(sym, amt, 'sell')
        cash += amt - fee
        p.execute("INSERT INTO paper_trade(account,date,symbol,side,shares,price,amount,cost,reason)"
                  " VALUES('A',?,?,'sell',?,?,?,?,'rebalance-out')",
                  (fdate, sym, shares, px, amt, fee))
        p.execute("DELETE FROM paper_position WHERE account='A' AND symbol=?", (sym,))
    # 2) 等权买入 top-K(D+1 开盘)
    tradable = []
    for sym in tk:
        px, fdate = _next_open(src, sym, date)
        if px is not None:
            tradable.append((sym, px, fdate))
    if tradable:
        budget = cash / len(tradable)
        for sym, px, fdate in tradable:
            raw = budget / px
            shares = int(raw // LOT) * LOT
            if shares <= 0:
                continue
            amt = shares * px
            fee = trade_cost(sym, amt, 'buy')
            if amt + fee > cash:
                shares -= LOT; amt = shares * px; fee = trade_cost(sym, amt, 'buy')
            if shares <= 0:
                continue
            cash -= amt + fee
            p.execute("INSERT INTO paper_position(account,symbol,shares,cost,available,lot_date)"
                      " VALUES('A',?,?,?,0,?)", (sym, shares, px, fdate))
            p.execute("INSERT INTO paper_trade(account,date,symbol,side,shares,price,amount,cost,reason)"
                      " VALUES('A',?,?,'buy',?,?,?,?,'rebalance-in')",
                      (fdate, sym, shares, px, amt, fee))
    p.execute("UPDATE paper_account SET cash=? WHERE account='A'", (cash,))
    p.commit()
    print(f"  🅰 {date} 调仓: 买入 {len(tradable)} 只等权, 余现金 {cash:,.0f}")


def _bench_a_period_return(p, src, date):
    """账户A基准(全票池等权收益指数)在调仓日推进一期: 用当日冻结的全票池,
    取 [D+1开盘, 下一调仓日或最新] 区间等权收益,扣一次往返成本。"""
    row = p.execute("SELECT base, launch_date FROM paper_bench_a").fetchone()
    base, launch = row[0], row[1]
    idx = _period_index(src, launch, date)
    if idx == 0 or idx % REBALANCE_DAYS != 0:
        return
    univ = [r[0] for r in p.execute(
        "SELECT symbol FROM paper_signals WHERE date=? AND account='A'", (date,)).fetchall()]
    if not univ:
        return
    rets = []
    for sym in univ:
        o, _ = _next_open(src, sym, date)
        c, _ = _close_on(src, sym, _latest_bar(src))
        if o and c:
            rets.append(c / o - 1)
    if rets:
        ew = sum(rets) / len(rets) - roundtrip_frac(etf=False)
        base = base * (1 + ew)
        p.execute("UPDATE paper_bench_a SET base=?", (base,))
        p.commit()


# ============ 账户B: 建议跟随 ============
def advance_account_b(p, src, date):
    """账户B: 按当日冻结 action(add/trim/hold)在 D+1 开盘执行,受 T+1 约束。
    破位止损: 若真实 positions.stop_loss 设定且 date 收盘破位 → 全卖。"""
    row = p.execute("SELECT cash FROM paper_account WHERE account='B'").fetchone()
    cash = row[0]
    sigs = {r[0]: r[1] for r in p.execute(
        "SELECT symbol, action FROM paper_signals WHERE date=? AND account='B'",
        (date,)).fetchall()}
    stops = {r[0]: r[1] for r in src.execute(
        "SELECT symbol, stop_loss FROM positions WHERE stop_loss IS NOT NULL").fetchall()}

    # 按 symbol 汇总(每 symbol 只决策/成交一次, 避免多 lot 重复执行)
    pos = p.execute(
        "SELECT symbol, SUM(shares), SUM(available) FROM paper_position WHERE account='B'"
        " AND shares>0 GROUP BY symbol").fetchall()
    for sym, shares, avail in pos:
        # 止损优先(纪律 > 模型)
        stop = stops.get(sym)
        cl, _ = _close_on(src, sym, date)
        forced = stop is not None and cl is not None and cl < stop
        action = 'stop' if forced else sigs.get(sym, 'hold')

        if action in ('trim', 'stop') and avail > 0:
            # 只在【可卖】股数上动手(T+1: 当日买入 lot available=0, 天然不参与)
            sell = avail if action == 'stop' else int((avail * B_TRIM_FRAC) // LOT) * LOT
            if sell <= 0:
                continue
            px, fdate = _next_open(src, sym, date)
            if px is None:
                continue
            amt = sell * px; fee = trade_cost(sym, amt, 'sell')
            cash += amt - fee
            _reduce_position(p, 'B', sym, sell)
            p.execute("INSERT INTO paper_trade(account,date,symbol,side,shares,price,amount,cost,reason)"
                      " VALUES('B',?,?,'sell',?,?,?,?,?)",
                      (fdate, sym, sell, px, amt, fee, action))
        elif action == 'add' and cash > 0:
            cur_val = shares * (cl or 0)
            budget = min(cash, B_ADD_FRAC * cur_val)
            px, fdate = _next_open(src, sym, date)
            if px is None or budget <= 0:
                continue
            buy = int((budget / px) // LOT) * LOT
            if buy <= 0:
                continue
            amt = buy * px; fee = trade_cost(sym, amt, 'buy')
            if amt + fee > cash:
                continue
            cash -= amt + fee
            p.execute("INSERT INTO paper_position(account,symbol,shares,cost,available,lot_date)"
                      " VALUES('B',?,?,?,0,?)", (sym, buy, px, fdate))
            p.execute("INSERT INTO paper_trade(account,date,symbol,side,shares,price,amount,cost,reason)"
                      " VALUES('B',?,?,'buy',?,?,?,?,'add')",
                      (fdate, sym, buy, px, amt, fee))
    p.execute("UPDATE paper_account SET cash=? WHERE account='B'", (cash,))
    p.commit()


def _reduce_position(p, account, symbol, qty):
    """按 FIFO 从该 symbol 【可卖】lot 里扣减 qty 股。

    只从 available>0 的 lot 扣(T+1: 当日买入 lot available=0, 结构性不可卖),
    故卖出永远不会动到锁定股, 与调用方计算的可卖额度自洽。
    """
    lots = p.execute(
        "SELECT id, shares, available FROM paper_position WHERE account=? AND symbol=?"
        " AND available>0 ORDER BY lot_date", (account, symbol)).fetchall()
    rem = qty
    for lid, shares, avail in lots:
        if rem <= 0:
            break
        take = min(avail, rem)      # 单 lot 最多卖出其可卖股
        newsh = shares - take
        newav = avail - take
        rem -= take
        if newsh <= 0:
            p.execute("DELETE FROM paper_position WHERE id=?", (lid,))
        else:
            p.execute("UPDATE paper_position SET shares=?, available=? WHERE id=?",
                      (newsh, newav, lid))


# ============ NAV ============
def compute_nav(p, src, date):
    """两账户 mark-to-market(当日收盘)+ 基准,写 paper_nav。"""
    latest = _latest_bar(src)
    # --- 账户A ---
    cashA = p.execute("SELECT cash FROM paper_account WHERE account='A'").fetchone()[0]
    navA = cashA; exdiv = 0
    for sym, (shares, _) in _acct_positions(p, 'A').items():
        cl, _ = _close_on(src, sym, date)
        if cl is None:
            continue
        navA += shares * cl
        exdiv |= _exdiv_hit(src, sym, date)
    baseA = p.execute("SELECT base FROM paper_bench_a").fetchone()[0]
    p.execute("INSERT OR REPLACE INTO paper_nav(account,date,nav,benchmark,cash,ex_div_flag)"
              " VALUES('A',?,?,?,?,?)", (date, navA, baseA, cashA, exdiv))

    # --- 账户B ---
    cashB = p.execute("SELECT cash FROM paper_account WHERE account='B'").fetchone()[0]
    navB = cashB; exdivB = 0
    for sym, (shares, _) in _acct_positions(p, 'B').items():
        cl, _ = _close_on(src, sym, date)
        if cl is None:
            continue
        navB += shares * cl
        exdivB |= _exdiv_hit(src, sym, date)
    # 基准B: launch 日镜像持仓原样买入持有
    benchB = 0.0
    for sym, shares in p.execute(
            "SELECT symbol, shares FROM paper_position WHERE account='B_bench'").fetchall():
        cl, _ = _close_on(src, sym, date)
        if cl is not None:
            benchB += shares * cl
    p.execute("INSERT OR REPLACE INTO paper_nav(account,date,nav,benchmark,cash,ex_div_flag)"
              " VALUES('B',?,?,?,?,?)", (date, navB, benchB, cashB, exdivB))
    p.commit()


def _exdiv_hit(src, symbol, date):
    """识别除权除息/拆股: 当日收盘相对前收盘跌幅超出涨跌停幅度 → 判为公司行为(而非真跌)。"""
    cl, ad = _close_on(src, symbol, date)
    if cl is None or ad != date:
        return 0
    pc = _prev_close(src, symbol, date)
    if pc is None or pc <= 0:
        return 0
    return 1 if (cl / pc - 1) < -(_limit_pct(symbol) + 0.005) else 0


# ============ 推进 / 重放 ============
def advance(date=None, load_model=True):
    src = _connect(DB_PATH)
    p = _connect(PAPER_DB)
    latest = _latest_bar(src)
    date = date or latest
    if not p.execute("SELECT 1 FROM paper_account WHERE account='A'").fetchone():
        src.close(); p.close()
        raise RuntimeError("paper.db 未初始化,请先 --init")
    print(f"▶️  advance {date} (最新bar={latest})")

    model_adv = None
    if load_model and date == latest:
        model_adv = _try_load_model()
    freeze_signals(p, src, date, model_adv)
    mark_available(p, date)
    _bench_a_period_return(p, src, date)
    advance_account_a(p, src, date)
    advance_account_b(p, src, date)
    compute_nav(p, src, date)
    a = p.execute("SELECT nav,benchmark FROM paper_nav WHERE account='A' AND date=?", (date,)).fetchone()
    b = p.execute("SELECT nav,benchmark FROM paper_nav WHERE account='B' AND date=?", (date,)).fetchone()
    print(f"  NAV A={a[0]:,.0f} (基准{a[1]:,.0f})   B={b[0]:,.0f} (基准{b[1]:,.0f})")
    src.close(); p.close()


def rebuild():
    """按已冻结 paper_signals 逐日重放,证明结果不依赖任何重算(无泄漏)。"""
    src = _connect(DB_PATH)
    p = _connect(PAPER_DB)
    launch = p.execute("SELECT launch_date FROM paper_account WHERE account='A'").fetchone()
    if not launch:
        raise RuntimeError("paper.db 未初始化")
    launch = launch[0]
    capital = p.execute("SELECT init_capital FROM paper_account WHERE account='A'").fetchone()[0]
    # 保留 signals,清账户态并重建镜像
    sig_rows = p.execute("SELECT DISTINCT date FROM paper_signals ORDER BY date").fetchall()
    dates = [r[0] for r in sig_rows]
    if not dates:
        print("无冻结信号,无需重放"); src.close(); p.close(); return
    # 重置(不动 paper_signals)
    for t in ('paper_position', 'paper_trade', 'paper_nav'):
        p.execute(f'DELETE FROM {t}')
    p.execute("UPDATE paper_account SET cash=init_capital WHERE account='A'")
    p.execute("DELETE FROM paper_bench_a")
    p.execute("INSERT INTO paper_bench_a VALUES(?,?)", (float(capital), launch))
    # 账户B镜像重建
    _rebuild_b_mirror(p, src, launch)
    p.commit()
    # 逐 signals 日重放(用已冻结信号,绝不 score)
    for d in dates:
        mark_available(p, d)
        _bench_a_period_return(p, src, d)
        advance_account_a(p, src, d)
        advance_account_b(p, src, d)
        compute_nav(p, src, d)
    print(f"♻️ 重放完成: {len(dates)} 个信号日 ({dates[0]}..{dates[-1]})")
    src.close(); p.close()


def _rebuild_b_mirror(p, src, launch):
    p.execute("DELETE FROM paper_position WHERE account IN('B','B_bench')")
    b_init = 0.0
    for sym, shares, cost in src.execute("SELECT symbol,shares,cost_price FROM positions").fetchall():
        px = _close_on(src, sym, launch)[0]
        if px is None:
            continue
        b_init += shares * px
        p.execute("INSERT INTO paper_position(account,symbol,shares,cost,available,lot_date)"
                  " VALUES('B',?,?,?,?,?)", (sym, shares, cost, shares, launch))
        p.execute("INSERT INTO paper_position(account,symbol,shares,cost,available,lot_date)"
                  " VALUES('B_bench',?,?,?,?,?)", (sym, shares, cost, shares, launch))
    p.execute("UPDATE paper_account SET cash=0, init_capital=? WHERE account='B'", (b_init,))


def _try_load_model():
    try:
        from strategy.add_advisor_ml import load_final_model, PURGE_DAYS
        from strategy.features import FeaturePipeline
        data = load_final_model()
        pipe = FeaturePipeline({
            'label': '日线', 'horizon': data['horizon'], 'db_table': 'kline_daily',
            'min_history': 120, 'purged_gap': PURGE_DAYS, 'north_shift_days': 1,
            'lstm_slim': True,
        })
        return {'data': data, 'pipeline': pipe}
    except Exception as e:
        print(f"  ⚠️ 模型未就绪, 账户B本日不冻结: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', action='store_true')
    ap.add_argument('--advance', action='store_true')
    ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--date', default=None)
    ap.add_argument('--capital', type=float, default=INIT_CAPITAL)
    ap.add_argument('--no-model', action='store_true', help='advance 时不加载模型(仅账户A)')
    args = ap.parse_args()

    if args.init:
        init_db(capital=args.capital, launch_date=args.date)
    elif args.rebuild:
        rebuild()
    elif args.advance:
        advance(date=args.date, load_model=not args.no_model)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
