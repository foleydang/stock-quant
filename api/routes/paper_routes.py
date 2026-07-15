"""纸面交易只读路由 — 暴露 paper.db 的 NAV 曲线 / 持仓 / 成交流水。

引擎(python/strategy/paper_trading.py)在 Mac/服务器各自 --init + 每日 --advance
维护 paper.db(host-local, gitignore)。本路由只读, 不推进不重算, 不碰真实账户库。
"""

from flask import Blueprint, jsonify, request
import os
import sqlite3

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))
from config_loader import get_db_path

paper_bp = Blueprint('paper', __name__)

_PAPER_DB = os.path.join(os.path.dirname(__file__), '../../python/data/paper.db')


def _paper_conn():
    if not os.path.exists(_PAPER_DB):
        return None
    c = sqlite3.connect(_PAPER_DB)
    c.row_factory = sqlite3.Row
    return c


def _not_ready():
    return jsonify({
        'status': 'error',
        'message': '纸面账户未初始化 (paper.db 缺失), 请在主机跑 '
                   'python strategy/paper_trading.py --init 后 --advance',
    }), 200


def _acct(req):
    a = (req.args.get('account', 'A') or 'A').upper()
    return a if a in ('A', 'B') else 'A'


def _names(symbols):
    """symbol -> 中文名 (stock_info 优先, 缺失回退 symbol)。"""
    if not symbols:
        return {}
    src = sqlite3.connect(get_db_path())
    q = ','.join('?' * len(symbols))
    out = {}
    try:
        for sym, nm in src.execute(
                f"SELECT symbol, name FROM stock_info WHERE symbol IN ({q})", list(symbols)):
            out[sym] = nm
        # positions.stock_name 兜底 (持仓可能不在 stock_info)
        for sym, nm in src.execute(
                f"SELECT symbol, stock_name FROM positions WHERE symbol IN ({q})", list(symbols)):
            out.setdefault(sym, nm)
    finally:
        src.close()
    return out


def _close_map(symbols, date):
    """symbol -> (close, actual_date) at ≤date (停牌取最近)。"""
    if not symbols:
        return {}
    src = sqlite3.connect(get_db_path())
    out = {}
    try:
        for sym in symbols:
            r = src.execute(
                "SELECT close, date FROM kline_daily WHERE symbol=? AND date<=? "
                "ORDER BY date DESC LIMIT 1", (sym, date)).fetchone()
            if r and r[0] is not None:
                out[sym] = (float(r[0]), r[1])
    finally:
        src.close()
    return out


@paper_bp.route('/paper/nav', methods=['GET'])
def paper_nav():
    """NAV 曲线 + 基准, 归一化到起点 1.0 (与 PortfolioBacktest 的 {date,value} 结构一致)。"""
    p = _paper_conn()
    if p is None:
        return _not_ready()
    acct = _acct(request)
    try:
        meta = p.execute(
            "SELECT init_capital, launch_date FROM paper_account WHERE account=?",
            (acct,)).fetchone()
        if meta is None:
            return _not_ready()
        rows = p.execute(
            "SELECT date, nav, benchmark, cash, ex_div_flag FROM paper_nav "
            "WHERE account=? ORDER BY date", (acct,)).fetchall()
    finally:
        p.close()

    base = meta['init_capital'] or 0.0
    # 基准归一: 账户A基准是收益指数(base=capital); 账户B基准是 launch 日镜像市值
    bench0 = rows[0]['benchmark'] if rows else base
    nav_curve, bench_curve, exdiv_dates = [], [], []
    for r in rows:
        d = r['date']
        if base > 0:
            nav_curve.append({'date': d, 'value': round(r['nav'] / base, 4)})
        if bench0 and bench0 > 0:
            bench_curve.append({'date': d, 'value': round(r['benchmark'] / bench0, 4)})
        if r['ex_div_flag']:
            exdiv_dates.append(d)

    last = rows[-1] if rows else None
    return jsonify({
        'status': 'success',
        'account': acct,
        'launchDate': meta['launch_date'],
        'initCapital': base,
        'navCurve': nav_curve,          # [{date, value}] 归一起点1.0
        'benchmarkCurve': bench_curve,  # [{date, value}]
        'exDivDates': exdiv_dates,      # 持有期内除权除息(未复权失真)提示
        'benchmarkLabel': '全票池等权(收益指数)' if acct == 'A' else 'launch日持仓买入持有',
        'latest': None if last is None else {
            'date': last['date'],
            'nav': round(last['nav'], 2),
            'benchmark': round(last['benchmark'], 2),
            'cash': round(last['cash'], 2),
            'totalReturn': round(last['nav'] / base - 1, 4) if base > 0 else None,
            'benchmarkReturn': round(last['benchmark'] / bench0 - 1, 4) if bench0 else None,
        },
        'caveat': ('前瞻纸面记账(信号as-of收盘冻结, D+1开盘成交, 已扣真实成本, T+1约束)。'
                   '价格未复权, 持有期除权除息以 exDivDates 标注不静默吸收。'
                   + ('账户A=系统化横截面top-K每20交易日调仓。'
                      if acct == 'A' else '账户B=镜像真实持仓按顾问建议补/减/止损, 自筹现金。')),
    })


@paper_bp.route('/paper/positions', methods=['GET'])
def paper_positions():
    """当前持仓 + mark-to-market 浮盈 (按最新交易日收盘)。"""
    p = _paper_conn()
    if p is None:
        return _not_ready()
    acct = _acct(request)
    try:
        cash = p.execute("SELECT cash FROM paper_account WHERE account=?", (acct,)).fetchone()
        if cash is None:
            return _not_ready()
        cash = cash['cash']
        rows = p.execute(
            "SELECT symbol, SUM(shares) sh, SUM(available) av, "
            "SUM(shares*cost)/SUM(shares) avgcost "
            "FROM paper_position WHERE account=? GROUP BY symbol HAVING SUM(shares)>0",
            (acct,)).fetchall()
    finally:
        p.close()

    syms = [r['symbol'] for r in rows]
    # mark-to-market: 用整库最新交易日收盘
    src = sqlite3.connect(get_db_path())
    latest = src.execute("SELECT MAX(date) FROM kline_daily").fetchone()[0]
    src.close()
    names = _names(syms)
    closes = _close_map(syms, latest)

    items, mkt = [], 0.0
    for r in rows:
        sym = r['symbol']; sh = r['sh']; av = r['av']; cost = r['avgcost']
        cl, cdate = closes.get(sym, (None, None))
        val = sh * cl if cl else 0.0
        mkt += val
        items.append({
            'symbol': sym,
            'name': names.get(sym, sym),
            'shares': round(sh, 0),
            'available': round(av, 0),
            'avgCost': round(cost, 3) if cost else None,
            'lastPrice': round(cl, 3) if cl else None,
            'priceDate': cdate,
            'marketValue': round(val, 2),
            'pnlPct': round((cl / cost - 1) * 100, 2) if (cl and cost) else None,
        })
    items.sort(key=lambda x: x['marketValue'], reverse=True)
    return jsonify({
        'status': 'success',
        'account': acct,
        'cash': round(cash, 2),
        'marketValue': round(mkt, 2),
        'totalValue': round(cash + mkt, 2),
        'markDate': latest,
        'positions': items,
    })


@paper_bp.route('/paper/trades', methods=['GET'])
def paper_trades():
    """成交流水 (最近在前)。"""
    p = _paper_conn()
    if p is None:
        return _not_ready()
    acct = _acct(request)
    try:
        if p.execute("SELECT 1 FROM paper_account WHERE account=?", (acct,)).fetchone() is None:
            return _not_ready()
        rows = p.execute(
            "SELECT date, symbol, side, shares, price, amount, cost, reason "
            "FROM paper_trade WHERE account=? ORDER BY date DESC, id DESC LIMIT 500",
            (acct,)).fetchall()
    finally:
        p.close()
    syms = list({r['symbol'] for r in rows})
    names = _names(syms)
    trades = [{
        'date': r['date'],
        'symbol': r['symbol'],
        'name': names.get(r['symbol'], r['symbol']),
        'side': r['side'],
        'shares': round(r['shares'], 0),
        'price': round(r['price'], 3),
        'amount': round(r['amount'], 2),
        'cost': round(r['cost'], 2),
        'reason': r['reason'],
    } for r in rows]
    return jsonify({'status': 'success', 'account': acct, 'trades': trades})
