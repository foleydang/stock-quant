"""补仓顾问路由 — 加载 add_advisor/model.pkl, 对持仓给出 补/不补/观望 建议

复用 python/strategy/add_advisor_ml.py 的打分逻辑 (score_holding / _verdict),
不重复实现特征/标签/隘口计算。模型由 Mac 完整训练后 git 提交, 服务器 pull 即用。
"""

from flask import Blueprint, jsonify, request
import sys
import os
import json
import time
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))

from config_loader import get_db_path
from strategy.features import FeaturePipeline
from strategy.add_advisor_ml import (
    load_final_model, score_holding, scan_universe, _verdict, load_holdings,
    PURGE_DAYS,
)

advisor_bp = Blueprint('advisor', __name__)

_advisor = None       # {model dict, pipeline}

# 扫描结果磁盘缓存 (全票池打分较慢, 缓存 6 小时; ?refresh=1 强制重算)
_SCAN_CACHE = os.path.join(
    os.path.dirname(__file__), '../../python/data/advisor_scan.json')
_SCAN_TTL = 6 * 3600

# Mac 离线算好的诚实盈利回测 (backtest_advisor.py), 服务器只读, 不实时跑 walk-forward
_BT_DIR = os.path.join(os.path.dirname(__file__), '../../python/models/add_advisor')
_BT_PORTFOLIO = os.path.join(_BT_DIR, 'backtest_portfolio.json')
_BT_SIGNALS = os.path.join(_BT_DIR, 'backtest_signals.json')
_bt_signals_cache = None   # 懒加载, 全局缓存 (~300KB)


def _load_bt_signals():
    global _bt_signals_cache
    if _bt_signals_cache is not None:
        return _bt_signals_cache
    if not os.path.exists(_BT_SIGNALS):
        return {}
    try:
        with open(_BT_SIGNALS) as f:
            _bt_signals_cache = json.load(f)
    except Exception:
        _bt_signals_cache = {}
    return _bt_signals_cache


def _load_advisor():
    """加载模型 + 构建 pipeline, 全局缓存 (与 forecast_routes 同范式)。

    lstm_slim=True: 用瘦身版 LSTM embeddings (~0.1MB), 避免 273MB 全量在
    1.8GB 服务器上 OOM; 最后一行 lstm 特征与全量逐位相同。
    """
    global _advisor
    if _advisor is not None:
        return _advisor
    try:
        data = load_final_model()
    except FileNotFoundError:
        return None
    pipeline = FeaturePipeline({
        'label': '日线', 'horizon': data['horizon'], 'db_table': 'kline_daily',
        'min_history': 120, 'purged_gap': PURGE_DAYS, 'north_shift_days': 1,
        'lstm_slim': True,
    })
    _advisor = {'data': data, 'pipeline': pipeline}
    return _advisor


@advisor_bp.route('/advisor/holdings', methods=['GET'])
def advisor_holdings():
    adv = _load_advisor()
    if adv is None:
        return jsonify({
            'status': 'error',
            'message': '补仓顾问模型未就绪 (models/add_advisor/model.pkl 缺失), '
                       '请在 Mac 跑 python strategy/add_advisor_ml.py 训练后提交',
        }), 200

    data = adv['data']
    pipeline = adv['pipeline']
    a2_ok = data.get('a2_usable', False)
    a3_ok = data.get('a3_usable', False)

    conn = sqlite3.connect(get_db_path())
    holdings = load_holdings(conn)
    items = []
    for sym, name, shares, cost in holdings:
        try:
            s = score_holding(conn, pipeline, sym, data['feat_names'],
                              data['reg'], data['clf_s'], data['clf_tb'])
        except Exception as e:
            s = None
            err = str(e)
        if s is None:
            items.append({'symbol': sym, 'name': name, 'ready': False})
            continue
        pnl_pct = (s['last'] - cost) / cost * 100 if cost else None
        items.append({
            'symbol': sym,
            'name': name,
            'ready': True,
            'dataDate': s['date'],
            'lastPrice': round(s['last'], 3),
            'costPrice': cost,
            'pnlPct': round(pnl_pct, 1) if pnl_pct is not None else None,
            'rsi': round(s['rsi'], 0),
            'candidate': bool(s['cand']),          # 补仓候选态 (跌破MA20+超卖)
            'ret20Pred': round(s['reg'], 4),        # 方案2: 预测20日收益
            'upProb': round(s['pup'], 3),           # 方案2: 上涨概率
            'tpProb': round(s['ptp'], 3),           # 方案3: P(先触止盈)
            'tpPrice': round(s['tp_price'], 3),
            'slPrice': round(s['sl_price'], 3),
            'verdict': _verdict(s, a2_ok, a3_ok),
        })
    conn.close()

    return jsonify({
        'status': 'success',
        'trainDate': data.get('train_date'),
        'cutoff': data.get('cutoff'),
        'horizon': data.get('horizon'),
        'a2Usable': a2_ok,
        'a3Usable': a3_ok,
        'caveat': 'edge 薄 + 港股/ETF无宏观情绪特征(填0) + 5只样本少→靠池化外推, 仅辅助排序',
        'holdings': items,
    })


# 分位分桶阈值 (edge 薄, 用横截面相对排名而非绝对阈值)
_BUCKETS = [
    (0.10, 'strong_buy', '强烈买入'),
    (0.25, 'buy', '买入'),
    (0.75, 'hold', '持有'),
    (0.90, 'sell', '卖出'),
    (1.01, 'strong_sell', '强烈卖出'),
]


def _bucket_by_rank(idx, n):
    """idx: 从高分到低分的 0-based 排名; 返回 (key, 中文信号)"""
    q = (idx + 0.5) / max(n, 1)
    for thr, key, label in _BUCKETS:
        if q < thr:
            return key, label
    return 'strong_sell', '强烈卖出'


def _build_scan_payload(scored, data):
    """scored: scan_universe 结果; 按预测 20 日收益(reg)横截面排名分桶"""
    items = [s for s in scored if s is not None]
    items.sort(key=lambda s: s['reg'], reverse=True)
    n = len(items)

    dist = {'strong_buy': 0, 'buy': 0, 'hold': 0, 'sell': 0, 'strong_sell': 0}
    signals = {'strong_buy': [], 'buy': [], 'sell': [], 'strong_sell': []}
    pred_date = ''
    for i, s in enumerate(items):
        key, label = _bucket_by_rank(i, n)
        dist[key] += 1
        pred_date = max(pred_date, s.get('date', '') or '')
        if key in signals:
            signals[key].append({
                'rank': i + 1,
                'symbol': s['sym'],
                'name': s.get('name', s['sym']),
                'score': round(s['reg'], 4),
                'signal': label,
                'upProb': round(s['pup'], 3),
                'tpProb': round(s['ptp'], 3),
                'candidate': bool(s['cand']),
            })
    pred_date = (pred_date or '')[:10].replace('-', '')
    return {
        'status': 'success',
        'predDate': pred_date,
        'totalStocks': n,
        'distribution': dist,
        'signals': signals,
        'trainDate': data.get('train_date'),
        'cutoff': data.get('cutoff'),
        'horizon': data.get('horizon'),
        'caveat': 'edge 薄(横截面 rank-IC≈0.05), 仅 A 股; 按预测20日收益相对排名分桶, 非绝对信号',
        'generatedAt': time.strftime('%Y-%m-%d %H:%M'),
    }


@advisor_bp.route('/advisor/scan', methods=['GET'])
def advisor_scan():
    """对全 A 股票池用 add_advisor 模型打分, 供每日信号页扫描 (~50+ 只)。

    结果磁盘缓存 6 小时; ?refresh=1 强制重算。
    """
    refresh = request.args.get('refresh') in ('1', 'true', 'yes')
    if not refresh and os.path.exists(_SCAN_CACHE):
        age = time.time() - os.path.getmtime(_SCAN_CACHE)
        if age < _SCAN_TTL:
            try:
                with open(_SCAN_CACHE) as f:
                    payload = json.load(f)
                payload['cached'] = True
                payload['cacheAgeMin'] = int(age / 60)
                return jsonify(payload)
            except Exception:
                pass

    adv = _load_advisor()
    if adv is None:
        return jsonify({
            'status': 'error',
            'message': '补仓顾问模型未就绪 (models/add_advisor/model.pkl 缺失)',
        }), 200

    data = adv['data']
    pipeline = adv['pipeline']
    conn = sqlite3.connect(get_db_path())
    try:
        scored = scan_universe(
            conn, pipeline, data['feat_names'],
            data['reg'], data['clf_s'], data['clf_tb'])
    finally:
        conn.close()

    payload = _build_scan_payload(scored, data)
    try:
        os.makedirs(os.path.dirname(_SCAN_CACHE), exist_ok=True)
        with open(_SCAN_CACHE, 'w') as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        pass
    payload['cached'] = False
    return jsonify(payload)


@advisor_bp.route('/advisor/backtest', methods=['GET'])
def advisor_backtest():
    """返回 Mac 离线算好的诚实盈利回测 (横截面 top-K / long-short / 基准)。

    walk-forward 太重不在 1.8GB 服务器实时跑; 直接读 backtest_portfolio.json。
    """
    if not os.path.exists(_BT_PORTFOLIO):
        return jsonify({
            'status': 'error',
            'message': '回测结果未就绪 (backtest_portfolio.json 缺失), '
                       '请在 Mac 跑 python strategy/backtest_advisor.py 后提交',
        }), 200
    try:
        with open(_BT_PORTFOLIO) as f:
            portfolio = json.load(f)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'回测文件解析失败: {e}'}), 200
    portfolio['status'] = 'success'
    return jsonify(portfolio)


@advisor_bp.route('/advisor/predict/<path:symbol>', methods=['GET'])
def advisor_predict(symbol):
    """单只 20 日预测: 当前信号 (score_holding) + OOS 历史摘要 (backtest_signals)。"""
    adv = _load_advisor()
    if adv is None:
        return jsonify({
            'status': 'error',
            'message': '补仓顾问模型未就绪 (models/add_advisor/model.pkl 缺失)',
        }), 200

    data = adv['data']
    pipeline = adv['pipeline']
    a2_ok = data.get('a2_usable', False)
    a3_ok = data.get('a3_usable', False)

    conn = sqlite3.connect(get_db_path())
    try:
        s = score_holding(conn, pipeline, symbol, data['feat_names'],
                          data['reg'], data['clf_s'], data['clf_tb'])
    except Exception as e:
        s = None
        err = str(e)
    finally:
        conn.close()

    current = None
    if s is not None:
        current = {
            'dataDate': s['date'],
            'lastPrice': round(s['last'], 3),
            'rsi': round(s['rsi'], 0),
            'candidate': bool(s['cand']),
            'ret20Pred': round(s['reg'], 4),
            'upProb': round(s['pup'], 3),
            'tpProb': round(s['ptp'], 3),
            'tpPrice': round(s['tp_price'], 3),
            'slPrice': round(s['sl_price'], 3),
            'verdict': _verdict(s, a2_ok, a3_ok),
        }

    oos = _load_bt_signals().get(symbol)

    return jsonify({
        'status': 'success',
        'symbol': symbol,
        'horizon': data.get('horizon'),
        'trainDate': data.get('train_date'),
        'a2Usable': a2_ok,
        'a3Usable': a3_ok,
        'current': current,
        'oos': oos,   # {n, dir_acc, hit_rate_up, mean_ret_up_net, series:[{date,pred,actual}]}
        'caveat': 'edge 薄(横截面 rank-IC≈0.05); 单只择时不如买入持有, 仅作方向参考。'
                  '港股/ETF 无宏观情绪特征(填0), 置信度更低。已扣成本。',
    })
