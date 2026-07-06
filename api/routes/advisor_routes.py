"""补仓顾问路由 — 加载 add_advisor/model.pkl, 对持仓给出 补/不补/观望 建议

复用 python/strategy/add_advisor_ml.py 的打分逻辑 (score_holding / _verdict),
不重复实现特征/标签/隘口计算。模型由 Mac 完整训练后 git 提交, 服务器 pull 即用。
"""

from flask import Blueprint, jsonify
import sys
import os
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))

from config_loader import get_db_path
from strategy.features import FeaturePipeline
from strategy.add_advisor_ml import (
    load_final_model, score_holding, _verdict, load_holdings, PURGE_DAYS,
)

advisor_bp = Blueprint('advisor', __name__)

_advisor = None       # {model dict, pipeline}


def _load_advisor():
    """加载模型 + 构建 pipeline, 全局缓存 (与 forecast_routes 同范式)"""
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
