"""预测路由 — 全部已退役 (旧 lgb_hs300 v9 血统有数据泄漏)。

诚实口径改用 add_advisor:
  - 20 日预测 / OOS 验证:  GET /api/advisor/predict/<symbol>
  - 横截面盈利回测:        GET /api/advisor/backtest
保留这些 410 桩仅为给残留调用者(旧浏览器缓存等)明确交代, 不再加载任何模型。
"""

from flask import Blueprint, jsonify

forecast_bp = Blueprint('forecast', __name__)


@forecast_bp.route('/forecast/7days/<symbol>', methods=['GET'])
def forecast_7days(symbol):
    """[已退役] 旧 lgb_hs300(v9血统)有泄漏且上线特征名 0/56 匹配→恒输出常数。
    改用诚实的 add_advisor 模型: GET /api/advisor/predict/<symbol> (20日预测)。"""
    return jsonify({
        'status': 'error',
        'retired': True,
        'message': '该预测已退役 (旧模型有数据泄漏且特征不匹配)。'
                   '请改用 /api/advisor/predict/' + symbol + ' 获取诚实的 20 日预测。',
        'redirect': '/api/advisor/predict/' + symbol,
    }), 410


@forecast_bp.route('/forecast/history/<symbol>', methods=['GET'])
def forecast_history(symbol):
    """[已退役] 见 forecast_7days 注释。改用 /api/advisor/predict/<symbol> 的 oos.series。"""
    return jsonify({
        'status': 'error',
        'retired': True,
        'message': '该历史记录已退役 (旧模型有数据泄漏)。'
                   '请改用 /api/advisor/predict/' + symbol + ' 的 oos.series (OOS 预测 vs 实际)。',
        'redirect': '/api/advisor/predict/' + symbol,
    }), 410


@forecast_bp.route('/forecast/stats', methods=['GET'])
def forecast_stats():
    """[已退役] 基于泄漏的 lgb_hs300(v9血统)统计,已不可信。
    诚实口径改用 add_advisor 的横截面回测: GET /api/advisor/backtest。"""
    return jsonify({
        'status': 'error',
        'retired': True,
        'message': '该统计已退役 (旧模型有数据泄漏)。请改用 /api/advisor/backtest 获取诚实的横截面回测口径。',
        'redirect': '/api/advisor/backtest',
    }), 410


@forecast_bp.route('/forecast/accuracy/<symbol>', methods=['GET'])
def forecast_accuracy(symbol):
    """已退役: 旧准确性验证基于泄漏的 lgb_hs300(v9)模型 + kline_30m, 上线特征名
    0/56 匹配→全零输入→恒定输出, 非诚实。前端 预测验证 页已改指向
    /api/advisor/predict/<symbol>(add_advisor 的样本外 OOS 验证)。"""
    return jsonify({
        'status': 'error',
        'gone': True,
        'message': '该接口已退役 (旧模型泄漏且上线特征不匹配)。'
                   '请改用 /api/advisor/predict/%s 查看诚实的样本外验证。' % symbol,
        'redirect': '/api/advisor/predict/' + symbol,
    }), 410
