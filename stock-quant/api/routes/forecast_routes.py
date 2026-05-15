"""预测准确性验证路由 - LGBM模型预测曲线 vs 真实曲线"""

from flask import Blueprint, jsonify, request
import sys
import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))

forecast_bp = Blueprint('forecast', __name__)

# 全局加载模型（避免每次请求重新加载）
_model = None
_feature_engineer = None
_filtered_feature_names = None

# 需要过滤的时间特征
TIME_FEATURES = ['day_of_week', 'day_of_month', 'hour', 'minute', 'is_morning',
                  'is_afternoon', 'is_first_hour', 'is_last_hour']


def _load_model():
    """加载LGBM模型（单例）"""
    global _model, _feature_engineer, _filtered_feature_names
    if _model is not None:
        return _model, _feature_engineer, _filtered_feature_names

    model_path = '/root/github/stock-quant/stock-quant/python/models/lgb_hs300/model.pkl'
    if not os.path.exists(model_path):
        return None, None, None

    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        _model = model_data.get('model')
        if _model is None:
            print(f'⚠ 模型文件中无model字段: {list(model_data.keys())}')
            return None, None, None
        # 获取模型训练时使用的特征名
        _filtered_feature_names = model_data.get('feature_names', None)
    except Exception as e:
        print(f'⚠ 模型加载异常: {e}')
        return None, None, None

    try:
        from strategy.train_lgb_enhanced import EnhancedFeatureEngineer
        _feature_engineer = EnhancedFeatureEngineer
    except ImportError as e:
        print(f'⚠ FeatureEngineer导入失败: {e}')
        _feature_engineer = None

    return _model, _feature_engineer, _filtered_feature_names


@forecast_bp.route('/forecast/accuracy/<symbol>', methods=['GET'])
def forecast_accuracy(symbol):
    """
    预测准确性验证：
    1. 用最近1年数据作为模型输入
    2. 对每个30分钟K线预测上涨概率
    3. 将预测概率转换为模拟价格曲线
    4. 与真实价格曲线对比
    
    参数：
      - symbol: 股票代码
      - months: 验证月数（默认1，即最近1个月）
      - step: 验证步长（默认1，逐条K线预测）
    """
    months = int(request.args.get('months', 1))
    step = int(request.args.get('step', 1))

    model, feature_engineer, filtered_feature_names = _load_model()
    if model is None or feature_engineer is None:
        return jsonify({'status': 'error', 'message': '模型未加载'}), 500

    try:
        from config_loader import get_db_path
        db_path = get_db_path()
        conn = sqlite3.connect(db_path)

        # 获取股票名称
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM stock_info WHERE symbol=?", (symbol,))
        row_name = cursor.fetchone()
        stock_name = row_name[0] if row_name and row_name[0] else symbol

        # 获取全部30分钟数据（1年以上，为特征计算留足历史）
        df = pd.read_sql_query(
            'SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol=? ORDER BY date',
            conn, params=(symbol,)
        )
        conn.close()

        if df.empty or len(df) < 150:
            return jsonify({'status': 'error', 'message': f'数据不足（需要至少150条，当前{len(df)}条）'}), 404

        # 计算特征
        features = feature_engineer.calculate_features(df)

        # 过滤时间特征（v2模型不使用时间特征）
        if filtered_feature_names:
            # 用模型训练时的特征列表过滤
            missing_cols = [c for c in filtered_feature_names if c not in features.columns]
            for c in missing_cols:
                features[c] = 0  # 补齐缺失列
            features = features[filtered_feature_names]
        else:
            # 旧模型兼容：移除已知时间特征
            keep_cols = [c for c in features.columns if c not in TIME_FEATURES]
            features = features[keep_cols]

        # 确定验证区间：最近 months 个月
        df['date'] = pd.to_datetime(df['date'])
        cutoff_date = df['date'].max() - timedelta(days=30 * months)
        # 特征计算需要120条历史，所以验证开始位置需要足够的前置数据
        min_history = 120  # 特征计算需要的最少历史条数

        # 验证区间的起始索引
        cutoff_idx = df[df['date'] >= cutoff_date].index[0] if len(df[df['date'] >= cutoff_date]) > 0 else min_history
        # 确保验证起点有足够的历史数据用于特征计算
        start_idx = max(cutoff_idx, min_history)

        # 逐条预测（每 step 条取样一次）
        predictions = []     # 预测概率
        actual_directions = []  # 实际涨跌
        actual_prices = []   # 真实价格
        actual_dates = []    # 真实日期

        # 评估参数（与训练一致）
        horizon = 3      # 预测3根K线后
        threshold = 0.015  # 涨跌阈值1.5%

        for i in range(start_idx, len(df) - horizon, step):
            try:
                feat_row = features.iloc[i]
                if feat_row.isna().any():
                    feat_row = feat_row.fillna(0)

                up_prob = model.predict_proba([feat_row.values])[0][1]

                # 真实涨跌：3根K线后 close vs 当前 close，阈值1.5%
                future_close = float(df.iloc[i + horizon]['close'])
                current_close = float(df.iloc[i]['close'])
                ret = (future_close - current_close) / current_close
                if ret > threshold:
                    actual_up = 1   # 明确上涨
                elif ret < -threshold:
                    actual_up = 0   # 明确下跌
                else:
                    continue  # 震荡区间跳过，不参与评估

                predictions.append(float(up_prob))
                actual_directions.append(int(actual_up))
                actual_prices.append(float(df.iloc[i]['close']))
                actual_dates.append(str(df.iloc[i]['date']))

            except Exception as e:
                continue

        if not predictions:
            return jsonify({'status': 'error', 'message': '验证区间无有效预测'}), 404

        # === 准确率指标 ===
        pred_labels = [1 if p >= 0.5 else 0 for p in predictions]
        correct = sum(1 for p, a in zip(pred_labels, actual_directions) if p == a)
        accuracy = correct / len(predictions) * 100

        # 涨跌分别的准确率
        up_pred_correct = sum(1 for p, a in zip(pred_labels, actual_directions) if p == 1 and a == 1)
        up_pred_total = sum(1 for p in pred_labels if p == 1)
        up_precision = up_pred_correct / up_pred_total * 100 if up_pred_total > 0 else 0

        down_pred_correct = sum(1 for p, a in zip(pred_labels, actual_directions) if p == 0 and a == 0)
        down_pred_total = sum(1 for p in pred_labels if p == 0)
        down_precision = down_pred_correct / down_pred_total * 100 if down_pred_total > 0 else 0

        # === 模拟预测价格曲线 ===
        # 思路：每个时间点，模型预测上涨概率p
        # 如果p>=0.5 → 预测价格涨，幅度基于历史平均涨幅
        # 如果p<0.5 → 预测价格跌，幅度基于历史平均跌幅
        # 用实际涨跌幅的平均值来模拟预测的价格变动

        # 计算验证区间的平均涨跌幅
        changes_pct = []
        for i in range(start_idx, len(df) - 1, step):
            pct_change = (float(df.iloc[i + 1]['close']) - float(df.iloc[i]['close'])) / float(df.iloc[i]['close'])
            changes_pct.append(pct_change)

        avg_up_change = np.mean([c for c in changes_pct if c > 0]) if any(c > 0 for c in changes_pct) else 0.001
        avg_down_change = np.mean([c for c in changes_pct if c < 0]) if any(c < 0 for c in changes_pct) else -0.001

        # 从验证起点开始，逐条模拟预测价格
        predicted_prices = [actual_prices[0]]  # 起点用真实价格
        for i in range(1, len(predictions)):
            p = predictions[i - 1]  # 用上一条的预测概率决定当前涨跌
            if p >= 0.5:
                # 预测上涨，幅度用概率加权
                change = avg_up_change * (p / 0.5)
            else:
                # 预测下跌
                change = avg_down_change * ((1 - p) / 0.5)
            predicted_prices.append(predicted_prices[-1] * (1 + change))

        # === 最终偏离度 ===
        final_actual = actual_prices[-1]
        final_predicted = predicted_prices[-1]
        deviation_pct = (final_predicted - final_actual) / final_actual * 100

        # === 日级汇总（减少数据点，前端更易展示）===
        # 把30分钟数据汇总成日线
        daily_data = []
        current_date = None
        day_actual_prices = []
        day_predicted_prices = []
        day_probs = []
        day_directions = []

        for i in range(len(actual_dates)):
            d = actual_dates[i][:10]  # YYYY-MM-DD
            if d != current_date:
                if current_date is not None:
                    daily_data.append({
                        'date': current_date,
                        'actualClose': float(np.mean(day_actual_prices[-4:])) if len(day_actual_prices) >= 4 else float(day_actual_prices[-1]),
                        'predictedClose': float(np.mean(day_predicted_prices[-4:])) if len(day_predicted_prices) >= 4 else float(day_predicted_prices[-1]),
                        'avgProb': float(np.mean(day_probs)),
                        'directionAccuracy': sum(1 for p, a in zip(
                            [1 if pp >= 0.5 else 0 for pp in day_probs],
                            day_directions
                        ) if p == a) / len(day_directions) * 100 if day_directions else 0
                    })
                current_date = d
                day_actual_prices = []
                day_predicted_prices = []
                day_probs = []
                day_directions = []

            day_actual_prices.append(actual_prices[i])
            day_predicted_prices.append(predicted_prices[i])
            day_probs.append(predictions[i])
            day_directions.append(actual_directions[i])

        # 最后一天的数据
        if current_date and day_actual_prices:
            daily_data.append({
                'date': current_date,
                'actualClose': float(np.mean(day_actual_prices[-4:])) if len(day_actual_prices) >= 4 else float(day_actual_prices[-1]),
                'predictedClose': float(np.mean(day_predicted_prices[-4:])) if len(day_predicted_prices) >= 4 else float(day_predicted_prices[-1]),
                'avgProb': float(np.mean(day_probs)),
                'directionAccuracy': sum(1 for p, a in zip(
                    [1 if pp >= 0.5 else 0 for pp in day_probs],
                    day_directions
                ) if p == a) / len(day_directions) * 100 if day_directions else 0
            })

        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'stockName': stock_name,
            'summary': {
                'totalBars': len(predictions),
                'overallAccuracy': round(accuracy, 1),
                'upPrecision': round(up_precision, 1),
                'downPrecision': round(down_precision, 1),
                'avgUpProb': round(float(np.mean(predictions)) * 100, 1),
                'finalDeviation': round(deviation_pct, 2),
                'finalActualPrice': round(final_actual, 2),
                'finalPredictedPrice': round(final_predicted, 2),
                'months': months,
                'avgUpChange': round(avg_up_change * 100, 4),
                'avgDownChange': round(avg_down_change * 100, 4),
            },
            'dailyComparison': daily_data,
            'rawPredictions': {
                'dates': actual_dates[-40:],      # 最近40条原始数据
                'probs': predictions[-40:],
                'actualDirs': actual_directions[-40:],
                'actualPrices': actual_prices[-40:],
                'predictedPrices': predicted_prices[-40:],
            }
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500