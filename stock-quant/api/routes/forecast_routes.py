"""预测准确性验证路由 - 支持v3集成模型"""

from flask import Blueprint, jsonify, request
import sys
import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
from datetime import timedelta
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))

forecast_bp = Blueprint('forecast', __name__)

_model_data = None
_feature_engineer = None
_filtered_feature_names = None
TIME_FEATURES = ['day_of_week', 'day_of_month', 'hour', 'minute',
                  'is_morning', 'is_afternoon', 'is_first_hour', 'is_last_hour']


def _load_model():
    """加载模型（支持v3集成和v2单模型）"""
    global _model_data, _feature_engineer, _filtered_feature_names
    if _model_data is not None:
        return _model_data, _feature_engineer, _filtered_feature_names

    model_path = '/root/github/stock-quant/stock-quant/python/models/lgb_hs300/model.pkl'
    if not os.path.exists(model_path):
        return None, None, None

    try:
        with open(model_path, 'rb') as f:
            _model_data = pickle.load(f)
    except Exception as e:
        print(f'⚠ 模型加载异常: {e}')
        return None, None, None

    # 判断模型版本
    if 'models' in _model_data:
        # v3 集成模型
        print(f'✓ 加载 v3 集成模型 ({len(_model_data["models"])}个子模型)')
        _filtered_feature_names = _model_data.get('feature_names')
    elif 'model' in _model_data:
        # v2 单模型
        model = _model_data.get('model')
        print(f'✓ 加载 v2 单模型 (n_estimators={model.n_estimators})')
        _filtered_feature_names = _model_data.get('feature_names')
    else:
        print(f'⚠ 未识别的模型格式: {list(_model_data.keys())}')
        return None, None, None

    try:
        from strategy.train_lgb_enhanced import EnhancedFeatureEngineer
        _feature_engineer = EnhancedFeatureEngineer
    except ImportError as e:
        print(f'⚠ FeatureEngineer导入失败: {e}')
        _feature_engineer = None

    return _model_data, _feature_engineer, _filtered_feature_names


def _predict_proba(feat_row, model_data):
    """预测上涨概率（支持集成和单模型）"""
    if 'models' in model_data:
        # v3 集成: 平均概率
        probs = []
        for model in model_data['models']:
            try:
                probs.append(model.predict_proba([feat_row])[0][1])
            except Exception:
                probs.append(0.5)
        return np.mean(probs)
    else:
        # v2/v1 单模型
        model = model_data['model']
        return model.predict_proba([feat_row])[0][1]


def _predict_direction(feat_row, model_data):
    """预测方向（集成用投票，单模型用阈值）"""
    if 'models' in model_data:
        # v3 集成投票
        preds = []
        for model in model_data['models']:
            try:
                preds.append(int(model.predict([feat_row])[0]))
            except Exception:
                preds.append(1)
        return Counter(preds).most_common(1)[0][0]
    else:
        model = model_data['model']
        return int(model.predict([feat_row])[0])


@forecast_bp.route('/forecast/accuracy/<symbol>', methods=['GET'])
def forecast_accuracy(symbol):
    """预测准确性验证"""
    months = int(request.args.get('months', 1))
    step = int(request.args.get('step', 1))

    model_data, feature_engineer, filtered_feature_names = _load_model()
    if model_data is None or feature_engineer is None:
        return jsonify({'status': 'error', 'message': '模型未加载'}), 500

    try:
        from config_loader import get_db_path
        db_path = get_db_path()
        conn = sqlite3.connect(db_path)

        cursor = conn.cursor()
        cursor.execute("SELECT name FROM stock_info WHERE symbol=?", (symbol,))
        row_name = cursor.fetchone()
        stock_name = row_name[0] if row_name and row_name[0] else symbol

        df = pd.read_sql_query(
            'SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol=? ORDER BY date',
            conn, params=(symbol,)
        )
        conn.close()

        if df.empty or len(df) < 150:
            return jsonify({'status': 'error', 'message': f'数据不足（需150条，当前{len(df)}条）'}), 404

        # 计算特征
        features = feature_engineer.calculate_features(df)

        # 高级特征(v3)
        try:
            sys.path.insert(0, '/root/github/stock-quant/stock-quant/python')
            from strategy.train_lgb_v3 import AdvancedFeatureEngineer
            adv_features = AdvancedFeatureEngineer.calculate_advanced_features(df)
            features = pd.concat([features, adv_features], axis=1)
        except Exception:
            pass  # v2模型不需要高级特征

        # 过滤时间特征
        if filtered_feature_names:
            missing = [c for c in filtered_feature_names if c not in features.columns]
            for c in missing:
                features[c] = 0
            features = features[filtered_feature_names]
        else:
            features = features[[c for c in features.columns if c not in TIME_FEATURES]]

        # 评估参数
        horizon = model_data.get('horizon', 3)
        threshold = model_data.get('threshold', 0.015)
        min_history = 150

        df['date'] = pd.to_datetime(df['date'])
        cutoff_date = df['date'].max() - timedelta(days=30 * months)
        cutoff_idx = df[df['date'] >= cutoff_date].index[0] if len(df[df['date'] >= cutoff_date]) > 0 else min_history
        start_idx = max(cutoff_idx, min_history)

        predictions = []
        actual_directions = []
        actual_prices = []
        actual_dates = []

        for i in range(start_idx, len(df) - horizon, step):
            try:
                feat_row = features.iloc[i]
                if feat_row.isna().any():
                    feat_row = feat_row.fillna(0)

                up_prob = _predict_proba(feat_row.values, model_data)

                future_close = float(df.iloc[i + horizon]['close'])
                current_close = float(df.iloc[i]['close'])
                ret = (future_close - current_close) / current_close
                if ret > threshold:
                    actual_up = 1
                elif ret < -threshold:
                    actual_up = 0
                else:
                    continue

                predictions.append(float(up_prob))
                actual_directions.append(int(actual_up))
                actual_prices.append(float(df.iloc[i]['close']))
                actual_dates.append(str(df.iloc[i]['date']))
            except Exception:
                continue

        if not predictions:
            return jsonify({'status': 'error', 'message': '验证区间无有效预测'}), 404

        # 准确率指标
        pred_labels = [_predict_direction(features.iloc[start_idx + j * step].fillna(0).values, model_data)
                       for j in range(len(predictions))]
        correct = sum(1 for p, a in zip(pred_labels, actual_directions) if p == a)
        accuracy = correct / len(predictions) * 100

        up_correct = sum(1 for p, a in zip(pred_labels, actual_directions) if p == 1 and a == 1)
        up_total = sum(1 for p in pred_labels if p == 1)
        up_precision = up_correct / up_total * 100 if up_total > 0 else 0

        down_correct = sum(1 for p, a in zip(pred_labels, actual_directions) if p == 0 and a == 0)
        down_total = sum(1 for p in pred_labels if p == 0)
        down_precision = down_correct / down_total * 100 if down_total > 0 else 0

        # 模拟价格曲线
        changes_pct = [(actual_prices[i+1] - actual_prices[i]) / actual_prices[i]
                       for i in range(len(actual_prices)-1)] if len(actual_prices) > 1 else []
        avg_up = np.mean([c for c in changes_pct if c > 0]) if any(c > 0 for c in changes_pct) else 0.001
        avg_down = np.mean([c for c in changes_pct if c < 0]) if any(c < 0 for c in changes_pct) else -0.001

        predicted_prices = [actual_prices[0]]
        for i in range(1, len(predictions)):
            p = predictions[i - 1]
            change = avg_up_change * (p / 0.5) if p >= 0.5 else avg_down * ((1 - p) / 0.5)
            predicted_prices.append(predicted_prices[-1] * (1 + change))

        final_deviation = (predicted_prices[-1] - actual_prices[-1]) / actual_prices[-1] * 100

        # 日级汇总
        daily_data = []
        current_date = None
        day_actual, day_pred, day_probs, day_dirs = [], [], [], []

        for i in range(len(actual_dates)):
            d = actual_dates[i][:10]
            if d != current_date:
                if current_date and day_actual:
                    daily_data.append({
                        'date': current_date,
                        'actualClose': float(np.mean(day_actual[-4:])) if len(day_actual) >= 4 else float(day_actual[-1]),
                        'predictedClose': float(np.mean(day_pred[-4:])) if len(day_pred) >= 4 else float(day_pred[-1]),
                        'avgProb': float(np.mean(day_probs)),
                        'directionAccuracy': sum(1 for p, a in zip(
                            [1 if pp >= 0.5 else 0 for pp in day_probs], day_dirs
                        ) if p == a) / len(day_dirs) * 100 if day_dirs else 0
                    })
                current_date = d
                day_actual, day_pred, day_probs, day_dirs = [], [], [], []

            day_actual.append(actual_prices[i])
            day_pred.append(predicted_prices[i])
            day_probs.append(predictions[i])
            day_dirs.append(actual_directions[i])

        if current_date and day_actual:
            daily_data.append({
                'date': current_date,
                'actualClose': float(np.mean(day_actual[-4:])) if len(day_actual) >= 4 else float(day_actual[-1]),
                'predictedClose': float(np.mean(day_pred[-4:])) if len(day_pred) >= 4 else float(day_pred[-1]),
                'avgProb': float(np.mean(day_probs)),
                'directionAccuracy': sum(1 for p, a in zip(
                    [1 if pp >= 0.5 else 0 for pp in day_probs], day_dirs
                ) if p == a) / len(day_dirs) * 100 if day_dirs else 0
            })

        model_version = 'v3-ensemble' if 'models' in model_data else 'v2-single'
        n_sub_models = len(model_data.get('models', []))

        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'stockName': stock_name,
            'modelVersion': model_version,
            'nSubModels': n_sub_models,
            'summary': {
                'totalBars': len(predictions),
                'overallAccuracy': round(accuracy, 1),
                'upPrecision': round(up_precision, 1),
                'downPrecision': round(down_precision, 1),
                'avgUpProb': round(float(np.mean(predictions)) * 100, 1),
                'finalDeviation': round(final_deviation, 2),
                'finalActualPrice': round(actual_prices[-1], 2),
                'finalPredictedPrice': round(predicted_prices[-1], 2),
                'months': months,
            },
            'dailyComparison': daily_data,
            'rawPredictions': {
                'dates': actual_dates[-40:],
                'probs': predictions[-40:],
                'actualDirs': actual_directions[-40:],
                'actualPrices': actual_prices[-40:],
                'predictedPrices': predicted_prices[-40:],
            }
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500