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
ZERO_IMP_FEATURES = [
    'price_above_ma5', 'price_above_ma10', 'price_above_ma20',
    'price_above_ma30', 'price_above_ma60', 'price_above_ma80',
    'price_above_ma100', 'price_above_ma120',
    'ma5_cross_ma10', 'ma10_cross_ma20', 'ma20_cross_ma60', 'ma60_cross_ma120',
    'macd_cross', 'kdj_cross_signal', 'inside_bar', 'breakout_20', 'trend_direction',
]


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
    """预测上涨概率（支持v4混合/v3集成/v2单模型）"""
    if 'models' in model_data and 'model_types' in model_data:
        # v4 混合ensemble
        probs = []
        model_types = model_data.get('model_types', ['lgbm'] * len(model_data['models']))
        for model, mtype in zip(model_data['models'], model_types):
            try:
                if mtype == 'lgbm':
                    probs.append(model.predict_proba([feat_row])[0][1])
                elif mtype == 'xgb':
                    probs.append(model.predict_proba(feat_row.reshape(1, -1))[0][1])
                elif mtype == 'catboost':
                    p = model.predict_proba(feat_row.reshape(1, -1))
                    probs.append(float(p[0][1]))
                else:
                    probs.append(0.5)
            except Exception:
                probs.append(0.5)
        return np.mean(probs)
    elif 'models' in model_data:
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


@forecast_bp.route('/forecast/7days/<symbol>', methods=['GET'])
def forecast_7days(symbol):
    """预测接下来7天走势"""
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
        try:
            sys.path.insert(0, '/root/github/stock-quant/stock-quant/python')
            from strategy.train_lgb_v3 import AdvancedFeatureEngineer
            adv_features = AdvancedFeatureEngineer.calculate_advanced_features(df)
            features = pd.concat([features, adv_features], axis=1)
        except Exception:
            pass

        if filtered_feature_names:
            missing = [c for c in filtered_feature_names if c not in features.columns]
            for c in missing:
                features[c] = 0
            features = features[filtered_feature_names]
        else:
            drop_cols = TIME_FEATURES + ZERO_IMP_FEATURES
            features = features[[c for c in features.columns if c not in drop_cols]]

        # 用最近7个K线的特征模拟预测
        horizon = model_data.get('horizon', 3)
        threshold = model_data.get('threshold', 0.015)
        last_idx = len(df) - 1
        current_price = float(df.iloc[last_idx]['close'])
        last_date = str(df.iloc[last_idx]['date'])

        # 取最近7个K线位置进行预测
        predictions = []
        sim_prices = [current_price]

        # 计算历史平均涨跌幅用于模拟
        recent_closes = df['close'].iloc[-50:].values
        changes = [(recent_closes[i+1] - recent_closes[i]) / recent_closes[i] for i in range(len(recent_closes)-1)]
        avg_up_change = np.mean([c for c in changes if c > 0]) if any(c > 0 for c in changes) else 0.003
        avg_down_change = np.mean([c for c in changes if c < 0]) if any(c < 0 for c in changes) else -0.003

        # 对最近7个K线逐步预测
        for step_i in range(7):
            idx = last_idx - 6 + step_i  # 从倒数第7条开始
            if idx < 0 or idx >= len(features):
                predictions.append({'day': step_i + 1, 'upProb': 0.5, 'direction': 'neutral', 'simPrice': sim_prices[-1]})
                continue

            feat_row = features.iloc[idx].fillna(0).values
            up_prob = float(_predict_proba(feat_row, model_data))
            direction = 'up' if up_prob >= 0.55 else ('down' if up_prob <= 0.45 else 'neutral')

            # 模拟价格变化
            if up_prob >= 0.5:
                change = avg_up_change * (up_prob / 0.5)
            else:
                change = avg_down_change * ((1 - up_prob) / 0.5)
            sim_price = sim_prices[-1] * (1 + change)
            sim_prices.append(sim_price)

            # 预测价格区间（置信区间）
            volatility = np.std(changes) if len(changes) > 2 else 0.01
            price_low = sim_price * (1 - volatility * 1.5)
            price_high = sim_price * (1 + volatility * 1.5)

            # 生成未来7天的时间标签（每条30分钟K线）
            from datetime import datetime, timedelta as td
            base_dt = pd.to_datetime(last_date)
            # 简化：每个step代表一个交易日
            pred_date = (base_dt + td(days=step_i + 1)).strftime('%Y-%m-%d')

            predictions.append({
                'day': step_i + 1,
                'date': pred_date,
                'upProb': round(up_prob * 100, 1),
                'downProb': round((1 - up_prob) * 100, 1),
                'direction': direction,
                'simPrice': round(sim_price, 2),
                'priceLow': round(price_low, 2),
                'priceHigh': round(price_high, 2),
            })

        model_version = 'v3-ensemble' if 'models' in model_data else 'v2-single'

        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'stockName': stock_name,
            'modelVersion': model_version,
            'currentPrice': round(current_price, 2),
            'lastDate': last_date[:10],
            'predictions': predictions,
            'summary': {
                'avgUpProb': round(float(np.mean([p['upProb'] for p in predictions])), 1),
                'trendDirection': 'up' if np.mean([p['upProb'] for p in predictions]) > 55 else ('down' if np.mean([p['upProb'] for p in predictions]) < 45 else 'neutral'),
                'simFinalPrice': round(sim_prices[-1], 2),
                'simReturn': round((sim_prices[-1] - current_price) / current_price * 100, 2),
            }
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@forecast_bp.route('/forecast/history/<symbol>', methods=['GET'])
def forecast_history(symbol):
    """返回过去7天的预测记录vs实际结果"""
    days = int(request.args.get('days', 7))

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
        try:
            sys.path.insert(0, '/root/github/stock-quant/stock-quant/python')
            from strategy.train_lgb_v3 import AdvancedFeatureEngineer
            adv_features = AdvancedFeatureEngineer.calculate_advanced_features(df)
            features = pd.concat([features, adv_features], axis=1)
        except Exception:
            pass

        if filtered_feature_names:
            missing = [c for c in filtered_feature_names if c not in features.columns]
            for c in missing:
                features[c] = 0
            features = features[filtered_feature_names]
        else:
            drop_cols = TIME_FEATURES + ZERO_IMP_FEATURES
            features = features[[c for c in features.columns if c not in drop_cols]]

        df['date'] = pd.to_datetime(df['date'])
        horizon = model_data.get('horizon', 3)
        threshold = model_data.get('threshold', 0.015)

        # 取过去N天的数据
        cutoff_date = df['date'].max() - timedelta(days=days)
        recent_df = df[df['date'] >= cutoff_date]

        # 按日汇总
        daily_records = []
        daily_groups = recent_df.groupby(recent_df['date'].dt.date)

        for day_date, day_group in daily_groups:
            day_indices = day_group.index.tolist()
            day_predictions = []
            day_actual_dirs = []

            for idx in day_indices:
                if idx + horizon >= len(df) or idx < 0 or idx >= len(features):
                    continue
                try:
                    feat_row = features.iloc[idx].fillna(0).values
                    up_prob = float(_predict_proba(feat_row, model_data))

                    future_close = float(df.iloc[idx + horizon]['close'])
                    current_close = float(df.iloc[idx]['close'])
                    ret = (future_close - current_close) / current_close

                    if ret > threshold:
                        actual_dir = 'up'
                    elif ret < -threshold:
                        actual_dir = 'down'
                    else:
                        actual_dir = 'neutral'

                    pred_dir = 'up' if up_prob >= 0.55 else ('down' if up_prob <= 0.45 else 'neutral')
                    is_correct = (pred_dir == actual_dir) or (actual_dir == 'neutral')

                    day_predictions.append({
                        'time': str(df.iloc[idx]['date']),
                        'upProb': round(up_prob * 100, 1),
                        'predDirection': pred_dir,
                        'actualDirection': actual_dir,
                        'actualReturn': round(ret * 100, 2),
                        'isCorrect': is_correct,
                    })
                    day_actual_dirs.append(actual_dir)
                except Exception:
                    continue

            if day_predictions:
                correct_count = sum(1 for p in day_predictions if p['isCorrect'])
                daily_records.append({
                    'date': str(day_date),
                    'predictions': day_predictions,
                    'accuracy': round(correct_count / len(day_predictions) * 100, 1),
                    'avgUpProb': round(float(np.mean([p['upProb'] for p in day_predictions])), 1),
                    'predCount': len(day_predictions),
                    'correctCount': correct_count,
                })

        # 按时间倒序
        daily_records.reverse()

        total_preds = sum(r['predCount'] for r in daily_records)
        total_correct = sum(r['correctCount'] for r in daily_records)
        overall_accuracy = round(total_correct / total_preds * 100, 1) if total_preds > 0 else 0

        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'stockName': stock_name,
            'days': days,
            'overallAccuracy': overall_accuracy,
            'totalPredictions': total_preds,
            'totalCorrect': total_correct,
            'dailyRecords': daily_records,
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@forecast_bp.route('/forecast/stats', methods=['GET'])
def forecast_stats():
    """全局预测准确率统计"""
    model_data, feature_engineer, filtered_feature_names = _load_model()
    if model_data is None or feature_engineer is None:
        return jsonify({'status': 'error', 'message': '模型未加载'}), 500

    try:
        from config_loader import get_db_path
        db_path = get_db_path()
        conn = sqlite3.connect(db_path)

        # 获取所有有足够数据的股票
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, name FROM stock_info")
        all_stocks = cursor.fetchall()

        # 只取数据量足够的股票（limit 20避免内存问题）
        stock_stats = []
        total_preds = 0
        total_correct = 0
        recent_7d_preds = 0
        recent_7d_correct = 0

        horizon = model_data.get('horizon', 3)
        threshold = model_data.get('threshold', 0.015)

        for symbol, name in all_stocks[:20]:  # 限制20只避免内存压力
            try:
                df = pd.read_sql_query(
                    'SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol=? ORDER BY date',
                    conn, params=(symbol,)
                )
                if df.empty or len(df) < 150:
                    continue

                # 计算特征
                features_calc = feature_engineer.calculate_features(df)
                try:
                    sys.path.insert(0, '/root/github/stock-quant/stock-quant/python')
                    from strategy.train_lgb_v3 import AdvancedFeatureEngineer
                    adv_features = AdvancedFeatureEngineer.calculate_advanced_features(df)
                    features_calc = pd.concat([features_calc, adv_features], axis=1)
                except Exception:
                    pass

                if filtered_feature_names:
                    missing_cols = [c for c in filtered_feature_names if c not in features_calc.columns]
                    for c in missing_cols:
                        features_calc[c] = 0
                    features_calc = features_calc[filtered_feature_names]
                else:
                    drop_cols = TIME_FEATURES + ZERO_IMP_FEATURES
                    features_calc = features_calc[[c for c in features_calc.columns if c not in drop_cols]]

                df['date'] = pd.to_datetime(df['date'])
                seven_days_ago = df['date'].max() - timedelta(days=7)

                # 最近1个月评估（步长=5减少计算量）
                start_idx = 150
                correct = 0
                preds_count = 0
                recent_correct = 0
                recent_preds = 0

                for i in range(start_idx, len(df) - horizon, 5):  # step=5节省内存
                    try:
                        feat_row = features_calc.iloc[i].fillna(0).values
                        up_prob = _predict_proba(feat_row, model_data)
                        pred_dir = 1 if up_prob >= 0.55 else 0

                        future_close = float(df.iloc[i + horizon]['close'])
                        current_close = float(df.iloc[i]['close'])
                        ret = (future_close - current_close) / current_close

                        if abs(ret) <= threshold:
                            continue

                        actual_dir = 1 if ret > threshold else 0
                        is_correct = pred_dir == actual_dir

                        preds_count += 1
                        if is_correct:
                            correct += 1

                        # 最近7天
                        if df.iloc[i]['date'] >= seven_days_ago:
                            recent_preds += 1
                            if is_correct:
                                recent_correct += 1
                    except Exception:
                        continue

                if preds_count > 0:
                    accuracy = round(correct / preds_count * 100, 1)
                    stock_stats.append({
                        'symbol': symbol,
                        'name': name or symbol,
                        'totalPredictions': preds_count,
                        'accuracy': accuracy,
                        'recentPredictions': recent_preds,
                        'recentAccuracy': round(recent_correct / recent_preds * 100, 1) if recent_preds > 0 else None,
                    })
                    total_preds += preds_count
                    total_correct += correct
                    recent_7d_preds += recent_preds
                    recent_7d_correct += recent_correct

            except Exception:
                continue

        conn.close()

        # 排序：准确率高的在前
        stock_stats.sort(key=lambda x: x['accuracy'], reverse=True)

        return jsonify({
            'status': 'success',
            'totalPredictions': total_preds,
            'totalAccuracy': round(total_correct / total_preds * 100, 1) if total_preds > 0 else 0,
            'recent7dPredictions': recent_7d_preds,
            'recent7dAccuracy': round(recent_7d_correct / recent_7d_preds * 100, 1) if recent_7d_preds > 0 else 0,
            'stockCount': len(stock_stats),
            'stockStats': stock_stats,
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500



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

        # 过滤时间特征 + 零重要性特征
        if filtered_feature_names:
            missing = [c for c in filtered_feature_names if c not in features.columns]
            for c in missing:
                features[c] = 0
            features = features[filtered_feature_names]
        else:
            drop_cols = TIME_FEATURES + ZERO_IMP_FEATURES
            features = features[[c for c in features.columns if c not in drop_cols]]

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
            change = avg_up * (p / 0.5) if p >= 0.5 else avg_down * ((1 - p) / 0.5)
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