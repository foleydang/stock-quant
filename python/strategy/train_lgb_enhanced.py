#!/usr/bin/env python3
"""
增强版 LightGBM 模型训练
特点：
1. 更复杂的特征工程（50+特征）
2. 支持沪深300数据
3. 交叉验证 + 特征选择
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/stock_data.db')


class EnhancedFeatureEngineer:
    """增强版特征工程"""

    FEATURE_NAMES = None

    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算增强版特征（50+特征）
        """
        features = pd.DataFrame(index=df.index)

        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float)
        open_price = df['open'].values.astype(float)

        # ========================================
        # 1. 收益率特征 (10个)
        # ========================================
        for period in [1, 2, 3, 5, 10, 15, 20, 30, 40, 60]:
            features[f'return_{period}'] = pd.Series(close).pct_change(period)

        # ========================================
        # 2. 对数收益率 (5个)
        # ========================================
        for period in [1, 3, 5, 10, 20]:
            features[f'log_return_{period}'] = np.log(pd.Series(close) / pd.Series(close).shift(period))

        # ========================================
        # 3. 波动率特征 (8个)
        # ========================================
        returns = pd.Series(close).pct_change()
        for period in [5, 10, 20, 30, 40, 60, 80, 100]:
            features[f'volatility_{period}'] = returns.rolling(period).std()

        # Parkinson 波动率
        features['parkinson_vol'] = np.sqrt(
            (np.log(pd.Series(high) / pd.Series(low)) ** 2).rolling(20).mean() / (4 * np.log(2))
        )

        # ========================================
        # 4. 均线系统 (16个)
        # ========================================
        for period in [5, 10, 20, 30, 60, 80, 100, 120]:
            ma = pd.Series(close).rolling(period).mean()
            features[f'ma{period}_ratio'] = close / ma - 1
            features[f'price_above_ma{period}'] = (close > ma).astype(int)

        # 均线交叉
        for fast, slow in [(5, 10), (10, 20), (20, 60), (60, 120)]:
            ma_fast = pd.Series(close).rolling(fast).mean()
            ma_slow = pd.Series(close).rolling(slow).mean()
            features[f'ma{fast}_ma{slow}'] = ma_fast / ma_slow - 1
            features[f'ma{fast}_cross_ma{slow}'] = ((ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))).astype(int)

        # ========================================
        # 5. RSI 系列 (4个)
        # ========================================
        for period in [6, 14, 24, 50]:
            delta = pd.Series(close).diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # RSI 背离
        rsi_14 = features['rsi_14']
        price_change = pd.Series(close, index=df.index).diff(20)
        rsi_change = rsi_14.diff(20)
        features['rsi_divergence'] = np.where(
            (price_change.values < 0) & (rsi_change.values > 0), 1,
            np.where((price_change.values > 0) & (rsi_change.values < 0), -1, 0)
        )

        # ========================================
        # 6. MACD (4个)
        # ========================================
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal

        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = histogram
        features['macd_hist_slope'] = histogram.diff()

        # MACD 交叉信号
        features['macd_cross'] = ((macd > signal) & (macd.shift(1) <= signal.shift(1))).astype(int)

        # ========================================
        # 7. KDJ (5个)
        # ========================================
        low_min = pd.Series(low).rolling(9).min()
        high_max = pd.Series(high).rolling(9).max()
        rsv = (close - low_min) / (high_max - low_min + 1e-10) * 100

        features['kdj_k'] = rsv.ewm(com=2).mean()
        features['kdj_d'] = features['kdj_k'].ewm(com=2).mean()
        features['kdj_j'] = 3 * features['kdj_k'] - 2 * features['kdj_d']
        features['kdj_cross'] = features['kdj_k'] - features['kdj_d']
        features['kdj_cross_signal'] = ((features['kdj_k'] > features['kdj_d']) &
                                        (features['kdj_k'].shift(1) <= features['kdj_d'].shift(1))).astype(int)

        # ========================================
        # 8. 布林带 (5个)
        # ========================================
        for period in [20, 30]:
            ma = pd.Series(close).rolling(period).mean()
            std = pd.Series(close).rolling(period).std()
            upper = ma + 2 * std
            lower = ma - 2 * std

            features[f'bb_upper_{period}'] = (upper - close) / close
            features[f'bb_lower_{period}'] = (close - lower) / close
            features[f'bb_width_{period}'] = (upper - lower) / ma
            features[f'bb_position_{period}'] = (close - lower) / (upper - lower + 1e-10)

        # ========================================
        # 9. ATR (3个)
        # ========================================
        tr = pd.concat([
            pd.Series(high) - pd.Series(low),
            pd.Series(high) - pd.Series(close).shift(1),
            pd.Series(close).shift(1) - pd.Series(low)
        ], axis=1).max(axis=1)

        for period in [10, 14, 20]:
            features[f'atr_{period}'] = tr.rolling(period).mean()

        features['atr_ratio'] = features['atr_14'] / pd.Series(close)

        # ========================================
        # 10. 成交量特征 (10个)
        # ========================================
        vol = pd.Series(volume)

        for period in [5, 10, 20, 30, 60]:
            features[f'volume_ma{period}'] = vol.rolling(period).mean()
            features[f'volume_ratio_{period}'] = vol / (features[f'volume_ma{period}'] + 1e-10)

        # 成交量变化率
        features['volume_change'] = vol.pct_change()
        features['volume_acceleration'] = features['volume_change'].diff()

        # OBV
        obv = np.zeros(len(close))
        obv[0] = volume[0]
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]

        features['obv_ma10'] = pd.Series(obv).rolling(10).mean()
        features['obv_ma30'] = pd.Series(obv).rolling(30).mean()
        features['obv_trend'] = pd.Series(obv).diff(10)

        # ========================================
        # 11. 价格形态 (8个)
        # ========================================
        # 影线
        features['upper_shadow'] = (high - np.maximum(open_price, close)) / (close + 1e-10)
        features['lower_shadow'] = (np.minimum(open_price, close) - low) / (close + 1e-10)
        features['body_size'] = np.abs(close - open_price) / (close + 1e-10)

        # 跳空
        features['gap'] = (open_price - pd.Series(close).shift(1)) / (pd.Series(close).shift(1) + 1e-10)

        # 价格位置
        for period in [10, 20, 60]:
            high_roll = pd.Series(high).rolling(period).max()
            low_roll = pd.Series(low).rolling(period).min()
            features[f'price_position_{period}'] = (close - low_roll) / (high_roll - low_roll + 1e-10)
            features[f'high_{period}_ratio'] = (close - high_roll) / (high_roll + 1e-10)

        # ========================================
        # 12. 动量指标 (4个)
        # ========================================
        # 动量
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = close - pd.Series(close).shift(period)

        # CCI
        tp = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
        features['cci'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())

        # ========================================
        # 13. 时间特征 (5个)
        # ========================================
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
            features['hour'] = dates.dt.hour
            features['minute'] = dates.dt.minute
            features['day_of_week'] = dates.dt.dayofweek
            features['day_of_month'] = dates.dt.day
            features['is_month_end'] = dates.dt.is_month_end.astype(int)

            # 交易时段
            features['morning_session'] = ((dates.dt.hour >= 9) & (dates.dt.hour < 12)).astype(int)
            features['afternoon_session'] = ((dates.dt.hour >= 13) & (dates.dt.hour < 15)).astype(int)

        # ========================================
        # 14. 趋势强度 (3个)
        # ========================================
        # ADX
        plus_dm = pd.Series(high).diff()
        minus_dm = pd.Series(low).diff() * -1
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr = features['atr_14']
        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr + 1e-10))
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        features['adx'] = dx.rolling(14).mean()

        # 趋势方向
        features['trend_direction'] = np.where(
            plus_di.values > minus_di.values, 1, np.where(plus_di.values < minus_di.values, -1, 0)
        )

        # 趋势强度
        features['trend_strength'] = features['adx'] * features['trend_direction']

        # 缓存特征名称
        if EnhancedFeatureEngineer.FEATURE_NAMES is None:
            EnhancedFeatureEngineer.FEATURE_NAMES = features.columns.tolist()

        return features


class MarketFeatureEngineer:
    """市场/板块特征工程
    从DB中读取北向资金、大盘涨跌、板块数据，
    合并到30分钟线DataFrame中。
    """

    MARKET_FEATURE_NAMES = None

    @staticmethod
    def calculate_market_features(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        计算市场/板块特征（约8个）
        
        Args:
            df: 30分钟K线DataFrame，必须有 'date' 列
            symbol: 股票代码（如 '600036.SH'），用于查询板块映射
        """
        features = pd.DataFrame(index=df.index)

        # 1. 从date列提取trade_date
        if 'date' not in df.columns:
            return features

        df_dates = pd.to_datetime(df['date'])
        # kline_30m用YYYY-MM-DD, hs300_daily用YYYYMMDD， north_flow用YYYY-MM-DD
        trade_dates_ymd = df_dates.dt.strftime('%Y-%m-%d')
        trade_dates_raw8 = df_dates.dt.strftime('%Y%m%d')  # BaoStock格式

        # 2. 查询北向资金（从north_flow表，YYYY-MM-DD格式）
        try:
            conn = sqlite3.connect(DB_PATH)
            min_date_ymd = trade_dates_ymd.min()
            max_date_ymd = trade_dates_ymd.max()
            north_df = pd.read_sql(
                "SELECT trade_date, total_net FROM north_flow "
                "WHERE total_net IS NOT NULL AND total_net != 0 "
                f"AND trade_date >= '{min_date_ymd}' AND trade_date <= '{max_date_ymd}'",
                conn
            )
            conn.close()

            if len(north_df) > 0:
                north_df['total_net_billion'] = north_df['total_net'] / 10000  # 万元转亿元
                north_map = dict(zip(north_df['trade_date'], north_df['total_net_billion']))

                features['north_flow'] = trade_dates_ymd.map(north_map).fillna(0)
                features['north_flow_cum_3'] = features['north_flow'].rolling(6, min_periods=1).sum()
                features['north_flow_change'] = features['north_flow'].diff(6)
            else:
                features['north_flow'] = 0
                features['north_flow_cum_3'] = 0
                features['north_flow_change'] = 0
        except Exception as e:
            features['north_flow'] = 0
            features['north_flow_cum_3'] = 0
            features['north_flow_change'] = 0

        # 3. 大盘涨跌幅（从hs300_daily表，YYYYMMDD格式）
        try:
            conn = sqlite3.connect(DB_PATH)
            min_date_raw8 = trade_dates_raw8.min()
            max_date_raw8 = trade_dates_raw8.max()
            hs300_df = pd.read_sql(
                "SELECT trade_date, pct_chg, avg_pct_chg, up_count, down_count FROM hs300_daily "
                f"WHERE trade_date >= '{min_date_raw8}' AND trade_date <= '{max_date_raw8}'",
                conn
            )
            conn.close()

            if len(hs300_df) > 0:
                # 优先用pct_chg（BaoStock真实沪深300涨跌幅），fallback用avg_pct_chg（kline_30m推算）
                # 字段名可能不同，需要适配
                pct_col = 'pct_chg' if 'pct_chg' in hs300_df.columns else 'avg_pct_chg'
                hs300_map_pct = dict(zip(hs300_df['trade_date'], hs300_df[pct_col]))
                
                up_col = 'up_count' if 'up_count' in hs300_df.columns else None
                down_col = 'down_count' if 'down_count' in hs300_df.columns else None

                features['market_pct_chg'] = trade_dates_raw8.map(hs300_map_pct).fillna(0)
                if up_col:
                    hs300_map_up = dict(zip(hs300_df['trade_date'], hs300_df[up_col]))
                    hs300_map_down = dict(zip(hs300_df['trade_date'], hs300_df[down_col]))
                    features['market_up_ratio'] = trade_dates_raw8.map(hs300_map_up).fillna(0) / \
                                                  (trade_dates_raw8.map(hs300_map_down).fillna(0) + trade_dates_raw8.map(hs300_map_up).fillna(0) + 1e-10)
                else:
                    features['market_up_ratio'] = 0  # 无涨跌家数数据
                features['market_momentum_3'] = features['market_pct_chg'].rolling(6, min_periods=1).sum()
            else:
                features['market_pct_chg'] = 0
                features['market_up_ratio'] = 0
                features['market_momentum_3'] = 0
        except Exception as e:
            features['market_pct_chg'] = 0
            features['market_up_ratio'] = 0
            features['market_momentum_3'] = 0

        # 4. 个股vs大盘超额收益
        if 'close' in df.columns and 'market_pct_chg' in features.columns:
            stock_pct = df['close'].pct_change()
            features['alpha_vs_market'] = stock_pct - features['market_pct_chg'] / 100
            features['alpha_cum_3'] = features['alpha_vs_market'].rolling(6, min_periods=1).sum()
        else:
            features['alpha_vs_market'] = 0
            features['alpha_cum_3'] = 0

        # 5. 板块信息（从stock_sector表）
        try:
            if symbol:
                conn = sqlite3.connect(DB_PATH)
                sector_row = conn.execute(
                    "SELECT industry FROM stock_sector WHERE symbol=?", (symbol,)
                ).fetchone()
                conn.close()
                industry = sector_row[0] if sector_row else '其他'
            else:
                industry = '其他'
        except:
            industry = '其他'

        # 板块是否为强势行业（编码为0/1）
        strong_industries = {'电子', '电力设备', '计算机', '通信', '医药生物', '国防军工', '汽车'}
        features['sector_is_strong'] = 1 if industry in strong_industries else 0

        if MarketFeatureEngineer.MARKET_FEATURE_NAMES is None:
            MarketFeatureEngineer.MARKET_FEATURE_NAMES = features.columns.tolist()

        return features

    @staticmethod
    def calculate_target(df: pd.DataFrame, horizon: int = 3, threshold: float = 0.008) -> np.ndarray:
        """
        计算预测目标
        horizon: 预测周期（3根K线 = 90分钟）
        threshold: 涨跌阈值（0.8%）
        """
        close = df['close'].values
        target = np.zeros(len(close))

        for i in range(len(close) - horizon):
            ret = (close[i + horizon] - close[i]) / close[i]
            if ret > threshold:
                target[i] = 1  # 上涨
            elif ret < -threshold:
                target[i] = 0  # 下跌
            else:
                target[i] = -1  # 震荡（标记为-1，后续过滤）

        return target


def load_data_from_db(db_path: str) -> Dict[str, pd.DataFrame]:
    """从数据库加载所有股票数据"""
    import sqlite3

    all_data = {}

    if not os.path.exists(db_path):
        print(f"数据库不存在: {db_path}")
        return all_data

    conn = sqlite3.connect(db_path)

    # 获取所有股票列表
    cursor = conn.execute("SELECT DISTINCT symbol FROM kline_30m ORDER BY symbol")
    symbols = [row[0] for row in cursor.fetchall()]
    print(f"数据库中共有 {len(symbols)} 只股票")

    for i, symbol in enumerate(symbols):
        try:
            cursor = conn.execute(
                "SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol = ? ORDER BY date",
                (symbol,)
            )
            rows = cursor.fetchall()

            if len(rows) < 100:
                continue

            df = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

            # 数值转换
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            all_data[symbol] = df

        except Exception as e:
            print(f"  加载失败 {symbol}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  已加载 {i + 1}/{len(symbols)} 只股票")

    conn.close()
    print(f"成功加载 {len(all_data)} 只股票数据")

    return all_data


def load_data(cache_dir: str = '') -> Dict[str, pd.DataFrame]:
    """加载数据（从数据库）

    不再使用CSV/pkl缓存，直接从DB读取。
    """
    db_path = os.path.join(os.path.dirname(__file__), '../data/stock_data.db')
    if os.path.exists(db_path):
        return load_data_from_db(db_path)

    print("数据库不存在，无法加载数据")
    return {}


def prepare_training_data(all_data: Dict[str, pd.DataFrame], horizon: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """准备训练数据"""
    all_features = []
    all_targets = []

    print("计算特征...")

    for i, (symbol, df) in enumerate(all_data.items()):
        try:
            # 基础技术特征
            features = EnhancedFeatureEngineer.calculate_features(df)
            # 市场/板块特征
            market_features = MarketFeatureEngineer.calculate_market_features(df, symbol=symbol)
            # 合并
            features = pd.concat([features, market_features], axis=1)

            target = EnhancedFeatureEngineer.calculate_target(df, horizon=horizon)

            # 过滤无效数据
            valid_mask = ~(features.isna().any(axis=1)) & (target >= 0)
            features_valid = features[valid_mask]
            target_valid = target[valid_mask].astype(int)

            # 过滤前120行（特征不完整）
            features_valid = features_valid.iloc[120:]
            target_valid = target_valid[120:]

            if len(features_valid) > 50:
                all_features.append(features_valid.values)
                all_targets.append(target_valid)

        except Exception as e:
            print(f"  特征计算失败 {symbol}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  已处理 {i + 1}/{len(all_data)} 只股票")

    if not all_features:
        return None, None

    X = np.vstack(all_features)
    y = np.concatenate(all_targets)

    print(f"总样本数: {len(X)}")
    print(f"  上涨: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
    print(f"  下跌: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")

    return X, y


def train_model(X: np.ndarray, y: np.ndarray) -> Dict:
    """训练模型"""
    # 时序交叉验证
    tscv = TimeSeriesSplit(n_splits=5)

    # 优化后的参数
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 500,
        'max_depth': 8,
        'min_child_samples': 30,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1
    }

    print("\n训练 LightGBM 模型（5折交叉验证）...")

    cv_scores = []
    models = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores.append(accuracy)
        models.append(model)

        print(f"  Fold {fold + 1}: Accuracy = {accuracy:.4f}")

    avg_accuracy = np.mean(cv_scores)
    print(f"\n平均交叉验证准确率: {avg_accuracy:.4f}")

    # 使用最后一个模型
    final_model = models[-1]

    # 整体评估
    y_pred_all = final_model.predict(X)
    print(f"\n整体评估:")
    print(f"  准确率: {accuracy_score(y, y_pred_all):.2%}")
    print(f"\n分类报告:")
    print(classification_report(y, y_pred_all, target_names=['下跌', '上涨']))

    # 特征重要性
    feature_importance = dict(zip(
        EnhancedFeatureEngineer.FEATURE_NAMES or [],
        final_model.feature_importances_
    ))

    return {
        'model': final_model,
        'cv_accuracy': avg_accuracy,
        'cv_scores': cv_scores,
        'feature_importance': feature_importance,
        'feature_names': EnhancedFeatureEngineer.FEATURE_NAMES,
        'params': params,
        'train_samples': len(X)
    }


def save_model(model_data: Dict, model_dir: str):
    """保存模型"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n模型已保存到: {model_path}")


def main():
    print("=" * 60)
    print("增强版 LightGBM 模型训练")
    print("=" * 60)
    print(f"数据源: SQLite 数据库")
    print(f"特征数量: 50+")
    print(f"预测目标: 未来3根K线（90分钟）走势")
    print("=" * 60)

    # 加载数据
    db_path = os.path.join(os.path.dirname(__file__), '../data/stock_data.db')
    all_data = load_data_from_db(db_path)

    if not all_data:
        print("未加载到任何数据")
        return

    print(f"\n加载了 {len(all_data)} 只股票，开始训练...")

    # 准备训练数据
    X, y = prepare_training_data(all_data, horizon=3)

    if X is None or len(X) < 500:
        print(f"训练数据不足 ({len(X) if X is not None else 0} 条)")
        return

    # 训练模型
    model_data = train_model(X, y)

    # 保存模型
    model_dir = os.path.join(os.path.dirname(__file__), '../models/lgb_hs300')
    save_model(model_data, model_dir)

    # 显示特征重要性
    print("\n特征重要性 Top 20:")
    importance = sorted(model_data['feature_importance'].items(), key=lambda x: x[1], reverse=True)
    for name, score in importance[:20]:
        print(f"  {name}: {score}")

    print("\n训练完成!")
    print(f"交叉验证平均准确率: {model_data['cv_accuracy']:.2%}")


if __name__ == "__main__":
    main()