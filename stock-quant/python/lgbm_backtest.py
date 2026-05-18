#!/usr/bin/env python3
"""
基于 LGBM 模型的回测系统 - 性能优化版 + 逻辑修复版 + 日志规范
优化点：
1. 预计算特征，避免重复计算
2. 时间匹配修复：用时间值而非全局索引
3. 使用 logging 模块规范日志
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

from strategy.train_lgb_enhanced import EnhancedFeatureEngineer
from strategy.train_lgb_v3 import AdvancedFeatureEngineer, ZERO_IMP_FEATURES, TIME_FEATURES

# 配置日志
logger = logging.getLogger(__name__)  
logger.setLevel(logging.DEBUG)  # 设置为 DEBUG，允许所有级别

# 控制台输出（简洁）
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))

# 文件输出（详细）
log_dir = os.path.join(os.path.dirname(__file__), '../logs')
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, 'backtest.log'))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger.addHandler(console_handler)
logger.addHandler(file_handler)


@dataclass
class Position:
    """持仓"""
    symbol: str
    stock_name: str
    shares: int
    cost_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    entry_time: str
    entry_idx: int
    available: bool = False


@dataclass
class Trade:
    """交易记录"""
    symbol: str
    stock_name: str
    trade_type: str
    price: float
    shares: int
    amount: float
    time: str
    reason: str
    profit: float = 0.0
    hold_periods: int = 0


class LGBMBacktesterOptimized:
    """优化版回测引擎 - 预计算特征 + 时间匹配修复"""

    def __init__(self, initial_capital: float = 100000, model_path: str = None):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_values = []

        # 加载模型 (支持v4混合/v3 ensemble)
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'models/lgb_hs300/model.pkl')

        # v4混合模型 (需要用use_v4=True参数启用)
        v4_path = model_path.replace('model.pkl', 'model_v4.pkl')
        if os.path.exists(v4_path) and getattr(self, 'use_v4', False):
            model_path = v4_path
            logger.info(f"✓ 使用v4混合模型")

        self.model_data = self._load_model(model_path)
        self.models = self.model_data.get('models', []) if self.model_data else []
        self.feature_names = self.model_data.get('feature_names', []) if self.model_data else []
        self.keep_features = self.model_data.get('keep_features', []) if self.model_data else []

        # H5 双确认模型 (备用)
        h5_path = model_path.replace('lgb_hs300', 'lgb_hs300_h5')
        if os.path.exists(h5_path):
            with open(h5_path, 'rb') as f:
                self.model_data_h5 = pickle.load(f)
            self.models_h5 = self.model_data_h5.get('models', [])

        # 参数 - 短线策略
        self.position_pct = 0.30  # 仓位比例
        self.max_positions = 3    # 集中持仓
        self.min_hold_periods = 6   # 最小持仓6根K线=3小时
        self.max_hold_periods = 24  # 最大持仓24根K线=12小时
        self.stop_loss_pct = 0.015   # 止损1.5%
        self.take_profit_pct = 0.035  # 止盈3.5%
        self.buy_threshold = 0.55   # 买入阈值(最优值,回测验证)
        self.sell_threshold = 0.35  # 卖出阈值
        self.dynamic_stop_loss = False  # 动态止损(验证无效,关闭)

        # 特征缓存
        self.features_cache: Dict[str, pd.DataFrame] = {}
        # 时间索引映射
        self.time_index_map: Dict[str, Dict[datetime, int]] = {}

    def _load_model(self, model_path: str) -> Optional[Dict]:
        """加载模型"""
        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            return None
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return None

    def _get_model_prediction(self, symbol: str, local_idx: int) -> Tuple[float, str]:
        """获取模型预测 - 支持v4混合+LR Stacking"""
        if not self.models:
            return 0.5, "模型未加载"

        features = self.features_cache.get(symbol)
        if features is None or local_idx >= len(features):
            return 0.5, "无特征数据"

        try:
            last_row = features.iloc[local_idx]
            if last_row.isna().any():
                last_row = last_row.fillna(0)

            # 对齐特征到模型期望的列(v4:107列, v3:118列)
            if self.feature_names and len(last_row) != len(self.feature_names):
                last_row = last_row.reindex(self.feature_names).fillna(0)

            # 每个模型类型用不同的predict方式
            probs = []
            model_types = self.model_data.get('model_types', ['lgbm'] * len(self.models))

            for model, mtype in zip(self.models, model_types):
                try:
                    if mtype == 'lgbm':
                        p = model.predict_proba([last_row.values])[0]
                        probs.append(p[1] if len(p) > 1 else p[0])
                    elif mtype == 'xgb':
                        p = model.predict_proba(last_row.values.reshape(1, -1))[0]
                        probs.append(p[1] if len(p) > 1 else p[0])
                    elif mtype == 'catboost':
                        p = model.predict_proba(last_row.values.reshape(1, -1))
                        probs.append(float(p[0][1]) if len(p[0]) > 1 else float(p[0][0]))
                    else:
                        probs.append(0.5)
                except Exception:
                    probs.append(0.5)

            avg_prob = float(np.mean(probs))

            # LR Stacking元模型 (如果可用且有效)
            # 注意: v4的LR Stacking在训练集上98%准确率但回测0%收益,
            # 严重过拟合, 故禁用LR, 改用简单加权平均
            lr_meta = self.model_data.get('lr_meta')
            use_lr_stacking = self.model_data.get('use_lr_stacking', False)
            if lr_meta is not None and use_lr_stacking:
                std_prob = float(np.std(probs))
                signal_strength = avg_prob - 0.5
                strong_up = 1.0 if avg_prob > 0.6 else 0.0
                strong_down = 1.0 if avg_prob < 0.4 else 0.0
                stacking_input = np.array([
                    *probs, avg_prob, std_prob,
                    signal_strength, strong_up, strong_down,
                ]).reshape(1, -1)
                up_prob = float(lr_meta.predict_proba(stacking_input)[0][1])
                method = f"LR-stacking"
            elif lr_meta is not None:
                # 禁用LR, 用加权平均(LGBM权重1.0, XGB权重0.8)
                weights = []
                for mt in model_types:
                    if mt == 'lgbm':
                        weights.append(1.0)
                    elif mt == 'xgb':
                        weights.append(0.8)
                    elif mt == 'catboost':
                        weights.append(0.9)
                    else:
                        weights.append(1.0)
                total_w = sum(weights)
                weighted_avg = sum(p * w for p, w in zip(probs, weights)) / total_w
                up_prob = weighted_avg
                method = f"加权混合({len(probs)}模型)"
            else:
                up_prob = avg_prob
                n_types = Counter(model_types)
                type_str = '+'.join(f"{cnt}{t}" for t, cnt in n_types.items())
                method = f"混合:{type_str}"

            return up_prob, f"上涨概率:{up_prob:.1%}({method})"

        except Exception as e:
            return 0.5, f"预测错误:{e}"

    def _get_model_prediction_h5(self, symbol: str, local_idx: int) -> float:
        """获取H5模型的上涨概率 (双确认用)"""
        if not self.models_h5:
            return 0.5

        features = self.features_cache.get(symbol)
        if features is None or local_idx >= len(features):
            return 0.5

        try:
            last_row = features.iloc[local_idx]
            if last_row.isna().any():
                return 0.5

            probs = []
            for model in self.models_h5:
                p = model.predict_proba([last_row.values])[0]
                probs.append(p[1] if len(p) > 1 else p[0])
            return np.mean(probs)
        except Exception:
            return 0.5

    def load_data(self, symbol: str) -> pd.DataFrame:
        """加载数据 - 优先从数据库读取"""
        import sqlite3
        from config_loader import get_db_path
        
        db_path = get_db_path()
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol=? ORDER BY date ASC',
                    (symbol,)
                )
                rows = cursor.fetchall()
                conn.close()
                
                if rows:
                    df = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date').reset_index(drop=True)
                    return df
            except Exception as e:
                logger.warning(f"数据库读取失败: {e}")
        
        # Fallback: CSV缓存
        cache_path = os.path.join(os.path.dirname(__file__), f'data/{symbol}_30m.csv')
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            return df
        
        return None

    def preload_features(self, all_data: Dict[str, pd.DataFrame]):
        """预计算所有特征 + 建立时间索引映射"""
        logger.info("\n预计算特征...")
        for symbol, df in all_data.items():
            logger.debug(f"计算 {symbol} 特征...")
            # 使用 v3 的组合特征 (基础+高级，过滤时间和零重要性)
            base_features = EnhancedFeatureEngineer.calculate_features(df)
            adv_features = AdvancedFeatureEngineer.calculate_advanced_features(df)
            all_features = pd.concat([base_features, adv_features], axis=1)
            drop_cols = TIME_FEATURES + ZERO_IMP_FEATURES
            keep_cols = [c for c in all_features.columns if c not in drop_cols]
            all_features = all_features[keep_cols]
            all_features = all_features.fillna(0)
            self.features_cache[symbol] = all_features
            # 建立时间到索引的映射
            self.time_index_map[symbol] = {row['date']: idx for idx, row in df.iterrows()}
        logger.info(f"特征预计算完成，共 {len(self.features_cache)} 只股票")

    def _get_stock_local_idx(self, symbol: str, time: datetime) -> Optional[int]:
        """获取股票在指定时间的局部索引"""
        time_map = self.time_index_map.get(symbol)
        if time_map is None:
            return None
        return time_map.get(time)

    def run_backtest(self, stocks: List[Dict], start_date: Optional[str] = None, end_date: Optional[str] = None):
        """执行回测 - 支持日期范围"""
        logger.info("=" * 70)
        logger.info("LGBM 模型回测系统 (优化版)")
        logger.info("=" * 70)
        logger.info(f"初始资金: {self.initial_capital:.2f} 元")
        if self.model_data:
            logger.info(f"模型准确率: {self.model_data.get('cv_accuracy', 0):.2%}")
        else:
            logger.warning("模型未加载")
        logger.info(f"买入阈值: 上涨概率 > {self.buy_threshold:.0%}")
        logger.info("=" * 70)

        # 加载数据
        all_data = {}
        for stock in stocks:
            symbol = stock['symbol']
            logger.info(f"\n加载 {stock['name']} ({symbol})...")
            df = self.load_data(symbol)
            if df is not None and len(df) >= 60:
                # 日期范围过滤（兼容不同时间格式）
                if start_date:
                    start_dt = pd.to_datetime(start_date)
                    # 只匹配日期部分，忽略时间
                    df = df[df['date'].dt.date >= start_dt.date()]
                if end_date:
                    end_dt = pd.to_datetime(end_date)
                    df = df[df['date'].dt.date <= end_dt.date()]
                
                # 降低最小数据要求
                if len(df) >= 20:
                    # 重置索引，确保特征计算正确
                    df = df.reset_index(drop=True)
                    all_data[symbol] = df
                    logger.info(f"  ✓ 数据量: {len(df)} 条")
                else:
                    logger.warning(f"  ⚠️ 日期范围后数据不足({len(df)}条)")
            else:
                logger.warning(f"  ⚠️ 数据不足")

        if not all_data:
            logger.error("无有效数据")
            return

        # 预计算特征
        self.preload_features(all_data)

        # 获取所有时间点
        all_times = sorted(set(
            row['date'] for df in all_data.values() for row in df.to_dict('records')
        ))

        logger.info(f"\n共 {len(all_times)} 个时间点")
        logger.info("\n开始回测...\n")

        # 遍历时间点
        for i, current_time in enumerate(all_times):
            if i % 50 == 0:
                total_value = self._calculate_total_value(all_data, current_time)
                logger.info(f"[{i}/{len(all_times)}] {current_time.strftime('%m-%d %H:%M')} | "
                           f"市值:{total_value:.0f} | 现金:{self.cash:.0f} | 持仓:{len(self.positions)}只")

            self._update_availability(all_data, current_time)
            self._check_sell(all_data, current_time)
            self._check_buy_ml(all_data, current_time, stocks)

            total_value = self._calculate_total_value(all_data, current_time)
            self.daily_values.append({
                'time': current_time,
                'value': total_value,
                'cash': self.cash,
                'positions': len(self.positions)
            })

        self._print_results()

    def _update_availability(self, all_data: Dict, current_time: datetime):
        """更新T+1"""
        for symbol, pos in self.positions.items():
            if not pos.available:
                current_idx = self._get_stock_local_idx(symbol, current_time)
                if current_idx is None:
                    continue
                periods_held = current_idx - pos.entry_idx
                if periods_held >= self.min_hold_periods:
                    pos.available = True

    def _check_sell(self, all_data: Dict, current_time: datetime):
        """检查卖出 - 支持动态止损"""
        for symbol, pos in list(self.positions.items()):
            if not pos.available:
                continue

            df = all_data.get(symbol)
            if df is None:
                continue

            current_idx = self._get_stock_local_idx(symbol, current_time)
            if current_idx is None:
                continue

            current_price = float(df['close'].iloc[current_idx])
            pos.current_price = current_price

            hold_periods = current_idx - pos.entry_idx
            sell_reason = None

            loss_pct = (current_price - pos.cost_price) / pos.cost_price

            # 动态止损: 根据近期波动率调整止损幅度
            if self.dynamic_stop_loss and current_idx >= 10:
                recent_prices = df['close'].iloc[current_idx-10:current_idx+1].values
                recent_vol = np.std(np.diff(recent_prices) / recent_prices[:-1])  # 10根K线的波动率
                # 高波动: 止损放宽到2.5%; 低波动: 收紧到1.2%
                dynamic_stop = min(0.025, max(0.012, recent_vol * 3))  # 3倍波动率作为止损
            else:
                dynamic_stop = self.stop_loss_pct

            if loss_pct <= -dynamic_stop:
                sell_reason = f"止损(动态{dynamic_stop:.1%})"
            elif current_price >= pos.take_profit:
                sell_reason = "止盈"
            elif hold_periods >= self.max_hold_periods:
                sell_reason = "到期"
            elif hold_periods >= self.min_hold_periods:
                up_prob, _ = self._get_model_prediction(symbol, current_idx)
                if up_prob < self.sell_threshold:
                    sell_reason = f"模型看跌({up_prob:.0%})"

            if sell_reason:
                self._sell_position(symbol, current_time, current_price, sell_reason, hold_periods)

    def _check_market_trend(self, all_data: Dict, current_time: datetime) -> str:
        """判断大盘趋势: 'up'/'down'/'neutral'"""
        # 用平安银行作为大盘代理
        proxy = '000001.SZ'
        df = all_data.get(proxy)
        if df is None or len(df) < 20:
            return 'neutral'  # 无数据时不限制

        idx = self._get_stock_local_idx(proxy, current_time)
        if idx is None or idx < 10:
            return 'neutral'

        # 最近10根K线的趋势 (5小时=约1.5个交易日)
        recent_close = df['close'].iloc[idx-10:idx+1].values
        if len(recent_close) < 5:
            return 'neutral'

        ret_10 = (recent_close[-1] - recent_close[0]) / recent_close[0]
        if ret_10 > 0.008:   # 5小时内涨>0.8% = 上涨趋势
            return 'up'
        elif ret_10 < -0.008:  # 5小时内跌>0.8% = 下跌趋势
            return 'down'
        else:
            return 'neutral'

    def _check_buy_ml(self, all_data: Dict, current_time: datetime, stocks: List[Dict]):
        """检查买入 - 加入多时间框架确认"""
        if len(self.positions) >= self.max_positions:
            return

        for stock in stocks:
            symbol = stock['symbol']
            if symbol in self.positions:
                continue

            df = all_data.get(symbol)
            if df is None:
                continue

            current_idx = self._get_stock_local_idx(symbol, current_time)
            if current_idx is None or current_idx < 120:
                continue

            up_prob, reason = self._get_model_prediction(symbol, current_idx)

            if up_prob > self.buy_threshold:
                # 置信度决定仓位比例
                confidence = up_prob - 0.5
                if confidence > 0.20:   # >70%: 重仓40%
                    pos_pct = 0.40
                elif confidence > 0.10: # >60%: 中仓30%
                    pos_pct = 0.30
                else:                  # 55-60%: 轻仓20%
                    pos_pct = 0.20

                current_price = float(df['close'].iloc[current_idx])
                self._buy_stock(symbol, stock['name'], current_time, current_price, current_idx, reason, pos_pct=pos_pct)

    def _buy_stock(self, symbol: str, stock_name: str, entry_time: datetime,
                   entry_price: float, entry_idx: int, reason: str, pos_pct: float = None):
        """买入 - 支持置信度动态仓位"""
        if self.cash <= 0:
            return

        # 动态仓位
        if pos_pct is None:
            pos_pct = self.position_pct
        max_invest = min(self.cash * 0.9, self.initial_capital * pos_pct)
        shares = int(max_invest / entry_price / 100) * 100  # 100股整数倍
        if shares < 100:  # 最少100股（交易所规则）
            return

        actual_amount = shares * entry_price
        if actual_amount > self.cash:
            shares = int(self.cash / entry_price / 100) * 100  # 100股整数倍
            if shares < 100:
                return
            actual_amount = shares * entry_price

        stop_loss = entry_price * (1 - self.stop_loss_pct)
        take_profit = entry_price * (1 + self.take_profit_pct)

        self.cash -= actual_amount

        self.positions[symbol] = Position(
            symbol=symbol,
            stock_name=stock_name,
            shares=shares,
            cost_price=entry_price,
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=str(entry_time),
            entry_idx=entry_idx,
            available=False
        )

        self.trades.append(Trade(
            symbol=symbol,
            stock_name=stock_name,
            trade_type="buy",
            price=entry_price,
            shares=shares,
            amount=actual_amount,
            time=str(entry_time),
            reason=reason
        ))

        # DEBUG 级别：详细交易记录写入文件，不打印到控制台
        logger.debug(f"买入 {stock_name}: {shares}股 @ {entry_price:.2f} | {reason}")

    def _sell_position(self, symbol: str, sell_time: datetime, sell_price: float,
                       reason: str, hold_periods: int):
        """卖出"""
        pos = self.positions.get(symbol)
        if pos is None or not pos.available:
            return

        sell_amount = pos.shares * sell_price
        profit = (sell_price - pos.cost_price) * pos.shares

        self.cash += sell_amount

        self.trades.append(Trade(
            symbol=symbol,
            stock_name=pos.stock_name,
            trade_type="sell",
            price=sell_price,
            shares=pos.shares,
            amount=sell_amount,
            time=str(sell_time),
            reason=reason,
            profit=profit,
            hold_periods=hold_periods
        ))

        del self.positions[symbol]

        # DEBUG 级别：详细交易记录写入文件
        result = "盈利" if profit > 0 else "亏损"
        logger.debug(f"卖出 {pos.stock_name}: {pos.shares}股 @ {sell_price:.2f} | "
                    f"{result}:{profit:.0f} | {reason}")

    def _calculate_total_value(self, all_data: Dict, current_time: datetime) -> float:
        """计算总市值"""
        total = self.cash
        for symbol, pos in self.positions.items():
            df = all_data.get(symbol)
            if df is None:
                total += pos.current_price * pos.shares
                continue
            current_idx = self._get_stock_local_idx(symbol, current_time)
            if current_idx is not None:
                current_price = float(df['close'].iloc[current_idx])
                total += current_price * pos.shares
            else:
                total += pos.current_price * pos.shares
        return total

    def _print_results(self):
        """输出结果"""
        logger.info("\n" + "=" * 70)
        logger.info("回测结果汇总")
        logger.info("=" * 70)

        final_value = self.daily_values[-1]['value'] if self.daily_values else self.initial_capital
        total_return = final_value - self.initial_capital
        return_pct = total_return / self.initial_capital * 100

        logger.info(f"\n【资金统计】")
        logger.info(f"  初始资金: {self.initial_capital:.2f} 元")
        logger.info(f"  最终市值: {final_value:.2f} 元")
        logger.info(f"  总盈亏: {total_return:.2f} 元")
        logger.info(f"  收益率: {return_pct:.2f}%")

        buy_trades = [t for t in self.trades if t.trade_type == "buy"]
        sell_trades = [t for t in self.trades if t.trade_type == "sell"]

        wins = [t for t in sell_trades if t.profit > 0]
        losses = [t for t in sell_trades if t.profit <= 0]

        if sell_trades:
            win_rate = len(wins) / len(sell_trades) * 100
            total_profit = sum(t.profit for t in sell_trades)
            avg_win = sum(t.profit for t in wins) / len(wins) if wins else 0
            avg_loss = sum(t.profit for t in losses) / len(losses) if losses else 0

            logger.info(f"\n【交易统计】")
            logger.info(f"  买入次数: {len(buy_trades)}")
            logger.info(f"  卖出次数: {len(sell_trades)}")
            logger.info(f"  盈利次数: {len(wins)} | 亏损次数: {len(losses)}")
            logger.info(f"  胜率: {win_rate:.1f}%")
            logger.info(f"  总盈亏: {total_profit:.2f} 元")
            logger.info(f"  平均盈利: {avg_win:.2f} | 平均亏损: {avg_loss:.2f}")
            if avg_loss != 0:
                logger.info(f"  盈亏比: {abs(avg_win/avg_loss):.2f}")

        logger.info("-" * 70)


WATCHLIST = [
    {"symbol": "300015.SZ", "name": "爱尔眼科"},
    {"symbol": "300124.SZ", "name": "汇川技术"},
    {"symbol": "600048.SH", "name": "保利发展"},
    {"symbol": "600519.SH", "name": "贵州茅台"},
    {"symbol": "000001.SZ", "name": "平安银行"},
    {"symbol": "000333.SZ", "name": "美的集团"},
    {"symbol": "002594.SZ", "name": "比亚迪"},
    {"symbol": "601318.SH", "name": "中国平安"},
    {"symbol": "3690.HK", "name": "美团-W"},
    {"symbol": "0700.HK", "name": "腾讯控股"},
    {"symbol": "9988.HK", "name": "阿里巴巴-W"},
]


def main():
    backtester = LGBMBacktesterOptimized(initial_capital=500000)  # 50万初始资金
    backtester.run_backtest(WATCHLIST)


if __name__ == "__main__":
    main()