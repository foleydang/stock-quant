#!/usr/bin/env python3
"""
v8 回归模型预测器 — 飞书 Bot 集成

功能:
1. 加载 v8 回归模型
2. 对所有股票预测未来收益率
3. 截面排序 → Top买入候选 + 持仓加减仓建议
4. 输出飞书卡片格式
"""

import os
import sys
import pickle
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'strategy'))

from train_lgb_enhanced import EnhancedFeatureEngineer

# ====== 自适应信号阈值 (基于大盘状态) ======
# 固定阈值的问题: 牛市0.5%太保守, 熊市0.5%太乐观
# 修复: 根据大盘趋势动态调整

# 基准阈值(震荡市)
BASE_BUY_THRESHOLD = 0.003       # 基准买入阈值 0.3%
BASE_STRONG_BUY = 0.005          # 基准强买阈值 0.5%
BASE_SELL_THRESHOLD = -0.003     # 基准卖出阈值 -0.3%
BASE_STRONG_SELL = -0.005        # 基准强卖阈值 -0.5%

TOP_N_CANDIDATES = 5
POSITION_RANK_WARN = 300


def get_market_regime() -> dict:
    """获取大盘状态 (基于沪深300日线)
    
    Returns:
        {'regime': 'bull'|'bear'|'sideways',
         'trend_strength': float,       # 趋势强度
         'daily_vol': float,            # 日波动率
         'adjustment': float}           # 阈值调整系数
    """
    try:
        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data/stock_data.db'
        )
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            "SELECT trade_date, close, pct_chg FROM hs300_daily ORDER BY trade_date DESC LIMIT 120",
            conn
        )
        conn.close()

        if df.empty or len(df) < 60:
            return {'regime': 'sideways', 'trend_strength': 0, 'daily_vol': 0.015, 'adjustment': 1.0}

        df = df.sort_values('trade_date').reset_index(drop=True)
        close = df['close'].values

        # 趋势判断: 20日MA vs 60日MA
        ma20 = np.mean(close[-20:])
        ma60 = np.mean(close[-60:]) if len(close) >= 60 else np.mean(close)
        current = close[-1]

        # 趋势强度 = (MA20偏离MA60的程度)
        trend_strength = (ma20 - ma60) / ma60

        if ma20 > ma60 and current > ma20:
            regime = 'bull'
        elif ma20 < ma60 and current < ma20:
            regime = 'bear'
        else:
            regime = 'sideways'

        # 日波动率
        returns = df['pct_chg'].values / 100
        daily_vol = np.std(returns[-20:]) if len(returns) >= 20 else 0.015

        # 阈值调整系数
        # 牛市: 放宽买入阈值(容易涨), 收紧卖出阈值(少卖)
        # 熊市: 收紧买入阈值(容易跌), 放宽卖出阈值(多卖)
        if regime == 'bull':
            adjustment = 0.7  # 阈值 × 0.7 → 更容易触发买入
        elif regime == 'bear':
            adjustment = 1.5  # 阈值 × 1.5 → 更难买入, 更容易卖出
        else:
            adjustment = 1.0

        return {
            'regime': regime,
            'trend_strength': round(float(trend_strength), 4),
            'daily_vol': round(float(daily_vol), 4),
            'adjustment': adjustment,
        }

    except Exception as e:
        return {'regime': 'sideways', 'trend_strength': 0, 'daily_vol': 0.015, 'adjustment': 1.0}


def get_adaptive_thresholds() -> dict:
    """获取自适应阈值"""
    regime_info = get_market_regime()
    adj = regime_info['adjustment']

    return {
        'strong_buy': BASE_STRONG_BUY * adj,
        'buy': BASE_BUY_THRESHOLD * adj,
        'hold': 0,  # 中性
        'reduce': BASE_SELL_THRESHOLD * adj,
        'strong_sell': BASE_STRONG_SELL * adj,
        'regime': regime_info['regime'],
        'trend_strength': regime_info['trend_strength'],
        'daily_vol': regime_info['daily_vol'],
    }


class V8Predictor:
    """v8 回归模型预测器"""

    def __init__(self, model_path: str = None):
        self.model = None
        self.feature_names = None
        self.model_data = None

        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'models/lgb_hs300/model.pkl'
            )

        if os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """加载模型"""
        with open(model_path, 'rb') as f:
            self.model_data = pickle.load(f)

        self.model = self.model_data.get('model')
        self.feature_names = self.model_data.get('feature_names')
        model_type = self.model_data.get('model_type', 'classification')
        version = self.model_data.get('model_version', 'unknown')

        print(f"[V8Predictor] 加载模型: {version} ({model_type})")
        if model_type == 'regression':
            print(f"[V8Predictor] Spearman={self.model_data.get('cv_spearman', 0):.4f}, "
                  f"特征={len(self.feature_names or [])}")

        return self.model is not None

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict_return(self, symbol: str) -> Optional[float]:
        """
        预测单只股票的未来收益率
        
        Returns:
            float: 预测的90分钟收益率, 或 None
        """
        if not self.model:
            return None

        try:
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data/stock_data.db'
            )
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(
                'SELECT date, open, high, low, close, volume FROM kline_30m '
                'WHERE symbol=? ORDER BY date DESC LIMIT 200',
                conn, params=(symbol,)
            )
            conn.close()

            if df.empty or len(df) < 150:
                return None

            df = df.sort_values('date').reset_index(drop=True)
            features = EnhancedFeatureEngineer.calculate_features(df)
            features = features.fillna(method='ffill').fillna(0)

            if self.feature_names:
                missing = [c for c in self.feature_names if c not in features.columns]
                for c in missing:
                    features[c] = 0
                features = features[self.feature_names]

            last_row = features.iloc[-1].values
            prediction = float(self.model.predict([last_row])[0])
            return prediction

        except Exception as e:
            print(f"[V8Predictor] 预测失败 {symbol}: {e}")
            return None

    def predict_all(self, limit: int = None) -> List[Dict]:
        """
        预测所有股票的收益率 → 按预测值排序
        
        使用自适应阈值: 根据大盘状态(牛/熊/震荡)动态调整
        """
        if not self.model:
            return []

        # 获取自适应阈值
        thresholds = get_adaptive_thresholds()

        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data/stock_data.db'
        )
        conn = sqlite3.connect(db_path)

        symbols = [row[0] for row in
                   conn.execute("SELECT DISTINCT symbol FROM kline_30m ORDER BY symbol").fetchall()]

        if limit:
            symbols = symbols[:limit]

        results = []
        for sym in symbols:
            pred = self.predict_return(sym)
            if pred is not None:
                name_row = conn.execute("SELECT name FROM stock_info WHERE symbol=?", (sym,)).fetchone()
                name = name_row[0] if name_row and name_row[0] else sym
                results.append({'symbol': sym, 'name': name, 'predicted_return': pred})

        conn.close()

        results.sort(key=lambda x: x['predicted_return'], reverse=True)

        # 自适应信号
        for rank, r in enumerate(results):
            r['rank'] = rank + 1
            ret = r['predicted_return']

            if ret > thresholds['strong_buy']:
                r['signal'] = 'strong_buy'
                r['signal_text'] = '🔥 强烈买入'
            elif ret > thresholds['buy']:
                r['signal'] = 'buy'
                r['signal_text'] = '📈 可关注'
            elif ret > thresholds['reduce']:
                r['signal'] = 'hold'
                r['signal_text'] = '➖ 持有'
            elif ret > thresholds['strong_sell']:
                r['signal'] = 'sell'
                r['signal_text'] = '📉 建议卖出'
            else:
                r['signal'] = 'strong_sell'
                r['signal_text'] = '🚨 强烈卖出'

        # 附加大盘状态
        self._regime_info = thresholds

        return results

    def get_position_advice(self, positions: List[Dict]) -> List[Dict]:
        """对持仓股票给出加减仓建议 (自适应阈值)"""
        if not self.model:
            return [dict(p, signal='unknown', signal_text='❓ 模型未加载') for p in positions]

        # 自适应阈值
        thresholds = get_adaptive_thresholds()
        all_rankings = self.predict_all()

        rank_map = {r['symbol']: r for r in all_rankings}

        advice = []
        for pos in positions:
            sym = pos['symbol']
            ranking = rank_map.get(sym, {})

            predicted_return = ranking.get('predicted_return')
            rank = ranking.get('rank')

            if predicted_return is None:
                signal = 'unknown'
                signal_text = '❓ 数据不足'
            elif predicted_return > thresholds['strong_buy']:
                signal = 'add'
                signal_text = f'🔥 建议加仓'
            elif predicted_return > thresholds['buy']:
                signal = 'hold_add'
                signal_text = f'📈 持有观察'
            elif predicted_return > thresholds['reduce']:
                signal = 'hold'
                signal_text = f'➖ 继续持有'
            elif predicted_return > thresholds['strong_sell']:
                signal = 'reduce'
                signal_text = f'⚠️ 减仓'
            else:
                signal = 'sell'
                signal_text = f'🚨 建议清仓'

            # 预测值
            pred_str = f"+{predicted_return:.2%}" if predicted_return and predicted_return > 0 else f"{predicted_return:.2%}" if predicted_return else "?"

            # 排名警告
            rank_warning = ''
            if rank and rank > POSITION_RANK_WARN:
                rank_warning = f' ⚡排名{rank}/372(靠后)'

            advice.append({
                **pos,
                'predicted_return': predicted_return,
                'predicted_return_str': pred_str,
                'rank': rank,
                'signal': signal,
                'signal_text': signal_text + rank_warning,
            })

        advice.sort(key=lambda x: x.get('predicted_return', -999), reverse=True)
        return advice

    def get_buy_candidates(self, existing_positions: List[str] = None, n: int = None) -> List[Dict]:
        """
        获取买入候选（排除已持仓的股票）

        Args:
            existing_positions: 已持仓的 symbol 列表
            n: Top N

        Returns:
            Top N 买入候选
        """
        if n is None:
            n = TOP_N_CANDIDATES

        all_rankings = self.predict_all()
        existing = set(existing_positions or [])

        candidates = [r for r in all_rankings
                      if r['symbol'] not in existing
                      and r['predicted_return'] > BUY_THRESHOLD]

        return candidates[:n]


def format_feishu_message(rankings: List[Dict], positions_advice: List[Dict], spearman: float = None, regime_info: dict = None) -> str:
    """格式化为飞书消息（Markdown格式）"""
    lines = ["**📊 v8 模型预测 (90分钟)**\n"]

    # 大盘状态
    if regime_info:
        regime = regime_info.get('regime', '?')
        emoji = {'bull': '🐂', 'bear': '🐻', 'sideways': '📊'}.get(regime, '📊')
        cn = {'bull': '牛市', 'bear': '熊市', 'sideways': '震荡'}.get(regime, '?')
        lines.append(f"{emoji} 大盘: **{cn}** | 趋势强度: {regime_info.get('trend_strength', 0):.2%} | 阈值×{regime_info.get('adjustment', 1.0):.1f}\n")

    # 买入候选
    candidates = [r for r in rankings if r['signal'] in ('strong_buy', 'buy')][:TOP_N_CANDIDATES]
    if candidates:
        lines.append("**🔥 买入候选**")
        for r in candidates:
            lines.append(f"  {r['rank']}. {r['name']} — 预期收益 **{r['predicted_return']:.2%}**")
        lines.append("")

    # 持仓建议
    if positions_advice:
        lines.append("**💼 持仓建议**")
        for p in positions_advice[:10]:
            profit = p.get('profit_pct', 0)
            profit_str = f" (浮{profit:+.1f}%)" if profit else ""
            pred = p.get('predicted_return')
            pred_str = f"→ 预期" + (f"{pred:.2%}" if pred is not None else "?")
            lines.append(f"  {p['name']} — {p['signal_text']}{profit_str}")
        lines.append("")

    # 整体信号
    strong_buy_count = sum(1 for r in rankings if r['signal'] == 'strong_buy')
    buy_count = sum(1 for r in rankings if r['signal'] == 'buy')
    sell_count = sum(1 for r in rankings if r['signal'] in ('sell', 'strong_sell'))

    lines.append(f"**📈 信号分布:** 🔥{strong_buy_count} 📈{buy_count} ➖持仓 📉{sell_count}")
    if spearman:
        lines.append(f"*模型: v8 回归, Spearman={spearman:.4f}*")

    return '\n'.join(lines)


# 全局实例
_predictor: Optional[V8Predictor] = None


def get_predictor() -> V8Predictor:
    global _predictor
    if _predictor is None:
        _predictor = V8Predictor()
    return _predictor


# 测试入口
if __name__ == '__main__':
    p = get_predictor()
    if not p.is_loaded():
        print("模型未加载, 请先训练 v8 回归模型")
        sys.exit(1)

    rankings = p.predict_all(limit=50)
    print(f"\nTop 10 预测:")
    for r in rankings[:10]:
        print(f"  {r['rank']:>3}. {r['name']:<8} {r['signal_text']:<12} ({r['predicted_return']:.4f})")

    print(f"\nBottom 5:")
    for r in rankings[-5:]:
        print(f"  {r['rank']:>3}. {r['name']:<8} {r['signal_text']:<12} ({r['predicted_return']:.4f})")