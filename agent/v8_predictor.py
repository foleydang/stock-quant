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

# ====== 信号阈值 ======
STRONG_BUY_THRESHOLD = 0.005    # 预测收益率 > 0.5%/90min → 强买入
BUY_THRESHOLD = 0.002           # 预测收益率 > 0.2%/90min → 关注
HOLD_THRESHOLD = -0.001         # -0.1% ~ 0.2% → 持有
REDUCE_THRESHOLD = -0.003       # -0.3% ~ -0.1% → 减仓
STRONG_SELL_THRESHOLD = -0.005  # < -0.5% → 清仓

TOP_N_CANDIDATES = 5            # 展示Top N买入候选
POSITION_RANK_WARN = 300        # 持仓排名 >300 触发警告


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

        Returns:
            [{symbol, name, predicted_return, rank, signal}, ...]
        """
        if not self.model:
            return []

        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data/stock_data.db'
        )
        conn = sqlite3.connect(db_path)

        # 获取股票列表
        symbols = [row[0] for row in
                   conn.execute("SELECT DISTINCT symbol FROM kline_30m ORDER BY symbol").fetchall()]

        if limit:
            symbols = symbols[:limit]

        results = []
        for sym in symbols:
            pred = self.predict_return(sym)
            if pred is not None:
                # 获取股票名称
                name_row = conn.execute("SELECT name FROM stock_info WHERE symbol=?", (sym,)).fetchone()
                name = name_row[0] if name_row and name_row[0] else sym
                results.append({'symbol': sym, 'name': name, 'predicted_return': pred})

        conn.close()

        # 按预测收益率降序排列
        results.sort(key=lambda x: x['predicted_return'], reverse=True)

        # 添加排名和信号
        for rank, r in enumerate(results):
            r['rank'] = rank + 1
            ret = r['predicted_return']
            if ret > STRONG_BUY_THRESHOLD:
                r['signal'] = 'strong_buy'
                r['signal_text'] = '🔥 强烈买入'
            elif ret > BUY_THRESHOLD:
                r['signal'] = 'buy'
                r['signal_text'] = '📈 可关注'
            elif ret > HOLD_THRESHOLD:
                r['signal'] = 'hold'
                r['signal_text'] = '➖ 持有'
            elif ret > REDUCE_THRESHOLD:
                r['signal'] = 'reduce'
                r['signal_text'] = '⚠️ 建议减仓'
            elif ret > STRONG_SELL_THRESHOLD:
                r['signal'] = 'sell'
                r['signal_text'] = '📉 建议卖出'
            else:
                r['signal'] = 'strong_sell'
                r['signal_text'] = '🚨 强烈卖出'

        return results

    def get_position_advice(self, positions: List[Dict]) -> List[Dict]:
        """
        对持仓股票给出加减仓建议

        Args:
            positions: [{symbol, name, shares, cost_price, current_price}, ...]

        Returns:
            带建议的持仓列表, 按预测收益率排序
        """
        if not self.model:
            return [dict(p, signal='unknown', signal_text='❓ 模型未加载') for p in positions]

        all_rankings = self.predict_all()

        # 建立排名查找表
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
            elif predicted_return > STRONG_BUY_THRESHOLD:
                signal = 'add'
                signal_text = f'🔥 建议加仓 (+{predicted_return:.2%})'
            elif predicted_return > BUY_THRESHOLD:
                signal = 'hold_add'
                signal_text = f'📈 持有观察 (+{predicted_return:.2%})'
            elif predicted_return > HOLD_THRESHOLD:
                signal = 'hold'
                signal_text = f'➖ 继续持有 ({predicted_return:.2%})'
            elif predicted_return > REDUCE_THRESHOLD:
                signal = 'reduce'
                signal_text = f'⚠️ 减仓 ({predicted_return:.2%})'
            else:
                signal = 'sell'
                signal_text = f'🚨 建议清仓 ({predicted_return:.2%})'

            # 排名警告
            rank_warning = ''
            if rank and rank > POSITION_RANK_WARN:
                rank_warning = f' ⚡排名{rank}/372(靠后)'

            advice.append({
                **pos,
                'predicted_return': predicted_return,
                'rank': rank,
                'signal': signal,
                'signal_text': signal_text + rank_warning,
            })

        # 按预测收益率排序
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


def format_feishu_message(rankings: List[Dict], positions_advice: List[Dict], spearman: float = None) -> str:
    """
    格式化为飞书消息（Markdown格式）
    """
    lines = ["**📊 v8 模型预测 (90分钟)**\n"]

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