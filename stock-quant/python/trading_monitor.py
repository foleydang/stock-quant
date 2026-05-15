#!/usr/bin/env python3
"""
交易监控系统
功能:
1. 更新持仓股票实时价格
2. 熊市策略：不自动止损，关注补仓机会
3. 15万现金补仓策略
4. 邮件通知

使用方法:
    python trading_monitor.py          # 实时监控
    python trading_monitor.py --update # 更新数据
"""

import os
import sys
import json
import pickle
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# 使用统一配置
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import get_base_dir, get_db_path, get_available_cash, get_watchlist, get_strategy_params, get_email_config

BASE_DIR = get_base_dir()
DB_PATH = get_db_path()
MODEL_PATH = os.path.join(BASE_DIR, 'models/lgb_hs300/model.pkl')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
sys.path.insert(0, BASE_DIR)

from data.data_handler import DataHandler

try:
    from strategy.email_notifier import EmailNotifier, create_email_notifier_from_env
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

try:
    from strategy.train_lgb_enhanced import EnhancedFeatureEngineer
    FEATURE_ENGINEER_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEER_AVAILABLE = False

# 做T策略
try:
    from strategy.t_strategy import TStrategy, TTradeSuggestion, format_t_suggestion, format_t_suggestions_batch
    T_STRATEGY_AVAILABLE = True
except ImportError:
    T_STRATEGY_AVAILABLE = False


@dataclass
class Position:
    """持仓"""
    symbol: str
    stock_name: str
    shares: int
    cost_price: float
    current_price: float
    entry_date: str = ""

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def cost_value(self) -> float:
        return self.shares * self.cost_price

    @property
    def profit(self) -> float:
        return (self.current_price - self.cost_price) * self.shares

    @property
    def profit_pct(self) -> float:
        if self.cost_price == 0:
            return 0
        return (self.current_price - self.cost_price) / self.cost_price * 100


class TradingMonitor:
    """交易监控 - 熊市策略"""

    def __init__(self):
        self.db_path = DB_PATH
        self.model_path = MODEL_PATH
        self.data_handler = DataHandler(force_refresh=True)

        # 账户参数
        self.available_cash = get_available_cash()  # 从配置读取

        # 策略参数（熊市策略）
        params = get_strategy_params()
        self.add_position_threshold = params.get('add_position_threshold', -0.20)
        self.add_position_up_prob = params.get("add_position_prob", 0.55)
        self.max_add_ratio = params.get("max_add_ratio", 0.30)

        # 模型
        self.model_data = None
        self._load_model()

        # 做T策略
        self.t_strategy = TStrategy() if T_STRATEGY_AVAILABLE else None

        # 邮件
        self.email_notifier = create_email_notifier_from_env() if EMAIL_AVAILABLE else None

        # 关注股票
        self.watchlist = get_watchlist()  # 从配置读取

    def _load_model(self):
        """加载模型（支持v3集成和v2单模型）"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                self.model_data = model_data
                if 'models' in model_data:
                    # v3 集成
                    print(f"✓ 加载v3集成模型 ({len(model_data['models'])}个子模型)")
                elif 'model' in model_data:
                    # v2 单模型
                    self.model = model_data.get('model')
                    print(f"✓ 加载v2单模型")
                else:
                    print(f"⚠ 未识别的模型格式")
            except Exception as e:
                print(f"⚠ 模型加载失败: {e}")

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def update_prices(self):
        """Update prices from minute-bar data"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, stock_name FROM positions")
        positions = cursor.fetchall()
        all_symbols = list(positions) + [(w["symbol"], w["name"]) for w in self.watchlist]

        for symbol, name in all_symbols:
            print(f"  {name}({symbol})...", end=" ")
            cursor.execute("SELECT close FROM kline_30m WHERE symbol=? ORDER BY date DESC LIMIT 1", (symbol,))
            row = cursor.fetchone()
            if row:
                price = float(row[0])
                cursor.execute("UPDATE positions SET current_price=? WHERE symbol=?", (price, symbol))
                print(f"done {price:.2f}")
            else:
                prices = self.data_handler.get_realtime_prices([symbol])
                if prices and symbol in prices:
                    price = prices[symbol]["price"]
                    cursor.execute("UPDATE positions SET current_price=? WHERE symbol=?", (price, symbol))
                    print(f"rt {price:.2f}")
                else:
                    print("fail")
        conn.commit()
        conn.close()

    def get_positions(self) -> Dict[str, Position]:
        """获取持仓"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT symbol, stock_name, shares, cost_price, current_price FROM positions')

        positions = {}
        for row in cursor.fetchall():
            pos = Position(
                symbol=row[0],
                stock_name=row[1],
                shares=row[2],
                cost_price=row[3],
                current_price=row[4]
            )
            positions[pos.symbol] = pos
        conn.close()
        return positions

    def _predict_up_prob(self, symbol: str) -> Optional[float]:
        """预测上涨概率（支持v3集成和v2单模型）"""
        if self.model_data is None or not FEATURE_ENGINEER_AVAILABLE:
            return None

        conn = self._get_conn()
        df = pd.read_sql_query('SELECT * FROM kline_30m WHERE symbol=? ORDER BY date', conn, params=(symbol,))
        conn.close()

        if len(df) < 150:
            return None

        try:
            # 基础特征
            features = EnhancedFeatureEngineer.calculate_features(df)

            # v3 高级特征
            try:
                from strategy.train_lgb_v3 import AdvancedFeatureEngineer, TIME_FEATURES
                adv_features = AdvancedFeatureEngineer.calculate_advanced_features(df)
                features = pd.concat([features, adv_features], axis=1)
            except Exception:
                pass

            # 过滤时间特征 + 使用模型特征名
            feature_names = self.model_data.get('feature_names')
            if feature_names:
                missing = [c for c in feature_names if c not in features.columns]
                for c in missing:
                    features[c] = 0
                features = features[feature_names]
            else:
                TIME_FEATS = ['day_of_week', 'day_of_month', 'hour', 'minute', 'is_morning', 'is_afternoon', 'is_first_hour', 'is_last_hour']
                features = features[[c for c in features.columns if c not in TIME_FEATS]]

            last_features = features.iloc[-1].fillna(0)

            # 预测
            if 'models' in self.model_data:
                # v3 集成：平均概率
                probs = []
                for m in self.model_data['models']:
                    try:
                        probs.append(m.predict_proba([last_features.values])[0][1])
                    except Exception:
                        probs.append(0.5)
                return float(np.mean(probs))
            else:
                # v2/v1 单模型
                return self.model.predict_proba([last_features.values])[0][1]
        except Exception as e:
            sys.stderr.write(f"预测失败 {symbol}: {e}\n")
            return None

    def analyze_positions(self) -> List[Dict]:
        """分析持仓，给出操作建议"""
        positions = self.get_positions()
        suggestions = []

        for symbol, pos in positions.items():
            up_prob = self._predict_up_prob(symbol)
            profit_pct = pos.profit_pct

            suggestion = {
                'symbol': symbol,
                'stock_name': pos.stock_name,
                'shares': pos.shares,
                'cost_price': pos.cost_price,
                'current_price': pos.current_price,
                'profit': pos.profit,
                'profit_pct': profit_pct,
                'up_prob': up_prob or 0,
                'action': '持有',
                'reason': ''
            }

            # 熊市策略：浮亏超过20%且模型看涨，建议补仓
            if profit_pct <= -20 and up_prob and up_prob >= self.add_position_up_prob:
                add_shares = int(pos.shares * self.max_add_ratio / 100) * 100
                add_amount = add_shares * pos.current_price

                if add_amount <= self.available_cash:
                    suggestion['action'] = '补仓'
                    suggestion['reason'] = f"浮亏{profit_pct:.0f}%，模型看涨{up_prob:.0%}，建议补仓{add_shares}股"
                    suggestion['add_shares'] = add_shares
                    suggestion['add_amount'] = add_amount

            # 浮亏严重但模型看跌，提示风险
            elif profit_pct <= -25 and up_prob and up_prob < 0.45:
                suggestion['action'] = '观望'
                suggestion['reason'] = f"浮亏{profit_pct:.0f}%，但模型看跌，暂不补仓"

            # 浮盈超过15%且模型看跌，提示减仓机会
            elif profit_pct >= 15 and up_prob and up_prob < 0.45:
                suggestion['action'] = '减仓'
                suggestion['reason'] = f"浮盈{profit_pct:.0f}%，模型看跌，可考虑减仓"

            else:
                suggestion['reason'] = f"持有观望，等待机会"

            suggestions.append(suggestion)

        return suggestions

    def analyze_t_opportunities(self) -> List[Dict]:
        """
        分析做T机会，给出具体操作建议

        Returns:
            做T建议列表，包含具体买入价位、卖出价位、操作数量
        """
        if not self.t_strategy:
            return []

        positions = self.get_positions()
        conn = self._get_conn()

        t_suggestions = []
        for symbol, pos in positions.items():
            # 获取30分钟数据
            df = pd.read_sql_query(
                'SELECT date, open, high, low, close, volume FROM kline_30m WHERE symbol=? ORDER BY date',
                conn, params=(symbol,)
            )

            if len(df) < 50:
                continue

            # 分析做T机会
            position_dict = {
                'symbol': symbol,
                'stock_name': pos.stock_name,
                'shares': pos.shares,
                'cost_price': pos.cost_price,
                'current_price': pos.current_price
            }

            suggestion = self.t_strategy.analyze(df, position_dict)

            t_suggestions.append({
                'symbol': symbol,
                'stock_name': pos.stock_name,
                'action': suggestion.action,
                'current_price': suggestion.current_price,
                'cost_price': suggestion.cost_price,
                'profit_pct': suggestion.profit_pct,
                'buy_price': suggestion.buy_price,
                'sell_price': suggestion.sell_price,
                'buy_shares': suggestion.buy_shares,
                'sell_shares': suggestion.sell_shares,
                'support_price': suggestion.support_price,
                'resistance_price': suggestion.resistance_price,
                'intraday_range': suggestion.intraday_range,
                'trend': suggestion.trend,
                'reason': suggestion.reason,
                'risk_level': suggestion.risk_level
            })

        conn.close()
        return t_suggestions

    def run(self, send_email: bool = True):
        """执行监控"""
        print("\n" + "=" * 70)
        print(f"交易监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        # 更新价格
        self.update_prices()

        # 获取持仓
        positions = self.get_positions()

        # 计算汇总
        total_cost = sum(p.cost_value for p in positions.values())
        total_value = sum(p.market_value for p in positions.values())
        total_profit = total_value - total_cost
        total_profit_pct = total_profit / total_cost * 100 if total_cost > 0 else 0

        print(f"\n【账户汇总】")
        print(f"  投入本金: ¥{total_cost:,.0f}")
        print(f"  持仓市值: ¥{total_value:,.0f}")
        print(f"  浮动盈亏: ¥{total_profit:,.0f} ({total_profit_pct:.1f}%)")
        print(f"  可用现金: ¥{self.available_cash:,.0f}")
        print(f"  总资产: ¥{total_value + self.available_cash:,.0f}")

        # 分析持仓
        suggestions = self.analyze_positions()

        # 分析做T机会
        t_suggestions = self.analyze_t_opportunities()

        print(f"\n【持仓分析】")
        for s in suggestions:
            action_emoji = {'补仓': '🟢', '减仓': '🔴', '持有': '⚪', '观望': '⚠️'}.get(s['action'], '⚪')
            profit_emoji = '✅' if s['profit'] > 0 else '❌'
            print(f"  {profit_emoji} {s['stock_name']}: {s['shares']}股 @ ¥{s['cost_price']:.2f} → ¥{s['current_price']:.2f}")
            print(f"     盈亏: ¥{s['profit']:,.0f} ({s['profit_pct']:.1f}%) | 模型预测: {s['up_prob']:.0%}")
            print(f"     {action_emoji} {s['action']}: {s['reason']}")

        # 做T建议
        if t_suggestions:
            print(f"\n【做T操作建议】")
            for t in t_suggestions:
                action_emoji = {'适合做T': '🟢', '可减仓': '🔵', '观望': '⚠️', '不建议': '❌'}.get(t['action'], '⚪')
                support_str = f"¥{t['support_price']:.2f}" if t['support_price'] else "-"
                resistance_str = f"¥{t['resistance_price']:.2f}" if t['resistance_price'] else "-"
                print(f"  {action_emoji} {t['stock_name']}: 现价¥{t['current_price']:.2f} | 支撑{support_str} | 阻力{resistance_str}")
                print(f"     {t['reason']}")
                if t['buy_price'] and t['buy_shares']:
                    print(f"     💰 建议买入: ¥{t['buy_price']:.2f} × {t['buy_shares']}股")
                if t['sell_price'] and t['sell_shares']:
                    print(f"     💵 建议卖出: ¥{t['sell_price']:.2f} × {t['sell_shares']}股")

        # 关注股票
        print(f"\n【关注股票】")
        conn = self._get_conn()
        for stock in self.watchlist:
            df = pd.read_sql_query('SELECT close FROM kline_30m WHERE symbol=? ORDER BY date DESC LIMIT 1', conn, params=(stock['symbol'],))
            if not df.empty:
                price = float(df['close'].iloc[0])
                print(f"  👀 {stock['name']}: ¥{price:.2f}")
        conn.close()

        # 发送邮件
        if send_email and self.email_notifier:
            self._send_email(positions, suggestions, total_cost, total_value, total_profit, t_suggestions)

        # 保存结果
        self._save_result(positions, suggestions, total_cost, total_value, total_profit)

        return suggestions

    def _send_email(self, positions, suggestions, total_cost, total_value, total_profit, t_suggestions=None):
        """发送邮件"""
        if t_suggestions is None:
            t_suggestions = []

        subject = f"【持仓日报】{datetime.now().strftime('%m-%d')} - 浮亏¥{total_profit:,.0f}"

        # 获取关注股票价格
        conn = self._get_conn()
        watchlist_prices = []
        for stock in self.watchlist:
            df = pd.read_sql_query('SELECT close FROM kline_30m WHERE symbol=? ORDER BY date DESC LIMIT 1', conn, params=(stock['symbol'],))
            if not df.empty:
                price = float(df['close'].iloc[0])
                watchlist_prices.append({'symbol': stock['symbol'], 'name': stock['name'], 'price': price})
        conn.close()

        # 构建持仓表格
        rows = ""
        for s in sorted(suggestions, key=lambda x: x['profit_pct'], reverse=True):
            color = "green" if s['profit'] > 0 else "red"
            action_color = {'补仓': '#28a745', '减仓': '#dc3545', '持有': '#6c757d', '观望': '#ffc107'}[s['action']]
            # 预测上涨概率颜色
            up_prob = s['up_prob']
            prob_color = "green" if up_prob >= 0.55 else "red" if up_prob < 0.45 else "gray"
            prob_text = f"看涨{up_prob:.0%}" if up_prob >= 0.55 else f"看跌{up_prob:.0%}" if up_prob < 0.45 else f"中性{up_prob:.0%}"
            rows += f"""
            <tr>
                <td>{s['stock_name']}</td>
                <td>{s['symbol']}</td>
                <td style="text-align:right">{s['shares']:,}</td>
                <td style="text-align:right">¥{s['cost_price']:.3f}</td>
                <td style="text-align:right">¥{s['current_price']:.3f}</td>
                <td style="text-align:right;color:{color}">¥{s['profit']:,.0f}</td>
                <td style="text-align:right;color:{color}">{s['profit_pct']:.1f}%</td>
                <td style="text-align:center;color:{prob_color}">{prob_text}</td>
                <td style="text-align:center;color:{action_color};font-weight:bold">{s['action']}</td>
            </tr>
            """

        # 关注股票表格
        watchlist_rows = ""
        for w in watchlist_prices:
            watchlist_rows += f"<tr><td>{w['name']}</td><td>{w['symbol']}</td><td style='text-align:right'>¥{w['price']:.2f}</td></tr>"

        # 做T建议（具体的操作建议）
        t_tips = []
        t_rows = ""

        # 分类显示做T建议
        can_t = [t for t in t_suggestions if t['action'] == '适合做T']
        can_reduce = [t for t in t_suggestions if t['action'] == '可减仓']
        watch = [t for t in t_suggestions if t['action'] == '观望']

        for t in can_t:
            action_color = '#28a745'
            support_str = f"¥{t['support_price']:.2f}" if t['support_price'] else "-"
            resistance_str = f"¥{t['resistance_price']:.2f}" if t['resistance_price'] else "-"
            buy_price_str = f"¥{t['buy_price']:.2f}" if t['buy_price'] else "-"
            sell_price_str = f"¥{t['sell_price']:.2f}" if t['sell_price'] else "-"
            t_rows += f"""
            <tr style="background:#e8f5e9">
                <td style="color:{action_color};font-weight:bold">🟢 适合做T</td>
                <td>{t['stock_name']}</td>
                <td style="text-align:right">¥{t['current_price']:.2f}</td>
                <td style="text-align:right">{support_str}</td>
                <td style="text-align:right">{resistance_str}</td>
                <td style="text-align:right;color:red">{t['profit_pct']:.1f}%</td>
                <td style="text-align:right">{t['buy_shares'] if t['buy_shares'] else '-'}</td>
                <td style="text-align:right">{buy_price_str}</td>
                <td style="text-align:right">{sell_price_str}</td>
            </tr>
            <tr><td colspan="9" style="padding-left:20px;font-size:12px;color:#666">📝 {t['reason']}</td></tr>
            """

        for t in can_reduce:
            action_color = '#17a2b8'
            support_str = f"¥{t['support_price']:.2f}" if t['support_price'] else "-"
            resistance_str = f"¥{t['resistance_price']:.2f}" if t['resistance_price'] else "-"
            sell_price_str = f"¥{t['sell_price']:.2f}" if t['sell_price'] else "-"
            t_rows += f"""
            <tr style="background:#e3f2fd">
                <td style="color:{action_color};font-weight:bold">🔵 可减仓</td>
                <td>{t['stock_name']}</td>
                <td style="text-align:right">¥{t['current_price']:.2f}</td>
                <td style="text-align:right">{support_str}</td>
                <td style="text-align:right">{resistance_str}</td>
                <td style="text-align:right;color:red">{t['profit_pct']:.1f}%</td>
                <td style="text-align:right">{t['sell_shares'] if t['sell_shares'] else '-'}</td>
                <td style="text-align:right">-</td>
                <td style="text-align:right">{sell_price_str}</td>
            </tr>
            <tr><td colspan="9" style="padding-left:20px;font-size:12px;color:#666">📝 {t['reason']}</td></tr>
            """

        for t in watch:
            support_str = f"¥{t['support_price']:.2f}" if t['support_price'] else "-"
            resistance_str = f"¥{t['resistance_price']:.2f}" if t['resistance_price'] else "-"
            t_rows += f"""
            <tr>
                <td style="color:#ffc107;font-weight:bold">⚠️ 观望</td>
                <td>{t['stock_name']}</td>
                <td style="text-align:right">¥{t['current_price']:.2f}</td>
                <td style="text-align:right">{support_str}</td>
                <td style="text-align:right">{resistance_str}</td>
                <td style="text-align:right;color:red">{t['profit_pct']:.1f}%</td>
                <td colspan="3" style="text-align:center">等待更好机会</td>
            </tr>
            <tr><td colspan="9" style="padding-left:20px;font-size:12px;color:#666">📝 {t['reason']}</td></tr>
            """

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; padding: 20px; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; border-radius: 10px; overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); color: white; padding: 25px; }}
        .header h1 {{ margin: 0; font-size: 24px; }}
        .content {{ padding: 20px; }}
        .summary {{ display: flex; justify-content: space-around; padding: 20px; background: #f8f9fa; border-radius: 8px; margin-bottom: 20px; }}
        .summary-item {{ text-align: center; }}
        .summary-item .value {{ font-size: 22px; font-weight: bold; }}
        .summary-item .label {{ font-size: 12px; color: #666; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th {{ background: #667eea; color: white; padding: 10px 6px; font-size: 12px; }}
        td {{ padding: 10px 6px; border-bottom: 1px solid #eee; font-size: 13px; }}
        .footer {{ padding: 15px; background: #f8f9fa; font-size: 12px; color: #666; text-align: center; }}
        .suggestion {{ margin-top: 20px; padding: 15px; background: #e7f3ff; border-radius: 8px; }}
        .t-tip {{ margin-top: 20px; padding: 15px; background: #fff3cd; border-radius: 8px; }}
        .watchlist-section {{ margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 持仓日报</h1>
            <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 熊市深套，坚持做T降成本</p>
        </div>

        <div class="content">
            <div class="summary">
                <div class="summary-item">
                    <div class="value">¥{total_cost:,.0f}</div>
                    <div class="label">投入本金</div>
                </div>
                <div class="summary-item">
                    <div class="value">¥{total_value:,.0f}</div>
                    <div class="label">持仓市值</div>
                </div>
                <div class="summary-item">
                    <div class="value" style="color: red">¥{total_profit:,.0f}</div>
                    <div class="label">浮动盈亏</div>
                </div>
                <div class="summary-item">
                    <div class="value">¥{self.available_cash:,.0f}</div>
                    <div class="label">可用现金</div>
                </div>
            </div>

            <h3 style="margin: 20px 0 10px 0; border-bottom: 2px solid #667eea; padding-bottom: 5px;">📈 持仓明细</h3>
            <table>
                <tr>
                    <th>股票</th><th>代码</th><th>持股</th><th>成本</th><th>现价</th><th>盈亏</th><th>幅度</th><th>预测上涨</th><th>建议</th>
                </tr>
                {rows}
            </table>

            <div class="watchlist-section">
                <h3 style="margin: 20px 0 10px 0; border-bottom: 2px solid #17a2b8; padding-bottom: 5px;">👀 关注股票</h3>
                <table>
                    <tr><th>股票</th><th>代码</th><th>现价</th></tr>
                    {watchlist_rows}
                </table>
            </div>

            <div class="t-tip">
                <h4 style="margin: 0 0 10px 0">🔄 做T操作建议（具体价位）</h4>
                {'<table><tr><th>操作</th><th>股票</th><th>现价</th><th>支撑</th><th>阻力</th><th>浮亏</th><th>数量</th><th>买入价</th><th>卖出价</th></tr>' + t_rows + '</table>' if t_rows else '<p style="color:#666">当前暂无适合做T的机会</p>'}
                <ul style="margin: 10px 0 0 0; padding-left: 20px;">
                <li><b>做T原则</b>: 先买后卖（正T），在支撑位附近买入，反弹到阻力位卖出原有持仓</li>
                <li><b>风险提示</b>: 做T需确保有足够现金，每次操作数量不超过持仓的1/3</li>
                <li><b>技术分析</b>: 支撑位=日内低点附近，阻力位=日内高点附近</li>
                </ul>
            </div>

            <div class="suggestion">
                <h4 style="margin: 0 0 10px 0">💡 操作建议</h4>
                <ul style="margin: 0; padding-left: 20px;">
                {''.join([f"<li><b>{s['stock_name']}</b>: {s['action']} - {s['reason']}</li>" for s in suggestions if s['action'] != '持有'])}
                </ul>
                <p style="margin: 10px 0 0 0; font-size: 12px; color: #666;">
                <b>预测上涨</b>列说明: 模型预测该股票未来上涨概率，>55%看涨(绿色)，<45%看跌(红色)，其余中性
                </p>
            </div>
        </div>

        <div class="footer">
            总投入 ¥{total_cost:,.0f} | 总浮亏 ¥{total_profit:,.0f} ({total_profit/total_cost*100:.1f}%) | 可用现金 ¥{self.available_cash:,.0f}<br>
            熊市策略：不割肉、坚持做T、逢低补仓
        </div>
    </div>
</body>
</html>
"""

        NL = "\n"
        holdings_text = NL.join([f"{s['stock_name']}: {s['shares']:,}股 @ ¥{s['cost_price']:.3f} → ¥{s['current_price']:.3f} | 盈亏¥{s['profit']:,.0f}({s['profit_pct']:.1f}%) | 预测上涨{s['up_prob']:.0%}" for s in suggestions])
        watchlist_text = NL.join([f"{w['name']}: ¥{w['price']:.2f}" for w in watchlist_prices])
        
        text = f"""
持仓日报 - {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*60}
投入本金: ¥{total_cost:,.0f}
持仓市值: ¥{total_value:,.0f}
浮动盈亏: ¥{total_profit:,.0f} ({total_profit/total_cost*100:.1f}%)
可用现金: ¥{self.available_cash:,.0f}
{'='*60}

【持仓明细】
{holdings_text}

【关注股票】
{watchlist_text}

【做T操作建议】
"""

        # 添加具体的做T建议
        if t_suggestions:
            for t in t_suggestions:
                action_emoji = {'适合做T': '🟢', '可减仓': '🔵', '观望': '⚠️', '不建议': '❌'}.get(t['action'], '⚪')
                support_str = f"¥{t['support_price']:.2f}" if t['support_price'] else "-"
                resistance_str = f"¥{t['resistance_price']:.2f}" if t['resistance_price'] else "-"
                text += f"{action_emoji} {t['stock_name']}: 现价¥{t['current_price']:.2f} 支撑{support_str} 阻力{resistance_str}\n"
                text += f"   {t['reason']}\n"
                if t['buy_price'] and t['buy_shares']:
                    text += f"   💰 建议买入: ¥{t['buy_price']:.2f} × {t['buy_shares']}股\n"
                if t['sell_price'] and t['sell_shares']:
                    text += f"   💵 建议卖出: ¥{t['sell_price']:.2f} × {t['sell_shares']}股\n"
        else:
            text += "当前暂无适合做T的机会\n"

        text += """
做T原则: 先买后卖（正T），在支撑位附近买入，反弹到阻力位卖出原有持仓
"""

        self.email_notifier.send(subject, text, html)
        print("✓ 邮件已发送")

    def _save_result(self, positions, suggestions, total_cost, total_value, total_profit):
        """保存结果"""
        os.makedirs(LOGS_DIR, exist_ok=True)
        result = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_cost': total_cost,
                'total_value': total_value,
                'total_profit': total_profit,
                'available_cash': self.available_cash
            },
            'positions': [asdict(p) for p in positions.values()],
            'suggestions': suggestions
        }
        result_file = os.path.join(LOGS_DIR, f'monitor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✓ 结果已保存")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='交易监控')
    parser.add_argument('--update', action='store_true', help='仅更新数据')
    parser.add_argument('--no-email', action='store_true', help='不发送邮件')
    args = parser.parse_args()

    monitor = TradingMonitor()

    if args.update:
        monitor.update_prices()
    else:
        monitor.run(send_email=not args.no_email)


if __name__ == "__main__":
    main()