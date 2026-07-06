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
MODEL_PATH = os.path.join(BASE_DIR, 'python/models/lgb_30m/model.pkl')
DAILY_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lgb_hs300_enhanced', 'model.pkl')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
sys.path.insert(0, BASE_DIR)

from data.data_handler import DataHandler

try:
    from strategy.email_notifier import EmailNotifier, create_email_notifier_from_env
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

try:
    from strategy.features import FeaturePipeline, rename_features_for_model
    FEATURE_ENGINEER_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEER_AVAILABLE = False

# 补仓顾问(诚实模型) — 方向信号唯一可信来源, 取代泄漏的 lgb_30m/v9
try:
    from strategy.add_advisor_ml import (
        load_final_model as _load_advisor_model,
        score_holding as _advisor_score,
        PURGE_DAYS as _ADVISOR_PURGE,
    )
    ADVISOR_AVAILABLE = True
except ImportError:
    ADVISOR_AVAILABLE = False

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
        self.data_handler = DataHandler(force_refresh=True)

        # 账户参数
        self.available_cash = get_available_cash()  # 从配置读取

        # 策略参数（熊市策略）
        params = get_strategy_params()
        self.add_position_threshold = params.get('add_position_threshold', -0.20)
        self.add_position_up_prob = params.get("add_position_prob", 0.01)
        self.max_add_ratio = params.get("max_add_ratio", 0.30)

        # 方向模型(诚实补仓顾问)
        self._load_model()

        # 做T策略
        self.t_strategy = TStrategy() if T_STRATEGY_AVAILABLE else None

        # 邮件
        self.email_notifier = create_email_notifier_from_env() if EMAIL_AVAILABLE else None

        # 关注股票
        self.watchlist = get_watchlist()  # 从配置读取

    def _load_model(self):
        """加载方向模型 —— 只用诚实的补仓顾问模型。
        旧的 lgb_30m / v9日线 已停用: 其报告 IC≈0.38 是泄漏假象, 真实样本外 edge≈0。
        """
        # 补仓顾问(诚实模型): 方案2上涨概率/预期收益 + 方案3候选态P(止盈)
        # 这是方向信号的唯一可信来源, 与网页 /advisor/holdings 同一模型
        self.advisor = None
        if ADVISOR_AVAILABLE and FEATURE_ENGINEER_AVAILABLE:
            try:
                adv = _load_advisor_model()
                adv['pipeline'] = FeaturePipeline({
                    'label': '日线', 'horizon': adv['horizon'], 'db_table': 'kline_daily',
                    'min_history': 120, 'purged_gap': _ADVISOR_PURGE, 'north_shift_days': 1,
                })
                self.advisor = adv
                print(f"✓ 补仓顾问模型: 训练截至 {adv.get('cutoff')} "
                      f"方案2{'可用' if adv.get('a2_usable') else '薄'}/"
                      f"方案3{'可用' if adv.get('a3_usable') else '薄'}")
            except FileNotFoundError:
                print("⚠ 补仓顾问模型缺失(models/add_advisor/model.pkl), 方向信号不可用")
            except Exception as e:
                print(f"⚠ 补仓顾问模型加载失败: {e}")

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

    def _predict_advisor(self, symbol: str) -> Optional[Dict]:
        """诚实模型打分: 返回 pup(上涨概率)/reg(预期20日收益)/ptp(P止盈)/cand(候选态)"""
        if not self.advisor:
            return None
        conn = self._get_conn()
        try:
            s = _advisor_score(
                conn, self.advisor['pipeline'], symbol,
                self.advisor['feat_names'], self.advisor['reg'],
                self.advisor['clf_s'], self.advisor['clf_tb'])
        except Exception:
            s = None
        finally:
            conn.close()
        return s

    def _load_ml_signals(self) -> Optional[Dict]:
        """加载 ML 日线策略信号"""
        signal_file = os.path.join(BASE_DIR, 'data', 'daily_ml_signals.json')
        if not os.path.exists(signal_file):
            return None
        try:
            with open(signal_file) as f:
                data = json.load(f)
            # 检查是否是今天的信号
            today = datetime.now().strftime('%Y-%m-%d')
            if data.get('date') == today:
                return data
        except Exception:
            pass
        return None

    def analyze_positions(self) -> List[Dict]:
        """分析持仓，给出操作建议"""
        positions = self.get_positions()
        suggestions = []

        a3_ok = bool(self.advisor and self.advisor.get('a3_usable'))
        for symbol, pos in positions.items():
            adv = self._predict_advisor(symbol)  # 诚实模型: pup/reg/ptp/cand
            profit_pct = pos.profit_pct

            pup = adv['pup'] if adv else None          # 上涨概率 0-1
            ret20 = adv['reg'] if adv else None         # 预期20日收益
            ptp = adv['ptp'] if adv else None           # P(先触止盈)
            cand = adv['cand'] if adv else False        # 候选态(超卖破MA20)

            suggestion = {
                'symbol': symbol,
                'stock_name': pos.stock_name,
                'shares': pos.shares,
                'cost_price': pos.cost_price,
                'current_price': pos.current_price,
                'profit': pos.profit,
                'profit_pct': profit_pct,
                'up_prob': pup if pup is not None else 0,
                'daily_pred': ret20,       # 预期20日收益
                'tp_prob': ptp,
                'candidate': cand,
                'action': '持有',
                'reason': ''
            }

            if adv is None:
                suggestion['reason'] = "补仓顾问模型未就绪, 仅规则持有(旧泄漏模型已停用)"
                suggestions.append(suggestion)
                continue

            # 补仓: 深浮亏 + 候选态(超卖破MA20) + 方案3占优(P止盈≥0.55)且预期为正
            # 与网页 _verdict "可小仓试探" 同口径 —— 网页与邮件说同一件事
            if profit_pct <= -20 and cand and a3_ok and ptp >= 0.55 and ret20 > 0:
                add_shares = int(pos.shares * self.max_add_ratio / 100) * 100
                add_amount = add_shares * pos.current_price
                if add_amount <= self.available_cash:
                    suggestion['action'] = '补仓'
                    suggestion['reason'] = (f"浮亏{profit_pct:.0f}%+超卖候选态, "
                                            f"P(止盈){ptp:.2f}占优, 可小仓试探{add_shares}股(严格止损)")
                    suggestion['add_shares'] = add_shares
                    suggestion['add_amount'] = add_amount
                else:
                    suggestion['reason'] = f"浮亏{profit_pct:.0f}%达补仓条件, 但可用资金不足"

            # 深浮亏但模型不占优 —— 不接飞刀
            elif profit_pct <= -25 and (pup < 0.45 or ret20 < -0.005):
                suggestion['action'] = '观望'
                suggestion['reason'] = f"浮亏{profit_pct:.0f}%, 模型不占优(涨概率{pup:.2f}), 不补等企稳"

            # 浮盈且模型看跌 —— 减仓机会
            elif profit_pct >= 15 and (pup < 0.45 or ret20 < -0.005):
                suggestion['action'] = '减仓'
                suggestion['reason'] = f"浮盈{profit_pct:.0f}%, 模型看跌(涨概率{pup:.2f}), 可考虑减仓"

            else:
                suggestion['reason'] = f"持有观望(涨概率{pup:.2f}, 预期20日{ret20:+.1%})"

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

        # 分析做T机会（异常不应阻断邮件）
        try:
            t_suggestions = self.analyze_t_opportunities()
        except Exception as e:
            sys.stderr.write(f"⚠️ 做T分析异常: {e}\n")
            t_suggestions = []

        print(f"\n【持仓分析】")
        for s in suggestions:
            action_emoji = {'补仓': '🟢', '减仓': '🔴', '持有': '⚪', '观望': '⚠️'}.get(s['action'], '⚪')
            profit_emoji = '✅' if s['profit'] > 0 else '❌'
            print(f"  {profit_emoji} {s['stock_name']}: {s['shares']}股 @ ¥{s['cost_price']:.2f} → ¥{s['current_price']:.2f}")
            print(f"     盈亏: ¥{s['profit']:,.0f} ({s['profit_pct']:.1f}%) | 涨概率: {s['up_prob']:.2f}" + (f" | 预期20日: {s['daily_pred']:+.1%}" if s.get('daily_pred') is not None else "") + (f" | P(止盈): {s['tp_prob']:.2f}" if s.get('tp_prob') is not None else ""))
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

        # ── ML 日线信号 ──
        ml_signals = self._load_ml_signals()
        if ml_signals:
            print(f"\n【ML 日线量化信号】")
            print(f"  模型: {ml_signals.get('model', {}).get('model', '?')}"
                  f" | RankIC={ml_signals.get('model', {}).get('RankIC', '?')}")

            for s in ml_signals.get('signals', []):
                icon = {'BUY': '🟢', 'SELL': '🔴', 'ADD': '🔵'}.get(s['action'], '⚪')
                extra = ''
                if s['action'] == 'BUY':
                    extra = f" | {s.get('shares', '?')}股 ¥{s.get('amount', 0):,.0f}"
                elif s['action'] == 'ADD':
                    extra = f" | +{s.get('add_shares', '?')}股 ¥{s.get('add_amount', 0):,.0f}"
                elif s['action'] == 'SELL':
                    extra = f" | 持仓{s.get('hold_days', '?')}天 | {s.get('pnl', '?')}"
                print(f"  {icon} {s['action']:4s} | {s['symbol']:12s} {s.get('name', '')[:8]:8s}"
                      f" | @{s['price']:.2f} | 排名:{s['rank']} | {s['reason']}{extra}")

            if not ml_signals.get('signals'):
                print(f"  ⚪ 无交易信号")

            top5 = ml_signals.get('top5', [])
            if top5:
                print(f"\n  ML Top-5 推荐:")
                for t in top5:
                    print(f"    {t['rank']:3d}. {t['symbol']:12s} 分数:{t['score']:.4f}")

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
            # 预期收益率颜色
            pred_ret = s['up_prob']
            daily_pred = s.get('daily_pred')
            prob_color = "green" if pred_ret >= 0.01 else "red" if pred_ret < -0.005 else "gray"
            prob_text = f"30m:{pred_ret:+.3f}"
            if daily_pred is not None:
                prob_text += f" 日线:{daily_pred:+.3f}"
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
                <b>预期收益</b>列说明: 模型预测该股票预期收益率，>1%看涨(绿色)，<-0.5%看跌(红色)，其余中性
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