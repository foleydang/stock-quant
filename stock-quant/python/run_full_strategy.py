#!/usr/bin/env python3
"""
完整策略执行系统
包括：数据管理、模型训练、回测、选股、邮件通知

用法:
    python3 run_full_strategy.py
"""

import os
import sys
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_PATH = os.path.join(DATA_DIR, 'stock_data.db')
MODEL_DIR = os.path.join(BASE_DIR, 'models/lgb_hs300')

# ============ 邮件配置 ============
EMAIL_CONFIG = {
    'smtp_server': 'smtp.qq.com',  # QQ邮箱SMTP
    'smtp_port': 465,
    'sender': '',  # 发件人邮箱
    'password': '',  # 授权码
    'receiver': '',  # 收件人
    'enabled': False  # 是否启用邮件
}

# ============ 策略参数 ============
STRATEGY_CONFIG = {
    'initial_capital': 100000,  # 初始资金10万
    'position_pct': 0.20,       # 每次建仓20%
    'stop_loss_pct': 0.10,      # 止损10%
    'take_profit_pct': 0.10,    # 止盈10%
    'buy_threshold': 0.55,      # 买入阈值
    'sell_threshold': 0.40,     # 卖出阈值
    'strong_buy_threshold': 0.65,
    'strong_sell_threshold': 0.35,
    'min_hold_periods': 16,     # T+1
    'min_trade_amount': 5000,   # 最小交易金额
    'trade_cooldown': 8,        # 交易冷却期
}


class EmailNotifier:
    """邮件通知器"""

    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get('enabled', False)

    def send_trade_alert(self, subject: str, trades: List[Dict], summary: Dict):
        """发送交易提醒"""
        if not self.enabled or not self.config.get('sender'):
            print(f"[邮件未配置] {subject}")
            return

        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['sender']
            msg['To'] = self.config['receiver']
            msg['Subject'] = subject

            # 构建邮件内容
            content = f"""
            <html>
            <body>
            <h2>LGBM量化交易系统 - {datetime.now().strftime('%Y-%m-%d %H:%M')}</h2>

            <h3>策略汇总</h3>
            <table border="1" cellpadding="5">
                <tr><td>初始资金</td><td>¥{summary['initial_capital']:,.0f}</td></tr>
                <tr><td>最终市值</td><td>¥{summary['final_value']:,.0f}</td></tr>
                <tr><td>总盈亏</td><td style="color: {'green' if summary['profit'] >= 0 else 'red'}">¥{summary['profit']:,.0f}</td></tr>
                <tr><td>收益率</td><td style="color: {'green' if summary['profit_rate'] >= 0 else 'red'}">{summary['profit_rate']:.2f}%</td></tr>
                <tr><td>交易次数</td><td>{summary['trade_count']}笔</td></tr>
                <tr><td>胜率</td><td>{summary['win_rate']:.1f}%</td></tr>
            </table>

            <h3>交易记录</h3>
            <table border="1" cellpadding="5">
                <tr><th>时间</th><th>股票</th><th>操作</th><th>价格</th><th>数量</th><th>盈亏</th></tr>
                {''.join([f"<tr><td>{t['date'][:16]}</td><td>{t['symbol']}</td><td style='color:{'green' if t['type']=='BUY' else 'red'}'>{t['type']}</td><td>¥{t['price']:.2f}</td><td>{t['shares']}</td><td>{t.get('profit', '--')}</td></tr>" for t in trades[-20:]])}
            </table>

            <h3>当前持仓</h3>
            <table border="1" cellpadding="5">
                <tr><th>股票</th><th>数量</th><th>成本</th><th>市值</th></tr>
                {''.join([f"<tr><td>{p['symbol']}</td><td>{p['shares']}</td><td>¥{p['avg_cost']:.2f}</td><td>¥{p['value']:,.0f}</td></tr>" for p in summary.get('positions', [])])}
            </table>
            </body>
            </html>
            """

            msg.attach(MIMEText(content, 'html', 'utf-8'))

            with smtplib.SMTP_SSL(self.config['smtp_server'], self.config['smtp_port']) as server:
                server.login(self.config['sender'], self.config['password'])
                server.send_message(msg)

            print(f"[邮件发送成功] {subject}")

        except Exception as e:
            print(f"[邮件发送失败] {e}")


class StrategyRunner:
    """策略执行器"""

    def __init__(self):
        self.notifier = EmailNotifier(EMAIL_CONFIG)
        self.results = {}

    def load_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """从数据库加载股票数据"""
        import sqlite3

        if not os.path.exists(DB_PATH):
            return None

        try:
            conn = sqlite3.connect(DB_PATH)
            query = '''
                SELECT date, open, high, low, close, volume
                FROM kline_30m WHERE symbol = ?
                ORDER BY date
            '''
            df = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()

            if df.empty:
                return None

            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date').reset_index(drop=True)

        except Exception as e:
            print(f"加载数据失败 {symbol}: {e}")
            return None

    def get_stock_list(self) -> List[Dict]:
        """获取股票列表"""
        import sqlite3

        if not os.path.exists(DB_PATH):
            return []

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT symbol, COUNT(*) as cnt
                FROM kline_30m
                GROUP BY symbol
                HAVING cnt >= 500
                ORDER BY cnt DESC
            ''')
            stocks = [{'symbol': row[0], 'count': row[1]} for row in cursor.fetchall()]
            conn.close()
            return stocks
        except:
            return []

    def run_backtest(self, symbol: str, df: pd.DataFrame) -> Dict:
        """执行单只股票回测"""
        import pickle

        # 加载模型
        model_path = os.path.join(MODEL_DIR, 'model.pkl')
        if not os.path.exists(model_path):
            return {'error': '模型不存在'}

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        model = model_data.get('model')

        # 导入特征工程
        sys.path.insert(0, BASE_DIR)
        from strategy.train_lgb_enhanced import EnhancedFeatureEngineer

        # 回测参数
        config = STRATEGY_CONFIG
        cash = config['initial_capital']
        holding_shares = 0
        total_cost = 0
        position_records = []
        last_trade_idx = -999

        trades = []
        buy_points = []
        sell_points = []
        portfolio_values = []

        # 遍历数据
        for i in range(150, len(df)):
            current_time = df['date'].iloc[i]
            current_price = round(df['close'].iloc[i], 2)

            # 计算特征和预测
            df_slice = df.iloc[:i+1]
            try:
                features = EnhancedFeatureEngineer.calculate_features(df_slice)
                if features.iloc[-1].isna().any():
                    up_prob = 0.5
                else:
                    prob = model.predict_proba([features.iloc[-1].values])[0]
                    up_prob = prob[1] if len(prob) > 1 else prob[0]
            except:
                up_prob = 0.5

            # 持仓状态
            avg_cost = total_cost / holding_shares if holding_shares > 0 else 0
            profit_pct = (current_price - avg_cost) / avg_cost if avg_cost > 0 else 0

            # T+1可用
            available_shares = 0
            for pr in position_records:
                if i - pr['entry_idx'] >= config['min_hold_periods']:
                    available_shares += pr['shares']

            in_cooldown = (i - last_trade_idx) < config['trade_cooldown']

            # 卖出逻辑
            sell_reason = None
            sell_shares = 0

            if holding_shares > 0 and available_shares > 0 and not in_cooldown:
                if up_prob < config['strong_sell_threshold']:
                    sell_reason = f"强烈看跌({up_prob:.0%})"
                    sell_shares = holding_shares
                elif up_prob < config['sell_threshold']:
                    sell_reason = f"看跌({up_prob:.0%})"
                    sell_shares = min(int(holding_shares * 0.5 / 100) * 100, available_shares)
                elif profit_pct >= config['take_profit_pct'] and up_prob < 0.50:
                    sell_reason = f"止盈({profit_pct*100:.1f}%)"
                    sell_shares = min(int(holding_shares * 0.5 / 100) * 100, available_shares)
                elif profit_pct <= -config['stop_loss_pct'] and up_prob < 0.48:
                    sell_reason = f"止损({profit_pct*100:.1f}%)"
                    sell_shares = holding_shares
                elif profit_pct <= -0.15:
                    sell_reason = f"深度止损({profit_pct*100:.1f}%)"
                    sell_shares = holding_shares

                if sell_shares > 0 and sell_shares * current_price < config['min_trade_amount']:
                    if sell_shares * current_price < config['min_trade_amount'] * 0.5:
                        sell_shares = 0
                        sell_reason = None
                    else:
                        sell_shares = min(available_shares, holding_shares)

                if sell_reason and sell_shares >= 100:
                    sell_amount = sell_shares * current_price
                    profit = (current_price - avg_cost) * sell_shares
                    cash = round(cash + sell_amount, 2)

                    # 更新持仓记录
                    remaining = sell_shares
                    new_records = []
                    for pr in position_records:
                        if remaining <= 0:
                            new_records.append(pr)
                        elif pr['shares'] <= remaining:
                            remaining -= pr['shares']
                        else:
                            new_pr = pr.copy()
                            new_pr['shares'] = pr['shares'] - remaining
                            remaining = 0
                            new_records.append(new_pr)
                    position_records = new_records

                    holding_shares -= sell_shares
                    total_cost = avg_cost * holding_shares if holding_shares > 0 else 0
                    last_trade_idx = i

                    sell_points.append({
                        'date': str(current_time),
                        'price': current_price,
                        'shares': sell_shares,
                        'profit': profit,
                        'reason': sell_reason
                    })
                    trades.append({
                        'date': str(current_time),
                        'symbol': symbol,
                        'type': 'SELL',
                        'price': current_price,
                        'shares': sell_shares,
                        'profit': profit,
                        'reason': sell_reason
                    })

            # 买入逻辑
            buy_reason = None
            buy_amount = 0

            if not in_cooldown:
                if up_prob > config['strong_buy_threshold']:
                    if holding_shares == 0:
                        buy_amount = config['initial_capital'] * config['position_pct']
                        buy_reason = f"强烈买入({up_prob:.0%})"
                    else:
                        buy_amount = config['initial_capital'] * 0.15
                        buy_reason = f"加仓({up_prob:.0%})"
                elif up_prob > config['buy_threshold'] and holding_shares == 0:
                    buy_amount = config['initial_capital'] * config['position_pct']
                    buy_reason = f"建仓({up_prob:.0%})"
                elif holding_shares > 0 and profit_pct < -0.05 and up_prob > config['buy_threshold']:
                    buy_amount = config['initial_capital'] * 0.10
                    buy_reason = f"补仓({up_prob:.0%})"

                if buy_amount > 0 and buy_amount < config['min_trade_amount']:
                    buy_amount = config['min_trade_amount']

                if buy_reason and cash >= buy_amount:
                    shares = int(buy_amount / current_price / 100) * 100
                    actual_amount = shares * current_price

                    if shares >= 100 and actual_amount <= cash:
                        cash = round(cash - actual_amount, 2)
                        if holding_shares == 0:
                            total_cost = actual_amount
                        else:
                            total_cost += actual_amount
                        holding_shares += shares

                        position_records.append({
                            'shares': shares,
                            'entry_idx': i,
                            'amount': actual_amount
                        })

                        avg_cost = total_cost / holding_shares
                        last_trade_idx = i

                        buy_points.append({
                            'date': str(current_time),
                            'price': current_price,
                            'shares': shares,
                            'reason': buy_reason
                        })
                        trades.append({
                            'date': str(current_time),
                            'symbol': symbol,
                            'type': 'BUY',
                            'price': current_price,
                            'shares': shares,
                            'reason': buy_reason
                        })

            # 记录市值
            stock_value = holding_shares * current_price if holding_shares > 0 else 0
            portfolio_values.append({
                'date': str(current_time),
                'value': round(cash + stock_value, 2),
                'cash': cash,
                'stock_value': stock_value
            })

        # 最终结果
        final_price = df['close'].iloc[-1]
        final_stock_value = holding_shares * final_price if holding_shares > 0 else 0
        final_value = round(cash + final_stock_value, 2)

        sell_trades = [t for t in trades if t['type'] == 'SELL']
        wins = [t for t in sell_trades if t.get('profit', 0) > 0]

        return {
            'symbol': symbol,
            'trades': trades,
            'buy_points': buy_points,
            'sell_points': sell_points,
            'portfolio_values': portfolio_values,
            'summary': {
                'initial_capital': config['initial_capital'],
                'final_value': final_value,
                'profit': round(final_value - config['initial_capital'], 2),
                'profit_rate': round((final_value - config['initial_capital']) / config['initial_capital'] * 100, 2),
                'trade_count': len(trades),
                'win_rate': round(len(wins) / max(len(sell_trades), 1) * 100, 1),
                'holding_shares': holding_shares,
                'avg_cost': round(avg_cost, 2) if holding_shares > 0 else 0,
                'positions': [{'symbol': symbol, 'shares': holding_shares, 'avg_cost': avg_cost, 'value': final_stock_value}] if holding_shares > 0 else []
            }
        }

    def run_stock_selection(self, stocks: List[Dict], top_n: int = 10) -> List[Dict]:
        """选股策略"""
        print(f"\n{'='*60}")
        print(f"开始选股 - 从 {len(stocks)} 只股票中选择前 {top_n} 只")
        print(f"{'='*60}")

        results = []
        for i, stock in enumerate(stocks[:50]):  # 最多分析50只
            symbol = stock['symbol']
            print(f"[{i+1}/{min(len(stocks), 50)}] 分析 {symbol}...", end=" ")

            df = self.load_stock_data(symbol)
            if df is None or len(df) < 500:
                print("数据不足")
                continue

            try:
                result = self.run_backtest(symbol, df)
                if 'error' not in result:
                    results.append({
                        'symbol': symbol,
                        'profit_rate': result['summary']['profit_rate'],
                        'win_rate': result['summary']['win_rate'],
                        'trade_count': result['summary']['trade_count'],
                        'trades': result['trades'],
                        'summary': result['summary']
                    })
                    print(f"收益率 {result['summary']['profit_rate']:.2f}%")
            except Exception as e:
                print(f"错误: {e}")

        # 按收益率排序
        results.sort(key=lambda x: x['profit_rate'], reverse=True)

        print(f"\n{'='*60}")
        print(f"选股完成，前 {min(top_n, len(results))} 只股票:")
        print(f"{'='*60}")
        for i, r in enumerate(results[:top_n], 1):
            print(f"  #{i} {r['symbol']}: 收益率 {r['profit_rate']:.2f}%, 胜率 {r['win_rate']:.1f}%")

        return results[:top_n]

    def run(self):
        """执行完整策略"""
        print(f"\n{'='*60}")
        print(f"LGBM量化交易系统")
        print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        # 1. 获取股票列表
        stocks = self.get_stock_list()
        print(f"\n股票池: {len(stocks)} 只股票")

        if not stocks:
            print("无可用股票数据，请先运行数据收集")
            return

        # 2. 选股
        selected = self.run_stock_selection(stocks, top_n=5)

        if not selected:
            print("选股失败")
            return

        # 3. 汇总结果
        best = selected[0]
        all_trades = []
        for s in selected:
            all_trades.extend(s.get('trades', []))

        summary = {
            'initial_capital': STRATEGY_CONFIG['initial_capital'],
            'final_value': best['summary']['final_value'],
            'profit': best['summary']['profit'],
            'profit_rate': best['summary']['profit_rate'],
            'trade_count': sum(s['trade_count'] for s in selected),
            'win_rate': sum(s['win_rate'] for s in selected) / len(selected),
            'positions': best['summary'].get('positions', [])
        }

        print(f"\n{'='*60}")
        print(f"策略执行完成")
        print(f"{'='*60}")
        print(f"最佳股票: {best['symbol']}")
        print(f"收益率: {summary['profit_rate']:.2f}%")
        print(f"总盈亏: ¥{summary['profit']:,.0f}")
        print(f"胜率: {summary['win_rate']:.1f}%")

        # 4. 发送邮件通知
        self.notifier.send_trade_alert(
            f"【量化策略】{datetime.now().strftime('%Y-%m-%d')} 策略执行报告",
            all_trades,
            summary
        )

        # 5. 保存结果
        result_path = os.path.join(BASE_DIR, 'logs', f'strategy_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump({
                'selected_stocks': selected,
                'summary': summary,
                'config': STRATEGY_CONFIG,
                'time': datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存: {result_path}")

        return selected


def main():
    runner = StrategyRunner()
    runner.run()


if __name__ == "__main__":
    main()