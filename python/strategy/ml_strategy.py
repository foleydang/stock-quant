#!/usr/bin/env python3
"""
ML 日线策略 v1 — 基于 Optuna LightGBM 的智能选股+交易

核心思路:
  ML 模型输出 cs_rank 预测 [0,1]，越高代表截面排名越靠前
  - 排名只是选股信号，不是交易信号
  - 交易信号 = ML排名 + 持仓盈亏 + 持仓时间 + 风控

策略规则:
  🟢 BUY:   ML排名前 buy_pct  AND 不持仓 AND 有仓位空位
  🔴 SELL:  ML排名跌出 sell_pct  AND 持仓>min_hold_days
            AND (浮亏>stop_loss OR 浮盈>take_profit)
  🔵 ADD:   ML排名前 add_pct  AND 浮亏>add_loss AND 现金充足
  ⚪ HOLD:  其他情况（不动就是最好的操作）

用法:
  from strategy.ml_strategy import MLStrategy
  strategy = MLStrategy()
  signals = strategy.generate_signals()
"""

import os, sys, json, sqlite3, pickle, warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np, pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from config_loader import get_db_path, get_available_cash, get_watchlist

DB_PATH = get_db_path()
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'lgb_daily')
SIGNAL_FILE = os.path.join(PROJECT_ROOT, 'data', 'daily_ml_signals.json')


# ═══════════════════════════════════════════════
# 策略参数 — 可调
# ═══════════════════════════════════════════════
class StrategyConfig:
    # 选股
    MAX_POSITIONS = 5           # 最大持仓数
    BUY_PCT = 0.10              # 买入：ML排名前 10%
    SELL_PCT = 0.70             # 卖出：ML排名掉出前 70%（即后30%）
    ADD_PCT = 0.40              # 补仓：ML排名仍在 40% 以内

    # 交易限制
    MIN_HOLD_DAYS = 10          # 最短持有天数（避免频繁交易）
    MAX_DAILY_TURNOVER = 2      # 每日最多交易 2 只（控制换手率）

    # 风控
    STOP_LOSS = -0.08           # 止损线：浮亏 8%
    TAKE_PROFIT = 0.25          # 止盈线：浮盈 25%
    ADD_LOSS = -0.15            # 补仓触发：浮亏 15% 以上

    # 仓位
    POSITION_SIZE_PCT = 0.20    # 单只股票仓位占比（等权 20% = 5只）
    AVAILABLE_CASH = get_available_cash()

    # 模型
    MODEL_DIR = MODEL_DIR


# ═══════════════════════════════════════════════
# ML 策略引擎
# ═══════════════════════════════════════════════
class MLStrategy:
    def __init__(self, config=None):
        self.config = config or StrategyConfig()
        self.model = None
        self.feature_names = None
        self.meta = None
        self._load_model()

    def _load_model(self):
        """加载 LightGBM 模型"""
        with open(os.path.join(self.config.MODEL_DIR, 'meta.json')) as f:
            self.meta = json.load(f)
        with open(os.path.join(self.config.MODEL_DIR, 'feature_names.json')) as f:
            fn = json.load(f)
        self.feature_names = fn['features']
        self.model = lgb.Booster(
            model_file=os.path.join(self.config.MODEL_DIR, 'model.txt')
        )

    def _load_aux(self, conn: sqlite3.Connection) -> dict:
        """加载辅助数据"""
        aux = {}
        # 基本面
        try:
            fund = pd.read_sql(
                "SELECT symbol, trade_date, roe, net_profit_yoy, debt_ratio, revenue_yoy "
                "FROM fundamental_daily", conn)
            fund['trade_date'] = pd.to_datetime(fund['trade_date'])
            aux['fund'] = fund.set_index(['symbol', 'trade_date']).sort_index()
        except Exception:
            aux['fund'] = None

        # 行业
        try:
            sector = pd.read_sql("SELECT symbol, industry FROM stock_sector", conn)
            aux['sector'] = sector.set_index('symbol')['industry'].to_dict()
        except Exception:
            aux['sector'] = {}

        # 宏观
        try:
            macro = pd.read_sql(
                "SELECT trade_date, hs300_close, shibor_1w, shibor_1m, cn_10y, "
                "cn_us_spread, usdcny, us_10y FROM macro_daily ORDER BY trade_date", conn)
            macro['trade_date'] = pd.to_datetime(macro['trade_date'])
            aux['macro'] = macro.set_index('trade_date')
        except Exception:
            aux['macro'] = None

        # 北向
        try:
            north = pd.read_sql(
                "SELECT trade_date, north_net, total_net FROM north_flow ORDER BY trade_date", conn)
            north['trade_date'] = pd.to_datetime(north['trade_date'])
            aux['north'] = north.set_index('trade_date')
        except Exception:
            aux['north'] = None

        # 情绪（最近30天）
        try:
            sent = pd.read_sql(
                "SELECT symbol, trade_date, is_limit_up, is_limit_down, vol_ratio_20, "
                "lhb_net_buy, margin_balance_chg, abnormal_ret FROM sentiment_daily "
                "WHERE trade_date >= date('now', '-30 days')", conn)
            sent['trade_date'] = pd.to_datetime(sent['trade_date'])
            aux['sent'] = sent.set_index(['symbol', 'trade_date']).sort_index()
        except Exception:
            aux['sent'] = None

        return aux

    def _add_aux(self, row: dict, sym: str, date: str, aux: dict):
        """添加辅助特征到 row dict"""
        ds = pd.Timestamp(str(date)[:10])

        # 基本面
        fund = aux.get('fund')
        if fund is not None and sym in fund.index:
            try:
                fs = fund.loc[sym]; fb = fs[fs.index <= ds]
                if len(fb) > 0:
                    l = fb.iloc[-1]
                    row['fund_roe'] = float(l.get('roe', 0) or 0)
                    row['fund_np_yoy'] = float(l.get('net_profit_yoy', 0) or 0)
                    row['fund_debt'] = float(l.get('debt_ratio', 0) or 0)
                    row['fund_rev_yoy'] = float(l.get('revenue_yoy', 0) or 0)
                else:
                    row['fund_roe'] = row['fund_np_yoy'] = \
                        row['fund_debt'] = row['fund_rev_yoy'] = 0
            except Exception:
                row['fund_roe'] = row['fund_np_yoy'] = \
                    row['fund_debt'] = row['fund_rev_yoy'] = 0
        else:
            row['fund_roe'] = row['fund_np_yoy'] = \
                row['fund_debt'] = row['fund_rev_yoy'] = 0

        row['sector_code'] = float(hash(aux.get('sector', {}).get(sym, '未知')) % 100) / 100

        macro = aux.get('macro')
        if macro is not None and ds in macro.index:
            m = macro.loc[ds]
            idx = macro.index.get_loc(ds)
            prev_close = macro.iloc[idx - 1]['hs300_close'] if idx > 0 else m['hs300_close']
            row['macro_hs300_chg'] = float((m['hs300_close'] - prev_close) / prev_close) if prev_close else 0
            row['macro_shibor_1w'] = float(m.get('shibor_1w', 0) or 0)
            row['macro_shibor_1m'] = float(m.get('shibor_1m', 0) or 0)
            row['macro_cn_10y'] = float(m.get('cn_10y', 0) or 0)
            row['macro_us_10y'] = float(m.get('us_10y', 0) or 0)
            row['macro_cn_us_spread'] = float(m.get('cn_us_spread', 0) or 0)
            row['macro_usdcny'] = float(m.get('usdcny', 0) or 0)
        else:
            for k in ['macro_hs300_chg', 'macro_shibor_1w', 'macro_shibor_1m',
                      'macro_cn_10y', 'macro_us_10y', 'macro_cn_us_spread', 'macro_usdcny']:
                row[k] = 0

        north = aux.get('north')
        if north is not None and ds in north.index:
            row['north_net'] = float(north.loc[ds, 'north_net'] or 0)
            row['north_total_net'] = float(north.loc[ds, 'total_net'] or 0)
        else:
            row['north_net'] = row['north_total_net'] = 0

        sent = aux.get('sent')
        if sent is not None and sym in sent.index:
            try:
                ss = sent.loc[sym]
                if ds in ss.index:
                    s = ss.loc[ds]
                    row['sent_limit_up'] = float(s.get('is_limit_up', 0) or 0)
                    row['sent_limit_down'] = float(s.get('is_limit_down', 0) or 0)
                    row['sent_vol_ratio'] = float(s.get('vol_ratio_20', 0) or 0)
                    row['sent_lhb_net'] = float(s.get('lhb_net_buy', 0) or 0)
                    row['sent_margin_chg'] = float(s.get('margin_balance_chg', 0) or 0)
                    row['sent_abnormal_ret'] = float(s.get('abnormal_ret', 0) or 0)
                    return
            except Exception:
                pass
        for k in ['sent_limit_up', 'sent_limit_down', 'sent_vol_ratio',
                  'sent_lhb_net', 'sent_margin_chg', 'sent_abnormal_ret']:
            row[k] = 0

    def predict_scores(self, conn: sqlite3.Connection) -> pd.DataFrame:
        """对所有股票进行 ML 预测，返回排名 DataFrame"""
        from qlib_pipeline.features_daily import compute_features

        aux = self._load_aux(conn)
        symbols = [r[0] for r in conn.execute(
            "SELECT DISTINCT symbol FROM kline_daily ORDER BY symbol").fetchall()]

        results = []
        for sym in symbols:
            df = pd.read_sql(
                "SELECT date, open, high, low, close, volume FROM kline_daily "
                "WHERE symbol=? ORDER BY date", conn, params=(sym,))
            if len(df) < 120:
                continue

            feats = compute_features(
                df['close'].values, df['high'].values,
                df['low'].values, df['volume'].values)
            if feats is None or len(feats) < 20:
                continue

            self._add_aux(feats, sym, df['date'].iloc[-1], aux)

            X = pd.DataFrame([feats])
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[self.feature_names].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)

            score = float(self.model.predict(X)[0])
            if np.isnan(score) or np.isinf(score):
                continue

            results.append({
                'symbol': sym,
                'score': score,
                'close': float(df['close'].iloc[-1]),
            })

        df = pd.DataFrame(results)
        if len(df) == 0:
            return df
        df['rank'] = df['score'].rank(ascending=False, method='min').astype(int)
        df['rank_pct'] = df['rank'] / len(df)  # 排名百分比，越小越好
        return df.sort_values('score', ascending=False)

    def get_positions(self, conn: sqlite3.Connection) -> Dict[str, dict]:
        """获取当前持仓"""
        positions = {}
        try:
            rows = conn.execute(
                "SELECT symbol, stock_name, shares, cost_price, entry_date "
                "FROM positions").fetchall()
            for r in rows:
                positions[r[0]] = {
                    'name': r[1], 'shares': r[2], 'cost': r[3],
                    'entry_date': r[4] or '',
                }
        except Exception:
            pass
        return positions

    def generate_signals(self, conn: sqlite3.Connection,
                         scores: pd.DataFrame) -> List[dict]:
        """
        生成交易信号

        决策逻辑:
        1. 对每个持仓股票，检查是否需要卖出
        2. 对未持仓的 Top-N 股票，检查是否需要买入
        3. 对持仓但亏损的股票，检查是否需要补仓
        """
        cfg = self.config
        positions = self.get_positions(conn)
        total_stocks = len(scores)
        today = datetime.now().strftime('%Y-%m-%d')

        signals = []
        held = set(positions.keys())
        turnover = 0

        # ── 1. 卖出检查 ──
        for sym, pos in positions.items():
            score_row = scores[scores['symbol'] == sym]
            if len(score_row) == 0:
                continue

            row = score_row.iloc[0]
            pnl = (row['close'] - pos['cost']) / pos['cost'] if pos['cost'] > 0 else 0

            # 计算持仓天数
            hold_days = 999
            if pos['entry_date']:
                try:
                    entry = datetime.strptime(str(pos['entry_date'])[:10], '%Y-%m-%d')
                    hold_days = (datetime.now() - entry).days
                except Exception:
                    pass

            sell = False
            reason = ''

            # 止损：任何情况都触发
            if pnl <= cfg.STOP_LOSS:
                sell = True
                reason = f'止损 ({pnl:.1%})'
            # 止盈：浮盈达标且排名下滑
            elif pnl >= cfg.TAKE_PROFIT and row['rank_pct'] > cfg.SELL_PCT:
                sell = True
                reason = f'止盈 ({pnl:.1%}) + 排名下滑({row["rank"]}/{total_stocks})'
            # 排名卖出：排名掉出前70% + 持仓超过最短持有期
            elif row['rank_pct'] > cfg.SELL_PCT and hold_days >= cfg.MIN_HOLD_DAYS and pnl < 0:
                sell = True
                reason = f'排名下滑({row["rank"]}/{total_stocks}) + 持仓{hold_days}天 + 浮亏{pnl:.1%}'
            # 极端情况：排名掉到最后10%
            elif row['rank_pct'] > 0.90 and hold_days >= cfg.MIN_HOLD_DAYS:
                sell = True
                reason = f'排名垫底({row["rank"]}/{total_stocks})'

            if sell and (turnover < cfg.MAX_DAILY_TURNOVER or pnl <= cfg.STOP_LOSS):
                # 止损永远不限制换手率
                signals.append({
                    'action': 'SELL', 'symbol': sym,
                    'name': pos['name'], 'price': round(row['close'], 2),
                    'score': round(row['score'], 4),
                    'rank': f"{row['rank']}/{total_stocks}",
                    'pnl': f"{pnl:.1%}", 'hold_days': hold_days,
                    'reason': reason,
                })
                held.discard(sym)
                turnover += 1

        # ── 2. 补仓检查 ──
        for sym, pos in positions.items():
            if sym not in held:
                continue  # 已被卖出
            score_row = scores[scores['symbol'] == sym]
            if len(score_row) == 0:
                continue
            row = score_row.iloc[0]
            pnl = (row['close'] - pos['cost']) / pos['cost'] if pos['cost'] > 0 else 0

            if pnl <= cfg.ADD_LOSS and row['rank_pct'] <= cfg.ADD_PCT:
                add_shares = min(int(pos['shares'] * 0.3 / 100) * 100, 100)  # 最多补 30%
                add_amount = add_shares * row['close']
                if add_amount <= cfg.AVAILABLE_CASH * 0.5 and add_shares >= 100:
                    signals.append({
                        'action': 'ADD', 'symbol': sym,
                        'name': pos['name'], 'price': round(row['close'], 2),
                        'score': round(row['score'], 4),
                        'rank': f"{row['rank']}/{total_stocks}",
                        'pnl': f"{pnl:.1%}",
                        'add_shares': add_shares,
                        'add_amount': round(add_amount, 2),
                        'reason': f'浮亏{pnl:.1%} + ML排名前{row["rank_pct"]:.0%}',
                    })
                    turnover += 1

        # ── 3. 买入检查 ──
        available_slots = cfg.MAX_POSITIONS - len(held)
        if available_slots > 0 and turnover < cfg.MAX_DAILY_TURNOVER:
            for _, row in scores.iterrows():
                if turnover >= cfg.MAX_DAILY_TURNOVER:
                    break
                sym = row['symbol']
                if sym in held:
                    continue
                if row['rank_pct'] > cfg.BUY_PCT:
                    continue

                # 计算仓位
                pos_value = cfg.AVAILABLE_CASH * cfg.POSITION_SIZE_PCT
                shares = int(pos_value / row['close'] / 100) * 100
                if shares < 100:
                    continue
                amount = shares * row['close']
                if amount > cfg.AVAILABLE_CASH * (1 - len(held) * cfg.POSITION_SIZE_PCT):
                    continue

                signals.append({
                    'action': 'BUY', 'symbol': sym,
                    'name': self._get_stock_name(conn, sym),
                    'price': round(row['close'], 2),
                    'score': round(row['score'], 4),
                    'rank': f"{row['rank']}/{total_stocks}",
                    'pnl': '-',
                    'shares': shares,
                    'amount': round(amount, 2),
                    'reason': f'ML排名 Top-{row["rank_pct"]:.0%}',
                })
                held.add(sym)
                turnover += 1

        return signals

    def _get_stock_name(self, conn: sqlite3.Connection, sym: str) -> str:
        try:
            r = conn.execute("SELECT name FROM stock_info WHERE symbol=?", (sym,)).fetchone()
            return r[0] if r else sym
        except Exception:
            return sym

    def run(self, top_k: int = 5) -> dict:
        """完整运行：预测 → 生成信号 → 保存"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        print("📡 ML 策略分析中...", flush=True)
        scores = self.predict_scores(conn)
        signals = self.generate_signals(conn, scores)

        today = datetime.now().strftime('%Y-%m-%d')

        # 打印结果
        positions = self.get_positions(conn)
        held = set(positions.keys())

        print(f"\n{'='*60}")
        print(f"📊 ML 日线策略 ({today})")
        print(f"  模型: {self.meta['model']} | RankIC={self.meta['RankIC']} | "
              f"{self.meta['features']}维")
        print(f"  股票池: {len(scores)} | 持仓: {len(positions)} | "
              f"信号: {len(signals)}")

        if signals:
            print(f"\n📋 交易信号:")
            for s in signals:
                icon = {'BUY': '🟢', 'SELL': '🔴', 'ADD': '🔵'}.get(s['action'], '⚪')
                extra = ''
                if s['action'] == 'BUY':
                    extra = f" | {s['shares']}股 ¥{s['amount']:,.0f}"
                elif s['action'] == 'ADD':
                    extra = f" | +{s['add_shares']}股 ¥{s['add_amount']:,.0f}"
                elif s['action'] == 'SELL':
                    extra = f" | 持仓{s['hold_days']}天 | {s['pnl']}"
                print(f"  {icon} {s['action']:4s} | {s['symbol']:12s} {s['name'][:8]:8s} | "
                      f"@{s['price']:.2f} | 排名:{s['rank']} | {s['reason']}{extra}")
        else:
            print(f"\n  ⚪ 无交易信号（持仓不动）")

        # 打印持仓状态
        if positions:
            print(f"\n📦 持仓状态:")
            for sym, pos in positions.items():
                score_row = scores[scores['symbol'] == sym]
                if len(score_row) > 0:
                    r = score_row.iloc[0]
                    pnl = (r['close'] - pos['cost']) / pos['cost'] if pos['cost'] > 0 else 0
                    hold_days = ''
                    if pos['entry_date']:
                        try:
                            entry = datetime.strptime(str(pos['entry_date'])[:10], '%Y-%m-%d')
                            hold_days = f" | {((datetime.now() - entry).days)}天"
                        except Exception:
                            pass
                    print(f"  {sym:12s} {pos['name'][:8]:8s} | "
                          f"@{r['close']:.2f} | 成本:{pos['cost']:.2f} | "
                          f"{pnl:+.1%} | 排名:{r['rank']}/{len(scores)}{hold_days}")

        print(f"\n🏆 ML Top-{top_k}:")
        for _, row in scores.head(top_k).iterrows():
            held_mark = ' ★' if row['symbol'] in held else ''
            name = self._get_stock_name(conn, row['symbol'])[:8]
            print(f"  {row['rank']:3d}. {row['symbol']:12s} {name:8s} "
                  f"分数:{row['score']:.4f}  价格:{row['close']:.2f}{held_mark}")

        conn.close()

        # 保存
        os.makedirs(os.path.dirname(SIGNAL_FILE), exist_ok=True)
        result = {
            'date': today, 'timestamp': datetime.now().isoformat(),
            'model': self.meta,
            'strategy': {
                'max_positions': self.config.MAX_POSITIONS,
                'buy_pct': self.config.BUY_PCT,
                'sell_pct': self.config.SELL_PCT,
                'stop_loss': self.config.STOP_LOSS,
                'take_profit': self.config.TAKE_PROFIT,
                'min_hold_days': self.config.MIN_HOLD_DAYS,
            },
            'signals': signals,
            'portfolio': [
                {'symbol': sym, 'name': pos['name'], 'shares': pos['shares'],
                 'cost': pos['cost'], 'entry_date': pos['entry_date']}
                for sym, pos in positions.items()
            ],
            'top5': [{'symbol': r['symbol'], 'score': round(r['score'], 4),
                      'close': round(r['close'], 2), 'rank': int(r['rank'])}
                     for _, r in scores.head(5).iterrows()],
        }
        with open(SIGNAL_FILE, 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 信号已保存: {SIGNAL_FILE}")
        return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()

    strategy = MLStrategy()
    strategy.run(top_k=args.top_k)