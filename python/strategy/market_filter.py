"""大盘过滤器: 基于 HS300 指数判断是否允许买入"""

import sqlite3
import numpy as np
import pandas as pd


class MarketFilter:
    def __init__(self, db_path):
        conn = sqlite3.connect(db_path)
        df = pd.read_sql(
            "SELECT CAST(trade_date AS TEXT) as td, close FROM hs300_daily ORDER BY td", conn)
        conn.close()

        df['td'] = df['td'].str.replace('.0', '', regex=False)
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close'])

        def _norm(d):
            d = d.strip()
            return d[:10] if '-' in d else f"{d[:4]}-{d[4:6]}-{d[6:8]}"

        df['date'] = df['td'].apply(_norm)
        df = df.drop_duplicates(subset='date', keep='first').reset_index(drop=True)

        close = df['close'].values
        ma20 = pd.Series(close).rolling(20).mean().values
        ma60 = pd.Series(close).rolling(60).mean().values
        ma200 = pd.Series(close).rolling(200).mean().values

        self._state = {}
        for i in range(len(df)):
            d = df['date'].iloc[i]
            above_ma20 = not np.isnan(ma20[i]) and close[i] > ma20[i]
            above_ma60 = not np.isnan(ma60[i]) and close[i] > ma60[i]
            above_ma200 = not np.isnan(ma200[i]) and close[i] > ma200[i]
            score = int(above_ma20) + int(above_ma60) + int(above_ma200)

            if score >= 2:
                label = 'bull'
            elif score == 0:
                label = 'bear'
            else:
                label = 'sideways'

            self._state[d] = {
                'close': close[i],
                'ma20': ma20[i],
                'ma60': ma60[i],
                'above_ma20': above_ma20,
                'market_regime': score,
                'regime_label': label,
            }

    def get_state(self, date_str):
        d = pd.Timestamp(date_str).strftime('%Y-%m-%d')
        return self._state.get(d)

    def should_allow_buy(self, date_str):
        state = self.get_state(date_str)
        if state is None:
            return False
        return state['above_ma20']

    def get_position_ratio(self, date_str):
        """仓位建议: regime 0→0%, 1→30%, 2→70%, 3→100%"""
        state = self.get_state(date_str)
        if state is None:
            return 0.0
        return {0: 0.0, 1: 0.3, 2: 0.7, 3: 1.0}[state['market_regime']]
