#!/usr/bin/env python3
"""
qlib 轻量预测器 — 只加载最近60天数据，适合 bot 定时调用
用法: from qlib_light_predictor import predict_top_stocks, format_feishu_card
"""
import os, sys, warnings, io, pickle, json
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
_stderr_backup = sys.stderr
sys.stderr = io.StringIO()
try:
    import gym
except Exception:
    pass
finally:
    sys.stderr = _stderr_backup

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'python'))

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.contrib.ops.high_freq import Cut, DayLast, FFillNan, IsNull

MODEL_DIR = os.path.join(ROOT, 'python', 'models', 'lgb_h3_binary')
BIN_DIR = os.path.expanduser('~/.qlib/qlib_data/cn_30min/bin')
TOP_K = 10

_model = None
_feature_config = None
_meta = None


def _load_model():
    global _model, _feature_config, _meta
    if _model is not None:
        return _model, _feature_config, _meta

    with open(os.path.join(MODEL_DIR, 'feature_config.json')) as f:
        _feature_config = json.load(f)
    with open(os.path.join(MODEL_DIR, 'meta.json')) as f:
        _meta = json.load(f)

    if not os.path.exists(BIN_DIR):
        raise FileNotFoundError(f"qlib 数据目录不存在: {BIN_DIR}")

    qlib.init(provider_uri=BIN_DIR, region=REG_CN, freq=_feature_config['freq'],
              custom_ops=[Cut, DayLast, FFillNan, IsNull], expression_cache=None)

    pkl_path = os.path.join(MODEL_DIR, 'model.pkl')
    with open(pkl_path, 'rb') as f:
        _model = pickle.load(f)

    return _model, _feature_config, _meta


def predict_top_stocks(top_k=TOP_K, use_yesterday=True):
    """预测全市场 Top-K 买入信号
    
    Args:
        top_k: 返回前K只股票
        use_yesterday: True=用昨天收盘数据预测今天（推荐，盘前推送）
                       False=用当前最新数据（盘中预测）
    """
    from qlib_pipeline.train import _IntradayHandler, create_dataset

    model, fc, meta = _load_model()

    ds_cfg = create_dataset(
        fc['horizon'],
        max_stocks=0,
        label_type=fc['label_type'],
    )

    now = datetime.now()
    
    if use_yesterday:
        # 找到最近一个交易日（跳过周末）
        last_trade_day = now
        for _ in range(10):
            last_trade_day = last_trade_day - timedelta(days=1)
            if last_trade_day.weekday() < 5:  # 周一到周五
                break
        end = last_trade_day.strftime('%Y-%m-%d 15:00:00')
    else:
        end = now.strftime('%Y-%m-%d %H:%M:%S')
    
    start = (datetime.strptime(end, '%Y-%m-%d %H:%M:%S') - timedelta(days=10)).strftime('%Y-%m-%d %H:%M:%S')

    ds_cfg['kwargs']['handler']['kwargs']['start_time'] = start
    ds_cfg['kwargs']['handler']['kwargs']['end_time'] = end
    ds_cfg['kwargs']['handler']['kwargs']['fit_start_time'] = start
    ds_cfg['kwargs']['handler']['kwargs']['fit_end_time'] = end
    # 覆盖 segments 中的 test 时间范围（默认是到2026-06-16）
    ds_cfg['kwargs']['segments'] = {
        'train': (start, end),
        'valid': (start, end),
        'test': (start, end),
    }

    dataset = init_instance_by_config(ds_cfg)
    pred = model.predict(dataset)

    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0] if pred.shape[1] == 1 else pred['score']

    # binary → probability
    if fc.get('label_type') == 'binary':
        pred = 1 / (1 + np.exp(-pred))

    # 取最新截面
    latest_time = pred.index.get_level_values('datetime').max()
    latest_pred = pred.xs(latest_time, level='datetime')
    latest_pred = latest_pred.sort_values(ascending=False).dropna()

    top_k = min(top_k, len(latest_pred))
    top = latest_pred.head(top_k)

    mean_score = latest_pred.mean()
    std_score = latest_pred.std()

    results = []
    for rank, (inst, score) in enumerate(top.items()):
        z = (score - mean_score) / std_score if std_score > 0 else 0
        results.append({
            'rank': rank + 1,
            'symbol': inst,
            'score': round(float(score), 4),
            'prob_up': round(float(score), 4),
            'z_score': round(float(z), 2),
            'confidence': 'high' if z > 1.5 else ('medium' if z > 0.5 else 'low'),
        })

    return {
        'timestamp': str(latest_time),
        'model': meta['model'],
        'horizon': meta['horizon'],
        'ic': meta['IC'],
        'rank_ic': meta['RankIC'],
        'n_stocks': len(latest_pred),
        'mean_score': round(float(mean_score), 4),
        'signals': results,
    }


def _get_stock_name(symbol):
    """从数据库查股票名称"""
    import sqlite3
    try:
        conn = sqlite3.connect(os.path.join(ROOT, 'python', 'data', 'stock_data.db'))
        r = conn.execute('SELECT name FROM stock_info WHERE symbol=?', (symbol,)).fetchone()
        conn.close()
        return r[0] if r else symbol
    except Exception:
        return symbol


def format_feishu_card(result):
    """格式化为飞书消息"""
    if not result or not result.get('signals'):
        return "❌ qlib 模型无预测结果"

    lines = [
        f"🧠 qlib 明日选股（基于昨日收盘数据）",
        f"模型: {result['model']} | 周期: {result['horizon']}×30min | IC: {result['ic']:.2%} | RankIC: {result['rank_ic']:.2%}",
        f"数据截止: {result['timestamp']} | 截面: {result['n_stocks']} 只",
        f"",
    ]

    for s in result['signals']:
        conf_emoji = {'high': '🔥', 'medium': '⭐', 'low': '💡'}.get(s['confidence'], '')
        name = _get_stock_name(s['symbol'])
        lines.append(f"  #{s['rank']} {name}（{s['symbol']}） 概率 {s['prob_up']:.2%}  z={s['z_score']} {conf_emoji}")

    return '\n'.join(lines)


if __name__ == '__main__':
    print("qlib 轻量预测器测试...")
    result = predict_top_stocks(10)
    print(format_feishu_card(result))