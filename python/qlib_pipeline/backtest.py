#!/usr/bin/env python3
"""
回测脚本: 基于训练好的模型信号进行回测
- Top-K 每日调仓策略
- 计算年化收益、夏普比率、最大回撤等
"""

import os, sys, warnings, io, argparse
import numpy as np
import pandas as pd

np.seterr(all='ignore')
warnings.filterwarnings('ignore')

_stderr_backup = sys.stderr
sys.stderr = io.StringIO()
try:
    import gym
except Exception:
    pass
finally:
    sys.stderr = _stderr_backup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord
from qlib.contrib.ops.high_freq import Cut, DayLast, FFillNan, IsNull
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.backtest import backtest, executor
from qlib.contrib.evaluate import risk_analysis

BIN_DIR = os.path.expanduser('~/.qlib/qlib_data/cn_30min/bin')
FREQ = '30min'
DAY_LENGTH = 8
START_TIME = '2020-01-02 09:30:00'
TRAIN_END = '2026-04-30 15:00:00'
VAL_END = '2026-05-31 15:00:00'
END_TIME = '2026-06-16 15:00:00'
HORIZON = 3

from qlib_pipeline.train import (
    _IntradayHandler, build_feature_expressions, MODEL_CONFIGS,
    eval_signal, create_dataset,
)


def run_backtest(horizon=3, top_k=30, model_name='LightGBM', max_stocks=200,
                 label_type='binary', ic_features=None):
    """训练模型并回测"""
    print(f"\n{'='*60}")
    print(f" 📈 回测: {model_name} | h={horizon} | Top-{top_k}")
    print(f"{'='*60}")

    qlib.init(provider_uri=BIN_DIR, region=REG_CN, freq=FREQ,
              custom_ops=[Cut, DayLast, FFillNan, IsNull], expression_cache=None)

    # 训练模型
    ds_cfg = create_dataset(horizon, max_stocks=max_stocks, label_type=label_type)
    model_config = MODEL_CONFIGS[model_name]

    with R.start(experiment_name=f"backtest_{model_name}_h{horizon}"):
        model = init_instance_by_config(model_config)
        dataset = init_instance_by_config(ds_cfg)
        model.fit(dataset)

        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        pred = recorder.load_object("pred.pkl")
        label = dataset.prepare('test', col_set=["label"])

        # 评估信号
        metrics = eval_signal(pred, label)
        print(f"  信号: IC={metrics['IC']:.4f} RankIC={metrics['RankIC']:.4f}")

    # ── 回测: 用 Top-K 策略 ──
    # 将预测转为每日截面信号
    if isinstance(pred, pd.DataFrame):
        signal = pred.iloc[:, 0] if pred.shape[1] == 1 else pred['score']
    else:
        signal = pred

    # 用 raw 标签获取实际未来收益率
    ds_raw_cfg = create_dataset(horizon, max_stocks=max_stocks, label_type='raw')
    ds_raw = init_instance_by_config(ds_raw_cfg)
    raw_label = ds_raw.prepare('test', col_set=["label"])
    if isinstance(raw_label, pd.DataFrame):
        raw_label = raw_label.iloc[:, 0]

    # 过滤: 只保留交易时段 + 非零标签
    signal = signal.dropna()
    raw_label = raw_label.dropna()
    common_idx = signal.index.intersection(raw_label.index)
    signal = signal.loc[common_idx]
    raw_label = raw_label.loc[common_idx]

    # 只保留交易时段 (9:30-15:00)
    hours = signal.index.get_level_values('datetime').hour
    trading_mask = ((hours >= 9) & (hours <= 14)) | ((hours == 15) & (signal.index.get_level_values('datetime').minute == 0))
    signal = signal[trading_mask]
    raw_label = raw_label[trading_mask]

    # 只保留非零标签
    non_zero = raw_label != 0
    signal = signal[non_zero]
    raw_label = raw_label[non_zero]

    # 按时间分组, 每组选 Top-K
    signal_df = signal.reset_index()
    signal_df.columns = ['instrument', 'datetime', 'score']
    label_df = raw_label.reset_index()
    label_df.columns = ['instrument', 'datetime', 'ret']

    portfolio_returns = []
    for dt, s_grp in signal_df.groupby('datetime'):
        l_grp = label_df[label_df['datetime'] == dt]
        if len(s_grp) < top_k or len(l_grp) < top_k:
            continue

        s_grp = s_grp.sort_values('score', ascending=False)
        top_stocks = s_grp.head(top_k)['instrument'].values
        bottom_stocks = s_grp.tail(top_k)['instrument'].values

        # 等权 Long Top-K, Short Bottom-K
        long_ret = l_grp[l_grp['instrument'].isin(top_stocks)]['ret'].mean()
        short_ret = l_grp[l_grp['instrument'].isin(bottom_stocks)]['ret'].mean()
        portfolio_returns.append((long_ret - short_ret) / 2)

    # ── 计算回测指标 ──
    if not portfolio_returns:
        print("  ⚠️ 没有足够的回测数据")
        return None

    rets = np.array(portfolio_returns)
    rets = rets[~np.isnan(rets)]

    if len(rets) < 5:
        print("  ⚠️ 交易天数不足")
        return None

    # 年化: 每 bar 30min, 每天 8 bar, 252 天/年
    bars_per_year = 252 * 8
    periods_per_year = bars_per_year / horizon  # horizon-bar holding periods

    total_return = (1 + rets).prod() - 1
    n_bars = len(rets)
    annual_return = (1 + total_return) ** (periods_per_year / n_bars) - 1
    annual_vol = rets.std() * np.sqrt(periods_per_year)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    # 最大回撤
    cumulative = (1 + rets).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # 胜率
    win_rate = (rets > 0).mean()
    avg_win = rets[rets > 0].mean() if (rets > 0).any() else 0
    avg_loss = rets[rets < 0].mean() if (rets < 0).any() else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    # Calmar 比率
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    result = {
        'model': model_name,
        'horizon': horizon,
        'top_k': top_k,
        'n_bars': n_bars,
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'calmar': calmar,
        'win_rate': win_rate,
        'avg_ret': rets.mean(),
        'profit_factor': profit_factor,
        'IC': metrics['IC'],
        'RankIC': metrics['RankIC'],
    }

    print(f"\n{'─'*50}")
    print(f" 📊 回测结果")
    print(f"{'─'*50}")
    print(f"  交易次数:    {n_bars}")
    print(f"  累计收益率:  {total_return*100:+.2f}%")
    print(f"  年化收益率:  {annual_return*100:+.2f}%")
    print(f"  年化波动率:  {annual_vol*100:.2f}%")
    print(f"  夏普比率:    {sharpe:.3f}")
    print(f"  最大回撤:    {max_drawdown*100:.2f}%")
    print(f"  Calmar:      {calmar:.3f}")
    print(f"  胜率:        {win_rate*100:.1f}%")
    print(f"  盈亏比:      {profit_factor:.2f}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='LightGBM', choices=['LightGBM', 'XGBoost', 'CatBoost'])
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--top-k', type=int, default=30)
    parser.add_argument('--max-stocks', type=int, default=200)
    parser.add_argument('--label-type', default='binary')
    args = parser.parse_args()

    results = run_backtest(
        horizon=args.horizon, top_k=args.top_k,
        model_name=args.model, max_stocks=args.max_stocks,
        label_type=args.label_type,
    )

    return results


if __name__ == '__main__':
    main()