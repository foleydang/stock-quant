#!/usr/bin/env python3
"""
Intraday inference script — 每 30 分钟运行一次, 输出 Top-5 持仓信号
- 加载训练好的 LightGBM 模型 (models/weekly_ranking/)
- 对全市场 372 只股票进行截面排名预测
- 输出 Top-5 买入信号, JSON 格式

用法:
  python qlib_pipeline/infer.py                          # 默认输出 Top-5
  python qlib_pipeline/infer.py --top-k 10               # Top-10
  python qlib_pipeline/infer.py --model-dir models/weekly_ranking
  python qlib_pipeline/infer.py --json > signals.json    # 重定向到文件

部署 (cron, 每 30 分钟):
  */30 9-15 * * 1-5 cd /path/to/python && python qlib_pipeline/infer.py >> logs/signals.log
"""

import os, sys, warnings, io, json, argparse, pickle
from datetime import datetime

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
sys.path.insert(0, ROOT)

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.contrib.ops.high_freq import Cut, DayLast, FFillNan, IsNull

MODEL_DIR = os.path.join(ROOT, 'models', 'lgb_h3_binary')
BIN_DIR = os.path.expanduser('~/.qlib/qlib_data/cn_30min/bin')
TOP_K = 5


def load_model(model_dir=None):
    """加载模型和配置"""
    if model_dir is None:
        model_dir = MODEL_DIR

    # 加载配置
    with open(os.path.join(model_dir, 'feature_config.json')) as f:
        feature_config = json.load(f)
    with open(os.path.join(model_dir, 'meta.json')) as f:
        meta = json.load(f)

    # 初始化 qlib
    if not os.path.exists(BIN_DIR):
        raise FileNotFoundError(f"数据目录不存在: {BIN_DIR}")

    qlib.init(provider_uri=BIN_DIR, region=REG_CN, freq=feature_config['freq'],
              custom_ops=[Cut, DayLast, FFillNan, IsNull], expression_cache=None)

    # 加载 pickle 模型 (包含完整 qlib wrapper)
    pkl_path = os.path.join(model_dir, 'model.pkl')
    with open(pkl_path, 'rb') as f:
        model = pickle.load(f)

    return model, feature_config, meta


def predict(model, feature_config, instruments=None):
    """对全市场进行截面预测"""
    from qlib_pipeline.train import _IntradayHandler, create_dataset

    ds_cfg = create_dataset(
        feature_config['horizon'],
        max_stocks=0,
        label_type=feature_config['label_type'],
    )

    # 更新时间范围: 从 2020 到当前时间
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ds_cfg['kwargs']['handler']['kwargs']['start_time'] = '2020-01-02 09:30:00'
    ds_cfg['kwargs']['handler']['kwargs']['end_time'] = now
    ds_cfg['kwargs']['handler']['kwargs']['fit_start_time'] = '2020-01-02 09:30:00'
    ds_cfg['kwargs']['handler']['kwargs']['fit_end_time'] = now

    if instruments:
        ds_cfg['kwargs']['handler']['kwargs']['instruments'] = instruments

    dataset = init_instance_by_config(ds_cfg)
    pred = model.predict(dataset)

    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0] if pred.shape[1] == 1 else pred['score']

    # binary 模型输出 raw margin, 需要 sigmoid 转为概率
    if feature_config.get('label_type') == 'binary':
        pred = 1 / (1 + np.exp(-pred))

    return pred


def get_latest_signals(pred, top_k=TOP_K):
    """获取最新时间点的 Top-K 信号"""
    latest_time = pred.index.get_level_values('datetime').max()
    latest_pred = pred.xs(latest_time, level='datetime')

    # 按分数降序排列
    latest_pred = latest_pred.sort_values(ascending=False)
    latest_pred = latest_pred.dropna()

    if len(latest_pred) == 0:
        return None, latest_time, None

    top_k = min(top_k, len(latest_pred))
    top = latest_pred.head(top_k)

    # 截面排名统计
    mean_score = latest_pred.mean()
    std_score = latest_pred.std()

    signals = []
    for i, (inst, score) in enumerate(top.items()):
        z_score = (score - mean_score) / std_score if std_score > 0 else 0
        signals.append({
            'rank': i + 1,
            'stock': inst,
            'score': round(float(score), 4),
            'prob_up': round(float(score), 4),  # binary 模型: P(未来上涨)
            'z_score': round(float(z_score), 2),
            'confidence': 'high' if z_score > 1.5 else ('medium' if z_score > 0.5 else 'low'),
        })

    return signals, latest_time, {
        'n_stocks': len(latest_pred),
        'mean': round(float(mean_score), 4),
        'std': round(float(std_score), 4),
        'min': round(float(latest_pred.min()), 4),
        'max': round(float(latest_pred.max()), 4),
    }


def main():
    parser = argparse.ArgumentParser(description='Intraday inference')
    parser.add_argument('--model-dir', default=MODEL_DIR, help='模型目录')
    parser.add_argument('--top-k', type=int, default=TOP_K, help='输出 Top-K')
    parser.add_argument('--json', action='store_true', default=True, help='JSON 输出')
    parser.add_argument('--text', action='store_true', help='纯文本输出')
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        print(f"❌ 模型目录不存在: {args.model_dir}", file=sys.stderr)
        sys.exit(1)

    model, feature_config, meta = load_model(args.model_dir)

    pred = predict(model, feature_config)
    signals, latest_time, stats = get_latest_signals(pred, args.top_k)

    if signals is None:
        print("❌ 无法生成预测信号", file=sys.stderr)
        sys.exit(1)

    result = {
        'timestamp': str(latest_time),
        'generated_at': datetime.now().isoformat(),
        'model': {
            'name': meta['model'],
            'horizon': meta['horizon'],
            'label_type': meta['label_type'],
            'ic': meta['IC'],
            'rank_ic': meta['RankIC'],
        },
        'stats': stats,
        'signals': signals,
    }

    if args.text:
        print(f"时间: {latest_time}")
        print(f"模型: {meta['model']} h={meta['horizon']} (IC={meta['IC']}, RankIC={meta['RankIC']})")
        print(f"截面: {stats['n_stocks']} stocks, mean={stats['mean']}, std={stats['std']}")
        print(f"\nTop-{args.top_k} 信号:")
        for s in signals:
            print(f"  #{s['rank']} {s['stock']}  prob={s['prob_up']:.2%}  z={s['z_score']}  [{s['confidence']}]")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()