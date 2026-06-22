#!/usr/bin/env python3
"""
全市场训练脚本 — 周频截面排名 + 日内推理
- 全市场 372 只股票, 30分钟K线
- 预测 1 天 (8 bar) 后收益率, 截面排名
- LightGBM (快速 + 小模型 + CPU友好)
- 导出模型到 models/weekly_ranking/

用法:
  python qlib_pipeline/train_full.py                  # 默认 horizon=8 (1天)
  python qlib_pipeline/train_full.py --horizon 24     # 3天
  python qlib_pipeline/train_full.py --horizon 40     # 5天 (一周)
"""

import os, sys, warnings, io, pickle, json, argparse, time
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
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import QlibDataLoader

from qlib_pipeline.train import (
    _IntradayHandler, build_feature_expressions, MODEL_CONFIGS,
    create_dataset, eval_signal,
    BIN_DIR, FREQ, DAY_LENGTH, START_TIME, TRAIN_END, VAL_END, END_TIME,
)

OUTPUT_DIR = os.path.join(ROOT, 'models', 'weekly_ranking')
HORIZON = 8  # 1 天 = 8 bar (30min), 可改为 24 (3天) 或 40 (5天)
MAX_STOCKS = 0  # 0 = 全市场
LABEL_TYPE = 'binary'  # 涨跌二分类 (比 cs_rank 更稳定, IC~0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=int, default=HORIZON)
    parser.add_argument('--output', default=OUTPUT_DIR)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"{'='*60}")
    print(f" 全市场训练")
    print(f"   horizon={args.horizon} ({args.horizon*30}min = {args.horizon/8:.1f}天)")
    print(f"   股票: 全市场 (372只)")
    print(f"   标签: {LABEL_TYPE}")
    print(f"   输出: {args.output}")
    print(f"{'='*60}")

    qlib.init(provider_uri=BIN_DIR, region=REG_CN, freq=FREQ,
              custom_ops=[Cut, DayLast, FFillNan, IsNull], expression_cache=None)

    # ── 训练 LightGBM ──
    print(f"\n▶ 训练 LightGBM (全市场)...")
    t0 = time.time()

    ds_cfg = create_dataset(args.horizon, max_stocks=MAX_STOCKS, label_type=LABEL_TYPE)
    model_config = MODEL_CONFIGS['LightGBM']

    # 调整参数: 全市场数据更多, 使用 binary 分类 loss
    model_config = dict(model_config)
    model_config['kwargs'] = dict(model_config['kwargs'])
    model_config['kwargs']['loss'] = 'binary'

    with R.start(experiment_name=f"full_ranking_h{args.horizon}"):
        model = init_instance_by_config(model_config)
        dataset = init_instance_by_config(ds_cfg)

        if args.debug:
            print(f"  特征数: {len(dataset.handler._feature_names)}")
            print(f"  训练样本: {dataset.prepare('train', col_set=['feature']).shape}")

        model.fit(dataset)
        elapsed = time.time() - t0

        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        pred = recorder.load_object("pred.pkl")
        label = dataset.prepare('test', col_set=["label"])
        metrics = eval_signal(pred, label)

        print(f"  训练完成: {elapsed:.0f}s")
        print(f"  IC={metrics['IC']:.4f}  RankIC={metrics['RankIC']:.4f}")

        # 特征重要性
        importance = model.get_feature_importance()
        feat_names = dataset.handler._feature_names
        if len(importance) == len(feat_names):
            top = sorted(zip(feat_names, importance), key=lambda x: -x[1])[:20]
        else:
            top = sorted(enumerate(importance), key=lambda x: -x[1])[:20]
            top = [(feat_names[i] if i < len(feat_names) else f'f{i}', v) for i, v in top]

        print(f"\n  Top 20 特征:")
        for name, imp in top:
            print(f"    {name:<30s} {imp:.4f}")

    # ── 保存模型 ──
    print(f"\n💾 保存模型...")

    # LightGBM native (最快)
    lgb_path = os.path.join(args.output, 'model.txt')
    model.model.save_model(lgb_path)
    size_mb = os.path.getsize(lgb_path) / 1024 / 1024
    print(f"  ✓ LightGBM native: {lgb_path} ({size_mb:.1f} MB)")

    # Pickle
    pkl_path = os.path.join(args.output, 'model.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ Pickle: {pkl_path}")

    # 特征配置
    feature_config = {
        'horizon': args.horizon,
        'label_type': LABEL_TYPE,
        'day_length': DAY_LENGTH,
        'freq': FREQ,
        'feature_names': list(dataset.handler._feature_names),
        'feature_fields': list(dataset.handler._feature_fields),
        'columns': ['$open', '$high', '$low', '$close'],
    }
    with open(os.path.join(args.output, 'feature_config.json'), 'w') as f:
        json.dump(feature_config, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 特征配置: {len(feature_config['feature_names'])} features")

    # 元信息
    meta = {
        'model': 'LightGBM',
        'horizon': args.horizon,
        'label_type': LABEL_TYPE,
        'universe': 'all (372 stocks)',
        'IC': round(metrics['IC'], 4),
        'RankIC': round(metrics['RankIC'], 4),
        'train_time_s': int(elapsed),
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'top_features': [(name, float(imp)) for name, imp in top[:10]],
    }
    with open(os.path.join(args.output, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ 元信息: meta.json")

    print(f"\n{'='*60}")
    print(f" ✅ 训练完成!")
    print(f"   IC={metrics['IC']:.4f}  RankIC={metrics['RankIC']:.4f}")
    print(f"   模型: {args.output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()