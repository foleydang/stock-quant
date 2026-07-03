#!/usr/bin/env python3
"""XGBoost h=3 推理脚本 — 在服务器上加载模型进行预测"""

import os, sys, pickle, json, warnings, io
import numpy as np

warnings.filterwarnings('ignore')
_stderr_backup = sys.stderr
sys.stderr = io.StringIO()
try:
    import gym
except Exception:
    pass
finally:
    sys.stderr = _stderr_backup

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.contrib.ops.high_freq import Cut, DayLast, FFillNan, IsNull

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# 加载配置
with open(os.path.join(MODEL_DIR, 'feature_config.json')) as f:
    FEATURE_CONFIG = json.load(f)

with open(os.path.join(MODEL_DIR, 'meta.json')) as f:
    META = json.load(f)


def load_model():
    """加载模型"""
    from qlib_pipeline.train import _IntradayHandler, MODEL_CONFIGS

    # 初始化 qlib
    BIN_DIR = os.path.expanduser('~/.qlib/qlib_data/cn_30min/bin')
    if not os.path.exists(BIN_DIR):
        raise FileNotFoundError(f"❌ 数据目录不存在: {BIN_DIR}")

    qlib.init(provider_uri=BIN_DIR, region=REG_CN, freq=FEATURE_CONFIG['freq'],
              custom_ops=[Cut, DayLast, FFillNan, IsNull], expression_cache=None)

    # 加载模型
    pkl_path = os.path.join(MODEL_DIR, 'model.pkl')
    with open(pkl_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ 模型已加载: {META['model']} h={META['horizon']} "
          f"(IC={META['IC']}, RankIC={META['RankIC']})")
    return model


def predict(model, instruments=None):
    """批量预测"""
    from qlib_pipeline.train import create_dataset

    ds_cfg = create_dataset(
        FEATURE_CONFIG['horizon'],
        max_stocks=0,  # all stocks
        label_type=FEATURE_CONFIG['label_type'],
        ic_features=set(FEATURE_CONFIG['ic_features']),
    )

    # 更新时间范围到最新
    import datetime
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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

    return pred


if __name__ == '__main__':
    import pandas as pd
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.width', 120)

    model = load_model()
    pred = predict(model)

    print(f"\n📊 预测结果 (最新截面):")
    # 取最新时间点的预测
    latest_time = pred.index.get_level_values('datetime').max()
    latest_pred = pred.xs(latest_time, level='datetime')
    top20 = latest_pred.sort_values(ascending=False).head(20)
    print(f"  时间: {latest_time}")
    print(f"  Top-20 买入信号:")
    for inst, score in top20.items():
        print(f"    {inst}: {score:.4f}")
