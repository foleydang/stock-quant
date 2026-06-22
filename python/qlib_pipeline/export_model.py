#!/usr/bin/env python3
"""
训练并导出最佳模型: XGBoost h=3, binary labels, Top-50 IC features
输出到 models/xgb_h3_binary/ 目录
"""

import os, sys, warnings, io, pickle, json, shutil

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

import numpy as np
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord
from qlib.contrib.ops.high_freq import Cut, DayLast, FFillNan, IsNull

from qlib_pipeline.train import (
    _IntradayHandler, build_feature_expressions, MODEL_CONFIGS,
    create_dataset, eval_signal, feature_ic_screening,
    BIN_DIR, FREQ, DAY_LENGTH, START_TIME, TRAIN_END, VAL_END, END_TIME,
)

OUTPUT_DIR = os.path.join(ROOT, 'models', 'xgb_h3_binary')
HORIZON = 3
LABEL_TYPE = 'binary'
MAX_STOCKS = 200
TOP_K_FEATURES = 50


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"{'='*60}")
    print(f" 🚀 训练导出: XGBoost h={HORIZON} | binary | Top-{TOP_K_FEATURES} features")
    print(f"   输出: {OUTPUT_DIR}")
    print(f"{'='*60}")

    qlib.init(provider_uri=BIN_DIR, region=REG_CN, freq=FREQ,
              custom_ops=[Cut, DayLast, FFillNan, IsNull], expression_cache=None)

    # ── Step 1: 特征 IC 筛选 ──
    print(f"\n🔍 特征 IC 筛选...")
    ic_features, ic_all = feature_ic_screening(
        HORIZON, quick=False, top_k=TOP_K_FEATURES, label_type=LABEL_TYPE)

    # ── Step 2: 训练模型 ──
    print(f"\n▶ 训练 XGBoost h={HORIZON}...")
    ds_cfg = create_dataset(HORIZON, max_stocks=MAX_STOCKS, label_type=LABEL_TYPE,
                            ic_features=ic_features)
    model_config = MODEL_CONFIGS['XGBoost']

    with R.start(experiment_name=f"export_xgb_h{HORIZON}"):
        model = init_instance_by_config(model_config)
        dataset = init_instance_by_config(ds_cfg)
        model.fit(dataset)

        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        pred = recorder.load_object("pred.pkl")
        label = dataset.prepare('test', col_set=["label"])

        metrics = eval_signal(pred, label)
        print(f"  IC={metrics['IC']:.4f} RankIC={metrics['RankIC']:.4f}")

    # ── Step 3: 保存模型 ──
    print(f"\n💾 保存模型...")

    # 3a. XGBoost 原生格式 (推荐, 跨平台兼容)
    xgb_native_path = os.path.join(OUTPUT_DIR, 'xgb_model.json')
    model.model.save_model(xgb_native_path)
    print(f"  ✓ XGBoost native: {xgb_native_path}")

    # 3b. Pickle 格式 (备选)
    pkl_path = os.path.join(OUTPUT_DIR, 'model.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ Pickle: {pkl_path}")

    # ── Step 4: 保存特征配置 ──
    features = {
        'horizon': HORIZON,
        'label_type': LABEL_TYPE,
        'day_length': DAY_LENGTH,
        'freq': FREQ,
        'ic_features': list(ic_features),
        'feature_names': dataset.handler._feature_names,
        'feature_fields': dataset.handler._feature_fields,
        'columns': ['$open', '$high', '$low', '$close'],
    }
    with open(os.path.join(OUTPUT_DIR, 'feature_config.json'), 'w') as f:
        json.dump(features, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 特征配置: feature_config.json ({len(features['feature_names'])} features)")

    # ── Step 5: 保存元信息 ──
    meta = {
        'model': 'XGBoost',
        'horizon': HORIZON,
        'label_type': LABEL_TYPE,
        'max_stocks': MAX_STOCKS,
        'top_k_features': TOP_K_FEATURES,
        'IC': round(metrics['IC'], 4),
        'RankIC': round(metrics['RankIC'], 4),
        'train_samples': metrics.get('n_samples', 0),
        'timestamp': __import__('datetime').datetime.now().isoformat(),
    }
    with open(os.path.join(OUTPUT_DIR, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ 元信息: meta.json")

    # ── Step 6: 保存推理脚本 ──
    infer_script = '''#!/usr/bin/env python3
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

    print(f"\\n📊 预测结果 (最新截面):")
    # 取最新时间点的预测
    latest_time = pred.index.get_level_values('datetime').max()
    latest_pred = pred.xs(latest_time, level='datetime')
    top20 = latest_pred.sort_values(ascending=False).head(20)
    print(f"  时间: {latest_time}")
    print(f"  Top-20 买入信号:")
    for inst, score in top20.items():
        print(f"    {inst}: {score:.4f}")
'''
    with open(os.path.join(OUTPUT_DIR, 'predict.py'), 'w') as f:
        f.write(infer_script)
    print(f"  ✓ 推理脚本: predict.py")

    # ── 回测验证 ──
    print(f"\n─── 回测验证 ───")
    from qlib_pipeline.backtest import run_backtest
    bt_result = run_backtest(
        horizon=HORIZON, top_k=30, model_name='XGBoost',
        max_stocks=MAX_STOCKS, label_type=LABEL_TYPE,
    )

    print(f"\n{'='*60}")
    print(f" ✅ 模型导出完成!")
    print(f"   输出目录: {OUTPUT_DIR}")
    print(f"   文件列表:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"     {f} ({size:,} bytes)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()