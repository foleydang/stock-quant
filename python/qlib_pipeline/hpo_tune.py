#!/usr/bin/env python3
"""
超参数调优脚本 — 针对 h=5 的 LightGBM
随机搜索 20 组参数, 记录 IC/RankIC/训练时间
"""

import os, sys, time, json, warnings, argparse, itertools
import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord
from qlib.contrib.ops.high_freq import Cut, DayLast, FFillNan, IsNull

# ============ 配置 ============
BIN_DIR = os.path.expanduser('~/.qlib/qlib_data/cn_30min/bin')
FREQ = '30min'
DAY_LENGTH = 8
START_TIME = '2020-01-02 09:30:00'
TRAIN_END = '2026-04-30 15:00:00'
VAL_END = '2026-05-31 15:00:00'
END_TIME = '2026-06-16 15:00:00'
HORIZON = 5  # 固定 h=5 调参

# 参数搜索空间
PARAM_SPACE = {
    'learning_rate': [0.005, 0.01, 0.02, 0.03, 0.05],
    'num_leaves': [63, 95, 127, 191, 255],
    'max_depth': [6, 8, 10, 12, -1],
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.4, 0.5, 0.6, 0.7],
    'min_child_samples': [20, 50, 100, 200],
    'reg_alpha': [0.0, 0.05, 0.1, 0.5, 1.0],
    'reg_lambda': [0.0, 0.1, 0.5, 1.0, 5.0],
    'min_split_gain': [0.0, 0.001, 0.01],
}


def sample_params():
    """随机采样一组参数"""
    p = {}
    for k, v in PARAM_SPACE.items():
        p[k] = float(np.random.choice(v)) if isinstance(v[0], float) else int(np.random.choice(v))
    return p


def run_one(params, trial_id):
    """跑一组参数, 返回 IC/RankIC"""
    model_config = {
        'class': 'LGBModel',
        'module_path': 'qlib.contrib.model.gbdt',
        'kwargs': {
            'loss': 'mse',
            'n_estimators': 5000,
            'early_stopping_rounds': 200,
            'verbosity': -1,
            'seed': 42,
            'n_jobs': 4,
            **params,
        }
    }
    
    dataset_config = {
        'class': 'DatasetH',
        'module_path': 'qlib.data.dataset',
        'kwargs': {
            'handler': {
                'class': 'IntradayHandler',
                'module_path': 'qlib_pipeline.train',
                'kwargs': {
                    'start_time': START_TIME, 'end_time': END_TIME,
                    'fit_start_time': START_TIME, 'fit_end_time': TRAIN_END,
                    'instruments': 'all', 'day_length': DAY_LENGTH, 'freq': FREQ,
                    'columns': ['$open', '$high', '$low', '$close'],
                    'horizon': HORIZON, 'quiet': True,
                },
            },
            'segments': {
                'train': (START_TIME, TRAIN_END),
                'valid': (TRAIN_END, VAL_END),
                'test': (VAL_END, END_TIME),
            },
        }
    }
    
    try:
        t0 = time.time()
        with R.start(experiment_name=f"hpo_h5_trial{trial_id}"):
            model = init_instance_by_config(model_config)
            dataset = init_instance_by_config(dataset_config)
            model.fit(dataset)
            train_time = time.time() - t0
            
            recorder = R.get_recorder()
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()
            sar = SigAnaRecord(recorder)
            sar.generate()
            
            # 读取 IC 结果
            ic = recorder.list_metrics().get('IC', np.nan)
            rank_ic = recorder.list_metrics().get('Rank IC', np.nan)
            
            # 默认用绝对值排序 (正向+负向都有用)
            ic_abs = abs(ic) if not np.isnan(ic) else 0
            rank_ic_abs = abs(rank_ic) if not np.isnan(rank_ic) else 0
            
            return {
                'trial': trial_id,
                'IC': ic, 'RankIC': rank_ic,
                'IC_abs': ic_abs, 'RankIC_abs': rank_ic_abs,
                'train_s': round(train_time, 1),
                'params': params,
                'error': None,
            }
    except Exception as e:
        return {
            'trial': trial_id,
            'IC': np.nan, 'RankIC': np.nan,
            'IC_abs': 0, 'RankIC_abs': 0,
            'train_s': 0, 'params': params,
            'error': str(e),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=20, help='搜索组数')
    parser.add_argument('--quick', action='store_true', help='csi300 快速模式')
    args = parser.parse_args()
    
    print(f"🔧 超参数调优: h={HORIZON}, {args.trials} 组随机搜索")
    if args.quick:
        print("⚡ 快速模式 (csi300)")
    
    qlib.init(provider_uri=BIN_DIR, region=REG_CN, freq=FREQ,
              custom_ops=[Cut, DayLast, FFillNan, IsNull], expression_cache=None)
    
    results = []
    t_start = time.time()
    
    for i in range(args.trials):
        params = sample_params()
        print(f"\n--- Trial {i+1}/{args.trials} ---")
        print(f"    lr={params['learning_rate']} leaves={params['num_leaves']} depth={params['max_depth']} "
              f"subsample={params['subsample']} colsample={params['colsample_bytree']} "
              f"min_child={params['min_child_samples']} alpha={params['reg_alpha']} lambda={params['reg_lambda']}")
        
        r = run_one(params, i)
        results.append(r)
        
        print(f"    IC={r['IC']:.4f} RankIC={r['RankIC']:.4f} "
              f"|IC_abs|={r['IC_abs']:.4f} 耗时={r['train_s']}s"
              + (f" ❌ {r['error']}" if r['error'] else ""))
    
    # 排序: 按 RankIC 绝对值
    results.sort(key=lambda x: -x['RankIC_abs'])
    
    print(f"\n{'='*70}")
    print(f" 📊 调参结果 (按 |RankIC| 排序)")
    print(f"{'='*70}")
    print(f"{'排名':<5} {'IC':>8} {'RankIC':>8} {'|IC|':>7} {'|RkIC|':>7} {'耗时':>6} {'lr':>6} {'leaves':>6} {'depth':>5} {'sub':>5} {'cols':>5} {'mc':>5} {'a':>5} {'l':>5}")
    print(f"{'-'*95}")
    
    for i, r in enumerate(results[:15]):
        p = r['params']
        print(f"{i+1:<5} {r['IC']:>8.4f} {r['RankIC']:>8.4f} {r['IC_abs']:>7.4f} {r['RankIC_abs']:>7.4f} "
              f"{r['train_s']:>5.0f}s {p['learning_rate']:>6.4f} {p['num_leaves']:>6} {p['max_depth']:>5} "
              f"{p['subsample']:>.2f} {p['colsample_bytree']:>.2f} {p['min_child_samples']:>5} "
              f"{p['reg_alpha']:>.2f} {p['reg_lambda']:>.2f}")
    
    best = results[0]
    print(f"\n🏆 最优参数: {json.dumps(best['params'], indent=2)}")
    print(f"   IC={best['IC']:.4f} RankIC={best['RankIC']:.4f}")
    
    # 保存结果
    out = os.path.join(ROOT, 'experiments', f'hpo_h5_results.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump([{k: v for k, v in r.items() if k != 'params'} | {'params': r['params']} for r in results], f, indent=2)
    print(f"\n总耗时: {time.time()-t_start:.0f}s | 结果已保存: {out}")


if __name__ == '__main__':
    main()