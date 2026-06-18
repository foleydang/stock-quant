#!/usr/bin/env python3
"""
特征筛选: 按重要性截断, 测试 Top-K 特征效果
"""

import os, sys, time, json, warnings, argparse, io, csv
import numpy as np

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
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord
from qlib.contrib.ops.high_freq import Cut, DayLast, FFillNan, IsNull

BIN_DIR = os.path.expanduser('~/.qlib/qlib_data/cn_30min/bin')
FREQ = '30min'
DAY_LENGTH = 8
START_TIME = '2020-01-02 09:30:00'
TRAIN_END = '2026-04-30 15:00:00'
VAL_END = '2026-05-31 15:00:00'
END_TIME = '2026-06-16 15:00:00'
HORIZON = 5


def load_importance(path):
    """加载特征重要性 CSV"""
    fi = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fi.append((row['feature'], float(row['combined'])))
    fi.sort(key=lambda x: -x[1])
    return fi


def get_top_features(fi, k):
    """获取 Top-K 特征名"""
    return [f[0] for f in fi[:k]]


def create_filtered_handler_class(top_features):
    """动态创建只包含 top_features 的 Handler"""
    from qlib_pipeline.train import IntradayHandler
    
    class FilteredHandler(IntradayHandler):
        def get_feature_config(self):
            fields, names = super().get_feature_config()
            # 只保留 top_features 中的特征
            filtered_fields, filtered_names = [], []
            for f, n in zip(fields, names):
                if n in top_features:
                    filtered_fields.append(f)
                    filtered_names.append(n)
            return filtered_fields, filtered_names
    
    return FilteredHandler


def train_with_k(k, top_features, FilteredHandler):
    """用 Top-K 特征训练"""
    ds_cfg = {
        'class': 'DatasetH', 'module_path': 'qlib.data.dataset',
        'kwargs': {
            'handler': {
                'class': 'FilteredHandler',
                'module_path': 'qlib_pipeline.feature_select',
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
    
    model_cfg = {
        'class': 'LGBModel', 'module_path': 'qlib.contrib.model.gbdt',
        'kwargs': {
            'loss': 'mse', 'num_leaves': 127, 'max_depth': -1,
            'learning_rate': 0.03, 'n_estimators': 5000, 'early_stopping_rounds': 200,
            'subsample': 0.6, 'colsample_bytree': 0.6,
            'reg_alpha': 0.0, 'reg_lambda': 5.0, 'min_child_samples': 50,
            'min_split_gain': 0.001, 'verbosity': -1, 'seed': 42, 'n_jobs': 4,
        }
    }
    
    t0 = time.time()
    with R.start(experiment_name=f"feat_sel_k{k}"):
        model = init_instance_by_config(model_cfg)
        dataset = init_instance_by_config(ds_cfg)
        model.fit(dataset)
        elapsed = time.time() - t0
        
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        sar = SigAnaRecord(recorder)
        sar.generate()
        
        metrics = recorder.list_metrics()
        ic = metrics.get('IC', np.nan)
        rank_ic = metrics.get('Rank IC', np.nan)
        
        return {'k': k, 'IC': ic, 'RankIC': rank_ic, 'train_s': elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--importance', default=None, help='特征重要性 CSV 路径')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    
    # 默认路径
    if args.importance is None:
        args.importance = os.path.join(ROOT, 'experiments', 'feature_importance_ensemble.csv')
        if not os.path.exists(args.importance):
            args.importance = os.path.join(ROOT, 'experiments', 'feature_importance_h5.csv')
    
    if not os.path.exists(args.importance):
        print(f"❌ 特征重要性文件不存在: {args.importance}")
        print("   请先运行: python qlib_pipeline/ensemble.py")
        sys.exit(1)
    
    fi = load_importance(args.importance)
    print(f"📋 加载 {len(fi)} 个特征重要性 | 来源: {args.importance}")
    
    # 注册 FilteredHandler 到模块
    # 需要 hack: 把 FilteredHandler 放到当前模块
    import qlib_pipeline.feature_select as this_module
    
    K_VALUES = [10, 15, 20, 25, 30, 40, 50, len(fi)]
    print(f"\n🔬 特征筛选实验: K={K_VALUES}")
    print(f"   HORIZON={HORIZON}")
    
    qlib.init(provider_uri=BIN_DIR, region=REG_CN, freq=FREQ,
              custom_ops=[Cut, DayLast, FFillNan, IsNull], expression_cache=None)
    
    results = []
    for k in K_VALUES:
        top = get_top_features(fi, k)
        # 动态创建 handler 类
        this_module.FilteredHandler = create_filtered_handler_class(set(top))
        
        print(f"\n▶ Top-{k} 特征 ({k}/{len(fi)})...")
        r = train_with_k(k, top, this_module.FilteredHandler)
        results.append(r)
        print(f"  K={k}: IC={r['IC']:.4f} RankIC={r['RankIC']:.4f} ({r['train_s']:.0f}s)")
    
    # 汇总
    print(f"\n{'='*60}")
    print(f" 📊 特征筛选结果")
    print(f"{'='*60}")
    print(f"{'K':<6} {'特征数':<8} {'IC':>8} {'|IC|':>8} {'RankIC':>8} {'|RkIC|':>8} {'耗时':>6}")
    print(f"{'-'*55}")
    for r in results:
        print(f"{r['k']:<6} {r['k']:<8} {r['IC']:>8.4f} {abs(r['IC']):>8.4f} "
              f"{r['RankIC']:>8.4f} {abs(r['RankIC']):>8.4f} {r['train_s']:>5.0f}s")
    
    best = max(results, key=lambda r: abs(r['RankIC']))
    print(f"\n🏆 最优 K={best['k']} |RankIC|={abs(best['RankIC']):.4f} (全量: |RankIC|={abs(results[-1]['RankIC']):.4f})")


if __name__ == '__main__':
    main()