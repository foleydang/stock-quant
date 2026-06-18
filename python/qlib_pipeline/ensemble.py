#!/usr/bin/env python3
"""
多模型集成训练 + 特征筛选
- LGBM + XGBoost 平均
- 对比单模型 vs 集成效果
"""

import os, sys, time, json, warnings, argparse, io
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

MODEL_CONFIGS = {
    'LightGBM': {
        'class': 'LGBModel', 'module_path': 'qlib.contrib.model.gbdt',
        'kwargs': {
            'loss': 'mse', 'num_leaves': 127, 'max_depth': -1,
            'learning_rate': 0.03, 'n_estimators': 5000, 'early_stopping_rounds': 200,
            'subsample': 0.6, 'colsample_bytree': 0.6,
            'reg_alpha': 0.0, 'reg_lambda': 5.0, 'min_child_samples': 50,
            'min_split_gain': 0.001, 'verbosity': -1, 'seed': 42, 'n_jobs': 4,
        }
    },
    'XGBoost': {
        'class': 'XGBModel', 'module_path': 'qlib.contrib.model.xgboost',
        'kwargs': {
            'objective': 'reg:squarederror', 'max_depth': 8,
            'learning_rate': 0.03, 'n_estimators': 5000, 'early_stopping_rounds': 200,
            'subsample': 0.7, 'colsample_bytree': 0.6,
            'reg_alpha': 0.0, 'reg_lambda': 5.0, 'min_child_weight': 5,
            'verbosity': 0, 'seed': 42, 'n_jobs': 4,
        }
    },
}


def create_dataset():
    return {
        'class': 'DatasetH', 'module_path': 'qlib.data.dataset',
        'kwargs': {
            'handler': {
                'class': 'IntradayHandler', 'module_path': 'qlib_pipeline.train',
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


def eval_signal(pred, label):
    """计算 IC/RankIC"""
    pred = pred.values.flatten()
    label = label.values.flatten()
    mask = ~(np.isnan(pred) | np.isnan(label))
    pred, label = pred[mask], label[mask]
    if len(pred) < 10:
        return {'IC': np.nan, 'RankIC': np.nan}
    ic = np.corrcoef(pred, label)[0, 1]
    from scipy.stats import spearmanr
    rank_ic = spearmanr(pred, label)[0]
    return {'IC': float(ic), 'RankIC': float(rank_ic)}


def train_and_predict(name, model_config, dataset_config):
    """训练单个模型, 返回预测和标签"""
    t0 = time.time()
    with R.start(experiment_name=f"ensemble_{name}_h{HORIZON}"):
        model = init_instance_by_config(model_config)
        dataset = init_instance_by_config(dataset_config)
        model.fit(dataset)
        
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        
        pred = recorder.load_object("pred.pkl")
        # 获取测试集标签
        label = dataset.prepare('test', col_set=["label"])
        
        elapsed = time.time() - t0
        
        # 特征重要性
        fi = {}
        if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
            fnames = dataset.handler.get_feature_config()[1]
            fi = {n: float(v) for n, v in zip(fnames, model.model.feature_importances_)}
    
    return pred, label, elapsed, fi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--top-k', type=int, default=0, help='只用Top-K特征 (0=全部)')
    args = parser.parse_args()
    
    print(f"🤝 多模型集成: LGBM + XGBoost | h={HORIZON}")
    print(f"   特征筛选: {'Top-' + str(args.top_k) if args.top_k > 0 else '全部'}")
    
    qlib.init(provider_uri=BIN_DIR, region=REG_CN, freq=FREQ,
              custom_ops=[Cut, DayLast, FFillNan, IsNull], expression_cache=None)
    
    results = {}
    predictions = {}
    labels = None
    
    for name in ['LightGBM', 'XGBoost']:
        print(f"\n▶ 训练 {name}...")
        ds_cfg = create_dataset()
        pred, label, elapsed, fi = train_and_predict(name, MODEL_CONFIGS[name], ds_cfg)
        
        if labels is None:
            labels = label
        
        metrics = eval_signal(pred, label)
        predictions[name] = pred
        results[name] = {**metrics, 'train_s': elapsed, 'importance': fi}
        print(f"  {name}: IC={metrics['IC']:.4f} RankIC={metrics['RankIC']:.4f} ({elapsed:.0f}s)")
    
    # ── 集成: 平均预测 ──
    print(f"\n─── 集成: LGBM + XGBoost 平均 ───")
    ensemble_pred = (predictions['LightGBM'] + predictions['XGBoost']) / 2
    ensemble_metrics = eval_signal(ensemble_pred, labels)
    results['Ensemble'] = {**ensemble_metrics, 'train_s': results['LightGBM']['train_s'] + results['XGBoost']['train_s']}
    print(f"  Ensemble: IC={ensemble_metrics['IC']:.4f} RankIC={ensemble_metrics['RankIC']:.4f}")
    
    # ── 特征重要性合并 ──
    if results['LightGBM'].get('importance') and results['XGBoost'].get('importance'):
        lgb_fi = results['LightGBM']['importance']
        xgb_fi = results['XGBoost']['importance']
        all_features = set(lgb_fi.keys()) | set(xgb_fi.keys())
        # 归一化后取平均
        lgb_max = max(lgb_fi.values()) if lgb_fi else 1
        xgb_max = max(xgb_fi.values()) if xgb_fi else 1
        combined = {}
        for f in all_features:
            lgb_norm = lgb_fi.get(f, 0) / lgb_max
            xgb_norm = xgb_fi.get(f, 0) / xgb_max
            combined[f] = (lgb_norm + xgb_norm) / 2
        ranked = sorted(combined.items(), key=lambda x: -x[1])
        
        print(f"\n📊 合并特征重要性 Top 20:")
        for i, (name, imp) in enumerate(ranked[:20]):
            bar = '█' * int(imp * 20)
            print(f"  {i+1:>2}. {name:<25s} {imp:.4f} {bar}")
        
        # 保存
        out_dir = os.path.join(ROOT, 'experiments')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'feature_importance_ensemble.csv'), 'w') as f:
            f.write("feature,lgb_importance,xgb_importance,combined\n")
            for name, imp in ranked:
                f.write(f"{name},{lgb_fi.get(name,0):.6f},{xgb_fi.get(name,0):.6f},{imp:.6f}\n")
        print(f"  已保存: experiments/feature_importance_ensemble.csv")
    
    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(f" 📊 结果汇总")
    print(f"{'='*60}")
    print(f"{'模型':<15} {'IC':>8} {'RankIC':>8} {'耗时':>6}")
    print(f"{'-'*40}")
    for name in ['LightGBM', 'XGBoost', 'Ensemble']:
        r = results[name]
        print(f"{name:<15} {r['IC']:>8.4f} {r['RankIC']:>8.4f} {r['train_s']:>5.0f}s")
    
    best_name = max(results, key=lambda n: abs(results[n]['RankIC']))
    best_ric = abs(results[best_name]['RankIC'])
    print(f"\n🏆 最优: {best_name} (|RankIC|={best_ric:.4f})")


if __name__ == '__main__':
    main()