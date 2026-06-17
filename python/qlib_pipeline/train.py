#!/usr/bin/env python3
"""
Qlib 分钟级择时训练 — HighFreqGeneralHandler 专用

特征: 分钟级归一化价格 + 成交量 (适配 day_length=8, 30min K线)
模型: LGBM/GRU/LSTM/Transformer/TabNet

用法:
  python qlib_pipeline/train.py                    # 默认 LGBM
  python qlib_pipeline/train.py --model GRU         # 换 GRU
  python qlib_pipeline/train.py --model LSTM        # 换 LSTM
  python qlib_pipeline/train.py --quick             # 快速验证(沪深300)
"""

import os, sys, argparse, time, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

import qlib
from qlib.constant import REG_CN
from qlib.config import C
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.contrib.data.highfreq_handler import HighFreqGeneralHandler
from qlib.contrib.ops.high_freq import Cut
from qlib.data.dataset import DatasetH

# ============ 配置 ============
BIN_DIR = os.path.expanduser('~/.qlib/qlib_data/cn_30min/bin')
FREQ = '30min'
DAY_LENGTH = 8  # 30分钟K线, 每天4小时 = 8根
EXPERIMENT_NAME = 'intraday_30min_hf'
MODEL_DIR = os.path.join(ROOT, 'models', 'qlib_intraday')

# 时间范围
START_TIME = '2025-06-01 09:30:00'
TRAIN_END = '2026-05-20 15:00:00'
VAL_END = '2026-06-14 15:00:00'
END_TIME = '2026-06-17 15:00:00'

# 特征数 (HighFreqGeneralHandler: 4列×2天 + 2成交量 = 10)
N_FEAT = 10

# 模型配置
MODEL_CONFIGS = {
    'LightGBM': {
        'class': 'LGBModel',
        'module_path': 'qlib.contrib.model.gbdt',
        'kwargs': {
            'loss': 'mse',
            'num_leaves': 127,
            'max_depth': 9,
            'learning_rate': 0.001,
            'n_estimators': 5000,
            'early_stopping_rounds': 200,
            'subsample': 0.6,
            'colsample_bytree': 0.5,
            'reg_alpha': 0.1,
            'reg_lambda': 0.5,
            'min_child_samples': 50,
            'verbosity': 1,
            'seed': 42,
            'n_jobs': 4,
        }
    },
    'GRU': {
        'class': 'GRU',
        'module_path': 'qlib.contrib.model.pytorch_gru',
        'kwargs': {
            'd_feat': N_FEAT,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'n_epochs': 100,
            'lr': 0.001,
            'early_stop': 20,
            'batch_size': 2048,
            'GPU': 0,
            'seed': 42,
        }
    },
    'LSTM': {
        'class': 'LSTM',
        'module_path': 'qlib.contrib.model.pytorch_lstm',
        'kwargs': {
            'd_feat': N_FEAT,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'n_epochs': 100,
            'lr': 0.001,
            'early_stop': 20,
            'batch_size': 2048,
            'GPU': 0,
            'seed': 42,
        }
    },
    'Transformer': {
        'class': 'Transformer',
        'module_path': 'qlib.contrib.model.pytorch_transformer',
        'kwargs': {
            'd_feat': N_FEAT,
            'd_model': 128,
            'n_head': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'n_epochs': 100,
            'lr': 0.0001,
            'early_stop': 20,
            'batch_size': 2048,
            'GPU': 0,
            'seed': 42,
        }
    },
    'TabNet': {
        'class': 'TabNetModel',
        'module_path': 'qlib.contrib.model.pytorch_tabnet',
        'kwargs': {
            'd_feat': N_FEAT,
            'n_d': 32,
            'n_a': 32,
            'n_steps': 3,
            'gamma': 1.3,
            'n_epochs': 100,
            'lr': 0.001,
            'early_stop': 20,
            'batch_size': 2048,
            'GPU': 0,
            'seed': 42,
        }
    },
}


def get_dataset_config(quick: bool = False):
    """构建 Dataset 配置 (HighFreqGeneralHandler)"""
    handler_kwargs = {
        'start_time': START_TIME,
        'end_time': END_TIME,
        'fit_start_time': START_TIME,
        'fit_end_time': TRAIN_END,
        'instruments': 'all',
        'day_length': DAY_LENGTH,
        'freq': FREQ,
        'columns': ['$open', '$high', '$low', '$close'],  # 无 $vwap 数据
    }

    if quick:
        handler_kwargs['instruments'] = 'csi300'

    return {
        'class': 'DatasetH',
        'module_path': 'qlib.data.dataset',
        'kwargs': {
            'handler': {
                'class': 'HighFreqGeneralHandler',
                'module_path': 'qlib.contrib.data.highfreq_handler',
                'kwargs': handler_kwargs,
            },
            'segments': {
                'train': (START_TIME, TRAIN_END),
                'valid': (TRAIN_END, VAL_END),
                'test': (VAL_END, END_TIME),
            },
        }
    }


def get_port_analysis_config():
    """回测配置 (30min专用)"""
    return {
        'executor': {
            'class': 'SimulatorExecutor',
            'module_path': 'qlib.backtest.executor',
            'kwargs': {
                'time_per_step': FREQ,
                'generate_portfolio_metrics': True,
            },
        },
        'strategy': {
            'class': 'TopkDropoutStrategy',
            'module_path': 'qlib.contrib.strategy.signal_strategy',
            'kwargs': {
                'topk': 50,
                'n_drop': 5,
                'method': 'topk',
            },
        },
        'backtest': {
            'start_time': VAL_END,
            'end_time': END_TIME,
            'account': 1000000,
            'exchange_kwargs': {
                'freq': 'day',  # 回测用日频撮合, 避免 30min→1min 报错
                'limit_threshold': 0.095,
                'deal_price': 'close',
                'open_cost': 0.0005,
                'close_cost': 0.0015,
                'min_cost': 5,
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Qlib 分钟级择时训练')
    parser.add_argument('--model', default='LightGBM',
                        choices=list(MODEL_CONFIGS.keys()),
                        help='模型类型')
    parser.add_argument('--bin-dir', default=BIN_DIR, help='Qlib .bin 数据目录')
    parser.add_argument('--quick', action='store_true', help='快速验证模式')
    parser.add_argument('--no-backtest', action='store_true', help='跳过回测')
    args = parser.parse_args()

    if not os.path.exists(args.bin_dir):
        print(f"❌ 数据目录不存在: {args.bin_dir}")
        print(f"  请先运行: sh scripts/sync_qlib_data.sh")
        sys.exit(1)

    print(f"{'='*60}")
    print(f" Qlib 分钟级择时训练: {args.model} (HighFreqGeneralHandler)")
    print(f" 频率: {FREQ} | day_length: {DAY_LENGTH} | 数据: {args.bin_dir}")
    if args.quick:
        print(f" ⚡ 快速验证模式")
    print(f"{'='*60}")

    # 初始化 Qlib
    qlib.init(
        provider_uri=args.bin_dir,
        region=REG_CN,
        freq=FREQ,
    )

    dataset_config = get_dataset_config(args.quick)
    model_config = MODEL_CONFIGS[args.model]

    print(f"\n📦 模型: {model_config['class']}")
    print(f"📊 特征: HighFreqGeneralHandler (day_length={DAY_LENGTH}, {N_FEAT} features)")

    exp_name = f"{EXPERIMENT_NAME}_{args.model}"
    if args.quick:
        exp_name += '_quick'

    t0 = time.time()

    with R.start(experiment_name=exp_name):
        model = init_instance_by_config(model_config)
        dataset = init_instance_by_config(dataset_config)

        print(f"\n🏋️ 训练模型...")
        model.fit(dataset)
        train_time = time.time() - t0
        print(f"  训练耗时: {train_time:.0f}s")

        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        print(f"\n📊 信号分析...")
        sar = SigAnaRecord(recorder)
        sar.generate()

        if not args.no_backtest:
            print(f"\n📈 回测...")
            port_config = get_port_analysis_config()
            try:
                par = PortAnaRecord(recorder, port_config, FREQ)
                par.generate()
            except Exception as e:
                print(f"  ⚠️ 回测跳过: {e}")

        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f"{args.model.lower()}_{FREQ}.pkl")
        R.save_objects(**{'model': model})
        print(f"\n💾 模型已保存: {model_path}")

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f" ✅ 完成! 总耗时: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()