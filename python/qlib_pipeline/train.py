#!/usr/bin/env python3
"""
Qlib 分钟级择时训练 — 一步到位替代手写 pipeline

对比手写版本:
  - 特征工程: Alpha158 因子库 (158因子) 替代手写 459个
  - 回测: 内置 TopkDropoutStrategy 替代手写回测
  - 模型切换: 一行配置改模型

用法:
  python qlib_pipeline/train.py                          # 默认 LGBM
  python qlib_pipeline/train.py --model GRU               # 换 GRU
  python qlib_pipeline/train.py --model Transformer       # 换 Transformer
  python qlib_pipeline/train.py --model LightGBM --quick  # 快速验证
"""

import os, sys, argparse, time, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH

# ============ 配置 ============
BIN_DIR = os.path.expanduser('~/.qlib/qlib_data/cn_30min/bin')
FREQ = '30min'
EXPERIMENT_NAME = 'intraday_30min_q'
MODEL_DIR = os.path.join(ROOT, 'models', 'qlib_intraday')

# 时间范围
START_TIME = '2025-06-01 09:30:00'
TRAIN_END = '2026-05-20 15:00:00'
VAL_END = '2026-06-14 15:00:00'
END_TIME = '2026-06-17 15:00:00'

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
            'd_feat': 158,
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
            'd_feat': 158,
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
            'd_feat': 158,
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
            'd_feat': 158,
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


def get_dataset_config(model_name: str, quick: bool = False):
    """构建 Dataset 配置"""
    handler_kwargs = {
        'start_time': START_TIME,
        'end_time': END_TIME,
        'fit_start_time': START_TIME,
        'fit_end_time': TRAIN_END,
        'instruments': 'all',
    }

    if quick:
        # 快速模式: 只选成交量前50的股票
        handler_kwargs['instruments'] = 'csi300'

    return {
        'class': 'DatasetH',
        'module_path': 'qlib.data.dataset',
        'kwargs': {
            'handler': {
                'class': 'Alpha158',
                'module_path': 'qlib.contrib.data.handler',
                'kwargs': handler_kwargs,
            },
            'segments': {
                'train': (START_TIME, TRAIN_END),
                'valid': (TRAIN_END, VAL_END),
                'test': (VAL_END, END_TIME),
            },
        }
    }


def get_port_analysis_config(model_name: str):
    """回测配置"""
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
                'freq': FREQ,
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
    parser.add_argument('--freq', default=FREQ, help='K线频率')
    parser.add_argument('--quick', action='store_true', help='快速验证模式')
    parser.add_argument('--no-backtest', action='store_true', help='跳过回测')
    args = parser.parse_args()

    if not os.path.exists(args.bin_dir):
        print(f"❌ 数据目录不存在: {args.bin_dir}")
        print(f"  请先运行: python qlib_pipeline/convert_data.py")
        sys.exit(1)

    print(f"{'='*60}")
    print(f" Qlib 分钟级择时训练: {args.model}")
    print(f" 频率: {args.freq} | 数据: {args.bin_dir}")
    if args.quick:
        print(f" ⚡ 快速验证模式")
    print(f"{'='*60}")

    # 初始化 Qlib
    qlib.init(
        provider_uri=args.bin_dir,
        region=REG_CN,
        freq=args.freq,
    )

    # 数据集配置
    dataset_config = get_dataset_config(args.model, args.quick)
    model_config = MODEL_CONFIGS[args.model]

    print(f"\n📦 模型: {model_config['class']}")
    print(f"  {json.dumps(model_config['kwargs'], indent=2)[:200]}...")

    # 开始实验
    exp_name = f"{EXPERIMENT_NAME}_{args.model}"
    if args.quick:
        exp_name += '_quick'

    t0 = time.time()

    with R.start(experiment_name=exp_name):
        model = init_instance_by_config(model_config)
        dataset = init_instance_by_config(dataset_config)

        # 训练
        print(f"\n🏋️ 训练模型...")
        model.fit(dataset)
        train_time = time.time() - t0
        print(f"  训练耗时: {train_time:.0f}s")

        # 预测
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # 信号分析
        print(f"\n📊 信号分析...")
        sar = SigAnaRecord(recorder)
        sar.generate()

        # 回测
        if not args.no_backtest:
            print(f"\n📈 回测...")
            port_config = get_port_analysis_config(args.model)
            try:
                par = PortAnaRecord(recorder, port_config, args.freq)
                par.generate()
            except Exception as e:
                print(f"  ⚠️ 回测跳过: {e}")

        # 保存模型
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f"{args.model.lower()}_{args.freq}.pkl")
        R.save_objects(**{'model': model})
        print(f"\n💾 模型已保存: {model_path}")

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f" ✅ 完成! 总耗时: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()