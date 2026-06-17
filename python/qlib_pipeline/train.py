#!/usr/bin/env python3
"""
Qlib 分钟级择时训练 — HighFreqGeneralHandler + 自定义标签

基于 Qlib 官方高频框架, 适配 30min K线:
  - day_length=8 (每天8根30min K线)
  - 特征: OHLC 归一化价格 + 成交量
  - 标签: 未来 N 根K线收益率

用法:
  python qlib_pipeline/train.py                    # 默认 LGBM
  python qlib_pipeline/train.py --model GRU         # 换 GRU
  python qlib_pipeline/train.py --horizon 3         # 预测3根K线后
"""

import os, sys, argparse, time, json, copy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.contrib.data.highfreq_handler import HighFreqGeneralHandler
from qlib.contrib.ops.high_freq import Cut, DayLast, FFillNan, IsNull
from qlib.data.dataset import DatasetH

# ============ 配置 ============
BIN_DIR = os.path.expanduser('~/.qlib/qlib_data/cn_30min/bin')
FREQ = '30min'
DAY_LENGTH = 8  # 30分钟K线, 每天4小时 = 8根
EXPERIMENT_NAME = 'intraday_30min_hf'
MODEL_DIR = os.path.join(ROOT, 'models', 'qlib_intraday')

START_TIME = '2025-06-01 09:30:00'
TRAIN_END = '2026-05-20 15:00:00'
VAL_END = '2026-06-14 15:00:00'
END_TIME = '2026-06-17 15:00:00'

N_FEAT = 10  # 默认值, 会被实际特征数覆盖


class IntradayHandler(HighFreqGeneralHandler):
    """丰富的分钟级特征 + 标签处理器 (~120+ features)"""

    def __init__(self, horizon=3, **kwargs):
        self.horizon = horizon
        self.day_length = kwargs.pop('day_length', DAY_LENGTH)
        self.columns = kwargs.pop('columns', ['$open', '$high', '$low', '$close'])
        freq = kwargs.pop('freq', FREQ)
        kwargs.pop('fit_start_time', None)
        kwargs.pop('fit_end_time', None)

        from qlib.data.dataset.loader import QlibDataLoader
        from qlib.data.dataset.handler import DataHandlerLP
        feature_fields, feature_names = self.get_feature_config()
        label_fields, label_names = self.get_label_config()
        data_loader = QlibDataLoader(
            config={'feature': (feature_fields, feature_names),
                    'label': (label_fields, label_names)},
            swap_level=False, freq=freq,
        )
        DataHandlerLP.__init__(self, data_loader=data_loader, **kwargs)

    def get_feature_config(self):
        """~50 Qlib 表达式特征: 收益/波动/均线/技术指标/量价"""
        fields, names = HighFreqGeneralHandler.get_feature_config(self)
        EPS = '1e-6'

        def add(expr, name):
            fields.append(expr)
            names.append(name)

        # ── 收益 (多周期) ──
        for p in [1, 2, 3, 5, 10, 15, 20, 30, 40, 60]:
            add(f"$close / Ref($close, {p}) - 1", f"ret_{p}")

        # ── 波动率 ──
        ret1 = "$close / Ref($close, 1) - 1"
        for p in [5, 10, 20, 30, 60]:
            add(f"Std({ret1}, {p})", f"vol_{p}")

        # ── 均线偏离 ──
        for p in [5, 10, 20, 30, 60]:
            add(f"$close / Mean($close, {p}) - 1", f"ma_dist_{p}")

        # ── 均线斜率 ──
        for p in [5, 10, 20, 60]:
            add(f"(Mean($close, {p}) - Ref(Mean($close, {p}), 3)) / (Mean($close, {p}) + {EPS})", f"ma_slope_{p}")

        # ── 价格位置 (Donchian Channel) ──
        for p in [10, 20, 60, 120]:
            add(f"($close - Min($low, {p})) / (Max($high, {p}) - Min($low, {p}) + {EPS})", f"pos_{p}")

        # ── 振幅 ──
        for p in [10, 20, 60]:
            add(f"(Max($high, {p}) - Min($low, {p})) / (Mean($close, {p}) + {EPS})", f"range_{p}")

        # ── 量比 ──
        for p in [5, 10, 20, 60]:
            add(f"$volume / (Mean($volume, {p}) + {EPS}) - 1", f"vol_ratio_{p}")

        # ── 量变化 ──
        for p in [1, 3, 5, 10, 20]:
            add(f"$volume / (Ref($volume, {p}) + {EPS}) - 1", f"vol_chg_{p}")

        # ── 量趋势 ──
        add(f"Mean($volume, 5) / (Mean($volume, 20) + {EPS}) - 1", "vol_trend_5_20")
        add(f"Mean($volume, 10) / (Mean($volume, 60) + {EPS}) - 1", "vol_trend_10_60")

        # ── RSI (用 If 替代 Max 避免 parser 歧义) ──
        for p in [6, 14, 24]:
            up = f"If(Gt($close - Ref($close, 1), 0), $close - Ref($close, 1), 0)"
            down = f"If(Gt(Ref($close, 1) - $close, 0), Ref($close, 1) - $close, 0)"
            add(f"100 - 100 / (1 + Mean({up}, {p}) / (Mean({down}, {p}) + {EPS}))", f"rsi_{p}")

        # ── MACD ──
        add("EMA($close, 12) - EMA($close, 26)", "macd")
        add("EMA(EMA($close, 12) - EMA($close, 26), 9)", "macd_signal")
        add("(EMA($close, 12) - EMA($close, 26)) - EMA(EMA($close, 12) - EMA($close, 26), 9)", "macd_hist")

        # ── 布林带 ──
        for p in [10, 20, 60]:
            add(f"($close - Mean($close, {p}) + 2*Std($close, {p})) / (4*Std($close, {p}) + {EPS})", f"bb_pos_{p}")
            add(f"4*Std($close, {p}) / (Mean($close, {p}) + {EPS})", f"bb_width_{p}")

        # ── ROC / MOM ──
        for p in [5, 10, 20, 60]:
            add(f"($close / Ref($close, {p}) - 1) * 100", f"roc_{p}")
        for p in [5, 10, 20]:
            add(f"$close - Ref($close, {p})", f"mom_{p}")

        # ── CCI ──
        for p in [14, 20]:
            tp = f"($high + $low + $close) / 3"
            add(f"({tp} - Mean({tp}, {p})) / (0.015 * Std({tp}, {p}) + {EPS})", f"cci_{p}")

        # ── Williams %R ──
        for p in [6, 14]:
            add(f"(Max($high, {p}) - $close) / (Max($high, {p}) - Min($low, {p}) + {EPS}) * -100", f"wr_{p}")

        # ── 量价相关性 ──
        for p in [20, 60]:
            add(f"Corr($close, $volume, {p})", f"corr_cv_{p}")

        # ── 高低价/开盘特征 ──
        add(f"($high - $low) / ($close + {EPS})", "hl_range")
        add(f"$open / (Ref($close, 1) + {EPS}) - 1", "open_gap")
        add(f"($close - $open) / ($open + {EPS})", "close_vs_open")

        # ── 加速/减速 ──
        for p in [3, 5, 10, 20]:
            add(f"($close / Ref($close, {p}) - 1) - Ref($close / Ref($close, {p}) - 1, {p})", f"accel_{p}")

        # ── 涨跌统计 ──
        for p in [5, 10, 20]:
            add(f"Sum(Gt($close, Ref($close, 1)), {p})", f"up_streak_{p}")
            add(f"Sum(Lt($close, Ref($close, 1)), {p})", f"down_streak_{p}")

        return fields, names

    def get_label_config(self):
        """未来 horizon 根K线收益率"""
        label_expr = f"Ref($close, -{self.horizon}) / Ref($close, -1) - 1"
        return [label_expr], ["LABEL0"]


MODEL_CONFIGS = {
    'LightGBM': {
        'class': 'LGBModel',
        'module_path': 'qlib.contrib.model.gbdt',
        'kwargs': {
            'loss': 'mse', 'num_leaves': 127, 'max_depth': 9,
            'learning_rate': 0.001, 'n_estimators': 5000, 'early_stopping_rounds': 200,
            'subsample': 0.6, 'colsample_bytree': 0.5,
            'reg_alpha': 0.1, 'reg_lambda': 0.5, 'min_child_samples': 50,
            'verbosity': 1, 'seed': 42, 'n_jobs': 4,
        }
    },
    'GRU': {
        'class': 'GRU', 'module_path': 'qlib.contrib.model.pytorch_gru',
        'kwargs': {'d_feat': N_FEAT, 'hidden_size': 128, 'num_layers': 2,
                   'dropout': 0.2, 'n_epochs': 100, 'lr': 0.001,
                   'early_stop': 20, 'batch_size': 2048, 'GPU': 0, 'seed': 42}
    },
    'LSTM': {
        'class': 'LSTM', 'module_path': 'qlib.contrib.model.pytorch_lstm',
        'kwargs': {'d_feat': N_FEAT, 'hidden_size': 128, 'num_layers': 2,
                   'dropout': 0.2, 'n_epochs': 100, 'lr': 0.001,
                   'early_stop': 20, 'batch_size': 2048, 'GPU': 0, 'seed': 42}
    },
    'Transformer': {
        'class': 'Transformer', 'module_path': 'qlib.contrib.model.pytorch_transformer',
        'kwargs': {'d_feat': N_FEAT, 'd_model': 128, 'n_head': 4, 'num_layers': 2,
                   'dropout': 0.1, 'n_epochs': 100, 'lr': 0.0001,
                   'early_stop': 20, 'batch_size': 2048, 'GPU': 0, 'seed': 42}
    },
    'TabNet': {
        'class': 'TabNetModel', 'module_path': 'qlib.contrib.model.pytorch_tabnet',
        'kwargs': {'d_feat': N_FEAT, 'n_d': 32, 'n_a': 32, 'n_steps': 3,
                   'gamma': 1.3, 'n_epochs': 100, 'lr': 0.001,
                   'early_stop': 20, 'batch_size': 2048, 'GPU': 0, 'seed': 42}
    },
}


def get_dataset_config(horizon: int, quick: bool = False):
    handler_kwargs = {
        'start_time': START_TIME, 'end_time': END_TIME,
        'fit_start_time': START_TIME, 'fit_end_time': TRAIN_END,
        'instruments': 'all', 'day_length': DAY_LENGTH, 'freq': FREQ,
        'columns': ['$open', '$high', '$low', '$close'],
        'horizon': horizon,
    }
    if quick:
        handler_kwargs['instruments'] = 'csi300'

    return {
        'class': 'DatasetH',
        'module_path': 'qlib.data.dataset',
        'kwargs': {
            'handler': {
                'class': 'IntradayHandler',
                'module_path': 'qlib_pipeline.train',
                'kwargs': handler_kwargs,
            },
            'segments': {
                'train': (START_TIME, TRAIN_END),
                'valid': (TRAIN_END, VAL_END),
                'test': (VAL_END, END_TIME),
            },
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Qlib 分钟级择时训练')
    parser.add_argument('--model', default='LightGBM', choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--bin-dir', default=BIN_DIR, help='Qlib .bin 数据目录')
    parser.add_argument('--horizon', type=int, default=3, help='预测未来几根K线')
    parser.add_argument('--quick', action='store_true', help='快速验证')
    parser.add_argument('--no-backtest', action='store_true', help='跳过回测')
    args = parser.parse_args()

    if not os.path.exists(args.bin_dir):
        print(f"❌ 数据目录不存在: {args.bin_dir}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f" Qlib 分钟级择时: {args.model} (HighFreqGeneralHandler)")
    print(f" 频率: {FREQ} | day_length: {DAY_LENGTH} | horizon: {args.horizon}")
    print(f" 数据: {args.bin_dir}")
    if args.quick: print(f" ⚡ 快速验证")
    print(f"{'='*60}")

    qlib.init(provider_uri=args.bin_dir, region=REG_CN, freq=FREQ,
              custom_ops=[Cut, DayLast, FFillNan, IsNull], expression_cache=None)

    dataset_config = get_dataset_config(args.horizon, args.quick)
    model_config = MODEL_CONFIGS[args.model]

    print(f"\n📦 模型: {model_config['class']} | 📊 特征: {N_FEAT} | 🎯 标签: 未来{args.horizon}根K线收益率")

    exp_name = f"{EXPERIMENT_NAME}_{args.model}_h{args.horizon}"
    if args.quick: exp_name += '_quick'

    t0 = time.time()
    with R.start(experiment_name=exp_name):
        model = init_instance_by_config(model_config)
        dataset = init_instance_by_config(dataset_config)

        print(f"\n🏋️ 训练模型...")
        model.fit(dataset)
        print(f"  训练耗时: {time.time()-t0:.0f}s")

        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        print(f"\n📊 信号分析...")
        sar = SigAnaRecord(recorder)
        sar.generate()

        if not args.no_backtest:
            print(f"\n📈 回测...")
            port_config = {
                'executor': {'class': 'SimulatorExecutor', 'module_path': 'qlib.backtest.executor',
                             'kwargs': {'time_per_step': FREQ, 'generate_portfolio_metrics': True}},
                'strategy': {'class': 'TopkDropoutStrategy', 'module_path': 'qlib.contrib.strategy.signal_strategy',
                             'kwargs': {'topk': 50, 'n_drop': 5, 'method': 'topk'}},
                'backtest': {'start_time': VAL_END, 'end_time': END_TIME, 'account': 1000000,
                             'exchange_kwargs': {'freq': 'day', 'limit_threshold': 0.095, 'deal_price': 'close',
                                                 'open_cost': 0.0005, 'close_cost': 0.0015, 'min_cost': 5}},
            }
            try:
                par = PortAnaRecord(recorder, port_config, FREQ)
                par.generate()
            except Exception as e:
                print(f"  ⚠️ 回测跳过: {e}")

        os.makedirs(MODEL_DIR, exist_ok=True)
        R.save_objects(**{'model': model})
        print(f"\n💾 模型已保存: {MODEL_DIR}/{args.model.lower()}_{FREQ}.pkl")

    print(f"\n{'='*60}")
    print(f" ✅ 完成! 总耗时: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()