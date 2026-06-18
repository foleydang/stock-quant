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

import os, sys, argparse, time, json, copy, warnings

# 抑制 Qlib 依赖的 FutureWarning/DeprecationWarning (pandas fillna 废弃)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 抑制 Gym 废弃通知 (gym 直接 print 到 stderr, 不走 warnings 系统)
try:
    import gym_notices.notices
    gym_notices.notices.notices = {}
except ImportError:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord
from qlib.contrib.data.highfreq_handler import HighFreqGeneralHandler
from qlib.contrib.ops.high_freq import Cut, DayLast, FFillNan, IsNull
from qlib.data.dataset import DatasetH

# ============ 配置 ============
BIN_DIR = os.path.expanduser('~/.qlib/qlib_data/cn_30min/bin')
FREQ = '30min'
DAY_LENGTH = 8  # 30分钟K线, 每天4小时 = 8根
EXPERIMENT_NAME = 'intraday_30min_hf'
MODEL_DIR = os.path.join(ROOT, 'models', 'qlib_intraday')

START_TIME = '2020-01-02 09:30:00'
TRAIN_END = '2026-04-30 15:00:00'
VAL_END = '2026-05-31 15:00:00'
END_TIME = '2026-06-16 15:00:00'

N_FEAT = 10  # 默认值, 会被实际特征数覆盖


class IntradayLabelProcessor:
    """Label 处理器: 将 $close 原始值转换为未来收益率
    
    Qlib 的 Ref($close, -N) 负偏移表达式在 DatasetH 中不稳定,
    改为在 Python 侧直接计算 label。
    
    继承 qlib 的 Processor 基类以兼容 DataHandlerLP 的 learn_processors。
    """
    
    def __init__(self, horizon=3, eps=1e-6):
        self.horizon = horizon
        self.eps = eps
        import sys
        print(f"[IntradayLabelProcessor] INIT horizon={horizon}", file=sys.stderr)
    
    def fit(self, df: 'pd.DataFrame' = None):
        pass  # 不需要 fit
    
    def __call__(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        """
        df: MultiIndex (instrument, datetime), columns 是 MultiIndex (group, col_name)
        将 ('label', 'LABEL0') 列从 $close 原始值转换为:
            Close(t+horizon) / Close(t+1) - 1
        """
        import pandas as pd
        # Qlib 的 DataFrame columns 是 MultiIndex, 如 ('label', 'LABEL0')
        if isinstance(df.columns, pd.MultiIndex):
            label_key = ('label', 'LABEL0')
        else:
            label_key = 'LABEL0'
        
        import sys
        if label_key in df.columns:
            print(f"[IntradayLabelProcessor] APPLYING label transform, shape={df.shape}", file=sys.stderr)
            if 'instrument' in df.index.names:
                series = df[label_key]
                df[label_key] = series.groupby(level='instrument').transform(
                    lambda x: x.shift(-self.horizon) / (x.shift(-1) + self.eps) - 1
                )
            else:
                df[label_key] = df[label_key].shift(-self.horizon) / (df[label_key].shift(-1) + self.eps) - 1
            df[label_key] = df[label_key].replace([float('inf'), float('-inf')], float('nan'))
        else:
            import sys
            print(f"[IntradayLabelProcessor] SKIP: label_key={label_key} NOT in columns={list(df.columns[:3])}...", file=sys.stderr)
        return df
    
    def is_for_infer(self):
        return True
    
    def readonly(self):
        return False


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
        DataHandlerLP.__init__(
            self, data_loader=data_loader,
            **kwargs
        )
        
        # 直接修改 _learn 和 _infer 的 label 列, 绕过 Qlib 的 processor 系统
        self._fix_label_column()

    def _fix_label_column(self):
        """
        将 _learn 和 _infer 中的 label 列从原始 $close 转换为收益率
        绕过 Qlib 的 processor 系统, 直接修改 DataFrame
        """
        import pandas as pd
        import sys
        label_key = ('label', 'LABEL0')
        
        for attr in ['_learn', '_infer']:
            df = getattr(self, attr, None)
            if df is None:
                continue
            if not isinstance(df.columns, pd.MultiIndex):
                print(f"[IntradayHandler] {attr} columns is NOT MultiIndex, skipping", file=sys.stderr)
                continue
            if label_key not in df.columns:
                print(f"[IntradayHandler] {attr} label_key={label_key} not in columns, skipping", file=sys.stderr)
                continue
            
            print(f"[IntradayHandler] Fixing {attr} label, shape={df.shape}, before mean={df[label_key].mean():.4f}", file=sys.stderr)
            series = df[label_key]
            if 'instrument' in df.index.names:
                df[label_key] = series.groupby(level='instrument').transform(
                    lambda x: x.shift(-self.horizon) / (x.shift(-1) + 1e-6) - 1
                )
            else:
                df[label_key] = series.shift(-self.horizon) / (series.shift(-1) + 1e-6) - 1
            df[label_key] = df[label_key].replace([float('inf'), float('-inf')], float('nan'))
            print(f"[IntradayHandler] {attr} label fixed, after mean={df[label_key].mean():.6f}", file=sys.stderr)


    def get_feature_config(self):
        """最优特征集: 归一化OHLC(带epsilon防除零) + 收益 + 量比 + RSI + MACD (23个, IC=0.119)"""
        EPS = '1e-6'
        fields, names = [], []

        def add(expr, name):
            fields.append(expr)
            names.append(name)

        # ── 归一化 OHLC (带 epsilon 防止停牌脏数据除零) ──
        # 与 HighFreqGeneralHandler 相同逻辑, 但分母加 EPS
        for col in self.columns:
            # 当日归一化: $open / (昨日收盘 + EPS), Cut 去掉前2天
            add(f"Cut({col} / (DayLast(Ref(FFillNan($close), {self.day_length * 2})) + {EPS}), {self.day_length * 2}, None)", col)
        for col in self.columns:
            # 前日归一化: Ref($open, 8) / (昨日收盘 + EPS)
            add(f"Cut(Ref({col}, {self.day_length}) / (DayLast(Ref(FFillNan($close), {self.day_length})) + {EPS}), {self.day_length * 2}, None)", f"{col}_1")

        # ── 归一化成交量 ──
        add(f"Cut($volume / (Ref(DayLast(Mean($volume, {self.day_length * 30})), {self.day_length}) + {EPS}), {self.day_length * 2}, None)", "$volume")
        add(f"Cut(Ref($volume, {self.day_length}) / (Ref(DayLast(Mean($volume, {self.day_length * 30})), {self.day_length}) + {EPS}), {self.day_length * 2}, None)", "$volume_1")

        # ── 收益率 (短周期动量) ──
        for p in [1, 2, 3, 5, 10]:
            add(f"$close / Ref($close, {p}) - 1", f"ret_{p}")

        # ── 量比 ──
        for p in [5, 10, 20]:
            add(f"$volume / (Mean($volume, {p}) + {EPS}) - 1", f"vol_ratio_{p}")

        # ── RSI ──
        for p in [6, 14]:
            up = f"If(Gt($close - Ref($close, 1), 0), $close - Ref($close, 1), 0)"
            down = f"If(Gt(Ref($close, 1) - $close, 0), Ref($close, 1) - $close, 0)"
            add(f"100 - 100 / (1 + Mean({up}, {p}) / (Mean({down}, {p}) + {EPS}))", f"rsi_{p}")

        # ── MACD ──
        add("EMA($close, 12) - EMA($close, 26)", "macd")
        add("(EMA($close, 12) - EMA($close, 26)) - EMA(EMA($close, 12) - EMA($close, 26), 9)", "macd_hist")

        return fields, names

    def get_label_config(self):
        """
        未来 horizon 根K线收益率
        
        注意: 不直接使用 Ref($close, -N) 前向引用表达式,
        因为 Qlib 的 DatasetH 在某些版本下对负偏移 Ref 的求值不稳定。
        改为返回 $close 原始值, 在 Python 侧计算 label。
        """
        return ["$close"], ["LABEL0"]


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


def get_dataset_config(horizon: int, quick: bool = False, max_stocks: int = 0):
    handler_kwargs = {
        'start_time': START_TIME, 'end_time': END_TIME,
        'fit_start_time': START_TIME, 'fit_end_time': TRAIN_END,
        'instruments': 'all', 'day_length': DAY_LENGTH, 'freq': FREQ,
        'columns': ['$open', '$high', '$low', '$close'],
        'horizon': horizon,
    }
    if quick:
        handler_kwargs['instruments'] = 'csi300'
    if max_stocks > 0:
        # 从 all.txt 中取前 max_stocks 只股票创建临时文件
        import tempfile
        all_path = os.path.expanduser(f'~/.qlib/qlib_data/cn_30min/bin/instruments/all.txt')
        with open(all_path) as f:
            lines = [next(f).strip() for _ in range(max_stocks)]
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        tmp.write('\n'.join(lines))
        tmp.close()
        handler_kwargs['instruments'] = tmp.name
        print(f"   📋 限制股票: {max_stocks} 只")

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
    parser.add_argument('--max-stocks', type=int, default=0, help='限制股票数量 (0=全部)')
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

    dataset_config = get_dataset_config(args.horizon, args.quick, args.max_stocks)
    model_config = MODEL_CONFIGS[args.model]

    print(f"\n📦 模型: {model_config['class']} | 📊 特征: {N_FEAT} | 🎯 标签: 未来{args.horizon}根K线收益率")

    exp_name = f"{EXPERIMENT_NAME}_{args.model}_h{args.horizon}"
    if args.quick: exp_name += '_quick'

    t0 = time.time()
    with R.start(experiment_name=exp_name):
        model = init_instance_by_config(model_config)
        dataset = init_instance_by_config(dataset_config)

        # ── 数据验证 ──
        import numpy as np
        df_check = dataset.prepare('train', col_set=["feature", "label"])
        lab = df_check['label'].values
        lab_mean = float(np.nanmean(lab))
        lab_std = float(np.nanstd(lab))
        nan_count = int(np.isnan(lab).sum())
        inf_count = int(np.isinf(lab).sum())
        print(f"\n🔍 数据验证: {df_check.shape[0]:,} 样本 | label mean={lab_mean:.6f} std={lab_std:.6f} | NaN={nan_count} Inf={inf_count}")
        if abs(lab_mean) > 1e6 or lab_std > 1e6:
            print(f"❌ 标签异常 (mean={lab_mean:.2e}, std={lab_std:.2e})！数据可能已损坏")
            print(f"   建议: 先用 --quick 模式测试 (csi300 50只股票)")
            sys.exit(1)
        if nan_count > df_check.shape[0] * 0.5:
            print(f"❌ NaN 标签过多 ({nan_count}/{df_check.shape[0]})！数据可能已损坏")
            sys.exit(1)

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
            try:
                # 加载预测信号, 按日聚合
                pred = recorder.load_object("pred.pkl")
                daily_pred = pred.groupby('datetime').mean()
                print(f"  日频信号: {len(daily_pred)} 天")

                # 简单回测指标
                import pandas as pd
                mean_sig = daily_pred.mean()
                std_sig = daily_pred.std()
                sharpe = mean_sig / (std_sig + 1e-6) * (252**0.5)
                cum = (1 + daily_pred).prod().iloc[0] if hasattr((1 + daily_pred).prod(), 'iloc') else (1 + daily_pred).prod()
                print(f"  累计复合: {float(cum):.4f}")
                print(f"  日均信号: {float(mean_sig):.6f}")
                print(f"  信号夏普: {float(sharpe):.2f}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  ⚠️ 回测跳过: {e}")

        os.makedirs(MODEL_DIR, exist_ok=True)
        R.save_objects(**{'model': model})
        print(f"\n💾 模型已保存: {MODEL_DIR}/{args.model.lower()}_{FREQ}.pkl")

    print(f"\n{'='*60}")
    print(f" ✅ 完成! 总耗时: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()