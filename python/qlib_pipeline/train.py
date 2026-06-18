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

import os, sys, argparse, time, json, copy, warnings, io, contextlib
import numpy as np

# 抑制 numpy 除零/自由度警告
np.seterr(all='ignore')
warnings.filterwarnings('ignore')

# 抑制 Gym 刷屏警告 (gym.__init__ 直接 print 到 stderr)
_stderr_backup = sys.stderr
sys.stderr = io.StringIO()
try:
    import gym
except Exception:
    pass
finally:
    sys.stderr = _stderr_backup

# 抑制 Gym 废弃通知 (旧版 gym 兼容)
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

N_FEAT = 0  # 运行时根据实际特征数动态设置


class IntradayHandler(HighFreqGeneralHandler):
    """丰富的分钟级特征 + 标签处理器 (~120+ features)"""

    def __init__(self, horizon=3, quiet=False, top_features=None, **kwargs):
        self.horizon = horizon
        self.quiet = quiet
        self._top_features = set(top_features) if top_features else None
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
        用 numpy 逐股票计算, 确保不跨股票边界
        """
        import numpy as np
        import sys
        label_key = ('label', 'LABEL0')
        
        processed_ids = set()
        
        for attr in ['_learn', '_infer']:
            df = getattr(self, attr, None)
            if df is None:
                continue
            obj_id = id(df)
            if obj_id in processed_ids:
                continue
            processed_ids.add(obj_id)
            
            if label_key not in df.columns:
                continue
            
            if not self.quiet:
                print(f"[IntradayHandler] Fixing {attr} label, shape={df.shape}, before mean={df[label_key].mean():.4f}", file=sys.stderr)
            
            # 按 instrument 分组, 逐组用 numpy 向量化计算
            close_series = df[label_key]
            inst_level = 'instrument' if 'instrument' in df.index.names else df.index.names[0]
            
            new_labels = close_series.copy()
            for inst, idx in close_series.groupby(level=inst_level).groups.items():
                grp = close_series.loc[idx].sort_index(level=1)
                vals = grp.values.astype(np.float64)
                n = len(vals)
                result = np.full(n, np.nan)
                if n > self.horizon:
                    n_valid = n - self.horizon
                    # close(t+horizon) / close(t) - 1
                    # h=1: close(t+1)/close(t)-1, h=3: close(t+3)/close(t)-1
                    future_close = vals[self.horizon:self.horizon + n_valid]
                    cur_close = vals[:n_valid]
                    mask = (cur_close > 0) & (future_close > 0)
                    result[:n_valid][mask] = future_close[mask] / cur_close[mask] - 1.0
                new_labels.loc[grp.index] = result
            
            df[label_key] = new_labels
            df[label_key] = df[label_key].replace([float('inf'), float('-inf')], float('nan'))
            # XGBoost 不接受 NaN 标签, 填 0 (LightGBM 自动忽略 NaN)
            df[label_key] = df[label_key].fillna(0.0)
            
            # 清理所有特征列中的 inf (XGBoost 不接受 inf 值)
            feat_cols = [c for c in df.columns if c != label_key]
            df[feat_cols] = df[feat_cols].replace([float('inf'), float('-inf')], float('nan'))
            
            if not self.quiet:
                print(f"[IntradayHandler] {attr} label fixed, after mean={df[label_key].mean():.6f}, nan={df[label_key].isna().sum()}", file=sys.stderr)


    def get_feature_config(self):
        """
        扩展特征集 (~80+ 特征):
        - 归一化 OHLC (当日 + 前日)
        - 多周期收益率 (1/2/3/5/8/10/15/20/30 根K线)
        - 量比 + 量加速度
        - RSI (6/14/24)
        - MACD + 信号线 + 柱
        - 波动率 (滚动std)
        - 日内位置 (在当日高低价区间的位置)
        - 布林带 (%B)
        - 价格加速度
        """
        EPS = '1e-6'
        DL = self.day_length
        fields, names = [], []

        def add(expr, name):
            fields.append(expr)
            names.append(name)

        # ── 归一化 OHLC (当日/昨日) ──
        for col in self.columns:
            add(f"Cut({col} / (DayLast(Ref(FFillNan($close), {DL * 2})) + {EPS}), {DL * 2}, None)", col)
        for col in self.columns:
            add(f"Cut(Ref({col}, {DL}) / (DayLast(Ref(FFillNan($close), {DL})) + {EPS}), {DL * 2}, None)", f"{col}_1")

        # ── 归一化成交量 ──
        add(f"Cut($volume / (Ref(DayLast(Mean($volume, {DL * 30})), {DL}) + {EPS}), {DL * 2}, None)", "$volume")
        add(f"Cut(Ref($volume, {DL}) / (Ref(DayLast(Mean($volume, {DL * 30})), {DL}) + {EPS}), {DL * 2}, None)", "$volume_1")

        # ── 多周期收益率 (动量) ──
        for p in [1, 2, 3, 5, 8, 10, 15, 20, 30]:
            add(f"$close / Ref($close, {p}) - 1", f"ret_{p}")

        # ── 收益率加速度 (ret_1 - ret_N) ──
        for p in [2, 3, 5]:
            add(f"($close / Ref($close, 1) - 1) - (Ref($close, 1) / Ref($close, {p + 1}) - 1)", f"ret_acc_{p}")

        # ── 量比 ──
        for p in [3, 5, 8, 10, 20, 30]:
            add(f"$volume / (Mean($volume, {p}) + {EPS}) - 1", f"vol_ratio_{p}")

        # ── 量加速度 ──
        for p in [3, 5, 10]:
            add(f"($volume / Ref($volume, 1) - 1) - (Ref($volume, {p}) / Ref($volume, {p + 1}) - 1)", f"vol_acc_{p}")

        # ── RSI ──
        for p in [6, 14, 24]:
            up = f"If(Gt($close - Ref($close, 1), 0), $close - Ref($close, 1), 0)"
            down = f"If(Gt(Ref($close, 1) - $close, 0), Ref($close, 1) - $close, 0)"
            add(f"100 - 100 / (1 + Mean({up}, {p}) / (Mean({down}, {p}) + {EPS}))", f"rsi_{p}")

        # ── MACD (多周期) ──
        for fast, slow, sig in [(12, 26, 9), (6, 13, 5), (24, 52, 18)]:
            add(f"EMA($close, {fast}) - EMA($close, {slow})", f"macd_{fast}_{slow}")
            add(f"(EMA($close, {fast}) - EMA($close, {slow})) - EMA(EMA($close, {fast}) - EMA($close, {slow}), {sig})", f"macd_hist_{fast}_{slow}")

        # ── 波动率 (滚动标准差) ──
        for p in [5, 10, 20, 40]:
            add(f"Std($close / Ref($close, 1) - 1, {p})", f"vol_{p}")

        # ── 日内位置特征 ──
        # 当前价格在日内区间的位置
        add(f"($close - DayLast(Ref(FFillNan($low), {DL * 2}))) / (DayLast(Ref(FFillNan(FFillNan($high) - FFillNan($low)), {DL * 2})) + {EPS})", "day_pos")
        # 日内涨跌幅
        add(f"$close / DayLast(Ref(FFillNan($close), {DL * 2})) - 1", "day_ret")
        # 日内振幅
        add(f"(DayLast(Ref(FFillNan($high) - FFillNan($low), {DL * 2}))) / (DayLast(Ref(FFillNan($close), {DL * 2})) + {EPS})", "day_range")

        # ── 布林带 %B ──
        for p in [20, 40]:
            ma = f"Mean($close, {p})"
            std = f"Std($close, {p})"
            add(f"($close - ({ma} - 2 * {std})) / (4 * ({std} + {EPS}))", f"bb_pct_{p}")

        # ── 高低价差 ──
        add(f"($high - $low) / ($close + {EPS})", "hl_ratio")
        add(f"($close - $open) / ($open + {EPS})", "co_ratio")

        # ── 价格与均线偏离 ──
        for p in [5, 10, 20, 40]:
            add(f"$close / (Mean($close, {p}) + {EPS}) - 1", f"ma_dev_{p}")

        # ── KDJ ──
        for p in [9, 14]:
            highest = f"Max($high, {p})"
            lowest = f"Min($low, {p})"
            rsv = f"100 * ($close - {lowest}) / ({highest} - {lowest} + {EPS})"
            k = f"Mean({rsv}, 3)"
            d = f"Mean({k}, 3)"
            add(k, f"kdj_k_{p}")
            add(d, f"kdj_d_{p}")
            add(f"3 * {k} - 2 * {d}", f"kdj_j_{p}")

        # ── 隔夜跳空 ──
        for col in ['$open']:
            add(f"({col} / Ref($close, 1) - 1)", "overnight_gap")

        # ── 日内累计收益 ──
        add(f"$close / DayLast(Ref(FFillNan($open), {DL * 2})) - 1", "intraday_ret")

        # ── 相对强弱 ──
        for p in [5, 10, 20]:
            add(f"($close / Ref($close, 1) - 1) - (Mean($close, {p}) / Ref(Mean($close, {p}), 1) - 1)", f"rel_strength_{p}")

        # ── 价格百分位 ──
        for p in [20, 40]:
            highest = f"Max($high, {p})"
            lowest = f"Min($low, {p})"
            add(f"($close - {lowest}) / ({highest} - {lowest} + {EPS})", f"price_pct_{p}")

        # ── 均线斜率 (均线变化率) ──
        for p in [5, 10, 20]:
            add(f"Mean($close, {p}) / Ref(Mean($close, {p}), 1) - 1", f"ma_slope_{p}")

        # ── 量价背离 ──
        for p in [5, 10, 20]:
            add(f"($close / Ref($close, {p}) - 1) * ($volume / (Mean($volume, {p}) + {EPS}) - 1)", f"pv_div_{p}")

        # ── 高低价位置 ──
        for p in [5, 10, 20]:
            add(f"($close - Min($low, {p})) / (Max($high, {p}) - Min($low, {p}) + {EPS})", f"hl_pos_{p}")

        # ── 特征筛选 (如果指定了 top_features) ──
        if self._top_features is not None:
            filtered = [(f, n) for f, n in zip(fields, names) if n in self._top_features]
            if filtered:
                fields, names = zip(*filtered)
                fields, names = list(fields), list(names)

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
            'loss': 'mse', 'num_leaves': 127, 'max_depth': -1,
            'learning_rate': 0.03, 'n_estimators': 5000, 'early_stopping_rounds': 200,
            'subsample': 0.6, 'colsample_bytree': 0.6,
            'reg_alpha': 0.0, 'reg_lambda': 5.0, 'min_child_samples': 50,
            'min_split_gain': 0.001, 'verbosity': -1, 'seed': 42, 'n_jobs': 4,
        }
    },
    'XGBoost': {
        'class': 'XGBModel',
        'module_path': 'qlib.contrib.model.xgboost',
        'kwargs': {
            'objective': 'reg:squarederror', 'max_depth': 8,
            'learning_rate': 0.03, 'n_estimators': 5000, 'early_stopping_rounds': 200,
            'subsample': 0.7, 'colsample_bytree': 0.6,
            'reg_alpha': 0.0, 'reg_lambda': 5.0, 'min_child_weight': 5,
            'verbosity': 0, 'seed': 42, 'n_jobs': 4,
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


def get_dataset_config(horizon: int, quick: bool = False, max_stocks: int = 0, quiet: bool = False):
    handler_kwargs = {
        'start_time': START_TIME, 'end_time': END_TIME,
        'fit_start_time': START_TIME, 'fit_end_time': TRAIN_END,
        'instruments': 'all', 'day_length': DAY_LENGTH, 'freq': FREQ,
        'columns': ['$open', '$high', '$low', '$close'],
        'horizon': horizon,
        'quiet': quiet,
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
    parser.add_argument('--quiet', action='store_true', help='减少输出')
    parser.add_argument('--feature-limit', type=int, default=0, help='只使用 Top-K 特征 (0=全部, 需先跑 ensemble.py 生成重要性文件)')
    parser.add_argument('--feature-importance', default='', help='特征重要性 CSV 路径')
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

    dataset_config = get_dataset_config(args.horizon, args.quick, args.max_stocks, args.quiet)
    
    # ── 特征筛选: 加载重要性文件, 取 Top-K ──
    top_features = None
    if args.feature_limit > 0:
        fi_path = args.feature_importance or os.path.join(ROOT, 'experiments', 'feature_importance_ensemble.csv')
        if not os.path.exists(fi_path):
            fi_path = os.path.join(ROOT, 'experiments', 'feature_importance_h5.csv')
        if os.path.exists(fi_path):
            import csv
            with open(fi_path) as f:
                rows = list(csv.DictReader(f))
            # 用 combined 列或 importance 列
            col = 'combined' if 'combined' in rows[0] else 'importance'
            rows.sort(key=lambda r: -float(r[col]))
            top_features = [r['feature'] for r in rows[:args.feature_limit]]
            print(f"📋 特征筛选: Top-{args.feature_limit} (来自 {fi_path})")
        else:
            print(f"⚠️ 特征重要性文件不存在: {fi_path}, 请先运行 ensemble.py")
    
    if top_features:
        dataset_config['kwargs']['handler']['kwargs']['top_features'] = top_features
    model_config = MODEL_CONFIGS[args.model]

    t0 = time.time()
    with R.start(experiment_name=f"{EXPERIMENT_NAME}_{args.model}_h{args.horizon}" + ('_quick' if args.quick else '')):
        # 先创建 dataset, 获取实际特征数
        dataset = init_instance_by_config(dataset_config)
        n_feat = len(dataset.handler.get_feature_config()[0])
        
        # 更新深度学习模型的 d_feat
        if 'd_feat' in model_config.get('kwargs', {}):
            model_config['kwargs']['d_feat'] = n_feat
        
        print(f"\n📦 模型: {model_config['class']} | 📊 特征: {n_feat} | 🎯 标签: 未来{args.horizon}根K线收益率")
        
        model = init_instance_by_config(model_config)

        # ── 数据验证 ──
        import numpy as np
        df_check = dataset.prepare('train', col_set=["feature", "label"])
        lab = df_check['label'].values
        lab_mean = float(np.nanmean(lab))
        lab_std = float(np.nanstd(lab))
        nan_count = int(np.isnan(lab).sum())
        inf_count = int(np.isinf(lab).sum())
        print(f"🔍 数据验证: {df_check.shape[0]:,} 样本 | label mean={lab_mean:.6f} std={lab_std:.6f} | NaN={nan_count} Inf={inf_count}")
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

        # ── 特征重要性 (LGBM/XGBoost/CatBoost) ──
        feature_names = dataset.handler.get_feature_config()[1]
        if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
            importances = model.model.feature_importances_
            ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])
            print(f"\n📊 特征重要性 Top 20:")
            for i, (name, imp) in enumerate(ranked[:20]):
                bar = '█' * int(imp / ranked[0][1] * 20)
                print(f"  {i+1:>2}. {name:<25s} {imp:.6f} {bar}")
            print(f"  ... (共 {len(ranked)} 个特征)")
            # 保存到文件
            out_dir = os.path.join(ROOT, 'experiments')
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, 'feature_importance_h5.csv'), 'w') as f:
                f.write("feature,importance\n")
                for name, imp in ranked:
                    f.write(f"{name},{imp}\n")
            print(f"  特征重要性已保存: experiments/feature_importance_h5.csv")

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