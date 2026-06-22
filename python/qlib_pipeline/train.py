#!/usr/bin/env python3
"""
统一训练脚本 — 多模型 + 多周期 + 特征筛选 + 加权集成

功能:
  - 多模型: LGBM / XGBoost / CatBoost
  - 多周期: h=1/3/5/10 同时训练
  - 特征筛选: 自动 IC 筛选 + 特征重要性筛选
  - 标签: 波动率归一化收益率 (更稳定)
  - 损失: Huber loss (对异常值鲁棒)
  - 集成: 按验证集 RankIC 加权

用法:
  python qlib_pipeline/train.py                      # 默认: 全部模型, 全部horizon
  python qlib_pipeline/train.py --quick              # 快速验证 (csi300)
  python qlib_pipeline/train.py --horizons 1,3       # 只跑 h=1,3
  python qlib_pipeline/train.py --models LightGBM    # 只跑 LightGBM
  python qlib_pipeline/train.py --no-hpo --no-ensemble  # 只跑基础训练
"""

import os, sys, time, json, warnings, argparse, io, csv, hashlib
import numpy as np
import pandas as pd

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

# ============ 全局配置 ============
BIN_DIR = os.path.expanduser('~/.qlib/qlib_data/cn_30min/bin')
FREQ = '30min'
DAY_LENGTH = 8
START_TIME = '2020-01-02 09:30:00'
TRAIN_END = '2026-04-30 15:00:00'
VAL_END = '2026-05-31 15:00:00'
END_TIME = '2026-06-16 15:00:00'
DEFAULT_HORIZONS = [1, 3, 5, 10]

# ============ 特征表达式 (~180+ 特征) ============

def build_feature_expressions(columns=None, day_length=DAY_LENGTH):
    """构建所有特征表达式, 返回 (fields, names)"""
    if columns is None:
        columns = ['$open', '$high', '$low', '$close']
    EPS = '1e-6'
    DL = day_length
    fields, names = [], []

    def add(expr, name):
        fields.append(expr)
        names.append(name)

    # 归一化 OHLC (当日/昨日)
    for col in columns:
        add(f"Cut({col} / (DayLast(Ref(FFillNan($close), {DL * 2})) + {EPS}), {DL * 2}, None)", col)
    for col in columns:
        add(f"Cut(Ref({col}, {DL}) / (DayLast(Ref(FFillNan($close), {DL})) + {EPS}), {DL * 2}, None)", f"{col}_1")

    # 归一化成交量
    add(f"Cut($volume / (Ref(DayLast(Mean($volume, {DL * 30})), {DL}) + {EPS}), {DL * 2}, None)", "$volume")
    add(f"Cut(Ref($volume, {DL}) / (Ref(DayLast(Mean($volume, {DL * 30})), {DL}) + {EPS}), {DL * 2}, None)", "$volume_1")

    # 多周期收益率
    for p in [1, 2, 3, 5, 8, 10, 15, 20, 30]:
        add(f"$close / Ref($close, {p}) - 1", f"ret_{p}")

    # 收益率加速度
    for p in [2, 3, 5]:
        add(f"($close / Ref($close, 1) - 1) - (Ref($close, 1) / Ref($close, {p + 1}) - 1)", f"ret_acc_{p}")

    # 量比
    for p in [3, 5, 8, 10, 20, 30]:
        add(f"$volume / (Mean($volume, {p}) + {EPS}) - 1", f"vol_ratio_{p}")

    # 量加速度
    for p in [3, 5, 10]:
        add(f"($volume / Ref($volume, 1) - 1) - (Ref($volume, {p}) / Ref($volume, {p + 1}) - 1)", f"vol_acc_{p}")

    # RSI
    for p in [6, 14, 24]:
        up = f"If(Gt($close - Ref($close, 1), 0), $close - Ref($close, 1), 0)"
        down = f"If(Gt(Ref($close, 1) - $close, 0), Ref($close, 1) - $close, 0)"
        add(f"100 - 100 / (1 + Mean({up}, {p}) / (Mean({down}, {p}) + {EPS}))", f"rsi_{p}")

    # MACD
    for fast, slow, sig in [(12, 26, 9), (6, 13, 5), (24, 52, 18)]:
        add(f"EMA($close, {fast}) - EMA($close, {slow})", f"macd_{fast}_{slow}")
        add(f"(EMA($close, {fast}) - EMA($close, {slow})) - EMA(EMA($close, {fast}) - EMA($close, {slow}), {sig})", f"macd_hist_{fast}_{slow}")

    # 波动率
    for p in [5, 10, 20, 40]:
        add(f"Std($close / Ref($close, 1) - 1, {p})", f"vol_{p}")

    # 日内位置
    add(f"($close - DayLast(Ref(FFillNan($low), {DL * 2}))) / (DayLast(Ref(FFillNan(FFillNan($high) - FFillNan($low)), {DL * 2})) + {EPS})", "day_pos")
    add(f"$close / DayLast(Ref(FFillNan($close), {DL * 2})) - 1", "day_ret")
    add(f"(DayLast(Ref(FFillNan($high) - FFillNan($low), {DL * 2}))) / (DayLast(Ref(FFillNan($close), {DL * 2})) + {EPS})", "day_range")

    # 布林带 %B
    for p in [20, 40]:
        ma = f"Mean($close, {p})"
        std = f"Std($close, {p})"
        add(f"($close - ({ma} - 2 * {std})) / (4 * ({std} + {EPS}))", f"bb_pct_{p}")

    # 高低价差
    add(f"($high - $low) / ($close + {EPS})", "hl_ratio")
    add(f"($close - $open) / ($open + {EPS})", "co_ratio")

    # 均线偏离
    for p in [5, 10, 20, 40]:
        add(f"$close / (Mean($close, {p}) + {EPS}) - 1", f"ma_dev_{p}")

    # KDJ
    for p in [9, 14]:
        highest = f"Max($high, {p})"
        lowest = f"Min($low, {p})"
        rsv = f"100 * ($close - {lowest}) / ({highest} - {lowest} + {EPS})"
        k = f"Mean({rsv}, 3)"
        d = f"Mean({k}, 3)"
        add(k, f"kdj_k_{p}")
        add(d, f"kdj_d_{p}")
        add(f"3 * {k} - 2 * {d}", f"kdj_j_{p}")

    # 隔夜跳空
    add("($open / Ref($close, 1) - 1)", "overnight_gap")

    # 日内累计收益
    add(f"$close / DayLast(Ref(FFillNan($open), {DL * 2})) - 1", "intraday_ret")

    # 相对强弱
    for p in [5, 10, 20]:
        add(f"($close / Ref($close, 1) - 1) - (Mean($close, {p}) / Ref(Mean($close, {p}), 1) - 1)", f"rel_strength_{p}")

    # 价格百分位
    for p in [20, 40]:
        highest = f"Max($high, {p})"
        lowest = f"Min($low, {p})"
        add(f"($close - {lowest}) / ({highest} - {lowest} + {EPS})", f"price_pct_{p}")

    # 均线斜率
    for p in [5, 10, 20]:
        add(f"Mean($close, {p}) / Ref(Mean($close, {p}), 1) - 1", f"ma_slope_{p}")

    # 量价背离
    for p in [5, 10, 20]:
        add(f"($close / Ref($close, {p}) - 1) * ($volume / (Mean($volume, {p}) + {EPS}) - 1)", f"pv_div_{p}")

    # 高低价位置
    for p in [5, 10, 20]:
        add(f"($close - Min($low, {p})) / (Max($high, {p}) - Min($low, {p}) + {EPS})", f"hl_pos_{p}")

    return fields, names


# ============ Handler: 特征 + 波动率归一化标签 ============

from qlib.data.dataset.loader import QlibDataLoader
from qlib.data.dataset.handler import DataHandlerLP


class _IntradayHandler(DataHandlerLP):
    """分钟级特征 + 波动率归一化标签处理器

    标签: (close[t+h]/close[t] - 1) / rolling_volatility[t]
    波动率归一化让不同波动率股票的收益率可比, 提升模型稳定性
    """
    def __init__(self, horizon=3, quiet=False, top_features=None,
                 ic_features=None, label_type='cs_rank', **kwargs):
        self.horizon = horizon
        self.quiet = quiet
        self.label_type = label_type  # 'raw', 'vol_norm', 'cs_rank', 'binary'
        self.day_length = kwargs.pop('day_length', DAY_LENGTH)
        self._columns = kwargs.pop('columns', ['$open', '$high', '$low', '$close'])
        freq = kwargs.pop('freq', FREQ)
        kwargs.pop('fit_start_time', None)
        kwargs.pop('fit_end_time', None)

        feature_fields, feature_names = build_feature_expressions(self._columns, self.day_length)

        # 特征筛选
        if ic_features is not None:
            filtered = [(f, n) for f, n in zip(feature_fields, feature_names) if n in ic_features]
            if filtered:
                feature_fields, feature_names = zip(*filtered)
                feature_fields, feature_names = list(feature_fields), list(feature_names)
        elif top_features is not None:
            top_set = set(top_features)
            filtered = [(f, n) for f, n in zip(feature_fields, feature_names) if n in top_set]
            if filtered:
                feature_fields, feature_names = zip(*filtered)
                feature_fields, feature_names = list(feature_fields), list(feature_names)

        self._feature_fields = feature_fields
        self._feature_names = feature_names

        label_fields = ["$close"]
        label_names = ["LABEL0"]

        data_loader = QlibDataLoader(
            config={'feature': (feature_fields, feature_names),
                    'label': (label_fields, label_names)},
            swap_level=False, freq=freq,
        )
        DataHandlerLP.__init__(self, data_loader=data_loader, **kwargs)
        self._fix_labels()

    def _fix_labels(self):
        """将原始 $close 转为标签

        支持三种标签类型:
          - 'raw': 原始未来收益率 close[t+h]/close[t] - 1
          - 'vol_norm': 波动率归一化收益率
          - 'cs_rank': 截面排序 (每个时间点对所有股票排名, 归一化到 [0,1])
          - 'binary': 涨跌二分类 (涨=1, 跌=0)
        """
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

            close_series = df[label_key]
            inst_level = 'instrument' if 'instrument' in df.index.names else df.index.names[0]
            time_level = 'datetime' if 'datetime' in df.index.names else df.index.names[1]

            new_labels = close_series.copy()

            if self.label_type == 'cs_rank':
                # 截面排序: 先算每只股票的原始收益率, 再在每个时间点排序
                # Step 1: 逐股票算未来收益率
                raw_rets = close_series.copy()
                for inst, idx in close_series.groupby(level=inst_level).groups.items():
                    grp = close_series.loc[idx].sort_index(level=time_level)
                    vals = grp.values.astype(np.float64)
                    n = len(vals)
                    result = np.full(n, np.nan)
                    if n > self.horizon:
                        n_valid = n - self.horizon
                        future_close = vals[self.horizon:self.horizon + n_valid]
                        cur_close = vals[:n_valid]
                        mask = (cur_close > 0) & (future_close > 0)
                        result[:n_valid][mask] = future_close[mask] / cur_close[mask] - 1.0
                    raw_rets.loc[grp.index] = result

                # Step 2: 每个时间点截面排序
                for dt, dt_idx in raw_rets.groupby(level=time_level).groups.items():
                    dt_vals = raw_rets.loc[dt_idx]
                    valid = dt_vals.notna()
                    if valid.sum() < 5:
                        new_labels.loc[dt_idx] = 0.5
                        continue
                    ranks = dt_vals[valid].rank(pct=True)  # percentile rank [0, 1]
                    new_labels.loc[dt_idx[valid]] = ranks.values

                new_labels = new_labels.fillna(0.5)

            elif self.label_type == 'binary':
                # 逐股票算未来收益率, 然后二值化
                for inst, idx in close_series.groupby(level=inst_level).groups.items():
                    grp = close_series.loc[idx].sort_index(level=time_level)
                    vals = grp.values.astype(np.float64)
                    n = len(vals)
                    result = np.full(n, np.nan)
                    if n > self.horizon:
                        n_valid = n - self.horizon
                        future_close = vals[self.horizon:self.horizon + n_valid]
                        cur_close = vals[:n_valid]
                        mask = (cur_close > 0) & (future_close > 0)
                        rets = np.full(n_valid, np.nan)
                        rets[mask] = future_close[mask] / cur_close[mask] - 1.0
                        result[:n_valid] = (rets > 0).astype(float)  # 涨=1, 跌=0
                    new_labels.loc[grp.index] = result
                new_labels = new_labels.fillna(0.5)

            else:
                # 逐股票算未来收益率
                for inst, idx in close_series.groupby(level=inst_level).groups.items():
                    grp = close_series.loc[idx].sort_index(level=time_level)
                    vals = grp.values.astype(np.float64)
                    n = len(vals)
                    result = np.full(n, np.nan)
                    if n > self.horizon:
                        n_valid = n - self.horizon
                        future_close = vals[self.horizon:self.horizon + n_valid]
                        cur_close = vals[:n_valid]
                        mask = (cur_close > 0) & (future_close > 0)
                        rets = np.full(n_valid, np.nan)
                        rets[mask] = future_close[mask] / cur_close[mask] - 1.0

                        if self.label_type == 'vol_norm':
                            past_rets = np.full(n, np.nan)
                            for i in range(1, n):
                                if vals[i-1] > 0 and vals[i] > 0:
                                    past_rets[i] = vals[i] / vals[i-1] - 1.0
                            past_vol = pd.Series(past_rets).rolling(20, min_periods=5).std().values
                            vol_slice = past_vol[:n_valid]
                            vol_mask = ~np.isnan(vol_slice) & (vol_slice > 1e-8)
                            rets[vol_mask] = rets[vol_mask] / vol_slice[vol_mask]

                        result[:n_valid] = rets
                    new_labels.loc[grp.index] = result

                new_labels = new_labels.fillna(0.0)

            df[label_key] = new_labels
            df[label_key] = df[label_key].replace([float('inf'), float('-inf')], float('nan'))
            df[label_key] = df[label_key].fillna(0.0 if self.label_type in ('raw', 'vol_norm') else 0.5)

            # 清理特征中的 inf
            feat_cols = [c for c in df.columns if c != label_key]
            df[feat_cols] = df[feat_cols].replace([float('inf'), float('-inf')], float('nan'))

    def get_feature_config(self):
        return self._feature_fields, self._feature_names


# ============ 模型配置 ============

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
    'CatBoost': {
        'class': 'CatBoostModel', 'module_path': 'qlib.contrib.model.catboost_model',
        'kwargs': {
            'loss_function': 'RMSE', 'max_depth': 8,
            'learning_rate': 0.03, 'iterations': 5000, 'early_stopping_rounds': 200,
            'subsample': 0.7, 'colsample_bylevel': 0.6,
            'l2_leaf_reg': 5.0, 'min_data_in_leaf': 50,
            'random_seed': 42, 'thread_count': 4,
        }
    },
}

# HPO 搜索空间 (LightGBM)
HPO_SPACE = {
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


# ============ 工具函数 ============

def eval_signal(pred, label):
    """计算 IC / RankIC / ICIR"""
    pred = pred.values.flatten()
    label = label.values.flatten()
    mask = ~(np.isnan(pred) | np.isnan(label))
    pred, label = pred[mask], label[mask]
    if len(pred) < 10:
        return {'IC': np.nan, 'RankIC': np.nan, 'ICIR': np.nan, 'RankICIR': np.nan}
    ic = np.corrcoef(pred, label)[0, 1]
    from scipy.stats import spearmanr
    rank_ic = spearmanr(pred, label)[0]

    # ICIR: IC / std(IC) 按天分组
    pred_series = pd.Series(pred, index=pred.index[:len(pred)] if hasattr(pred, 'index') else None)
    label_series = pd.Series(label, index=label.index[:len(label)] if hasattr(label, 'index') else None)
    try:
        # 按天计算 IC
        daily_ic = []
        for dt, group in pred_series.groupby(level='datetime'):
            if dt in label_series.index.get_level_values('datetime'):
                l = label_series.loc[dt]
                if len(group) > 1 and len(l) > 1:
                    common_idx = group.index.intersection(l.index)
                    if len(common_idx) > 5:
                        d_ic = np.corrcoef(group[common_idx], l[common_idx])[0, 1]
                        if not np.isnan(d_ic):
                            daily_ic.append(d_ic)
        if len(daily_ic) > 1:
            daily_ic = np.array(daily_ic)
            icir = daily_ic.mean() / (daily_ic.std() + 1e-8)
        else:
            icir = np.nan
    except Exception:
        icir = np.nan

    return {'IC': float(ic), 'RankIC': float(rank_ic), 'ICIR': float(icir) if not np.isnan(icir) else np.nan,
            'RankICIR': float(icir) if not np.isnan(icir) else np.nan}


def create_dataset(horizon, quick=False, max_stocks=0, top_features=None, ic_features=None, label_type='cs_rank'):
    """创建 DatasetH 配置"""
    handler_kwargs = {
        'start_time': START_TIME, 'end_time': END_TIME,
        'fit_start_time': START_TIME, 'fit_end_time': TRAIN_END,
        'instruments': 'all', 'day_length': DAY_LENGTH, 'freq': FREQ,
        'columns': ['$open', '$high', '$low', '$close'],
        'horizon': horizon, 'quiet': True, 'label_type': label_type,
    }
    if quick:
        # 用 max_stocks 方式: 取前 50 只股票
        import tempfile
        all_path = os.path.expanduser(f'~/.qlib/qlib_data/cn_30min/bin/instruments/all.txt')
        with open(all_path) as f:
            lines = [next(f).strip() for _ in range(50)]
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        tmp.write('\n'.join(lines))
        tmp.close()
        handler_kwargs['instruments'] = tmp.name.replace('.txt', '')  # qlib 自动加 .txt
    if max_stocks > 0:
        import tempfile
        all_path = os.path.expanduser(f'~/.qlib/qlib_data/cn_30min/bin/instruments/all.txt')
        with open(all_path) as f:
            lines = [next(f).strip() for _ in range(max_stocks)]
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        tmp.write('\n'.join(lines))
        tmp.close()
        handler_kwargs['instruments'] = tmp.name.replace('.txt', '')  # qlib 自动加 .txt

    if top_features is not None:
        handler_kwargs['top_features'] = top_features
    if ic_features is not None:
        handler_kwargs['ic_features'] = ic_features

    return {
        'class': 'DatasetH', 'module_path': 'qlib.data.dataset',
        'kwargs': {
            'handler': {
                'class': '_IntradayHandler',
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


def train_one(model_name, model_config, horizon, quick=False, max_stocks=0,
              top_features=None, ic_features=None, label_type='cs_rank'):
    """训练单个模型, 返回 (pred, label, elapsed, metrics)"""
    t0 = time.time()
    ds_cfg = create_dataset(horizon, quick, max_stocks, top_features, ic_features, label_type)
    exp_name = f"unified_{model_name}_h{horizon}"

    with R.start(experiment_name=exp_name):
        model = init_instance_by_config(model_config)
        dataset = init_instance_by_config(ds_cfg)

        # 数据验证
        df_check = dataset.prepare('train', col_set=["feature", "label"])
        lab = df_check['label'].values
        lab_mean = float(np.nanmean(lab))
        lab_std = float(np.nanstd(lab))
        nan_count = int(np.isnan(lab).sum())
        if nan_count > df_check.shape[0] * 0.5:
            print(f"  ⚠️ {model_name} h={horizon}: NaN labels={nan_count}/{df_check.shape[0]}, 跳过")
            return None, None, 0, {'IC': np.nan, 'RankIC': np.nan, 'error': 'too_many_nan'}

        model.fit(dataset)
        elapsed = time.time() - t0

        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        pred = recorder.load_object("pred.pkl")
        label = dataset.prepare('test', col_set=["label"])

        metrics = eval_signal(pred, label)
        metrics['train_s'] = round(elapsed, 1)
        metrics['n_samples'] = df_check.shape[0]

        # 特征重要性
        fi = {}
        if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
            fnames = dataset.handler.get_feature_config()[1]
            fi = {n: float(v) for n, v in zip(fnames, model.model.feature_importances_)}

        return pred, label, elapsed, metrics, fi


# ============ 特征 IC 筛选 ============

def feature_ic_screening(horizon=1, quick=True, top_k=60, label_type='cs_rank'):
    """计算每个特征的 IC, 保留 |IC| 最高的 top_k 个"""
    print(f"\n🔍 特征 IC 筛选 (h={horizon}, {'csi300' if quick else 'all'}, label={label_type})...")

    all_fields, all_names = build_feature_expressions()
    print(f"   总特征数: {len(all_names)}")

    # 创建完整数据集
    ds_cfg = create_dataset(horizon, quick=quick, label_type=label_type)
    qlib.init(provider_uri=BIN_DIR, region=REG_CN, freq=FREQ,
              custom_ops=[Cut, DayLast, FFillNan, IsNull], expression_cache=None)
    dataset = init_instance_by_config(ds_cfg)

    # 获取训练集特征和标签
    df = dataset.prepare('train', col_set=["feature", "label"])
    feat_cols = [c for c in df.columns if c != ('label', 'LABEL0')]
    label = df[('label', 'LABEL0')].values

    ic_results = []
    for col in feat_cols:
        feat_vals = df[col].values
        mask = ~(np.isnan(feat_vals) | np.isnan(label))
        if mask.sum() < 100:
            ic_results.append((col[1], 0.0))
            continue
        ic = np.corrcoef(feat_vals[mask], label[mask])[0, 1]
        ic_results.append((col[1], abs(ic) if not np.isnan(ic) else 0.0))

    ic_results.sort(key=lambda x: -x[1])
    selected = [f[0] for f in ic_results[:top_k]]

    print(f"   保留 Top-{top_k} 特征:")
    for i, (name, ic) in enumerate(ic_results[:top_k]):
        print(f"   {i+1:>2}. {name:<25s} |IC|={ic:.4f}")
    print(f"   丢弃 {len(all_names) - top_k} 个 |IC|<{ic_results[top_k-1][1]:.4f} 的特征")

    return selected, ic_results


# ============ HPO 超参数搜索 ============

def hpo_search(horizon, quick=False, n_trials=10, label_type='cs_rank'):
    """随机搜索 LightGBM 超参数"""
    print(f"\n🔧 HPO 超参数搜索 (h={horizon}, {n_trials} trials)...")

    results = []
    for i in range(n_trials):
        params = {}
        for k, v in HPO_SPACE.items():
            params[k] = float(np.random.choice(v)) if isinstance(v[0], float) else int(np.random.choice(v))

        model_cfg = {
            'class': 'LGBModel', 'module_path': 'qlib.contrib.model.gbdt',
            'kwargs': {
                'loss': 'mse', 'n_estimators': 5000, 'early_stopping_rounds': 200,
                'verbosity': -1, 'seed': 42, 'n_jobs': 4, **params,
            }
        }

        try:
            pred, label, elapsed, metrics, _ = train_one(
                'LightGBM', model_cfg, horizon, quick=quick, label_type=label_type
            )
            results.append({
                'trial': i, 'IC': metrics['IC'], 'RankIC': metrics['RankIC'],
                'train_s': metrics['train_s'], 'params': params,
            })
            print(f"   Trial {i+1}/{n_trials}: IC={metrics['IC']:.4f} RankIC={metrics['RankIC']:.4f} "
                  f"lr={params['learning_rate']} leaves={params['num_leaves']}")
        except Exception as e:
            print(f"   Trial {i+1}/{n_trials}: ❌ {e}")

    results.sort(key=lambda x: -abs(x['RankIC']))
    best = results[0] if results else None
    if best:
        print(f"   🏆 最优: IC={best['IC']:.4f} RankIC={best['RankIC']:.4f}")
        print(f"   参数: {json.dumps(best['params'])}")
    return results


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(description='统一训练脚本')
    parser.add_argument('--models', default='LightGBM,XGBoost,CatBoost',
                        help='模型列表, 逗号分隔 (默认: LightGBM,XGBoost,CatBoost)')
    parser.add_argument('--horizons', default='1,3,5,10',
                        help='预测周期, 逗号分隔 (默认: 1,3,5,10)')
    parser.add_argument('--quick', action='store_true', help='快速验证 (csi300)')
    parser.add_argument('--max-stocks', type=int, default=0, help='限制股票数')
    parser.add_argument('--no-hpo', action='store_true', help='跳过 HPO')
    parser.add_argument('--no-ensemble', action='store_true', help='跳过集成')
    parser.add_argument('--no-feature-ic', action='store_true', help='跳过特征 IC 筛选')
    parser.add_argument('--feature-top-k', type=int, default=60, help='特征 IC 筛选保留数')
    parser.add_argument('--label-type', default='binary', choices=['raw', 'vol_norm', 'cs_rank', 'binary'],
                        help='标签类型: raw=原始收益率, vol_norm=波动率归一化, cs_rank=截面排序, binary=涨跌分类')
    parser.add_argument('--hpo-trials', type=int, default=10, help='HPO 搜索次数')
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(',')]
    horizons = [int(h) for h in args.horizons.split(',')]
    label_type = args.label_type

    print(f"{'='*70}")
    print(f" 🚀 统一训练: {', '.join(model_names)} | h={horizons}")
    print(f"   标签类型: {label_type}")
    print(f"   特征IC筛选: {'✗' if args.no_feature_ic else f'Top-{args.feature_top_k}'}")
    print(f"   HPO: {'✗' if args.no_hpo else f'{args.hpo_trials} trials'}")
    if args.quick:
        print(f"   ⚡ 快速模式 (csi300)")
    print(f"{'='*70}")

    qlib.init(provider_uri=BIN_DIR, region=REG_CN, freq=FREQ,
              custom_ops=[Cut, DayLast, FFillNan, IsNull], expression_cache=None)

    # ── Step 1: 特征 IC 筛选 ──
    ic_features = None
    if not args.no_feature_ic:
        ic_features, ic_all = feature_ic_screening(
            horizon=1, quick=args.quick, top_k=args.feature_top_k, label_type=label_type
        )
        print(f"\n✅ 特征 IC 筛选完成, 保留 {len(ic_features)} 个特征")

    # ── Step 2: HPO 超参数搜索 ──
    if not args.no_hpo:
        best_hpo_params = {}
        for h in horizons:
            results = hpo_search(h, quick=args.quick, n_trials=args.hpo_trials, label_type=label_type)
            if results:
                best_hpo_params[h] = results[0]['params']
                # 更新 LightGBM 配置
                MODEL_CONFIGS['LightGBM']['kwargs'].update(best_hpo_params[h])
        print(f"\n✅ HPO 完成")

    # ── Step 3: 训练所有模型 × 所有周期 ──
    all_results = {}  # {(model, horizon): metrics}
    all_predictions = {}  # {(model, horizon): pred}
    all_labels = {}  # {horizon: label}

    for h in horizons:
        for m_name in model_names:
            if m_name not in MODEL_CONFIGS:
                print(f"⚠️ 未知模型: {m_name}, 跳过")
                continue

            print(f"\n▶ 训练 {m_name} h={h}...")
            pred, label, elapsed, metrics, fi = train_one(
                m_name, MODEL_CONFIGS[m_name], h,
                quick=args.quick, max_stocks=args.max_stocks,
                ic_features=ic_features, label_type=label_type,
            )

            if pred is None:
                continue

            all_results[(m_name, h)] = metrics
            all_predictions[(m_name, h)] = pred
            if h not in all_labels:
                all_labels[h] = label

            ic_str = f"IC={metrics['IC']:.4f}" if not np.isnan(metrics['IC']) else "IC=NaN"
            ric_str = f"RankIC={metrics['RankIC']:.4f}" if not np.isnan(metrics['RankIC']) else "RankIC=NaN"
            print(f"  {m_name} h={h}: {ic_str} {ric_str} ({metrics['train_s']:.0f}s)")

    if not all_results:
        print("❌ 没有成功的训练结果")
        sys.exit(1)

    # ── Step 4: 加权集成 ──
    if not args.no_ensemble and len(model_names) > 1:
        for h in horizons:
            h_preds = {m: all_predictions.get((m, h)) for m in model_names if (m, h) in all_predictions}
            if len(h_preds) < 2:
                continue

            # 按验证集 RankIC 绝对值加权
            weights = {}
            for m_name in h_preds:
                ric = abs(all_results[(m_name, h)].get('RankIC', 0))
                weights[m_name] = max(ric, 0.001)  # 避免零权重

            total_w = sum(weights.values())
            weights = {k: v / total_w for k, v in weights.items()}

            ensemble_pred = sum(h_preds[m] * weights[m] for m in h_preds)
            ensemble_metrics = eval_signal(ensemble_pred, all_labels[h])
            ensemble_metrics['train_s'] = sum(all_results[(m, h)]['train_s'] for m in h_preds)
            all_results[('Ensemble', h)] = ensemble_metrics

            print(f"\n─── 集成 h={h} (权重: {', '.join(f'{m}={w:.2f}' for m, w in weights.items())}) ───")
            print(f"  Ensemble: IC={ensemble_metrics['IC']:.4f} RankIC={ensemble_metrics['RankIC']:.4f}")

    # ── Step 5: 结果汇总 ──
    print(f"\n{'='*70}")
    print(f" 📊 训练结果汇总")
    print(f"{'='*70}")
    header = f"{'模型':<12} {'h':>3} {'IC':>8} {'RankIC':>8} {'|IC|':>7} {'|RkIC|':>7} {'ICIR':>7} {'耗时':>6}"
    print(header)
    print(f"{'-'*65}")

    # 按模型和horizon分组
    all_models = model_names + (['Ensemble'] if not args.no_ensemble else [])
    for h in horizons:
        for m in all_models:
            key = (m, h)
            if key not in all_results:
                continue
            r = all_results[key]
            ic = r['IC']
            ric = r['RankIC']
            icir = r.get('ICIR', np.nan)
            print(f"{m:<12} {h:>3} {ic:>8.4f} {ric:>8.4f} {abs(ic):>7.4f} {abs(ric):>7.4f} "
                  f"{icir if not np.isnan(icir) else 0:>7.2f} {r['train_s']:>5.0f}s")

    # 找最优
    best_key = max(all_results, key=lambda k: abs(all_results[k]['RankIC']))
    best_ric = abs(all_results[best_key]['RankIC'])
    print(f"\n🏆 最优: {best_key[0]} h={best_key[1]} (|RankIC|={best_ric:.4f})")

    # ── Step 6: 保存结果 ──
    out_dir = os.path.join(ROOT, 'experiments')
    os.makedirs(out_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    # CSV
    csv_path = os.path.join(out_dir, f'results_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'horizon', 'ic', 'rank_ic', 'icir', 'rank_icir', 'train_s', 'label_type'])
        for (m, h), r in all_results.items():
            writer.writerow([m, h, r['IC'], r['RankIC'], r.get('ICIR', ''), r.get('RankICIR', ''),
                             r['train_s'], label_type])
    print(f"\n💾 结果已保存: {csv_path}")

    # 特征 IC 结果
    if ic_features is not None:
        ic_path = os.path.join(out_dir, f'feature_ic_{timestamp}.csv')
        with open(ic_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['feature', 'abs_ic'])
            writer.writerows(ic_all)
        print(f"💾 特征 IC: {ic_path}")


if __name__ == '__main__':
    main()