#!/usr/bin/env python3
"""
LGBM 统一训练脚本 — 回归 + 截面排序

双层架构:
  日线模型 → 预测N日收益率 → 截面排序选股 (α层)
  30分钟模型 → 预测90分钟收益率 → 截面排序择时 (γ层)

评估指标: Rank IC (Spearman 排序相关性)

用法:
  python strategy/train.py --model daily
  python strategy/train.py --model 30m
  python strategy/train.py --model daily --quick
"""

import sys, os, argparse, pickle, json, sqlite3, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategy.features import EnhancedFeatureEngineer, AdvancedFeatureEngineer, MarketFeatureEngineer

warnings.filterwarnings('ignore')
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/stock_data.db')

# ============ 配置 ============
CONFIG_30M = {
    'db_table': 'kline_30m', 'model_dir': 'models/lgb_30m', 'label': '30分钟',
    'horizon': 3, 'min_history': 150, 'min_samples': 200,
    'n_estimators': 2000, 'early_stopping': 80,
    'search_sample': 0.25, 'search_estimators': 500,
    'optuna_trials': 100,
    'features': 'enhanced+advanced',  # 纯30分钟特征
    'purged_gap': 3,
}

CONFIG_DAILY = {
    'db_table': 'kline_daily', 'model_dir': 'models/lgb_daily', 'label': '日线',
    'horizon': 5, 'min_history': 120, 'min_samples': 200,
    'n_estimators': 1500, 'early_stopping': 80,
    'search_sample': 0.5, 'search_estimators': 500,
    'optuna_trials': 100,
    'features': 'enhanced+advanced+market',  # 日线含北向资金+情绪
    'purged_gap': 1,
}


# ============ 特征 ============
def compute_features(df: pd.DataFrame, symbol: str, cfg: dict) -> pd.DataFrame:
    base = EnhancedFeatureEngineer.calculate_features(df)
    adv = AdvancedFeatureEngineer.calculate_advanced_features(df)
    feats = pd.concat([base, adv], axis=1)
    if 'market' in cfg['features']:
        market = MarketFeatureEngineer.calculate_market_features(df, symbol=symbol)
        feats = pd.concat([feats, market], axis=1)
    time_cols = ['day_of_week', 'day_of_month', 'is_month_end', 'is_month_start',
                 'hour', 'minute', 'is_morning', 'is_afternoon']
    return feats.drop(columns=[c for c in time_cols if c in feats.columns], errors='ignore')


def load_sentiment(conn) -> pd.DataFrame:
    try:
        df = pd.read_sql("SELECT symbol, trade_date as date, lhb_flag, lhb_net_buy, "
                         "lhb_net_buy_ratio, lhb_ret_5d, is_limit_up, is_limit_down, "
                         "vol_ratio_20, abnormal_ret, consecutive_limit_up FROM sentiment_daily", conn)
        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'], format='mixed').dt.strftime('%Y-%m-%d')
            return df
    except Exception:
        pass
    return pd.DataFrame()


# ============ 数据加载 ============
def load_data(db_path: str, table: str) -> Dict[str, pd.DataFrame]:
    conn = sqlite3.connect(db_path)
    symbols = [r[0] for r in conn.execute(f"SELECT DISTINCT symbol FROM {table}")]
    data = {}
    for sym in symbols:
        try:
            df = pd.read_sql(f"SELECT * FROM {table} WHERE symbol=? ORDER BY date", conn, params=(sym,))
            if len(df) >= 120:
                df['date'] = pd.to_datetime(df['date'], format='mixed')
                df = df.sort_values('date').reset_index(drop=True)
                data[sym] = df
        except Exception:
            continue
    conn.close()
    print(f"加载了 {len(data)} 只股票 (表: {table})")
    return data


def prepare_data(data: Dict, cfg: dict, conn) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """准备回归训练数据 — 目标为连续收益率"""
    X_list, y_list, feature_names = [], [], None
    sent_df = load_sentiment(conn)
    has_sent = len(sent_df) > 0
    if has_sent:
        print(f" 含情绪特征")
    success = 0
    horizon = cfg['horizon']

    for i, (sym, df) in enumerate(data.items()):
        try:
            feats = compute_features(df, sym, cfg)
            if feature_names is None:
                feature_names = list(feats.columns)

            close = df['close'].values.astype(float)
            target = np.full(len(close), np.nan)
            for j in range(len(close) - horizon):
                target[j] = (close[j + horizon] - close[j]) / close[j]

            # 合并情绪特征
            if has_sent:
                dates = df['date'].dt.strftime('%Y-%m-%d')
                sent = sent_df[sent_df['symbol'] == sym].set_index('date')
                for col in sent.columns:
                    if col not in ('symbol', 'date'):
                        feats[f'sent_{col}'] = dates.map(lambda d: sent.loc[d, col] if d in sent.index else 0).fillna(0).values
                feature_names = list(feats.columns)

            feats = feats.fillna(method='ffill').fillna(0)
            valid = ~np.isnan(target)
            feats_v, target_v = feats[valid], target[valid]
            if len(feats_v) > cfg['min_history']:
                feats_v, target_v = feats_v.iloc[cfg['min_history']:], target_v[cfg['min_history']:]
            if len(feats_v) > 50:
                X_list.append(feats_v.values)
                y_list.append(target_v)
                success += 1
        except Exception:
            continue
        if (i + 1) % 100 == 0:
            print(f"  处理 {i+1}/{len(data)} 只 (成功{success})")

    if not X_list:
        return None, None, None
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    valid = np.abs(y) < 0.15
    X, y = X[valid], y[valid]

    print(f"\n训练数据: {len(X):,}条, 特征: {len(feature_names)}, "
          f"目标: mean={y.mean():.4f}, std={y.std():.4f}")
    return X, y, feature_names


# ============ 模型训练 ============
def train(X: np.ndarray, y: np.ndarray, feature_names: List[str],
          cfg: dict, quick: bool = False) -> Dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_trials = 20 if quick else cfg['optuna_trials']
    tscv = TimeSeriesSplit(n_splits=5, gap=cfg['purged_gap'])

    print(f"\nOptuna超参搜索 ({n_trials}次, "
          f"5折PurgedCV(gap={cfg['purged_gap']}), 目标=Spearman)...")

    # 搜索用80%数据
    split = int(len(X) * 0.8)
    X_s, y_s = X[:split], y[:split]

    def objective(trial):
        p = {
            'objective': 'regression_l1', 'metric': 'mae',
            'boosting_type': 'gbdt', 'verbosity': -1, 'n_jobs': -1,
            'random_state': 42, 'n_estimators': cfg['search_estimators'],
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.95),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        }
        # 采样加速
        if cfg['search_sample'] < 1.0 and len(X_s) > 100000:
            n = int(len(X_s) * cfg['search_sample'])
            idx = np.random.RandomState(42 + trial.number).choice(len(X_s), n, replace=False)
            Xt, yt = X_s[idx], y_s[idx]
        else:
            Xt, yt = X_s, y_s

        scores = []
        for tr, te in TimeSeriesSplit(n_splits=3, gap=cfg['purged_gap']).split(Xt):
            m = lgb.LGBMRegressor(**p)
            m.fit(Xt[tr], yt[tr], eval_set=[(Xt[te], yt[te])],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
            pred = m.predict(Xt[te])
            if len(pred) > 2:
                corr, _ = spearmanr(pred, yt[te])
                scores.append(corr if not np.isnan(corr) else 0)
            else:
                scores.append(0)
        return np.mean(scores) if scores else 0

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_val = study.best_value
    print(f"\n最优Spearman: {best_val:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # ====== 5折CV ======
    final_p = {
        'objective': 'regression_l1', 'metric': 'mae',
        'boosting_type': 'gbdt', 'verbosity': -1, 'n_jobs': -1,
        'random_state': 42, 'n_estimators': cfg['n_estimators'],
    }
    final_p.update(study.best_params)

    print(f"\n5折Purged交叉验证...")
    cv_s, cv_r, cv_m, models = [], [], [], []
    for fold, (tr, te) in enumerate(tscv.split(X)):
        m = lgb.LGBMRegressor(**final_p)
        m.fit(X[tr], y[tr], eval_set=[(X[te], y[te])],
              callbacks=[lgb.early_stopping(cfg['early_stopping'], verbose=False)])
        pred = m.predict(X[te])
        rmse = np.sqrt(mean_squared_error(y[te], pred))
        mae = mean_absolute_error(y[te], pred)
        corr, _ = spearmanr(pred, y[te])
        if np.isnan(corr): corr = 0
        cv_s.append(corr); cv_r.append(rmse); cv_m.append(mae); models.append(m)
        print(f"  Fold {fold+1}: Spearman={corr:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    avg_s, avg_r, avg_m = np.mean(cv_s), np.mean(cv_r), np.mean(cv_m)
    print(f"\n平均: Spearman={avg_s:.4f}, RMSE={avg_r:.4f}, MAE={avg_m:.4f}")

    # ====== 特征选择 ======
    if len(feature_names) > 20:
        # 去冗余 (先做，再切 Xp)
        v = int(len(X) * 0.8)
        Xv = X[:v]
        cm = np.corrcoef(Xv.T)
        rm = set()
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                if abs(cm[i, j]) > 0.95 and i not in rm and j not in rm:
                    rm.add(j)
        if rm:
            keep = np.ones(len(feature_names), dtype=bool)
            keep[list(rm)] = False
            X, feature_names = X[:, keep], [fn for fn, m in zip(feature_names, keep) if m]
            print(f"特征去冗余: {sum(keep)}/{sum(keep)+len(rm)}")

        # SelectFromModel (基于去冗余后的 X)
        v2 = int(len(X) * 0.8)
        Xp = X[:v2]
        sel = lgb.LGBMRegressor(**final_p)
        sel.fit(Xp[:-50], y[:v2][:-50], eval_set=[(Xp[-50:], y[:v2][-50:])],
                callbacks=[lgb.early_stopping(30, verbose=False)])
        sf = SelectFromModel(sel, threshold='median', prefit=True)
        X = sf.transform(X)
        feature_names = [fn for fn, m in zip(feature_names, sf.get_support()) if m]
        print(f"特征选择: {len(feature_names)} 个")

    # 最终模型
    final_model = models[-1]
    imp = final_model.feature_importances_
    top = np.argsort(imp)[-20:][::-1]
    print(f"\nTop 20 特征:")
    for idx in top:
        print(f"  {feature_names[idx]}: {imp[idx]:.0f}")

    return {
        'model': final_model, 'feature_names': feature_names,
        'best_params': study.best_params,
        'cv_spearman': round(avg_s, 4), 'cv_rmse': round(avg_r, 4),
        'cv_mae': round(avg_m, 4), 'horizon': cfg['horizon'],
        'n_features': len(feature_names), 'n_samples': len(X),
    }


def save(model_data: Dict, cfg: dict, model_type: str):
    d = cfg['model_dir']; os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'model.pkl'), 'wb') as f:
        pickle.dump(model_data, f)
    size = os.path.getsize(os.path.join(d, 'model.pkl')) / 1024 / 1024

    meta = {
        'model_type': model_type, 'label': cfg['label'],
        'horizon': model_data['horizon'],
        'n_features': model_data['n_features'],
        'n_samples': model_data['n_samples'],
        'cv_spearman': model_data['cv_spearman'],
        'cv_rmse': model_data['cv_rmse'],
        'cv_mae': model_data['cv_mae'],
        'best_params': model_data['best_params'],
        'feature_names': model_data['feature_names'][:50],
        'trained_at': datetime.now().isoformat(),
        'role': 'α选股层' if model_type == 'daily' else 'γ择时层',
    }
    with open(os.path.join(d, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 模型已保存: {d}/model.pkl ({size:.1f} MB)")
    print(f"  Rank IC: {model_data['cv_spearman']:.4f}")


# ============ 主入口 ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['daily', '30m'], default='30m')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    cfg = CONFIG_DAILY if args.model == 'daily' else CONFIG_30M

    print("=" * 60)
    print(f" LGBM {cfg['label']}模型训练 (回归 + 截面排序)")
    print(f" 预测未来{cfg['horizon']}根K线收益率 | 评估: Rank IC")
    print("=" * 60)

    print(f"\n数据库: {DB_PATH}")
    data = load_data(DB_PATH, cfg['db_table'])

    conn = sqlite3.connect(DB_PATH)
    X, y, fn = prepare_data(data, cfg, conn)
    conn.close()

    if X is None:
        print("❌ 数据准备失败"); return

    md = train(X, y, fn, cfg, quick=args.quick)
    save(md, cfg, args.model)

    print("\n" + "=" * 60)
    print(f" 🎉 完成! Rank IC: {md['cv_spearman']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()