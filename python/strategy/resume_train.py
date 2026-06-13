#!/usr/bin/env python3
"""续跑脚本 — 跳过Optuna和CV，用已有参数完成特征选择+保存

用法:
  python strategy/resume_train.py --model daily --params '{"num_leaves":213,...}'
  python strategy/resume_train.py --model 30m --params '{"num_leaves":150,...}'

或者把参数写进 strategy/best_params.json，格式:
  {"daily": {...}, "30m": {...}}
"""

import sys, os, argparse, pickle, json, sqlite3, warnings
import numpy as np
import lightgbm as lgb
from datetime import datetime
from sklearn.feature_selection import SelectFromModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategy.train import load_data, prepare_data

warnings.filterwarnings('ignore')

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/stock_data.db')

CONFIGS = {
    'daily': {
        'db_table': 'kline_daily', 'model_dir': 'models/lgb_daily',
        'label': '日线', 'horizon': 5, 'early_stopping': 80,
        'role': 'α选股层', 'cv_spearman': 0.3862, 'cv_rmse': 0.0442, 'cv_mae': 0.0327,
    },
    '30m': {
        'db_table': 'kline_30m', 'model_dir': 'models/lgb_30m',
        'label': '30分钟', 'horizon': 3, 'early_stopping': 80,
        'role': 'γ择时层',
        # 以下需要替换为30m模型的实际CV结果
        'cv_spearman': None, 'cv_rmse': None, 'cv_mae': None,
    },
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['daily', '30m'], default='daily')
    parser.add_argument('--params', type=str, default='',
                        help='JSON字符串, 例如: \'{"num_leaves":213,...}\'')
    parser.add_argument('--cvs', type=str, default='',
                        help='CV结果JSON: \'{"spearman":0.38,"rmse":0.044,"mae":0.033}\'')
    args = parser.parse_args()

    cfg = CONFIGS[args.model]

    # --- 加载最优参数 ---
    if args.params:
        best_params = json.loads(args.params)
    else:
        # 尝试从文件加载
        params_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_params.json')
        if os.path.exists(params_file):
            with open(params_file) as f:
                all_params = json.load(f)
            best_params = all_params.get(args.model, {})
        else:
            print("❌ 请用 --params 传入最优参数, 或创建 strategy/best_params.json")
            return

    if not best_params:
        print(f"❌ 未找到 {args.model} 的最优参数")
        return

    # 填充 LGBM 固定参数
    best_params.setdefault('n_estimators', 1500)
    best_params.setdefault('verbosity', -1)
    best_params.setdefault('random_state', 42)
    best_params.setdefault('force_row_wise', True)
    best_params.setdefault('num_threads', -1)

    # --- 覆盖CV结果 ---
    if args.cvs:
        cvs = json.loads(args.cvs)
        cfg['cv_spearman'] = cvs.get('spearman', cfg['cv_spearman'])
        cfg['cv_rmse'] = cvs.get('rmse', cfg['cv_rmse'])
        cfg['cv_mae'] = cvs.get('mae', cfg['cv_mae'])

    print("=" * 60)
    print(f" 续跑 {cfg['label']} 模型 (跳过Optuna + CV)")
    print(f" 参数来源: {'--params' if args.params else 'best_params.json'}")
    print("=" * 60)

    # --- 加载数据 ---
    print("\n加载数据...")
    data = load_data(DB_PATH, cfg['db_table'])
    conn = sqlite3.connect(DB_PATH)
    X, y, feature_names = prepare_data(data, cfg, conn)
    conn.close()

    if X is None:
        print("❌ 数据准备失败"); return

    print(f"数据: {len(X)}条, {X.shape[1]}特征")

    # --- 去冗余 ---
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

    # --- SelectFromModel ---
    v2 = int(len(X) * 0.8)
    Xp = X[:v2]
    sel = lgb.LGBMRegressor(**best_params)
    sel.fit(Xp[:-50], y[:v2][:-50], eval_set=[(Xp[-50:], y[:v2][-50:])],
            callbacks=[lgb.early_stopping(30, verbose=False)])
    sf = SelectFromModel(sel, threshold='median', prefit=True)
    X = sf.transform(X)
    feature_names = [fn for fn, m in zip(feature_names, sf.get_support()) if m]
    print(f"特征选择: {len(feature_names)} 个")

    # --- 最终模型 ---
    print("\n训练最终模型...")
    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X, y)

    imp = final_model.feature_importances_
    top = np.argsort(imp)[-20:][::-1]
    print(f"\nTop 20 特征:")
    for idx in top:
        print(f"  {feature_names[idx]}: {imp[idx]:.0f}")

    # --- 保存 ---
    d = cfg['model_dir']
    os.makedirs(d, exist_ok=True)

    core_params = {k: v for k, v in best_params.items()
                   if k not in ('verbosity', 'random_state', 'force_row_wise', 'num_threads', 'n_estimators')}

    model_data = {
        'model': final_model, 'feature_names': feature_names,
        'best_params': core_params,
        'cv_spearman': cfg['cv_spearman'],
        'cv_rmse': cfg['cv_rmse'],
        'cv_mae': cfg['cv_mae'],
        'horizon': cfg['horizon'],
        'n_features': len(feature_names),
        'n_samples': len(X),
    }
    with open(os.path.join(d, 'model.pkl'), 'wb') as f:
        pickle.dump(model_data, f)
    size = os.path.getsize(os.path.join(d, 'model.pkl')) / 1024 / 1024

    meta = {
        'model_type': args.model, 'label': cfg['label'],
        'horizon': model_data['horizon'],
        'n_features': model_data['n_features'],
        'n_samples': model_data['n_samples'],
        'cv_spearman': model_data['cv_spearman'],
        'cv_rmse': model_data['cv_rmse'],
        'cv_mae': model_data['cv_mae'],
        'best_params': model_data['best_params'],
        'feature_names': model_data['feature_names'][:50],
        'trained_at': datetime.now().isoformat(),
        'role': cfg['role'],
    }
    with open(os.path.join(d, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 模型已保存: {d}/model.pkl ({size:.1f} MB)")
    if cfg['cv_spearman']:
        print(f"  Rank IC: {cfg['cv_spearman']:.4f}")

if __name__ == '__main__':
    main()