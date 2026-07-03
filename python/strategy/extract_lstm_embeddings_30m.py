#!/usr/bin/env python3
"""提取 LSTM embeddings — 仅覆盖 kline_30m 中的股票 (372只, ~3分钟)"""

import os, sys, pickle, sqlite3, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from lstm_encoder import LSTMPredictor, compute_lstm_inputs, SEQ_LEN, MODEL_DIR, DB_PATH

def main():
    model_path = os.path.join(MODEL_DIR, 'encoder.pt')
    stats_path = os.path.join(MODEL_DIR, 'norm_stats.pt')
    output_path = os.path.join(os.path.dirname(DB_PATH), 'lstm_embeddings.pkl')

    print("🔮 提取 LSTM embeddings (30分钟股票范围)")
    print(f"   模型: {model_path}")
    print(f"   输出: {output_path}")

    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    input_dim = state_dict['lstm.weight_ih_l0'].shape[1]
    model = LSTMPredictor(input_dim=input_dim, hidden_dim=64)
    model.load_state_dict(state_dict)
    model.eval()

    norm_stats = torch.load(stats_path, map_location='cpu', weights_only=False)

    conn = sqlite3.connect(DB_PATH)
    # 只覆盖 kline_30m 中的股票
    symbols = [r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM kline_30m ORDER BY symbol"
    ).fetchall()]
    print(f"   {len(symbols)} 只股票 (kline_30m)")

    embeddings = {}
    t0 = time.time()
    n_processed = 0
    n_total_emb = 0

    for sym in symbols:
        try:
            df = pd.read_sql(
                "SELECT date, open, high, low, close, volume FROM kline_daily "
                "WHERE symbol=? ORDER BY date", conn, params=(sym,))
            if len(df) < SEQ_LEN + 20:
                continue
            df['date'] = pd.to_datetime(df['date'], format='mixed')

            feats = compute_lstm_inputs(df).fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)

            mean = np.squeeze(norm_stats['x_mean'])
            std = np.squeeze(norm_stats['x_std'])
            feats = np.nan_to_num((feats - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)

            emb_dict = {}
            seqs = []
            date_list = []
            for i in range(SEQ_LEN, len(feats)):
                seq = feats[i - SEQ_LEN:i]
                if np.isnan(seq).any():
                    continue
                seqs.append(seq)
                date_list.append(str(df['date'].iloc[i])[:10])

            if seqs:
                batch = torch.FloatTensor(np.array(seqs))
                with torch.no_grad():
                    embs = model.encode(batch)
                for d, e in zip(date_list, embs):
                    emb_dict[d] = e.astype(np.float32) if hasattr(e, 'astype') else np.array(e.cpu()).astype(np.float32)

            embeddings[sym] = emb_dict
            n_processed += 1
            n_total_emb += len(emb_dict)

            if n_processed % 50 == 0:
                elapsed = time.time() - t0
                print(f"   [{n_processed}/{len(symbols)}] {elapsed:.0f}s, {n_total_emb:,} embeddings")

        except Exception as e:
            if n_processed < 5:
                print(f"   ⚠️ {sym}: {e}")
            continue

    conn.close()

    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    elapsed = time.time() - t0
    print(f"\n✅ 完成! {elapsed:.0f}s, {n_processed} 只股票, {n_total_emb:,} embeddings")
    print(f"   文件: {output_path} ({os.path.getsize(output_path)/1024/1024:.1f} MB)")

if __name__ == '__main__':
    main()