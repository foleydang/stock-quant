#!/usr/bin/env python3
"""轻量级 LSTM embedding 提取 — 内存友好版本"""

import os, sys, pickle, sqlite3, time, gc
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
torch.set_num_threads(1)  # 单线程避免内存竞争

from lstm_encoder import LSTMPredictor, compute_lstm_inputs, SEQ_LEN, MODEL_DIR, DB_PATH

def main():
    model_path = os.path.join(MODEL_DIR, 'encoder.pt')
    stats_path = os.path.join(MODEL_DIR, 'norm_stats.pt')
    output_path = os.path.join(os.path.dirname(DB_PATH), 'lstm_embeddings.pkl')

    print("🔮 提取 LSTM embeddings (轻量模式)")
    
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    input_dim = state_dict['lstm.weight_ih_l0'].shape[1]
    model = LSTMPredictor(input_dim=input_dim, hidden_dim=64)
    model.load_state_dict(state_dict)
    model.eval()

    norm_stats = torch.load(stats_path, map_location='cpu', weights_only=False)
    mean = np.squeeze(norm_stats['x_mean'])
    std = np.squeeze(norm_stats['x_std'])

    # 分批处理，每批50只股票
    conn = sqlite3.connect(DB_PATH)
    symbols = [r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM kline_30m ORDER BY symbol"
    ).fetchall()]
    conn.close()
    
    print(f"   {len(symbols)} 只股票，分批处理")
    
    embeddings = {}
    t0 = time.time()
    n_processed = 0
    batch_size = 50

    for batch_start in range(0, len(symbols), batch_size):
        batch = symbols[batch_start:batch_start + batch_size]
        conn = sqlite3.connect(DB_PATH)
        
        for sym in batch:
            try:
                df = pd.read_sql(
                    "SELECT date, open, high, low, close, volume FROM kline_daily "
                    "WHERE symbol=? ORDER BY date", conn, params=(sym,))
                if len(df) < SEQ_LEN + 20:
                    continue
                df['date'] = pd.to_datetime(df['date'], format='mixed')

                feats = compute_lstm_inputs(df).fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
                feats = np.nan_to_num((feats - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)

                seqs = []
                date_list = []
                for i in range(SEQ_LEN, len(feats)):
                    seq = feats[i - SEQ_LEN:i]
                    if np.isnan(seq).any():
                        continue
                    seqs.append(seq)
                    date_list.append(str(df['date'].iloc[i])[:10])

                if seqs:
                    batch_tensor = torch.FloatTensor(np.array(seqs))
                    with torch.no_grad():
                        embs = model.encode(batch_tensor)
                    emb_dict = {}
                    for d, e in zip(date_list, embs):
                        emb_dict[d] = e.astype(np.float32) if hasattr(e, 'astype') else np.array(e).astype(np.float32)
                    embeddings[sym] = emb_dict

                n_processed += 1
                del df, feats, seqs, date_list
                
            except Exception as e:
                if n_processed < 5:
                    print(f"   ⚠️ {sym}: {e}")
                continue

        conn.close()
        gc.collect()
        
        elapsed = time.time() - t0
        print(f"   [{n_processed}/{len(symbols)}] {elapsed:.0f}s, {sum(len(v) for v in embeddings.values()):,} embeddings, mem={len(pickle.dumps(embeddings))//1024}KB")

    # 保存
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    elapsed = time.time() - t0
    total_emb = sum(len(v) for v in embeddings.values())
    print(f"\n✅ 完成! {elapsed:.0f}s, {len(embeddings)} 只股票, {total_emb:,} embeddings")
    print(f"   文件: {output_path} ({os.path.getsize(output_path)/1024/1024:.1f} MB)")

if __name__ == '__main__':
    main()