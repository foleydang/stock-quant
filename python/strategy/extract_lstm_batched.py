#!/usr/bin/env python3
"""LSTM embedding 提取 — 分批写入磁盘，内存友好"""

import os, sys, pickle, sqlite3, time, gc, glob, tempfile
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
torch.set_num_threads(1)

from lstm_encoder import LSTMPredictor, compute_lstm_inputs, SEQ_LEN, MODEL_DIR, DB_PATH

def process_batch(symbols_batch, batch_idx, model, mean, std, tmpdir):
    """处理一批股票，写入临时文件"""
    embeddings = {}
    conn = sqlite3.connect(DB_PATH)
    
    for sym in symbols_batch:
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

        except Exception as e:
            continue
    
    conn.close()
    
    # 写入临时文件
    tmpfile = os.path.join(tmpdir, f'batch_{batch_idx:04d}.pkl')
    with open(tmpfile, 'wb') as f:
        pickle.dump(embeddings, f)
    
    return len(embeddings), sum(len(v) for v in embeddings.values())

def main():
    model_path = os.path.join(MODEL_DIR, 'encoder.pt')
    stats_path = os.path.join(MODEL_DIR, 'norm_stats.pt')
    output_path = os.path.join(os.path.dirname(DB_PATH), 'lstm_embeddings.pkl')

    print("🔮 提取 LSTM embeddings (分批写入)")
    
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    input_dim = state_dict['lstm.weight_ih_l0'].shape[1]
    model = LSTMPredictor(input_dim=input_dim, hidden_dim=64)
    model.load_state_dict(state_dict)
    model.eval()

    norm_stats = torch.load(stats_path, map_location='cpu', weights_only=False)
    mean = np.squeeze(norm_stats['x_mean'])
    std = np.squeeze(norm_stats['x_std'])

    conn = sqlite3.connect(DB_PATH)
    symbols = [r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM kline_30m ORDER BY symbol"
    ).fetchall()]
    conn.close()
    
    print(f"   {len(symbols)} 只股票")
    
    batch_size = 20
    n_batches = (len(symbols) + batch_size - 1) // batch_size
    t0 = time.time()
    total_stocks = 0
    total_embs = 0
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(0, len(symbols), batch_size):
            batch_idx = i // batch_size
            batch = symbols[i:i + batch_size]
            n, m = process_batch(batch, batch_idx, model, mean, std, tmpdir)
            total_stocks += n
            total_embs += m
            gc.collect()
            
            elapsed = time.time() - t0
            eta = elapsed / (batch_idx + 1) * (n_batches - batch_idx - 1) if batch_idx < n_batches - 1 else 0
            print(f"   [{batch_idx+1}/{n_batches}] {total_stocks} stocks, {total_embs:,} embs, {elapsed:.0f}s, eta={eta:.0f}s")
        
        # 合并所有临时文件
        print("   合并临时文件...")
        all_embeddings = {}
        for f in sorted(glob.glob(os.path.join(tmpdir, 'batch_*.pkl'))):
            with open(f, 'rb') as fh:
                all_embeddings.update(pickle.load(fh))
        
        with open(output_path, 'wb') as f:
            pickle.dump(all_embeddings, f)
    
    elapsed = time.time() - t0
    print(f"\n✅ 完成! {elapsed:.0f}s, {len(all_embeddings)} stocks, {sum(len(v) for v in all_embeddings.values()):,} embeddings")
    print(f"   文件: {output_path} ({os.path.getsize(output_path)/1024/1024:.1f} MB)")

if __name__ == '__main__':
    main()