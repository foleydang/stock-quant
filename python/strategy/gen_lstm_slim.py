#!/usr/bin/env python3
"""生成瘦身版 LSTM embeddings — 每只股票只保留最新一天的 embedding。

背景: data/lstm_embeddings.pkl 是全历史 {sym: {date: array(64)}} ~273MB,
在 1.8GB 服务器上整体载入会 OOM。而"今日打分/扫描"只用每只股票的最后一根
bar 的特征,lstm_* 是终端特征(下游 interact/macro_interact 不依赖它),故只需
每只股票最新一天的 embedding。瘦身文件 ~0.1MB, 且最后一行的 lstm 特征与全量文件
逐位相同(取的都是该股票可用的最新 embedding 日期)。

每次在 Mac 重算 lstm_embeddings.pkl 后, 重跑本脚本, 一并 commit 瘦身文件, 服务器
git pull 即用。

用法: python strategy/gen_lstm_slim.py
"""
import os
import pickle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL = os.path.join(ROOT, 'data', 'lstm_embeddings.pkl')
SLIM = os.path.join(ROOT, 'data', 'lstm_embeddings_latest.pkl')


def main():
    with open(FULL, 'rb') as f:
        full = pickle.load(f)
    slim = {}
    for sym, emb in full.items():
        if not isinstance(emb, dict) or not emb:
            continue
        mx = max(emb.keys())
        slim[sym] = {mx: emb[mx]}
    with open(SLIM, 'wb') as f:
        pickle.dump(slim, f)
    size_mb = os.path.getsize(SLIM) / 1024 / 1024
    print(f"✅ 瘦身 embeddings: {len(slim)} 只 → {SLIM} ({size_mb:.2f} MB)")


if __name__ == '__main__':
    main()
