#!/usr/bin/env python3
"""
意图分类模型训练 v2 - 批量 embedding API

流程:
1. 读取 training_data.jsonl
2. 批量调用百炼 Embedding API (50条/次)
3. 训练 sklearn LogisticRegression
4. 保存模型
"""

import json, os, sys, time, pickle, requests
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 加载 .env
from dotenv import load_dotenv
for p in [os.path.join(SCRIPT_DIR, '..', '.env'), os.path.join(SCRIPT_DIR, '..', '..', '.env')]:
    if os.path.exists(p):
        load_dotenv(p)
        break

API_KEY = os.environ.get('DASHSCOPE_API_KEY', '')
BATCH_SIZE = 10  # API 限制每批最多 10 条


def get_embeddings_batch(texts: list) -> list:
    """批量获取 embedding"""
    resp = requests.post(
        'https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding',
        headers={'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'},
        json={
            'model': 'text-embedding-v4',
            'input': {'texts': texts},
            'parameters': {'text_type': 'query'},
        },
        timeout=30,
    )
    data = resp.json()
    if data.get('output') and data.get('output').get('embeddings'):
        return [e['embedding'] for e in data['output']['embeddings']]
    raise Exception(f"Embedding error: {data}")


def main():
    data_path = os.path.join(SCRIPT_DIR, 'training_data.jsonl')
    emb_cache = os.path.join(SCRIPT_DIR, 'embeddings.npy')

    # 1. 加载数据
    print("1. 加载数据", flush=True)
    texts, labels = [], []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            texts.append(item['text'])
            labels.append(item['intent'])
    print(f"   共 {len(texts)} 条, {len(set(labels))} 类", flush=True)

    # 2. Embedding (批量)
    if os.path.exists(emb_cache):
        print(f"\n2. 加载缓存: {emb_cache}", flush=True)
        X = np.load(emb_cache)
    else:
        print(f"\n2. 批量计算 embedding (batch={BATCH_SIZE})...", flush=True)
        all_embs = []
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            print(f"   [{batch_num}/{total_batches}] {len(batch)} 条...", end=' ', flush=True)
            try:
                embs = get_embeddings_batch(batch)
                all_embs.extend(embs)
                print(f"✓", flush=True)
            except Exception as e:
                print(f"✗ {e}", flush=True)
                all_embs.extend([[0.0] * 1024] * len(batch))
            time.sleep(0.1)
        X = np.array(all_embs, dtype=np.float32)
        np.save(emb_cache, X)
        print(f"   保存: {emb_cache} ({X.shape})", flush=True)

    # 3. 训练
    print("\n3. 训练分类器", flush=True)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f"   类别: {list(le.classes_)}", flush=True)
    print(f"   数据: {X.shape[0]} 条, {X.shape[1]} 维", flush=True)

    clf = LogisticRegression(
        C=10.0, max_iter=1000, class_weight='balanced', random_state=42,
    )
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"   ✅ 5-fold CV accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})", flush=True)

    clf.fit(X, y)

    # 4. 保存
    model_path = os.path.join(SCRIPT_DIR, 'intent_classifier.pkl')
    label_path = os.path.join(SCRIPT_DIR, 'label_map.json')
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    label_map = {i: label for i, label in enumerate(le.classes_)}
    with open(label_path, 'w') as f:
        json.dump(label_map, f, ensure_ascii=False)
    print(f"\n4. 模型保存: {model_path}", flush=True)
    print(f"   标签: {label_map}", flush=True)

    # 5. 测试
    print("\n5. 测试", flush=True)
    tests = [
        "茅台多少钱", "茅台可以买吗", "茅台最近有新闻吗",
        "我的持仓怎么样", "今天大盘如何", "茅台和五粮液对比",
        "帮助", "今天赚了多少",
    ]
    test_embs = get_embeddings_batch(tests)
    for text, emb in zip(tests, test_embs):
        e = np.array(emb).reshape(1, -1)
        proba = clf.predict_proba(e)[0]
        idx = proba.argmax()
        print(f"   [{proba[idx]:.2f}] {le.inverse_transform([idx])[0]:15s} | {text}", flush=True)

    print("\n✅ 训练完成", flush=True)


if __name__ == '__main__':
    main()