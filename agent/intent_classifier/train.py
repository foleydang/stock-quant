#!/usr/bin/env python3
"""
意图分类模型训练 v3 - 纯本地 TF-IDF + LogisticRegression

零外部依赖，零 API 调用，模型 < 200KB
"""

import json, os, pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    data_path = os.path.join(SCRIPT_DIR, 'training_data.jsonl')

    # 1. 加载数据
    print("1. 加载数据", flush=True)
    texts, labels = [], []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            texts.append(item['text'])
            labels.append(item['intent'])
    print(f"   共 {len(texts)} 条, {len(set(labels))} 类", flush=True)

    # 2. 构建 Pipeline: TF-IDF + 分类器
    print("\n2. 训练 Pipeline(TF-IDF + LogisticRegression)", flush=True)
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 3),  # 1-3 gram 捕捉短语
            analyzer='char_wb',   # char-level with word boundaries (对中文好)
            min_df=2,
        )),
        ('clf', LogisticRegression(
            C=10.0, max_iter=1000, class_weight='balanced', random_state=42,
        )),
    ])

    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f"   类别: {list(le.classes_)}", flush=True)

    # 交叉验证
    scores = cross_val_score(pipeline, texts, y, cv=5, scoring='accuracy')
    print(f"   ✅ 5-fold CV accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})", flush=True)

    # 全量训练
    pipeline.fit(texts, y)

    # 3. 保存
    model_path = os.path.join(SCRIPT_DIR, 'intent_classifier.pkl')
    label_path = os.path.join(SCRIPT_DIR, 'label_map.json')
    
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    label_map = {i: label for i, label in enumerate(le.classes_)}
    with open(label_path, 'w') as f:
        json.dump(label_map, f, ensure_ascii=False)
    
    # 模型大小
    size_kb = os.path.getsize(model_path) / 1024
    print(f"\n3. 模型保存: {model_path} ({size_kb:.0f} KB)", flush=True)
    print(f"   标签: {label_map}", flush=True)

    # 4. 测试
    print("\n4. 测试", flush=True)
    tests = [
        "茅台多少钱", "茅台可以买吗", "茅台最近有新闻吗",
        "我的持仓怎么样", "今天大盘如何", "茅台和五粮液对比",
        "帮助", "今天赚了多少", "比较下茅台五粮液",
        "茅台跟五粮液", "茅台五粮液", "汇川技术怎么操作",
        "港股通互联网", "这个股票怎么样", "推荐个股票",
        "毛台多少钱", "珈仓茅台", "瞅瞅持仓",
    ]
    for text in tests:
        proba = pipeline.predict_proba([text])[0]
        idx = proba.argmax()
        pred = le.inverse_transform([idx])[0]
        print(f"   [{proba[idx]:.2f}] {pred:15s} | {text}", flush=True)

    print("\n✅ 训练完成", flush=True)


if __name__ == '__main__':
    main()