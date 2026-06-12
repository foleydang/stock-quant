#!/usr/bin/env python3
"""
意图分类模型训练 v5 - 纯 TF-IDF + LogisticRegression (简化最优)

改动: 只用 TF-IDF，不加手工特征，SVM/LR都试
"""

import json, os, pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    data_path = os.path.join(SCRIPT_DIR, 'training_data.jsonl')

    print("1. 加载数据", flush=True)
    texts, labels = [], []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            texts.append(item['text'])
            labels.append(item['intent'])
    
    from collections import Counter
    for k, v in sorted(Counter(labels).items()):
        print(f"   {k}: {v}", flush=True)
    print(f"   共 {len(texts)} 条", flush=True)
    
    le = LabelEncoder()
    y = le.fit_transform(labels)

    print("\n2. 特征: TF-IDF (char_wb, 1-3 gram, 3000维)", flush=True)
    tfidf = TfidfVectorizer(
        max_features=3000, ngram_range=(1, 3),
        analyzer='char_wb', min_df=2, sublinear_tf=True,
    )
    X = tfidf.fit_transform(texts)
    print(f"   维度: {X.shape}", flush=True)

    print("\n3. 训练对比", flush=True)
    best_name, best_score, best_clf = '', 0, None
    
    for name, clf in [
        ("LR", LogisticRegression(C=10.0, max_iter=1000, class_weight='balanced', random_state=42)),
        ("SVM", LinearSVC(C=1.0, max_iter=2000, class_weight='balanced', dual=False, random_state=42)),
    ]:
        scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        print(f"   {name}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})", flush=True)
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_name = name
            best_clf = clf
    
    best_clf.fit(X, y)
    print(f"\n   ✅ 最佳: {best_name} ({best_score:.3f})", flush=True)

    # 保存
    model_path = os.path.join(SCRIPT_DIR, 'intent_classifier.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'tfidf': tfidf, 'clf': best_clf, 'le': le}, f)
    
    label_map = {i: label for i, label in enumerate(le.classes_)}
    with open(os.path.join(SCRIPT_DIR, 'label_map.json'), 'w') as f:
        json.dump(label_map, f, ensure_ascii=False)
    
    print(f"\n4. 模型: {model_path} ({os.path.getsize(model_path)/1024:.0f} KB)", flush=True)

    # 测试
    print("\n5. 测试", flush=True)
    tests = [
        "茅台多少钱", "茅台可以买吗", "茅台最近有新闻吗",
        "我的持仓怎么样", "今天大盘如何", "茅台和五粮液对比",
        "帮助", "今天赚了多少", "比较下茅台五粮液",
        "茅台跟五粮液", "茅台五粮液", "汇川技术怎么操作",
        "港股通互联网", "毛台多少钱", "珈仓茅台",
        "汇川技术操作指南", "港股通互联网消息面",
        "平安和招行比较", "这个股票怎么样",
    ]
    for text in tests:
        x = tfidf.transform([text])
        if hasattr(best_clf, 'predict_proba'):
            proba = best_clf.predict_proba(x)[0]
            idx = proba.argmax()
            conf = proba[idx]
        else:
            scores = best_clf.decision_function(x)[0]
            idx = scores.argmax()
            # softmax pseudo-probability
            e = np.exp(scores - scores.max())
            conf = e[idx] / e.sum()
        pred = le.inverse_transform([idx])[0]
        print(f"   [{conf:.2f}] {pred:15s} | {text}", flush=True)

    print("\n✅ 训练完成", flush=True)


if __name__ == '__main__':
    main()