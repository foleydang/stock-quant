#!/usr/bin/env python3
"""
意图分类模型推理器 v5 - 纯 TF-IDF + LR
纯本地，< 1ms，~300KB
"""

import json, os, pickle
import numpy as np
from typing import Tuple, Dict, Optional


class IntentClassifier:
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'intent_classifier')
        
        model_path = os.path.join(model_dir, 'intent_classifier.pkl')
        label_path = os.path.join(model_dir, 'label_map.json')
        
        with open(model_path, 'rb') as f:
            components = pickle.load(f)
        
        self.tfidf = components['tfidf']
        self.clf = components['clf']
        self.le = components['le']
        
        with open(label_path, 'r') as f:
            self.label_map = json.load(f)
        self.idx_to_label = {int(k): v for k, v in self.label_map.items()}
    
    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        x = self.tfidf.transform([text])
        
        if hasattr(self.clf, 'predict_proba'):
            proba = self.clf.predict_proba(x)[0]
        else:
            scores = self.clf.decision_function(x)[0]
            e = np.exp(scores - scores.max())
            proba = e / e.sum()
        
        idx = proba.argmax()
        intent = self.idx_to_label[idx]
        confidence = float(proba[idx])
        probs = {self.idx_to_label[i]: float(p) for i, p in enumerate(proba)}
        return intent, confidence, probs


_classifier: Optional[IntentClassifier] = None


def get_classifier() -> Optional[IntentClassifier]:
    global _classifier
    if _classifier is None:
        try:
            _classifier = IntentClassifier()
        except Exception as e:
            import logging
            logging.getLogger('feishu_bot').warning(f"模型加载失败: {e}")
            _classifier = None
    return _classifier