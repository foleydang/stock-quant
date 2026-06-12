#!/usr/bin/env python3
"""
意图分类模型推理器 v3 - 纯本地 TF-IDF + LogisticRegression

零 API 调用，推理 < 1ms，模型 < 200KB
"""

import json, os, pickle
from typing import Tuple, Dict, Optional


class IntentClassifier:
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'intent_classifier')
        
        model_path = os.path.join(model_dir, 'intent_classifier.pkl')
        label_path = os.path.join(model_dir, 'label_map.json')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型不存在: {model_path}，请先运行 train.py")
        
        with open(model_path, 'rb') as f:
            self.pipeline = pickle.load(f)
        with open(label_path, 'r') as f:
            self.label_map = json.load(f)
        
        self.idx_to_label = {int(k): v for k, v in self.label_map.items()}
    
    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """预测意图: (intent, confidence, {intent: prob})"""
        proba = self.pipeline.predict_proba([text])[0]
        idx = proba.argmax()
        intent = self.idx_to_label[idx]
        confidence = float(proba[idx])
        probs = {self.idx_to_label[i]: float(p) for i, p in enumerate(proba)}
        return intent, confidence, probs


# ========== Singleton ==========

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