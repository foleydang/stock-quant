#!/usr/bin/env python3
"""
意图分类模型 - 推理器

使用百炼 Embedding API + sklearn 分类器
"""

import json
import os
import pickle
import numpy as np
import requests
from typing import Tuple, Dict, Optional


class IntentClassifier:
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'intent_classifier')
        
        # 加载分类器
        model_path = os.path.join(model_dir, 'intent_classifier.pkl')
        label_path = os.path.join(model_dir, 'label_map.json')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型不存在: {model_path}，请先运行 train.py")
        
        with open(model_path, 'rb') as f:
            self.clf = pickle.load(f)
        with open(label_path, 'r') as f:
            self.label_map = json.load(f)
        
        self.idx_to_label = {int(k): v for k, v in self.label_map.items()}
        self.labels = sorted(self.idx_to_label.values())
        
        # API Key
        api_key = os.environ.get('DASHSCOPE_API_KEY', '')
        if not api_key:
            # 尝试从 .env 加载
            env_paths = [
                os.path.join(os.path.dirname(__file__), '..', '.env'),
                os.path.join(os.path.dirname(__file__), '..', '..', '.env'),
            ]
            for p in env_paths:
                if os.path.exists(p):
                    from dotenv import load_dotenv
                    load_dotenv(p)
                    break
            api_key = os.environ.get('DASHSCOPE_API_KEY', '')
        
        self.api_key = api_key
        self.embedding_url = 'https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding'
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本的 embedding 向量（百炼 API）"""
        resp = requests.post(
            self.embedding_url,
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            },
            json={
                'model': 'text-embedding-v4',
                'input': {'texts': [text]},
                'parameters': {'text_type': 'query'},
            },
            timeout=10,
        )
        data = resp.json()
        if data.get('output') and data.get('output').get('embeddings'):
            return np.array(data['output']['embeddings'][0]['embedding'], dtype=np.float32)
        raise Exception(f"Embedding API error: {data}")
    
    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        预测意图
        
        Returns:
            (intent, confidence, {intent: probability})
        """
        emb = self._get_embedding(text).reshape(1, -1)
        proba = self.clf.predict_proba(emb)[0]
        pred_idx = proba.argmax()
        
        intent = self.idx_to_label[pred_idx]
        confidence = float(proba[pred_idx])
        
        probs = {self.idx_to_label[i]: float(p) for i, p in enumerate(proba)}
        
        return intent, confidence, probs
    
    def classify(self, text: str, threshold: float = 0.5) -> Tuple[str, Dict]:
        """
        分类意图（兼容 intent_router 接口）
        
        Returns:
            (intent, params_dict)
        """
        intent, confidence, probs = self.predict(text)
        
        params = {}
        if confidence < threshold:
            # 低置信度 → 回退到 LLM
            from intent_router import llm_classify
            return llm_classify(text)
        
        # 提取股票代码
        if intent in ('stock_brief', 'stock_deep', 'stock_news'):
            from intent_router import extract_symbol
            symbol = extract_symbol(text)
            if symbol:
                params['symbol'] = symbol
        
        return intent, params


# ========== Singleton ==========

_classifier: Optional[IntentClassifier] = None


def get_classifier() -> Optional[IntentClassifier]:
    global _classifier
    if _classifier is None:
        try:
            _classifier = IntentClassifier()
        except Exception as e:
            import logging
            logging.getLogger('feishu_bot').warning(f"模型加载失败，回退到关键词: {e}")
            _classifier = None
    return _classifier