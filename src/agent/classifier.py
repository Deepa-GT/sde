"""
Intent Classifier module for Apple Support AI Agent.
Hybrid TF-IDF + Logistic Regression trained on historical pairs, with keyword fallback.
"""

import os
from typing import Dict, Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.agent.taxonomy import Intent, INTENT_KEYWORDS, get_all_intents


class IntentClassifier:
    """Classifies customer tweets using ML trained on brand history plus keyword fallback."""

    def __init__(self, data_path: str = "data/raw_support_tweets.csv"):
        self.intents = get_all_intents()
        self.vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
        self.clf = LogisticRegression(max_iter=500, class_weight="balanced")
        self._trained = False
        self._train(data_path)

    def _train(self, data_path: str) -> None:
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            texts = df["customer_tweet"].fillna("").tolist()
            labels = df["intent"].fillna(Intent.UNKNOWN_OTHER.value).tolist()
            if len(texts) >= 10:
                X = self.vectorizer.fit_transform(texts)
                self.clf.fit(X, labels)
                self._trained = True

    def _keyword_classify(self, text: str) -> Dict[str, Any]:
        clean_text = text.lower()
        scores: Dict[str, float] = {intent: 0.0 for intent in self.intents}

        for intent_enum, keywords in INTENT_KEYWORDS.items():
            intent_val = intent_enum.value
            if intent_val not in scores:
                continue
            for kw in keywords:
                if kw in clean_text:
                    weight = 2.0 if " " in kw else 1.0
                    scores[intent_val] += weight

        total_score = sum(scores.values())
        best_intent = max(scores, key=scores.get) if total_score > 0 else Intent.UNKNOWN_OTHER.value
        confidence = (scores[best_intent] / total_score) if total_score > 0 else 0.3

        if total_score == 0:
            if any(greeting in clean_text for greeting in ["thanks", "thank you", "hello", "hi", "hey"]):
                best_intent = Intent.UNKNOWN_OTHER.value
                confidence = 0.8
            elif any(prod in clean_text for prod in ["iphone", "mac", "ipad", "watch"]):
                best_intent = Intent.DEVICE_TROUBLESHOOTING.value
                confidence = 0.5
            else:
                best_intent = Intent.UNKNOWN_OTHER.value
                confidence = 0.4

        return {
            "predicted_intent": best_intent,
            "confidence": round(confidence, 3),
            "intent_scores": {k: round(v, 2) for k, v in scores.items()},
        }

    def classify(self, text: str) -> Dict[str, Any]:
        """Classify input text; prefer ML model when trained, blend with keyword signals."""
        if not self._trained:
            return self._keyword_classify(text)

        vec = self.vectorizer.transform([text])
        ml_intent = str(self.clf.predict(vec)[0])
        ml_probs = self.clf.predict_proba(vec)[0]
        ml_conf = float(max(ml_probs))

        kw_result = self._keyword_classify(text)
        kw_intent = kw_result["predicted_intent"]
        kw_conf = kw_result["confidence"]

        if ml_intent == kw_intent:
            return {
                "predicted_intent": ml_intent,
                "confidence": round(max(ml_conf, kw_conf), 3),
                "intent_scores": kw_result["intent_scores"],
            }

        if ml_conf >= 0.55 and ml_conf >= kw_conf:
            return {
                "predicted_intent": ml_intent,
                "confidence": round(ml_conf, 3),
                "intent_scores": kw_result["intent_scores"],
            }

        return kw_result
