"""
Simple ML Baseline Agent (Baseline 2).
Uses TF-IDF + Logistic Regression for intent classification, 1-NN lookup for replies, and primitive keyword escalation.
"""

import os
import pandas as pd
from typing import Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class SimpleMLBaselineAgent:
    """Simple ML baseline model for intent classification, retrieval, and heuristic escalation."""

    def __init__(self, data_path: str = "data/raw_support_tweets.csv"):
        self.data_path = data_path
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.clf = LogisticRegression(max_iter=200)
        self.df = None
        self._train()

    def _train(self):
        """Fits TF-IDF and Logistic Regression on historical support pairs."""
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
        else:
            self.df = pd.DataFrame([
                {"customer_tweet": "I was charged twice.", "intent": "billing_subscription", "historical_reply": "Visit http://reportaproblem.apple.com"},
                {"customer_tweet": "My battery dies fast.", "intent": "device_troubleshooting", "historical_reply": "Restart your phone and check battery settings."}
            ])
            
        X_texts = self.df["customer_tweet"].fillna("").tolist()
        y_labels = self.df["intent"].fillna("unknown_other").tolist()
        
        X_vec = self.vectorizer.fit_transform(X_texts)
        self.clf.fit(X_vec, y_labels)

    def process(self, customer_text: str) -> Dict[str, Any]:
        vec = self.vectorizer.transform([customer_text])
        predicted_intent = str(self.clf.predict(vec)[0])

        # Simple 1-NN Lookup for reply
        subset = self.df[self.df["intent"] == predicted_intent]
        if not subset.empty:
            drafted_reply = subset.iloc[0]["historical_reply"]
        else:
            drafted_reply = "Please visit http://support.apple.com for Apple Support guidance."

        # Primitive heuristic escalation (only escalates if explicit keywords are present)
        text_lower = customer_text.lower()
        should_escalate = any(kw in text_lower for kw in ["escalate", "human", "manager", "representative", "lawyer"])
        escalation_reason = "Simple ML baseline escalation triggered by explicit human request keyword." if should_escalate else "Simple ML baseline auto-handles message."

        return {
            "predicted_intent": predicted_intent,
            "should_escalate": should_escalate,
            "escalation_reason": escalation_reason,
            "drafted_reply": drafted_reply
        }
