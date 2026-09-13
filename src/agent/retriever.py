"""
Historical Support RAG Retriever module for Apple Support AI Agent.
Retrieves top-k historical support interaction pairs using TF-IDF & Cosine Similarity.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SupportRetriever:
    """Retrieves top-k historical resolved support tweets grounded in past brand interactions."""

    def __init__(self, data_path: str = "data/raw_support_tweets.csv"):
        self.data_path = data_path
        self.df = None
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = None
        self._load_and_index()

    def _load_and_index(self):
        """Loads historical support pairs and computes TF-IDF index matrix."""
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
        else:
            # Fallback inline memory index if file missing
            self.df = pd.DataFrame([
                {
                    "customer_tweet": "I was charged twice for subscription.",
                    "intent": "billing_subscription",
                    "historical_reply": "Check purchase history and request refund at http://reportaproblem.apple.com."
                },
                {
                    "customer_tweet": "My iPhone battery drains fast.",
                    "intent": "device_troubleshooting",
                    "historical_reply": "Check Settings > Battery and force restart your device."
                }
            ])
            
        corpus = self.df["customer_tweet"].fillna("").tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 2, filter_intent: str = None) -> List[Dict[str, Any]]:
        """
        Retrieves top-k most similar historical support pairs for a customer query.
        Optionally filters by intent.
        """
        query_vec = self.vectorizer.transform([query])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Sort indices by similarity descending
        top_indices = np.argsort(sim_scores)[::-1]
        
        results = []
        for idx in top_indices:
            if len(results) >= top_k:
                break
                
            row = self.df.iloc[idx]
            row_intent = row.get("intent", "")
            
            # Optional intent filter
            if filter_intent and row_intent and row_intent != filter_intent:
                continue
                
            results.append({
                "similarity_score": float(round(sim_scores[idx], 4)),
                "historical_query": row["customer_tweet"],
                "historical_reply": row["historical_reply"],
                "intent": row_intent
            })
            
        # Fallback if filter yielded no results
        if not results and top_indices.size > 0:
            best_idx = top_indices[0]
            row = self.df.iloc[best_idx]
            results.append({
                "similarity_score": float(round(sim_scores[best_idx], 4)),
                "historical_query": row["customer_tweet"],
                "historical_reply": row["historical_reply"],
                "intent": row.get("intent", "")
            })

        return results
