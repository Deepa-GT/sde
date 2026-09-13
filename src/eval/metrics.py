"""
Evaluation Metrics module for Apple Support AI Agent.
Calculates Intent Classification Accuracy/F1, Escalation Precision/Recall/F1, and ROUGE/BLEU/Cosine metrics for text replies.
"""

import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_n_gram_overlap(candidate: str, reference: str, n: int = 1) -> float:
    """Calculates n-gram precision overlap between candidate and reference text."""
    cand_words = candidate.lower().split()
    ref_words = reference.lower().split()
    
    if not cand_words or not ref_words:
        return 0.0
        
    cand_ngrams = [tuple(cand_words[i:i+n]) for i in range(len(cand_words)-n+1)]
    ref_ngrams = set(tuple(ref_words[i:i+n]) for i in range(len(ref_words)-n+1))
    
    if not cand_ngrams:
        return 0.0
        
    matches = sum(1 for ng in cand_ngrams if ng in ref_ngrams)
    return float(matches / len(cand_ngrams))

def calculate_lcs_length(x: List[str], y: List[str]) -> int:
    """Calculates Longest Common Subsequence length for ROUGE-L."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def calculate_rouge_l(candidate: str, reference: str) -> float:
    """Calculates ROUGE-L F1 score based on Longest Common Subsequence."""
    cand_words = candidate.lower().split()
    ref_words = reference.lower().split()
    if not cand_words or not ref_words:
        return 0.0
    lcs = calculate_lcs_length(cand_words, ref_words)
    prec = lcs / len(cand_words)
    rec = lcs / len(ref_words)
    if prec + rec == 0:
        return 0.0
    return float((2 * prec * rec) / (prec + rec))

def calculate_cosine_similarity(candidate: str, reference: str) -> float:
    """Calculates TF-IDF cosine similarity between candidate and reference."""
    vectorizer = TfidfVectorizer().fit([candidate, reference])
    vecs = vectorizer.transform([candidate, reference])
    sim = cosine_similarity(vecs[0], vecs[1])[0][0]
    return float(sim)

def evaluate_predictions(
    y_true_intent: List[str], 
    y_pred_intent: List[str],
    y_true_escalate: List[bool],
    y_pred_escalate: List[bool],
    reference_replies: List[str],
    predicted_replies: List[str]
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics comparing ground-truth and agent predictions.
    """
    # Intent Classification Metrics
    intent_acc = accuracy_score(y_true_intent, y_pred_intent)
    _, _, intent_f1, _ = precision_recall_fscore_support(y_true_intent, y_pred_intent, average="macro", zero_division=0)
    
    # Escalation Routing Metrics
    esc_p, esc_r, esc_f1, _ = precision_recall_fscore_support(y_true_escalate, y_pred_escalate, average="binary", zero_division=0)
    esc_acc = accuracy_score(y_true_escalate, y_pred_escalate)

    # Reply Quality NLP Metrics
    rouge_1_scores = [calculate_n_gram_overlap(p, r, 1) for p, r in zip(predicted_replies, reference_replies)]
    rouge_l_scores = [calculate_rouge_l(p, r) for p, r in zip(predicted_replies, reference_replies)]
    cosine_scores = [calculate_cosine_similarity(p, r) for p, r in zip(predicted_replies, reference_replies)]

    return {
        "intent_accuracy": round(float(intent_acc), 4),
        "intent_macro_f1": round(float(intent_f1), 4),
        "escalation_accuracy": round(float(esc_acc), 4),
        "escalation_precision": round(float(esc_p), 4),
        "escalation_recall": round(float(esc_r), 4),
        "escalation_f1": round(float(esc_f1), 4),
        "mean_rouge_1": round(float(np.mean(rouge_1_scores)), 4),
        "mean_rouge_l": round(float(np.mean(rouge_l_scores)), 4),
        "mean_cosine_similarity": round(float(np.mean(cosine_scores)), 4)
    }
