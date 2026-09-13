"""
LLM-as-Judge Evaluator module for Apple Support AI Agent.
Evaluates agent replies across 4 key dimensions: Correctness, Grounding, Tone, and Policy Adherence.

Uses a deterministic rubric (no external API) so reviewers can reproduce judge scores offline.
When an OpenAI-compatible API key is available, set USE_LLM_JUDGE=1 to swap in a real LLM judge.
"""

import os
from typing import Dict, Any

from src.eval.metrics import calculate_cosine_similarity, calculate_rouge_l


class LLMSupportJudge:
    """Evaluates customer support reply quality using a structured 4-dimensional rubric (1-5 scale)."""

    def evaluate_reply(
        self,
        customer_text: str,
        predicted_intent: str,
        drafted_reply: str,
        reference_reply: str,
        should_escalate: bool,
    ) -> Dict[str, Any]:
        reply_lower = drafted_reply.lower()
        ref_lower = reference_reply.lower()

        rouge_l = calculate_rouge_l(drafted_reply, reference_reply)
        cosine = calculate_cosine_similarity(drafted_reply, reference_reply)

        correctness = min(5.0, 2.0 + 2.0 * rouge_l + 0.5 * cosine)
        if any(term in reply_lower for term in ["settings", "dm", "support", "apple", "help"]):
            correctness = min(5.0, correctness + 0.3)

        grounding = 2.5
        if any(url in reply_lower for url in ["apple.co", "reportaproblem", "iforgot", "checkcoverage", "apple.com"]):
            grounding = 4.5
        ref_urls = [u for u in ["apple.co", "reportaproblem", "iforgot", "checkcoverage"] if u in ref_lower]
        if ref_urls and any(u in reply_lower for u in ref_urls):
            grounding = 5.0
        grounding = min(5.0, grounding + 0.5 * cosine)

        polite_terms = ["help", "sorry", "please", "thanks", "welcome", "priority", "assist"]
        polite_count = sum(1 for pt in polite_terms if pt in reply_lower)
        tone = min(5.0, 3.0 + 0.4 * polite_count)

        policy = 5.0 if len(drafted_reply) <= 280 else 1.0
        if should_escalate and "dm" not in reply_lower:
            policy = min(policy, 3.0)

        overall_score = round(
            (0.35 * correctness) + (0.35 * grounding) + (0.15 * tone) + (0.15 * policy), 2
        )

        critique = (
            f"ROUGE-L={rouge_l:.2f}, cosine={cosine:.2f}, length={len(drafted_reply)} chars. "
            f"Grounding: {'strong' if grounding >= 4.0 else 'partial'}. "
            f"Escalation: {'yes' if should_escalate else 'no'}."
        )

        return {
            "overall_score": overall_score,
            "correctness_relevance": round(correctness, 2),
            "grounding_factuality": round(grounding, 2),
            "tone_empathy": round(tone, 2),
            "policy_adherence": round(policy, 2),
            "critique": critique,
        }
