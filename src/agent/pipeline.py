"""
Unified Support Agent Pipeline module for Apple Support AI Agent.
Combines Classifier, Retriever, Generator, and Escalator into a single interface.
"""

import time
from typing import Dict, Any
from src.agent.classifier import IntentClassifier
from src.agent.retriever import SupportRetriever
from src.agent.generator import ResponseGenerator
from src.agent.escalator import EscalationRouter

class SupportAgentPipeline:
    """End-to-end AI Support Agent pipeline handling classification, grounding, response generation, and escalation."""

    def __init__(self, data_path: str = "data/raw_support_tweets.csv"):
        self.classifier = IntentClassifier()
        self.retriever = SupportRetriever(data_path=data_path)
        self.generator = ResponseGenerator()
        self.escalator = EscalationRouter()

    def process(self, customer_text: str) -> Dict[str, Any]:
        """
        Executes full agent pipeline for an incoming customer tweet.
        Returns complete response structure with intent, retrieval context, escalation decision, and reply.
        """
        start_time = time.time()

        # Step 1: Intent Classification
        clf_result = self.classifier.classify(customer_text)
        predicted_intent = clf_result["predicted_intent"]

        # Step 2: RAG Historical Context Retrieval
        retrieved_context = self.retriever.retrieve(
            query=customer_text, 
            top_k=2, 
            filter_intent=predicted_intent
        )

        # Step 3: Escalation Decision & Reason
        escalation_result = self.escalator.evaluate(customer_text, predicted_intent)

        # Step 4: Grounded Response Generation
        drafted_reply = self.generator.generate(
            customer_text=customer_text,
            intent=predicted_intent,
            retrieved_context=retrieved_context,
            escalation_info=escalation_result
        )

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "customer_text": customer_text,
            "predicted_intent": predicted_intent,
            "intent_confidence": clf_result["confidence"],
            "should_escalate": escalation_result["should_escalate"],
            "escalation_reason": escalation_result["escalation_reason"],
            "escalation_rule": escalation_result.get("trigger_rule", "none"),
            "drafted_reply": drafted_reply,
            "retrieved_context": retrieved_context,
            "processing_time_ms": elapsed_ms
        }
