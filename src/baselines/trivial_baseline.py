"""
Trivial Baseline Agent (Baseline 1).
Uses basic keyword lookup, generic canned response, and never escalates.
"""

from typing import Dict, Any

class TrivialBaselineAgent:
    """Trivial baseline using primitive rules and canned responses."""

    def process(self, customer_text: str) -> Dict[str, Any]:
        text_lower = customer_text.lower()

        # Simple naive keyword matching
        if "charge" in text_lower or "bill" in text_lower or "money" in text_lower:
            intent = "billing_subscription"
        elif "battery" in text_lower or "screen" in text_lower or "phone" in text_lower:
            intent = "device_troubleshooting"
        elif "password" in text_lower or "login" in text_lower:
            intent = "account_apple_id"
        else:
            intent = "unknown_other"

        # Canned static response
        reply = "Thank you for reaching out to Apple Support! We are glad to help. Please check our support website at http://support.apple.com for assistance."

        # Always auto-handles (never escalates)
        return {
            "predicted_intent": intent,
            "should_escalate": False,
            "escalation_reason": "Trivial baseline auto-handles all incoming customer messages.",
            "drafted_reply": reply
        }
