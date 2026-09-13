"""
Escalation Router module for Apple Support AI Agent.
Determines whether incoming messages require human escalation and provides explicit rationales.
"""

from typing import Dict, Any

from src.agent.taxonomy import Intent


class EscalationRouter:
    """Evaluates customer messages and intent context to decide on auto-handling vs human escalation."""

    def evaluate(self, customer_text: str, intent: str) -> Dict[str, Any]:
        """Return should_escalate boolean and human-readable escalation_reason."""
        text_lower = customer_text.lower()

        safety_keywords = [
            "smoke", "smoking", "fire", "explode", "exploded", "burning", "hot",
            "hazard", "sparks", "spark", "swollen", "melting", "electrocut",
        ]
        if any(kw in text_lower for kw in safety_keywords):
            return {
                "should_escalate": True,
                "escalation_reason": "Critical physical safety hazard (device thermal/battery risk) requires emergency human escalation.",
                "trigger_rule": "safety_thermal_hazard",
            }

        fraud_keywords = [
            "stolen", "unauthorized", "fraud", "hacked", "hacker", "stolen card",
            "compromised", "identity theft", "someone accessed",
        ]
        if any(kw in text_lower for kw in fraud_keywords):
            return {
                "should_escalate": True,
                "escalation_reason": "High-value financial fraud or unauthorized transaction requires immediate human security agent investigation.",
                "trigger_rule": "financial_fraud_security",
            }

        if intent == Intent.ACCOUNT_APPLE_ID.value or "2fa" in text_lower or "verification code" in text_lower:
            if any(kw in text_lower for kw in [
                "changed number", "lost phone", "stuck", "recovery taking",
                "no access to phone", "locked out", "cant receive",
            ]):
                return {
                    "should_escalate": True,
                    "escalation_reason": "Complex 2FA authentication factor loss requires manual identity verification by senior support staff.",
                    "trigger_rule": "account_2fa_identity_lock",
                }

        hardware_damage = [
            "water", "lake", "pool", "dropped", "shattered", "broken screen",
            "crack", "cracked", "won't turn on", "dead", "sticking",
        ]
        if intent == Intent.HARDWARE_REPAIR_WARRANTY.value or any(kw in text_lower for kw in hardware_damage):
            if any(kw in text_lower for kw in hardware_damage):
                return {
                    "should_escalate": True,
                    "escalation_reason": "Physical hardware damage or liquid exposure requires store repair appointment or mail-in service inspection.",
                    "trigger_rule": "hardware_physical_damage",
                }

        if intent == Intent.ORDER_SHIPPING.value or "order" in text_lower:
            if any(kw in text_lower for kw in [
                "missing", "never received", "stolen package", "says delivered but",
                "lost", "not delivered", "didn't arrive",
            ]):
                return {
                    "should_escalate": True,
                    "escalation_reason": "Lost package claim marked delivered requires human logistics claim resolution.",
                    "trigger_rule": "logistics_missing_delivery",
                }

        frustration_keywords = [
            "human", "manager", "representative", "lawyer", "sue",
            "terrible support", "useless agent", "speak to someone",
        ]
        if any(kw in text_lower for kw in frustration_keywords):
            return {
                "should_escalate": True,
                "escalation_reason": "Customer explicitly requested human specialist or expressed severe dissatisfaction.",
                "trigger_rule": "explicit_human_request",
            }

        return {
            "should_escalate": False,
            "escalation_reason": f"Routine {intent} query resolvable via standard self-service portal or automated troubleshooting guidance.",
            "trigger_rule": "auto_handle_routine",
        }
