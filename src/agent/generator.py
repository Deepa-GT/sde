"""
Response Generator module for Apple Support AI Agent.
Drafts replies grounded in retrieved historical resolutions while adhering to Twitter character limits.
"""

from typing import List, Dict, Any

from src.agent.taxonomy import Intent


class ResponseGenerator:
    """Generates customer support replies grounded in RAG retrieval context and policy constraints."""

    def __init__(self, brand_handle: str = "@AppleSupport"):
        self.brand_handle = brand_handle

    def generate(
        self,
        customer_text: str,
        intent: str,
        retrieved_context: List[Dict[str, Any]],
        escalation_info: Dict[str, Any],
    ) -> str:
        """Generate a customer-facing tweet response grounded in historical resolutions."""
        should_escalate = escalation_info.get("should_escalate", False)

        if should_escalate:
            trigger = escalation_info.get("trigger_rule", "")
            if trigger == "safety_thermal_hazard":
                reply = "Your safety is our top priority! Please turn off your device immediately, disconnect chargers, and send us a DM right away so we can assist. http://apple.co/DM"
            elif trigger == "financial_fraud_security":
                reply = "We take security very seriously. Please DM us your Apple ID email address immediately so our account security team can investigate. http://apple.co/DM"
            elif trigger == "hardware_physical_damage":
                reply = "We can help explore your service options. Please visit http://apple.co/Repair to check estimate costs or DM us to locate an Authorized Provider."
            elif trigger == "logistics_missing_delivery":
                reply = "We're sorry to hear about your delivery! Please DM us your order number and shipping address so our logistics team can track it down. http://apple.co/DM"
            else:
                reply = "We want to help resolve this for you right away. Please send us a Direct Message with your account details so a specialist can take a look: http://apple.co/DM"
        else:
            best_historical_reply = ""
            if retrieved_context:
                best_historical_reply = retrieved_context[0].get("historical_reply", "")

            if best_historical_reply and len(best_historical_reply) >= 30:
                reply = best_historical_reply
            elif intent == Intent.BILLING_SUBSCRIPTION.value:
                reply = "You can view your purchase history and request a refund directly at http://reportaproblem.apple.com. To manage subscriptions, check Settings > Apple ID."
            elif intent == Intent.DEVICE_TROUBLESHOOTING.value:
                reply = "Let's try a force restart! Quickly press Vol Up, Vol Down, then hold Side Button until the Apple logo appears. More tips: http://apple.co/Troubleshoot"
            elif intent == Intent.ACCOUNT_APPLE_ID.value:
                reply = "You can quickly reset your Apple ID password or unlock your account at http://iforgot.apple.com from any trusted browser."
            elif intent == Intent.HARDWARE_REPAIR_WARRANTY.value:
                reply = "Check your warranty coverage status by entering your serial number at http://checkcoverage.apple.com or book a Genius Bar visit: http://apple.co/GeniusBar"
            elif intent == Intent.SOFTWARE_UPDATE.value:
                reply = "Ensure your device is connected to Wi-Fi & power, then check Settings > General > Software Update. If stuck, delete download & retry."
            elif intent == Intent.ORDER_SHIPPING.value:
                reply = "Track your online order shipment and delivery status anytime by signing in at http://apple.co/OrderStatus."
            else:
                reply = "Thanks for reaching out! Please check http://apple.co/Support for helpful guides, or DM us your device model for specific tips."

        if len(reply) > 280:
            reply = reply[:277] + "..."

        return reply
