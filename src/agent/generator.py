"""
Response Generator module for Apple Support AI Agent.
Drafts friendly, warm, empathetic, and responsive replies grounded in official Apple resolutions
while strictly adhering to Twitter's 280-character limit.
"""

from typing import List, Dict, Any
from src.agent.taxonomy import Intent


class ResponseGenerator:
    """Generates customer support replies with a friendly, empathetic, and responsive brand voice."""

    def __init__(self, brand_handle: str = "@AppleSupport"):
        self.brand_handle = brand_handle

    def generate(
        self,
        customer_text: str,
        intent: str,
        retrieved_context: List[Dict[str, Any]],
        escalation_info: Dict[str, Any],
    ) -> str:
        """
        Generate a customer-facing tweet reply that is friendly, empathetic, actionable,
        and strictly compliant with Twitter's 280-character boundary.
        """
        should_escalate = escalation_info.get("should_escalate", False)

        if should_escalate:
            trigger = escalation_info.get("trigger_rule", "")
            if trigger == "safety_thermal_hazard":
                reply = (
                    "Your safety is our absolute priority! Please power off your device immediately, "
                    "disconnect any chargers, and send us a quick DM right away so our safety team can "
                    "assist you personally: http://apple.co/DM"
                )
            elif trigger == "financial_fraud_security":
                reply = (
                    "We know unexpected charges are alarming, and we take security very seriously. "
                    "Please DM us your Apple ID email right away so our dedicated security team can "
                    "investigate and protect your account: http://apple.co/DM"
                )
            elif trigger == "hardware_physical_damage":
                reply = (
                    "We're so sorry to hear about the damage! We'd love to help explore your repair options. "
                    "Check estimated service costs at http://apple.co/Repair or send us a DM so we can "
                    "help arrange an appointment for you."
                )
            elif trigger == "logistics_missing_delivery":
                reply = (
                    "We know how exciting it is to receive your order, and we're so sorry for the delay! "
                    "Please DM us your order number and shipping address so our logistics team can track "
                    "it down for you: http://apple.co/DM"
                )
            else:
                reply = (
                    "We're right here and want to help resolve this for you as quickly as possible! "
                    "Please join us in a Direct Message with your details so a specialist can take a "
                    "closer look with you: http://apple.co/DM"
                )
        else:
            # Check if retrieved historical context provides a high-quality reply
            best_historical_reply = ""
            if retrieved_context:
                best_historical_reply = retrieved_context[0].get("historical_reply", "")

            # If historical reply is available, ensure it has a warm tone
            if best_historical_reply and 40 <= len(best_historical_reply) <= 240:
                reply = best_historical_reply
                if not any(w in reply.lower() for w in ["help", "happy", "sorry", "welcome", "glad"]):
                    reply = f"We're happy to help! {reply}"
            elif intent == Intent.BILLING_SUBSCRIPTION.value:
                reply = (
                    "We're happy to help with your billing! You can review your purchase history and "
                    "request a refund anytime at http://reportaproblem.apple.com. To manage active "
                    "subscriptions, visit Settings > Apple ID > Subscriptions."
                )
            elif intent == Intent.DEVICE_TROUBLESHOOTING.value:
                reply = (
                    "We want your device running smoothly! A quick restart often does wonders: press Vol Up, "
                    "Vol Down, then hold the Side button until the Apple logo appears. More helpful tips: "
                    "http://apple.co/Troubleshoot"
                )
            elif intent == Intent.ACCOUNT_APPLE_ID.value:
                reply = (
                    "We're here to help you get back into your account safely! You can quickly reset your "
                    "password or unlock your Apple ID anytime at http://iforgot.apple.com from any trusted "
                    "browser."
                )
            elif intent == Intent.HARDWARE_REPAIR_WARRANTY.value:
                reply = (
                    "We're glad to help explore your coverage! You can check your warranty status at "
                    "http://checkcoverage.apple.com or let us help you book a Genius Bar visit: "
                    "http://apple.co/GeniusBar. We're always here for you!"
                )
            elif intent == Intent.SOFTWARE_UPDATE.value:
                reply = (
                    "We're here to help get your update finished! Make sure you're on Wi-Fi and power, "
                    "then head to Settings > General > Software Update. If stuck, delete the download in "
                    "iPhone Storage and retry."
                )
            elif intent == Intent.ORDER_SHIPPING.value:
                reply = (
                    "We're so excited for your new gear! You can track your shipment and delivery updates "
                    "anytime by signing in at http://apple.co/OrderStatus. Let us know if you need any extra help!"
                )
            else:
                reply = (
                    "Thanks for reaching out to us! We're always here and happy to help. Check out helpful "
                    "guides at http://apple.co/Support or send us a quick DM with your device model so we "
                    "can assist you directly!"
                )

        # Strict Twitter 280-character boundary guarantee
        if len(reply) > 280:
            reply = reply[:277].rstrip() + "..."

        return reply
