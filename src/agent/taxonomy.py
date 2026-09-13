"""
Intent Taxonomy definition for Apple Support AI Agent.
Defines intent categories, descriptions, keywords, and escalation rules.
"""

from enum import Enum
from typing import Dict, List, Any

class Intent(str, Enum):
    BILLING_SUBSCRIPTION = "billing_subscription"
    DEVICE_TROUBLESHOOTING = "device_troubleshooting"
    ACCOUNT_APPLE_ID = "account_apple_id"
    HARDWARE_REPAIR_WARRANTY = "hardware_repair_warranty"
    SOFTWARE_UPDATE = "software_update"
    ORDER_SHIPPING = "order_shipping_inquiry"
    UNKNOWN_OTHER = "unknown_other"

INTENT_DESCRIPTIONS: Dict[Intent, str] = {
    Intent.BILLING_SUBSCRIPTION: "Queries regarding App Store purchases, subscriptions, refunds, billing errors, or payment methods.",
    Intent.DEVICE_TROUBLESHOOTING: "Issues with iPhone, iPad, Mac, or Apple Watch performance, battery drain, Bluetooth, Wi-Fi, or screen freezes.",
    Intent.ACCOUNT_APPLE_ID: "Problems signing into Apple ID, two-factor authentication, locked accounts, or password resets.",
    Intent.HARDWARE_REPAIR_WARRANTY: "Physical damage, AppleCare coverage, broken screens, battery replacement, or repair appointment scheduling.",
    Intent.SOFTWARE_UPDATE: "Issues related to iOS, macOS, watchOS updates, installation errors, or feature bugs post-update.",
    Intent.ORDER_SHIPPING: "Questions about new Apple Store online orders, delivery status, shipping delays, or trade-in kits.",
    Intent.UNKNOWN_OTHER: "Out-of-scope inquiries, general compliments, or unintelligible/spam messages."
}

INTENT_KEYWORDS: Dict[Intent, List[str]] = {
    Intent.BILLING_SUBSCRIPTION: [
        "charged", "charge", "refund", "subscription", "app store", "billing", "receipt",
        "payment", "invoice", "double charge", "cost", "cancel subscription", "money"
    ],
    Intent.DEVICE_TROUBLESHOOTING: [
        "battery", "drain", "freeze", "frozen", "screen", "wifi", "bluetooth", "touch",
        "lag", "restart", "black screen", "camera", "speaker", "sound", "mic", "overheating"
    ],
    Intent.ACCOUNT_APPLE_ID: [
        "apple id", "password", "locked", "disabled", "sign in", "login", "2fa", "verification",
        "security code", "icloudd", "account", "reset", "verification code"
    ],
    Intent.HARDWARE_REPAIR_WARRANTY: [
        "broken", "crack", "cracked", "repair", "applecare", "warranty", "store appointment",
        "genius bar", "replace battery", "water damage", "drop", "dropped", "hardware"
    ],
    Intent.SOFTWARE_UPDATE: [
        "ios", "update", "macos", "watchos", "software", "install", "bug", "beta",
        "stuck on update", "upgraded", "ios 17", "ios 18"
    ],
    Intent.ORDER_SHIPPING: [
        "order", "shipping", "delivery", "track", "ups", "fedex", "shipment", "trade-in",
        "trade in", "apple store online", "status", "arriving"
    ],
    Intent.UNKNOWN_OTHER: [
        "thanks", "thank you", "hello", "hi", "hey", "cool", "great", "ok"
    ]
}

def get_all_intents() -> List[str]:
    """Returns a list of string values for all supported intents."""
    return [intent.value for intent in Intent if intent != Intent.UNKNOWN_OTHER]
