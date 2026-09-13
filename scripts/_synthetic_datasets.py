"""Synthetic fallback dataset generator when Kaggle download is unavailable."""

import json
import os
import random

import pandas as pd


def generate_historical_dataset() -> pd.DataFrame:
    templates = [
        ("I was charged twice for Apple Music subscription this month!", "billing_subscription",
         "We can check your billing history. Please DM us with your Apple ID email. http://apple.co/DM"),
        ("My iPhone 14 battery is draining super fast after the latest update.", "device_troubleshooting",
         "Battery consumption may increase for 48 hours after an update. Check Settings > Battery."),
        ("My Apple ID is locked for security reasons and I can't log in.", "account_apple_id",
         "Unlock or reset your password at http://iforgot.apple.com from any trusted browser."),
        ("I dropped my iPhone 15 Pro and the back glass shattered completely.", "hardware_repair_warranty",
         "Schedule a service appointment at http://apple.co/Repair or an Authorized Provider."),
        ("iOS 17.5 update gets stuck on 'Preparing Update...' for hours.", "software_update",
         "Delete the update in Settings > General > iPhone Storage, restart, and retry over Wi-Fi."),
        ("Where is my order W123456789? It was supposed to arrive yesterday.", "order_shipping_inquiry",
         "Track delivery status at http://apple.co/OrderStatus."),
    ]
    rows = []
    id_counter = 1000
    for _ in range(25):
        for text, intent, reply in templates:
            id_counter += 1
            prefix = random.choice(["", "Hey @AppleSupport, ", "@AppleSupport ", "Need help: "])
            rows.append({
                "tweet_id": f"TW_{id_counter}",
                "author_id": f"User_{random.randint(10000, 99999)}",
                "customer_tweet": f"{prefix}{text}",
                "intent": intent,
                "historical_reply": reply,
                "in_response_to_tweet_id": None,
            })
    return pd.DataFrame(rows)


def generate_golden_eval_set():
    seeds = [
        {"text": "@AppleSupport I was double charged $14.99 for Apple TV+ this month. Need refund!",
         "intent": "billing_subscription", "should_escalate": False,
         "escalation_reason": "Standard refund via reportaproblem.apple.com.",
         "reference_reply": "Review purchase history and submit a refund at http://reportaproblem.apple.com.",
         "human_quality_score": 5},
        {"text": "@AppleSupport Someone stole my credit card and made $500 of unauthorized App Store purchases!",
         "intent": "billing_subscription", "should_escalate": True,
         "escalation_reason": "Fraudulent transaction requires security escalation.",
         "reference_reply": "Please DM us immediately with your phone number so a specialist can secure your account.",
         "human_quality_score": 5},
        {"text": "@AppleSupport My iPhone screen went red, started smoking, and got burning hot!",
         "intent": "device_troubleshooting", "should_escalate": True,
         "escalation_reason": "Thermal hazard requires safety escalation.",
         "reference_reply": "Turn off the device immediately, disconnect chargers, and DM us your contact info.",
         "human_quality_score": 5},
    ]
    items = []
    counter = 1
    while len(items) < 200:
        for item in seeds:
            if counter > 200:
                break
            items.append({
                "id": f"GOLD_{counter:03d}",
                "customer_text": item["text"],
                "ground_truth_intent": item["intent"],
                "should_escalate": item["should_escalate"],
                "escalation_reason": item["escalation_reason"],
                "reference_reply": item["reference_reply"],
                "human_quality_score": item["human_quality_score"],
            })
            counter += 1
    return items


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_historical_dataset()
    df.to_csv("data/raw_support_tweets.csv", index=False)
    golden = generate_golden_eval_set()
    with open("data/golden_eval_set.json", "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2)
    print(f"Saved {len(df)} synthetic pairs and {len(golden)} golden cases.")
