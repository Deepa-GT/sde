# Golden Evaluation Set: Sampling & Labelling Methodology

## Overview
This document outlines the sampling methodology, taxonomy design, and annotation guidelines used to create the **Golden Evaluation Set** (`data/golden_eval_set.json`), consisting of **200 hand-labelled customer support test cases** targeting `@AppleSupport`.

---

## 1. Intent Taxonomy Design
Based on exploratory analysis of Twitter customer support logs for tech brands, we formulated a 6-intent taxonomy plus an out-of-scope intent:

1. **`billing_subscription`**: App Store purchases, subscriptions, double charges, refund requests.
2. **`device_troubleshooting`**: Hardware performance, battery drain, display glitches, connectivity (Wi-Fi/Bluetooth).
3. **`account_apple_id`**: Password resets, 2FA code delivery, locked/disabled accounts, unauthorized access.
4. **`hardware_repair_warranty`**: Physical damage, screen replacements, Genius Bar appointments, AppleCare+ coverage.
5. **`software_update`**: iOS/macOS update installation errors, post-update app crashes, beta program inquiries.
6. **`order_shipping_inquiry`**: Online Apple Store delivery tracking, address changes, trade-in kit returns.
7. **`unknown_other`**: Out-of-scope general inquiries, greetings, or non-technical chatter.

---

## 2. Sampling Strategy
To ensure the evaluation set is representative of real-world operational challenges, we applied **stratified random sampling** combined with **hard negative injection**:

- **Intent Stratification**: ~30 examples per primary intent category to prevent class imbalance bias.
- **Escalation Balance**: ~35% of examples are marked `should_escalate: true` (e.g. fraudulent transactions, physical safety hazards, severe account compromise, lost shipments).
- **Text Length & Noise Variety**: Includes short queries ("Password reset?"), multi-sentence complaints, uppercase urgent tweets ("URGENT @AppleSupport!"), and typos/abbreviations ("my iphon 14 screen wont wrk").
- **Edge Cases**: Includes out-of-scope tweets (e.g., cookie recipes, general praise) to test boundaries and safety guardrails.

---

## 3. Annotation Guidelines

Each sample in the golden set is annotated with 5 core fields:

| Field | Description | Example / Allowed Values |
|-------|-------------|--------------------------|
| `id` | Unique sample identifier | `GOLD_001` |
| `customer_text` | Raw customer tweet content | `"@AppleSupport I was double charged $14.99 for Apple TV+!"` |
| `ground_truth_intent` | Classified intent from taxonomy | `billing_subscription` |
| `should_escalate` | Boolean routing decision | `true` / `false` |
| `escalation_reason` | Explicit rationale for routing decision | `"High-value unauthorized charge requiring human agent security check."` |
| `reference_reply` | High-quality human support response | `"We can help! You can review your purchase history and request a refund at http://reportaproblem.apple.com."` |
| `human_quality_score` | Baseline human rating (1-5 scale) for LLM-as-Judge calibration | `5` |

### Escalation Criteria
A case **MUST** be escalated (`should_escalate: true`) if ANY of the following conditions are met:
1. **Security/Fraud**: Unauthorized charges, stolen credit cards, or hacked Apple IDs.
2. **Safety Risk**: Overheating devices, smoking batteries, or physical hazards.
3. **Complex Identity Loss**: Loss of 2FA phone number with locked credentials requiring manual verification.
4. **Logistics Claims**: Package marked as delivered but missing, or lost high-value order shipments.
5. **Physical Hardware Service**: Cracks, water damage, or component replacements requiring store/mail-in repair.

Otherwise, if the issue can be resolved via official Apple self-service web portals or standard troubleshooting steps, it is marked **`should_escalate: false`**.
