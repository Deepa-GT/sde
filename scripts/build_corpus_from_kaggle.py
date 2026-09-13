"""
Build AppleSupport RAG corpus and golden evaluation set from the Kaggle
Customer Support on Twitter dataset (thoughtvector/customer-support-on-twitter).

Usage:
    python scripts/build_corpus_from_kaggle.py [--kaggle-path PATH]

If --kaggle-path is omitted, attempts kagglehub download automatically.
Outputs:
    data/raw_support_tweets.csv   (~800 resolved customer/reply pairs)
    data/golden_eval_set.json     (200 held-out labelled test cases)
"""

import argparse
import json
import os
import random
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.agent.taxonomy import Intent, INTENT_KEYWORDS  # noqa: E402


BRAND = "AppleSupport"
RANDOM_SEED = 42
CORPUS_SIZE = 800
GOLDEN_SIZE = 200


def resolve_kaggle_path(explicit: str | None) -> str:
    if explicit and os.path.exists(explicit):
        return explicit
    try:
        import kagglehub

        base = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
        candidate = os.path.join(base, "twcs", "twcs.csv")
        if os.path.exists(candidate):
            return candidate
    except Exception as exc:
        raise FileNotFoundError(
            "Could not locate Kaggle CSV. Pass --kaggle-path or install kagglehub."
        ) from exc
    raise FileNotFoundError(f"twcs.csv not found under {base}")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "@AppleSupport", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:500]


def label_intent(text: str) -> str:
    lower = text.lower()
    scores = {intent.value: 0.0 for intent in Intent}
    for intent_enum, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                scores[intent_enum.value] += 2.0 if " " in kw else 1.0
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else Intent.UNKNOWN_OTHER.value


def should_escalate(text: str, intent: str) -> tuple[bool, str]:
    lower = text.lower()
    safety = ["smoke", "smoking", "fire", "explode", "exploded", "burning", "sparks", "swollen"]
    if any(k in lower for k in safety):
        return True, "Physical safety hazard requires human escalation."
    fraud = ["stolen", "unauthorized", "fraud", "hacked", "hacker", "compromised"]
    if any(k in lower for k in fraud):
        return True, "Security or fraud concern requires human investigation."
    if intent == Intent.ACCOUNT_APPLE_ID.value and any(
        k in lower for k in ["2fa", "verification code", "lost phone", "changed number", "no access"]
    ):
        return True, "Complex Apple ID / 2FA recovery requires identity verification."
    if intent == Intent.HARDWARE_REPAIR_WARRANTY.value and any(
        k in lower for k in ["crack", "cracked", "shattered", "water", "dropped", "broken screen"]
    ):
        return True, "Physical hardware damage requires repair service."
    if intent == Intent.ORDER_SHIPPING.value and any(
        k in lower for k in ["never received", "missing", "lost", "stolen package", "delivered but"]
    ):
        return True, "Lost or missing shipment requires logistics investigation."
    if any(k in lower for k in ["lawyer", "sue", "manager", "speak to human", "representative"]):
        return True, "Customer explicitly requested human specialist."
    return False, f"Routine {intent} query suitable for self-service guidance."


def load_apple_pairs(csv_path: str) -> pd.DataFrame:
    """Extract inbound customer tweets paired with AppleSupport replies."""
    frames = []
    for chunk in pd.read_csv(csv_path, chunksize=100_000, low_memory=False):
        chunk["author_id"] = chunk["author_id"].astype(str)
        chunk["text"] = chunk["text"].astype(str)
        apple_mask = chunk["author_id"].str.contains(BRAND, case=False, na=False) | chunk[
            "text"
        ].str.contains(f"@{BRAND}", case=False, na=False)
        frames.append(chunk[apple_mask])
    df = pd.concat(frames, ignore_index=True)
    def normalize_id(val):
        if pd.isna(val) or str(val).strip().lower() in ("nan", ""):
            return None
        raw = str(val).replace(",", " ").split()
        return raw[0] if raw else None

    df["tweet_id"] = df["tweet_id"].apply(normalize_id)
    df["response_tweet_id"] = df["response_tweet_id"].apply(normalize_id)
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].apply(normalize_id)
    df = df.dropna(subset=["tweet_id"])

    by_id = df.set_index("tweet_id", drop=False)
    pairs = []
    customers = df[(df["inbound"] == True) & df["text"].str.contains(f"@{BRAND}", case=False, na=False)]

    for _, row in customers.iterrows():
        reply_text = None
        resp_id = row.get("response_tweet_id")
        if resp_id and resp_id in by_id.index:
            reply_row = by_id.loc[resp_id]
            if isinstance(reply_row, pd.DataFrame):
                reply_row = reply_row.iloc[0]
            if BRAND.lower() in str(reply_row["author_id"]).lower():
                reply_text = clean_text(reply_row["text"])

        if not reply_text and row.get("in_response_to_tweet_id"):
            parent_id = row["in_response_to_tweet_id"]
            if parent_id in by_id.index:
                parent = by_id.loc[parent_id]
                if isinstance(parent, pd.DataFrame):
                    parent = parent.iloc[0]
                if parent.get("inbound") == False and BRAND.lower() in str(parent["author_id"]).lower():
                    reply_text = clean_text(parent["text"])

        customer_text = clean_text(row["text"])
        if len(customer_text) < 15 or not reply_text or len(reply_text) < 20:
            continue
        intent = label_intent(customer_text)
        if intent == Intent.UNKNOWN_OTHER.value:
            continue
        pairs.append(
            {
                "tweet_id": row["tweet_id"],
                "author_id": row["author_id"],
                "customer_tweet": customer_text,
                "intent": intent,
                "historical_reply": reply_text[:280],
                "in_response_to_tweet_id": row.get("in_response_to_tweet_id"),
            }
        )

    return pd.DataFrame(pairs).drop_duplicates(subset=["customer_tweet"])


def build_golden_set(pairs_df: pd.DataFrame, corpus_ids: set) -> list:
    """Sample held-out real tweets for golden evaluation."""
    held_out = pairs_df[~pairs_df["tweet_id"].isin(corpus_ids)].copy()
    held_out["should_escalate"] = False
    held_out["escalation_reason"] = ""
    for idx, row in held_out.iterrows():
        esc, reason = should_escalate(row["customer_tweet"], row["intent"])
        held_out.at[idx, "should_escalate"] = esc
        held_out.at[idx, "escalation_reason"] = reason

    random.seed(RANDOM_SEED)
    stratified = []
    intents = [i.value for i in Intent if i != Intent.UNKNOWN_OTHER]
    per_intent = GOLDEN_SIZE // len(intents)
    esc_target = int(GOLDEN_SIZE * 0.35)

    for intent in intents:
        pool = held_out[held_out["intent"] == intent]
        if len(pool) >= per_intent:
            stratified.append(pool.sample(n=per_intent, random_state=RANDOM_SEED))
        elif len(pool) > 0:
            stratified.append(pool)

    golden_df = pd.concat(stratified, ignore_index=True).drop_duplicates(subset=["customer_tweet"])
    if len(golden_df) < GOLDEN_SIZE:
        remaining = held_out[~held_out["customer_tweet"].isin(golden_df["customer_tweet"])]
        extra = remaining.sample(n=min(GOLDEN_SIZE - len(golden_df), len(remaining)), random_state=RANDOM_SEED)
        golden_df = pd.concat([golden_df, extra], ignore_index=True)

    golden_df = golden_df.head(GOLDEN_SIZE)

    esc_count = golden_df["should_escalate"].sum()
    if esc_count < esc_target and len(held_out) > len(golden_df):
        non_esc = golden_df[~golden_df["should_escalate"]]
        esc_pool = held_out[held_out["should_escalate"]].head(esc_target - esc_count)
        golden_df = pd.concat([golden_df[golden_df["should_escalate"]], esc_pool, non_esc]).drop_duplicates(
            subset=["customer_tweet"]
        ).head(GOLDEN_SIZE)

    items = []
    for i, row in golden_df.reset_index(drop=True).iterrows():
        human_score = 5 if row["should_escalate"] or "http" in row["historical_reply"].lower() else 4
        items.append(
            {
                "id": f"GOLD_{i+1:03d}",
                "customer_text": row["customer_tweet"],
                "ground_truth_intent": row["intent"],
                "should_escalate": bool(row["should_escalate"]),
                "escalation_reason": row["escalation_reason"],
                "reference_reply": row["historical_reply"],
                "human_quality_score": human_score,
            }
        )
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-path", default=None, help="Path to twcs/twcs.csv")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    csv_path = resolve_kaggle_path(args.kaggle_path)
    print(f"Loading AppleSupport pairs from {csv_path} ...")
    pairs_df = load_apple_pairs(csv_path)
    print(f"Found {len(pairs_df)} unique customer/reply pairs.")

    random.seed(RANDOM_SEED)
    if len(pairs_df) > CORPUS_SIZE + GOLDEN_SIZE:
        corpus_df = pairs_df.sample(n=CORPUS_SIZE, random_state=RANDOM_SEED)
    else:
        corpus_df = pairs_df.sample(frac=0.8, random_state=RANDOM_SEED)

    corpus_ids = set(corpus_df["tweet_id"].tolist())
    corpus_df.to_csv("data/raw_support_tweets.csv", index=False)
    print(f"Saved {len(corpus_df)} pairs to data/raw_support_tweets.csv")

    golden = build_golden_set(pairs_df, corpus_ids)
    with open("data/golden_eval_set.json", "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(golden)} golden eval cases to data/golden_eval_set.json")


if __name__ == "__main__":
    main()
