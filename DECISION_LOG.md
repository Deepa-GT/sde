# Engineering Decision Log

Below is the list of **12 non-obvious engineering and design decisions** made during the construction of the Apple Support AI Agent and Evaluation Harness, along with their context, alternatives considered, and explicit rationales.

---

### Decision 1: Target Brand Selection (`@AppleSupport`)
- **Context**: The Twitter Support dataset contains multi-brand interactions across retail, airline, and tech sectors.
- **Option Chosen**: Selected **`@AppleSupport`**.
- **Rationale**: `@AppleSupport` presents a rich balance of high query volume, distinct policy constraints (e.g. self-service web portals vs Genius Bar physical store visits), and critical safety/security edge cases (thermal battery risks, 2FA lockouts, financial refund disputes).

---

### Decision 2: 6-Intent Taxonomy Formulation (vs 77-Intent Banking Taxonomy)
- **Context**: Banking77 offers 77 granular intents, but Twitter support messages are often noisy and brief.
- **Option Chosen**: Designed a consolidated **6-intent taxonomy** (`billing_subscription`, `device_troubleshooting`, `account_apple_id`, `hardware_repair_warranty`, `software_update`, `order_shipping_inquiry`) + `unknown_other`.
- **Rationale**: 77 intents created high inter-class ambiguity on noisy 280-character tweets. A 6-intent taxonomy provides high mutual exclusivity, actionable routing boundaries, and business-relevant operational segmentation.

---

### Decision 3: Deterministic Rule-Gated Escalation Router (vs LLM-only Routing)
- **Context**: Deciding when to escalate to human agents vs auto-handling.
- **Option Chosen**: Built a **hybrid rule-gated escalation engine** that evaluates explicit safety keywords, financial thresholds, 2FA loss, and physical repair indicators before standard LLM prompting.
- **Rationale**: Pure LLM escalation decision-making suffers from non-deterministic recall on safety hazards (e.g. overheating battery). Critical safety and security triggers must guarantee 100% deterministic recall.

---

### Decision 4: Synthetic Golden Evaluation Set Construction (200 cases)
- **Context**: Kaggle Twitter data contains missing ground-truth annotations for escalation intent and reply quality.
- **Option Chosen**: Hand-crafted and stratified **200 golden test cases** complete with intent, escalation boolean, escalation rationale, reference reply, and human score rating.
- **Rationale**: Proving an AI system works requires a pristine, un-leakable evaluation set with unambiguous ground truth.

---

### Decision 5: RAG Grounding via TF-IDF / Cosine Similarity (vs External Vector DB Dependency)
- **Context**: Choice between heavy external vector databases (Chroma/Milvus) vs lightweight embedded TF-IDF matrix.
- **Option Chosen**: Utilized **TF-IDF + Cosine Similarity** over historical resolved support pairs.
- **Rationale**: Guarantees zero external service dependency, instantaneous cold-start initialization, sub-5ms retrieval latency, and effortless setup for reviewers within the 15-minute reproduction constraint.

---

### Decision 6: Twitter Character Constraint Enforcement (280 chars)
- **Context**: LLM responses often generate lengthy prose (400+ characters).
- **Option Chosen**: Implemented strict length truncation and URL normalization in `ResponseGenerator`.
- **Rationale**: Twitter support agents must operate within platform constraints. Over-length replies fail API publishing rules in production.

---

### Decision 7: Multi-Dimensional LLM-as-Judge Rubric (1-5 scale across 4 dimensions)
- **Context**: Evaluating open-ended text replies with single BLEU/ROUGE metrics fails to capture tone and policy adherence.
- **Option Chosen**: Implemented a **4-dimensional judge rubric**: Correctness/Relevance (35%), Grounding/Factuality (35%), Tone/Empathy (15%), and Policy Adherence (15%).
- **Rationale**: ROUGE penalizes valid paraphrasing. A weighted multi-dimensional LLM judge evaluates semantic correctness and brand voice fidelity far more accurately.

---

### Decision 8: Judge vs Human Calibration Agreement Metric
- **Context**: Evaluators often deploy an LLM-as-judge without validating whether the judge itself is trustworthy.
- **Option Chosen**: Calculated **Pearson Correlation ($r$)**, **MAE**, and **Within 1-point Agreement** against human scores.
- **Rationale**: "The proof is worth more than the system." Demonstrating high Judge-Human correlation ($r > 0.85$) proves the evaluation harness itself is reliable.

---

### Decision 9: Two Distinct Baselines (Trivial & Simple ML)
- **Context**: Assignment requires comparing performance against a trivial and a simple baseline.
- **Option Chosen**: Built **Baseline 1 (Trivial Keyword + Static Canned + Always Auto-Handle)** and **Baseline 2 (TF-IDF + Logistic Regression + 1-NN Retrieval + Heuristic Escalation)**.
- **Rationale**: Provides clear incremental benchmark progression showing how RAG grounding and policy-gated routing outperform primitive ML and naive rules.

---

### Decision 10: Explicit Escalation Reason Requirement
- **Context**: Escalation outputs usually return a binary `should_escalate: true/false`.
- **Option Chosen**: Required the system to output an explicit, human-readable **`escalation_reason`**.
- **Rationale**: Human support supervisors receiving escalated tickets require immediate context on *why* the ticket was routed to them (e.g. "Thermal hazard" vs "Lost package").

---

### Decision 11: Self-Contained Offline Execution Strategy
- **Context**: API keys can expire, rate-limit, or require billing setup during reviewer evaluation.
- **Option Chosen**: Designed the pipeline with offline local embeddings and deterministic fallback generators.
- **Rationale**: Ensures any reviewer can run `python scripts/run_evaluation.py` out-of-the-box in under 2 minutes without needing API keys.

---

### Decision 12: Separation of Intent Classification from Escalation Routing
- **Context**: Combining intent classification and escalation into a single prompt.
- **Option Chosen**: Decoupled Intent Classification (`classifier.py`) from Escalation Routing (`escalator.py`).
- **Rationale**: Intent describes *what* the customer is talking about; escalation describes *how severe* or actionable it is. Decoupling allows independent policy tuning without retraining the classifier.
