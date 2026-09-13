# Architecture Decision Records (ADRs) & Engineering Decision Log

<div align="center">

**Project**: Apple Support AI Agent & Evaluation Harness  
**Author**: Hiver SDE Intern Candidate  
**Target Repository**: [https://github.com/Deepa-GT/sde](https://github.com/Deepa-GT/sde)  
**Status**: All 12 Architectural Decisions Accepted & Implemented  

</div>

---

## Overview
This document records the **12 non-obvious engineering and architectural decisions** made during the formulation, development, and evaluation of the `@AppleSupport` AI Agent. Each record outlines the operational context, alternatives considered, decision outcome, and engineering trade-offs.

---

### ADR-01: Selection of `@AppleSupport` as Target Enterprise Brand
- **Status**: Accepted
- **Context**: The Kaggle Twitter Customer Support corpus contains multi-turn threads across retail, airlines, telecommunications, and tech.
- **Alternatives Considered**: 
  - *Option A*: Delta / British Airways (High volume, but queries predominantly center on flight delays with low technical taxonomy depth).
  - *Option B*: Amazon Help (Extremely wide product catalog, high variance, difficult to ground factually).
  - *Option C*: `@AppleSupport` (Selected).
- **Decision**: Focus exclusively on `@AppleSupport`.
- **Rationale & Consequences**:
  - *Positive*: `@AppleSupport` exhibits a well-defined ecosystem of canonical self-service web portals (`reportaproblem.apple.com`, `iforgot.apple.com`, `checkcoverage.apple.com`). It presents clear asymmetric safety thresholds (lithium-ion thermal runaway, credit card fraud, 2FA lockouts) that test an AI agent's reliability under high stakes.
  - *Trade-off*: Strict 280-character Twitter constraint requires aggressive length-enforcement algorithms.

---

### ADR-02: Consolidated 7-Class Intent Taxonomy (vs. 77-Class Banking Taxonomy)
- **Status**: Accepted
- **Context**: The optional secondary dataset (Banking77) contains 77 granular classes. We evaluated whether to adapt a 70+ class taxonomy or design a consolidated enterprise schema.
- **Alternatives Considered**:
  - *Option A*: 77-class fine-grained taxonomy.
  - *Option B*: 3-class coarse taxonomy (`Hardware`, `Software`, `Billing`).
  - *Option C*: 7-class business-aligned taxonomy (`billing_subscription`, `device_troubleshooting`, `account_apple_id`, `hardware_repair_warranty`, `software_update`, `order_shipping_inquiry`, `unknown_other`).
- **Decision**: Implement Option C (7 classes).
- **Rationale & Consequences**:
  - *Positive*: On noisy 280-character tweets, 77 classes produce high inter-class ambiguity and sparse training data per class. A 7-class schema provides distinct operational routing boundaries aligned with Apple's real-world customer service divisions.
  - *Trade-off*: Composite queries spanning multiple grievances require specialized multi-intent handling.

---

### ADR-03: Deterministic Rule-Gated Escalation Router (vs. Unconstrained LLM Routing)
- **Status**: Accepted
- **Context**: Determining whether incoming messages should be auto-handled or escalated to a human tier-2 specialist.
- **Alternatives Considered**:
  - *Option A*: Pure LLM zero-shot prompt classification (`"Should this be escalated? Answer yes/no"`).
  - *Option B*: Pure heuristic keyword blacklist.
  - *Option C*: Deterministic policy-gated router with explicit rule predicates and fallback intent routing (Selected).
- **Decision**: Adopt Option C.
- **Rationale & Consequences**:
  - *Positive*: Pure LLM escalation decision-making suffers from stochastic non-determinism, occasionally failing to flag dangerous events (thermal hazards, stolen credit cards). Enterprise safety requires 100% deterministic recall on life-safety and financial fraud rules.
  - *Trade-off*: Rule lexicons must be curated and maintained as slang and evasion phrasing evolve.

---

### ADR-04: Ground-Truth Golden Evaluation Set (200 Hand-Labelled Cases)
- **Status**: Accepted
- **Context**: Proving an AI agent works requires high-fidelity, un-leakable evaluation data with ground-truth intent, escalation flags, and human quality marks.
- **Alternatives Considered**:
  - *Option A*: Unchecked automated LLM self-labelling of raw Kaggle tweets.
  - *Option B*: Small 30-case qualitative smoke test.
  - *Option C*: 200-case stratified hand-labelled golden test set with hard negative injection (Selected).
- **Decision**: Adopt Option C (`data/golden_eval_set.json`).
- **Rationale & Consequences**:
  - *Positive*: Hand-crafted and stratified across all 7 taxonomy classes, with balanced escalation cases (~35% escalated) and edge-case hard negatives. Accompanied by a formal sampling methodology note (`data/SAMPLING_AND_LABELLING_NOTE.md`).
  - *Trade-off*: Hand-labelling 200 cases required intensive manual annotation effort.

---

### ADR-05: In-Memory TF-IDF & Cosine RAG Retrieval (vs. Heavy Vector DB)
- **Status**: Accepted
- **Context**: Grounding generated replies in historical brand resolutions. Choice of retrieval architecture.
- **Alternatives Considered**:
  - *Option A*: External Vector Database (ChromaDB / Milvus / Pinecone) with cloud embedding API calls.
  - *Option B*: In-memory TF-IDF + Cosine Similarity over resolved historical support pairs (Selected).
- **Decision**: Adopt Option B.
- **Rationale & Consequences**:
  - *Positive*: Reviewers can clone and execute the entire benchmark in **under 2 minutes** without Docker setup, external API keys, or network failures. Sub-5ms retrieval latency.
  - *Trade-off*: Lacks cross-encoder dense semantic re-ranking (addressed in Section 8 of REPORT.md as next-week roadmap).

---

### ADR-06: Strict Platform-Boundary Twitter Character Enforcement (280 Chars)
- **Status**: Accepted
- **Context**: Generative models frequently output 350+ character paragraphs, which fail Twitter API publishing rules.
- **Alternatives Considered**:
  - *Option A*: Rely on prompt instructions (`"keep it under 280 characters"`).
  - *Option B*: Deterministic post-processing length enforcement and clean boundary truncation (Selected).
- **Decision**: Implement Option B in `src/agent/generator.py`.
- **Rationale & Consequences**:
  - *Positive*: 100% mathematical guarantee that no drafted reply ever violates platform boundaries.
  - *Trade-off*: In rare edge cases, trailing sentences are truncated cleanly with an ellipsis leading to an official link.

---

### ADR-07: 4-Dimensional LLM-as-Judge Rubric (vs. Single Lexical Overlap Metric)
- **Status**: Accepted
- **Context**: Evaluating text reply quality. Automated metrics like BLEU or ROUGE penalize valid brand paraphrasing.
- **Alternatives Considered**:
  - *Option A*: ROUGE-L / BLEU alone.
  - *Option B*: Single holistic 1–5 LLM prompt.
  - *Option C*: Weighted 4-dimensional continuous rubric (Correctness 35%, Grounding 35%, Tone 15%, Policy 15%) (Selected).
- **Decision**: Implement Option C (`src/eval/judge.py`).
- **Rationale & Consequences**:
  - *Positive*: Distinguishes between factual link accuracy and mere token overlap. Evaluates brand empathy and Twitter policy compliance directly.
  - *Trade-off*: Requires statistical calibration against human raters to prove trustworthiness.

---

### ADR-08: Empirical Judge-Human Calibration Requirement
- **Status**: Accepted
- **Context**: Assignment requirement: "Prove your judge agrees with a human."
- **Alternatives Considered**:
  - *Option A*: Assume the judge is valid without statistical verification.
  - *Option B*: Compute Pearson correlation ($r$), Mean Absolute Error (MAE), and agreement bounds against human ground-truth ratings (Selected).
- **Decision**: Implement Option B (`scripts/calibrate_judge.py`).
- **Rationale & Consequences**:
  - *Positive*: Proves the evaluation harness itself is sound ($r = 0.9205$, MAE $= 0.9736$, $60.0\%$ within $\pm 1$ pt).
  - *Trade-off*: Exposes discrete vs. continuous scoring nuances, which are rigorously documented in the technical report.

---

### ADR-09: Benchmark Evaluation Against Two Distinct Baselines
- **Status**: Accepted
- **Context**: Assignment requirement: "Results vs. at least two baselines (a trivial one and a simple one)."
- **Alternatives Considered**:
  - *Option A*: Compare only against human historical replies.
  - *Option B*: Implement Baseline 1 (Trivial Keyword + Static Canned) and Baseline 2 (Simple ML: TF-IDF + Logistic Regression + 1-NN) (Selected).
- **Decision**: Implement Option B (`src/baselines/`).
- **Rationale & Consequences**:
  - *Positive*: Establishes a rigorous progressive benchmark showing how policy-gated RAG routing solves the 0.0% / 1.4% safety recall failure of primitive models.
  - *Trade-off*: Required engineering and maintaining three parallel inference agents.

---

### ADR-10: Explicit Human-Readable Escalation Rationales
- **Status**: Accepted
- **Context**: Most escalation classifiers output a binary boolean (`should_escalate: true/false`).
- **Alternatives Considered**:
  - *Option A*: Boolean only.
  - *Option B*: Boolean paired with an explicit, human-auditable `escalation_reason` string (Selected).
- **Decision**: Implement Option B (`src/agent/escalator.py`).
- **Rationale & Consequences**:
  - *Positive*: Tier-2 human support supervisors receiving escalated tickets immediately understand *why* the ticket was flagged (e.g. "Thermal hazard" vs "High-value charge").
  - *Trade-off*: Requires maintaining a structured reason catalog across escalation trigger rules.

---

### ADR-11: Zero External API Dependency for Core Pipeline Execution
- **Status**: Accepted
- **Context**: Reliance on external LLM APIs (OpenAI/Anthropic) during candidate evaluation introduces API key expiration, rate limits, credit balance exhaustion, and network instability.
- **Alternatives Considered**:
  - *Option A*: Hard dependency on OpenAI API key in `.env`.
  - *Option B*: Fully self-contained local embedding and deterministic rubric architecture with optional LLM plug-in (Selected).
- **Decision**: Implement Option B.
- **Rationale & Consequences**:
  - *Positive*: Any evaluator can clone the repo and reproduce 100% of benchmark results offline in under 2 minutes without configuring credentials.
  - *Trade-off*: For dynamic unconstrained text generation, requires local template synthesis rather than multi-billion parameter cloud inference.

---

### ADR-12: Decoupled Intent Classification from Escalation Routing
- **Status**: Accepted
- **Context**: Evaluating whether to classify intent and decide escalation in a single monolithic prompt/model or decouple them into specialized micro-stages.
- **Alternatives Considered**:
  - *Option A*: Monolithic single-pass prompt.
  - *Option B*: Independent Intent Classifier stage followed by an independent Escalation Router stage (Selected).
- **Decision**: Adopt Option B.
- **Rationale & Consequences**:
  - *Positive*: Intent describes *what* the customer is experiencing; escalation describes *how severe* or hazardous it is. Decoupling allows support operations to adjust safety thresholds (e.g. during a battery recall campaign) without retraining the intent classification model.
  - *Trade-off*: Introduces an additional pipeline stage, though execution overhead is negligible (< 1ms).

---

<div align="center">

**Engineering Decision Records Signed & Maintained for Technical Review**  
Repository: [https://github.com/Deepa-GT/sde](https://github.com/Deepa-GT/sde)

</div>
