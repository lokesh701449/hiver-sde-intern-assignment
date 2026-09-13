# AmazonHelp AI Customer Support Agent & Hybrid RAG System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Local LLM](https://img.shields.io/badge/Ollama-Qwen2.5--3B-orange.svg)](https://ollama.ai/)
[![Vector Index](https://img.shields.io/badge/FAISS-MiniLM--L6--v2-green.svg)](https://github.com/facebookresearch/faiss)
[![Evaluation Benchmark](https://img.shields.io/badge/Benchmark-Frozen%20Heldout%20(n%3D200)-purple.svg)](./evaluation/heldout_test_final.csv)

> **Hybrid RAG Support Agent Prototype — 65.5% intent accuracy, 95.0% escalation recall, and 58.0% end-to-end decision accuracy on a frozen 200-example heldout test set.**

---

## 1. Executive Summary & Benchmark Comparison

This repository contains an end-to-end **AI Customer Support Agent prototype** for e-commerce (`@AmazonHelp`), featuring multi-task **Intent Classification**, **Deterministic Escalation Routing**, **Historical RAG Retrieval**, and an **LLM-as-Judge Reply Quality Evaluation**.

### Final Heldout Benchmark Results (n=200 Frozen Test Set)

| System / Baseline | Architecture / Method | Evaluation Split | Intent Acc | Intent Macro F1 | Escalation Acc | Escalation Recall | Escalation F1 | Combined Acc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule-Based V1** | Keyword Heuristics | **HELDOUT** | 60.50% | 0.5447 | 78.50% | 84.38% | 0.8060 | 52.50% |
| **Rule-Based V2** | Refined Pattern Rules | **HELDOUT** | 58.50% | 0.4643 | 76.50% | 78.13% | 0.8418 | 50.50% |
| **TF-IDF + LogReg** | ML Classifier (1,000 feat) | **HELDOUT** | 53.00% | 0.4030 | **86.00%** | 93.75% | **0.9146** | 48.00% |
| **Hybrid RAG (Final)** | Ollama Qwen2.5:3B + Policy | **HELDOUT** | **65.50%** | **0.5836** | 84.00% | **95.00%** | 0.9048 | **58.00%** |
| *DistilBERT (Dev)* | Fine-Tuned Transformer | *DEV / CV Only* | *75.00%* | *0.6959* | *65.00%* | *N/A* | *0.7813* | *47.50%* |

> [!NOTE]
> **Transformer Model Note**: Fine-tuned DistilBERT results are excluded from the comparable HELDOUT table because only DEV/CV results were recorded (`golden_set_final.csv` 80/20 Val n=40; 5-Fold CV: `71.50% ± 4.64%`). They are not comparable heldout benchmark numbers.

---

## 2. Evaluation Design & Data Splits

To ensure evaluation rigor and prevent benchmark contamination, the data architecture enforces strict separation:

```
TWCS Dataset (@AmazonHelp Subset: 374,304 tweets, 82,636 threads)
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
Historical RAG Index             Development Set                    Frozen Heldout Benchmark
(25,544 resolved cases)        (golden_set_final.csv, n=200)       (heldout_test_final.csv, n=200)
Used for FAISS vector          Used for Taxonomy & Policy           Strictly Frozen Final Benchmark
retrieval evidence             Development (Seed 42)                0 Dev Overlap (Seed 2026)
```

- **Development Set (`golden_set_final.csv`, n=200)**: 100% human-audited (93 confirmed pre-labels, 107 corrected). Used for prompt design, taxonomy definition, and escalation policy rules.
- **Frozen Heldout Set (`heldout_test_final.csv`, n=200)**: Independently human-audited. Kept completely untouched during development. Zero thread overlap with development set (340 root conversations excluded). Heldout labels were strictly hidden during inference and tuning.
- **Baseline Alignment**: The exact same 200-example heldout set was used to evaluate Rule-Based V1, Rule-Based V2, TF-IDF + Logistic Regression, and the Hybrid RAG system.

> [!IMPORTANT]
> **Evaluation Comparability Note**: Direct numerical comparison with external benchmarks or alternative submissions must be interpreted with caution. Reported metrics depend heavily on specific intent taxonomy definitions (e.g. 5-class vs 11-class), sampling distributions, thread filtering heuristics, and whether evaluation was performed on a true frozen heldout set or an un-audited development split. Across all baselines in this repository, evaluation was conducted on the exact same 200-example frozen heldout test set (`heldout_test_final.csv`).

---

## 3. System Architecture & Technical Flow

```
Incoming Customer Conversation Thread
        │
        ▼
Conversation Reconstruction & Context Preprocessing
        │
        ▼
Historical Retrieval (FAISS + all-MiniLM-L6-v2) ──► Top-3 Resolved Historical Cases
(Current Conversation ID Excluded)
        │
        ▼
Local LLM Inference (Ollama Qwen2.5 3B)
        ├─► Intent Classification
        ├─► Public Reply Draft
        └─► Raw Escalation Suggestion
        │
        ▼
Deterministic Escalation Policy Engine (Final Authority)
        ├─► Order / Tracking Lookup Rule
        ├─► Account Security / Credential Rule
        ├─► Refund / Monetary Action Rule
        └─► Third-Party Dispute Rule
        │
        ▼
Final Output: Intent + Final Escalation (True/False) + Public Reply + Reason
```

### Technical Design Clarifications:
1. **Role of Historical Evidence**: Retrieved cases provide resolution patterns, formatting, and style grounding. They are **NOT** used as hidden intent ground-truth labels.
2. **Escalation Authority**: The **Deterministic Escalation Policy Engine is the final authority** for escalation decisions. Raw LLM escalation output is retained for comparison logging but does NOT override the policy engine.
3. **Retrieval Leakage Prevention**: During evaluation, the current conversation ID is explicitly excluded from vector search to prevent retrieving an example's own historical turns.

---

## 4. Engineering Rigor & Implementation Highlights

- **Data Leakage Prevention**: Enforces explicit conversation-ID exclusion in `src/retrieval.py` during FAISS vector search to prevent an example from retrieving its own historical turns.
- **Deterministic Escalation Authority**: Decouples LLM intent prediction from escalation policy enforcement. A deterministic rule engine evaluates domain rules (order lookups, account security, refunds, seller disputes) as the final authority, preventing LLM safety bypasses.
- **Reproducible Artifact Logging**: Every run writes structured JSON metrics, per-example CSV predictions, confusion matrices, and detailed execution logs to `evaluation/results/`.
- **Comprehensive Decision Log**: Documents 13 non-obvious engineering decisions ([`evaluation/results/decision_log.md`](./evaluation/results/decision_log.md)), detailing trade-offs, architecture choices, and evaluation split protocols.
- **Interactive Human Annotation Web UI**: Includes a standalone, zero-dependency local web application ([`evaluation/reply_quality/annotate_replies.py`](./evaluation/reply_quality/annotate_replies.py)) for human annotation, complete with progress tracking, guideline cards, and real-time state persistence.
- **Test Suite**: Includes unit and smoke testing scripts for LLM connectivity, retrieval accuracy, latency benchmarks, and prompt schema validation (`src/test_ollama.py`, `src/test_retrieval.py`, `src/test_prompt_smoke.py`).

---

## 5. Intent Taxonomy (11 Classes)

1. `delivery_shipping_delay`: Order status, late shipments, tracking inquiries.
2. `returns_refund_inquiry`: Return procedures, refund timelines, return labels.
3. `item_condition_issue`: Damaged items, missing parts, wrong item received.
4. `device_hardware_support`: Kindle, Fire TV, Echo device troubleshooting.
5. `prime_membership_billing`: Prime subscriptions, membership charges.
6. `digital_streaming_services`: Prime Video, Amazon Music, digital content.
7. `account_access_security`: Password resets, locked accounts, suspicious activity.
8. `payment_promo_giftcard`: Failed payments, promotional codes, gift card issues.
9. `order_modification_cancellation`: Order cancellation requests, address updates.
10. `marketplace_third_party_seller`: Third-party seller disputes, fulfillment issues.
11. `unclear_other`: Ambiguous, incomplete, or generic customer statements.

---

## 6. Failure Analysis & Error Patterns

### Dominant Failure Pattern: Over-Prediction of Delivery Issues
The primary intent classification weakness is the systematic over-prediction of `delivery_shipping_delay` (59/200 heldout examples, 100% recall, 56.19% precision).

### Major Heldout Confusions & Real Examples:

1. **Returns / Refunds Mistaken for Delivery** (11 cases)
   - *Customer*: `"@AmazonHelp I expect a refund for my delivery charge."`
   - *Gold*: `returns_refund_inquiry` | *Predicted*: `delivery_shipping_delay`
   - *Hypothesis*: The classifier appears to overweight salient delivery keywords ("delivery charge") even when the primary actionable request is a refund.

2. **Unclear / General Inquiries Mistaken for Delivery** (9 cases)
   - *Customer*: `"@AmazonHelp May I ask why they're listed as prime if you can't deliver any the next day? I'd understand 1 item? But all 4 seems a poor service?"`
   - *Gold*: `unclear_other` | *Predicted*: `delivery_shipping_delay`
   - *Hypothesis*: Generic feedback containing delivery timing terms is misclassified as an active delivery delay inquiry.

3. **Item Condition Issues Mistaken for Delivery** (8 cases)
   - *Customer*: `"@AmazonHelp It finally got here, 20 min late &amp; it’s missing food 😡"`
   - *Gold*: `item_condition_issue` | *Predicted*: `delivery_shipping_delay`
   - *Hypothesis*: Arrival timing mentions ("20 min late") overshadow the missing item condition complaint.

4. **Order Modification / Cancellation Mistaken for Delivery** (4 cases)
   - *Customer*: `"@AmazonHelp Asked for you to contact courier? Put notes and directions on the system? Asked for different postcode to be applied but then hung up on me"`
   - *Gold*: `order_modification_cancellation` | *Predicted*: `delivery_shipping_delay`
   - *Hypothesis*: Logistics terms ("courier", "postcode") steer prediction toward delivery issues rather than address modification.

5. **Marketplace Seller Disputes Mistaken for Delivery** (4 cases)
   - *Customer*: `"@AmazonHelp I’m not talking about delivery charges but the MRP rates. Even though seller indicates this doesn’t it mean you validate this stuff?"`
   - *Gold*: `marketplace_third_party_seller` | *Predicted*: `delivery_shipping_delay`
   - *Hypothesis*: Mentioning "delivery charges" while disputing seller pricing causes the model to catch delivery keywords instead of seller validation intent.

*Note: These hypotheses represent observed-error patterns based on prediction logs rather than proven causal mechanisms.*

---

## 7. Reply Quality Evaluation & Human Agreement

An **LLM-as-Judge evaluation** paired with a **Human Agreement Study** was conducted on a representative 50-example heldout subset across 6 dimensions (1–5 scale).

### Human vs. Judge Metrics Summary (n=50)

| Dimension | LLM Judge Mean (1-5) | Human Mean (1-5) | Mean Abs Diff | Weighted Cohen's Kappa | Exact Agreement % | Within-1 Agreement % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Relevance** | 3.180 | 3.90 | 1.40 | 0.0598 | 22.0% | 58.0% |
| **Groundedness** | 3.200 | 4.06 | 1.14 | 0.0535 | 14.0% | 72.0% |
| **Helpfulness** | 2.620 | 2.90 | 0.64 | 0.2430 | 42.0% | 94.0% |
| **Tone** | 4.580 | 3.20 | 1.38 | 0.0670 | 10.0% | 52.0% |
| **Safety / Escalation** | **4.980** | **4.94** | **0.04** | **0.4845** | **96.0%** | **100.0%** |
| **Overall Quality** | **3.580** | **3.38** | **0.48** | **0.2214** | **56.0%** | **96.0%** |

> [!IMPORTANT]
> **Secondary Evaluator Framing**: The LLM judge is treated strictly as a **secondary evaluation signal**, NOT an authoritative replacement for human judgment. Human-judge agreement was strongest for objective **Safety Routing** (Kappa: `0.4845`, Exact: `96.0%`, Within-1: `100.0%`), but weaker on subjective dimensions (Relevance, Tone).

---

## 8. What Is Misleading About My Headline Number?

The **65.50%** intent accuracy is useful but incomplete:
1. **Class Imbalance Distortion**: Over-predicting the majority class (`delivery_shipping_delay`, 100% recall) inflates overall accuracy while masking weak performance on minority classes (`unclear_other` 17.65% recall, `marketplace_third_party_seller` 25.00% recall). The Macro F1 of **0.5836** exposes this imbalance.
2. **Downstream Decision Gap**: Combined decision accuracy is **58.00%**, demonstrating that correct intent classification does not always translate into a correct downstream escalation decision.
3. **Evaluation Snapshot**: The heldout benchmark consists of 200 heldout examples; results represent an evaluation snapshot rather than a production guarantee.

---

## 9. What I Would Improve Next

To systematically improve performance without contaminating the frozen heldout benchmark:

1. **Hard-Negative Training & Few-Shot Examples**: Add explicit hard-negative few-shot examples differentiating delivery delays from refund requests, item condition complaints, and marketplace disputes.
2. **Ambiguity Handling**: Introduce structured clarifying prompts for generic queries before assigning a functional intent.
3. **Fresh Benchmark Evaluation**: Evaluate all future model improvements on a **NEW, untouched test set** rather than retuning against the current heldout set.
4. **Calibrated Multi-Judge Ensembling**: Improve LLM judge calibration using anchored rubric examples and multi-judge consensus scoring.
5. **Hybrid Vector & Lexical Retrieval**: Benchmark BM25 + dense hybrid search against pure dense retrieval to improve case grounding.

---

## 10. Reproducibility & Quickstart

> [!NOTE]
> The recorded frozen heldout evaluation completed in approximately **11.3 minutes** (679.7s) after setup prerequisites and the retrieval index were available.

### Setup Prerequisites
1. **Python 3.10+**
2. **Ollama local LLM runtime**: Pull `qwen2.5:3b` (`ollama pull qwen2.5:3b`)
3. **Dataset Note**: The raw TWCS CSV (~516 MB) is intentionally excluded from git tracking. The processed `@AmazonHelp` index files (`models/amazonhelp_faiss.index`, `models/amazonhelp_metadata.json`) are included in the repository for immediate evaluation.

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Build Vector Index (Optional — pre-built index included)
```bash
python3 src/build_index.py
```

### Step 3: Run Frozen Heldout Evaluation
```bash
python3 evaluation/evaluate_heldout_hybrid.py
```
*Outputs written to `evaluation/results/hybrid_rag_heldout_evaluation.json` and `evaluation/results/hybrid_rag_heldout_predictions.csv`.*

### Step 4: Launch Human Annotation Web Application
```bash
python3 evaluation/reply_quality/annotate_replies.py --port 8500
```
*Access interface at `http://localhost:8500`.*

---

## 11. Repository Structure

```
hiver-sde-intern-assignment/
├── README.md                              # Project overview & guide
├── FINAL_REPORT.md                        # Full detailed technical report
├── requirements.txt                       # Python dependencies
├── src/                                   # System source code
│   ├── config.py                          # Global paths & settings
│   ├── llm_agent.py                       # Ollama LLM provider & agent logic
│   ├── retrieval.py                       # FAISS historical retriever
│   ├── escalation_policy.py               # Deterministic escalation rules
│   └── build_index.py                     # Vector index builder
├── evaluation/                            # Benchmark datasets & scripts
│   ├── golden_set_final.csv               # 200 human-audited dev examples
│   ├── heldout_test_final.csv             # 200 frozen heldout test examples
│   ├── evaluate_heldout_hybrid.py         # Heldout benchmark runner
│   ├── reply_quality/                     # Reply quality & human study files
│   │   ├── annotate_replies.py            # Local annotation web application
│   │   ├── human_annotation_template.csv  # 50-example human template
│   │   └── annotation_guidelines.md       # Annotator guidelines
│   └── results/                           # Evaluation outputs & logs
│       ├── hybrid_rag_heldout_evaluation.json
│       ├── hybrid_rag_heldout_predictions.csv
│       ├── reply_quality_human_agreement.json
│       ├── baseline_comparison.md
│       └── decision_log.md
└── models/                                # Index metadata & vector files
```
