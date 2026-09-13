# AmazonHelp AI Customer Support Agent & Hybrid RAG System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Local LLM](https://img.shields.io/badge/Ollama-Qwen2.5--3B-orange.svg)](https://ollama.ai/)
[![Vector Index](https://img.shields.io/badge/FAISS-MiniLM--L6--v2-green.svg)](https://github.com/facebookresearch/faiss)
[![Evaluation Benchmark](https://img.shields.io/badge/Benchmark-Frozen%20Heldout%20(n%3D200)-purple.svg)](./evaluation/heldout_test_final.csv)

An end-to-end, production-grade AI Customer Support Agent for e-commerce (`@AmazonHelp`), featuring multi-task **Intent Classification**, **Deterministic Escalation Routing**, **Historical RAG Retrieval**, and **LLM-as-Judge Reply Quality Evaluation**.

---

## 1. Executive Summary & Headline Results

| System / Baseline | Architecture | Evaluation Split | Intent Acc | Intent Macro F1 | Escalation Acc | Escalation F1 | Combined Acc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule-Based V1** | Keyword / Heuristics | **HELDOUT** | 60.50% | 0.5447 | 78.50% | 0.8060 | 52.50% |
| **Rule-Based V2** | Refined Heuristics | **HELDOUT** | 58.50% | 0.4643 | 76.50% | 0.8418 | 50.50% |
| **TF-IDF + LogReg** | ML Classifier (1000 feat) | **HELDOUT** | 53.00% | 0.4030 | **86.00%** | **0.9146** | 48.00% |
| **Hybrid RAG (Final)** | Ollama Qwen2.5:3B + Policy | **HELDOUT** | **65.50%** | **0.5836** | 84.00% | 0.9048 | **58.00%** |
| *DistilBERT (Dev)* | Fine-Tuned Transformer | *DEV / CV Only* | *75.00%* | *0.6959* | *65.00%* | *0.7813* | *47.50%* |

> [!IMPORTANT]
> **Evaluation Protocol**: The final heldout benchmark (`heldout_test_final.csv`, n=200) was strictly frozen. No heldout labels were available during inference, and current conversation IDs were excluded during historical vector retrieval to prevent data leakage.

---

## 2. Problem Framing & Objectives

1. **Multi-Turn Conversation Context**: Operates on full multi-turn conversation threads rather than isolated single tweets.
2. **Intent Taxonomy (11 Classes)**: Categorizes customer inquiries into 11 functional e-commerce classes including `unclear_other`.
3. **Safe Escalation Routing**: Separates public inquiries (resolved publicly) from private/sensitive inquiries (routed to human support/DM).
4. **Public Reply Drafting**: Generates helpful public support responses grounded in historical `@AmazonHelp` support cases.

---

## 3. System Architecture

```
Incoming Customer Conversation
        │
        ▼
Context Reconstruction & Preprocessing
        │
        ▼
Vector Retrieval (FAISS + MiniLM-L6-v2) ──► Top-3 Historical Cases
        │
        ▼
Local LLM Inference (Ollama Qwen2.5 3B)
        ├─► Intent Classification
        └─► Public Reply Draft
        │
        ▼
Deterministic Escalation Policy Engine
        ├─► Order / Tracking Lookup Detection
        ├─► Account Security / Credential Check
        ├─► Refund / Monetary Inquiry Rule
        └─► Third-Party Dispute Verification
        │
        ▼
Final Decision: Intent + Escalation (True/False) + Public Reply + Reason
```

---

## 4. Intent Taxonomy (11 Classes)

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

## 5. Dataset, Subsampling & Golden Set

- **Raw Source**: Twitter Customer Support (TWCS) dataset (2.81M tweets).
- **Brand Filter**: `@AmazonHelp` subset (374,304 tweets, 82,636 threads).
- **Golden Development Set**: 200 conversation threads (`golden_set_final.csv`), 100% human-reviewed (93 confirmed, 107 corrected).
- **Frozen Heldout Test Set**: 200 conversation threads (`heldout_test_final.csv`, Seed 2026), 0 thread overlap with development set.

---

## 6. Reply Quality Evaluation & Human Agreement

Evaluated on 50 representative heldout examples across 6 dimensions (1-5 scale):

| Dimension | LLM Judge Mean (1-5) | Human Mean (1-5) | Mean Abs Diff | Weighted Cohen's Kappa | Within-1 Agreement % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Relevance** | 3.180 | 3.90 | 1.40 | 0.0598 | 58.0% |
| **Groundedness** | 3.200 | 4.06 | 1.14 | 0.0535 | 72.0% |
| **Helpfulness** | 2.620 | 2.90 | 0.64 | 0.2430 | 94.0% |
| **Tone** | 4.580 | 3.20 | 1.38 | 0.0670 | 52.0% |
| **Safety / Escalation** | **4.980** | **4.94** | **0.04** | **0.4845** | **100.0%** |
| **Overall Quality** | **3.580** | **3.38** | **0.48** | **0.2214** | **96.0%** |

---

## 7. Failure Modes & Limitations

### Top 5 Error Patterns:
1. **Returns/Refunds Mistaken for Delivery** (11 cases)
2. **Unclear/General Inquiries Mistaken for Delivery** (9 cases)
3. **Item Condition Issues Mistaken for Delivery** (8 cases)
4. **Order Modification Mistaken for Delivery** (4 cases)
5. **Marketplace Seller Disputes Mistaken for Delivery** (4 cases)

*What is Misleading About the Headline Number?*  
The **65.50%** intent accuracy is driven by high recall on `delivery_shipping_delay` (100% recall), while Macro F1 (**0.5836**) reveals lower performance on rare intents (`unclear_other` 17.65% recall). Downstream combined decision accuracy is **58.00%**.

---

## 8. Reproducibility & Commands

### Prerequisites
- Python 3.10+
- Ollama running locally with `qwen2.5:3b` model (`ollama pull qwen2.5:3b`)

### Build Vector Index
```bash
python3 src/build_index.py
```

### Run Heldout Evaluation
```bash
python3 evaluation/evaluate_heldout_hybrid.py
```

### Launch Human Annotation Web Tool
```bash
python3 evaluation/reply_quality/annotate_replies.py --port 8500
```

---

## 9. Repository Structure

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

---

## 10. Future Work (One-Week Horizon)

1. **Rare Intent Data Expansion**: Collect 500+ labelled examples for minority intent classes.
2. **Negative Keyword Constraints**: Improve discrimination between delivery delays and refund requests.
3. **Cross-Encoder Reranking**: Re-rank retrieved historical cases to improve grounding.
4. **Explicit Escalation Messaging**: Enhance public replies for escalated cases to explain routing.
