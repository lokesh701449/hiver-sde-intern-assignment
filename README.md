# AmazonHelp AI Customer Support Agent & Hybrid RAG System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Local LLM](https://img.shields.io/badge/Ollama-Qwen2.5--3B-orange.svg)](https://ollama.ai/)
[![Vector Index](https://img.shields.io/badge/FAISS-MiniLM--L6--v2-green.svg)](https://github.com/facebookresearch/faiss)
[![Evaluation Benchmark](https://img.shields.io/badge/Benchmark-Frozen%20Heldout%20(n%3D200)-purple.svg)](./evaluation/heldout_test_final.csv)

An end-to-end **AI Customer Support Agent prototype** for e-commerce (`@AmazonHelp`), featuring multi-task **Intent Classification**, **Deterministic Escalation Routing**, **Historical RAG Retrieval**, and **LLM-as-Judge Reply Quality Evaluation**.

---

## 1. Executive Summary & Headline Results

| System | Intent Acc | Intent Macro F1 | Esc Acc | Esc F1 | Combined |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Rule V1** | 60.50% | 0.5447 | 78.50% | 0.8060 | 52.50% |
| **Rule V2** | 58.50% | 0.4643 | 76.50% | 0.8418 | 50.50% |
| **TF-IDF + Logistic Regression** | 53.00% | 0.4030 | **86.00%** | **0.9146** | 48.00% |
| **Hybrid RAG + LLM** | **65.50%** | **0.5836** | 84.00% | 0.9048 | **58.00%** |

> [!NOTE]
> **Transformer Model Note**:
> Transformer results (Fine-tuned DistilBERT) are excluded from this comparable HELDOUT table because only DEV/CV results were recorded (`golden_set_final.csv` 80/20 Val n=40; Intent Acc: 75.00%, Macro F1: 0.6959, Esc Acc: 65.00%; 5-Fold CV: 71.50% ± 4.64%). They are not comparable heldout benchmark numbers.

---

## 2. Problem Framing & Objectives

1. **Multi-Turn Conversation Context**: Operates on full multi-turn conversation threads rather than isolated single tweets.
2. **Intent Taxonomy (11 Classes)**: Categorizes customer inquiries into 11 functional e-commerce classes including `unclear_other`.
3. **Safe Escalation Routing**: Separates public inquiries (resolved publicly) from private/sensitive inquiries (routed to human support/DM). Deterministic policy rules provide predictable escalation behavior and improve safety for sensitive customer actions, while the heldout evaluation shows 95.0% escalation recall.
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

*Note*: The human evaluation and LLM judge show strong agreement on safety. The LLM judge is treated as a secondary evaluator rather than a replacement for human judgment.

---

## 7. Failure Modes (with Real Heldout Examples)

### Top 5 Failure Modes:

1. **Returns / Refunds Mistaken for Delivery** (11 cases)
   - *Customer*: `"@AmazonHelp I expect a refund for my delivery charge."`
   - *Gold*: `returns_refund_inquiry` | *Predicted*: `delivery_shipping_delay`
   - *Hypothesis*: The presence of "delivery charge" triggered delivery keyword heuristics despite the request being a refund inquiry.

2. **Unclear / General Inquiries Mistaken for Delivery** (9 cases)
   - *Customer*: `"@AmazonHelp May I ask why they're listed as prime if you can't deliver any the next day? I'd understand 1 item? But all 4 seems a poor service?"`
   - *Gold*: `unclear_other` | *Predicted*: `delivery_shipping_delay`
   - *Hypothesis*: Generic feedback containing delivery timing language causes misclassification as an active delivery delay.

3. **Item Condition Issues Mistaken for Delivery** (8 cases)
   - *Customer*: `"@AmazonHelp It finally got here, 20 min late &amp; it’s missing food 😡"`
   - *Gold*: `item_condition_issue` | *Predicted*: `delivery_shipping_delay`
   - *Hypothesis*: Late arrival mentions overshadow the missing item condition complaint.

4. **Order Modification / Cancellation Mistaken for Delivery** (4 cases)
   - *Customer*: `"@AmazonHelp Asked for you to contact courier? Put notes and directions on the system? Asked for different postcode to be applied but then hung up on me"`
   - *Gold*: `order_modification_cancellation` | *Predicted*: `delivery_shipping_delay`
   - *Hypothesis*: Logistics terms like "courier" and "postcode" steer prediction toward delivery issues rather than address modification.

5. **Marketplace Seller Disputes Mistaken for Delivery** (4 cases)
   - *Customer*: `"@AmazonHelp I’m not talking about delivery charges but the MRP rates. Even though seller indicates this doesn’t it mean you validate this stuff?"`
   - *Gold*: `marketplace_third_party_seller` | *Predicted*: `delivery_shipping_delay`
   - *Hypothesis*: Mentioning "delivery charges" while disputing third-party seller pricing causes the model to catch delivery keywords instead of seller validation intent.

*What is Misleading About the Headline Number?*  
The 65.50% intent accuracy is useful but incomplete. It averages over an uneven intent distribution and hides weak performance on lower-frequency intents; Macro F1 of 0.5836 exposes this. Combined decision accuracy is only 58.0%, showing that correct intent classification does not always translate into the correct downstream escalation decision. The heldout set contains 200 examples, so the result should be interpreted as an evaluation snapshot rather than a production performance estimate.

---

## 8. Reproducibility & Commands

> [!NOTE]
> The recorded frozen heldout evaluation completed in approximately **11.3 minutes** after prerequisites and the retrieval index were available. Setup prerequisites include Python dependencies, local Ollama runtime (`qwen2.5:3b`), and the vector index.

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
4. **Action-Aware Reply Generation**: Enhance public replies for escalated cases to explain routing.
5. **Production Benchmarking**: Expand heldout evaluation to a 1,000-example benchmark.
