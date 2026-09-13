# AmazonHelp AI Support Agent — Final Project Report

> **Author**: Chalasani Lokesh  
> **Repository**: [hiver-sde-intern-assignment](https://github.com/lokesh701449/hiver-sde-intern-assignment.git)  
> **System Architecture**: Hybrid RAG Prototype (Ollama Qwen2.5 3B + FAISS Historical Evidence + Deterministic Escalation Policy)

---

## 1. Problem and Framing

Customer support teams operating on public channels (e.g., Twitter / X) face two critical challenges:
1. **Semantic Intent Recognition**: Categorizing customer inquiries into actionable e-commerce support domains (e.g. delivery delays, refunds, streaming issues, security).
2. **Safe Escalation Routing**: Determining whether an inquiry can be resolved publicly with self-service guidance or must be **escalated** to a private human support agent (e.g., when order lookups, refunds, or private account credentials are required).

We formulate this task as a **Multi-Task E-Commerce Support Automation Pipeline**:
- **Task A (Intent Classification)**: Predict one of 11 mutually exclusive support intents for an incoming customer conversation.
- **Task B (Escalation Decision)**: Predict a binary escalation signal (`True` = route to human/DM, `False` = respond publicly) along with an explicit escalation rationale.
- **Task C (Public Reply Generation)**: Generate a professional, grounded public reply for non-escalated cases or appropriate routing guidance for escalated cases.

---

## 2. Dataset, Sampling & Evaluation Design

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

### 2.1 Brand Scope & Subsampling
- **Source**: Twitter Customer Support (TWCS) dataset (~2.81M tweets across 70+ brands).
- **Brand Scope**: Filtered exclusively to **`@AmazonHelp`** interactions (~374k tweets across 82,636 conversation threads; ~75,897 clean English threads).

### 2.2 Golden Development Set (n=200)
- **Sampling**: Stratified sampling across thread lengths and keywords (`golden_set_final.csv`, Seed 42).
- **Human Audit**: 100% human-reviewed. Confirmed 93 pre-labels and corrected 107.
- **Intent Distribution**: `delivery_shipping_delay` (56), `unclear_other` (27), `item_condition_issue` (24), `returns_refund_inquiry` (21), `account_access_security` (17), `digital_streaming_services` (16), `payment_promo_giftcard` (13), `device_hardware_support` (8), `prime_membership_billing` (8), `order_modification_cancellation` (6), `marketplace_third_party_seller` (4).
- **Human Escalation**: `True` = 135 (67.5%), `False` = 65 (32.5%).

### 2.3 Frozen Heldout Test Set (n=200)
- **Sampling**: Independent 200-example sample (`heldout_test_final.csv`, Seed 2026).
- **Leakage Prevention**: Zero thread overlap with development set (340 root conversations excluded).
- **Human Verification**: 100% independently human-audited. Kept completely frozen during development; heldout labels were strictly hidden during inference and tuning.

---

## 3. Approach (Hybrid RAG Architecture)

The final system uses a **Hybrid RAG Pipeline** combining local LLM semantic understanding with deterministic policy safety:

```
Incoming Customer Conversation
        │
        ▼
Context Reconstruction & Vector Search (FAISS + MiniLM-L6-v2) ──► Top-3 Resolved Historical Cases
(Current Conversation ID Excluded)
        │
        ▼
Ollama Qwen2.5 3B (Local LLM)
        ├─► Predicted Intent
        ├─► Public Reply Draft
        └─► Raw Escalation Suggestion
        │
        ▼
Deterministic Escalation Policy Engine (Final Authority)
        ├─► Order/Tracking Lookup Check
        ├─► Private Account / Security Check
        ├─► Refund / Monetary Action Check
        └─► Marketplace Dispute Check
        │
        ▼
Final Escalation Decision + Rationale Output
```

### Architectural Principles:
1. **Local LLM (`qwen2.5:3b`)**: Zero-shot semantic intent classification and public reply drafting via structured JSON output.
2. **FAISS Historical Retrieval**: Indexes 25,544 resolved AmazonHelp conversations using `sentence-transformers/all-MiniLM-L6-v2` (384-dim). During evaluation, the current conversation ID is strictly excluded from retrieval to prevent data leakage. Historical cases provide resolution style and formatting grounding, but are **not** used as hidden intent labels.
3. **Deterministic Escalation Policy**: Deterministic policy rules provide predictable escalation behavior and improve safety for sensitive customer actions. The policy engine is the **final authority** for escalation decisions; raw LLM escalation suggestions are logged for comparison but do not override policy rules.

---

## 4. Baselines

We compare our Hybrid RAG system against pre-existing baselines evaluated on the exact same 200-example frozen heldout test set:

1. **Rule-Based Baseline V1**: Keyword and pattern heuristics.
2. **Rule-Based Baseline V2**: Refined keyword and regex rules.
3. **Machine Learning Baseline**: TF-IDF (1,000 features) + Logistic Regression (trained on 200 dev examples).

*Note: Fine-tuned DistilBERT results (Intent Acc: 75.00%, Macro F1: 0.6959) are from an 80/20 Dev Split (n=40 val) and 5-Fold CV (`71.50% ± 4.64%`), and are marked as DEV/CV evidence only.*

---

## 5. Final Frozen Heldout Benchmark Results

### 5.1 System Benchmark Comparison (n=200 Heldout)

| System | Intent Acc | Intent Macro F1 | Esc Acc | Esc Precision | Esc Recall | Esc F1 | Combined Acc |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Rule V1** | 60.50% | 0.5447 | 78.50% | 0.7714 | 0.8438 | 0.8060 | 52.50% |
| **Rule V2** | 58.50% | 0.4643 | 76.50% | 0.8170 | 0.7813 | 0.8418 | 50.50% |
| **TF-IDF + Logistic Regression** | 53.00% | 0.4030 | **86.00%** | **0.8929** | 0.9375 | **0.9146** | 48.00% |
| **Hybrid RAG + LLM** | **65.50%** | **0.5836** | 84.00% | 0.8636 | **0.9500** | 0.9048 | **58.00%** |

- **Escalation Confusion Matrix (Hybrid RAG)**: TP = 152, FN = 8, FP = 24, TN = 16 (Precision: 0.8636, Recall: 0.9500).
- **Recorded Runtime**: The recorded frozen heldout evaluation completed in approximately **11.3 minutes** (679.7 seconds) after prerequisites and the retrieval index were available. Avg LLM Latency: 3.18s (Median: 2.92s). Avg Retrieval Latency: ~210 ms.

### 5.2 Per-Intent Performance (Hybrid RAG Heldout)

| Intent | Support | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| `digital_streaming_services` | 20 | 0.9286 | 0.6500 | **0.7647** |
| `account_access_security` | 10 | 0.7778 | 0.7000 | **0.7368** |
| `delivery_shipping_delay` | 59 | 0.5619 | 1.0000 | **0.7195** |
| `payment_promo_giftcard` | 14 | 0.7143 | 0.7143 | **0.7143** |
| `item_condition_issue` | 21 | 1.0000 | 0.5238 | **0.6875** |
| `returns_refund_inquiry` | 26 | 0.8235 | 0.5385 | **0.6512** |
| `device_hardware_support` | 9 | 0.6250 | 0.5556 | **0.5882** |
| `prime_membership_billing` | 6 | 0.5000 | 0.6667 | **0.5714** |
| `marketplace_third_party_seller` | 12 | 1.0000 | 0.2500 | **0.4000** |
| `unclear_other` | 17 | 1.0000 | 0.1765 | **0.3000** |
| `order_modification_cancellation` | 6 | 0.2500 | 0.3333 | **0.2857** |

---

## 6. Reply Quality Evaluation & Human Agreement

Evaluated on a 50-example representative heldout subset across 6 dimensions (1-5 scale):

| Dimension | LLM Judge Mean (1-5) | Human Mean (1-5) | Mean Abs Diff | Weighted Cohen's Kappa | Exact Agreement % | Within-1 Agreement % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Relevance** | 3.180 | 3.90 | 1.40 | 0.0598 | 22.0% | 58.0% |
| **Groundedness** | 3.200 | 4.06 | 1.14 | 0.0535 | 14.0% | 72.0% |
| **Helpfulness** | 2.620 | 2.90 | 0.64 | 0.2430 | 42.0% | 94.0% |
| **Tone** | 4.580 | 3.20 | 1.38 | 0.0670 | 10.0% | 52.0% |
| **Safety / Escalation** | 4.980 | 4.94 | 0.04 | **0.4845** | **96.0%** | **100.0%** |
| **Overall Quality** | **3.580** | **3.38** | **0.48** | **0.2214** | **56.0%** | **96.0%** |

### Key Evaluation Insights:
- **Strong Safety Alignment**: The human evaluation and LLM judge show strong agreement on safety (weighted kappa = `0.4845`, `96.0%` exact agreement, `100.0%` within-1 agreement).
- **Secondary Evaluator Role**: The LLM judge is treated as a secondary evaluator rather than a replacement for human judgment due to modest agreement on subjective dimensions (Tone, Relevance).

---

## 7. Top 5 Failure Modes (with Real Heldout Examples)

### 1. Returns / Refunds Mistaken for Delivery (11 examples)
- **Example**:  
  Customer: `"@AmazonHelp I expect a refund for my delivery charge."`  
  Gold intent: `returns_refund_inquiry` | Predicted intent: `delivery_shipping_delay`  
- **Hypothesis**: The classifier appears to overweight salient delivery keywords ("delivery charge") even when the primary actionable request is a refund.

### 2. Unclear / General Inquiries Mistaken for Delivery (9 examples)
- **Example**:  
  Customer: `"@AmazonHelp May I ask why they're listed as prime if you can't deliver any the next day? I'd understand 1 item? But all 4 seems a poor service?"`  
  Gold intent: `unclear_other` | Predicted intent: `delivery_shipping_delay`  
- **Hypothesis**: Generic customer feedback containing delivery timing terms is misclassified as an active delivery delay inquiry.

### 3. Item Condition Issues Mistaken for Delivery (8 examples)
- **Example**:  
  Customer: `"@AmazonHelp It finally got here, 20 min late &amp; it’s missing food 😡"`  
  Gold intent: `item_condition_issue` | Predicted intent: `delivery_shipping_delay`  
- **Hypothesis**: Arrival timing mentions ("20 min late") overshadow the missing item condition complaint.

### 4. Order Modification / Cancellation Mistaken for Delivery (4 examples)
- **Example**:  
  Customer: `"@AmazonHelp Asked for you to contact courier? Put notes and directions on the system? Asked for different postcode to be applied but then hung up on me"`  
  Gold intent: `order_modification_cancellation` | Predicted intent: `delivery_shipping_delay`  
- **Hypothesis**: Logistics terms ("courier", "postcode") steer prediction toward delivery issues rather than address modification.

### 5. Marketplace Seller Disputes Mistaken for Delivery (4 examples)
- **Example**:  
  Customer: `"@AmazonHelp I’m not talking about delivery charges but the MRP rates. Even though seller indicates this doesn’t it mean you validate this stuff?"`  
  Gold intent: `marketplace_third_party_seller` | Predicted intent: `delivery_shipping_delay`  
- **Hypothesis**: Mentioning "delivery charges" while disputing seller pricing causes the model to catch delivery keywords instead of seller validation intent.

*Note: These hypotheses represent observed-error patterns based on prediction logs rather than proven causal mechanisms.*

---

## 8. What Is Misleading About My Headline Number?

The **65.50%** intent accuracy is useful but incomplete. It averages over an uneven intent distribution and hides weak performance on lower-frequency intents; Macro F1 of **0.5836** exposes this. Combined decision accuracy is only **58.00%**, showing that correct intent classification does not always translate into the correct downstream escalation decision. The heldout set contains 200 examples, so the result should be interpreted as an evaluation snapshot rather than a production performance estimate.

---

## 9. Decision Log Summary

1. **Conversation-Level Framing**: Evaluated complete multi-turn threads rather than single tweets for realistic support context.
2. **Domain Focus**: Restricted dataset to `@AmazonHelp` for clean e-commerce domain consistency.
3. **11-Intent Taxonomy**: Created an actionable 11-class taxonomy including `unclear_other`.
4. **80/10/10 Split Protocol**: Prevented data leakage across index building, golden development, and heldout evaluation.
5. **100% Human Audit**: Corrected 107 out of 200 pre-labels to ensure a clean ground truth.
6. **Retrieval Leakage Prevention**: Excluded current conversation IDs from vector retrieval during evaluation.
7. **Hybrid Architecture**: Decoupled LLM reply drafting from deterministic escalation rules to improve escalation safety.

---

## 10. What I Would Improve Next

1. **Hard-Negative Training & Few-Shot Examples**: Add explicit hard-negative few-shot examples differentiating delivery delays from refund requests, item condition complaints, and marketplace disputes.
2. **Ambiguity Handling**: Introduce structured clarifying prompts for generic queries before assigning a functional intent.
3. **Fresh Benchmark Evaluation**: Evaluate all future model improvements on a **NEW, untouched test set** rather than retuning against the current heldout set.
4. **Calibrated Multi-Judge Ensembling**: Improve LLM judge calibration using anchored rubric examples and multi-judge consensus scoring.
5. **Hybrid Vector & Lexical Retrieval**: Benchmark BM25 + dense hybrid search against pure dense retrieval to improve case grounding.

---

## 11. Limitations

- **Domain Scope**: Tailored to `@AmazonHelp` e-commerce workflows; adapting to telecom or banking requires re-indexing.
- **Majority Class Bias**: Heuristics favor `delivery_shipping_delay` under high uncertainty.
- **Latency**: Local LLM inference takes ~3.18s per query, requiring asynchronous processing in production.
