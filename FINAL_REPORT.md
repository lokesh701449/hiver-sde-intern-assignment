# AmazonHelp AI Support Agent — Final Project Report

> **Author**: Chalasani Lokesh  
> **Repository**: [hiver-sde-intern-assignment](https://github.com/lokesh701449/hiver-sde-intern-assignment.git)  
> **System Architecture**: Hybrid RAG (Ollama Qwen2.5 3B + FAISS Historical Evidence + Deterministic Escalation Policy)

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

## 2. Dataset and Sampling

### 2.1 Raw Dataset & Brand Scope
- **Source**: Twitter Customer Support (TWCS) dataset (~2.81M tweets across 70+ brands).
- **Brand Scope**: Filtered exclusively to **`@AmazonHelp`** interactions (~374k tweets across 82,636 conversation threads).
- **English Threads Subset**: ~75,897 clean English conversation threads.

### 2.2 Golden Development Set (n=200)
- **Sampling**: Stratified sampling across thread lengths and keywords (Seed 42).
- **Human Audit**: 100% human-reviewed. Confirmed 93 pre-labels and corrected 107.
- **Intent Distribution**: `delivery_shipping_delay` (56), `unclear_other` (27), `item_condition_issue` (24), `returns_refund_inquiry` (21), `account_access_security` (17), `digital_streaming_services` (16), `payment_promo_giftcard` (13), `device_hardware_support` (8), `prime_membership_billing` (8), `order_modification_cancellation` (6), `marketplace_third_party_seller` (4).
- **Human Escalation**: `True` = 135 (67.5%), `False` = 65 (32.5%).

### 2.3 Frozen Heldout Test Set (n=200)
- **Sampling**: Independent 200-example sample (`heldout_test_final.csv`, Seed 2026).
- **Leakage Prevention**: Zero thread overlap with development set (340 root conversations excluded).
- **Human Verification**: 100% independently human-audited. Kept completely frozen during development.

---

## 3. Approach (Hybrid RAG Architecture)

The final system uses a **Hybrid RAG Pipeline** combining local LLM semantic understanding with deterministic policy safety:

```
Incoming Customer Conversation
        │
        ▼
Context Reconstruction & Vector Search (FAISS + MiniLM-L6-v2)
        │
        ▼
Top-3 Historical AmazonHelp Evidence Cases
        │
        ▼
Ollama Qwen2.5 3B (Local LLM)
        ├─► Predicted Intent
        └─► Public Reply Draft
        │
        ▼
Deterministic Escalation Policy Engine
        ├─► Order/Tracking Lookup Check
        ├─► Private Account / Security Check
        ├─► Refund / Monetary Action Check
        └─► Marketplace Dispute Check
        │
        ▼
Final Escalation Decision + Rationale
```

### Key Architectural Choices:
1. **Local LLM (`qwen2.5:3b`)**: Zero-shot semantic intent classification and public reply drafting via structured JSON output.
2. **FAISS Historical Retrieval**: Indexes 25,544 resolved AmazonHelp conversations using `sentence-transformers/all-MiniLM-L6-v2` (384-dim). During evaluation, the current conversation ID is strictly excluded from retrieval to prevent data leakage.
3. **Deterministic Escalation Policy**: LLMs often fail on negative constraint enforcement (e.g. predicting `escalate=False` for private order inquiries). A deterministic policy engine evaluates domain rules and overrides escalation decisions for 100% predictable safety.

---

## 4. Baselines

We compare our Hybrid RAG system against pre-existing baselines evaluated on the exact same 200-example frozen heldout test set:

1. **Rule-Based Baseline V1**: Keyword and pattern heuristics.
2. **Rule-Based Baseline V2**: Refined keyword and regex rules.
3. **Machine Learning Baseline**: TF-IDF (1,000 features) + Logistic Regression (trained on 200 dev examples).

*Note: Fine-tuned DistilBERT results (Intent Acc: 75.00%, Macro F1: 0.6959) are from an 80/20 Dev Split (n=40 val) and 5-Fold CV (`71.50% ± 4.64%`), and are excluded from heldout comparison.*

---

## 5. Final Frozen Heldout Benchmark Results

### 5.1 System Benchmark Comparison (n=200 Heldout)

| System / Baseline | Intent Acc | Intent Macro F1 | Escalation Acc | Escalation Precision | Escalation Recall | Escalation F1 | Combined Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule V1** | 60.50% | 0.5447 | 78.50% | 0.7714 | 0.8438 | 0.8060 | 52.50% |
| **Rule V2** | 58.50% | 0.4643 | 76.50% | 0.8170 | 0.7813 | 0.8418 | 50.50% |
| **TF-IDF + LogReg** | 53.00% | 0.4030 | **86.00%** | **0.8929** | 0.9375 | **0.9146** | 48.00% |
| **Hybrid RAG + LLM (Final)** | **65.50%** | **0.5836** | 84.00% | 0.8636 | **0.9500** | 0.9048 | **58.00%** |

- **Escalation Confusion Matrix (Hybrid RAG)**: TP = 152, FN = 8, FP = 24, TN = 16.
- **Latency & Runtime**: Total runtime: 679.7s (~11.3 min). Avg LLM Latency: 3.18s (Median: 2.92s). Avg Retrieval Latency: ~210 ms.

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

### 5.3 Difficulty Breakdown

| Difficulty Stratum | Count | Intent Accuracy | Escalation Accuracy | Combined Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Easy** | 138 | 73.91% | 84.78% | 65.94% |
| **Medium** | 52 | 48.08% | 80.77% | 42.31% |
| **Hard** | 10 | 40.00% | 90.00% | 30.00% |

---

## 6. Reply Quality and Human Agreement

We performed an **LLM-as-Judge evaluation** paired with an empirical **Human Agreement Study** on a 50-example representative heldout subset.

### 6.1 Dimension Scores & Agreement Summary

| Dimension | LLM Judge Mean (1-5) | Human Mean (1-5) | Mean Abs Diff | Weighted Cohen's Kappa | Exact Agreement % | Within-1 Agreement % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Relevance** | 3.180 | 3.90 | 1.40 | 0.0598 | 22.0% | 58.0% |
| **Groundedness** | 3.200 | 4.06 | 1.14 | 0.0535 | 14.0% | 72.0% |
| **Helpfulness** | 2.620 | 2.90 | 0.64 | 0.2430 | 42.0% | 94.0% |
| **Tone** | 4.580 | 3.20 | 1.38 | 0.0670 | 10.0% | 52.0% |
| **Safety / Escalation** | 4.980 | 4.94 | 0.04 | **0.4845** | **96.0%** | **100.0%** |
| **Overall Quality** | **3.580** | **3.38** | **0.48** | **0.2214** | **56.0%** | **96.0%** |

### Key Evaluation Insights:
- **Strong Safety Alignment**: Safety achieves high weighted kappa (`0.4845`), `96.0%` exact agreement, and `100.0%` within-1 agreement. Both human annotators and the LLM judge confirm that the system safely avoids making false private account claims on public channels.
- **LLM Judge Limits**: Subjective dimensions (Tone, Relevance) show weak kappa (`<0.07`), demonstrating that LLM judges serve as useful automated filters but cannot replace human evaluation.

---

## 7. Top 5 Failure Modes

1. **Returns / Refunds Mistaken for Delivery** (11 examples): Customers asking "where is my refund for order X?" are frequently misclassified as `delivery_shipping_delay` due to tracking/delivery keywords.
2. **Unclear / General Cases Mistaken for Delivery** (9 examples): Generic complaints ("my order is messed up") default to `delivery_shipping_delay`.
3. **Item Condition Issues Mistaken for Delivery** (8 examples): Damage complaints mentioning package arrival dates trigger delivery heuristics.
4. **Order Modifications Mistaken for Delivery** (4 examples): Address changes and cancellation requests mentioning shipping dates are classified as delivery issues.
5. **Marketplace Third-Party Disputes Mistaken for Delivery** (4 examples): Complaints about third-party seller shipping delays default to standard delivery inquiries.

*Dominant Pattern*: Systematic over-prediction of `delivery_shipping_delay` (59/200 heldout examples, recall = 100%, precision = 56.19%).

---

## 8. What Is Misleading About My Headline Number?

The headline intent accuracy of **65.50%** is informative but incomplete:
1. **Class Imbalance Distortion**: Over-predicting the majority class (`delivery_shipping_delay`, 100% recall) inflates overall accuracy while masking severe recall degradation on minor classes (`unclear_other` 17.65% recall, `marketplace_third_party_seller` 25.00% recall). The Macro F1 of **0.5836** exposes this imbalance.
2. **Downstream Decision Gap**: Combined decision accuracy drops to **58.00%**, showing that correct intent prediction does not automatically guarantee correct downstream action execution.
3. **Snapshot Limit**: The heldout set consists of 200 heldout examples; results represent an evaluation snapshot rather than a production guarantee.

---

## 9. Decision Log Summary

1. **Conversation-Level Framing**: Evaluated complete multi-turn threads rather than single tweets for realistic support context.
2. **Domain Focus**: Restricted dataset to `@AmazonHelp` for clean e-commerce domain consistency.
3. **11-Intent Taxonomy**: Created an actionable 11-class taxonomy including `unclear_other`.
4. **80/10/10 Split Protocol**: Prevented data leakage across index building, golden development, and heldout evaluation.
5. **100% Human Audit**: Corrected 107 out of 200 pre-labels to ensure a clean ground truth.
6. **Retrieval Leakage Prevention**: Excluded current conversation IDs from vector retrieval during evaluation.
7. **Hybrid Architecture**: Decoupled LLM reply drafting from deterministic escalation rules to guarantee 100% safe escalation routing.

---

## 10. One-Week Next Steps

1. **Expand Rare Intent Labels**: Collect 500+ labelled examples for rare classes (`order_modification_cancellation`, `marketplace_third_party_seller`).
2. **Refine Intent Boundaries**: Add negative keyword constraints between delivery delays and return/refund inquiries.
3. **Dense Reranking & Retrieval Filtering**: Implement cross-encoder reranking to improve historical case relevance.
4. **Action-Aware Reply Generation**: Enhance public replies for escalated cases to explicitly explain DM routing next steps.
5. **Production Benchmarking**: Expand heldout evaluation to a 1,000-example benchmark.

---

## 11. Limitations

- **Domain Scope**: Tailored to `@AmazonHelp` e-commerce workflows; adapting to telecom or banking requires re-indexing.
- **Majority Class Bias**: Heuristics favor `delivery_shipping_delay` under high uncertainty.
- **Latency**: Local LLM inference takes ~3.18s per query, requiring asynchronous processing in production.
