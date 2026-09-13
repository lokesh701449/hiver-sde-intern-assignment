# Hybrid RAG Evaluation Results (Development Set)

> [!IMPORTANT]
> **Architecture**: Hybrid RAG (Ollama `qwen2.5:3b` for Intent + Reply; Deterministic Policy for Escalation).
> **Dev Set**: `golden_set_final.csv` (n=200). `heldout_test_final.csv` was **NOT accessed**.

## 1. Overall Metrics Summary

| Metric | Value |
| :--- | :---: |
| Intent Accuracy | **68.50%** (137/200) |
| Intent Macro F1 | **0.6514** |
| Final Escalation Accuracy | **75.50%** (151/200) |
| Final Escalation Precision | **0.7471** |
| Final Escalation Recall | **0.9630** |
| Final Escalation F1 | **0.8414** |
| Combined Decision Accuracy | **59.50%** (119/200) |
| Total Runtime | 670.5s (11.2 min) |

## 2. Escalation Comparison & Disagreements

| Category | Count |
| :--- | :---: |
| Deterministic Policy Escalate=True | 174 |
| Deterministic Policy Escalate=False | 26 |
| Raw LLM Escalate=True | 25 |
| Raw LLM Escalate=False | 175 |
| LLM vs Policy Disagreements | 151 |
| Intent Correct & Escalation Wrong | 18 |
| Intent Wrong & Escalation Correct | 32 |

## 3. Escalation Confusion Matrix (2x2)

| | Predicted Escalate | Predicted No Escalate |
| :--- | :---: | :---: |
| **Actual Escalate** | 130 (TP) | 5 (FN) |
| **Actual No Escalate** | 44 (FP) | 21 (TN) |

## 4. Per-Intent Performance

| Intent | Support | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: | :---: |
| `account_access_security` | 17 | 0.9375 | 0.8824 | **0.9091** |
| `returns_refund_inquiry` | 21 | 0.8571 | 0.8571 | **0.8571** |
| `device_hardware_support` | 8 | 0.7778 | 0.8750 | **0.8235** |
| `payment_promo_giftcard` | 13 | 0.8182 | 0.6923 | **0.7500** |
| `delivery_shipping_delay` | 56 | 0.5714 | 1.0000 | **0.7273** |
| `prime_membership_billing` | 8 | 0.8333 | 0.6250 | **0.7143** |
| `digital_streaming_services` | 16 | 0.7333 | 0.6875 | **0.7097** |
| `marketplace_third_party_seller` | 4 | 1.0000 | 0.5000 | **0.6667** |
| `item_condition_issue` | 24 | 0.7500 | 0.3750 | **0.5000** |
| `order_modification_cancellation` | 6 | 0.2857 | 0.3333 | **0.3077** |
| `unclear_other` | 27 | 1.0000 | 0.1111 | **0.2000** |

## 5. Top 10 Intent Confusions

| Rank | Actual Intent | Predicted Intent | Count |
| :---: | :--- | :--- | :---: |
| 1 | `unclear_other` | `delivery_shipping_delay` | 17 |
| 2 | `item_condition_issue` | `delivery_shipping_delay` | 12 |
| 3 | `digital_streaming_services` | `delivery_shipping_delay` | 4 |
| 4 | `unclear_other` | `order_modification_cancellation` | 3 |
| 5 | `unclear_other` | `digital_streaming_services` | 3 |
| 6 | `prime_membership_billing` | `delivery_shipping_delay` | 3 |
| 7 | `order_modification_cancellation` | `delivery_shipping_delay` | 2 |
| 8 | `payment_promo_giftcard` | `delivery_shipping_delay` | 2 |
| 9 | `returns_refund_inquiry` | `payment_promo_giftcard` | 1 |
| 10 | `returns_refund_inquiry` | `delivery_shipping_delay` | 1 |
