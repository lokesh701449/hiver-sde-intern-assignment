# Final Heldout Test Set Evaluation Results (Frozen Hybrid RAG System)

> [!IMPORTANT]
> **Evaluation Scope**: Held-out test set (`heldout_test_final.csv`, n=200).
> **System Status**: 100% FROZEN (no code, prompt, model, or rule modifications).
> **Strict Isolation**: Ground truth labels were **NOT accessed during inference**.

## 1. Overall Performance Metrics

| Metric | Heldout Result | Dev Baseline | Delta |
| :--- | :---: | :---: | :---: |
| **Intent Accuracy** | **65.50%** | 68.50% | -3.00% |
| **Intent Macro F1** | **0.5836** | 0.6514 | -0.0678 |
| **Escalation Accuracy** | **84.00%** | 75.50% | +8.50% |
| **Escalation Precision** | **0.8636** | 0.7471 | +0.1165 |
| **Escalation Recall** | **0.9500** | 0.9630 | -0.0130 |
| **Escalation F1** | **0.9048** | 0.8414 | +0.0634 |
| **Combined Decision Accuracy** | **58.00%** | 59.50% | -1.50% |

## 2. Difficulty Breakdown

| Difficulty | Count | Intent Acc | Escalation Acc | Combined Acc |
| :--- | :---: | :---: | :---: | :---: |
| `Easy` | 138 | 73.9% | 84.8% | 65.9% |
| `Hard` | 10 | 40.0% | 90.0% | 30.0% |
| `Medium` | 52 | 48.1% | 80.8% | 42.3% |

## 3. Escalation Confusion Matrix (2x2)

| | Predicted Escalate | Predicted No Escalate |
| :--- | :---: | :---: |
| **Actual Escalate** | 152 (TP) | 8 (FN) |
| **Actual No Escalate** | 24 (FP) | 16 (TN) |

## 4. Per-Intent Performance

| Intent | Support | Precision | Recall | F1 |
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

## 5. Top 10 Intent Confusions

| Rank | Actual Intent | Predicted Intent | Count |
| :---: | :--- | :--- | :---: |
| 1 | `returns_refund_inquiry` | `delivery_shipping_delay` | 11 |
| 2 | `unclear_other` | `delivery_shipping_delay` | 9 |
| 3 | `item_condition_issue` | `delivery_shipping_delay` | 8 |
| 4 | `order_modification_cancellation` | `delivery_shipping_delay` | 4 |
| 5 | `marketplace_third_party_seller` | `delivery_shipping_delay` | 4 |
| 6 | `device_hardware_support` | `delivery_shipping_delay` | 4 |
| 7 | `digital_streaming_services` | `device_hardware_support` | 3 |
| 8 | `payment_promo_giftcard` | `delivery_shipping_delay` | 2 |
| 9 | `payment_promo_giftcard` | `prime_membership_billing` | 2 |
| 10 | `prime_membership_billing` | `delivery_shipping_delay` | 2 |
