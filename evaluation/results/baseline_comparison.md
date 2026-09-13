# Baseline Comparison — Frozen Heldout Benchmark (n=200)

> [!IMPORTANT]
> **Evaluation Benchmark**: Strict Heldout Test Set (`heldout_test_final.csv`, n=200, Seed 2026).
> **System Status**: FROZEN. All metrics evaluated under zero-shot / leak-free evaluation protocol.

## 1. Comparable Heldout Benchmark Comparison

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

## 2. Metric Rationale & Key Takeaways

1. **Intent Classification Superiority**: The **Hybrid RAG system** achieves the highest Intent Accuracy (**65.50%**) and Intent Macro F1 (**0.5836**), outperforming rule-based V1 by +5.0% accuracy and ML TF-IDF by +12.5% accuracy.
2. **Escalation Precision & Recall**: Both TF-IDF + LogReg and Hybrid RAG achieve strong escalation handling (Escalation F1 > 0.90), successfully capturing sensitive requests (DM routing, order verification).
3. **Combined Decision Accuracy**: Hybrid RAG achieves the highest downstream Combined Accuracy (**58.00%**), demonstrating that LLM semantic intent prediction combined with deterministic policy safety provides the best overall end-to-end support automation.
