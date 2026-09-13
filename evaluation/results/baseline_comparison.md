# Baseline Comparison — Frozen Heldout Benchmark (n=200)

> [!IMPORTANT]
> **Evaluation Benchmark**: Strict Heldout Test Set (`heldout_test_final.csv`, n=200, Seed 2026).
> **System Status**: FROZEN. All metrics evaluated under zero-shot / leak-free evaluation protocol.

## 1. Comparable Heldout Benchmark Comparison

| System / Baseline | Architecture / Method | Intent Acc | Intent Macro F1 | Escalation Acc | Escalation F1 | Combined Acc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Rule-Based Baseline V1** | Keyword / Heuristic Rules | 60.50% | 0.5447 | 78.50% | 0.8060 | 52.50% |
| **Rule-Based Baseline V2** | Refined Pattern Matching | 58.50% | 0.4643 | 76.50% | 0.8418 | 50.50% |
| **ML Baseline** | TF-IDF + Logistic Regression | 53.00% | 0.4030 | **86.00%** | **0.9146** | 48.00% |
| **Hybrid RAG + LLM (Final)** | Ollama `qwen2.5:3b` + RAG + Deterministic Policy | **65.50%** | **0.5836** | 84.00% | 0.9048 | **58.00%** |

> [!NOTE]
> **Transformer Model Note**:
> Transformer results (Fine-tuned DistilBERT) are excluded from the comparable HELDOUT table above because DistilBERT was evaluated on an 80/20 Stratified Development Split (`golden_set_final.csv`, Val n=40; Intent Acc: 75.00%, Macro F1: 0.6959, Esc Acc: 65.00%) and 5-Fold Cross-Validation (`71.50% ± 4.64%`). It was not evaluated on the frozen 200-example heldout test benchmark.

---

## 2. Metric Rationale & Key Takeaways

1. **Intent Classification Superiority**: The **Hybrid RAG system** achieves the highest Intent Accuracy (**65.50%**) and Intent Macro F1 (**0.5836**), outperforming rule-based V1 by +5.0% accuracy and ML TF-IDF by +12.5% accuracy.
2. **Escalation Precision & Recall**: Both TF-IDF + LogReg and Hybrid RAG achieve strong escalation handling (Escalation F1 > 0.90), successfully capturing sensitive requests (DM routing, order verification).
3. **Combined Decision Accuracy**: Hybrid RAG achieves the highest downstream Combined Accuracy (**58.00%**), demonstrating that LLM semantic intent prediction combined with deterministic policy safety provides the best overall end-to-end support automation.
