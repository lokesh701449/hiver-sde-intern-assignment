# Reply-Quality LLM-as-Judge Evaluation & Human Agreement Study

> [!IMPORTANT]
> **Scope**: 50 representative sample examples from `heldout_test_final.csv` predictions.
> **Judge Model**: `ollama/qwen2.5:3b`
> **System Status**: FROZEN (No code, prompt, model, or escalation policy changes).

## 1. LLM-as-Judge Dimension Averages (1-5 Scale)

| Dimension | Average Score (1-5) | Description |
| :--- | :---: | :--- |
| **Relevance** | **3.180** | Directly addresses customer's primary issue |
| **Groundedness** | **3.200** | Consistent with retrieved historical evidence |
| **Helpfulness** | **2.620** | Provides clear next steps or escalation routing |
| **Tone** | **4.580** | Professional, empathetic, and concise |
| **Safety / Escalation** | **4.980** | Avoids false private claims; routes properly |
| **Overall Quality** | **3.580** | Reasonable first-draft agent response |

## 2. Strengths & Weaknesses Analysis

- **Strongest Dimension**: `safety` (Score: 4.980 / 5.0)
- **Weakest Dimension**: `helpfulness` (Score: 2.620 / 5.0)

### Recurring Reply-Quality Observations
1. **High Safety & Groundedness**: The system cleanly avoids hallucinating private account actions on social media and routes private issues via DM.
2. **Concise / Empty Public Replies**: For escalated cases, the model defaults to empty or short public replies, relying on the deterministic policy to trigger agent routing.

## 3. Human Agreement Study Setup

A human annotation template has been prepared at `evaluation/reply_quality/human_annotation_template.csv`.
The template displays the current conversation, generated reply, and top retrieved historical evidence, while hiding the LLM judge's scores to prevent annotator bias.

### Human Agreement Statistics

| Dimension | Human Mean | Judge Mean | Mean Abs Diff | Weighted Kappa | Exact Acc % | Within-1 Acc % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `relevance` | 3.90 | 3.18 | 1.40 | **0.0598** | 22.0% | 58.0% |
| `groundedness` | 4.06 | 3.20 | 1.14 | **0.0535** | 14.0% | 72.0% |
| `helpfulness` | 2.90 | 2.62 | 0.64 | **0.2430** | 42.0% | 94.0% |
| `tone` | 3.20 | 4.58 | 1.38 | **0.0670** | 10.0% | 52.0% |
| `safety` | 4.94 | 4.98 | 0.04 | **0.4845** | 96.0% | 100.0% |
| `overall` | 3.38 | 3.58 | 0.48 | **0.2214** | 56.0% | 96.0% |
