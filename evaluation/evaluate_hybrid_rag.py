"""
evaluate_hybrid_rag.py
----------------------
Evaluate Hybrid RAG (Ollama qwen2.5:3b + Deterministic Escalation Policy)
on the FULL 200-example DEVELOPMENT SET (golden_set_final.csv).

Protocol & Constraints:
  - Uses ONLY golden_set_final.csv
  - NEVER accesses or modifies heldout_test_final.csv
  - Excludes current conversation_id from top-3 FAISS retrieval
  - LLM predicts intent and reply; deterministic policy produces final escalation decision
  - Checkpoint enabled per-example for 100% crash resilience
"""

import csv
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import ALL_INTENTS, LLM_PROVIDER
from retrieval import ConversationRetriever
from llm_agent import LLMAgent, get_provider, MockLLMProvider, OllamaProvider

DEV_SET_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'hiver-sde-assignment',
    'evaluation', 'golden_set_final.csv'
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, 'hybrid_rag_checkpoint.json')

MAX_RETRIES = 5
RETRY_DELAY_S = 5.0


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def load_dev_set(path):
    if not os.path.exists(path):
        print(f"ERROR: Dev set not found at {path}")
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} dev examples from: {path}")
    return rows


def call_with_retry(agent, message, context, cases):
    """Call LLM with exponential backoff on errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return agent.classify(message, context, cases)
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY_S * attempt
                print(f"    [Retry {attempt}/{MAX_RETRIES}] API issue — waiting {wait:.0f}s ... ({str(e)[:80]})")
                time.sleep(wait)
            else:
                print(f"    [Failed all retries]: {e}")
                raise
    return None


def save_checkpoint(predictions, start_time):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump({'predictions': predictions, 'start_time': start_time}, f)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH) as f:
                data = json.load(f)
            preds = data.get('predictions', [])
            if preds:
                print(f"Checkpoint found: {len(preds)} examples already completed — resuming.")
                return preds, data.get('start_time', time.time())
        except Exception:
            pass
    return [], time.time()


def run_evaluation():
    print("=" * 80)
    print("  HYBRID RAG EVALUATION — FULL 200 DEVELOPMENT EXAMPLES")
    print("  Model: Ollama / qwen2.5:3b  |  Architecture: Hybrid (LLM Intent + Policy Escalation)")
    print("=" * 80)

    # Provider setup
    provider = OllamaProvider()
    print(f"LLM Provider : {provider.name}")

    # Load retriever & agent
    print("\nLoading Conversation Retriever ...")
    retriever = ConversationRetriever()
    print(f"Index loaded: {retriever.num_indexed:,} conversations")

    agent = LLMAgent(provider=provider)

    rows = load_dev_set(DEV_SET_PATH)
    total = len(rows)

    predictions, t_start = load_checkpoint()
    completed_ids = {p['example_id'] for p in predictions}

    n_failed = sum(1 for p in predictions if p.get('failed'))
    retrieval_times = [p['retrieval_latency_ms'] for p in predictions]
    llm_times = [p['llm_latency_ms'] for p in predictions if p['llm_latency_ms'] > 0]
    confidences = [p['confidence'] for p in predictions if p.get('confidence', 0) > 0]

    remaining = [r for r in rows if r['example_id'] not in completed_ids]
    print(f"Examples remaining to process: {len(remaining)}\n")

    for i, row in enumerate(remaining, len(completed_ids) + 1):
        eid = row['example_id']
        cid = str(row.get('conversation_id', ''))
        msg = row['customer_message']
        ctx = row['conversation_context']
        gt_i = row['intent']
        gt_e = row['expected_escalation']

        # Retrieval with conversation_id exclusion
        t_ret = time.time()
        cases = retriever.retrieve(msg, ctx, top_k=3, exclude_ids={cid} if cid else None)
        ret_ms = (time.time() - t_ret) * 1000
        retrieval_times.append(ret_ms)

        result = call_with_retry(agent, msg, ctx, cases)

        if result is None:
            n_failed += 1
            pred_i = 'unclear_other'
            pred_e = 'False'
            llm_e = 'False'
            pred_reason = 'LLM_CALL_FAILED'
            conf = 0.0
            llm_ms = 0.0
            print(f"  [{i:3d}/{total}] {eid} FAILED")
        else:
            pred_i = result.intent
            pred_e = str(result.escalate)
            llm_e = str(result.llm_escalate)
            pred_reason = result.escalation_reason
            conf = result.confidence
            llm_ms = result.latency_ms
            llm_times.append(llm_ms)
            confidences.append(conf)

            correct_i = "✓" if pred_i == gt_i else "✗"
            correct_e = "✓" if pred_e == gt_e else "✗"
            print(f"  [{i:3d}/{total}] ID:{eid:<4} | Intent {correct_i} {pred_i:<32} | Esc {correct_e} ({pred_e}) | {llm_ms:.0f}ms")

        predictions.append({
            'example_id': eid,
            'conversation_id': cid,
            'customer_message': msg[:200],
            'conversation_context': ctx[:200],
            'human_intent': gt_i,
            'predicted_intent': pred_i,
            'intent_correct': (pred_i == gt_i),
            'human_escalation': gt_e,
            'final_escalation': pred_e,
            'escalation_correct': (pred_e == gt_e),
            'llm_escalation': llm_e,
            'escalation_disagreement': (pred_e != llm_e),
            'combined_correct': (pred_i == gt_i) and (pred_e == gt_e),
            'policy_escalation_reason': pred_reason,
            'confidence': conf,
            'llm_latency_ms': llm_ms,
            'retrieval_latency_ms': ret_ms,
            'top1_similarity': cases[0].score if cases else 0.0,
            'top1_case_id': cases[0].conversation_id if cases else '',
            'top1_customer_text': cases[0].customer_text[:200] if cases else '',
            'top1_amazon_reply': cases[0].amazon_turns[0][:200] if cases and cases[0].amazon_turns else '',
            'failed': (result is None),
        })

        save_checkpoint(predictions, t_start)
        if len(predictions) % 20 == 0:
            cur_acc_i = sum(p['intent_correct'] for p in predictions) / len(predictions) * 100
            cur_acc_e = sum(p['escalation_correct'] for p in predictions) / len(predictions) * 100
            print(f"  [Checkpoint {len(predictions)}/{total}] Intent Acc: {cur_acc_i:.1f}% | Esc Acc: {cur_acc_e:.1f}%")

    t_total = time.time() - t_start

    # Metrics computation
    y_true_intent = [p['human_intent'] for p in predictions]
    y_pred_intent = [p['predicted_intent'] for p in predictions]
    y_true_esc = [p['human_escalation'] for p in predictions]
    y_pred_esc = [p['final_escalation'] for p in predictions]
    y_llm_esc = [p['llm_escalation'] for p in predictions]

    intent_correct = sum(p['intent_correct'] for p in predictions)
    esc_correct = sum(p['escalation_correct'] for p in predictions)
    combined_correct = sum(p['combined_correct'] for p in predictions)

    intent_acc = intent_correct / total
    esc_acc = esc_correct / total
    combined_acc = combined_correct / total

    # Per-intent metrics & confusion matrix
    cm = defaultdict(lambda: defaultdict(int))
    for g, p in zip(y_true_intent, y_pred_intent):
        cm[g][p] += 1

    gt_counts = Counter(y_true_intent)
    per_intent = {}
    f1_list = []
    for intent in ALL_INTENTS:
        tp = cm[intent][intent]
        fp = sum(cm[o][intent] for o in ALL_INTENTS if o != intent)
        fn = sum(cm[intent][o] for o in ALL_INTENTS if o != intent)
        p_, r_, f_ = prf(tp, fp, fn)
        per_intent[intent] = {
            'support': gt_counts[intent],
            'precision': round(p_, 4),
            'recall': round(r_, 4),
            'f1': round(f_, 4),
        }
        if gt_counts[intent] > 0:
            f1_list.append(f_)
    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

    # Escalation CM
    tp_e = sum(1 for g, p in zip(y_true_esc, y_pred_esc) if g == 'True' and p == 'True')
    fp_e = sum(1 for g, p in zip(y_true_esc, y_pred_esc) if g == 'False' and p == 'True')
    fn_e = sum(1 for g, p in zip(y_true_esc, y_pred_esc) if g == 'True' and p == 'False')
    tn_e = sum(1 for g, p in zip(y_true_esc, y_pred_esc) if g == 'False' and p == 'False')
    esc_p, esc_r, esc_f1 = prf(tp_e, fp_e, fn_e)

    # Distribution counts
    policy_true_cnt = sum(1 for p in y_pred_esc if p == 'True')
    policy_false_cnt = sum(1 for p in y_pred_esc if p == 'False')
    llm_true_cnt = sum(1 for p in y_llm_esc if p == 'True')
    llm_false_cnt = sum(1 for p in y_llm_esc if p == 'False')

    disagreement_cnt = sum(1 for p in predictions if p['escalation_disagreement'])
    intent_ok_esc_wrong = sum(1 for p in predictions if p['intent_correct'] and not p['escalation_correct'])
    intent_wrong_esc_ok = sum(1 for p in predictions if not p['intent_correct'] and p['escalation_correct'])

    avg_latency_ms = sum(llm_times) / len(llm_times) if llm_times else 0.0
    median_latency_ms = statistics.median(llm_times) if llm_times else 0.0

    # Top confusions
    confusion = Counter()
    for g, p in zip(y_true_intent, y_pred_intent):
        if g != p:
            confusion[(g, p)] += 1
    top_conf = [{'actual': g, 'predicted': p, 'count': c}
                for (g, p), c in confusion.most_common(10)]

    # Print Report
    print("\n" + "=" * 80)
    print("  HYBRID RAG EVALUATION RESULTS (Ollama qwen2.5:3b + Policy, Dev Set)")
    print("=" * 80)
    print(f"  Total examples           : {total}")
    print(f"  Successful examples      : {total - n_failed}")
    print(f"  Failed LLM calls         : {n_failed}")
    print(f"  Intent Accuracy          : {intent_acc*100:.2f}% ({intent_correct}/{total})")
    print(f"  Intent Macro F1          : {macro_f1:.4f}")
    print(f"  Final Escalation Acc     : {esc_acc*100:.2f}% ({esc_correct}/{total})")
    print(f"  Final Escalation Prec    : {esc_p:.4f}")
    print(f"  Final Escalation Rec     : {esc_r:.4f}")
    print(f"  Final Escalation F1      : {esc_f1:.4f}")
    print(f"  Combined Decision Acc    : {combined_acc*100:.2f}% ({combined_correct}/{total})")
    print(f"  Avg LLM latency          : {avg_latency_ms/1000:.2f} s ({avg_latency_ms:.0f} ms)")
    print(f"  Median LLM latency       : {median_latency_ms/1000:.2f} s ({median_latency_ms:.0f} ms)")
    print(f"  Total runtime            : {t_total:.1f} s ({t_total/60:.1f} min)")
    print("-" * 80)
    print("  ESCALATION COMPARISON & DISAGREEMENTS:")
    print(f"    - Policy Escalation counts  : Escalate=True: {policy_true_cnt} | Escalate=False: {policy_false_cnt}")
    print(f"    - LLM Escalation counts     : Escalate=True: {llm_true_cnt} | Escalate=False: {llm_false_cnt}")
    print(f"    - Policy vs LLM Disagreements : {disagreement_cnt} examples")
    print(f"    - Intent Correct, Esc Wrong : {intent_ok_esc_wrong} examples")
    print(f"    - Intent Wrong, Esc Correct : {intent_wrong_esc_ok} examples")
    print("=" * 80)

    # Save CSV, JSON, MD
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pred_path = os.path.join(RESULTS_DIR, 'llm_rag_hybrid_predictions.csv')
    with open(pred_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0].keys()))
        writer.writeheader()
        writer.writerows(predictions)

    eval_json = {
        'architecture': 'Hybrid RAG (Ollama qwen2.5:3b + Deterministic Escalation Policy)',
        'provider': provider.name,
        'total_examples': total,
        'successful_examples': total - n_failed,
        'failed_calls': n_failed,
        'overall_metrics': {
            'intent_accuracy': round(intent_acc, 4),
            'intent_macro_f1': round(macro_f1, 4),
            'escalation_accuracy': round(esc_acc, 4),
            'escalation_precision': round(esc_p, 4),
            'escalation_recall': round(esc_r, 4),
            'escalation_f1': round(esc_f1, 4),
            'combined_accuracy': round(combined_acc, 4),
        },
        'escalation_breakdown': {
            'policy_true': policy_true_cnt,
            'policy_false': policy_false_cnt,
            'llm_true': llm_true_cnt,
            'llm_false': llm_false_cnt,
            'disagreements_llm_vs_policy': disagreement_cnt,
            'intent_correct_escalation_wrong': intent_ok_esc_wrong,
            'intent_wrong_escalation_correct': intent_wrong_esc_ok,
            'escalation_confusion_matrix': {'tp': tp_e, 'fp': fp_e, 'fn': fn_e, 'tn': tn_e},
        },
        'per_intent': per_intent,
        'full_confusion_matrix': {g: dict(cm[g]) for g in ALL_INTENTS},
        'top_confusions': top_conf,
        'runtime': {
            'total_s': round(t_total, 1),
            'avg_llm_ms': round(avg_latency_ms, 1),
            'median_llm_ms': round(median_latency_ms, 1),
            'avg_retrieval_ms': round(sum(retrieval_times) / len(retrieval_times), 1),
        }
    }

    json_path = os.path.join(RESULTS_DIR, 'llm_rag_hybrid_evaluation.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(eval_json, f, indent=2)

    md_path = os.path.join(RESULTS_DIR, 'llm_rag_hybrid_results.md')
    _write_md_report(md_path, eval_json, predictions, per_intent, top_conf,
                     tp_e, fp_e, fn_e, tn_e, llm_times, retrieval_times, t_total, total)

    print(f"\nArtifacts saved:")
    print(f"  {pred_path}")
    print(f"  {json_path}")
    print(f"  {md_path}")

    return eval_json


def _write_md_report(path, eval_json, predictions, per_intent, top_conf,
                     tp_e, fp_e, fn_e, tn_e, llm_times, retrieval_times, t_total, total):
    m = eval_json['overall_metrics']
    eb = eval_json['escalation_breakdown']

    with open(path, 'w', encoding='utf-8') as f:
        f.write("# Hybrid RAG Evaluation Results (Development Set)\n\n")
        f.write("> [!IMPORTANT]\n")
        f.write(f"> **Architecture**: Hybrid RAG (Ollama `qwen2.5:3b` for Intent + Reply; Deterministic Policy for Escalation).\n")
        f.write(f"> **Dev Set**: `golden_set_final.csv` (n={total}). `heldout_test_final.csv` was **NOT accessed**.\n\n")

        f.write("## 1. Overall Metrics Summary\n\n")
        f.write("| Metric | Value |\n")
        f.write("| :--- | :---: |\n")
        f.write(f"| Intent Accuracy | **{m['intent_accuracy']*100:.2f}%** ({sum(p['intent_correct'] for p in predictions)}/{total}) |\n")
        f.write(f"| Intent Macro F1 | **{m['intent_macro_f1']:.4f}** |\n")
        f.write(f"| Final Escalation Accuracy | **{m['escalation_accuracy']*100:.2f}%** ({sum(p['escalation_correct'] for p in predictions)}/{total}) |\n")
        f.write(f"| Final Escalation Precision | **{m['escalation_precision']:.4f}** |\n")
        f.write(f"| Final Escalation Recall | **{m['escalation_recall']:.4f}** |\n")
        f.write(f"| Final Escalation F1 | **{m['escalation_f1']:.4f}** |\n")
        f.write(f"| Combined Decision Accuracy | **{m['combined_accuracy']*100:.2f}%** ({sum(p['combined_correct'] for p in predictions)}/{total}) |\n")
        f.write(f"| Total Runtime | {t_total:.1f}s ({t_total/60:.1f} min) |\n\n")

        f.write("## 2. Escalation Comparison & Disagreements\n\n")
        f.write("| Category | Count |\n")
        f.write("| :--- | :---: |\n")
        f.write(f"| Deterministic Policy Escalate=True | {eb['policy_true']} |\n")
        f.write(f"| Deterministic Policy Escalate=False | {eb['policy_false']} |\n")
        f.write(f"| Raw LLM Escalate=True | {eb['llm_true']} |\n")
        f.write(f"| Raw LLM Escalate=False | {eb['llm_false']} |\n")
        f.write(f"| LLM vs Policy Disagreements | {eb['disagreements_llm_vs_policy']} |\n")
        f.write(f"| Intent Correct & Escalation Wrong | {eb['intent_correct_escalation_wrong']} |\n")
        f.write(f"| Intent Wrong & Escalation Correct | {eb['intent_wrong_escalation_correct']} |\n\n")

        f.write("## 3. Escalation Confusion Matrix (2x2)\n\n")
        f.write("| | Predicted Escalate | Predicted No Escalate |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write(f"| **Actual Escalate** | {tp_e} (TP) | {fn_e} (FN) |\n")
        f.write(f"| **Actual No Escalate** | {fp_e} (FP) | {tn_e} (TN) |\n\n")

        f.write("## 4. Per-Intent Performance\n\n")
        f.write("| Intent | Support | Precision | Recall | F1 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for intent in sorted(per_intent, key=lambda x: -per_intent[x]['f1']):
            m2 = per_intent[intent]
            f.write(f"| `{intent}` | {m2['support']} | {m2['precision']:.4f} | {m2['recall']:.4f} | **{m2['f1']:.4f}** |\n")

        f.write("\n## 5. Top 10 Intent Confusions\n\n")
        f.write("| Rank | Actual Intent | Predicted Intent | Count |\n")
        f.write("| :---: | :--- | :--- | :---: |\n")
        for i, c in enumerate(top_conf, 1):
            f.write(f"| {i} | `{c['actual']}` | `{c['predicted']}` | {c['count']} |\n")


if __name__ == '__main__':
    run_evaluation()
