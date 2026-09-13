"""
evaluate_heldout_hybrid.py
--------------------------
FINAL FROZEN EVALUATION on HELDOUT TEST SET (heldout_test_final.csv).

System is 100% FROZEN:
  - Model: Ollama / qwen2.5:3b
  - Architecture: Hybrid RAG (LLM Intent + Deterministic Policy Escalation)
  - Retrieval: Top 3 historical cases (query conversation_id excluded)

Strict Isolation Protocol:
  - Ground truth labels (intent, expected_escalation, difficulty) are NOT accessed during inference
  - Results saved under evaluation/results/ with heldout suffix
"""

import csv
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import ALL_INTENTS
from retrieval import ConversationRetriever
from llm_agent import LLMAgent, OllamaProvider

HELDOUT_SET_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'hiver-sde-assignment',
    'evaluation', 'heldout_test_final.csv'
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, 'hybrid_rag_heldout_checkpoint.json')

MAX_RETRIES = 5
RETRY_DELAY_S = 5.0

DEV_BASELINE = {
    'intent_accuracy': 0.6850,
    'intent_macro_f1': 0.6514,
    'escalation_accuracy': 0.7550,
    'escalation_f1': 0.8414,
    'combined_accuracy': 0.5950,
}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def load_heldout_set(path):
    if not os.path.exists(path):
        print(f"ERROR: Heldout set not found at {path}")
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} heldout test examples from: {path}")
    return rows


def call_with_retry(agent, message, context, cases):
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


def run_heldout_evaluation():
    print("=" * 80)
    print("  FINAL HELDOUT TEST SET EVALUATION — FROZEN HYBRID RAG SYSTEM")
    print("  Model: Ollama / qwen2.5:3b  |  Dataset: heldout_test_final.csv (n=200)")
    print("=" * 80)

    provider = OllamaProvider()
    print(f"LLM Provider : {provider.name}")

    print("\nLoading Conversation Retriever ...")
    retriever = ConversationRetriever()
    print(f"Index loaded: {retriever.num_indexed:,} conversations")

    agent = LLMAgent(provider=provider)

    rows = load_heldout_set(HELDOUT_SET_PATH)
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
        # Feature Extraction (Strict Isolation: NO GT labels passed to agent)
        eid = row['example_id']
        cid = str(row.get('conversation_id', ''))
        msg = row['customer_message']
        ctx = row['conversation_context']

        # Ground Truth labels saved ONLY for evaluation harness AFTER prediction
        gt_i = row['intent']
        gt_e = str(row['expected_escalation'])
        gt_d = row.get('difficulty', 'unknown')

        t_ret = time.time()
        cases = retriever.retrieve(msg, ctx, top_k=3, exclude_ids={cid} if cid else None)
        ret_ms = (time.time() - t_ret) * 1000
        retrieval_times.append(ret_ms)

        # Inference Call — zero ground truth passed
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
            'difficulty': gt_d,
            'policy_escalation_reason': pred_reason,
            'confidence': conf,
            'llm_latency_ms': llm_ms,
            'retrieval_latency_ms': ret_ms,
            'top1_similarity': cases[0].score if cases else 0.0,
            'top1_case_id': cases[0].conversation_id if cases else '',
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

    # Per-intent metrics
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

    # Difficulty performance breakdown
    diff_perf = {}
    diff_groups = defaultdict(list)
    for p in predictions:
        diff_groups[p['difficulty']].append(p)
    for d, items in diff_groups.items():
        if items:
            i_acc = sum(x['intent_correct'] for x in items) / len(items)
            e_acc = sum(x['escalation_correct'] for x in items) / len(items)
            c_acc = sum(x['combined_correct'] for x in items) / len(items)
            diff_perf[d] = {
                'count': len(items),
                'intent_accuracy': round(i_acc, 4),
                'escalation_accuracy': round(e_acc, 4),
                'combined_accuracy': round(c_acc, 4),
            }

    # Distribution counts & Disagreements
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

    # Print Summary Report
    print("\n" + "=" * 80)
    print("  FINAL HELDOUT EVALUATION RESULTS — FROZEN HYBRID RAG SYSTEM")
    print("=" * 80)
    print(f"  Total examples           : {total}")
    print(f"  Successful examples      : {total - n_failed}")
    print(f"  Failed LLM calls         : {n_failed}")
    print(f"  Intent Accuracy          : {intent_acc*100:.2f}% (Dev: {DEV_BASELINE['intent_accuracy']*100:.2f}%)")
    print(f"  Intent Macro F1          : {macro_f1:.4f} (Dev: {DEV_BASELINE['intent_macro_f1']:.4f})")
    print(f"  Final Escalation Acc     : {esc_acc*100:.2f}% (Dev: {DEV_BASELINE['escalation_accuracy']*100:.2f}%)")
    print(f"  Final Escalation Prec    : {esc_p:.4f}")
    print(f"  Final Escalation Rec     : {esc_r:.4f}")
    print(f"  Final Escalation F1      : {esc_f1:.4f} (Dev: {DEV_BASELINE['escalation_f1']:.4f})")
    print(f"  Combined Decision Acc    : {combined_acc*100:.2f}% (Dev: {DEV_BASELINE['combined_accuracy']*100:.2f}%)")
    print(f"  Avg LLM latency          : {avg_latency_ms/1000:.2f} s ({avg_latency_ms:.0f} ms)")
    print(f"  Median LLM latency       : {median_latency_ms/1000:.2f} s ({median_latency_ms:.0f} ms)")
    print(f"  Avg retrieval latency    : {sum(retrieval_times)/len(retrieval_times):.1f} ms")
    print(f"  Total runtime            : {t_total:.1f} s ({t_total/60:.1f} min)")
    print("=" * 80)

    # Save outputs
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pred_path = os.path.join(RESULTS_DIR, 'hybrid_rag_heldout_predictions.csv')
    with open(pred_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0].keys()))
        writer.writeheader()
        writer.writerows(predictions)

    eval_json = {
        'evaluation_type': 'FINAL FROZEN HELDOUT EVALUATION',
        'architecture': 'Hybrid RAG (Ollama qwen2.5:3b + Deterministic Escalation Policy)',
        'provider': provider.name,
        'dataset': 'heldout_test_final.csv',
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
        'development_baseline_comparison': {
            'dev_intent_accuracy': DEV_BASELINE['intent_accuracy'],
            'dev_intent_macro_f1': DEV_BASELINE['intent_macro_f1'],
            'dev_escalation_accuracy': DEV_BASELINE['escalation_accuracy'],
            'dev_escalation_f1': DEV_BASELINE['escalation_f1'],
            'dev_combined_accuracy': DEV_BASELINE['combined_accuracy'],
            'delta_intent_acc': round(intent_acc - DEV_BASELINE['intent_accuracy'], 4),
            'delta_escalation_acc': round(esc_acc - DEV_BASELINE['escalation_accuracy'], 4),
            'delta_combined_acc': round(combined_acc - DEV_BASELINE['combined_accuracy'], 4),
        },
        'difficulty_performance': diff_perf,
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
            'avg_retrieval_ms': round(sum(retrieval_times)/len(retrieval_times), 1),
        }
    }

    json_path = os.path.join(RESULTS_DIR, 'hybrid_rag_heldout_evaluation.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(eval_json, f, indent=2)

    md_path = os.path.join(RESULTS_DIR, 'hybrid_rag_heldout_results.md')
    _write_md_report(md_path, eval_json, predictions, per_intent, top_conf,
                     tp_e, fp_e, fn_e, tn_e, llm_times, retrieval_times, t_total, total, diff_perf)

    print(f"\nFinal Heldout Artifacts Saved:")
    print(f"  {pred_path}")
    print(f"  {json_path}")
    print(f"  {md_path}")

    return eval_json


def _write_md_report(path, eval_json, predictions, per_intent, top_conf,
                     tp_e, fp_e, fn_e, tn_e, llm_times, retrieval_times, t_total, total, diff_perf):
    m = eval_json['overall_metrics']
    comp = eval_json['development_baseline_comparison']
    eb = eval_json['escalation_breakdown']

    with open(path, 'w', encoding='utf-8') as f:
        f.write("# Final Heldout Test Set Evaluation Results (Frozen Hybrid RAG System)\n\n")
        f.write("> [!IMPORTANT]\n")
        f.write(f"> **Evaluation Scope**: Held-out test set (`heldout_test_final.csv`, n={total}).\n")
        f.write(f"> **System Status**: 100% FROZEN (no code, prompt, model, or rule modifications).\n")
        f.write(f"> **Strict Isolation**: Ground truth labels were **NOT accessed during inference**.\n\n")

        f.write("## 1. Overall Performance Metrics\n\n")
        f.write("| Metric | Heldout Result | Dev Baseline | Delta |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Intent Accuracy** | **{m['intent_accuracy']*100:.2f}%** | 68.50% | {comp['delta_intent_acc']*100:+.2f}% |\n")
        f.write(f"| **Intent Macro F1** | **{m['intent_macro_f1']:.4f}** | 0.6514 | {m['intent_macro_f1']-0.6514:+.4f} |\n")
        f.write(f"| **Escalation Accuracy** | **{m['escalation_accuracy']*100:.2f}%** | 75.50% | {comp['delta_escalation_acc']*100:+.2f}% |\n")
        f.write(f"| **Escalation Precision** | **{m['escalation_precision']:.4f}** | 0.7471 | {m['escalation_precision']-0.7471:+.4f} |\n")
        f.write(f"| **Escalation Recall** | **{m['escalation_recall']:.4f}** | 0.9630 | {m['escalation_recall']-0.9630:+.4f} |\n")
        f.write(f"| **Escalation F1** | **{m['escalation_f1']:.4f}** | 0.8414 | {m['escalation_f1']-0.8414:+.4f} |\n")
        f.write(f"| **Combined Decision Accuracy** | **{m['combined_accuracy']*100:.2f}%** | 59.50% | {comp['delta_combined_acc']*100:+.2f}% |\n\n")

        f.write("## 2. Difficulty Breakdown\n\n")
        f.write("| Difficulty | Count | Intent Acc | Escalation Acc | Combined Acc |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for d, dp in sorted(diff_perf.items()):
            f.write(f"| `{d}` | {dp['count']} | {dp['intent_accuracy']*100:.1f}% | {dp['escalation_accuracy']*100:.1f}% | {dp['combined_accuracy']*100:.1f}% |\n")

        f.write("\n## 3. Escalation Confusion Matrix (2x2)\n\n")
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
    run_heldout_evaluation()
