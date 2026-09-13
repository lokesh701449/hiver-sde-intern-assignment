"""
evaluate_llm_rag.py — V1
--------------------------
Evaluate Gemini + RAG on the DEVELOPMENT SET (golden_set_final.csv).

Protocol:
  - Each example: customer_message + conversation_context → retrieve top-5 →
    Gemini classifies intent + escalation
  - NEVER uses held-out test set
  - NEVER passes human labels to the LLM
  - NEVER tunes the prompt based on individual errors
  - Rate-limited: 3s between calls to respect quota

This is EVALUATION V1 (prompt unchanged from initial design in llm_agent.py).

Usage:
    GEMINI_API_KEY=<key> GEMINI_MODEL=gemini-3.6-flash LLM_PROVIDER=gemini \
    python3 evaluation/evaluate_llm_rag.py

Output:
    evaluation/results/llm_rag_dev_predictions.csv
    evaluation/results/llm_rag_dev_evaluation.json
    evaluation/results/llm_rag_dev_results.md
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
from llm_agent import LLMAgent, get_provider, MockLLMProvider

# ─── Paths ────────────────────────────────────────────────────────────────────
DEV_SET_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'hiver-sde-assignment',
    'evaluation', 'golden_set_final.csv'
)
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), 'results')

# ─── Rate limiting ────────────────────────────────────────────────────────────
INTER_CALL_DELAY_S = 4.0      # seconds between LLM calls (quota guard for cloud APIs)
MAX_RETRIES        = 5
RETRY_DELAY_S      = 20.0     # base retry delay — exponential backoff applied
CHECKPOINT_PATH    = os.path.join(os.path.dirname(__file__), 'results', 'llm_rag_checkpoint.json')

# ─── Baselines (from sibling projects, clearly labelled by protocol) ──────────
BASELINES = {
    "Rule-Based V2 (dev full-set)": {
        "intent_acc": 0.615, "macro_f1": 0.5495,
        "esc_acc": 0.785, "esc_f1": 0.8436, "combined_acc": 0.520,
        "protocol": "Full 200-example dev set evaluation",
    },
    "TF-IDF + LR (5-fold CV dev)": {
        "intent_acc": 0.575, "macro_f1": 0.4279,
        "esc_acc": 0.805, "esc_f1": None, "combined_acc": None,
        "protocol": "5-fold stratified CV on 200 dev examples",
    },
    "DistilBERT (5-fold CV dev)": {
        "intent_acc": 0.420, "macro_f1": 0.2505,
        "esc_acc": 0.675, "esc_f1": None, "combined_acc": 0.315,
        "protocol": "5-fold CV on 200 dev examples (high variance ±5%)",
    },
    "DistilBERT (dev 80/20 split)": {
        "intent_acc": 0.750, "macro_f1": 0.6959,
        "esc_acc": 0.650, "esc_f1": 0.7813, "combined_acc": 0.475,
        "protocol": "Single 80/20 stratified split (n=40 val; may be high-variance)",
    },
}


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
    """Call LLM with exponential-backoff retry on 429/503/overload errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return agent.classify(message, context, cases)
        except Exception as e:
            err = str(e)
            is_retryable = (
                '429' in err or '503' in err
                or 'RESOURCE_EXHAUSTED' in err
                or 'UNAVAILABLE' in err
                or 'overload' in err.lower()
                or 'quota' in err.lower()
                or 'high demand' in err.lower()
            )
            if is_retryable and attempt < MAX_RETRIES:
                wait = RETRY_DELAY_S * attempt   # 20s, 40s, 60s …
                print(f"    [Retry {attempt}/{MAX_RETRIES}] API issue — waiting {wait:.0f}s ... ({err[:80]})")
                time.sleep(wait)
            else:
                raise
    return None   # unreachable but satisfies type checker


def save_checkpoint(predictions, start_time):
    """Save completed predictions so evaluation can resume after a crash."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump({'predictions': predictions, 'start_time': start_time}, f)


def load_checkpoint():
    """Load checkpoint if it exists and is not empty."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        preds = data.get('predictions', [])
        if preds:
            print(f"Checkpoint found: {len(preds)} examples already completed — resuming.")
            return preds, data.get('start_time', time.time())
    return [], time.time()


def run_evaluation(provider_name=None):
    # t_start is set by load_checkpoint() — either resumed or fresh

    print("  LLM + RAG EVALUATION — DEVELOPMENT SET (V1)")
    print("  EVALUATION PROTOCOL: Fixed V1 prompt, no post-hoc tuning")
    print("=" * 70)

    # ─── Guard: refuse to run mock as real ───────────────────────────────────
    provider = get_provider(provider_name)
    if isinstance(provider, MockLLMProvider):
        print("\nERROR: LLM_PROVIDER is 'mock'. This script requires a real LLM.")
        print("Set:   LLM_PROVIDER=ollama or LLM_PROVIDER=gemini")
        sys.exit(1)

    inter_call_delay = 0.1 if "ollama" in provider.name.lower() else INTER_CALL_DELAY_S

    print(f"\nLLM provider : {provider.name}")
    print(f"Rate limit   : {inter_call_delay}s between calls")

    # ─── Load retriever ───────────────────────────────────────────────────────
    print("\nLoading retriever ...")
    retriever = ConversationRetriever()
    print(f"Index: {retriever.num_indexed:,} conversations")

    # ─── Load agent ───────────────────────────────────────────────────────────
    agent = LLMAgent(provider=provider)

    # ─── Load dev set ─────────────────────────────────────────────────────────
    rows = load_dev_set(DEV_SET_PATH)
    total = len(rows)

    # ─── Checkpoint/resume ────────────────────────────────────────────────────
    predictions, t_start = load_checkpoint()
    completed_ids = {p['example_id'] for p in predictions}
    if completed_ids:
        print(f"Resuming from checkpoint: {len(completed_ids)}/{total} already done.")

    n_failed          = sum(1 for p in predictions if p.get('failed'))
    n_invalid_json    = 0
    retrieval_times   = [p['retrieval_latency_ms'] for p in predictions]
    llm_times         = [p['llm_latency_ms']       for p in predictions if p['llm_latency_ms'] > 0]
    retrieved_scores  = [p['top1_similarity']       for p in predictions if p.get('top1_similarity')]
    confidences       = [p['confidence']            for p in predictions if p['confidence'] > 0]

    remaining = [r for r in rows if r['example_id'] not in completed_ids]
    print(f"Examples remaining: {len(remaining)}\n")

    # ─── Pre-flight: wait until API is responsive ───────────────────────────
    print("Pre-flight: checking API connectivity ...")
    for probe_attempt in range(1, 12):
        try:
            _ = provider.complete(
                "You are a JSON API.",
                'Return {"status": "ok"}'
            )
            print(f"API ready (probe attempt {probe_attempt}).\n")
            break
        except Exception as e:
            wait = 20 * probe_attempt
            print(f"  [Probe {probe_attempt}/11] API not ready: {str(e)[:80]}")
            print(f"  Waiting {wait}s before retry ...")
            time.sleep(wait)
    else:
        print("ERROR: API unresponsive after 11 probes. Aborting.")
        sys.exit(1)

    for i, row in enumerate(remaining, len(completed_ids) + 1):
        eid     = row['example_id']
        msg     = row['customer_message']
        ctx     = row['conversation_context']
        gt_i    = row['intent']
        gt_e    = row['expected_escalation']
        gt_d    = row['difficulty']

        # Retrieval (exclude current conversation ID to prevent retrieval leakage)
        cid = str(row.get('conversation_id', ''))
        t_ret = time.time()
        cases = retriever.retrieve(msg, ctx, top_k=3, exclude_ids={cid} if cid else None)
        retrieval_times.append((time.time() - t_ret) * 1000)
        for c in cases:
            retrieved_scores.append(c.score)

        # LLM call with rate limiting
        if inter_call_delay > 0:
            time.sleep(inter_call_delay)
        result = call_with_retry(agent, msg, ctx, cases)

        if result is None:
            n_failed += 1
            pred_i = 'unclear_other'
            pred_e = 'False'
            pred_reason = 'LLM_CALL_FAILED'
            conf = 0.0
            llm_ms = 0.0
            print(f"  [{i:3d}/{total}] {eid} FAILED (all retries exhausted)")
        else:
            pred_i     = result.intent
            pred_e     = str(result.escalate)
            pred_reason = result.escalation_reason
            conf        = result.confidence
            llm_ms      = result.latency_ms
            llm_times.append(llm_ms)
            confidences.append(conf)

            correct_i = "✓" if pred_i == gt_i else "✗"
            correct_e = "✓" if pred_e == gt_e else "✗"
            print(f"  [{i:3d}/{total}] {eid} | Intent {correct_i} {pred_i:<35} | Esc {correct_e} | {llm_ms:.0f}ms")

        predictions.append({
            'example_id':            eid,
            'customer_message':      msg[:200],
            'conversation_context':  ctx[:200],
            'human_intent':          gt_i,
            'predicted_intent':      pred_i,
            'intent_correct':        pred_i == gt_i,
            'human_escalation':      gt_e,
            'predicted_escalation':  pred_e,
            'escalation_correct':    pred_e == gt_e,
            'combined_correct':      (pred_i == gt_i) and (pred_e == gt_e),
            'predicted_reason':      pred_reason,
            'confidence':            conf,
            'llm_latency_ms':        llm_ms,
            'retrieval_latency_ms':  retrieval_times[-1],
            'top1_similarity':       cases[0].score if cases else 0.0,
            'top1_case_id':          cases[0].conversation_id if cases else '',
            'top1_customer_text':    cases[0].customer_text[:200] if cases else '',
            'top1_amazon_reply':     cases[0].amazon_turns[0][:200] if cases and cases[0].amazon_turns else '',
            'failed':                result is None,
        })

        # Save checkpoint after EVERY example — resumable at single-example granularity
        save_checkpoint(predictions, t_start)
        done = len(predictions)
        if done % 10 == 0:
            correct_so_far = sum(p['intent_correct'] for p in predictions)
            print(f"  [Checkpoint] {done}/{total} done — running intent acc: {correct_so_far/done*100:.1f}%")

    t_total = time.time() - t_start

    # ─── Compute metrics ──────────────────────────────────────────────────────
    y_true_intent = [p['human_intent']       for p in predictions]
    y_pred_intent = [p['predicted_intent']   for p in predictions]
    y_true_esc    = [p['human_escalation']   for p in predictions]
    y_pred_esc    = [p['predicted_escalation'] for p in predictions]

    intent_correct = sum(p['intent_correct']  for p in predictions)
    esc_correct    = sum(p['escalation_correct'] for p in predictions)
    combined_correct = sum(p['combined_correct'] for p in predictions)

    intent_acc   = intent_correct  / total
    esc_acc      = esc_correct     / total
    combined_acc = combined_correct / total

    # Per-intent metrics
    cm = defaultdict(lambda: defaultdict(int))
    for g, p in zip(y_true_intent, y_pred_intent):
        cm[g][p] += 1

    gt_counts  = Counter(y_true_intent)
    per_intent = {}
    f1_list    = []
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

    # Escalation
    tp_e = sum(1 for g, p in zip(y_true_esc, y_pred_esc) if g=='True'  and p=='True')
    fp_e = sum(1 for g, p in zip(y_true_esc, y_pred_esc) if g=='False' and p=='True')
    fn_e = sum(1 for g, p in zip(y_true_esc, y_pred_esc) if g=='True'  and p=='False')
    tn_e = sum(1 for g, p in zip(y_true_esc, y_pred_esc) if g=='False' and p=='False')
    esc_p, esc_r, esc_f1 = prf(tp_e, fp_e, fn_e)

    # Latency stats
    avg_latency_ms    = sum(llm_times) / len(llm_times) if llm_times else 0.0
    median_latency_ms = statistics.median(llm_times) if llm_times else 0.0

    # Top confusions
    confusion = Counter()
    for g, p in zip(y_true_intent, y_pred_intent):
        if g != p:
            confusion[(g, p)] += 1
    top_conf = [{'actual': g, 'predicted': p, 'count': c}
                for (g, p), c in confusion.most_common(10)]

    # ─── Print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  RESULTS: LLM + RAG ({provider.name}, Dev Set)")
    print("=" * 70)
    print(f"  Total examples attempted : {total}")
    print(f"  Successful examples      : {total - n_failed}")
    print(f"  Failed LLM calls         : {n_failed}")
    print(f"  Intent Accuracy          : {intent_acc*100:.2f}%")
    print(f"  Intent Macro F1          : {macro_f1:.4f}")
    print(f"  Escalation Accuracy      : {esc_acc*100:.2f}%")
    print(f"  Escalation Precision     : {esc_p:.4f}")
    print(f"  Escalation Recall        : {esc_r:.4f}")
    print(f"  Escalation F1            : {esc_f1:.4f}")
    print(f"  Combined Decision Acc    : {combined_acc*100:.2f}%")
    print(f"  Average LLM latency      : {avg_latency_ms/1000:.2f} s ({avg_latency_ms:.0f} ms)")
    print(f"  Median LLM latency       : {median_latency_ms/1000:.2f} s ({median_latency_ms:.0f} ms)")
    print(f"  Avg retrieval latency    : {sum(retrieval_times)/len(retrieval_times):.1f} ms")
    print(f"  Total runtime            : {t_total:.1f} s ({t_total/60:.1f} min)")
    print("=" * 70)

    # ─── Save outputs ─────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    p_prefix = "ollama" if "ollama" in provider.name.lower() else "dev"

    # Predictions CSV
    pred_path = os.path.join(RESULTS_DIR, f'llm_rag_{p_prefix}_predictions.csv')
    with open(pred_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0].keys()))
        writer.writeheader(); writer.writerows(predictions)

    # JSON results
    eval_json = {
        'evaluation_version': 'V1',
        'protocol': 'Fixed V1 prompt — no post-hoc tuning',
        'provider': provider.name,
        'total_examples': total,
        'successful_examples': total - n_failed,
        'failed_calls': n_failed,
        'metrics': {
            'intent_accuracy': round(intent_acc, 4),
            'intent_macro_f1': round(macro_f1, 4),
            'escalation_accuracy': round(esc_acc, 4),
            'escalation_precision': round(esc_p, 4),
            'escalation_recall': round(esc_r, 4),
            'escalation_f1': round(esc_f1, 4),
            'combined_accuracy': round(combined_acc, 4),
        },
        'per_intent': per_intent,
        'escalation_cm': {'tp': tp_e, 'fp': fp_e, 'fn': fn_e, 'tn': tn_e},
        'full_confusion_matrix': {g: dict(cm[g]) for g in ALL_INTENTS},
        'top_confusions': top_conf,
        'runtime': {
            'total_s': round(t_total, 1),
            'avg_llm_ms': round(avg_latency_ms, 1),
            'median_llm_ms': round(median_latency_ms, 1),
            'avg_retrieval_ms': round(sum(retrieval_times)/len(retrieval_times), 1),
            'api_calls': total - n_failed,
        },
        'retrieval': {
            'avg_similarity': round(sum(retrieved_scores)/len(retrieved_scores) if retrieved_scores else 0, 4),
            'avg_confidence': round(sum(confidences)/len(confidences) if confidences else 0, 3),
        },
        'baselines': BASELINES,
    }
    json_path = os.path.join(RESULTS_DIR, f'llm_rag_{p_prefix}_evaluation.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(eval_json, f, indent=2)

    # Markdown report
    md_path = os.path.join(RESULTS_DIR, f'llm_rag_{p_prefix}_results.md')
    _write_md_report(md_path, eval_json, predictions, per_intent, top_conf,
                     tp_e, fp_e, fn_e, tn_e, llm_times, retrieval_times,
                     retrieved_scores, confidences, t_total, n_failed, total)

    print(f"\nArtifacts saved:")
    print(f"  {pred_path}")
    print(f"  {json_path}")
    print(f"  {md_path}")

    return eval_json


def _write_md_report(path, eval_json, predictions, per_intent, top_conf,
                     tp_e, fp_e, fn_e, tn_e, llm_times, retrieval_times,
                     retrieved_scores, confidences, t_total, n_failed, total):
    m     = eval_json['metrics']
    pname = eval_json['provider']
    import statistics

    with open(path, 'w', encoding='utf-8') as f:
        f.write("# LLM + RAG Evaluation — Development Set Results (V1)\n\n")
        f.write("> [!IMPORTANT]\n")
        f.write(f"> **Scope**: Development set only (`golden_set_final.csv`, n={total}). "
                f"`heldout_test_final.csv` was **NOT accessed**.\n")
        f.write(f"> **Prompt version**: V1 (fixed; no tuning based on individual development errors).\n")
        f.write(f"> **Provider**: `{pname}`\n\n")

        # ─── Comparison table ─────────────────────────────────────────────────
        f.write("## 1. All-Approaches Comparison (Development Estimates)\n\n")
        f.write("> [!WARNING]\n")
        f.write("> Metrics from different protocols (full dev-set, CV, 80/20 split) are NOT directly comparable.\n")
        f.write("> Each row is labelled with its evaluation protocol.\n\n")
        f.write("| Approach | Intent Acc | Macro F1 | Esc Acc | Esc F1 | Combined | Protocol |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :--- |\n")
        for name, b in BASELINES.items():
            ia  = f"{b['intent_acc']*100:.2f}%"
            mf  = f"{b['macro_f1']:.4f}"
            ea  = f"{b['esc_acc']*100:.2f}%"
            ef  = f"{b['esc_f1']:.4f}" if b.get('esc_f1') else "—"
            ca  = f"{b['combined_acc']*100:.2f}%" if b.get('combined_acc') else "—"
            f.write(f"| {name} | {ia} | {mf} | {ea} | {ef} | {ca} | {b['protocol']} |\n")
        f.write(f"| **LLM+RAG / Gemini 3.6 Flash (V1)** | "
                f"**{m['intent_accuracy']*100:.2f}%** | **{m['intent_macro_f1']:.4f}** | "
                f"**{m['escalation_accuracy']*100:.2f}%** | **{m['escalation_f1']:.4f}** | "
                f"**{m['combined_accuracy']*100:.2f}%** | Full 200-example dev set |\n")

        # ─── Per-intent ───────────────────────────────────────────────────────
        f.write("\n## 2. Per-Intent Metrics (LLM+RAG, Dev Set)\n\n")
        f.write("| Intent | Support | Precision | Recall | F1 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for intent in sorted(per_intent, key=lambda x: -per_intent[x]['f1']):
            m2 = per_intent[intent]
            f.write(f"| `{intent}` | {m2['support']} | {m2['precision']:.4f} | {m2['recall']:.4f} | **{m2['f1']:.4f}** |\n")

        # ─── Top confusions ───────────────────────────────────────────────────
        f.write("\n## 3. Top Intent Confusion Pairs (LLM+RAG)\n\n")
        f.write("| Rank | Actual | Predicted | Count |\n")
        f.write("| :---: | :--- | :--- | :---: |\n")
        for i, c in enumerate(top_conf, 1):
            f.write(f"| {i} | `{c['actual']}` | `{c['predicted']}` | {c['count']} |\n")

        # ─── Escalation CM ────────────────────────────────────────────────────
        f.write("\n## 4. Escalation Confusion Matrix\n\n")
        f.write("| | Predicted Escalate | Predicted No Escalate |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write(f"| **Actual Escalate** | {tp_e} (TP) | {fn_e} (FN) |\n")
        f.write(f"| **Actual No Escalate** | {fp_e} (FP) | {tn_e} (TN) |\n")

        # ─── Runtime ──────────────────────────────────────────────────────────
        f.write("\n## 5. Runtime & Retrieval Stats\n\n")
        f.write("| Metric | Value |\n")
        f.write("| :--- | :---: |\n")
        f.write(f"| Total runtime | {t_total:.0f}s |\n")
        f.write(f"| API calls made | {total - n_failed} / {total} |\n")
        f.write(f"| Failed LLM calls | {n_failed} |\n")
        f.write(f"| Avg LLM latency | {sum(llm_times)/len(llm_times) if llm_times else 0:.0f} ms |\n")
        f.write(f"| Avg retrieval latency | {sum(retrieval_times)/len(retrieval_times):.1f} ms |\n")
        f.write(f"| Avg retrieved similarity | {sum(retrieved_scores)/len(retrieved_scores) if retrieved_scores else 0:.4f} |\n")
        f.write(f"| Avg LLM confidence | {sum(confidences)/len(confidences) if confidences else 0:.3f} |\n")
        if confidences:
            f.write(f"| Confidence (min / max) | {min(confidences):.3f} / {max(confidences):.3f} |\n")
            f.write(f"| Confidence (median) | {statistics.median(confidences):.3f} |\n")

        # ─── Qualitative: 5 good retrieval examples ───────────────────────────
        f.write("\n## 6. Qualitative Analysis — Retrieval Usefulness\n\n")
        f.write("### 6a. Examples Where Retrieved Cases Appear USEFUL\n\n")
        useful = [p for p in predictions
                  if p['intent_correct'] and p['top1_similarity'] > 0.75
                  and not p['failed']][:5]
        for i, p in enumerate(useful, 1):
            f.write(f"**Example {i}** (id={p['example_id']}, similarity={p['top1_similarity']:.4f})\n\n")
            f.write(f"- Customer: `{p['customer_message'][:120]}...`\n")
            f.write(f"- Human intent: `{p['human_intent']}` → Predicted: `{p['predicted_intent']}` ✅\n")
            f.write(f"- Top retrieved case: `{p['top1_customer_text'][:120]}...`\n")
            f.write(f"- AmazonHelp replied: `{p['top1_amazon_reply'][:100]}...`\n\n")

        f.write("### 6b. Examples Where Retrieval Appears IRRELEVANT or MISLEADING\n\n")
        misleading = [p for p in predictions
                      if not p['intent_correct'] and not p['failed']][:5]
        for i, p in enumerate(misleading, 1):
            f.write(f"**Example {i}** (id={p['example_id']}, similarity={p['top1_similarity']:.4f})\n\n")
            f.write(f"- Customer: `{p['customer_message'][:120]}...`\n")
            f.write(f"- Human intent: `{p['human_intent']}` → Predicted: `{p['predicted_intent']}` ❌\n")
            f.write(f"- Top retrieved case: `{p['top1_customer_text'][:120]}...`\n")
            f.write(f"- Analysis: Retrieved case may have biased the LLM toward `{p['predicted_intent']}`\n\n")

        # ─── Summary answers ──────────────────────────────────────────────────
        beats_v2 = m['intent_accuracy'] > 0.615
        f.write("\n## 7. Summary Answers\n\n")
        f.write(f"| Question | Answer |\n")
        f.write(f"| :--- | :--- |\n")
        f.write(f"| A. Provider/model | `{pname}` |\n")
        f.write(f"| B. Intent accuracy | **{m['intent_accuracy']*100:.2f}%** |\n")
        f.write(f"| C. Macro F1 | **{m['intent_macro_f1']:.4f}** |\n")
        f.write(f"| D. Escalation acc / F1 | **{m['escalation_accuracy']*100:.2f}% / {m['escalation_f1']:.4f}** |\n")
        f.write(f"| E. Combined accuracy | **{m['combined_accuracy']*100:.2f}%** |\n")
        f.write(f"| F. Beats Rule V2? | {'✅ YES' if beats_v2 else '❌ NO'} ({m['intent_accuracy']*100:.2f}% vs 61.50%) |\n")
        avg_sim = sum(retrieved_scores)/len(retrieved_scores) if retrieved_scores else 0
        f.write(f"| G. Retrieval useful? | {'Likely YES' if avg_sim > 0.70 else 'Marginal'} (avg similarity={avg_sim:.4f}) |\n")
        f.write(f"| H. Proceed to reply generation? | {'Yes — intent acc ≥ V2, retrieval grounding works' if beats_v2 else 'Consider prompt iteration first (V2 prompt)'} |\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate LLM+RAG on development set.")
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["gemini", "ollama", "mock", "openai", "anthropic"],
        help="LLM provider to use (e.g. --provider ollama or --provider gemini)"
    )
    args = parser.parse_args()
    run_evaluation(provider_name=args.provider)
