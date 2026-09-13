"""
test_prompt_smoke.py
---------------------
Smoke test for prompt-only escalation experiment.
Runs ONLY 10 development examples with qwen2.5:3b and reports:
- predicted intent
- predicted escalate
- escalation reason
- latency
- count of True vs False escalation predictions
"""
import os
import sys
import time
import csv

sys.path.insert(0, os.path.dirname(__file__))
from config import OLLAMA_MODEL, OLLAMA_BASE_URL
from retrieval import ConversationRetriever
from llm_agent import OllamaProvider, LLMAgent

DEV_SET_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'hiver-sde-assignment',
    'evaluation', 'golden_set_final.csv'
)

def main():
    print("=" * 70)
    print("      PROMPT EXPERIMENT SMOKE TEST (10 DEV EXAMPLES)")
    print("=" * 70)

    print("Loading retriever...")
    retriever = ConversationRetriever()
    provider = OllamaProvider()
    agent = LLMAgent(provider=provider)

    print(f"Provider: {provider.name}\n")

    # Load first 10 examples from dev set
    with open(DEV_SET_PATH, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))[:10]

    results = []
    esc_counts = {True: 0, False: 0}

    print(f"{'#':<3} | {'Example ID':<10} | {'Predicted Intent':<32} | {'Escalate':<8} | {'Latency (s)':<10} | {'Reason'}")
    print("-" * 110)

    for idx, row in enumerate(rows, 1):
        eid = row['example_id']
        msg = row['customer_message']
        ctx = row['conversation_context']

        cases = retriever.retrieve(msg, ctx, top_k=5)

        t0 = time.perf_counter()
        res = agent.classify(msg, ctx, cases)
        elapsed_s = time.perf_counter() - t0

        esc_counts[res.escalate] += 1

        results.append({
            'idx': idx,
            'example_id': eid,
            'intent': res.intent,
            'escalate': res.escalate,
            'reason': res.escalation_reason,
            'latency_s': elapsed_s
        })

        print(f"{idx:<3} | {eid:<10} | {res.intent:<32} | {str(res.escalate):<8} | {elapsed_s:<10.2f} | {res.escalation_reason}")

    print("=" * 110)
    print("SUMMARY SUMMARY REPORT FOR 10-EXAMPLE SMOKE TEST:")
    print(f"Total examples tested        : {len(results)}")
    print(f"Escalations predicted True   : {esc_counts[True]}")
    print(f"Escalations predicted False  : {esc_counts[False]}")
    print(f"Average latency per request  : {sum(r['latency_s'] for r in results)/len(results):.2f} seconds")
    print("=" * 110)

if __name__ == '__main__':
    main()
