"""
test_ollama_latency.py
-----------------------
Measure latency across 3 sequential Ollama classification requests
using the same prompt format and retrieval as the evaluation script.
"""
import os
import sys
import time
import csv
import requests

sys.path.insert(0, os.path.dirname(__file__))
from config import OLLAMA_MODEL, OLLAMA_BASE_URL
from retrieval import ConversationRetriever
from llm_agent import OllamaProvider, LLMAgent

DEV_SET_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'hiver-sde-assignment',
    'evaluation', 'golden_set_final.csv'
)

def main():
    print("=" * 60)
    print("      OLLAMA 3-REQUEST SEQUENTIAL LATENCY TEST")
    print("=" * 60)

    # Load retriever and agent
    print("Loading retriever...")
    retriever = ConversationRetriever()
    provider = OllamaProvider()
    agent = LLMAgent(provider=provider)

    print(f"Model: {provider.name}\n")

    # Load 3 examples from dev set
    with open(DEV_SET_PATH, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))[:3]

    latencies = []
    results = []

    for idx, row in enumerate(rows, 1):
        msg = row['customer_message']
        ctx = row['conversation_context']

        print(f"--- Running Request {idx}/3 ---")
        cases = retriever.retrieve(msg, ctx, top_k=5)

        t_start = time.perf_counter()
        res = agent.classify(msg, ctx, cases)
        t_elapsed = time.perf_counter() - t_start

        latencies.append(t_elapsed)
        results.append(res)

        print(f"Request {idx} latency: {t_elapsed:.2f} s")
        print(f"  Predicted Intent  : {res.intent}")
        print(f"  Escalate          : {res.escalate}")
        print()

    req1_s = latencies[0]
    req2_s = latencies[1]
    req3_s = latencies[2]
    warm_avg_s = (req2_s + req3_s) / 2.0
    est_200_min = (warm_avg_s * 200) / 60.0

    print("=" * 60)
    print("                LATENCY SUMMARY REPORT")
    print("=" * 60)
    print(f"Request 1: {req1_s:.2f} seconds")
    print(f"Request 2: {req2_s:.2f} seconds")
    print(f"Request 3: {req3_s:.2f} seconds")
    print(f"Average warm latency: {warm_avg_s:.2f} seconds")
    print(f"Estimated time for 200 examples: {est_200_min:.1f} minutes")
    print("=" * 60)

    if req2_s > 10.0 or req3_s > 10.0:
        print("\n[NOTE] Warm latency is > 10 seconds.")
        print("Checking already-installed Ollama models locally...")
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=5)
            if resp.ok:
                models = [m['name'] for m in resp.json().get('models', [])]
                print(f"Installed Ollama models found ({len(models)}): {models}")
            else:
                print("Could not query installed models.")
        except Exception as e:
            print(f"Error querying installed models: {e}")

if __name__ == '__main__':
    main()
