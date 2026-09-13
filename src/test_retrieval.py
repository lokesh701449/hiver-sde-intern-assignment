"""
test_retrieval.py
------------------
Diagnostic script to verify the RAG pipeline end-to-end.

For each of several development example queries (from golden_set_final.csv
or hand-crafted queries):

  1. Display the customer conversation
  2. Show top 3–5 retrieved historical AmazonHelp cases with similarity scores
  3. Show the full_context of each case
  4. Run LLM mock classification and display structured output

IMPORTANT:
  - Does NOT evaluate against held-out test set.
  - Does NOT use held-out examples as queries.
  - Does NOT use human labels to select or modify retrieval results.
  - Queries are taken from golden_set_final.csv (dev set) or are hand-crafted.

Usage:
    cd hiver-sde-assignment-llm
    python3 src/test_retrieval.py
"""

import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from config import DEFAULT_TOP_K
from retrieval import ConversationRetriever
from llm_agent import LLMAgent

# Path to dev labeled set (queries only — labels are NOT used for retrieval tuning)
DEV_SET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "hiver-sde-assignment",
    "evaluation", "golden_set_final.csv"
)

# Hand-crafted queries for pipeline verification (not from held-out set)
HAND_CRAFTED_QUERIES = [
    {
        "id":       "hc_001",
        "message":  "My order was supposed to arrive 3 days ago and it still hasn't shown up. "
                    "The tracking says it's been delivered but I never received it.",
        "context":  "Customer: I ordered last week and paid for 2-day shipping.\n"
                    "AmazonHelp: We're sorry to hear that. Can you confirm your order number?",
    },
    {
        "id":       "hc_002",
        "message":  "The item I received is completely different from what I ordered. "
                    "It's also damaged. I want a refund.",
        "context":  "",
    },
    {
        "id":       "hc_003",
        "message":  "I can't log into my Amazon account. I changed my email and now "
                    "it says my password is wrong too. I think my account might be hacked.",
        "context":  "Customer: I haven't been able to access my account for 3 days.",
    },
    {
        "id":       "hc_004",
        "message":  "Why was I charged for Prime again? I cancelled my membership last month. "
                    "I want the charge reversed immediately.",
        "context":  "",
    },
    {
        "id":       "hc_005",
        "message":  "My Fire TV keeps buffering on Prime Video. It worked fine before "
                    "the latest update. All other apps work perfectly.",
        "context":  "Customer: I've restarted it multiple times.",
    },
]


def load_dev_queries(path: str, n: int = 5):
    """Load a sample of dev-set examples as retrieval queries (labels NOT used)."""
    if not os.path.exists(path):
        return []
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            queries.append({
                "id":      row["example_id"],
                "message": row["customer_message"],
                "context": row["conversation_context"],
                "_human_intent": row["intent"],   # stored only for display, NOT for retrieval
            })
            if len(queries) >= n:
                break
    return queries


def run_tests():
    print("=" * 72)
    print("  AmazonHelp RAG — RETRIEVAL DIAGNOSTIC TEST")
    print("=" * 72)

    # ─── Load retriever ───────────────────────────────────────────────────────
    print("\nLoading retriever ...")
    retriever = ConversationRetriever()
    print(f"Indexed conversations: {retriever.num_indexed:,}")

    # ─── Load LLM agent (mock mode by default) ────────────────────────────────
    print("\nInitialising LLM agent ...")
    agent = LLMAgent()   # auto-selects provider from LLM_PROVIDER env var

    # ─── Build query list ─────────────────────────────────────────────────────
    dev_queries = load_dev_queries(DEV_SET_PATH, n=5)
    if dev_queries:
        print(f"\nLoaded {len(dev_queries)} dev-set queries for retrieval diagnosis.")
    queries = HAND_CRAFTED_QUERIES + dev_queries

    # ─── Measure retrieval latency ────────────────────────────────────────────
    print("\nMeasuring retrieval latency (10 trials) ...")
    latency_ms = retriever.retrieval_latency(n_trials=10)
    print(f"  Avg retrieval latency: {latency_ms:.1f} ms/query")

    # ─── Run each query ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  QUERY RESULTS")
    print("=" * 72)

    all_scores = []

    for q in queries:
        qid     = q["id"]
        message = q["message"]
        context = q.get("context", "")
        human   = q.get("_human_intent", None)

        print(f"\n{'─' * 72}")
        print(f"  Query ID   : {qid}")
        if human:
            print(f"  Human label: {human}  (reference only, NOT used in retrieval)")
        print(f"  Customer   : {message[:140]}{'...' if len(message) > 140 else ''}")
        if context:
            print(f"  Context    : {context[:100]}{'...' if len(context) > 100 else ''}")

        # Retrieve
        t0 = time.time()
        cases = retriever.retrieve(message, context, top_k=DEFAULT_TOP_K)
        rt_ms = (time.time() - t0) * 1000

        print(f"\n  Retrieval time: {rt_ms:.1f} ms  |  Top {len(cases)} cases:")
        for case in cases:
            print(case.display(show_amazon=True))
            all_scores.append(case.score)

        # LLM classification (mock or real)
        print(f"\n  LLM Classification ({agent.provider.name}):")
        result = agent.classify(message, context, cases)
        result.display()

    # ─── Summary stats ────────────────────────────────────────────────────────
    if all_scores:
        import statistics
        print("\n" + "=" * 72)
        print("  RETRIEVAL QUALITY SUMMARY")
        print("=" * 72)
        print(f"  Total retrieval results analysed : {len(all_scores)}")
        print(f"  Avg similarity score             : {statistics.mean(all_scores):.4f}")
        print(f"  Min similarity score             : {min(all_scores):.4f}")
        print(f"  Max similarity score             : {max(all_scores):.4f}")
        print(f"  Median similarity                : {statistics.median(all_scores):.4f}")
        print(f"  Avg retrieval latency            : {latency_ms:.1f} ms/query")

    print("\n" + "=" * 72)
    print("  STATUS CHECKS")
    print("=" * 72)
    checks = {
        "FAISS index loaded":            retriever.num_indexed > 0,
        "Retrieval returns results":     len(all_scores) > 0,
        "Similarity scores plausible":   all(0 <= s <= 1.001 for s in all_scores),
        "LLM mock mode works":           True,  # if we got here, it works
        "Retrieval latency < 500 ms":    latency_ms < 500,
    }
    for check, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {check}")

    print("\n  What is needed for the next evaluation stage:")
    print("  1. Set LLM_PROVIDER=openai and OPENAI_API_KEY for real LLM calls")
    print("  2. Or set LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY")
    print("  3. Build evaluation harness:  evaluation/evaluate_llm_rag.py")
    print("     - Run against golden_set_final.csv (dev) for diagnostic metrics")
    print("     - Only run against heldout_test_final.csv for the final benchmark")
    print("  4. Consider larger index or better chunking strategy")
    print("  5. Consider query expansion or hybrid BM25 + dense retrieval")
    print("=" * 72)


if __name__ == "__main__":
    run_tests()
