"""
test_ollama.py
--------------
Small connectivity test for Ollama local LLM provider.
Makes ONE request and reports connectivity status, model, pass/fail, and latency.
"""
import os
import sys
import time
import requests

sys.path.insert(0, os.path.dirname(__file__))
from config import OLLAMA_MODEL, OLLAMA_BASE_URL
from llm_agent import OllamaProvider, LLMAgent

def test_ollama():
    base_url = OLLAMA_BASE_URL.rstrip('/')
    model_name = OLLAMA_MODEL

    print("=" * 60)
    print("           OLLAMA CONNECTIVITY TEST")
    print("=" * 60)

    # 1. Reachability check
    reachable = False
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        if resp.ok:
            reachable = True
            print(f"Ollama service reachable : YES ({base_url})")
        else:
            print(f"Ollama service reachable : NO (HTTP {resp.status_code})")
    except Exception as e:
        print(f"Ollama service reachable : NO ({e})")

    print(f"Ollama model configured   : {model_name}")

    if not reachable:
        print("Ollama connectivity: FAIL")
        print("\nReason: Local Ollama server is not running or unreachable at " + base_url)
        return False

    # 2. Single test request
    print("\nSending 1 test classification request to Ollama...")
    test_msg = "My package has not arrived yet and it's 3 days late. Where is my order?"
    test_ctx = "Customer asked for delivery status."

    t0 = time.time()
    try:
        provider = OllamaProvider()
        agent = LLMAgent(provider=provider)
        result = agent.classify(customer_message=test_msg, conversation_context=test_ctx)
        elapsed_ms = (time.time() - t0) * 1000

        print("\nClassification output:")
        result.display()

        print(f"\nApproximate response time : {elapsed_ms:.1f} ms ({elapsed_ms/1000:.2f} s)")

        # Verify output JSON schema
        if result.intent and result.confidence >= 0.0:
            print("\nOllama connectivity: PASS")
            return True
        else:
            print("\nOllama connectivity: FAIL")
            print("Reason: Invalid response structure returned by Ollama")
            return False

    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000
        print(f"Error during request     : {e}")
        print(f"Approximate response time : {elapsed_ms:.1f} ms")
        print("\nOllama connectivity: FAIL")
        return False

if __name__ == "__main__":
    success = test_ollama()
    sys.exit(0 if success else 1)
