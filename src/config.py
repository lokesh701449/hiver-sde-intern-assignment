"""
config.py
----------
Centralised configuration for the hiver-sde-assignment-llm project.
All paths are resolved relative to the project root or via absolute references
so the original TWCS dataset is NOT copied into this project.
"""
import os

# ─── Original dataset (NOT copied — referenced in-place) ──────────────────────
TWCS_CSV_PATH = os.environ.get(
    "TWCS_CSV_PATH",
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "hiver-sde-assignment",
        "data", "raw", "archive", "twcs", "twcs.csv"
    ),
)

# ─── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

# ─── Index artefacts ──────────────────────────────────────────────────────────
FAISS_INDEX_PATH    = os.path.join(MODELS_DIR, "amazonhelp_faiss.index")
METADATA_PATH       = os.path.join(MODELS_DIR, "amazonhelp_metadata.json")

# ─── Embedding model ──────────────────────────────────────────────────────────
# all-MiniLM-L6-v2: 384-dim, ~80 MB, fast CPU inference, excellent semantic quality
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
EMBEDDING_DIM = 384

# ─── Retrieval ────────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 5

# ─── Intent taxonomy ──────────────────────────────────────────────────────────
ALL_INTENTS = [
    "delivery_shipping_delay",
    "returns_refund_inquiry",
    "item_condition_issue",
    "device_hardware_support",
    "prime_membership_billing",
    "digital_streaming_services",
    "account_access_security",
    "payment_promo_giftcard",
    "order_modification_cancellation",
    "marketplace_third_party_seller",
    "unclear_other",
]

# ─── LLM provider ─────────────────────────────────────────────────────────────
# Set OPENAI_API_KEY / ANTHROPIC_API_KEY in environment for real LLM calls.
# Without any key the system defaults to MockLLMProvider.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "mock")   # "openai" | "anthropic" | "gemini" | "ollama" | "mock"
OPENAI_MODEL      = os.environ.get("OPENAI_MODEL",      "gpt-4o-mini")
ANTHROPIC_MODEL   = os.environ.get("ANTHROPIC_MODEL",   "claude-3-haiku-20240307")
GEMINI_MODEL      = os.environ.get("GEMINI_MODEL",      "gemini-2.0-flash")
OLLAMA_MODEL      = os.environ.get("OLLAMA_MODEL",      "qwen2.5:3b")
OLLAMA_BASE_URL   = os.environ.get("OLLAMA_BASE_URL",   "http://localhost:11434")
LLM_TEMPERATURE   = float(os.environ.get("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS    = int(os.environ.get("LLM_MAX_TOKENS", "512"))
