"""
build_index.py
--------------
Build a FAISS vector index of AmazonHelp historical conversations from TWCS.

Pipeline:
  1. Stream TWCS CSV (never loads all 2.8M tweets at once)
  2. Filter to AmazonHelp threads (English only)
  3. Reconstruct conversation-level context
     - customer messages (inbound)
     - AmazonHelp replies (outbound)
  4. Embed each conversation with sentence-transformers/all-MiniLM-L6-v2
  5. Build FAISS IndexFlatIP (cosine similarity via L2-normalised embeddings)
  6. Persist index + metadata JSON for reuse

Usage:
    cd hiver-sde-assignment-llm
    python3 src/build_index.py

Output:
    models/amazonhelp_faiss.index
    models/amazonhelp_metadata.json
"""

import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    TWCS_CSV_PATH, FAISS_INDEX_PATH, METADATA_PATH,
    EMBEDDING_MODEL_NAME, EMBEDDING_DIM, MODELS_DIR,
)

# ─── Constants ────────────────────────────────────────────────────────────────
NON_ASCII      = re.compile(r'[^\x00-\x7F]')
URL_PATTERN    = re.compile(r'https?://\S+')
MENTION_STRIP  = re.compile(r'@\w+\s*')
MIN_CUST_CHARS = 20          # skip trivially short customer messages
MAX_TURNS      = 6           # cap conversation length at 6 turns
MAX_CONVERSATIONS = 50_000   # cap index size for manageability


def clean_text(text: str) -> str:
    """Strip URLs and leading @mentions; normalise whitespace."""
    text = URL_PATTERN.sub("", text)
    text = re.sub(r'@\w+\s*', '', text, count=1)  # strip first mention only
    return text.strip()


def is_english(text: str) -> bool:
    return not NON_ASCII.search(text)


# ─── Step 1: Stream TWCS and index all tweets we care about ──────────────────

def load_amazon_tweets(twcs_path: str):
    """
    Single-pass load of TWCS rows relevant to AmazonHelp.
    Returns two dicts:
      tweet_map : tweet_id -> {author_id, inbound, text, in_response_to, response_tweet_id}
      amazon_outbound_ids : set of tweet_ids authored by AmazonHelp
    """
    print(f"Streaming TWCS from: {twcs_path}")
    if not os.path.exists(twcs_path):
        print(f"ERROR: TWCS dataset not found at {twcs_path}")
        print("Set TWCS_CSV_PATH environment variable to the correct path.")
        sys.exit(1)

    tweet_map: Dict[str, dict] = {}
    amazon_outbound_ids = set()

    t0 = time.time()
    with open(twcs_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            author = row["author_id"]
            tid    = row["tweet_id"]

            is_amazon_outbound = (author == "AmazonHelp")
            is_inbound_to_amazon = (
                row["inbound"] == "True"
                and "@AmazonHelp" in row["text"]
            )

            if is_amazon_outbound or is_inbound_to_amazon:
                tweet_map[tid] = {
                    "tweet_id":            tid,
                    "author_id":           author,
                    "inbound":             row["inbound"] == "True",
                    "text":                row["text"],
                    "in_response_to":      row["in_response_to_tweet_id"],
                    "response_tweet_id":   row["response_tweet_id"],
                }
                if is_amazon_outbound:
                    amazon_outbound_ids.add(tid)

            if (i + 1) % 500_000 == 0:
                elapsed = time.time() - t0
                print(f"  Processed {(i+1)//1000}K tweets in {elapsed:.0f}s | "
                      f"AmazonHelp-related: {len(tweet_map):,}")

    elapsed = time.time() - t0
    print(f"Done. Total tweets scanned: {i+1:,} in {elapsed:.0f}s")
    print(f"AmazonHelp-related tweets kept: {len(tweet_map):,}")
    print(f"AmazonHelp outbound tweets: {len(amazon_outbound_ids):,}")
    return tweet_map, amazon_outbound_ids


# ─── Step 2: Reconstruct conversations ───────────────────────────────────────

def build_conversations(tweet_map: Dict, amazon_outbound_ids: set) -> List[dict]:
    """
    Reconstruct conversation threads. Each conversation is anchored at an
    AmazonHelp response that is itself a reply to a customer inbound tweet.

    A conversation case contains:
      - conversation_id  : root customer tweet_id
      - customer_turns   : [clean customer messages]
      - amazon_turns     : [clean AmazonHelp replies]
      - customer_text    : concatenated customer messages (for embedding)
      - full_context     : formatted multi-turn string (for LLM prompt)
    """
    print("\nReconstructing conversation threads ...")

    # Build parent -> list of replies
    children: Dict[str, List[str]] = defaultdict(list)
    for tid, tweet in tweet_map.items():
        parent = tweet["in_response_to"]
        if parent:
            children[parent].append(tid)

    conversations = []
    visited_roots = set()

    # Start from each AmazonHelp outbound tweet, walk up to find root customer tweet
    for outbound_id in amazon_outbound_ids:
        outbound = tweet_map.get(outbound_id)
        if not outbound:
            continue
        # Only English
        if not is_english(outbound["text"]):
            continue

        # Walk up the chain to find the root inbound customer message
        chain = [outbound_id]
        current_id = outbound["in_response_to"]
        while current_id and current_id in tweet_map:
            chain.append(current_id)
            current_id = tweet_map[current_id]["in_response_to"]

        chain.reverse()  # oldest first
        root_id = chain[0]

        if root_id in visited_roots:
            continue
        visited_roots.add(root_id)

        # Build ordered turns
        customer_turns = []
        amazon_turns   = []
        full_turns     = []

        for tid in chain[:MAX_TURNS * 2]:
            tw = tweet_map.get(tid)
            if not tw:
                continue
            cleaned = clean_text(tw["text"])
            if not cleaned:
                continue
            if tw["author_id"] == "AmazonHelp":
                amazon_turns.append(cleaned)
                full_turns.append(f"AmazonHelp: {cleaned}")
            else:
                if is_english(cleaned):
                    customer_turns.append(cleaned)
                    full_turns.append(f"Customer: {cleaned}")

        if not customer_turns:
            continue
        customer_text = " ".join(customer_turns)
        if len(customer_text) < MIN_CUST_CHARS:
            continue

        conversations.append({
            "conversation_id":  root_id,
            "customer_turns":   customer_turns,
            "amazon_turns":     amazon_turns,
            "customer_text":    customer_text,
            "full_context":     "\n".join(full_turns),
            "has_resolution":   len(amazon_turns) > 0,
        })

        if len(conversations) >= MAX_CONVERSATIONS:
            break

    # Prefer resolved conversations (AmazonHelp responded)
    resolved   = [c for c in conversations if c["has_resolution"]]
    unresolved = [c for c in conversations if not c["has_resolution"]]
    conversations = resolved + unresolved   # resolved first

    print(f"Total conversation cases: {len(conversations):,}")
    print(f"  Resolved (AmazonHelp replied): {len(resolved):,}")
    print(f"  Unresolved (customer only):    {len(unresolved):,}")
    return conversations[:MAX_CONVERSATIONS]


# ─── Step 3: Embed + build FAISS index ───────────────────────────────────────

def embed_and_build_index(conversations: List[dict]):
    """Embed customer_text for each conversation, build FAISS index."""
    import faiss
    from sentence_transformers import SentenceTransformer

    print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [c["customer_text"] for c in conversations]
    print(f"Embedding {len(texts):,} conversations ...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalise for cosine similarity via IP
        convert_to_numpy=True,
    )
    elapsed = time.time() - t0
    print(f"Embedding done in {elapsed:.1f}s | shape: {embeddings.shape}")

    # Build FAISS flat inner-product index (cosine on normalised vecs)
    print("Building FAISS IndexFlatIP ...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings.astype(np.float32))
    print(f"FAISS index size: {index.ntotal:,} vectors")

    return index, embeddings


# ─── Step 4: Save ─────────────────────────────────────────────────────────────

def save_artifacts(index, conversations: List[dict]):
    import faiss
    os.makedirs(MODELS_DIR, exist_ok=True)

    faiss.write_index(index, FAISS_INDEX_PATH)
    idx_mb = os.path.getsize(FAISS_INDEX_PATH) / (1024 * 1024)
    print(f"FAISS index saved: {FAISS_INDEX_PATH}  ({idx_mb:.1f} MB)")

    # Strip embedding from metadata (stored in index)
    meta = [
        {k: v for k, v in c.items() if k not in ("customer_text",)}
        for c in conversations
    ]
    # Keep customer_text separately for retrieval display
    meta_with_text = conversations   # retain full dict

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_with_text, f, ensure_ascii=False, separators=(",", ":"))

    meta_mb = os.path.getsize(METADATA_PATH) / (1024 * 1024)
    print(f"Metadata saved:    {METADATA_PATH}  ({meta_mb:.1f} MB)")
    return idx_mb, meta_mb


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  BUILD AMAZONHELP RAG INDEX")
    print("  Embedding model: all-MiniLM-L6-v2 (384-dim, cosine similarity)")
    print("  Vector store:    FAISS IndexFlatIP (local, no external DB)")
    print("=" * 70)

    tweet_map, amazon_outbound_ids = load_amazon_tweets(TWCS_CSV_PATH)
    conversations = build_conversations(tweet_map, amazon_outbound_ids)
    index, _ = embed_and_build_index(conversations)
    idx_mb, meta_mb = save_artifacts(index, conversations)

    print("\n" + "=" * 70)
    print("  INDEX SUMMARY")
    print("=" * 70)
    print(f"  Conversations indexed  : {len(conversations):,}")
    print(f"  Resolved conversations : {sum(1 for c in conversations if c['has_resolution']):,}")
    print(f"  FAISS index size       : {idx_mb:.1f} MB")
    print(f"  Metadata size          : {meta_mb:.1f} MB")
    print(f"  Embedding model        : {EMBEDDING_MODEL_NAME}")
    print(f"  Embedding dimension    : {EMBEDDING_DIM}")
    print("\n  Next: python3 src/test_retrieval.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
