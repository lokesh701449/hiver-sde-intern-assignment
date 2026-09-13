"""
retrieval.py
-------------
ConversationRetriever: loads a pre-built FAISS index and returns the
top-K most semantically similar historical AmazonHelp conversations
for a given query.

Usage:
    from retrieval import ConversationRetriever
    retriever = ConversationRetriever()
    cases = retriever.retrieve("my package hasn't arrived", top_k=5)
    for case in cases:
        print(case.score, case.customer_text, case.full_context)
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    FAISS_INDEX_PATH, METADATA_PATH,
    EMBEDDING_MODEL_NAME, EMBEDDING_DIM, DEFAULT_TOP_K,
)


# ─── Data class returned by retrieval ─────────────────────────────────────────

@dataclass
class ConversationCase:
    """A single retrieved historical AmazonHelp conversation case."""
    rank:              int
    score:             float           # cosine similarity [0, 1]
    conversation_id:   str
    customer_turns:    List[str]
    amazon_turns:      List[str]
    customer_text:     str             # concatenated customer messages
    full_context:      str             # formatted multi-turn transcript
    has_resolution:    bool            # AmazonHelp sent at least one reply

    def display(self, show_amazon: bool = True) -> str:
        lines = [
            f"  ─── Case #{self.rank}  (score={self.score:.4f}, id={self.conversation_id}) ───",
            f"  Customer: {self.customer_text[:200]}{'...' if len(self.customer_text) > 200 else ''}",
        ]
        if show_amazon and self.amazon_turns:
            reply = self.amazon_turns[0]
            lines.append(f"  AmazonHelp: {reply[:200]}{'...' if len(reply) > 200 else ''}")
        return "\n".join(lines)


# ─── Retriever ────────────────────────────────────────────────────────────────

class ConversationRetriever:
    """
    Loads the pre-built FAISS index once and exposes a fast retrieve() method.

    Parameters
    ----------
    index_path : str
        Path to the FAISS binary index file.
    metadata_path : str
        Path to the JSON metadata file (list of conversation dicts).
    model_name : str
        Sentence-transformer model name (must match what was used at index build time).
    """

    def __init__(
        self,
        index_path:    str = FAISS_INDEX_PATH,
        metadata_path: str = METADATA_PATH,
        model_name:    str = EMBEDDING_MODEL_NAME,
    ):
        self._index_path    = index_path
        self._metadata_path = metadata_path
        self._model_name    = model_name
        self._index         = None
        self._metadata:     List[dict] = []
        self._model         = None
        self._load()

    def _load(self):
        import faiss
        from sentence_transformers import SentenceTransformer

        if not os.path.exists(self._index_path):
            raise FileNotFoundError(
                f"FAISS index not found: {self._index_path}\n"
                "Run:  python3 src/build_index.py  first."
            )
        if not os.path.exists(self._metadata_path):
            raise FileNotFoundError(
                f"Metadata not found: {self._metadata_path}\n"
                "Run:  python3 src/build_index.py  first."
            )

        t0 = time.time()
        self._index = faiss.read_index(self._index_path)
        with open(self._metadata_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)
        print(f"Index loaded: {self._index.ntotal:,} vectors in {time.time()-t0:.2f}s")

        print(f"Loading embedding model: {self._model_name} ...")
        self._model = SentenceTransformer(self._model_name)
        print("Retriever ready.")

    def retrieve(
        self,
        query_text: str,
        conversation_context: str = "",
        top_k: int = DEFAULT_TOP_K,
        exclude_ids: Optional[set] = None,
    ) -> List[ConversationCase]:
        """
        Retrieve the top-K most similar historical conversations.

        Parameters
        ----------
        query_text : str
            Customer message to search for.
        conversation_context : str
            Additional conversation context (multi-turn history).
        top_k : int
            Number of cases to return.
        exclude_ids : set, optional
            Conversation IDs to exclude (e.g., when query is from the dev set).

        Returns
        -------
        List[ConversationCase] sorted by descending similarity score.
        """
        # Combine query signals
        combined = query_text.strip()
        if conversation_context.strip():
            combined = f"{combined}\n{conversation_context.strip()}"

        # Embed and L2-normalise (index uses IP = cosine on unit vecs)
        vec = self._model.encode(
            [combined],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # Search (fetch more to allow filtering)
        fetch_k = min(top_k + 20, self._index.ntotal)
        scores, indices = self._index.search(vec, fetch_k)

        results: List[ConversationCase] = []
        rank = 1
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx]
            cid  = str(meta["conversation_id"])
            if exclude_ids and (cid in exclude_ids or meta["conversation_id"] in exclude_ids or cid in {str(x) for x in exclude_ids}):
                continue
            results.append(ConversationCase(
                rank=rank,
                score=float(score),
                conversation_id=cid,
                customer_turns=meta.get("customer_turns", []),
                amazon_turns=meta.get("amazon_turns", []),
                customer_text=meta.get("customer_text", ""),
                full_context=meta.get("full_context", ""),
                has_resolution=meta.get("has_resolution", False),
            ))
            rank += 1
            if len(results) >= top_k:
                break

        return results

    def retrieval_latency(self, n_trials: int = 10) -> float:
        """Return average retrieval latency in milliseconds over n_trials."""
        sample_queries = [
            "my order hasn't arrived",
            "I was charged twice for prime",
            "item arrived damaged",
            "can't sign in to my account",
            "want to return this product",
        ]
        times = []
        for i in range(n_trials):
            q = sample_queries[i % len(sample_queries)]
            t0 = time.time()
            self.retrieve(q, top_k=5)
            times.append((time.time() - t0) * 1000)
        return float(np.mean(times))

    @property
    def num_indexed(self) -> int:
        return self._index.ntotal if self._index else 0
