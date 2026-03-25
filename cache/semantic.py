"""
semantic.py — Semantic Cache Layer

Architecture:
- Prompts converted to compressed 32-dim vectors (via PCA on sentence embeddings)
- Cache stores vector + answer_id pointer (NOT full answer)
- Answer store holds actual answers separately — zero duplication
- LFU (Least Frequently Used) eviction when cache is full
- User-controlled clear — no forced TTL
- Similarity threshold is a research variable (tunable)

Why pointers:
- Two similar prompts share one answer in memory
- Wrong cache hit returns a short answer, not a long wrong essay
- Reduces blast radius of low-threshold mismatches
"""

import os
import numpy as np
import uuid
import time
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", 0.70))
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", 1000))

# Compressed vector dimensions (trade-off: smaller = faster search, less precise)
COMPRESSED_DIM = 384


class SemanticCache:
    def __init__(self, threshold: float = CACHE_SIMILARITY_THRESHOLD, max_size: int = CACHE_MAX_SIZE):
        self.threshold = threshold
        self.max_size = max_size

        # Cache store: list of {vector, answer_id, hit_count, prompt_snippet}
        self.cache_store = []

        # Answer store: {answer_id: compressed_answer}
        self.answer_store = {}

        # Stats
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # PCA compressor (fitted lazily on first use)
        self._pca = None
        self._embedding_model = None

    def _get_embedding_model(self):
        """Lazy load sentence transformer."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Embedding model loaded.")
            except Exception as e:
                print(f"Warning: Could not load sentence transformer: {e}")
                self._embedding_model = "fallback"
        return self._embedding_model

    def _embed(self, text: str) -> np.ndarray:
        """Convert text to embedding vector."""
        model = self._get_embedding_model()
        if model == "fallback":
            # Fallback: simple TF-IDF-like hash embedding
            return self._hash_embed(text)
        embedding = model.encode(text, show_progress_bar=False)
        return embedding

    def _hash_embed(self, text: str) -> np.ndarray:
        """Simple fallback embedding using character hashing."""
        vec = np.zeros(384)
        for i, char in enumerate(text.lower()):
            vec[ord(char) % 384] += 1
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _compress(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            return 0.0
        return float(dot / norm)

    def _find_similar(self, vector: np.ndarray) -> Optional[dict]:
        """
        Find most similar cached entry above threshold.
        Returns cache entry or None.
        """
        best_score = -1.0
        best_entry = None

        for entry in self.cache_store:
            score = self._cosine_similarity(vector, entry["vector"])
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self.threshold:
            return best_entry, best_score
        return None, best_score

    def _evict_lfu(self):
        """
        LFU Eviction: remove the least frequently used entry.
        If tie, remove oldest entry.
        """
        if not self.cache_store:
            return

        # Find entry with lowest hit_count
        min_hits = min(e["hit_count"] for e in self.cache_store)
        lfu_entries = [e for e in self.cache_store if e["hit_count"] == min_hits]

        # Among LFU entries, remove oldest
        oldest = min(lfu_entries, key=lambda e: e["created_at"])
        self.cache_store.remove(oldest)

        # Check if answer_id is still referenced
        answer_id = oldest["answer_id"]
        still_referenced = any(e["answer_id"] == answer_id for e in self.cache_store)
        if not still_referenced:
            self.answer_store.pop(answer_id, None)

    def lookup(self, prompt: str) -> dict:
        """
        Look up prompt in cache.
        Returns: {hit: bool, answer: str|None, similarity: float, answer_id: str|None}
        """
        self.total_requests += 1

        vector = self._compress(self._embed(prompt))
        entry, similarity = self._find_similar(vector)

        if entry is not None:
            # Cache HIT
            entry["hit_count"] += 1
            self.cache_hits += 1
            answer = self.answer_store.get(entry["answer_id"], None)
            return {
                "hit": True,
                "answer": answer,
                "similarity": round(similarity, 4),
                "answer_id": entry["answer_id"]
            }

        # Cache MISS
        self.cache_misses += 1
        return {
            "hit": False,
            "answer": None,
            "similarity": round(similarity, 4),
            "answer_id": None
        }

    def store(self, prompt: str, answer: str) -> str:
        """
        Store prompt-answer pair in cache.
        Returns answer_id (pointer).
        """
        # Check if we need to evict
        if len(self.cache_store) >= self.max_size:
            self._evict_lfu()

        # Generate answer_id pointer
        answer_id = str(uuid.uuid4())[:8]

        # Store answer once in answer store
        self.answer_store[answer_id] = answer

        # Store compressed vector + pointer in cache
        vector = self._compress(self._embed(prompt))
        self.cache_store.append({
            "vector": vector,
            "answer_id": answer_id,
            "hit_count": 0,
            "prompt_snippet": prompt[:50],
            "created_at": time.time()
        })

        return answer_id

    def clear(self):
        """User-controlled cache clear — wipes everything."""
        self.cache_store.clear()
        self.answer_store.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
        print("Cache cleared by user.")

    def get_stats(self) -> dict:
        """Return cache statistics."""
        hit_rate = (self.cache_hits / self.total_requests * 100) if self.total_requests > 0 else 0
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_pct": round(hit_rate, 2),
            "cache_size": len(self.cache_store),
            "answer_store_size": len(self.answer_store),
            "threshold": self.threshold,
            "max_size": self.max_size
        }

    def set_threshold(self, threshold: float):
        """Dynamically update similarity threshold (research variable)."""
        self.threshold = max(0.0, min(1.0, threshold))


# Singleton instance
_cache = None


def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache