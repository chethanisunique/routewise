"""
Pointer-based answer store with LFU eviction.

CACHE STORE   — holds {vector, answer_id, hit_count}
ANSWER STORE  — holds {answer_id: full_answer_text}

Two different prompts can point to the same answer_id → zero duplication.
LFU eviction removes the least-frequently-used entry when cache is full.
"""

import hashlib
import time
from collections import defaultdict


MAX_CACHE_SIZE = 100  # max entries in cache store


class AnswerStore:
    """Stores full answer text, keyed by a short hash ID."""

    def __init__(self):
        self._store: dict[str, str] = {}

    def save(self, answer: str) -> str:
        """Save answer and return its ID."""
        answer_id = hashlib.md5(answer.encode()).hexdigest()[:6]
        self._store[answer_id] = answer
        return answer_id

    def get(self, answer_id: str) -> str | None:
        return self._store.get(answer_id)

    def delete(self, answer_id: str):
        self._store.pop(answer_id, None)

    def clear(self):
        self._store.clear()

    def size(self) -> int:
        return len(self._store)


class CacheStore:
    """
    Cache store: maps compressed vectors to answer IDs.
    Supports similarity lookup and LFU eviction.
    """

    def __init__(self, max_size: int = MAX_CACHE_SIZE):
        self.max_size = max_size
        self._entries: list[dict] = []
        # entry schema: {vector, answer_id, hit_count, last_hit}

    def add(self, vector: list, answer_id: str):
        if len(self._entries) >= self.max_size:
            self._evict_lfu()
        self._entries.append({
            "vector": vector,
            "answer_id": answer_id,
            "hit_count": 0,
            "last_hit": time.time(),
        })

    def find(self, vector: list, threshold: float = 0.85) -> str | None:
        """
        Find the best matching answer_id for a given vector.
        Returns answer_id if similarity >= threshold, else None.
        """
        best_score = -1.0
        best_entry = None

        for entry in self._entries:
            score = _cosine_similarity(vector, entry["vector"])
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= threshold:
            best_entry["hit_count"] += 1
            best_entry["last_hit"] = time.time()
            return best_entry["answer_id"]

        return None

    def _evict_lfu(self):
        """Remove the entry with the lowest hit_count."""
        if not self._entries:
            return
        lfu = min(self._entries, key=lambda e: e["hit_count"])
        self._entries.remove(lfu)

    def clear(self):
        self._entries.clear()

    def size(self) -> int:
        return len(self._entries)

    def stats(self) -> dict:
        if not self._entries:
            return {"entries": 0, "total_hits": 0, "avg_hit_count": 0}
        total_hits = sum(e["hit_count"] for e in self._entries)
        return {
            "entries": len(self._entries),
            "total_hits": total_hits,
            "avg_hit_count": round(total_hits / len(self._entries), 2),
        }


def _cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
