"""
log.py — Request Logger

Logs every gateway request with:
- timestamp, prompt snippet, model used, routing reason,
  confidence score, latency, cache hit/miss, tokens used
"""

import json
import os
import time
from datetime import datetime
from typing import Optional

LOG_FILE = os.path.join(os.path.dirname(__file__), "../logs/requests.jsonl")


def ensure_log_dir():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


def log_request(
    prompt: str,
    answer: str,
    model_used: str,
    routing_decision: str,
    routing_reason: str,
    confidence: float,
    latency_ms: int,
    cache_hit: bool,
    similarity_score: float,
    tokens_used: int,
    answer_id: Optional[str] = None,
    error: Optional[str] = None
) -> dict:
    """Log a single gateway request and return the log entry."""
    ensure_log_dir()

    entry = {
        "id": str(int(time.time() * 1000)),
        "timestamp": datetime.utcnow().isoformat(),
        "prompt_snippet": prompt[:80] + "..." if len(prompt) > 80 else prompt,
        "prompt_length": len(prompt),
        "model_used": model_used,
        "routing_decision": routing_decision,
        "routing_reason": routing_reason,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "cache_hit": cache_hit,
        "similarity_score": similarity_score,
        "tokens_used": tokens_used,
        "answer_id": answer_id,
        "answer_length": len(answer),
        "error": error
    }

    # Append to JSONL file
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"Warning: Could not write log: {e}")

    return entry


def get_all_logs() -> list:
    """Read all logs from file."""
    ensure_log_dir()
    logs = []
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        logs.append(json.loads(line))
    except Exception as e:
        print(f"Warning: Could not read logs: {e}")
    return logs


def clear_logs():
    """Clear all logs."""
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)