"""
capable.py — Capable Model wrapper (Meta Llama 3.3 70B via OpenRouter)

Switched from Gemini — free tier quota is zero in India (region restriction).
OpenRouter provides free global access to Llama 3.3 70B.
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.env"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
CAPABLE_MODEL = "google/gemma-3-4b-it:free"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_capable_model(prompt: str, conversation_history: list = None) -> dict:
    if not OPENROUTER_API_KEY:
        return {
            "answer": "Error: OPENROUTER_API_KEY not set.",
            "tokens_used": 0,
            "latency_ms": 0,
            "model": CAPABLE_MODEL,
            "error": "missing_api_key"
        }

    messages = []
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "RouteWise Gateway"
    }

    payload = {
        "model": CAPABLE_MODEL,
        "messages": messages,
        "max_tokens": 2048
    }

    for attempt in range(3):
        start_time = time.time()
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            latency_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                data = response.json()
                answer = data["choices"][0]["message"]["content"]
                tokens = data.get("usage", {}).get("total_tokens", 0)
                return {
                    "answer": answer,
                    "tokens_used": tokens,
                    "latency_ms": latency_ms,
                    "model": CAPABLE_MODEL,
                    "error": None
                }
            elif response.status_code in [429, 503, 502]:
                time.sleep(3)
                continue
            else:
                return {
                    "answer": f"OpenRouter error: {response.status_code} — {response.text[:200]}",
                    "tokens_used": 0,
                    "latency_ms": latency_ms,
                    "model": CAPABLE_MODEL,
                    "error": f"api_error_{response.status_code}"
                }
        except Exception as e:
            if attempt == 2:
                return {
                    "answer": f"Capable model error: {str(e)}",
                    "tokens_used": 0,
                    "latency_ms": 0,
                    "model": CAPABLE_MODEL,
                    "error": str(e)
                }
            time.sleep(2)

    return {
        "answer": "Capable model failed after 3 retries. Try again in a moment.",
        "tokens_used": 0,
        "latency_ms": 0,
        "model": CAPABLE_MODEL,
        "error": "max_retries_exceeded"
    }