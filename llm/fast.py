"""
fast.py — Fast Model wrapper (Groq Llama 3.1 8B)

Why Groq:
- Runs on custom LPUs (Language Processing Units)
- ~0.3-0.8s response time vs 2-5s for standard APIs
- Free tier, no credit card required
- Best raw speed available in free tier
"""

import os
import time
import requests
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def call_fast_model(prompt: str, conversation_history: list = None) -> dict:
    """
    Call Groq Llama 3.1 8B (Fast model).
    
    Args:
        prompt: User prompt
        conversation_history: Full conversation history for multi-turn support
    
    Returns:
        {answer, tokens_used, latency_ms, model, error}
    """
    if not GROQ_API_KEY:
        return {
            "answer": "Error: GROQ_API_KEY not set. Please add it to your .env file.",
            "tokens_used": 0,
            "latency_ms": 0,
            "model": GROQ_MODEL,
            "error": "missing_api_key"
        }

    # Build messages — include history for multi-turn support
    messages = []
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.7
    }

    start_time = time.time()
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        latency_ms = int((time.time() - start_time) * 1000)

        if response.status_code == 200:
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)
            return {
                "answer": answer,
                "tokens_used": tokens,
                "latency_ms": latency_ms,
                "model": GROQ_MODEL,
                "error": None
            }
        else:
            return {
                "answer": f"Groq API error: {response.status_code} — {response.text}",
                "tokens_used": 0,
                "latency_ms": latency_ms,
                "model": GROQ_MODEL,
                "error": f"api_error_{response.status_code}"
            }
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "answer": f"Fast model error: {str(e)}",
            "tokens_used": 0,
            "latency_ms": latency_ms,
            "model": GROQ_MODEL,
            "error": str(e)
        }