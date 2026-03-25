"""
main.py — RouteWise AI Gateway Server

Single POST /chat endpoint that:
1. Checks semantic cache (compressed vector similarity)
2. If miss: runs routing model (logistic regression)
3. Routes to Fast (Groq Llama 3.1 8B) or Capable (Gemini 1.5 Flash)
4. Stores result in cache with pointer
5. Logs every decision
6. Returns answer + full metadata
"""

import time
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Import modules
from router.model import get_routing_model
from cache.semantic import get_cache
from llm.fast import call_fast_model
from llm.capable import call_capable_model
from logger.log import log_request, get_all_logs, clear_logs

app = FastAPI(
    title="RouteWise AI Gateway",
    description="Smart AI gateway that routes prompts to the right LLM",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ── Request / Response Models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    prompt: str
    conversation_history: Optional[List[dict]] = None
    force_model: Optional[str] = None  # "fast" or "capable" — override routing


class ChatResponse(BaseModel):
    answer: str
    model_used: str
    routing_decision: str
    routing_reason: str
    confidence: float
    latency_ms: int
    cache_hit: bool
    similarity_score: float
    tokens_used: int
    answer_id: Optional[str]


class CacheStatsResponse(BaseModel):
    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate_pct: float
    cache_size: int
    answer_store_size: int
    threshold: float
    max_size: int


# ── Initialize models on startup ───────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    print("Initializing RouteWise Gateway...")
    get_routing_model()  # Train/load routing model
    get_cache()          # Initialize semantic cache
    print("RouteWise Gateway ready.")


# ── Main Chat Endpoint ─────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main gateway endpoint.
    Routes prompt to Fast or Capable model based on complexity.
    Checks cache before routing.
    """
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    prompt = request.prompt.strip()
    start_time = time.time()

    cache = get_cache()
    routing_model = get_routing_model()

    # ── STEP 1: Cache Check ────────────────────────────────────────────────────
    cache_result = cache.lookup(prompt)

    if cache_result["hit"]:
        total_latency = int((time.time() - start_time) * 1000)
        answer = cache_result["answer"]

        log_entry = log_request(
            prompt=prompt,
            answer=answer,
            model_used="cache",
            routing_decision="cache_hit",
            routing_reason=f"Semantic cache hit (similarity={cache_result['similarity']})",
            confidence=cache_result["similarity"],
            latency_ms=total_latency,
            cache_hit=True,
            similarity_score=cache_result["similarity"],
            tokens_used=0,
            answer_id=cache_result["answer_id"]
        )

        return ChatResponse(
            answer=answer,
            model_used="cache",
            routing_decision="cache_hit",
            routing_reason=f"Cache hit — similarity {cache_result['similarity']}",
            confidence=cache_result["similarity"],
            latency_ms=total_latency,
            cache_hit=True,
            similarity_score=cache_result["similarity"],
            tokens_used=0,
            answer_id=cache_result["answer_id"]
        )

    # ── STEP 2: Routing Decision ───────────────────────────────────────────────
    if request.force_model:
        routing_result = {
            "decision": request.force_model,
            "confidence": 1.0,
            "reasoning": f"Manually forced to {request.force_model} model"
        }
    else:
        routing_result = routing_model.predict(prompt)

    decision = routing_result["decision"]
    confidence = routing_result["confidence"]
    reasoning = routing_result["reasoning"]

    # ── STEP 3: LLM Call ──────────────────────────────────────────────────────
    if decision == "fast":
        llm_result = call_fast_model(prompt, request.conversation_history)
    else:
        llm_result = call_capable_model(prompt, request.conversation_history)

    answer = llm_result["answer"]
    tokens_used = llm_result["tokens_used"]
    model_used = llm_result["model"]

    # ── STEP 4: Store in Cache ────────────────────────────────────────────────
    answer_id = cache.store(prompt, answer)

    # ── STEP 5: Log Request ───────────────────────────────────────────────────
    total_latency = int((time.time() - start_time) * 1000)

    log_request(
        prompt=prompt,
        answer=answer,
        model_used=model_used,
        routing_decision=decision,
        routing_reason=reasoning,
        confidence=confidence,
        latency_ms=total_latency,
        cache_hit=False,
        similarity_score=cache_result["similarity"],
        tokens_used=tokens_used,
        answer_id=answer_id,
        error=llm_result.get("error")
    )

    return ChatResponse(
        answer=answer,
        model_used=model_used,
        routing_decision=decision,
        routing_reason=reasoning,
        confidence=confidence,
        latency_ms=total_latency,
        cache_hit=False,
        similarity_score=cache_result["similarity"],
        tokens_used=tokens_used,
        answer_id=answer_id
    )


# ── Cache Management Endpoints ─────────────────────────────────────────────────

@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    """Get current cache statistics."""
    return get_cache().get_stats()


@app.post("/cache/clear")
async def clear_cache():
    """User-controlled cache clear."""
    get_cache().clear()
    return {"message": "Cache cleared successfully"}


@app.post("/cache/threshold/{value}")
async def set_threshold(value: float):
    """Dynamically update similarity threshold (research variable)."""
    if not 0.0 <= value <= 1.0:
        raise HTTPException(status_code=400, detail="Threshold must be between 0.0 and 1.0")
    get_cache().set_threshold(value)
    return {"message": f"Cache threshold updated to {value}"}


# ── Log Endpoints ──────────────────────────────────────────────────────────────

@app.get("/logs")
async def get_logs():
    """Get all request logs."""
    return get_all_logs()


@app.post("/logs/clear")
async def clear_request_logs():
    """Clear all request logs."""
    clear_logs()
    return {"message": "Logs cleared"}


# ── Health Check ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    cache_stats = get_cache().get_stats()
    return {
        "status": "healthy",
        "models": {
            "fast": "Groq Llama 3.1 8B",
            "capable": "Google Gemma 3 4B (OpenRouter)"
                  },
        "cache": cache_stats
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)