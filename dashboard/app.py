"""
dashboard/app.py — RouteWise Log Viewer

Streamlit dashboard showing:
- Live request log table
- Cache statistics
- Threshold slider (research variable)
- Aggregate metrics
"""

import streamlit as st
import requests
import pandas as pd
import json
import time
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.70
# Gateway URL
GATEWAY_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RouteWise — AI Gateway Dashboard",
    page_icon="🔀",
    layout="wide"
)

st.title("🔀 RouteWise — AI Gateway Dashboard")
st.caption("Smart routing between Fast (Groq Llama 3.1 8B) and Capable (Gemini 1.5 Flash) models")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Controls")

# Cache threshold slider
threshold = st.sidebar.slider(
    "Cache Similarity Threshold",
    min_value=0.1,
    max_value=1.0,
    value=st.session_state.threshold,
    step=0.05,
    help="High = fewer cache hits but safer. Low = more hits but risk of wrong answers."
)
st.session_state.threshold = threshold

if st.sidebar.button("Update Threshold"):
    try:
        r = requests.post(f"{GATEWAY_URL}/cache/threshold/{threshold}")
        if r.status_code == 200:
            st.sidebar.success(f"Threshold updated to {threshold}")
        else:
            st.sidebar.error("Failed to update threshold")
    except:
        st.sidebar.error("Gateway not reachable")

st.sidebar.divider()

if st.sidebar.button("🗑️ Clear Cache"):
    try:
        r = requests.post(f"{GATEWAY_URL}/cache/clear")
        if r.status_code == 200:
            st.sidebar.success("Cache cleared!")
        else:
            st.sidebar.error("Failed to clear cache")
    except:
        st.sidebar.error("Gateway not reachable")

if st.sidebar.button("🗑️ Clear Logs"):
    try:
        r = requests.post(f"{GATEWAY_URL}/logs/clear")
        if r.status_code == 200:
            st.sidebar.success("Logs cleared!")
        else:
            st.sidebar.error("Failed to clear logs")
    except:
        st.sidebar.error("Gateway not reachable")

auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=False)

# ── Try Prompt ─────────────────────────────────────────────────────────────────
st.subheader("💬 Try a Prompt")
col1, col2 = st.columns([4, 1])
with col1:
    prompt_input = st.text_input("Enter a prompt:", placeholder="Ask anything...")
with col2:
    force_model = st.selectbox("Force model:", ["auto", "fast", "capable"])

if st.button("Send", type="primary"):
    if prompt_input:
        with st.spinner("Routing..."):
            try:
                payload = {"prompt": prompt_input}
                if force_model != "auto":
                    payload["force_model"] = force_model

                r = requests.post(f"{GATEWAY_URL}/chat", json=payload, timeout=30)
                if r.status_code == 200:
                    resp = r.json()
                    st.success("Response received!")
                    

                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("Model Used", resp["model_used"].split("-")[0].upper())
                    col_b.metric("Latency", f"{resp['latency_ms']}ms")
                    col_c.metric("Cache Hit", "✅ YES" if resp["cache_hit"] else "❌ NO")
                    col_d.metric("Confidence", f"{resp['confidence']:.0%}")

                    st.info(f"**Routing reason:** {resp['routing_reason']}")
                    st.text_area("Answer:", resp["answer"], height=150)
                else:
                    st.error(f"Error: {r.status_code} — {r.text}")
            except Exception as e:
                st.error(f"Could not reach gateway: {e}")
    else:
        st.warning("Please enter a prompt")

st.divider()

# ── Cache Statistics ───────────────────────────────────────────────────────────
st.subheader("📊 Cache Statistics")

try:
    r = requests.get(f"{GATEWAY_URL}/cache/stats", timeout=5)
    if r.status_code == 200:
        stats = r.json()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Requests", stats["total_requests"])
        c2.metric("Cache Hits", stats["cache_hits"])
        c3.metric("Cache Misses", stats["cache_misses"])
        c4.metric("Hit Rate", f"{stats['hit_rate_pct']}%")
        c5.metric("Cache Size", f"{stats['cache_size']}/{stats['max_size']}")

        # Hit rate progress bar
        hit_rate = stats["hit_rate_pct"] / 100
        st.progress(hit_rate, text=f"Cache Hit Rate: {stats['hit_rate_pct']}% (target: >15%)")
except:
    st.warning("⚠️ Gateway not reachable. Start server with: `python main.py`")

st.divider()

# ── Request Log Table ──────────────────────────────────────────────────────────
st.subheader("📋 Request Log")

try:
    r = requests.get(f"{GATEWAY_URL}/logs", timeout=5)
    if r.status_code == 200:
        logs = r.json()

        if not logs:
            st.info("No requests logged yet. Send a prompt above to get started.")
        else:
            # Build dataframe
            df_data = []
            for log in reversed(logs):  # Most recent first
                df_data.append({
                    "Timestamp": log["timestamp"][:19].replace("T", " "),
                    "Prompt": log["prompt_snippet"],
                    "Model": log["model_used"],
                    "Decision": log["routing_decision"],
                    "Reason": log["routing_reason"][:60] + "..." if len(log.get("routing_reason", "")) > 60 else log.get("routing_reason", ""),
                    "Confidence": f"{log['confidence']:.0%}",
                    "Latency (ms)": log["latency_ms"],
                    "Cache": "✅ HIT" if log["cache_hit"] else "❌ MISS",
                    "Similarity": f"{log['similarity_score']:.3f}",
                    "Tokens": log["tokens_used"]
                })

            df = pd.DataFrame(df_data)

            # Color code by model
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Prompt": st.column_config.TextColumn(width="large"),
                    "Reason": st.column_config.TextColumn(width="large"),
                    "Latency (ms)": st.column_config.NumberColumn(format="%d ms"),
                }
            )

            # Summary metrics
            total = len(logs)
            fast_count = sum(1 for l in logs if l["routing_decision"] == "fast")
            capable_count = sum(1 for l in logs if l["routing_decision"] == "capable")
            cache_hits = sum(1 for l in logs if l["cache_hit"])

            st.caption(
                f"Total: {total} requests | "
                f"Fast model: {fast_count} ({fast_count/total*100:.0f}%) | "
                f"Capable model: {capable_count} ({capable_count/total*100:.0f}%) | "
                f"Cache hits: {cache_hits} ({cache_hits/total*100:.0f}%)"
            )
    else:
        st.error(f"Could not fetch logs: {r.status_code}")
except Exception as e:
    st.warning(f"⚠️ Gateway not reachable: {e}")

# Auto-refresh
if auto_refresh:
    time.sleep(5)
    st.rerun()