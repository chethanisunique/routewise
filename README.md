# RouteWise — Smart AI Gateway

> A smart AI gateway that routes every prompt to the right LLM and proves it made the right call.

---

## Models Used

| Label | Model | Handles | Why |
|---|---|---|---|
| **Fast model** | Groq Llama 3.1 8B | Simple queries, factual Q&A, definitions, short summaries | Fastest free-tier API (~400ms via Groq LPUs). No card required. |
| **Capable model** | Google Gemma 3 4B via OpenRouter | Code generation, reasoning, multi-step analysis, debugging | Strongest available free-tier model globally. Gemini 1.5 Flash has zero quota in India (confirmed: limit:0). No card required. |

Every request is tagged in logs and dashboard with which model handled it and why.

---

## Setup (5 commands)

```bash
git clone https://github.com/chethanisunique/routewise
cd routewise
pip install -r requirements.txt
cp .env.example .env
python main.py
```

---

## API Keys Required (both free, no card)

| Key | Where to get |
|---|---|
| `GROQ_API_KEY` | https://console.groq.com |
| `OPENROUTER_API_KEY` | https://openrouter.ai |

---

## Running the PoC (judges run this directly)

```bash
python poc.py
```

Runs standalone — no server needed. Completes in under 60 seconds on CPU.

**Results: 95% accuracy (19/20), 0 false negatives, 1 false positive**

---

## Running the Full System

```bash
# Terminal 1
python main.py

# Terminal 2
streamlit run dashboard/app.py
```

---

## Routing Model

**Type:** Logistic Regression (sklearn)

**Why Logistic Regression over Neural Network:**
- PDF requires CPU + under 60 second evaluation
- LR trains in under 1 second on 60 labelled examples
- Neural Network requires GPU and 10,000+ examples for same task
- Outputs natural probability = confidence score logged per request
- Fully explainable weights — each feature is auditable
- Under 1ms inference — routing adds zero noticeable latency

**8 Input Features:**

| Feature | Weight | Direction |
|---|---|---|
| token_count | +1.37 | Capable |
| has_reasoning_keywords | +1.28 | Capable |
| has_simple_keywords | −0.88 | Fast |
| avg_word_length | +0.69 | Capable |
| question_complexity | +0.66 | Capable |
| has_code_keywords | +0.35 | Capable |
| sentence_count | 0.00 | neutral |
| has_math_keywords | 0.00 | neutral |

**Decision logic:**
- P(capable) = sigmoid(w · features + b)
- P >= 0.50 → Route to Capable model
- P < 0.50 → Route to Fast model
- Confidence = max(P, 1−P)

**Training:** 60 labelled examples, StandardScaler normalization, balanced classes.

---

## Cache Layer

**Semantic Similarity Cache with Pointer-Based Answer Store**

- Prompts converted to 384-dim sentence embeddings
- Cache stores vector + answer_id pointer (not full answer)
- Answer store holds actual answers separately — zero duplication
- LFU eviction when cache reaches max size
- User-controlled clear via POST /cache/clear

**Threshold tuning (research finding):**
- 0.85 = too strict, missed paraphrases (real similarity was 0.7369)
- 0.70 = correct, catches paraphrases above this boundary (CHOSEN)
- 0.60 = too loose, risks serving wrong cached answers

**Compression bug found and fixed:** 32-dim index sampling destroyed semantic relationships. Restored to full 384-dim vectors.

---

## Research Questions Answered

**1. Did routing model work?**
95% accuracy (19/20). 0 false negatives. 1 false positive. Target >75% — PASSED.

**2. Cost difference?**
50% of requests went to fast model. Estimated 40% token cost reduction. Cache hits cost zero tokens. Target >25% — PASSED.

**3. Where did it fail?**
- "What year was Python programming language created?" — code keywords triggered capable route but task was a date lookup. Confidence 53.8%.
- "temp at which bubbles occur in water" — similarity 0.564 below threshold 0.70. Cache miss.
- Gemini 1.5 Flash — zero free-tier quota in India (limit:0 confirmed). Switched to Gemma 3 4B.

**4. Cache hit rate?**
25% hit rate (target >15% — PASSED). Threshold 0.70 caught 0.7369 similarity paraphrase correctly.

**5. What would you change?**
Replace logistic regression with DistilBERT on GPU. Add dynamic TTL by intent category. Add capable model fallback chain. Track real token costs per request.

---

## Performance Numbers

| Metric | Value | Target | Status |
|---|---|---|---|
| Routing accuracy | 95% | >75% | PASSED |
| False negatives | 0 | minimize | PASSED |
| Cache hit rate | 25% | >15% | PASSED |
| Cost reduction | ~40% | >25% | PASSED |
| Routing latency | 0.5ms | under LLM call | PASSED |
| Fast model latency | 406ms | — | measured |
| Cache hit latency | 25ms | — | measured |
| PoC eval time | 0.01s | <60s | PASSED |

---

## Project Structure

```
routewise/
├── main.py
├── poc.py
├── test_suite.json
├── router/
│   ├── model.py
│   └── features.py
├── cache/
│   └── semantic.py
├── llm/
│   ├── fast.py
│   └── capable.py
├── logger/
│   └── log.py
├── dashboard/
│   └── app.py
├── .env.example
└── requirements.txt
```
