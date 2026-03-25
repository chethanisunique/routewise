"""
features.py — Extract prompt features for the routing model

Features used (each justified as a complexity signal):
1. token_count        — longer prompts tend to be more complex
2. sentence_count     — multi-sentence = multi-step reasoning
3. has_code_keywords  — code tasks always go to Capable
4. has_reasoning_kw   — words like "explain", "analyze", "compare"
5. has_simple_kw      — words like "what is", "define", "when"
6. avg_word_length    — longer words = more technical/complex
7. question_word      — "why/how" = complex, "what/when/who" = simple
8. has_math           — math symbols or numbers in context
"""

import re
import numpy as np


# Keywords that strongly indicate complexity
REASONING_KEYWORDS = [
    "explain", "analyze", "analyse", "compare", "contrast",
    "evaluate", "design", "implement", "debug", "optimize",
    "why does", "how does", "what would happen", "difference between",
    "pros and cons", "trade-off", "architecture", "algorithm",
    "write a", "build a", "create a", "generate", "summarize",
    "multi-step", "step by step", "in detail", "comprehensive"
]

# Keywords that strongly indicate simplicity
SIMPLE_KEYWORDS = [
    "what is", "what are", "who is", "when was", "where is",
    "define", "definition", "meaning of", "capital of",
    "how many", "how much", "yes or no", "true or false",
    "spell", "translate", "convert"
]

# Code-related keywords — always complex
CODE_KEYWORDS = [
    "write code", "write a function", "write a class", "write a script",
    "write a program", "debug this", "debug the", "fix this code",
    "function that", "class that", "implement a", "implement the",
    "refactor", "unit test", "test case", "sql query", "python script",
    "javascript", "algorithm", "recursion", "write a loop",
    "def ", "class ", "import ", "```python", "```js"
]

# Math/logic keywords
MATH_KEYWORDS = [
    "calculate", "compute", "integral", "derivative", "matrix",
    "probability", "statistics", "proof", "equation", "formula",
    "solve", "factorial", "fibonacci", "prime"
]


def extract_features(prompt: str) -> np.ndarray:
    """
    Extract 8 numerical features from a prompt.
    Returns a numpy array of shape (8,)
    """
    text = prompt.lower().strip()
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Feature 1: Normalized token count (log scale to reduce outlier effect)
    token_count = min(np.log1p(len(words)) / np.log1p(200), 1.0)

    # Feature 2: Sentence count (normalized)
    sentence_count = min(len(sentences) / 10.0, 1.0)

    # Feature 3: Has code keywords
    has_code = float(any(kw in text for kw in CODE_KEYWORDS))

    # Feature 4: Has reasoning keywords
    has_reasoning = float(any(kw in text for kw in REASONING_KEYWORDS))

    # Feature 5: Has simple keywords
    has_simple = float(any(kw in text for kw in SIMPLE_KEYWORDS))

    # Feature 6: Average word length (normalized)
    avg_word_len = np.mean([len(w) for w in words]) / 15.0 if words else 0.0
    avg_word_len = min(avg_word_len, 1.0)

    # Feature 7: Question word complexity
    # "why", "how" = complex (0.8), "what", "when", "who", "where" = simple (0.2)
    question_complexity = 0.5  # default neutral
    if any(text.startswith(w) for w in ["why", "how would", "how do", "how can"]):
        question_complexity = 0.8
    elif any(text.startswith(w) for w in ["what is", "what are", "who", "when", "where"]):
        question_complexity = 0.2

    # Feature 8: Has math/logic keywords
    has_math = float(any(kw in text for kw in MATH_KEYWORDS))

    features = np.array([
        token_count,
        sentence_count,
        has_code,
        has_reasoning,
        has_simple,
        avg_word_len,
        question_complexity,
        has_math
    ], dtype=np.float32)

    return features


def get_feature_names() -> list:
    return [
        "token_count",
        "sentence_count",
        "has_code_keywords",
        "has_reasoning_keywords",
        "has_simple_keywords",
        "avg_word_length",
        "question_complexity",
        "has_math_keywords"
    ]