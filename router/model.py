"""
model.py — Logistic Regression Routing Model

Model type: Logistic Regression (sklearn)
Input: 8 extracted features from prompt text
Output: binary decision (fast/capable) + confidence score (0-1)

Why Logistic Regression:
- Trains in seconds on CPU
- Works well on small labelled datasets (50-100 examples)
- Naturally outputs probability = confidence score
- Fully explainable (each feature has a weight)
- Near-zero inference latency (<1ms per prompt)
"""

import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from router.features import extract_features, get_feature_names

MODEL_PATH = os.path.join(os.path.dirname(__file__), "routing_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

# Labels
FAST = "fast"
CAPABLE = "capable"

# Training data — 60 labelled examples
# Label 0 = fast (simple), Label 1 = capable (complex)
TRAINING_DATA = [
    # SIMPLE prompts (label=0, fast model)
    ("What is the capital of France?", 0),
    ("Who is Elon Musk?", 0),
    ("What does API stand for?", 0),
    ("When was Python created?", 0),
    ("What is 15 multiplied by 7?", 0),
    ("Define machine learning.", 0),
    ("What is the speed of light?", 0),
    ("Who wrote Harry Potter?", 0),
    ("What is the largest planet?", 0),
    ("Translate hello to Spanish.", 0),
    ("What year did World War 2 end?", 0),
    ("What is the boiling point of water?", 0),
    ("How many days in a leap year?", 0),
    ("What is the currency of Japan?", 0),
    ("Who invented the telephone?", 0),
    ("What is HTML?", 0),
    ("Define recursion in one sentence.", 0),
    ("What is the tallest mountain?", 0),
    ("How many continents are there?", 0),
    ("What is RAM?", 0),
    ("Who is the president of USA?", 0),
    ("What is photosynthesis?", 0),
    ("Convert 100 celsius to fahrenheit.", 0),
    ("What does CPU stand for?", 0),
    ("What is the square root of 144?", 0),
    ("When was the Eiffel Tower built?", 0),
    ("What is JSON?", 0),
    ("Define HTTP.", 0),
    ("What is the population of India?", 0),
    ("Who painted the Mona Lisa?", 0),

    # COMPLEX prompts (label=1, capable model)
    ("Write a Python function to implement binary search with error handling.", 1),
    ("Explain the difference between supervised and unsupervised learning with examples.", 1),
    ("Design a REST API architecture for a multi-tenant SaaS application.", 1),
    ("Analyze the trade-offs between microservices and monolithic architecture.", 1),
    ("Debug this code: def fib(n): return fib(n-1) + fib(n-2)", 1),
    ("Write a comprehensive guide on implementing JWT authentication in FastAPI.", 1),
    ("Compare gradient descent variants: SGD, Adam, RMSprop — when to use each.", 1),
    ("Implement a thread-safe LRU cache in Python with O(1) operations.", 1),
    ("Explain how transformer attention mechanisms work step by step.", 1),
    ("Design a database schema for an e-commerce platform with inventory management.", 1),
    ("Write a regex pattern to validate email addresses and explain each part.", 1),
    ("Analyze why my neural network is overfitting and suggest 5 solutions.", 1),
    ("Implement quicksort and explain its average vs worst case complexity.", 1),
    ("How would you architect a real-time chat system for 1 million users?", 1),
    ("Write unit tests for a payment processing module with edge cases.", 1),
    ("Explain the CAP theorem and how it applies to distributed database design.", 1),
    ("Create a Python class implementing the observer design pattern.", 1),
    ("Compare SQL vs NoSQL databases for a high-write social media application.", 1),
    ("Implement a rate limiter using the token bucket algorithm in Python.", 1),
    ("Explain how garbage collection works in Python vs Java — pros and cons.", 1),
    ("Write a comprehensive data pipeline for ETL with error handling and logging.", 1),
    ("Analyze the security vulnerabilities in this authentication flow and fix them.", 1),
    ("Design a recommendation system for an e-commerce platform.", 1),
    ("Implement a binary tree with insert, delete, and traversal methods.", 1),
    ("Explain SOLID principles with Python code examples for each.", 1),
    ("How do I optimize a slow SQL query with multiple joins and subqueries?", 1),
    ("Write a multi-threaded web scraper with rate limiting and retry logic.", 1),
    ("Explain the difference between process and thread with code examples.", 1),
    ("Design a CI/CD pipeline for a microservices application.", 1),
    ("Implement a distributed lock mechanism using Redis.", 1),
]


class RoutingModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self._load_or_train()

    def _load_or_train(self):
        """Load existing model or train a new one."""
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            with open(SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            self._train()

    def _train(self):
        """Train logistic regression on labelled examples."""
        print("Training routing model...")

        X = np.array([extract_features(prompt) for prompt, _ in TRAINING_DATA])
        y = np.array([label for _, label in TRAINING_DATA])

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train logistic regression
        self.model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_scaled, y)

        # Save model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)

        print(f"Routing model trained on {len(TRAINING_DATA)} examples.")
        print(f"Feature weights: {dict(zip(get_feature_names(), self.model.coef_[0].round(3)))}")

    def predict(self, prompt: str) -> dict:
        """
        Predict routing decision for a prompt.
        Returns: {
            decision: "fast" or "capable",
            confidence: float (0-1),
            features: dict of extracted features,
            reasoning: human-readable explanation
        }
        """
        features = extract_features(prompt)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Get prediction and probability
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = float(probabilities[prediction])

        decision = CAPABLE if prediction == 1 else FAST

        # Build human-readable reasoning
        feature_dict = dict(zip(get_feature_names(), features))
        reasoning = self._build_reasoning(feature_dict, decision)

        return {
            "decision": decision,
            "confidence": round(confidence, 3),
            "features": {k: round(float(v), 3) for k, v in feature_dict.items()},
            "reasoning": reasoning
        }

    def _build_reasoning(self, features: dict, decision: str) -> str:
        """Build a human-readable routing reason."""
        reasons = []

        if features["has_code_keywords"] > 0.5:
            reasons.append("contains code/programming task")
        if features["has_reasoning_keywords"] > 0.5:
            reasons.append("requires reasoning/analysis")
        if features["has_math_keywords"] > 0.5:
            reasons.append("involves math/logic")
        if features["token_count"] > 0.3:
            reasons.append("long/detailed prompt")
        if features["has_simple_keywords"] > 0.5:
            reasons.append("simple factual query")
        if features["question_complexity"] < 0.3:
            reasons.append("simple question word (what/who/when)")
        if features["question_complexity"] > 0.7:
            reasons.append("complex question word (why/how)")

        if not reasons:
            reasons.append("general complexity assessment")

        model_label = "Capable model" if decision == CAPABLE else "Fast model"
        return f"Routed to {model_label}: {', '.join(reasons)}"


# Singleton instance
_routing_model = None


def get_routing_model() -> RoutingModel:
    global _routing_model
    if _routing_model is None:
        _routing_model = RoutingModel()
    return _routing_model