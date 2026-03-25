"""
Training script for the logistic regression routing model.

Trains on a labelled dataset of prompts (simple=0, complex=1),
saves the model to router/model.pkl.

Run once before starting the gateway:
    python -m router.train
"""

import json
import pickle
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

from router.features import extract_features

# ---------------------------------------------------------------------------
# Embedded training data — 40 labelled prompts
# 0 = simple  →  Fast model
# 1 = complex →  Capable model
# ---------------------------------------------------------------------------
TRAINING_DATA = [
    # Simple prompts (label = 0)
    ("What is the capital of France?", 0),
    ("Who invented the telephone?", 0),
    ("What year did World War II end?", 0),
    ("Define photosynthesis.", 0),
    ("What is 15 multiplied by 8?", 0),
    ("Translate 'hello' to Spanish.", 0),
    ("What is the boiling point of water?", 0),
    ("Who is the current CEO of Tesla?", 0),
    ("What does API stand for?", 0),
    ("Name three planets in the solar system.", 0),
    ("What is the speed of light?", 0),
    ("What is Python?", 0),
    ("How many days are in a leap year?", 0),
    ("What is the currency of Japan?", 0),
    ("Who wrote Romeo and Juliet?", 0),
    ("What is HTML?", 0),
    ("List the primary colors.", 0),
    ("What is 2 to the power of 10?", 0),
    ("What is the largest ocean?", 0),
    ("Who painted the Mona Lisa?", 0),

    # Complex prompts (label = 1)
    ("Explain the difference between TCP and UDP, and when you would use each.", 1),
    ("Write a Python function that implements binary search and explain its time complexity.", 1),
    ("Compare microservices and monolithic architectures — pros, cons, and when to use each.", 1),
    ("Analyze the causes of the 2008 financial crisis and its long-term economic impact.", 1),
    ("Design a database schema for a multi-tenant SaaS application with role-based access.", 1),
    ("Explain how transformers work in modern NLP models, step by step.", 1),
    ("Debug this code and explain what is wrong: def fib(n): return fib(n-1) + fib(n-2)", 1),
    ("Write a detailed essay on the ethical implications of AI in hiring decisions.", 1),
    ("How would you architect a real-time chat system for 1 million concurrent users?", 1),
    ("Explain gradient descent and how it is used in training neural networks.", 1),
    ("Compare REST and GraphQL APIs. When should a team choose GraphQL over REST?", 1),
    ("Write a recursive algorithm to solve the Tower of Hanoi problem and analyze its complexity.", 1),
    ("Summarize the key arguments for and against universal basic income.", 1),
    ("Explain how public key cryptography works and why RSA is secure.", 1),
    ("Design a system to detect fraudulent transactions in real time at scale.", 1),
    ("What are the trade-offs between consistency and availability in distributed systems?", 1),
    ("Analyze this Python code for performance issues and recommend optimizations.", 1),
    ("Explain the CAP theorem and give a concrete example of how it affects database design.", 1),
    ("Write a step-by-step plan to migrate a legacy monolith to microservices safely.", 1),
    ("How does HTTPS work? Explain the TLS handshake in detail.", 1),
]


def train_and_save():
    print("Extracting features from training data...")
    X = np.array([extract_features(prompt) for prompt, _ in TRAINING_DATA])
    y = np.array([label for _, label in TRAINING_DATA])

    print(f"Training set: {len(y)} samples | Simple: {sum(y==0)} | Complex: {sum(y==1)}")

    # Cross-validation score before saving
    model_cv = LogisticRegression(max_iter=1000, random_state=42)
    cv_scores = cross_val_score(model_cv, X, y, cv=5, scoring="accuracy")
    print(f"Cross-validation accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

    # Train final model on full training set
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)

    train_preds = model.predict(X)
    print("\nTraining set classification report:")
    print(classification_report(y, train_preds, target_names=["Fast", "Capable"]))

    # Save model
    model_path = Path(__file__).parent / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

    return model


if __name__ == "__main__":
    train_and_save()
