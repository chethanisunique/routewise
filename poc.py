"""
poc.py — Standalone Routing Model Evaluator

Judges run this directly. No server needed.

Usage:
    python poc.py                          # uses default test_suite.json
    python poc.py --input test_suite.json  # specify input file
    python poc.py --threshold 0.5          # custom confidence threshold

Output:
    Per-prompt: routing decision, confidence, correct/incorrect
    Summary: accuracy, false positive rate, false negative rate
"""

import json
import sys
import time
import argparse
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from router.model import get_routing_model
from router.features import extract_features, get_feature_names


def run_poc(input_file: str, confidence_threshold: float = 0.5, verbose: bool = True):
    """
    Run routing model on test suite and report results.
    
    Args:
        input_file: Path to JSON/CSV file with prompts and ground_truth labels
        confidence_threshold: Minimum confidence to trust routing decision
        verbose: Print per-prompt results
    
    Returns:
        dict with full results
    """
    # Load test suite
    if not os.path.exists(input_file):
        print(f"Error: Test file not found: {input_file}")
        sys.exit(1)

    with open(input_file, "r") as f:
        test_cases = json.load(f)

    print("=" * 70)
    print("  RouteWise — Routing Model PoC Evaluator")
    print("=" * 70)
    print(f"  Test suite: {input_file}")
    print(f"  Total prompts: {len(test_cases)}")
    print(f"  Confidence threshold: {confidence_threshold}")
    print("=" * 70)

    # Load routing model
    print("\nLoading routing model...")
    routing_model = get_routing_model()
    print("Routing model ready.\n")

    # Run evaluation
    results = []
    correct = 0
    false_positives = 0   # Simple sent to Capable (wasteful)
    false_negatives = 0   # Complex sent to Fast (quality loss)

    start_time = time.time()

    if verbose:
        print(f"{'#':<4} {'PROMPT':<45} {'GT':<10} {'DECISION':<10} {'CONF':<7} {'RESULT'}")
        print("-" * 90)

    for case in test_cases:
        prompt = case["prompt"]
        ground_truth = case["ground_truth"]  # "simple" or "complex"

        # Get routing prediction
        pred = routing_model.predict(prompt)
        decision = pred["decision"]          # "fast" or "capable"
        confidence = pred["confidence"]
        reasoning = pred["reasoning"]

        # Map ground truth to model labels
        # simple → fast, complex → capable
        expected_decision = "fast" if ground_truth == "simple" else "capable"
        is_correct = decision == expected_decision

        if is_correct:
            correct += 1
            result_label = "✅ CORRECT"
        else:
            if ground_truth == "simple" and decision == "capable":
                false_positives += 1  # Simple → Capable (wasteful)
                result_label = "⚠️  FP (simple→capable)"
            else:
                false_negatives += 1  # Complex → Fast (quality loss)
                result_label = "❌ FN (complex→fast)"

        result_entry = {
            "id": case.get("id", "?"),
            "prompt": prompt,
            "prompt_snippet": prompt[:45] + "..." if len(prompt) > 45 else prompt,
            "ground_truth": ground_truth,
            "expected_decision": expected_decision,
            "routing_decision": decision,
            "confidence": confidence,
            "reasoning": reasoning,
            "is_correct": is_correct,
            "error_type": None if is_correct else ("false_positive" if ground_truth == "simple" else "false_negative"),
            "category": case.get("category", "unknown"),
            "notes": case.get("notes", "")
        }
        results.append(result_entry)

        if verbose:
            prompt_display = (prompt[:43] + "..") if len(prompt) > 45 else prompt
            print(f"{case.get('id', '?'):<4} {prompt_display:<45} {ground_truth:<10} {decision:<10} {confidence:<7.3f} {result_label}")

    total_time = time.time() - start_time

    # ── Summary Statistics ─────────────────────────────────────────────────────
    total = len(test_cases)
    accuracy = correct / total * 100
    fp_rate = false_positives / total * 100
    fn_rate = false_negatives / total * 100

    simple_count = sum(1 for c in test_cases if c["ground_truth"] == "simple")
    complex_count = sum(1 for c in test_cases if c["ground_truth"] == "complex")

    print("\n" + "=" * 70)
    print("  SUMMARY RESULTS")
    print("=" * 70)
    print(f"  Total prompts evaluated:    {total}")
    print(f"  Simple prompts:             {simple_count}")
    print(f"  Complex prompts:            {complex_count}")
    print(f"  Total evaluation time:      {total_time:.2f}s")
    print(f"  Avg time per prompt:        {total_time/total*1000:.1f}ms")
    print()
    print(f"  ✅ Correct routing:         {correct}/{total} ({accuracy:.1f}%)")
    print(f"  ⚠️  False positives:         {false_positives}/{total} ({fp_rate:.1f}%)")
    print(f"     (simple → capable, wasteful)")
    print(f"  ❌ False negatives:          {false_negatives}/{total} ({fn_rate:.1f}%)")
    print(f"     (complex → fast, quality loss)")
    print()

    # Success bar check
    target_accuracy = 75.0
    print(f"  Target accuracy (>75%):     {'✅ PASSED' if accuracy >= target_accuracy else '❌ FAILED'} ({accuracy:.1f}%)")
    print("=" * 70)

    # ── Feature Weight Analysis ────────────────────────────────────────────────
    print("\n  ROUTING MODEL FEATURE WEIGHTS")
    print("-" * 70)
    model = routing_model.model
    scaler = routing_model.scaler
    feature_names = get_feature_names()

    if hasattr(model, 'coef_'):
        weights = model.coef_[0]
        feature_importance = sorted(
            zip(feature_names, weights),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        for fname, weight in feature_importance:
            direction = "→ Capable" if weight > 0 else "→ Fast   "
            bar = "█" * int(abs(weight) * 5)
            print(f"  {fname:<30} {direction}  {weight:+.3f}  {bar}")

    # ── Failure Cases ──────────────────────────────────────────────────────────
    failures = [r for r in results if not r["is_correct"]]
    if failures:
        print(f"\n  FAILURE CASES ({len(failures)} mis-routes)")
        print("-" * 70)
        for i, f in enumerate(failures, 1):
            print(f"\n  Failure #{i} — {f['error_type'].upper()}")
            print(f"  Prompt:    {f['prompt']}")
            print(f"  Expected:  {f['expected_decision']} (ground truth: {f['ground_truth']})")
            print(f"  Got:       {f['routing_decision']} (confidence: {f['confidence']:.3f})")
            print(f"  Reason:    {f['reasoning']}")
            print(f"  Category:  {f['category']}")
            print(f"  Notes:     {f['notes']}")

    # ── Save Results ───────────────────────────────────────────────────────────
    output_file = "poc_results.json"
    output = {
        "summary": {
            "total": total,
            "correct": correct,
            "accuracy_pct": round(accuracy, 2),
            "false_positives": false_positives,
            "false_positive_rate_pct": round(fp_rate, 2),
            "false_negatives": false_negatives,
            "false_negative_rate_pct": round(fn_rate, 2),
            "total_time_seconds": round(total_time, 3),
            "avg_latency_ms": round(total_time / total * 1000, 1),
            "target_met": accuracy >= target_accuracy
        },
        "per_prompt_results": results
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {output_file}")
    print("=" * 70)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RouteWise Routing Model PoC Evaluator")
    parser.add_argument(
        "--input",
        default="test_suite.json",
        help="Path to test suite JSON file (default: test_suite.json)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for routing decision (default: 0.5)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-prompt output, show only summary"
    )

    args = parser.parse_args()
    run_poc(args.input, args.threshold, verbose=not args.quiet)