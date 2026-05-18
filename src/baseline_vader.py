"""
baseline_vader.py
Runs VADER (rule-based sentiment from 2014) on the test set.
Establishes the "lazy approach" baseline.

Run with: python src/baseline_vader.py
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from data_utils import load_split
from evaluate import evaluate_predictions

print("=" * 60)
print("BASELINE 1: VADER (rule-based)")
print("=" * 60)

# -----------------------------------------------------------------
# Load test set
# -----------------------------------------------------------------
print("\n[1] Loading test set...")
test = load_split("test")
print(f"    Loaded {len(test)} test sentences")

# -----------------------------------------------------------------
# Initialize VADER
# -----------------------------------------------------------------
print("\n[2] Initializing VADER analyzer...")
analyzer = SentimentIntensityAnalyzer()
print("    ✅ VADER ready")

# -----------------------------------------------------------------
# Define how to map VADER's scores to our 3 classes
# -----------------------------------------------------------------
def vader_to_label(text: str) -> str:
    """
    VADER returns a 'compound' score from -1 (negative) to +1 (positive).
    We bin it into 3 classes using standard thresholds:
        compound >= 0.05  -> positive
        compound <= -0.05 -> negative
        otherwise         -> neutral
    """
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"

# -----------------------------------------------------------------
# Run VADER on every test sentence
# -----------------------------------------------------------------
print("\n[3] Running VADER on test sentences...")
y_true = test["label_name"].tolist()
y_pred = [vader_to_label(text) for text in test["text"]]
print(f"    ✅ Predictions complete ({len(y_pred)} sentences)")

# -----------------------------------------------------------------
# Evaluate
# -----------------------------------------------------------------
metrics = evaluate_predictions(y_true, y_pred, model_name="vader")

# -----------------------------------------------------------------
# Quick comparison with majority baseline
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("VADER vs. NAIVE BASELINES")
print("=" * 60)
majority_acc = sum(1 for y in y_true if y == "neutral") / len(y_true)
print(f"\nMajority-class baseline (always predict 'neutral'): {majority_acc*100:.2f}%")
print(f"VADER accuracy:                                      {metrics['accuracy']*100:.2f}%")

if metrics["accuracy"] > majority_acc:
    diff = (metrics["accuracy"] - majority_acc) * 100
    print(f"\n✅ VADER beats majority baseline by {diff:.2f} points")
else:
    diff = (majority_acc - metrics["accuracy"]) * 100
    print(f"\n❌ VADER underperforms majority baseline by {diff:.2f} points")
    print("   (This is expected — VADER's vocabulary doesn't match financial language)")

print("\n" + "=" * 60)
print("BASELINE 1 COMPLETE")
print("=" * 60)