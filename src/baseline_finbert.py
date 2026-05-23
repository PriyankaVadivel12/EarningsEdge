"""
baseline_finbert.py
Runs FinBERT (ProsusAI/finbert) on the test set.
This is the "off-the-shelf specialist" baseline — what someone would
realistically deploy today for financial sentiment analysis.

Run with: python src/baseline_finbert.py

Note: First run downloads ~440MB. Subsequent runs use cached model.
"""
from transformers import pipeline
import torch

from data_utils import load_split
from evaluate import evaluate_predictions

print("=" * 60)
print("BASELINE 2: FinBERT (pre-trained on financial text)")
print("=" * 60)

# -----------------------------------------------------------------
# Detect device — FinBERT will run faster on Apple Silicon's MPS
# than CPU, but slower than CUDA. For this small test set, any is fine.
# -----------------------------------------------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"\n[1] Using device: {device}")

# -----------------------------------------------------------------
# Load FinBERT via the Hugging Face pipeline API
# -----------------------------------------------------------------
print("\n[2] Loading FinBERT model...")
print("    (First run: downloads ~440MB. Subsequent runs: instant.)")

classifier = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    device=device,
)
print("    ✅ FinBERT loaded")

# -----------------------------------------------------------------
# Load test set
# -----------------------------------------------------------------
print("\n[3] Loading test set...")
test = load_split("test")
print(f"    Loaded {len(test)} test sentences")

# -----------------------------------------------------------------
# Run FinBERT predictions
# -----------------------------------------------------------------
print("\n[4] Running FinBERT on test sentences (this takes ~30-60 seconds)...")

# FinBERT returns labels like 'positive', 'negative', 'neutral' — 
# already matches our label scheme. We just lowercase to be safe.
texts = test["text"].tolist()
raw_predictions = classifier(texts, truncation=True, max_length=512, batch_size=16)
y_pred = [pred["label"].lower() for pred in raw_predictions]
y_true = test["label_name"].tolist()

print(f"    ✅ Predictions complete ({len(y_pred)} sentences)")

# -----------------------------------------------------------------
# Sanity check: show a few example predictions vs. truth
# -----------------------------------------------------------------
print("\n[5] Sample predictions vs. ground truth:")
for i in range(5):
    text_preview = texts[i][:80] + ("..." if len(texts[i]) > 80 else "")
    correct = "✅" if y_pred[i] == y_true[i] else "❌"
    print(f"   {correct} True: {y_true[i]:>8} | Pred: {y_pred[i]:>8} | {text_preview}")

# -----------------------------------------------------------------
# Evaluate using our reusable framework
# -----------------------------------------------------------------
metrics = evaluate_predictions(y_true, y_pred, model_name="finbert")

# -----------------------------------------------------------------
# Comparison: FinBERT vs. previous baselines
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("BASELINE COMPARISON SO FAR")
print("=" * 60)

print(f"\n{'Model':<25} {'Accuracy':>10} {'F1 (wgt)':>10} {'F1 (mac)':>10}")
print("-" * 60)
print(f"{'Majority baseline':<25} {'61.36%':>10} {'N/A':>10} {'N/A':>10}")
print(f"{'VADER':<25} {'56.64%':>10} {'56.85%':>10} {'47.43%':>10}")
print(f"{'FinBERT':<25} {metrics['accuracy']*100:>9.2f}% {metrics['f1_weighted']*100:>9.2f}% {metrics['f1_macro']*100:>9.2f}%")

print("\n" + "=" * 60)
print("BASELINE 2 COMPLETE")
print("=" * 60)