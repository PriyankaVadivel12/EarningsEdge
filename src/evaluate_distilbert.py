"""
evaluate_distilbert.py
Evaluate the fine-tuned DistilBERT on the held-out test set.
This is the honest, never-seen-before evaluation.

Run with: python src/evaluate_distilbert.py
"""
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_utils import DATA_DIR, PROJECT_ROOT, LABEL_NAMES, ID_TO_LABEL, load_split
from evaluate import evaluate_predictions

# -----------------------------------------------------------------
# Config
# -----------------------------------------------------------------
MODEL_DIR = PROJECT_ROOT / "models" / "distilbert_finetuned" / "final"
TOKENIZED_DIR = DATA_DIR / "tokenized"
BATCH_SIZE = 32

print("=" * 60)
print("PHASE 3.3: EVALUATING FINE-TUNED DISTILBERT ON TEST SET")
print("=" * 60)

# -----------------------------------------------------------------
# Device selection
# -----------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"\nUsing device: {device}")

# -----------------------------------------------------------------
# Load the fine-tuned model
# -----------------------------------------------------------------
print(f"\n[1] Loading fine-tuned model from {MODEL_DIR.relative_to(PROJECT_ROOT)}...")
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model.to(device)
model.eval()  # set to evaluation mode (disables dropout etc)
print("    ✅ Model loaded")

# -----------------------------------------------------------------
# Load the test set
# -----------------------------------------------------------------
print(f"\n[2] Loading test set...")
tokenized = load_from_disk(str(TOKENIZED_DIR))
test_ds = tokenized["test"]
test_df = load_split("test")  # for getting original text + labels
print(f"    Loaded {len(test_ds)} test sentences")

# -----------------------------------------------------------------
# Run inference on the test set
# -----------------------------------------------------------------
print(f"\n[3] Running inference on test set (batch size {BATCH_SIZE})...")

all_predictions = []
all_confidences = []

# Convert to PyTorch tensors and run in batches
with torch.no_grad():
    for start in range(0, len(test_ds), BATCH_SIZE):
        batch = test_ds[start : start + BATCH_SIZE]
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # shape: [batch, 3]

        # Convert logits → probabilities → predicted class
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).cpu().numpy()
        confs = torch.max(probs, dim=-1).values.cpu().numpy()

        all_predictions.extend(preds.tolist())
        all_confidences.extend(confs.tolist())

# Convert integer predictions back to string labels
y_pred = [ID_TO_LABEL[p] for p in all_predictions]
y_true = test_df["label_name"].tolist()
print(f"    ✅ Predictions complete ({len(y_pred)} sentences)")

# -----------------------------------------------------------------
# Sanity check: show first 5 predictions
# -----------------------------------------------------------------
print("\n[4] Sample predictions vs. ground truth:")
for i in range(5):
    text = test_df.iloc[i]["text"]
    preview = text[:80] + ("..." if len(text) > 80 else "")
    correct = "✅" if y_pred[i] == y_true[i] else "❌"
    print(f"   {correct} True: {y_true[i]:>8} | Pred: {y_pred[i]:>8} (conf: {all_confidences[i]:.3f}) | {preview}")

# -----------------------------------------------------------------
# Evaluate
# -----------------------------------------------------------------
metrics = evaluate_predictions(y_true, y_pred, model_name="distilbert_finetuned")

# -----------------------------------------------------------------
# Final comparison: all FOUR models side-by-side
# -----------------------------------------------------------------
print("\n" + "=" * 70)
print("FINAL COMPARISON: ALL FOUR MODELS ON TEST SET")
print("=" * 70)

print(f"\n{'Model':<32} {'Accuracy':>10} {'F1 (wgt)':>10} {'F1 (mac)':>10}")
print("-" * 70)
print(f"{'Majority baseline':<32} {'61.36%':>10} {'N/A':>10} {'N/A':>10}")
print(f"{'VADER (rule-based)':<32} {'56.64%':>10} {'56.85%':>10} {'47.43%':>10}")
print(f"{'FinBERT (contam ⚠️)':<32} {'97.64%':>10} {'97.65%':>10} {'96.69%':>10}")
print(f"{'Llama 3.1 70B (zero-shot)':<32} {'97.64%':>10} {'97.65%':>10} {'97.49%':>10}")
print(f"{'DistilBERT (yours, fine-tuned)':<32} {metrics['accuracy']*100:>9.2f}% {metrics['f1_weighted']*100:>9.2f}% {metrics['f1_macro']*100:>9.2f}%")

print("\n" + "=" * 70)
print("PHASE 3.3 COMPLETE")
print("=" * 70)