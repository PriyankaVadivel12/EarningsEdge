"""
distilbert_error_analysis.py
Find DistilBERT's errors and compare them with FinBERT and Llama errors.
This is the key analysis for the project's main finding.
"""
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_utils import DATA_DIR, PROJECT_ROOT, ID_TO_LABEL, load_split, RESULTS_DIR

MODEL_DIR = PROJECT_ROOT / "models" / "distilbert_finetuned" / "final"
TOKENIZED_DIR = DATA_DIR / "tokenized"

# -----------------------------------------------------------------
# Reproduce DistilBERT predictions
# -----------------------------------------------------------------
print("=" * 70)
print("DISTILBERT ERROR ANALYSIS + CROSS-MODEL COMPARISON")
print("=" * 70)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"\nUsing device: {device}\n")

print("Loading fine-tuned DistilBERT...")
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model.to(device)
model.eval()

tokenized = load_from_disk(str(TOKENIZED_DIR))
test_ds = tokenized["test"]
test_df = load_split("test").reset_index(drop=True)

print("Running inference...")
predictions = []
confidences = []

with torch.no_grad():
    for start in range(0, len(test_ds), 32):
        batch = test_ds[start : start + 32]
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).cpu().numpy()
        confs = torch.max(probs, dim=-1).values.cpu().numpy()
        predictions.extend(preds.tolist())
        confidences.extend(confs.tolist())

predicted_labels = [ID_TO_LABEL[p] for p in predictions]
true_labels = test_df["label_name"].tolist()
texts = test_df["text"].tolist()

# -----------------------------------------------------------------
# Find DistilBERT's errors
# -----------------------------------------------------------------
distilbert_errors = []
for i, (t, p) in enumerate(zip(true_labels, predicted_labels)):
    if t != p:
        distilbert_errors.append({
            "index": i,
            "sentence": texts[i],
            "true": t,
            "predicted": p,
            "confidence": confidences[i],
        })

print("=" * 70)
print(f"DISTILBERT ERRORS: {len(distilbert_errors)} out of {len(texts)} sentences")
print("=" * 70)

for i, err in enumerate(distilbert_errors, 1):
    print(f"\n--- Error {i} (confidence: {err['confidence']:.3f}) ---")
    print(f"Text: {err['sentence']}")
    print(f"True: {err['true']} | Predicted: {err['predicted']}")

# -----------------------------------------------------------------
# Cross-model comparison
# -----------------------------------------------------------------
# FinBERT's 8 error sentence substrings (from earlier analysis)
finbert_error_subs = [
    "These developments partly reflect",
    "Operating loss was EUR 179mn",
    "The new location is",
    "Unit costs for flight operations fell",
    "Previously , EB delivered",
    "Finnish Talvivaara Mining",
    "However , the broker gave",
    "Operating profit totaled EUR 17.7",
]

# Llama's 8 error sentence substrings
llama_error_subs = [
    "The order was valued at over EUR15m",
    "YIT lodged counter claims",
    "With five different game modes",
    "Local government commissioner of",
    "The profit after taxes was EUR 57.7",
    "Purchase it for the 12MP snapper",
    "Altona stated that the private company",
    "The value of the orders is over EUR 25mn",
]

def find_match(substr: str) -> int:
    """Find test set index of a sentence containing substr."""
    for i, s in enumerate(texts):
        if substr in s:
            return i
    return -1

print("\n" + "=" * 70)
print("CROSS-MODEL OVERLAP ANALYSIS")
print("=" * 70)

# Did DistilBERT solve FinBERT's errors?
print("\n📍 Did DistilBERT solve FinBERT's 8 errors?\n")
solved_finbert = 0
for substr in finbert_error_subs:
    idx = find_match(substr)
    if idx == -1:
        print(f"  ⚠️  Sentence not found in test: \"{substr}...\"")
        continue
    correct = true_labels[idx] == predicted_labels[idx]
    if correct:
        solved_finbert += 1
        marker = "✅ SOLVED"
    else:
        marker = "❌ ALSO WRONG"
    print(f"  {marker}: \"{substr[:50]}...\"")
    print(f"     True: {true_labels[idx]} | DistilBERT: {predicted_labels[idx]}")

# Did DistilBERT solve Llama's errors?
print("\n📍 Did DistilBERT solve Llama's 8 errors?\n")
solved_llama = 0
for substr in llama_error_subs:
    idx = find_match(substr)
    if idx == -1:
        print(f"  ⚠️  Sentence not found in test: \"{substr}...\"")
        continue
    correct = true_labels[idx] == predicted_labels[idx]
    if correct:
        solved_llama += 1
        marker = "✅ SOLVED"
    else:
        marker = "❌ ALSO WRONG"
    print(f"  {marker}: \"{substr[:50]}...\"")
    print(f"     True: {true_labels[idx]} | DistilBERT: {predicted_labels[idx]}")

# How many of DistilBERT's errors overlap with FinBERT or Llama?
distilbert_error_texts = [e["sentence"] for e in distilbert_errors]

finbert_overlap = sum(
    1 for et in distilbert_error_texts
    if any(sub in et for sub in finbert_error_subs)
)
llama_overlap = sum(
    1 for et in distilbert_error_texts
    if any(sub in et for sub in llama_error_subs)
)

print("\n" + "=" * 70)
print("THE BIG NUMBERS")
print("=" * 70)
print(f"\nDistilBERT errors:                {len(distilbert_errors)}")
print(f"Of those, also in FinBERT errors:  {finbert_overlap}")
print(f"Of those, also in Llama errors:    {llama_overlap}")
print(f"\nFinBERT's 8 errors that DistilBERT solved: {solved_finbert}/8")
print(f"Llama's 8 errors that DistilBERT solved:   {solved_llama}/8")