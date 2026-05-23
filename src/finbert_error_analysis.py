"""
finbert_error_analysis.py
Identify which test sentences FinBERT got wrong.
Helps understand the hard cases.
"""
import torch
from transformers import pipeline

from data_utils import load_split

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

print("\nLoading FinBERT (cached)...")
classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)

test = load_split("test")
print(f"Test set: {len(test)} sentences\n")

# Get predictions
texts = test["text"].tolist()
raw_preds = classifier(texts, truncation=True, max_length=512, batch_size=16)
predictions = [p["label"].lower() for p in raw_preds]
confidences = [p["score"] for p in raw_preds]

# Find errors
test = test.copy()
test["predicted"] = predictions
test["confidence"] = confidences
errors = test[test["label_name"] != test["predicted"]].reset_index(drop=True)

print("=" * 70)
print(f"FINBERT ERRORS: {len(errors)} out of {len(test)} sentences")
print("=" * 70)

for i, row in errors.iterrows():
    print(f"\n--- Error {i+1} ---")
    print(f"Text: {row['text']}")
    print(f"True label:    {row['label_name']}")
    print(f"Predicted:     {row['predicted']} (confidence: {row['confidence']:.3f})")