"""
llama_error_analysis.py
Identify which test sentences Llama 3.1 got wrong.
"""
import json
from pathlib import Path
from data_utils import RESULTS_DIR

# Load the raw responses we saved earlier
with open(RESULTS_DIR / "llama_raw_responses.json") as f:
    data = json.load(f)

sentences = data["sentences"]
true_labels = data["true_labels"]
predicted = data["predicted_labels"]
raw = data["raw_responses"]

# Find errors
errors = []
for i, (t, p) in enumerate(zip(true_labels, predicted)):
    if t != p:
        errors.append({
            "index": i,
            "sentence": sentences[i],
            "true": t,
            "predicted": p,
            "raw_response": raw[i],
        })

print("=" * 70)
print(f"LLAMA ERRORS: {len(errors)} out of {len(sentences)} sentences")
print("=" * 70)

for i, err in enumerate(errors, 1):
    print(f"\n--- Error {i} ---")
    print(f"Text: {err['sentence']}")
    print(f"True label:    {err['true']}")
    print(f"Predicted:     {err['predicted']}")
    print(f"Raw response:  {err['raw_response']}")

# -----------------------------------------------------------------
# Compare Llama vs FinBERT — do they make the same mistakes?
# -----------------------------------------------------------------
print("\n" + "=" * 70)
print("OVERLAP ANALYSIS: Llama errors vs. FinBERT errors")
print("=" * 70)

# Quick check: see which Llama-error sentences match the FinBERT errors I noted
finbert_error_substrings = [
    "These developments partly reflect",
    "Operating loss was EUR 179mn",
    "The new location is",
    "Unit costs for flight operations fell",
    "Previously , EB delivered",
    "Finnish Talvivaara Mining",
    "However , the broker gave",
    "Operating profit totaled EUR 17.7",
]

print("\nDid Llama get FinBERT's errors right or also wrong?\n")
for substr in finbert_error_substrings:
    matches = [i for i, s in enumerate(sentences) if substr in s]
    if matches:
        idx = matches[0]
        same = "❌ ALSO WRONG" if true_labels[idx] != predicted[idx] else "✅ GOT IT RIGHT"
        print(f"  {same}: \"{substr}...\"")
        print(f"     True: {true_labels[idx]} | Llama: {predicted[idx]}")