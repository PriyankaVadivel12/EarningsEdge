"""
prepare_data.py
Tokenize the train/val/test splits for DistilBERT fine-tuning.
Saves tokenized datasets to disk for use in training.

Run with: python src/prepare_data.py
"""
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from data_utils import load_split, DATA_DIR, LABEL_TO_ID

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128  # max tokens per sentence; covers ~99% of our data
TOKENIZED_DIR = DATA_DIR / "tokenized"

print("=" * 60)
print("PHASE 3.1: TOKENIZATION + DATASET PREP")
print("=" * 60)

# -----------------------------------------------------------------
# Step 1: Load the splits
# -----------------------------------------------------------------
print("\n[1] Loading data splits...")
train_df = load_split("train")
val_df = load_split("val")
test_df = load_split("test")

print(f"    Train: {len(train_df)} sentences")
print(f"    Val:   {len(val_df)} sentences")
print(f"    Test:  {len(test_df)} sentences")

# -----------------------------------------------------------------
# Step 2: Load the tokenizer
# -----------------------------------------------------------------
print(f"\n[2] Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"    Vocabulary size: {tokenizer.vocab_size:,}")
print(f"    Max input length: {tokenizer.model_max_length}")

# -----------------------------------------------------------------
# Step 3: Show what tokenization looks like on a few examples
# -----------------------------------------------------------------
print("\n[3] Tokenization examples (so you can see what's happening):\n")
for i in range(3):
    sentence = train_df.iloc[i]["text"]
    label = train_df.iloc[i]["label_name"]
    
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.encode(sentence)
    
    preview = sentence[:80] + ("..." if len(sentence) > 80 else "")
    print(f"   Example {i+1} (label: {label}):")
    print(f"   Original: \"{preview}\"")
    print(f"   First 12 tokens: {tokens[:12]}")
    print(f"   Token IDs (first 12): {token_ids[:12]}")
    print(f"   Total tokens: {len(token_ids)}\n")

# -----------------------------------------------------------------
# Step 4: Convert pandas DataFrames to Hugging Face Datasets
# -----------------------------------------------------------------
print("[4] Converting to Hugging Face Dataset format...")

def to_hf_dataset(df: pd.DataFrame) -> Dataset:
    """Convert a pandas DataFrame to a HF Dataset with proper label format."""
    # Ensure label is integer (model wants int, not string)
    df = df.copy()
    df["label"] = df["label_name"].map(LABEL_TO_ID).astype(int)
    # Keep only the columns the model needs
    return Dataset.from_pandas(df[["text", "label"]], preserve_index=False)

dataset = DatasetDict({
    "train": to_hf_dataset(train_df),
    "validation": to_hf_dataset(val_df),
    "test": to_hf_dataset(test_df),
})

print(f"    ✅ Dataset created with splits: {list(dataset.keys())}")
print(f"    Sample train entry: {dataset['train'][0]}")

# -----------------------------------------------------------------
# Step 5: Tokenize each split
# -----------------------------------------------------------------
print("\n[5] Tokenizing all splits...")

def tokenize_batch(examples):
    """Tokenize a batch of sentences."""
    return tokenizer(
        examples["text"],
        truncation=True,        # cut off sentences longer than MAX_LENGTH
        padding="max_length",   # pad shorter sentences to MAX_LENGTH
        max_length=MAX_LENGTH,
    )

tokenized = dataset.map(
    tokenize_batch,
    batched=True,
    desc="Tokenizing",
)

print(f"    ✅ Tokenization complete")
print(f"    Columns now: {tokenized['train'].column_names}")

# -----------------------------------------------------------------
# Step 6: Inspect length distribution (sanity check)
# -----------------------------------------------------------------
print("\n[6] Checking token length distribution (before truncation)...")
import numpy as np

# Compute raw token lengths for each training sentence
raw_lengths = [
    len(tokenizer.encode(ex["text"], truncation=False))
    for ex in dataset["train"]
]

print(f"    Min:    {min(raw_lengths)}")
print(f"    Max:    {max(raw_lengths)}")
print(f"    Mean:   {np.mean(raw_lengths):.1f}")
print(f"    Median: {int(np.median(raw_lengths))}")
print(f"    P95:    {int(np.percentile(raw_lengths, 95))}")
print(f"    P99:    {int(np.percentile(raw_lengths, 99))}")
truncated = sum(1 for L in raw_lengths if L > MAX_LENGTH)
print(f"    Truncated by MAX_LENGTH={MAX_LENGTH}: {truncated}/{len(raw_lengths)} ({100*truncated/len(raw_lengths):.1f}%)")

# -----------------------------------------------------------------
# Step 7: Save the tokenized datasets to disk
# -----------------------------------------------------------------
print(f"\n[7] Saving tokenized datasets to {TOKENIZED_DIR}...")
TOKENIZED_DIR.mkdir(exist_ok=True)
tokenized.save_to_disk(str(TOKENIZED_DIR))
print(f"    ✅ Saved")

# -----------------------------------------------------------------
# Done
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 3.1 COMPLETE — Data ready for training")
print("=" * 60)
print(f"\nTokenized data location: {TOKENIZED_DIR.relative_to(DATA_DIR.parent)}")
print(f"Next: run training in Phase 3.2")