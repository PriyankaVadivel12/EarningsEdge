"""
load_data.py
Downloads FinancialPhraseBank (allagree subset), creates stratified
train/val/test splits, and saves them as CSVs in data/.

Run with: python src/load_data.py
"""
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from data_utils import save_split, ID_TO_LABEL


# Set a random seed everywhere so splits are reproducible
RANDOM_SEED = 42

print("=" * 60)
print("PHASE 1: DATA ACQUISITION")
print("=" * 60)

# -----------------------------------------------------------------
# Step 1: Download the dataset
# -----------------------------------------------------------------
print("\n[1] Downloading FinancialPhraseBank (allagree subset)...")
print("    Using gtfintechlab mirror (modern, no loading script issues)")

configs = ['5768', '78516', '944601']
all_parts = []

for config in configs:
    ds = load_dataset("gtfintechlab/financial_phrasebank_sentences_allagree", config)
    for split_name, split in ds.items():
        all_parts.append(split.to_pandas())

df = pd.concat(all_parts, ignore_index=True)
print(f"    Combined across configs: {len(df)} rows (with duplicates)")

# Find the text column (the mirror might call it 'sentence' or 'text')
text_col = "sentence" if "sentence" in df.columns else "text"
df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
print(f"    ✅ Final: {len(df)} unique sentences")

# -----------------------------------------------------------------
# Step 2: Standardize columns
# -----------------------------------------------------------------
print("\n[2] Standardizing column names...")

# Rename to consistent names
if text_col != "text":
    df = df.rename(columns={text_col: "text"})

# Some mirrors use 'labels' or 'sentiment' instead of 'label'
if "label" not in df.columns:
    for candidate in ["labels", "sentiment", "Label", "y"]:
        if candidate in df.columns:
            df = df.rename(columns={candidate: "label"})
            break

# If label is a string ('positive') instead of int, map it
if df["label"].dtype == object:
    print(f"    Label column is strings — converting to integers")
    LABEL_FROM_STRING = {"negative": 0, "neutral": 1, "positive": 2}
    df["label"] = df["label"].str.lower().map(LABEL_FROM_STRING)

# Add human-readable label_name column
df["label_name"] = df["label"].map(ID_TO_LABEL)

# Keep only the columns we need
df = df[["text", "label", "label_name"]].copy()

print(f"    ✅ Columns: {list(df.columns)}")
print(f"    ✅ First row:")
print(f"       text: {df.iloc[0]['text'][:80]}...")
print(f"       label: {df.iloc[0]['label']} ({df.iloc[0]['label_name']})")

# -----------------------------------------------------------------
# Step 3: Check class distribution
# -----------------------------------------------------------------
print("\n[3] Class distribution in full dataset:")
class_counts = df["label_name"].value_counts()
for label, count in class_counts.items():
    pct = 100 * count / len(df)
    print(f"    {label:>10}: {count:>5} ({pct:.1f}%)")

# -----------------------------------------------------------------
# Step 4: Stratified train/val/test split (70/15/15)
# -----------------------------------------------------------------
print("\n[4] Creating stratified splits (70/15/15)...")

# First split: separate test set (15%) from everything else (85%)
train_val, test = train_test_split(
    df,
    test_size=0.15,
    stratify=df["label"],
    random_state=RANDOM_SEED,
)

# Second split: from the 85%, take 15/85 ≈ 17.6% for val, rest for train
# This gives us roughly 70% train / 15% val / 15% test of the original
train, val = train_test_split(
    train_val,
    test_size=0.15 / 0.85,
    stratify=train_val["label"],
    random_state=RANDOM_SEED,
)

print(f"    ✅ Train: {len(train)} sentences ({100*len(train)/len(df):.1f}%)")
print(f"    ✅ Val:   {len(val)} sentences ({100*len(val)/len(df):.1f}%)")
print(f"    ✅ Test:  {len(test)} sentences ({100*len(test)/len(df):.1f}%)")

# -----------------------------------------------------------------
# Step 5: Verify stratification worked
# -----------------------------------------------------------------
print("\n[5] Class distribution per split (should be similar to overall):")
for split_name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
    dist = split_df["label_name"].value_counts(normalize=True)
    parts = [f"{label}={dist.get(label, 0)*100:.1f}%" for label in ["negative", "neutral", "positive"]]
    print(f"    {split_name:>6}: {', '.join(parts)}")

# -----------------------------------------------------------------
# Step 6: Save the splits
# -----------------------------------------------------------------
print("\n[6] Saving splits to data/ folder...")
save_split(train, "train")
save_split(val, "val")
save_split(test, "test")

print("\n" + "=" * 60)
print("PHASE 1 STEP 1 COMPLETE — Data downloaded and split")
print("=" * 60)
print("\nNext: Run python src/explore_data.py to analyze the data deeply.")