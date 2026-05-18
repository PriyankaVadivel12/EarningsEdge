"""
explore_data.py
Exploratory analysis of the FinancialPhraseBank dataset.
Prints statistics and saves a summary report.

Run with: python src/explore_data.py
"""
import pandas as pd

from data_utils import load_split, RESULTS_DIR

print("=" * 60)
print("PHASE 1: DATA EXPLORATION")
print("=" * 60)

# -----------------------------------------------------------------
# Load all three splits
# -----------------------------------------------------------------
train = load_split("train")
val = load_split("val")
test = load_split("test")

# Combined view for overall stats
full = pd.concat([train, val, test], ignore_index=True)
print(f"\nTotal sentences across all splits: {len(full)}")

# -----------------------------------------------------------------
# Section 1: Class distribution
# -----------------------------------------------------------------
print("\n" + "-" * 60)
print("CLASS DISTRIBUTION")
print("-" * 60)
for label in ["negative", "neutral", "positive"]:
    count = (full["label_name"] == label).sum()
    pct = 100 * count / len(full)
    bar = "█" * int(pct / 2)  # visual bar
    print(f"{label:>10}: {count:>5} ({pct:>5.1f}%) {bar}")

# Key insight: the majority baseline
majority_pct = 100 * (full["label_name"] == "neutral").sum() / len(full)
print(f"\n💡 If a model just predicts 'neutral' every time: {majority_pct:.1f}% accuracy")
print("   Our models need to beat this to be worthwhile.")

# -----------------------------------------------------------------
# Section 2: Sentence length statistics
# -----------------------------------------------------------------
print("\n" + "-" * 60)
print("SENTENCE LENGTH STATISTICS")
print("-" * 60)

full["word_count"] = full["text"].str.split().str.len()
full["char_count"] = full["text"].str.len()

print(f"\nWord count:")
print(f"   Min:    {full['word_count'].min()}")
print(f"   Max:    {full['word_count'].max()}")
print(f"   Mean:   {full['word_count'].mean():.1f}")
print(f"   Median: {full['word_count'].median():.0f}")

print(f"\nCharacter count:")
print(f"   Min:    {full['char_count'].min()}")
print(f"   Max:    {full['char_count'].max()}")
print(f"   Mean:   {full['char_count'].mean():.1f}")

# Length by class — are negative sentences longer than positive?
print("\nMean word count by class:")
for label in ["negative", "neutral", "positive"]:
    avg = full[full["label_name"] == label]["word_count"].mean()
    print(f"   {label:>10}: {avg:.1f} words")

# -----------------------------------------------------------------
# Section 3: Sample sentences from each class
# -----------------------------------------------------------------
print("\n" + "-" * 60)
print("SAMPLE SENTENCES (3 per class)")
print("-" * 60)

for label in ["negative", "neutral", "positive"]:
    print(f"\n📌 {label.upper()}:")
    samples = full[full["label_name"] == label]["text"].sample(3, random_state=42).tolist()
    for i, s in enumerate(samples, 1):
        print(f"   {i}. {s[:150]}{'...' if len(s) > 150 else ''}")

# -----------------------------------------------------------------
# Section 4: Vocabulary signals (finance-specific words)
# -----------------------------------------------------------------
print("\n" + "-" * 60)
print("FINANCE VOCABULARY CHECK")
print("-" * 60)
print("Counting key finance words to confirm this is real financial data...")

finance_words = ["EUR", "profit", "loss", "revenue", "earnings", "quarter", "sales", "growth", "decline", "shares"]
text_lower = full["text"].str.lower()

print()
for word in finance_words:
    count = text_lower.str.contains(word.lower()).sum()
    pct = 100 * count / len(full)
    print(f"   '{word:>10}' appears in {count:>4} sentences ({pct:.1f}%)")

# -----------------------------------------------------------------
# Section 5: Save summary to results/
# -----------------------------------------------------------------
RESULTS_DIR.mkdir(exist_ok=True)
summary_path = RESULTS_DIR / "data_summary.txt"

with open(summary_path, "w") as f:
    f.write("FinancialPhraseBank — Data Summary\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Total sentences: {len(full)}\n")
    f.write(f"Train / Val / Test: {len(train)} / {len(val)} / {len(test)}\n\n")
    f.write("Class distribution:\n")
    for label in ["negative", "neutral", "positive"]:
        count = (full["label_name"] == label).sum()
        pct = 100 * count / len(full)
        f.write(f"  {label:>10}: {count:>5} ({pct:.1f}%)\n")
    f.write(f"\nMajority baseline (always predict 'neutral'): {majority_pct:.1f}%\n")
    f.write(f"\nMean word count: {full['word_count'].mean():.1f}\n")
    f.write(f"Max word count: {full['word_count'].max()}\n")

print(f"\n✅ Summary saved → {summary_path.relative_to(RESULTS_DIR.parent)}")

print("\n" + "=" * 60)
print("PHASE 1 COMPLETE — Data downloaded, split, and explored")
print("=" * 60)