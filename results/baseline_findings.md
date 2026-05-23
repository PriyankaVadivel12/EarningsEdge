# Baseline Findings

## Baseline 1: VADER (rule-based, 2014)

### Headline Result
**VADER underperformed the majority-class baseline by 4.72 points** (56.64% vs 61.36% accuracy).
A rule-based sentiment tool from 2014 is *worse than predicting "neutral" every time* on financial text.

### Per-Class Performance

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Negative | 40.00% | 17.78% | 24.62% |
| Neutral | 78.67% | 56.73% | 65.92% |
| Positive | 39.05% | 76.74% | 51.76% |

### Failure Modes Identified

1. **Negative sentences misclassified as positive (20/45 = 44%)**
   VADER reads "Revenue totaled X, down 2%" and sees the positive-sounding "revenue" 
   without understanding "down 2%" reverses the meaning.

2. **Neutral sentences misclassified as positive (83/208 = 40%)**
   Factual statements with words like "achieved", "completed", "operates" trigger 
   VADER's positive-word heuristics, even when no sentiment is intended.

3. **Negative class recall only 17.78%**
   VADER misses 82% of negative cases — the most important class for a 
   risk-monitoring use case in finance.

### Key Insight
VADER's training corpus (social media) emphasizes emotional language. Financial text 
emphasizes factual, contextual language. The domain mismatch is severe enough that 
VADER actively introduces errors rather than detecting signal.

This motivates the rest of the project: financial sentiment requires either 
domain-specific pretraining (FinBERT) or sufficient general intelligence (LLMs) 
or fine-tuning.

### Files
- Predictions: `results/baseline_vader.json`


## Baseline 2: FinBERT (ProsusAI, pre-trained + fine-tuned on financial text)

### Headline Result
**FinBERT achieved 97.64% accuracy — a 41-point improvement over VADER (56.64%)
and 36 points over the majority baseline (61.36%).**

### Per-Class Performance

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Negative | 91.84% | 100.00% | 95.74% |
| Neutral | 99.51% | 98.56% | 99.03% |
| Positive | 96.43% | 94.19% | 95.29% |

### Confusion Matrix

|              | Pred Negative | Pred Neutral | Pred Positive |
|--------------|---------------|--------------|----------------|
| **True Negative** | 45 | 0 | 0 |
| **True Neutral**  | 0  | 205 | 3 |
| **True Positive** | 4  | 1 | 81 |

Only 8 misclassifications out of 339 sentences.

### ⚠️ Important Caveat: Train/Test Contamination

FinBERT was originally fine-tuned on the FinancialPhraseBank dataset (Araci, 2019).
Our test set is a random 15% split from the same dataset, meaning FinBERT has
almost certainly seen most of our test sentences during its own training.

This inflates FinBERT's reported performance. The original FinBERT paper reported
~86% on a held-out FinancialPhraseBank split using 60/20/20 splits. Our 97.64%
likely reflects significant test-set overlap.

**This is a known limitation of using FinBERT as a benchmark on FinancialPhraseBank.**
We include it because:
1. It represents what someone deploying off-the-shelf financial sentiment today
   would actually use
2. Our fine-tuned DistilBERT (Phase 3) will train ONLY on our train split, with
   strict test holdout — making our comparison artificially harder for us, not easier

### Comparison Table

| Model              | Accuracy | F1 (weighted) | F1 (macro) |
|--------------------|----------|---------------|------------|
| Majority baseline  | 61.36%   | N/A           | N/A        |
| VADER              | 56.64%   | 56.85%        | 47.43%     |
| FinBERT (caveat ⚠️)| 97.64%   | 97.65%        | 96.69%     |

### Key Insight
The jump from VADER (56.64%) to FinBERT (97.64%) — even with contamination caveats —
demonstrates the dramatic value of domain-specific pretraining. A model that has
*read* financial text understands "down 2%" reverses sentiment, and that 
"received a major order" is positive without explicit positive words.

This sets the bar high for the rest of the project.

### Files
- Predictions: `results/baseline_finbert.json`

## FinBERT Error Analysis (8 mistakes, deep dive)

Even with contamination advantage, FinBERT failed 8 of 339 sentences.
The failures cluster into 4 distinct patterns:

### Pattern 1: Arithmetic reasoning failures (2 errors)
FinBERT cannot reason about "this number compared to that number."

- "Operating loss was EUR 179mn, compared to a loss of EUR 188mn..." 
  → True: positive (loss shrank). Predicted: negative.
- "Operating profit totaled EUR 17.7 mn compared to EUR 17.6 mn..."
  → True: positive (profit grew). Predicted: negative.

**Insight:** BERT-family models lack arithmetic reasoning. "Loss decreased"
requires understanding that smaller-of-two-negatives is positive.

### Pattern 2: Semantic inversion failures (1 error)
Some verbs flip sentiment depending on subject.

- "Unit costs for flight operations fell by 6.4 percent."
  → True: positive (lower costs = good). Predicted: negative.

**Insight:** "Fell" is overwhelmingly negative in general English. The
business inversion — falling COSTS being positive — wasn't learned strongly enough.

### Pattern 3: Subtle/jargon positive signals (2 errors)
FinBERT misses sentiment conveyed via rare domain terms.

- "Previously delivered for LG... now making it commercially available
  for other mobile terminal vendors..." → True: positive. Predicted: neutral.
- "However, the broker gave an 'outperform' recommendation."
  → True: positive. Predicted: negative.

**Insight:** "Outperform" is a strong positive signal in finance but
appears rarely in general training. "Commercially available" suggests
business expansion but isn't a typical sentiment cue.

### Pattern 4: Over-positive bias on factual statements (3 errors)
FinBERT calls neutral statements positive when they contain
business-positive-sounding words.

- "...government's higher activity in the field of dividend policy."
  → True: neutral. Predicted: positive.
- "The new location isn't the only change Wellmont has in store..."
  → True: neutral. Predicted: positive.
- "Talvivaara picked BofA Merrill Lynch and JPMorgan as joint 
  bookrunners..." → True: neutral. Predicted: positive.

**Insight:** Big bank names, large numbers, and "active" verbs trigger
spurious positivity. This is a learned correlation from training data
that doesn't generalize properly to descriptive text.

### Implications for Phase 3

When evaluating our fine-tuned DistilBERT, we will specifically check
performance on these 8 sentences to determine:
1. Do the same patterns persist (structural problems)?
2. Does fine-tuning on the train split mitigate any patterns?
3. Are there NEW failure patterns introduced by our smaller model?