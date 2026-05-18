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