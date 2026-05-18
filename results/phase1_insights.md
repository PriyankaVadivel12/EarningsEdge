# Phase 1 Insights — EDA Findings

## Dataset Summary
- **Total sentences:** 2,259 (FinancialPhraseBank, allagree subset)
- **Splits:** 1,581 train / 339 val / 339 test (stratified)
- **Source:** Finnish/European financial news (32% mention EUR)

## Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Neutral | 1,386 | 61.4% |
| Positive | 570 | 25.2% |
| Negative | 303 | 13.4% |

**Heavily imbalanced** — neutral dominates. This will affect modeling decisions.

## Key Findings

### 1. Majority Baseline = 61.4%
A model that always predicts "neutral" already gets 61.4% accuracy.
This sets the floor: any meaningful model must beat this significantly.
We'll focus on **per-class F1 score** rather than raw accuracy.

### 2. Sentence Length Varies by Class
- Negative: 24.8 words on average
- Positive: 24.9 words on average
- Neutral: 20.9 words on average

Neutral sentences are typically shorter factual statements.
Positive/negative sentences need more context to convey sentiment.

### 3. Domain-Specific Vocabulary Required
Examples of sentences requiring finance knowledge:
- "Revenue totaled 27.4B, down 2 percent" → NEGATIVE (despite large number)
- "Received a major order" → POSITIVE (no obvious sentiment words)
- "The Annual Report contains financial statements" → NEUTRAL (boilerplate)

This is why off-the-shelf sentiment tools fail on financial text.

### 4. Bias Toward European Financial Language
- 32% of sentences contain "EUR"
- Many Finnish company names (KONE, Konecranes, CapMan, Basware)
- Model may underperform on US-style language ("up 2%", "$1.2B")

**Limitation to document in final write-up.**

### 5. Spurious Feature Risk
Money amounts are mentioned in most sentences. Model could learn
shortcuts based on number presence rather than true sentiment patterns.
**Mitigation:** Rigorous error analysis in Phase 4 will check this.