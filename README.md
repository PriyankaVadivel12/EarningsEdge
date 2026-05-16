# EarningsEdge - A Fine-Tubed Financial System

A financial sentiment analysis system that fine-tunes DistilBERT on financial text and benchmarks it against multiple baselines.

## Overview

Financial language uses domain-specific vocabulary where general-purpose sentiment tools fail (e.g., "crushed earnings" is positive, "headwinds" is negative). This project demonstrates that small, specialized models can match or beat much larger general-purpose models on narrow tasks.

## Approach

We benchmark four sentiment analysis approaches on the FinancialPhraseBank dataset:

1. **VADER** — Rule-based baseline (2014)
2. **FinBERT** — Pre-trained on financial text (ProsusAI)
3. **Llama 3.1 70B** — Large language model via Groq (zero-shot)
4. **Fine-tuned DistilBERT** — Our specialized model

## Tech Stack

- Python 3.13
- PyTorch + Hugging Face Transformers
- Groq API for LLM inference
- Streamlit for deployment
- Hosted on Hugging Face Spaces

## Status

🚧 **In active development** (May 16 – June 7, 2026)

- [x] Phase 0: Environment setup
- [ ] Phase 1: Data acquisition + exploration
- [ ] Phase 2: Baseline implementations
- [ ] Phase 3: Fine-tuning DistilBERT
- [ ] Phase 4: Evaluation + comparison
- [ ] Phase 5: Streamlit demo
- [ ] Phase 6: Deployment + write-up

## Author

[Priyanka Vadivel](https://www.linkedin.com/in/priyankavadivel) — MS Information Systems, Northeastern University