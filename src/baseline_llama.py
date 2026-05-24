"""
baseline_llama.py
Runs Llama 3.1 70B via Groq API on the test set.
This is the "giant generalist" baseline — testing whether a large
general-purpose LLM can match domain-specialized models on a narrow task.

Run with: python src/baseline_llama.py

Note: Takes ~12-15 minutes due to rate limiting on Groq's free tier.
"""
import os
import time
import json
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

from data_utils import load_split, RESULTS_DIR
from evaluate import evaluate_predictions

# -----------------------------------------------------------------
# Setup
# -----------------------------------------------------------------
print("=" * 60)
print("BASELINE 3: Llama 3.1 70B via Groq")
print("=" * 60)

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise SystemExit("❌ GROQ_API_KEY not found in .env file")

client = Groq(api_key=api_key)
MODEL = "llama-3.3-70b-versatile"  # Groq's current Llama model

# -----------------------------------------------------------------
# The prompt template
# -----------------------------------------------------------------
SYSTEM_PROMPT = """You are a financial sentiment classifier.

You read sentences from financial news and reports, and classify their sentiment about the company or financial event being described.

Categories:
- positive: favorable business outcomes (growth, profit, gains, success, beating expectations)
- negative: unfavorable business outcomes (losses, declines, problems, missing expectations)
- neutral: factual statements without clear positive or negative sentiment

Important: Consider business context. "Costs fell" is positive (lower costs = good). "Loss decreased" is positive (less loss = better). "Revenue fell" is negative.

You must respond with EXACTLY ONE WORD: positive, negative, or neutral."""


def classify_sentence(sentence: str, max_retries: int = 3) -> tuple[str, str]:
    """
    Send a sentence to Llama and parse the response.
    Returns (label, raw_response) tuple.
    Falls back to "neutral" if parsing fails.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f'Classify this sentence: "{sentence}"'},
                ],
                temperature=0,  # deterministic output
                max_tokens=10,  # we only need one word; cap response length
            )
            raw = response.choices[0].message.content.strip().lower()
            
            # Parse: look for our exact category words anywhere in the response
            # Order matters — check in order of specificity
            if "positive" in raw:
                return "positive", raw
            elif "negative" in raw:
                return "negative", raw
            elif "neutral" in raw:
                return "neutral", raw
            else:
                # Llama returned something unexpected; default to neutral
                return "neutral", f"UNPARSEABLE:{raw}"
        
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"   ⚠️ Retry {attempt+1}/{max_retries}: {e}")
                time.sleep(5)  # wait before retrying
            else:
                return "neutral", f"ERROR:{e}"


# -----------------------------------------------------------------
# Load test set
# -----------------------------------------------------------------
print("\n[1] Loading test set...")
test = load_split("test")
print(f"    Loaded {len(test)} test sentences")

# -----------------------------------------------------------------
# Quick API sanity check
# -----------------------------------------------------------------
print("\n[2] Testing API connection...")
test_label, test_raw = classify_sentence("Revenue grew by 25% this quarter.")
print(f"    Test sentence → predicted: {test_label} (raw: {test_raw!r})")
if test_label != "positive":
    print(f"    ⚠️  Expected 'positive' but got '{test_label}'. Continuing anyway.")
else:
    print("    ✅ API working as expected")

# -----------------------------------------------------------------
# Run on full test set with rate limiting
# -----------------------------------------------------------------
print(f"\n[3] Classifying {len(test)} sentences via Groq API...")
print(f"    Rate limiting: ~2 sec/request → estimated {len(test) * 2 / 60:.1f} minutes")
print(f"    Using model: {MODEL}\n")

predictions = []
raw_responses = []
unparseable_count = 0

# Use tqdm for a nice progress bar
for sentence in tqdm(test["text"].tolist(), desc="Classifying"):
    label, raw = classify_sentence(sentence)
    predictions.append(label)
    raw_responses.append(raw)
    if raw.startswith("UNPARSEABLE") or raw.startswith("ERROR"):
        unparseable_count += 1
    time.sleep(1.0)  # rate limit pacing (30 req/min = 2 sec, but we'll use 1 sec since Groq is fast)

print(f"\n    ✅ Classification complete")
print(f"    Unparseable responses: {unparseable_count}/{len(test)}")

# -----------------------------------------------------------------
# Save raw responses for debugging
# -----------------------------------------------------------------
raw_log_path = RESULTS_DIR / "llama_raw_responses.json"
RESULTS_DIR.mkdir(exist_ok=True)
with open(raw_log_path, "w") as f:
    json.dump({
        "model": MODEL,
        "sentences": test["text"].tolist(),
        "true_labels": test["label_name"].tolist(),
        "predicted_labels": predictions,
        "raw_responses": raw_responses,
    }, f, indent=2)
print(f"    💾 Raw responses saved to {raw_log_path.relative_to(RESULTS_DIR.parent)}")

# -----------------------------------------------------------------
# Sanity check: show a few example predictions
# -----------------------------------------------------------------
print("\n[4] Sample predictions vs. ground truth:")
y_true = test["label_name"].tolist()
texts = test["text"].tolist()
for i in range(5):
    text_preview = texts[i][:80] + ("..." if len(texts[i]) > 80 else "")
    correct = "✅" if predictions[i] == y_true[i] else "❌"
    print(f"   {correct} True: {y_true[i]:>8} | Pred: {predictions[i]:>8} | {text_preview}")

# -----------------------------------------------------------------
# Evaluate
# -----------------------------------------------------------------
metrics = evaluate_predictions(y_true, predictions, model_name="llama")

# -----------------------------------------------------------------
# Comparison: all three baselines side-by-side
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("ALL THREE BASELINES COMPARED")
print("=" * 60)

print(f"\n{'Model':<25} {'Accuracy':>10} {'F1 (wgt)':>10} {'F1 (mac)':>10}")
print("-" * 60)
print(f"{'Majority baseline':<25} {'61.36%':>10} {'N/A':>10} {'N/A':>10}")
print(f"{'VADER':<25} {'56.64%':>10} {'56.85%':>10} {'47.43%':>10}")
print(f"{'FinBERT (contam ⚠️)':<25} {'97.64%':>10} {'97.65%':>10} {'96.69%':>10}")
print(f"{'Llama 3.1 70B':<25} {metrics['accuracy']*100:>9.2f}% {metrics['f1_weighted']*100:>9.2f}% {metrics['f1_macro']*100:>9.2f}%")

print("\n" + "=" * 60)
print("BASELINE 3 COMPLETE")
print("=" * 60)