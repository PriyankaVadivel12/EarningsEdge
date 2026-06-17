"""
EarningsEdge — Interactive Demo
A Streamlit app comparing 4 sentiment models on financial text.

Run with:    streamlit run app/streamlit_app.py
Then open:   http://localhost:8501
"""
import os
import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import torch

# Make sure we can import from src/
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# =================================================================
# Page configuration
# =================================================================
st.set_page_config(
    page_title="EarningsEdge — Financial Sentiment AI",
    page_icon="📈",
    layout="wide",
)

# =================================================================
# Load all models (cached so they only load once)
# =================================================================
@st.cache_resource
def load_distilbert():
    """Load the fine-tuned DistilBERT model."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_dir = PROJECT_ROOT / "models" / "distilbert_finetuned" / "final"
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model.eval()
    return model, tokenizer


@st.cache_resource
def load_finbert():
    """Load the pre-trained FinBERT."""
    from transformers import pipeline
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")


@st.cache_resource
def load_vader():
    """Load VADER analyzer."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


@st.cache_resource
def load_groq_client():
    """Load Groq client if API key is set."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    from groq import Groq
    return Groq(api_key=api_key)


# =================================================================
# Inference functions for each model
# =================================================================
LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL_COLORS = {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}


def predict_distilbert(text: str) -> dict:
    """Run our fine-tuned DistilBERT."""
    model, tokenizer = load_distilbert()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
    return {
        "label": LABEL_NAMES[probs.argmax().item()],
        "probabilities": {name: float(probs[i]) for i, name in enumerate(LABEL_NAMES)},
    }


def predict_finbert(text: str) -> dict:
    """Run FinBERT via Hugging Face pipeline (returns all class scores)."""
    classifier = load_finbert()
    raw = classifier(text, top_k=None, truncation=True, max_length=512)
    probs = {item["label"].lower(): item["score"] for item in raw}
    # Make sure all 3 keys exist
    for label in LABEL_NAMES:
        probs.setdefault(label, 0.0)
    return {
        "label": max(probs, key=probs.get),
        "probabilities": probs,
    }


def predict_vader(text: str) -> dict:
    """Run VADER and convert compound score to 3-class probabilities."""
    analyzer = load_vader()
    scores = analyzer.polarity_scores(text)
    # VADER's compound score is -1 to +1
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    # Use VADER's pos/neg/neu directly as "probabilities"
    return {
        "label": label,
        "probabilities": {
            "negative": scores["neg"],
            "neutral":  scores["neu"],
            "positive": scores["pos"],
        },
    }


def predict_llama(text: str) -> dict:
    """Run Llama 3.1 via Groq."""
    client = load_groq_client()
    if client is None:
        return {"label": "error", "probabilities": {}, "error": "GROQ_API_KEY not set"}

    SYSTEM_PROMPT = """You are a financial sentiment classifier.
Classify financial sentences as exactly one of: positive, negative, neutral.
- positive: favorable business outcomes (growth, profit, gains)
- negative: unfavorable outcomes (losses, declines, problems)
- neutral: factual statements without clear sentiment
Respond with EXACTLY ONE WORD: positive, negative, or neutral."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f'Classify: "{text}"'},
            ],
            temperature=0,
            max_tokens=10,
        )
        raw = response.choices[0].message.content.strip().lower()
        if "positive" in raw:
            label = "positive"
        elif "negative" in raw:
            label = "negative"
        elif "neutral" in raw:
            label = "neutral"
        else:
            label = "neutral"
        # Llama doesn't give probabilities, so we use 1.0 for the predicted class
        probs = {name: (1.0 if name == label else 0.0) for name in LABEL_NAMES}
        return {"label": label, "probabilities": probs}
    except Exception as e:
        return {"label": "error", "probabilities": {}, "error": str(e)}


# =================================================================
# UI helpers
# =================================================================
def render_prediction_card(model_name: str, result: dict, badge: str = ""):
    """Render a prediction result as a card."""
    if "error" in result:
        st.error(f"**{model_name}**: {result['error']}")
        return

    label = result["label"]
    probs = result["probabilities"]
    color = LABEL_COLORS.get(label, "#666")

    # Header with model name and predicted label
    header = f"**{model_name}** {badge}"
    st.markdown(header)
    st.markdown(
        f"<div style='font-size:1.6em; font-weight:bold; color:{color}; "
        f"margin-bottom:8px;'>{label.upper()}</div>",
        unsafe_allow_html=True,
    )

    # Probability bars
    for name in LABEL_NAMES:
        prob = probs.get(name, 0.0)
        bar_color = LABEL_COLORS[name]
        st.markdown(
            f"<div style='display:flex; align-items:center; margin-bottom:4px;'>"
            f"<div style='width:80px; font-size:0.85em;'>{name}</div>"
            f"<div style='flex:1; background:#f0f0f0; border-radius:4px; height:14px; overflow:hidden;'>"
            f"<div style='width:{prob*100}%; background:{bar_color}; height:100%;'></div>"
            f"</div>"
            f"<div style='width:50px; text-align:right; font-size:0.85em;'>{prob*100:.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# =================================================================
# Main UI
# =================================================================
st.title("📈 EarningsEdge")
st.markdown(
    "**Financial Sentiment AI** — Compare 4 sentiment models on financial text. "
    "Built as a portfolio project demonstrating fine-tuning, evaluation, and deployment."
)

# Sidebar with project info
with st.sidebar:
    st.header("About the project")
    st.markdown(
        """
This demo lets you compare:

- **VADER** — Rule-based baseline (2014)
- **FinBERT** — Pre-trained on financial text
- **Llama 3.1 70B** — General LLM via Groq
- **DistilBERT (mine)** — Fine-tuned on FinancialPhraseBank

**Test set accuracy:**
- VADER: 56.6%
- FinBERT: 97.6% ⚠️
- Llama 3.1: 97.6%
- **DistilBERT (mine): 96.2%**

(⚠️ FinBERT's number is inflated due to train/test contamination on this dataset.)
        """
    )
    st.markdown("---")
    st.markdown(
        "**Built by Priyanka Vadivel**  \n"
        "MS Information Systems, Northeastern University  \n"
        "[GitHub](https://github.com/PriyankaVadivel12/EarningsEdge) · "
        "[LinkedIn](https://www.linkedin.com/in/priyankavadivel)"
    )

# Input area
st.markdown("### Try it yourself")

example_sentences = {
    "(choose an example…)": "",
    "Revenue growth": "Revenue grew by 15% year-over-year, beating analyst expectations.",
    "Loss decreased": "Operating loss was EUR 179mn, compared to a loss of EUR 188mn in the second quarter of 2009.",
    "Costs fell": "Unit costs for flight operations fell by 6.4 percent.",
    "Outperform recommendation": "However, the broker gave an 'outperform' recommendation on the stock.",
    "Neutral statement": "The Annual Report contains the financial statements and the report by the Board of Directors.",
    "Mixed signal": "Apple's revenue grew 15% but operating margins were compressed.",
}

choice = st.selectbox("Or pick an example:", list(example_sentences.keys()))
default_text = example_sentences[choice]

text = st.text_area(
    "Paste a financial sentence:",
    value=default_text,
    height=100,
    placeholder="e.g., 'Revenue grew by 12% year-over-year, exceeding expectations.'",
)

classify_button = st.button("🔍 Classify with all models", type="primary")

# Run inference when the button is clicked
if classify_button and text.strip():
    with st.spinner("Running 4 models..."):
        results = {
            "DistilBERT (mine)": (predict_distilbert(text), "🏆 Fine-tuned by me"),
            "FinBERT":           (predict_finbert(text), "Pre-trained baseline"),
            "Llama 3.1 70B":     (predict_llama(text), "Via Groq API"),
            "VADER":             (predict_vader(text), "Rule-based baseline"),
        }

    # Display in a 2x2 grid
    col1, col2 = st.columns(2)
    cols = [col1, col2, col1, col2]
    for (name, (result, badge)), col in zip(results.items(), cols):
        with col:
            with st.container(border=True):
                render_prediction_card(name, result, badge)

    # Show agreement / disagreement summary
    st.markdown("---")
    labels_only = [r[0]["label"] for r in results.values() if r[0]["label"] != "error"]
    unique_labels = set(labels_only)
    if len(unique_labels) == 1:
        st.success(f"✅ All models agree: **{labels_only[0].upper()}**")
    else:
        st.info(
            f"🤔 Models disagree: " + ", ".join(
                f"**{name}** → {result[0]['label']}"
                for name, result in results.items()
                if result[0]["label"] != "error"
            )
        )

elif classify_button and not text.strip():
    st.warning("Please enter a sentence to classify.")
else:
    st.info("👆 Paste a sentence (or pick an example) and click Classify.")

# Footer with technical details
with st.expander("How was this built?"):
    st.markdown(
        """
**Approach:**
1. Used FinancialPhraseBank dataset (~2,259 financial sentences, expert-labeled)
2. Split 70/15/15 train/val/test, stratified by class
3. Established 3 baselines (VADER, FinBERT, Llama 3.1 70B via Groq)
4. Fine-tuned DistilBERT (66M parameters) for 3 epochs on Apple Silicon MPS
5. Evaluated all 4 models on held-out test set with per-class metrics

**Key findings:**
- DistilBERT (mine) achieves 96.17% test accuracy with 1000× fewer parameters than Llama 3.1
- FinBERT and Llama have ZERO error overlap despite identical 97.64% accuracy
- DistilBERT shares architectural blind spots with FinBERT but solves 7/8 of Llama's errors
- Only 1 of 339 test sentences is misclassified by all three high-performing models

**Tech stack:** Python · PyTorch · Hugging Face Transformers · Groq API · scikit-learn · Streamlit
        """
    )