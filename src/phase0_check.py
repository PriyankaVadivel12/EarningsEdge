"""
Phase 0: Environment Verification
Run with: python phase0_check.py
"""
import sys
import os

print("=" * 60)
print("PHASE 0: EARNINGSEDGE ENVIRONMENT CHECK")
print("=" * 60)

# Check 1: Python version
print(f"\n[1] Python version: {sys.version.split()[0]}")

# Check 2: Core ML libraries
print("\n[2] Checking ML libraries...")
try:
    import transformers
    print(f"    ✅ transformers: {transformers.__version__}")
except ImportError:
    print("    ❌ transformers NOT installed")

try:
    import datasets
    print(f"    ✅ datasets: {datasets.__version__}")
except ImportError:
    print("    ❌ datasets NOT installed")

try:
    import torch
    print(f"    ✅ torch: {torch.__version__}")
    if torch.backends.mps.is_available():
        print("    ✅ Apple Silicon MPS available (good to know)")
except ImportError:
    print("    ❌ torch NOT installed")

try:
    import pandas
    print(f"    ✅ pandas: {pandas.__version__}")
except ImportError:
    print("    ❌ pandas NOT installed")

try:
    import sklearn
    print(f"    ✅ scikit-learn: {sklearn.__version__}")
except ImportError:
    print("    ❌ scikit-learn NOT installed")

# Check 3: Project-specific libraries
print("\n[3] Checking project libraries...")
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    print("    ✅ vaderSentiment")
except ImportError:
    print("    ❌ vaderSentiment NOT installed")

try:
    import groq
    print("    ✅ groq")
except ImportError:
    print("    ❌ groq NOT installed")

try:
    import streamlit
    print(f"    ✅ streamlit: {streamlit.__version__}")
except ImportError:
    print("    ❌ streamlit NOT installed")

# Check 4: Environment variables (API keys)
print("\n[4] Checking API keys from .env...")
try:
    from dotenv import load_dotenv
    load_dotenv()
    
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if hf_token:
        print(f"    ✅ HUGGINGFACE_TOKEN: ...{hf_token[-4:]}")
    else:
        print("    ❌ HUGGINGFACE_TOKEN missing from .env")
    
    if groq_key:
        print(f"    ✅ GROQ_API_KEY: ...{groq_key[-4:]}")
    else:
        print("    ❌ GROQ_API_KEY missing from .env")
except ImportError:
    print("    ❌ python-dotenv NOT installed")

# Check 5: VADER actually works
print("\n[5] Testing VADER sentiment (offline, instant)...")
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    result = analyzer.polarity_scores("Revenue exceeded expectations by 12%")
    print(f"    ✅ VADER output: {result}")
except Exception as e:
    print(f"    ❌ VADER error: {e}")

print("\n" + "=" * 60)
print("PHASE 0 CHECK COMPLETE")
print("=" * 60)
print("\nIf everything above shows ✅, you're ready for Phase 1!")