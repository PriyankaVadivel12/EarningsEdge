"""
upload_model.py
Upload the fine-tuned DistilBERT to Hugging Face Hub so the
deployed Space can load it.

Run once: python src/upload_model.py
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

from data_utils import PROJECT_ROOT

load_dotenv()
token = os.getenv("HUGGINGFACE_TOKEN")
if not token:
    raise SystemExit("HUGGINGFACE_TOKEN not found in .env")

# -----------------------------------------------------------------
# Configuration — change YOUR_USERNAME to your HF username
# -----------------------------------------------------------------
HF_USERNAME = "PriyankaVadivel"  
REPO_NAME = "earningsedge-distilbert"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

MODEL_DIR = PROJECT_ROOT / "models" / "distilbert_finetuned" / "final"

print(f"Uploading model from: {MODEL_DIR}")
print(f"To Hugging Face repo: {REPO_ID}")

# Verify essential model files exist
# Different tokenizer versions save different files, so we check for any valid combo
if not (MODEL_DIR / "config.json").exists():
    raise SystemExit("Missing config.json — model not properly saved")
if not (MODEL_DIR / "tokenizer_config.json").exists():
    raise SystemExit("Missing tokenizer_config.json — tokenizer not properly saved")

# Check for at least one of the tokenizer vocab formats
vocab_candidates = ["vocab.txt", "tokenizer.json", "tokenizer.model", "vocab.json"]
has_vocab = any((MODEL_DIR / v).exists() for v in vocab_candidates)
if not has_vocab:
    raise SystemExit(f"No tokenizer vocab file found (expected one of: {vocab_candidates})")

# Check for model weights (either .bin or .safetensors format)
weight_candidates = ["model.safetensors", "pytorch_model.bin"]
has_weights = any((MODEL_DIR / w).exists() for w in weight_candidates)
if not has_weights:
    raise SystemExit(f"No model weights found (expected one of: {weight_candidates})")

# Show what we're about to upload
print("\nFiles to upload:")
for f in sorted(MODEL_DIR.iterdir()):
    size_mb = f.stat().st_size / 1e6
    print(f"   {f.name}  ({size_mb:.1f} MB)")

# Create the repo (or skip if it already exists)
api = HfApi(token=token)
try:
    create_repo(REPO_ID, token=token, exist_ok=True, repo_type="model")
    print(f"✅ Repo {REPO_ID} ready")
except Exception as e:
    print(f"Repo creation: {e}")

# Upload everything in the final/ folder
print("\nUploading files...")
api.upload_folder(
    folder_path=str(MODEL_DIR),
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Initial upload: fine-tuned DistilBERT for financial sentiment",
)

print(f"\n✅ Model uploaded successfully")
print(f"View it at: https://huggingface.co/{REPO_ID}")