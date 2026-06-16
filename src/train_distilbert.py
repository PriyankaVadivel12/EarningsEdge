"""
train_distilbert.py
Fine-tune DistilBERT on FinancialPhraseBank for sentiment classification.

Run with: python src/train_distilbert.py

Settings:
- Model: distilbert-base-uncased (66M parameters)
- Task: 3-class sentiment (negative / neutral / positive)
- Training: 3 epochs, batch size 16, LR 2e-5
- Output: saves best model to models/distilbert_finetuned/
"""
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from data_utils import DATA_DIR, PROJECT_ROOT, LABEL_NAMES, ID_TO_LABEL

# -----------------------------------------------------------------
# Configuration — change these to experiment
# -----------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
USE_MPS = True          # Try Apple Silicon GPU. Set False if MPS gives errors.
SEED = 42
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100

TOKENIZED_DIR = DATA_DIR / "tokenized"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "distilbert_finetuned"
LOG_DIR = PROJECT_ROOT / "models" / "logs"

# -----------------------------------------------------------------
# Reproducibility: set seeds everywhere
# -----------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

print("=" * 60)
print("PHASE 3.2: FINE-TUNING DISTILBERT")
print("=" * 60)

# -----------------------------------------------------------------
# Device selection
# -----------------------------------------------------------------
if USE_MPS and torch.backends.mps.is_available():
    device = "mps"
    print(f"\n[Device] Using Apple Silicon MPS (faster, may be flaky)")
elif torch.cuda.is_available():
    device = "cuda"
    print(f"\n[Device] Using NVIDIA CUDA")
else:
    device = "cpu"
    print(f"\n[Device] Using CPU (slower but reliable)")

# -----------------------------------------------------------------
# Step 1: Load tokenized data
# -----------------------------------------------------------------
print(f"\n[1] Loading tokenized data from {TOKENIZED_DIR.relative_to(PROJECT_ROOT)}...")
tokenized = load_from_disk(str(TOKENIZED_DIR))
print(f"    Train: {len(tokenized['train'])} examples")
print(f"    Val:   {len(tokenized['validation'])} examples")
print(f"    Test:  {len(tokenized['test'])} examples")

# -----------------------------------------------------------------
# Step 2: Load model
# -----------------------------------------------------------------
print(f"\n[2] Loading DistilBERT with classification head (3 classes)...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID_TO_LABEL,
    label2id={name: i for i, name in ID_TO_LABEL.items()},
)
n_params = sum(p.numel() for p in model.parameters())
print(f"    Total parameters: {n_params:,} (~{n_params/1e6:.1f}M)")

# Move model to device
model = model.to(device)

# Load tokenizer for the data collator
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# -----------------------------------------------------------------
# Step 3: Define metric computation (called after each eval)
# -----------------------------------------------------------------
def compute_metrics(eval_pred):
    """Compute accuracy and F1 from predictions."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

# -----------------------------------------------------------------
# Step 4: Training configuration
# -----------------------------------------------------------------
print(f"\n[3] Setting up training configuration...")
print(f"    Epochs:        {EPOCHS}")
print(f"    Batch size:    {BATCH_SIZE}")
print(f"    Learning rate: {LEARNING_RATE}")
print(f"    Weight decay:  {WEIGHT_DECAY}")
print(f"    Warmup steps:  {WARMUP_STEPS}")

training_args = TrainingArguments(
    output_dir=str(MODEL_OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=WARMUP_STEPS,
    eval_strategy="epoch",            # evaluate after each epoch
    save_strategy="epoch",            # save checkpoint after each epoch
    load_best_model_at_end=True,      # load best model (by val accuracy)
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2,               # only keep last 2 checkpoints (saves disk)
    logging_dir=str(LOG_DIR),
    logging_steps=20,                 # print loss every 20 steps
    report_to="none",                 # don't use wandb/tensorboard for now
    seed=SEED,
    fp16=False,                       # MPS doesn't support fp16
)

# -----------------------------------------------------------------
# Step 5: Build the Trainer
# -----------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# -----------------------------------------------------------------
# Step 6: Train!
# -----------------------------------------------------------------
print(f"\n[4] Starting training...")
print(f"    Output dir: {MODEL_OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
print(f"    This will take ~10-40 minutes depending on device.\n")

train_result = trainer.train()

# -----------------------------------------------------------------
# Step 7: Final evaluation on validation set
# -----------------------------------------------------------------
print("\n[5] Final evaluation on validation set...")
val_metrics = trainer.evaluate()
print(f"\n   Final validation metrics:")
for key in ["eval_accuracy", "eval_f1_weighted", "eval_f1_macro", "eval_loss"]:
    print(f"      {key:>20}: {val_metrics[key]:.4f}")

# -----------------------------------------------------------------
# Step 8: Save the final model explicitly
# -----------------------------------------------------------------
final_dir = MODEL_OUTPUT_DIR / "final"
print(f"\n[6] Saving final model to {final_dir.relative_to(PROJECT_ROOT)}...")
trainer.save_model(str(final_dir))
tokenizer.save_pretrained(str(final_dir))
print(f"    ✅ Saved")

print("\n" + "=" * 60)
print("PHASE 3.2 COMPLETE — Model trained")
print("=" * 60)
print(f"\nFinal validation accuracy: {val_metrics['eval_accuracy']*100:.2f}%")
print(f"Next: run evaluation on test set in Phase 3.3")