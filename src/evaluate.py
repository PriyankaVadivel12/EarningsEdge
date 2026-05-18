"""
evaluate.py
Reusable evaluation framework for all sentiment models in this project.

Every model in this project (VADER, FinBERT, Llama, fine-tuned DistilBERT)
will use this same evaluation pipeline, ensuring fair comparison.
"""
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

from data_utils import RESULTS_DIR, LABEL_NAMES


def evaluate_predictions(
    y_true: list,
    y_pred: list,
    model_name: str,
    save_results: bool = True,
) -> dict:
    """
    Compute standard classification metrics and optionally save to JSON.
    
    Args:
        y_true: True labels (list of strings: 'negative', 'neutral', 'positive')
        y_pred: Predicted labels (same format as y_true)
        model_name: Name for saving and printing (e.g., 'vader', 'finbert')
        save_results: If True, save metrics to results/baseline_{model_name}.json
    
    Returns:
        Dictionary of metrics
    """
    # Compute core metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", labels=LABEL_NAMES)
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=LABEL_NAMES)
    
    # Per-class metrics (precision, recall, F1 for each class separately)
    per_class_f1 = f1_score(y_true, y_pred, labels=LABEL_NAMES, average=None)
    per_class_precision = precision_score(y_true, y_pred, labels=LABEL_NAMES, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=LABEL_NAMES, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_NAMES)
    
    # Package everything into a clean dict
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy),
        "f1_weighted": float(f1_weighted),
        "f1_macro": float(f1_macro),
        "per_class": {
            label: {
                "precision": float(per_class_precision[i]),
                "recall": float(per_class_recall[i]),
                "f1": float(per_class_f1[i]),
            }
            for i, label in enumerate(LABEL_NAMES)
        },
        "confusion_matrix": cm.tolist(),
        "n_samples": len(y_true),
    }
    
    # Print a nice summary
    print_metrics(metrics)
    
    # Save to file if requested
    if save_results:
        save_path = RESULTS_DIR / f"baseline_{model_name}.json"
        RESULTS_DIR.mkdir(exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n💾 Metrics saved to {save_path.relative_to(RESULTS_DIR.parent)}")
    
    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty-print metrics to console."""
    print("\n" + "=" * 60)
    print(f"RESULTS: {metrics['model'].upper()}")
    print("=" * 60)
    
    print(f"\nOverall Metrics:")
    print(f"   Accuracy:     {metrics['accuracy']*100:>6.2f}%")
    print(f"   F1 (weighted): {metrics['f1_weighted']*100:>6.2f}%")
    print(f"   F1 (macro):    {metrics['f1_macro']*100:>6.2f}%")
    print(f"   N samples:    {metrics['n_samples']}")
    
    print(f"\nPer-Class Metrics:")
    print(f"   {'Class':>10} {'Precision':>11} {'Recall':>9} {'F1':>7}")
    for label in LABEL_NAMES:
        pc = metrics["per_class"][label]
        print(f"   {label:>10} {pc['precision']*100:>10.2f}% {pc['recall']*100:>8.2f}% {pc['f1']*100:>6.2f}%")
    
    print(f"\nConfusion Matrix (rows=true, cols=predicted):")
    print(f"   {'':>10}", end="")
    for label in LABEL_NAMES:
        print(f"{label:>10}", end="")
    print()
    for i, true_label in enumerate(LABEL_NAMES):
        print(f"   {true_label:>10}", end="")
        for j in range(len(LABEL_NAMES)):
            print(f"{metrics['confusion_matrix'][i][j]:>10}", end="")
        print()