"""
visualize_baselines.py
Generate visualization comparing all baselines.
Produces:
- results/baseline_comparison.png (bar chart)
- results/confusion_matrices.png (side-by-side heatmaps)
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from data_utils import RESULTS_DIR, LABEL_NAMES

# -----------------------------------------------------------------
# Load all baseline metrics
# -----------------------------------------------------------------
def load_metrics(model_name: str) -> dict:
    path = RESULTS_DIR / f"baseline_{model_name}.json"
    with open(path) as f:
        return json.load(f)

vader = load_metrics("vader")
finbert = load_metrics("finbert")
llama = load_metrics("llama")

# Majority baseline (no JSON, computed manually)
majority = {
    "model": "majority",
    "accuracy": 0.6136,
    "f1_weighted": 0.4651,  # F1 weighted when always predicting neutral
    "f1_macro": 0.2538,     # F1 macro when always predicting neutral
}

models = [majority, vader, finbert, llama]
model_labels = ["Majority\nbaseline", "VADER", "FinBERT\n(contam ⚠️)", "Llama 3.1 70B"]

# -----------------------------------------------------------------
# Chart 1: Bar chart of overall metrics
# -----------------------------------------------------------------
print("Generating overall metrics bar chart...")

metrics_to_plot = ["accuracy", "f1_weighted", "f1_macro"]
metric_labels = ["Accuracy", "F1 (weighted)", "F1 (macro)"]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 6))

colors = ["#7f7f7f", "#1f77b4", "#2ca02c"]
for i, (metric, label, color) in enumerate(zip(metrics_to_plot, metric_labels, colors)):
    values = [m[metric] * 100 for m in models]
    bars = ax.bar(x + i * width - width, values, width, label=label, color=color, alpha=0.85)
    
    # Add value labels on top of bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("EarningsEdge — Baseline Comparison on FinancialPhraseBank Test Set", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=10)
ax.set_ylim(0, 105)
ax.legend(loc="lower right", framealpha=0.95)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

# Add note about contamination
ax.text(
    0.5, -0.15,
    "⚠️ FinBERT was fine-tuned on FinancialPhraseBank — its accuracy reflects test/train overlap.",
    transform=ax.transAxes,
    ha="center",
    fontsize=9,
    style="italic",
    color="gray",
)

plt.tight_layout()
output_path = RESULTS_DIR / "baseline_comparison.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"✅ Saved: {output_path.relative_to(RESULTS_DIR.parent)}")
plt.close()

# -----------------------------------------------------------------
# Chart 2: Side-by-side confusion matrices for FinBERT and Llama
# -----------------------------------------------------------------
print("Generating confusion matrix comparison...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, model_data, title in zip(
    axes,
    [finbert, llama],
    ["FinBERT (97.64%)", "Llama 3.1 70B (97.64%)"],
):
    cm = np.array(model_data["confusion_matrix"])
    
    # Use a clean colormap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=ax,
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
    )
    
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

fig.suptitle(
    "Identical Accuracy, Different Errors\n"
    "FinBERT and Llama disagree on which sentences are hard (0/8 error overlap)",
    fontsize=13,
    fontweight="bold",
)

plt.tight_layout()
output_path = RESULTS_DIR / "confusion_matrices.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"✅ Saved: {output_path.relative_to(RESULTS_DIR.parent)}")
plt.close()

# -----------------------------------------------------------------
# Chart 3: Per-class F1 comparison (the most interesting view)
# -----------------------------------------------------------------
print("Generating per-class F1 comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

per_class_data = {}
for label in LABEL_NAMES:
    per_class_data[label] = [
        vader["per_class"][label]["f1"] * 100,
        finbert["per_class"][label]["f1"] * 100,
        llama["per_class"][label]["f1"] * 100,
    ]

x = np.arange(3)  # 3 models (VADER, FinBERT, Llama — skip majority)
width = 0.27

model_short = ["VADER", "FinBERT", "Llama 3.1"]

class_colors = {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}
for i, label in enumerate(LABEL_NAMES):
    values = per_class_data[label]
    bars = ax.bar(x + i * width - width, values, width, label=label, color=class_colors[label], alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val:.1f}", ha="center", fontsize=9)

ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("F1 Score (%)", fontsize=12)
ax.set_title("Per-Class F1 Performance — Where Each Model Excels and Fails", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(model_short, fontsize=11)
ax.set_ylim(0, 110)
ax.legend(title="Class", loc="lower right", framealpha=0.95)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

plt.tight_layout()
output_path = RESULTS_DIR / "per_class_f1.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"✅ Saved: {output_path.relative_to(RESULTS_DIR.parent)}")
plt.close()

print("\n✅ All visualizations generated.")
print(f"Files in: {RESULTS_DIR}")