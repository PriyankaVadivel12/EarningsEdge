"""
data_utils.py
Reusable helpers for loading and saving project data.
"""
import pandas as pd
from pathlib import Path

# Project paths (resolved relative to the project root, not where the script is run from)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Label mappings (we'll standardize on lowercase strings everywhere)
LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_NAMES)}
ID_TO_LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}


def save_split(df: pd.DataFrame, name: str) -> None:
    """Save a dataframe as a CSV in the data folder."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"   ✅ Saved {len(df)} rows → {path.relative_to(PROJECT_ROOT)}")


def load_split(name: str) -> pd.DataFrame:
    """Load a saved split (train/val/test) from data folder."""
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run load_data.py first.")
    return pd.read_csv(path)