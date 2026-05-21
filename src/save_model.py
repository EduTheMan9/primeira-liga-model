"""Fit the Dixon-Coles model on all available data and save it to disk."""

import pickle
import sys
from pathlib import Path

# Make `src` importable as a package
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.dixon_coles import fit_dixon_coles

DATA_DIR = Path(__file__).parent.parent / "data"

df = pd.read_parquet(DATA_DIR / "matches.parquet")
model = fit_dixon_coles(df, xi=0.0)

with open(DATA_DIR / "fitted_model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"Saved fitted model with {len(model.teams)} teams.")