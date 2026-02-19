from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
ARTIFACT_MODEL: Any = joblib.load(DEFAULT_MODEL_PATH)

def predict_with_artifact_model(
    X: pd.DataFrame,
    threshold: float = 0.5,
) -> tuple[list[float], list[int]]:
    proba = ARTIFACT_MODEL.predict_proba(X)[:, 1]
    labels = (proba >= threshold).astype(int)
    return [float(p) for p in proba], [int(l) for l in labels]