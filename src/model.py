from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

DEFAULT_HF_REPO_ID = "leskimou/openclassrooms_project5_model"
HF_MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", DEFAULT_HF_REPO_ID)
HF_MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "model.joblib")
HF_MODEL_REVISION = os.getenv("HF_MODEL_REVISION")


def _load_model_from_hf() -> Any:
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub is not installed")

    model_path = hf_hub_download(
        repo_id=HF_MODEL_REPO_ID,
        filename=HF_MODEL_FILENAME,
        revision=HF_MODEL_REVISION,
    )
    return joblib.load(model_path)


ARTIFACT_MODEL: Any | None = None


def get_artifact_model() -> Any:
    global ARTIFACT_MODEL
    if ARTIFACT_MODEL is None:
        ARTIFACT_MODEL = _load_model_from_hf()
    return ARTIFACT_MODEL

def predict_with_artifact_model(
    X: pd.DataFrame,
    threshold: float = 0.5,
) -> tuple[list[float], list[int]]:
    model = get_artifact_model()
    proba = model.predict_proba(X)[:, 1]
    labels = (proba >= threshold).astype(int)
    return [float(p) for p in proba], [int(l) for l in labels]