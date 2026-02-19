from __future__ import annotations

import numpy as np
import pandas as pd

from src import model


class DummyModel:
    def __init__(self, proba_pos: list[float]):
        self._proba_pos = np.array(proba_pos, dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        probs = self._proba_pos[:n]
        return np.column_stack([1.0 - probs, probs])


def test_predict_with_artifact_model_returns_float_probabilities_and_int_labels(monkeypatch):
    monkeypatch.setattr(model, "ARTIFACT_MODEL", DummyModel([0.2, 0.8, 0.5]))
    X = pd.DataFrame({"x": [1, 2, 3]})

    proba, labels = model.predict_with_artifact_model(X=X, threshold=0.5)

    assert proba == [0.2, 0.8, 0.5]
    assert labels == [0, 1, 1]
    assert all(isinstance(v, float) for v in proba)
    assert all(isinstance(v, int) for v in labels)


def test_predict_with_artifact_model_threshold_is_applied(monkeypatch):
    monkeypatch.setattr(model, "ARTIFACT_MODEL", DummyModel([0.69, 0.70, 0.71]))
    X = pd.DataFrame({"x": [1, 2, 3]})

    _, labels = model.predict_with_artifact_model(X=X, threshold=0.7)

    assert labels == [0, 1, 1]
