from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src import utils


class FakeIsolationForest:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_predict(self, X):
        return np.ones(len(X), dtype=int)


class FakePipeline:
    def __init__(self, steps):
        self.steps = steps
        self.fitted_ = False

    def fit(self, X, y):
        self.fitted_ = True
        return self

    def predict_proba(self, X):
        n = len(X)
        probs = np.array([0.2 if i % 2 == 0 else 0.8 for i in range(n)], dtype=float)
        return np.column_stack([1.0 - probs, probs])


class FakeBayesSearchCV:
    def __init__(self, estimator, **kwargs):
        self.estimator = estimator
        self.kwargs = kwargs

    def fit(self, X, y):
        self.best_estimator_ = self.estimator
        self.best_params_ = {"dummy_param": 1}
        self.best_score_ = 0.42
        return self


class FakeBayesSearchCVNoProba:
    def __init__(self, estimator, **kwargs):
        self.estimator = estimator
        self.kwargs = kwargs

    def fit(self, X, y):
        class NoProbaEstimator:
            def fit(self, X, y):
                return self

        self.best_estimator_ = NoProbaEstimator()
        self.best_params_ = {"dummy_param": 1}
        self.best_score_ = 0.42
        return self


def _dataset_binary():
    X = pd.DataFrame(
        {
            "genre": ["H", "F", "H", "F", "H", "F", "H", "F"],
            "heure_supplementaires": ["Non", "Oui", "Non", "Oui", "Non", "Oui", "Non", "Oui"],
            "frequence_deplacement": ["Aucun", "Frequent", "Aucun", "Frequent", "Aucun", "Frequent", "Aucun", "Frequent"],
            "domaine_etude": ["Math", "Info", "Math", "Info", "Math", "Info", "Math", "Info"],
            "poste": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "departement": ["IT", "RH", "IT", "RH", "IT", "RH", "IT", "RH"],
            "statut_marital": ["Single", "Married", "Single", "Married", "Single", "Married", "Single", "Married"],
            "revenu_mensuel": [3000, 5000, 3200, 5100, 3100, 5050, 3300, 5200],
            "diff_salaire_vs_niveau_pct": [-0.2, 0.2, -0.1, 0.1, -0.15, 0.15, -0.05, 0.05],
            "ratio_salaire_anciennete": [600, 500, 610, 490, 605, 495, 615, 485],
            "augementation_salaire_precedente": [10, 5, 12, 4, 11, 6, 9, 5],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], name="target")
    return X, y

def _config():
    encoder_config = {
        "onehot": ["genre"],
        "ordinal": ["heure_supplementaires", "frequence_deplacement"],
        "target": ["domaine_etude", "poste", "departement", "statut_marital"],
    }
    ordinal_categories_map = {
        "heure_supplementaires": ["Non", "Oui"],
        "frequence_deplacement": ["Aucun", "Occasionnel", "Frequent"],
    }
    numerical_features = [
        "revenu_mensuel",
        "diff_salaire_vs_niveau_pct",
        "ratio_salaire_anciennete",
        "augementation_salaire_precedente",
    ]
    return encoder_config, ordinal_categories_map, numerical_features


def _patch_common(monkeypatch):
    monkeypatch.setattr(utils, "Pipeline", FakePipeline)
    monkeypatch.setattr(utils, "clone", lambda obj: obj)
    monkeypatch.setattr(utils, "IsolationForest", FakeIsolationForest)
    monkeypatch.setattr(utils.ConfusionMatrixDisplay, "from_predictions", MagicMock())

    monkeypatch.setattr(utils.plt, "figure", MagicMock())
    monkeypatch.setattr(utils.plt, "plot", MagicMock())
    monkeypatch.setattr(utils.plt, "scatter", MagicMock())
    monkeypatch.setattr(utils.plt, "xlabel", MagicMock())
    monkeypatch.setattr(utils.plt, "ylabel", MagicMock())
    monkeypatch.setattr(utils.plt, "title", MagicMock())
    monkeypatch.setattr(utils.plt, "grid", MagicMock())
    monkeypatch.setattr(utils.plt, "legend", MagicMock())
    monkeypatch.setattr(utils.plt, "tight_layout", MagicMock())
    monkeypatch.setattr(utils.plt, "show", MagicMock())
    monkeypatch.setattr(utils.sns, "heatmap", MagicMock())


def test_score_pondere_scalar():
    score = utils.score_pondere(recall=0.8, f1=0.6, accuracy=0.5)
    assert score == pytest.approx(0.64)


def test_score_pondere_from_pred_binary():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 0]
    score = utils._score_pondere_from_pred(y_true, y_pred)
    assert score == pytest.approx(0.6333333333, rel=1e-6)


@pytest.mark.parametrize(
    "metric,expected_threshold",
    [
        ("recall", 0.0),
        ("precision", 0.5),
        ("score_pondere", 0.0),
        ("f1", 0.0),
    ],
)
def test_best_threshold_from_proba(metric, expected_threshold):
    y_true = np.array([0, 1, 1])
    y_proba = np.array([0.2, 0.6, 0.4])

    threshold, df = utils._best_threshold_from_proba(
        y_true=y_true,
        y_proba_pos=y_proba,
        metric=metric,
        grid_size=3,
    )

    assert threshold == pytest.approx(expected_threshold)
    assert list(df.columns) == ["threshold", "precision", "recall", "f1", "accuracy"]
    assert len(df) == 3


def test_matrice_correlation_runs_plot_pipeline(monkeypatch):
    _patch_common(monkeypatch)
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [2.0, 4.0, 6.0, 8.0],
            "z": [4.0, 1.0, 3.0, 2.0],
        }
    )

    result = utils.matrice_correlation(df, method="spearman", seuil=0.5)

    assert result is None
    utils.plt.figure.assert_called_once()
    utils.sns.heatmap.assert_called_once()
    utils.plt.title.assert_called_once()
    utils.plt.show.assert_called_once()


@pytest.mark.parametrize("oversampler_value", [None, "passthrough", "randomoversampler", "smote"])
def test_evaluate_pipeline_cv_main_paths(monkeypatch, oversampler_value):
    X, y = _dataset_binary()
    encoder_config, ordinal_categories_map, numerical_features = _config()

    _patch_common(monkeypatch)

    summary = utils.evaluate_pipeline_cv(
        model=object(),
        X=X,
        y=y,
        encoder_config=encoder_config,
        ordinal_categories_map=ordinal_categories_map,
        numerical_features=numerical_features,
        scaler="passthrough",
        oversampler=oversampler_value,
        n_splits=2,
        k_best="all",
        holdout_size=0.25,
        seuil_decision=0.5,
        optimiser_seuil_cv=True,
        afficher_courbe_pr=False,
        score_seuil_cv="score_pondere",
        afficher_confusion_matrix=False,
    )

    assert isinstance(summary, pd.DataFrame)
    assert "metric" in summary.columns
    assert "holdout" in summary.columns
    assert "ROC_AUC" in summary["metric"].values
    assert summary.attrs["seuil_source"] == "cv_oof"
    assert "seuil_grid" in summary.attrs


def test_evaluate_pipeline_cv_raises_on_non_binary_target(monkeypatch):
    X, _ = _dataset_binary()
    y_non_binary = pd.Series([0, 1, 2, 0, 1, 2, 0, 2], name="target")
    encoder_config, ordinal_categories_map, numerical_features = _config()

    _patch_common(monkeypatch)

    with pytest.raises(ValueError, match="classification binaire"):
        utils.evaluate_pipeline_cv(
            model=object(),
            X=X,
            y=y_non_binary,
            encoder_config=encoder_config,
            ordinal_categories_map=ordinal_categories_map,
            numerical_features=numerical_features,
            n_splits=2,
            holdout_size=0.5,
            optimiser_seuil_cv=False,
            afficher_courbe_pr=False,
            afficher_confusion_matrix=False,
        )


def test_opti_pipeline_with_cv_threshold(monkeypatch):
    X, y = _dataset_binary()
    encoder_config, ordinal_categories_map, numerical_features = _config()

    _patch_common(monkeypatch)
    monkeypatch.setattr(utils, "BayesSearchCV", FakeBayesSearchCV)

    summary, bayes, X_holdout, y_holdout, X_train_full, y_train_full = utils.opti_pipeline(
        model=object(),
        X=X,
        y=y,
        search_space={"model__max_depth": (2, 4)},
        encoder_config=encoder_config,
        ordinal_categories_map=ordinal_categories_map,
        numerical_features=numerical_features,
        n_splits=2,
        k_best="all",
        holdout_size=0.25,
        seuil_decision=0.5,
        scoring_label="score_pondere",
        optimiser_seuil_cv=True,
        afficher_courbe_pr=False,
        score_seuil_cv="score_pondere",
        afficher_confusion_matrix=False,
    )

    assert isinstance(summary, pd.DataFrame)
    assert "metric" in summary.columns
    assert "holdout" in summary.columns
    assert summary.attrs["best_params"] == {"dummy_param": 1}
    assert summary.attrs["best_score"] == pytest.approx(0.42)
    assert summary.attrs["scoring"] == "score_pondere"
    assert summary.attrs["seuil_source"] == "cv_oof"
    assert len(X_holdout) > 0
    assert len(y_holdout) > 0
    assert len(X_train_full) > 0
    assert len(y_train_full) > 0
    assert bayes.best_score_ == pytest.approx(0.42)


def test_opti_pipeline_fixed_threshold_path(monkeypatch):
    X, y = _dataset_binary()
    encoder_config, ordinal_categories_map, numerical_features = _config()

    _patch_common(monkeypatch)
    monkeypatch.setattr(utils, "BayesSearchCV", FakeBayesSearchCV)

    summary, *_ = utils.opti_pipeline(
        model=object(),
        X=X,
        y=y,
        search_space={"model__max_depth": (2, 4)},
        encoder_config=encoder_config,
        ordinal_categories_map=ordinal_categories_map,
        numerical_features=numerical_features,
        n_splits=2,
        k_best="all",
        holdout_size=0.25,
        seuil_decision=0.6,
        scoring_label="score_pondere",
        optimiser_seuil_cv=False,
        afficher_courbe_pr=False,
        score_seuil_cv="score_pondere",
        afficher_confusion_matrix=False,
    )

    assert summary.attrs["seuil_source"] == "fixed"
    assert summary.attrs["seuil_grid"] is None
    assert summary.attrs["seuil_decision"] == pytest.approx(0.6)


def test_opti_pipeline_raises_if_no_predict_proba(monkeypatch):
    X, y = _dataset_binary()
    encoder_config, ordinal_categories_map, numerical_features = _config()

    _patch_common(monkeypatch)
    monkeypatch.setattr(utils, "BayesSearchCV", FakeBayesSearchCVNoProba)

    with pytest.raises(ValueError, match="predict_proba"):
        utils.opti_pipeline(
            model=object(),
            X=X,
            y=y,
            search_space={"model__max_depth": (2, 4)},
            encoder_config=encoder_config,
            ordinal_categories_map=ordinal_categories_map,
            numerical_features=numerical_features,
            n_splits=2,
            k_best="all",
            holdout_size=0.25,
            seuil_decision=0.5,
            scoring_label="score_pondere",
            optimiser_seuil_cv=True,
            afficher_courbe_pr=False,
            score_seuil_cv="score_pondere",
            afficher_confusion_matrix=False,
        )