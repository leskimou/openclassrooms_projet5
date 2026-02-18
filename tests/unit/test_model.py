from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier

from src import model


def _mock_input_frames():
    eval_df = pd.DataFrame({"eval_number": ["E_1", "E_2"]})

    sirh_df = pd.DataFrame(
        {
            "id_employee": [1, 2],
            "augementation_salaire_precedente": ["10%", "5%"],
            "nombre_heures_travailless": [40, 38],
            "nombre_employee_sous_responsabilite": [2, 1],
            "ayant_enfants": ["Oui", "Non"],
            "satisfaction_employee_environnement": [3, 4],
            "satisfaction_employee_nature_travail": [4, 4],
            "satisfaction_employee_equipe": [3, 5],
            "satisfaction_employee_equilibre_pro_perso": [2, 4],
            "note_evaluation_actuelle": [2, 3],
            "niveau_hierarchique_poste": ["N1", "N1"],
            "revenu_mensuel": [3000, 5000],
            "annee_experience_totale": [4, 9],
            "annees_dans_l_entreprise": [2, 6],
            "annes_sous_responsable_actuel": [1, 2],
            "a_quitte_l_entreprise": ["Non", "Oui"],
        }
    )

    sondage_df = pd.DataFrame({"code_sondage": [1, 2]})
    return eval_df, sirh_df, sondage_df


def test_build_training_data(monkeypatch):
    eval_df, sirh_df, sondage_df = _mock_input_frames()

    def fake_read_csv(path):
        p = str(path)
        if "eval" in p:
            return eval_df.copy()
        if "sirh" in p:
            return sirh_df.copy()
        if "sondage" in p:
            return sondage_df.copy()
        raise AssertionError(f"Unexpected file: {path}")

    monkeypatch.setattr(model.pd, "read_csv", fake_read_csv)

    X, y, num_features = model.build_training_data(data_dir=Path("dummy"))

    assert y.tolist() == [0, 1]
    assert "a_quitte_l_entreprise" not in X.columns
    assert "annees_dans_l_entreprise" not in X.columns
    assert "annee_experience_totale" not in X.columns
    assert "niveau_hierarchique_poste" not in X.columns
    assert "diff_salaire_vs_niveau_pct" in X.columns
    assert "ratio_salaire_anciennete" in X.columns
    assert "revenu_mensuel" in num_features
    assert "augementation_salaire_precedente" in num_features


def test_train_model_calls_opti_pipeline_and_returns_best_estimator(monkeypatch):
    X = pd.DataFrame(
        {
            "genre": ["H", "F"],
            "heure_supplementaires": ["Non", "Oui"],
            "frequence_deplacement": ["Aucun", "Frequent"],
            "domaine_etude": ["Math", "Info"],
            "poste": ["Analyst", "Manager"],
            "departement": ["IT", "RH"],
            "statut_marital": ["Single", "Married"],
            "revenu_mensuel": [3000, 5000],
            "diff_salaire_vs_niveau_pct": [-0.25, 0.25],
            "ratio_salaire_anciennete": [600.0, 500.0],
            "augementation_salaire_precedente": [10, 5],
        }
    )
    y = pd.Series([0, 1], name="a_quitte_l_entreprise")
    num_features = [
        "revenu_mensuel",
        "diff_salaire_vs_niveau_pct",
        "ratio_salaire_anciennete",
        "augementation_salaire_precedente",
    ]

    monkeypatch.setattr(model, "build_training_data", MagicMock(return_value=(X, y, num_features)))

    best_estimator = object()
    fake_bayes = SimpleNamespace(best_estimator_=best_estimator)
    opti_mock = MagicMock(return_value=(fake_bayes, None, None))
    monkeypatch.setattr(model, "opti_pipeline", opti_mock)

    estimator = model.train_model(data_dir=Path("dummy"))

    assert estimator is best_estimator
    assert opti_mock.call_count == 1
    kwargs = opti_mock.call_args.kwargs
    assert isinstance(kwargs["model"], BalancedRandomForestClassifier)
    assert kwargs["numerical_features"] == num_features
    assert kwargs["n_splits"] == 5
    assert kwargs["scoring_label"] == "score_pondere"


def test_save_model_creates_parent_and_calls_joblib_dump(monkeypatch, tmp_path):
    model_obj = {"name": "dummy"}
    model_path = tmp_path / "nested" / "model.joblib"

    dump_mock = MagicMock()
    monkeypatch.setattr(model.joblib, "dump", dump_mock)

    returned_path = model.save_model(model_obj, model_path=model_path)

    assert returned_path == model_path
    assert model_path.parent.exists()
    dump_mock.assert_called_once_with(model_obj, model_path)


def test_load_model_calls_joblib_load(monkeypatch, tmp_path):
    model_path = tmp_path / "model.joblib"
    expected = {"loaded": True}

    load_mock = MagicMock(return_value=expected)
    monkeypatch.setattr(model.joblib, "load", load_mock)

    loaded = model.load_model(model_path=model_path)

    assert loaded == expected
    load_mock.assert_called_once_with(model_path)