from pathlib import Path
from unittest.mock import MagicMock
import pandas as pd
from src import create_db


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
        }
    )
    sondage_df = pd.DataFrame({"code_sondage": [1, 2]})
    return eval_df, sirh_df, sondage_df


def test_build_dataset(monkeypatch):
    eval_df, sirh_df, sondage_df = _mock_input_frames()

    def fake_read_csv(path):
        p = str(path)
        if "eval" in p:
            return eval_df.copy()
        if "sirh" in p:
            return sirh_df.copy()
        if "sondage" in p:
            return sondage_df.copy()
        raise AssertionError("unexpected file")

    monkeypatch.setattr(create_db.pd, "read_csv", fake_read_csv)
    result = create_db.build_dataset(Path("dummy"))

    assert "diff_salaire_vs_niveau_pct" in result.columns
    assert "ratio_salaire_anciennete" in result.columns
    assert result["augementation_salaire_precedente"].tolist() == [10, 5]


def test_get_engine_from_env(monkeypatch):
    monkeypatch.setenv("ENV_FILE", "custom.env")
    monkeypatch.setenv("DB_USER", "u")
    monkeypatch.setenv("DB_PASSWORD", "p")
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.setenv("DB_NAME", "db")

    load_dotenv_mock = MagicMock()
    create_engine_mock = MagicMock(return_value="engine")

    monkeypatch.setattr(create_db, "load_dotenv", load_dotenv_mock)
    monkeypatch.setattr(create_db, "create_engine", create_engine_mock)

    assert create_db.get_engine_from_env() == "engine"
    create_engine_mock.assert_called_once()


def test_init_api_logging_tables(monkeypatch):
    create_all_mock = MagicMock()
    monkeypatch.setattr(create_db._metadata, "create_all", create_all_mock)
    create_db.init_api_logging_tables("engine")
    create_all_mock.assert_called_once_with("engine")


def test_log_request_and_prediction(monkeypatch):
    conn = MagicMock()
    engine = MagicMock()
    engine.begin.return_value.__enter__.return_value = conn
    monkeypatch.setattr(create_db.uuid, "uuid4", MagicMock(side_effect=["req", "pred"]))

    req_id, pred_id = create_db.log_request_and_prediction(
        engine,
        endpoint="/predict",
        payload={"records": [{"x": 1}]},
        proba_leave=[0.7],
        label=[1],
    )

    assert req_id == "req"
    assert pred_id == "pred"
    assert conn.execute.call_count == 2


def test_full_dataset_to_bdd(monkeypatch):
    df = pd.DataFrame({"a": [1]})
    df.to_sql = MagicMock()

    monkeypatch.setattr(create_db, "build_dataset", MagicMock(return_value=df))
    monkeypatch.setattr(create_db, "get_engine_from_env", MagicMock(return_value="engine"))

    out = create_db.full_dataset_to_bdd(Path("d"), "dataset_final")
    assert out is df
    df.to_sql.assert_called_once()