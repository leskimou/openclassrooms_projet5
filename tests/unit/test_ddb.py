from pathlib import Path
from unittest.mock import MagicMock
import pandas as pd
import pytest
from src import create_db


def _mock_input_frames():
    eval_df = pd.DataFrame({"eval_number": ["E_1", "E_2"]})
    sirh_df = pd.DataFrame(
        {
            "id_employee": [1, 2],
            "augementation_salaire_precedente": ["10%", "5%"],
            "revenu_mensuel": [3000, 5000],
            "annee_experience_totale": [4, 9],
            "annees_dans_l_entreprise": [2, 6],
            "annees_dans_le_poste_actuel": [1, 2],
        }
    )
    sondage_df = pd.DataFrame(
        {
            "code_sondage": [1, 2],
            "nombre_heures_travailless": [40, 38],
            "nombre_employee_sous_responsabilite": [2, 1],
            "ayant_enfants": ["Oui", "Non"],
            "satisfaction_employee_environnement": [3, 4],
            "satisfaction_employee_nature_travail": [4, 4],
            "satisfaction_employee_equipe": [3, 5],
            "satisfaction_employee_equilibre_pro_perso": [2, 4],
            "note_evaluation_actuelle": [2, 3],
            "niveau_hierarchique_poste": [1, 2],
            "annes_sous_responsable_actuel": [1, 2],
        }
    )
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


def test_api_requests_table_uses_feature_columns_not_payload():
    col_names = {col.name for col in create_db.api_requests.columns}

    assert "payload" not in col_names
    assert "note_evaluation_precedente" in col_names


def test_log_request_and_prediction_maps_feature_values(monkeypatch):
    conn = MagicMock()
    engine = MagicMock()
    engine.begin.return_value.__enter__.return_value = conn
    monkeypatch.setattr(create_db.uuid, "uuid4", MagicMock(side_effect=["req", "pred"]))

    payload = {
        "records": [
            {
                "note_evaluation_precedente": 2,
                "genre": "F",
            }
        ]
    }

    create_db.log_request_and_prediction(
        engine,
        endpoint="/predict",
        payload=payload,
        proba_leave=[0.7],
        label=[1],
    )

    request_call = conn.execute.call_args_list[0]
    request_params = request_call.args[0].compile().params

    assert request_params["note_evaluation_precedente"] == 2.0
    assert request_params["genre"] == "F"


def test_api_predictions_table_uses_typed_columns():
    col_names = {col.name for col in create_db.api_predictions.columns}

    assert "prediction_index" in col_names
    assert "proba_leave" in col_names
    assert "label" in col_names


def test_log_request_and_prediction_maps_prediction_values(monkeypatch):
    conn = MagicMock()
    engine = MagicMock()
    engine.begin.return_value.__enter__.return_value = conn
    monkeypatch.setattr(create_db.uuid, "uuid4", MagicMock(side_effect=["req", "pred"]))

    create_db.log_request_and_prediction(
        engine,
        endpoint="/predict",
        payload={"records": [{"note_evaluation_precedente": 2}]},
        proba_leave=[0.7],
        label=[1],
    )

    prediction_call = conn.execute.call_args_list[1]
    prediction_params = prediction_call.args[0].compile().params

    assert prediction_params["prediction_index"] == 0
    assert prediction_params["proba_leave"] == 0.7
    assert prediction_params["label"] == 1


def test_log_request_and_prediction_raises_when_lengths_mismatch(monkeypatch):
    engine = MagicMock()
    monkeypatch.setattr(create_db.uuid, "uuid4", MagicMock(return_value="req"))

    with pytest.raises(ValueError, match="même longueur"):
        create_db.log_request_and_prediction(
            engine,
            endpoint="/predict",
            payload={"records": [{"note_evaluation_precedente": 2}]},
            proba_leave=[0.7, 0.2],
            label=[1],
        )


def test_full_dataset_to_bdd(monkeypatch):
    df = pd.DataFrame({"a": [1]})
    df.to_sql = MagicMock()

    monkeypatch.setattr(create_db, "build_dataset", MagicMock(return_value=df))
    monkeypatch.setattr(create_db, "get_engine_from_env", MagicMock(return_value="engine"))

    out = create_db.full_dataset_to_bdd(Path("d"), "dataset_final")
    assert out is df
    df.to_sql.assert_called_once()


def test_build_features_engineering_variables(monkeypatch):
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

    result = create_db.build_features_engineering_variables(Path("dummy"))

    assert list(result.columns) == ["niveau_hierarchique_poste", "annee_experience_totale"]
    assert result.shape == (2, 2)


def test_features_engineering_variables_to_bdd(monkeypatch):
    df = pd.DataFrame(
        {
            "niveau_hierarchique_poste": ["N1"],
            "annee_experience_totale": [4],
        }
    )
    df.to_sql = MagicMock()

    monkeypatch.setattr(create_db, "build_features_engineering_variables", MagicMock(return_value=df))
    monkeypatch.setattr(create_db, "get_engine_from_env", MagicMock(return_value="engine"))

    out = create_db.features_engineering_variables_to_bdd(Path("d"), "features_engineering_variable")

    assert out is df
    df.to_sql.assert_called_once()


def test_init_feature_tables_if_missing_creates_both_tables(monkeypatch):
    engine = MagicMock()

    inspector = MagicMock()
    inspector.has_table = MagicMock(side_effect=[False, False])
    monkeypatch.setattr(create_db, "inspect", MagicMock(return_value=inspector))

    dataset_df = pd.DataFrame({"a": [1]})
    dataset_df.to_sql = MagicMock()
    monkeypatch.setattr(create_db, "build_dataset", MagicMock(return_value=dataset_df))

    features_df = pd.DataFrame(
        {
            "niveau_hierarchique_poste": ["N1"],
            "annee_experience_totale": [4],
        }
    )
    features_df.to_sql = MagicMock()
    monkeypatch.setattr(
        create_db,
        "build_features_engineering_variables",
        MagicMock(return_value=features_df),
    )

    create_db.init_feature_tables_if_missing(engine)

    dataset_df.to_sql.assert_called_once()
    features_df.to_sql.assert_called_once()


def test_init_feature_tables_if_missing_skips_existing_tables(monkeypatch):
    engine = MagicMock()

    inspector = MagicMock()
    inspector.has_table = MagicMock(side_effect=[True, True])
    monkeypatch.setattr(create_db, "inspect", MagicMock(return_value=inspector))

    build_dataset_mock = MagicMock()
    build_features_mock = MagicMock()
    monkeypatch.setattr(create_db, "build_dataset", build_dataset_mock)
    monkeypatch.setattr(create_db, "build_features_engineering_variables", build_features_mock)

    create_db.init_feature_tables_if_missing(engine)

    build_dataset_mock.assert_not_called()
    build_features_mock.assert_not_called()


def test_build_payload_from_bdd_row(monkeypatch):
    dataset_features = [
        feature
        for feature in create_db.FEATURE_SPECS.keys()
        if feature not in {"niveau_hierarchique_poste", "annee_experience_totale"}
    ]
    dataset_row: dict[str, float | str] = {}
    for feature in dataset_features:
        spec = create_db.FEATURE_SPECS[feature]
        feature_type = str(spec.get("type", "")).strip().lower()
        dataset_row[feature] = 1.0 if feature_type == "number" else "X"

    dataset_row["_drop_col_1"] = 999.0
    dataset_row["_drop_col_2"] = 888.0
    dataset_df = pd.DataFrame([dataset_row])

    features_df = pd.DataFrame(
        [{"niveau_hierarchique_poste": 3, "annee_experience_totale": 7}]
    )

    def fake_read_sql_query(query, con):
        _ = con
        if '"dataset_final"' in query:
            return dataset_df
        if '"features_engineering_variable"' in query:
            return features_df
        raise AssertionError("unexpected query")

    monkeypatch.setattr(create_db.pd, "read_sql_query", fake_read_sql_query)

    payload = create_db.build_payload_from_bdd_row(1, engine="engine")

    assert "records" in payload
    assert len(payload["records"]) == 1
    assert set(payload["records"][0].keys()) == set(create_db.FEATURE_SPECS.keys())
    assert payload["records"][0]["niveau_hierarchique_poste"] == 3.0
    assert payload["records"][0]["annee_experience_totale"] == 7.0


def test_build_payload_from_bdd_row_raises_when_line_missing(monkeypatch):
    def fake_read_sql_query(query, con):
        _ = (query, con)
        return pd.DataFrame()

    monkeypatch.setattr(create_db.pd, "read_sql_query", fake_read_sql_query)

    with pytest.raises(IndexError, match="Aucune ligne trouvée"):
        create_db.build_payload_from_bdd_row(5, engine="engine")


def test_build_payload_from_bdd_row_raises_when_row_number_invalid():
    with pytest.raises(ValueError, match=">= 1"):
        create_db.build_payload_from_bdd_row(0, engine="engine")


def test_build_payload_from_bdd_row_raises_when_features_columns_missing(monkeypatch):
    dataset_features = [
        feature
        for feature in create_db.FEATURE_SPECS.keys()
        if feature not in {"niveau_hierarchique_poste", "annee_experience_totale"}
    ]
    dataset_row = {}
    for feature in dataset_features:
        spec = create_db.FEATURE_SPECS[feature]
        feature_type = str(spec.get("type", "")).strip().lower()
        dataset_row[feature] = 1.0 if feature_type == "number" else "X"
    dataset_row["_drop_col_1"] = 1
    dataset_row["_drop_col_2"] = 2

    dataset_df = pd.DataFrame([dataset_row])
    features_df = pd.DataFrame([{"niveau_hierarchique_poste": 2}])

    def fake_read_sql_query(query, con):
        _ = con
        if '"dataset_final"' in query:
            return dataset_df
        if '"features_engineering_variable"' in query:
            return features_df
        raise AssertionError("unexpected query")

    monkeypatch.setattr(create_db.pd, "read_sql_query", fake_read_sql_query)

    with pytest.raises(KeyError, match="annee_experience_totale"):
        create_db.build_payload_from_bdd_row(1, engine="engine")