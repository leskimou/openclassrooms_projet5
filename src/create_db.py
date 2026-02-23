from pathlib import Path
import json
import os
import uuid
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
)
from sqlalchemy.engine import URL
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
SCHEMA_PATH = ARTIFACTS_DIR / "input_schema.json"


def _load_feature_specs(schema_path: Path = SCHEMA_PATH) -> dict[str, dict]:
    raw_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    features_raw = raw_schema.get("features", {})

    if not isinstance(features_raw, dict):
        return {}

    normalized_features: dict[str, dict] = {}
    for feature_name, spec in features_raw.items():
        clean_name = str(feature_name).strip()
        normalized_features[clean_name] = spec if isinstance(spec, dict) else {}

    return normalized_features


FEATURE_SPECS = _load_feature_specs()


def _build_request_feature_columns() -> list[Column]:
    columns: list[Column] = []
    for feature_name, spec in FEATURE_SPECS.items():
        feature_type = str(spec.get("type", "")).strip().lower()
        if feature_type == "number":
            columns.append(Column(feature_name, Float, nullable=True))
        else:
            columns.append(Column(feature_name, String, nullable=True))
    return columns


REQUEST_FEATURE_COLUMNS = _build_request_feature_columns()


def _resolve_env_file_path() -> Path | None:
    root_env = ROOT_DIR / ".env"
    if root_env.exists():
        return root_env

    confs_dir = ROOT_DIR / "confs"
    if confs_dir.exists():
        env_candidates = sorted(confs_dir.rglob(".env"))
        if env_candidates:
            return env_candidates[0]

        env_pattern_candidates = sorted(confs_dir.rglob(".env.*"))
        if env_pattern_candidates:
            return env_pattern_candidates[0]

    return None


def build_dataset(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    eval_df = pd.read_csv(data_dir / "extrait_eval.csv")
    sirh_df = pd.read_csv(data_dir / "extrait_sirh.csv")
    sondage_df = pd.read_csv(data_dir / "extrait_sondage.csv")

    eval_df["eval_number"] = eval_df["eval_number"].str.replace("E_", "", regex=False).astype(int)

    merge_df = (
        eval_df.merge(sirh_df, left_on="eval_number", right_on="id_employee", how="inner")
        .merge(sondage_df, left_on="eval_number", right_on="code_sondage", how="inner")
    )
    merge_df.drop(columns=["eval_number", "id_employee", "code_sondage"], inplace=True)

    merge_df["augementation_salaire_precedente"] = (
        merge_df["augementation_salaire_precedente"].str.replace("%", "", regex=False).astype(int)
    )

    col_to_drop = [
        "nombre_heures_travailless",
        "nombre_employee_sous_responsabilite",
        "ayant_enfants",
        "satisfaction_employee_environnement",
        "satisfaction_employee_nature_travail",
        "satisfaction_employee_equipe",
        "satisfaction_employee_equilibre_pro_perso",
        "note_evaluation_actuelle",
    ]
    merge_df.drop(columns=col_to_drop, inplace=True)

    salaire_moyen_par_niveau = merge_df.groupby("niveau_hierarchique_poste")["revenu_mensuel"].mean()
    merge_df["salaire_moyen_niveau"] = merge_df["niveau_hierarchique_poste"].map(salaire_moyen_par_niveau)

    merge_df["diff_salaire_vs_niveau"] = merge_df["revenu_mensuel"] - merge_df["salaire_moyen_niveau"]
    merge_df["diff_salaire_vs_niveau_pct"] = merge_df["diff_salaire_vs_niveau"] / merge_df["salaire_moyen_niveau"]
    merge_df.drop(columns=["salaire_moyen_niveau", "diff_salaire_vs_niveau"], inplace=True)

    merge_df["annee_experience_totale"] = merge_df["annee_experience_totale"] + 1
    merge_df["ratio_salaire_anciennete"] = merge_df["revenu_mensuel"] / merge_df["annee_experience_totale"]

    col_to_drop_after = [
        "annees_dans_l_entreprise",
        "annee_experience_totale",
        "annes_sous_responsable_actuel",
        "niveau_hierarchique_poste",
    ]
    merge_df.drop(columns=col_to_drop_after, inplace=True)

    return merge_df


def get_engine_from_env():
    dotenv_path = _resolve_env_file_path()
    if dotenv_path is not None:
        load_dotenv(dotenv_path=dotenv_path, override=False)

    url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST") or "127.0.0.1",
        port=int(os.getenv("DB_PORT") or 5432),
        database=os.getenv("DB_NAME"),
    )
    return create_engine(url)


# ---------------------------------------------------------------------
# API logging (requêtes + prédictions)
# ---------------------------------------------------------------------

_metadata = MetaData()

api_requests = Table(
    "api_requests",
    _metadata,
    Column("id", String, primary_key=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("endpoint", String, nullable=False),
    *REQUEST_FEATURE_COLUMNS,
)

api_predictions = Table(
    "api_predictions",
    _metadata,
    Column("id", String, primary_key=True),
    Column("request_id", String, ForeignKey("api_requests.id"), nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("prediction_index", Integer, nullable=False),
    Column("proba_leave", Float, nullable=False),
    Column("label", Integer, nullable=False),
)


def init_api_logging_tables(engine) -> None:
    """Crée les tables de logging si elles n'existent pas."""
    _metadata.create_all(engine)


def log_request_and_prediction(
    engine,
    *,
    endpoint: str,
    payload: dict,
    proba_leave: list[float],
    label: list[int],) -> tuple[str, str]:

    request_id = str(uuid.uuid4())
    if len(proba_leave) != len(label):
        raise ValueError("proba_leave et label doivent avoir la même longueur")

    prediction_ids: list[str] = []

    request_insert_values = {
        "id": request_id,
        "endpoint": endpoint,
    }

    records = payload.get("records", []) if isinstance(payload, dict) else []
    first_record = records[0] if records and isinstance(records[0], dict) else {}

    for feature_name, spec in FEATURE_SPECS.items():
        raw_value = first_record.get(feature_name)
        if raw_value is None:
            request_insert_values[feature_name] = None
            continue

        feature_type = str(spec.get("type", "")).strip().lower()
        if feature_type == "number":
            request_insert_values[feature_name] = float(raw_value)
        else:
            request_insert_values[feature_name] = str(raw_value)

    with engine.begin() as conn:
        conn.execute(
            api_requests.insert().values(
                **request_insert_values,
            )
        )
        for idx, (proba_value, label_value) in enumerate(zip(proba_leave, label)):
            prediction_id = str(uuid.uuid4())
            prediction_ids.append(prediction_id)
            conn.execute(
                api_predictions.insert().values(
                    id=prediction_id,
                    request_id=request_id,
                    prediction_index=idx,
                    proba_leave=float(proba_value),
                    label=int(label_value),
                )
            )

    first_prediction_id = prediction_ids[0] if prediction_ids else ""
    return request_id, first_prediction_id


def full_dataset_to_bdd(data_dir: Path = DATA_DIR, table_name: str = "dataset_final") -> pd.DataFrame:
    merge_df = build_dataset(data_dir=data_dir)

    engine = get_engine_from_env()
    merge_df.to_sql(name=table_name, con=engine, if_exists="replace", index=False)

    print(f"Dataset envoyé avec succès dans la table '{table_name}'")
    return merge_df


# Créer une fonction qui fait une requete sql pour aller chercher le salaire et le niveau hiérarchique puis qui va donner le salaire moyen par niveau hiérarchique


if __name__ == "__main__":
    full_dataset_to_bdd()


