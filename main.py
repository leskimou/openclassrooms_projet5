"""
main.py

API FastAPI + UI Gradio
Objectif :
- Exposer une API /predict (JSON) qui appelle un modèle scikit-learn sérialisé en joblib
- Exposer une interface web Gradio avec 1 champ par variable
- Définir l’interface (types, modalités, ordre) via artifacts/input_schema.json

Exécution (exemples) :
- Avec uv : uv run python -m uvicorn main:app --reload
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Union, Any, AsyncIterator

import pandas as pd
import gradio as gr
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.create_db import get_engine_from_env, init_api_logging_tables, log_request_and_prediction
from src.model import ARTIFACT_MODEL, predict_with_artifact_model
from src.utils import preprocess_record_for_model


APP_MODE = os.getenv("APP_MODE", "local").strip().lower()
IS_DEMO_MODE = APP_MODE == "demo"


# ---------------------------------------------------------------------
# 1) Chargement des artifacts
# ---------------------------------------------------------------------

ARTIFACTS_DIR = Path("artifacts")
SCHEMA_PATH = ARTIFACTS_DIR / "input_schema.json"

# Charge le modèle une seule fois au démarrage de l’app
model = ARTIFACT_MODEL


def _normalize_input_schema(raw_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Normalise le schéma d'entrée :
    - corrige la clé "colums_order" -> "columns_order" si besoin
    - supprime les espaces parasites sur les noms de features
    - garantit que columns_order contient toutes les features
    """
    features_raw = raw_schema.get("features", {})
    if not isinstance(features_raw, dict):
        raise ValueError("Le schéma d'entrée est invalide : 'features' doit être un objet.")

    features: dict[str, Any] = {}
    for feature_name, spec in features_raw.items():
        clean_name = str(feature_name).strip()
        features[clean_name] = spec

    columns_order_raw = raw_schema.get("columns_order") or raw_schema.get("colums_order") or list(features.keys())

    columns_order: list[str] = []
    for col in columns_order_raw:
        clean_col = str(col).strip()
        if clean_col in features and clean_col not in columns_order:
            columns_order.append(clean_col)

    for feature_name in features:
        if feature_name not in columns_order:
            columns_order.append(feature_name)

    return {
        "features": features,
        "columns_order": columns_order,
    }


INPUT_SCHEMA: dict[str, Any] = _normalize_input_schema(
    json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
)
FEATURE_SPECS: dict[str, Any] = INPUT_SCHEMA["features"]
REQUIRED_FEATURES_ORDER: list[str] = list(FEATURE_SPECS.keys())
REQUIRED_FEATURES_SET: set[str] = set(REQUIRED_FEATURES_ORDER)


# ---------------------------------------------------------------------
# 2) Schémas Pydantic pour l’API /predict
# ---------------------------------------------------------------------

# Valeurs acceptées dans le JSON d'entrée
Value = Union[str, int, float]

class PredictRequest(BaseModel):
    # "extra=forbid" => refuse les champs inattendus au niveau du modèle Pydantic
    model_config = ConfigDict(extra="forbid")
    # records = liste de lignes (chaque ligne = dict feature->value)
    records: List[Dict[str, Value]] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_records_against_input_schema(self) -> "PredictRequest":
        errors: list[str] = []

        for idx, record in enumerate(self.records):
            missing_features = [feature for feature in REQUIRED_FEATURES_ORDER if feature not in record]
            extra_features = [feature for feature in record if feature not in REQUIRED_FEATURES_SET]

            if missing_features:
                errors.append(f"records[{idx}] features manquantes: {missing_features}")
            if extra_features:
                errors.append(f"records[{idx}] features inattendues: {extra_features}")

            if missing_features:
                # Impossible de vérifier les types pour les champs absents
                continue

            for feature, spec in FEATURE_SPECS.items():
                expected_type = str(spec.get("type", "")).strip().lower()
                value = record[feature]

                if expected_type == "number":
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        errors.append(
                            f"records[{idx}].{feature} doit être un nombre (int|float), reçu {type(value).__name__}"
                        )
                    continue

                if expected_type == "category":
                    if not isinstance(value, str):
                        errors.append(
                            f"records[{idx}].{feature} doit être une chaîne (str), reçu {type(value).__name__}"
                        )
                        continue

                    choices = spec.get("choices", []) or []
                    if choices and value not in choices:
                        errors.append(
                            f"records[{idx}].{feature} doit être dans {choices}, reçu {value!r}"
                        )
                    continue

                errors.append(
                    f"records[{idx}].{feature} a un type de schéma non supporté: {expected_type!r}"
                )

        if errors:
            raise ValueError("Validation payload invalide: " + " | ".join(errors))

        return self


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # proba_leave = probabilité de la classe 1 ("quitte l’entreprise")
    proba_leave: List[float]
    # label = 1 si proba >= 0.5 sinon 0
    label: List[int]


# ---------------------------------------------------------------------
# 3) Helpers : schema d’inputs (pour la UI) + préparation DataFrame
# ---------------------------------------------------------------------

def _load_input_schema(schema_path: Path = SCHEMA_PATH) -> dict[str, Any]:
    _ = schema_path
    return INPUT_SCHEMA


def _expected_model_columns() -> list[str]:
    """
    Récupère les colonnes attendues par le modèle avec feature_names_in_. Ça permet de reconstruire X dans le bon ordre.
    """
    if hasattr(model, "feature_names_in_"):
        return [str(c) for c in list(getattr(model, "feature_names_in_"))]
    return []


def _to_model_dataframe(record: dict[str, Any]) -> pd.DataFrame:
    """
    Transforme un record (dict) en DataFrame prêt pour predict_proba.

    - Si le modèle fournit feature_names_in_ :
        - on ordonne exactement les colonnes comme attendu
        - on ajoute les colonnes manquantes à None
        - on ignore les colonnes en trop
    - Sinon :
        - on crée un DataFrame avec les clés fournies
    """
    expected = _expected_model_columns()

    if expected:
        aligned = {col: record.get(col, None) for col in expected}
        return pd.DataFrame([aligned], columns=expected)

    return pd.DataFrame([record])


# ---------------------------------------------------------------------
# 4) App FastAPI
# ---------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    _ = _app
    global _db_engine
    if IS_DEMO_MODE:
        _db_engine = None
        yield
        return

    try:
        _db_engine = get_engine_from_env()
        init_api_logging_tables(_db_engine)
    except Exception:
        _db_engine = None
    yield


app = FastAPI(lifespan=lifespan)

_db_engine = None

@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/gradio")

@app.get("/health")
def health():
    # Endpoint simple pour vérifier que l'app tourne
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request):
    try:
        processed_records = [preprocess_record_for_model(rec) for rec in payload.records]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Convertit la liste de records en DataFrame en respectant les colonnes attendues
    expected = _expected_model_columns()
    if expected:
        # On aligne chaque record sur les colonnes attendues
        rows = [{col: rec.get(col, None) for col in expected} for rec in processed_records]
        X = pd.DataFrame(rows, columns=expected)
    else:
        X = pd.DataFrame(processed_records)

    proba, labels = predict_with_artifact_model(X=X, threshold=0.5)

    # request + prediction
    if _db_engine is not None:
        log_request_and_prediction(
            _db_engine,
            endpoint=str(request.url.path),
            payload=payload.model_dump(),
            proba_leave=proba,
            label=labels,
        )

    return PredictResponse(
        proba_leave=proba,
        label=labels,
    )


# ---------------------------------------------------------------------
# 5) UI Gradio (1 champ par variable, ordre via columns_order)
# ---------------------------------------------------------------------

def _build_gradio_app() -> gr.Blocks:
    schema = _load_input_schema()
    features: dict[str, Any] = schema["features"]

    # Ordre d’affichage = columns_order (et on garde seulement celles présentes dans features)
    columns_order = schema.get("columns_order")
    cols_ui = [c for c in columns_order if c in features]

    # Defaults UI
    defaults: dict[str, Any] = {}
    for c in cols_ui:
        t = str(features[c].get("type", "text"))
        if t == "category":
            ch = features[c].get("choices", []) or []
            defaults[c] = ch[0] if ch else None
        elif t == "number":
            defaults[c] = None

    def predict_from_form(*values):
        # Reconstruit un record à partir des champs Gradio
        record = {col: val for col, val in zip(cols_ui, values)}

        try:
            processed_record = preprocess_record_for_model(record)
            # Aligne ce record sur les colonnes attendues par le modèle
            X = _to_model_dataframe(processed_record)

            # Prédiction
            proba_list, label_list = predict_with_artifact_model(X=X, threshold=0.5)
            proba = float(proba_list[0])
            label = int(label_list[0])

            # Log en base (mêmes tables que l'API)
            if _db_engine is not None:
                log_request_and_prediction(
                    _db_engine,
                    endpoint="/gradio",
                    payload={"records": [record]},
                    proba_leave=[proba],
                    label=[label],
                )
            return proba, label
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        except Exception as exc:
            raise gr.Error(f"Erreur lors de la prédiction: {exc}") from exc

    with gr.Blocks() as demo:
        gr.Markdown("## Prédiction de départ d'un employé (UI Gradio)")
      
        # Génère 1 composant par variable selon son type
        inputs = []
        for col in cols_ui:
            f = features[col]
            t = str(f.get("type", "text"))

            if t == "number":
                inputs.append(gr.Number(value=defaults[col], label=col))
            elif t == "category":
                # Choix unique parmi des modalités (plus simple et fiable pour ton modèle)
                inputs.append(gr.Dropdown(choices=f.get("choices", []), value=defaults[col], label=col))

        btn = gr.Button("Prédire")
        out_proba = gr.Number(label="Probabilité de départ (classe 1)", precision=8)
        out_label = gr.Number(label="Label (seuil 0.5)")

        btn.click(fn=predict_from_form, inputs=inputs, outputs=[out_proba, out_label])

    return demo

# Monte Gradio dans FastAPI sous /gradio
gradio_app = _build_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")