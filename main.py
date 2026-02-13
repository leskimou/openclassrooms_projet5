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
from pathlib import Path
from typing import Dict, List, Union, Any

import joblib
import pandas as pd
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------
# 1) Chargement des artifacts
# ---------------------------------------------------------------------

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
SCHEMA_PATH = ARTIFACTS_DIR / "input_schema.json"

# Charge le modèle une seule fois au démarrage de l’app
model = joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------
# 2) Schémas Pydantic pour l’API /predict
# ---------------------------------------------------------------------

# Valeurs acceptées dans le JSON d'entrée
Value = Union[str, int, float, bool, None]


class PredictRequest(BaseModel):
    # "extra=forbid" => refuse les champs inattendus au niveau du modèle Pydantic
    model_config = ConfigDict(extra="forbid")
    # records = liste de lignes (chaque ligne = dict feature->value)
    records: List[Dict[str, Value]] = Field(..., min_length=1)


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
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema introuvable: {schema_path}. "
            f"Crée {schema_path.as_posix()} pour définir les champs Gradio."
        )

    raw = json.loads(schema_path.read_text(encoding="utf-8"))

    if "features" not in raw or not isinstance(raw["features"], dict):
        raise ValueError('input_schema.json doit contenir un objet "features".')

    # Normalisation
    raw.setdefault("columns_order", list(raw["features"].keys()))
    return raw


def _build_schema_table(cols: list[str], features: dict[str, Any]) -> pd.DataFrame:
    """
    Construit un petit tableau (variable, type, modalités) affiché dans Gradio.
    """
    rows = []
    for col in cols:
        f = features[col]
        f_type = str(f.get("type", "text"))
        mods = f.get("choices", []) if f_type == "category" else []
        rows.append(
            {
                "variable": col,
                "type": f_type,
                "modalités possibles": ", ".join(map(str, mods)) if mods else "",
            }
        )
    return pd.DataFrame(rows)


def _expected_model_columns() -> list[str]:
    """
    Récupère les colonnes attendues par le modèle.

    Quand on entraîne une pipeline scikit-learn avec un DataFrame, certains estimateurs
    exposent feature_names_in_. Ça permet de reconstruire X dans le bon ordre.
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

    # Nettoyage léger : convertit "" -> None (souvent plus propre pour les pipelines)
    cleaned = {k: (None if v == "" else v) for k, v in record.items()}

    if expected:
        aligned = {col: cleaned.get(col, None) for col in expected}
        return pd.DataFrame([aligned], columns=expected)

    return pd.DataFrame([cleaned])


# ---------------------------------------------------------------------
# 4) App FastAPI
# ---------------------------------------------------------------------
app = FastAPI()

@app.get("/health")
def health():
    # Endpoint simple pour vérifier que l'app tourne
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    # Convertit la liste de records en DataFrame en respectant les colonnes attendues
    expected = _expected_model_columns()
    if expected:
        # On aligne chaque record sur les colonnes attendues
        rows = [{col: rec.get(col, None) for col in expected} for rec in payload.records]
        X = pd.DataFrame(rows, columns=expected)
    else:
        # Fallback : DataFrame direct (moins sûr si la pipeline attend un schéma précis)
        X = pd.DataFrame(payload.records)

    proba = model.predict_proba(X)[:, 1]
    labels = (proba >= 0.5).astype(int).tolist()

    return PredictResponse(
        proba_leave=[float(p) for p in proba],
        label=[int(l) for l in labels],
    )


# ---------------------------------------------------------------------
# 5) UI Gradio (1 champ par variable, ordre via columns_order)
# ---------------------------------------------------------------------

def _build_gradio_app() -> gr.Blocks:
    schema = _load_input_schema()
    features: dict[str, Any] = schema["features"]

    # Ordre d’affichage = columns_order (et on garde seulement celles présentes dans features)
    columns_order = schema.get("columns_order") or list(features.keys())
    cols_ui = [c for c in columns_order if c in features]

    # (Optionnel) Vérif : est-ce que le schéma couvre les colonnes attendues par le modèle ?
    # Ici on ne bloque pas forcément, mais tu peux décider de lever une erreur si tu veux.
    expected = _expected_model_columns()
    if expected:
        missing_in_schema = [c for c in expected if c not in features]
        if missing_in_schema:
            # On print pour aider au debug (visible dans la console uvicorn)
            print(
                "[WARN] input_schema.json ne contient pas toutes les colonnes attendues par le modèle : "
                + ", ".join(missing_in_schema)
            )

    # Defaults UI (puisqu’il n’y a pas de "default" dans le JSON)
    defaults: dict[str, Any] = {}
    for c in cols_ui:
        t = str(features[c].get("type", "text"))
        if t == "category":
            ch = features[c].get("choices", []) or []
            defaults[c] = ch[0] if ch else None
        elif t == "number":
            defaults[c] = None
        else:
            defaults[c] = ""

    schema_df = _build_schema_table(cols_ui, features)

    def predict_from_form(*values):
        # Reconstruit un record à partir des champs Gradio
        record = {col: val for col, val in zip(cols_ui, values)}

        # Aligne ce record sur les colonnes attendues par le modèle
        X = _to_model_dataframe(record)

        # Prédiction
        proba = float(model.predict_proba(X)[0, 1])
        label = int(proba >= 0.5)
        return proba, label

    with gr.Blocks() as demo:
        gr.Markdown("## Prédiction de départ (UI Gradio)")
        gr.Markdown("### Schéma des entrées (type + modalités)")
        gr.Dataframe(schema_df, interactive=False, wrap=True)

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
            else:
                inputs.append(gr.Textbox(value=str(defaults[col]), label=col))

        btn = gr.Button("Prédire")
        out_proba = gr.Number(label="Probabilité de départ (classe 1)")
        out_label = gr.Number(label="Label (seuil 0.5)")

        btn.click(fn=predict_from_form, inputs=inputs, outputs=[out_proba, out_label])

    return demo

# Monte Gradio dans FastAPI sous /gradio
gradio_app = _build_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")