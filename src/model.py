from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import joblib
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from skopt.space import Categorical, Integer, Real

from .utils import opti_pipeline

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "model.joblib"


def build_training_data(data_dir: Path = DATA_DIR) -> Tuple[pd.DataFrame, pd.Series, list[str]]:
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

    target_col = "a_quitte_l_entreprise"
    X = merge_df.drop(columns=[target_col])
    y = merge_df[target_col].map({"Non": 0, "Oui": 1}).astype(int)

    col_to_drop_after = [
        "annees_dans_l_entreprise",
        "annee_experience_totale",
        "annes_sous_responsable_actuel",
        "niveau_hierarchique_poste",
    ]
    X.drop(columns=col_to_drop_after, inplace=True)

    num_features = X.select_dtypes(include=["number"]).columns.tolist()
    return X, y, num_features


def train_model(data_dir: Path = DATA_DIR) -> Any:
    X, y, num_features = build_training_data(data_dir=data_dir)

    encoder_config = {
        "onehot": ["genre"],
        "ordinal": ["heure_supplementaires", "frequence_deplacement"],
        "target": ["domaine_etude", "poste", "departement", "statut_marital"],
    }

    ordinal_categories_map = {
        "heure_supplementaires": ["Non", "Oui"],
        "frequence_deplacement": ["Aucun", "Occasionnel", "Frequent"],
    }

    search_space = {
        "model__n_estimators": Integer(100, 800),
        "model__max_depth": Integer(2, 50),
        "model__min_samples_split": Integer(2, 20),
        "model__min_samples_leaf": Integer(1, 15),
        "model__max_features": Real(0.2, 1.0),
        "model__bootstrap": Categorical([True, False]),
        "model__class_weight": Categorical(["balanced", "balanced_subsample"]),
    }

    summary, bayes, *_ = opti_pipeline(
        model=BalancedRandomForestClassifier(random_state=42),
        X=X,
        y=y,
        search_space=search_space,
        encoder_config=encoder_config,
        ordinal_categories_map=ordinal_categories_map,
        numerical_features=num_features,
        n_splits=5,
        k_best="all",
        holdout_size=0.2,
        seuil_decision=0.50,
        scoring_label="score_pondere",
        optimiser_seuil_cv=False,
        score_seuil_cv="score_pondere",
        afficher_confusion_matrix=True,
    )

    estimator = bayes.best_estimator_
    return estimator


def save_model(model: Any, model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> Any:
    return joblib.load(model_path)


if __name__ == "__main__":
    model = train_model()
    path = save_model(model)
    print(f"Model saved to: {path}")