import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import TargetEncoder, OneHotEncoder, OrdinalEncoder
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.base import clone
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
)

from imblearn.over_sampling import RandomOverSampler, SMOTE

from sklearn.metrics import make_scorer
from skopt import BayesSearchCV

#---------------------------------------------------------------------------------------------------------------------------------

def score_pondere(
    recall,
    f1,
    accuracy,
):
    """
    Calcule un score pondéré entre 0 et 1 :
    score = 0.30 * recall + 0.50 * f1 + 0.20 * accuracy
    """

    return 0.30 * recall + 0.50 * f1 + 0.20 * accuracy

#---------------------------------------------------------------------------------------------------------------------------------
def _score_pondere_from_pred(y_true, y_pred, avg="binary"):
        r = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))
        return float(score_pondere(recall=r, f1=f1, accuracy=acc))

#---------------------------------------------------------------------------------------------------------------------------------

def _best_threshold_from_proba(
    y_true,
    y_proba_pos,
    metric="score_pondere",
    grid_size=101,
):
    """Renvoie le meilleur seuil de décision pour maximiser une métrique en fonction des probabilités prédites sur les out of fold.
        metric:
            - "f1": maximise le F1
            - "recall": maximise le rappel
            - "precision": maximise la précision
            - "score_pondere": maximise notre score pondéré
    """
    thresholds = np.linspace(0.0, 1.0, int(grid_size))
    rows = []
    for t in thresholds:
        y_pred = (y_proba_pos >= t).astype(int)
        p = float(precision_score(y_true, y_pred, zero_division=0))
        r = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))
        rows.append((float(t), p, r, f1, acc))

    df = pd.DataFrame(rows, columns=["threshold", "precision", "recall", "f1", "accuracy"])

    if metric == "recall":
        best = df.sort_values(["recall", "precision", "threshold"], ascending=[False, False, True]).iloc[0]
    elif metric == "precision":
        best = df.sort_values(["precision", "recall", "threshold"], ascending=[False, False, True]).iloc[0]
    elif metric == "score_pondere":
        best = df.assign(
            score_pondere=score_pondere(
                recall=df["recall"],
                f1=df["f1"],
                accuracy=df["accuracy"],
            )
        ).sort_values(["score_pondere", "recall", "f1", "accuracy", "threshold"], ascending=[False, False, False, False, True]).iloc[0]
    else:  # f1
        best = df.sort_values(["f1", "recall", "precision", "threshold"], ascending=[False, False, False, True]).iloc[0]

    return float(best["threshold"]), df


#--------------------------------------------------------------------------------------------------------------------
def matrice_correlation(df, method='spearman', seuil=0.7):
    """
    Affiche la matrice de corrélation (heatmap) et un tableau des corrélations
    strictement supérieures au seuil (en valeur absolue), sans doublons.

    Returns : Tableau des paires de variables dont |corr| > seuil.
    """

    corr = df.corr(method=method, numeric_only=True)

    # cache la diagonale et la partie inférieure
    mask = np.triu(np.ones_like(corr, dtype=bool), k=0)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.title(f"Matrice de corrélation ({method})")
    plt.show()

    # Tableau des corrélations au-dessus du seuil (sans diagonale et sans doublons)
    mask_upper = np.triu(np.ones(corr.shape), k=1).astype(bool)
    corr_upper = corr.where(mask_upper)

    corr_long = (
        corr_upper.stack()
        .reset_index()
        .rename(columns={"level_0": "variable_1", "level_1": "variable_2", 0: "correlation"})
    )

    corr_filtre = (
        corr_long.loc[corr_long["correlation"].abs() > seuil]
        .sort_values("correlation", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )

    if not corr_filtre.empty:
        print(f"Corrélations avec |corr| > {seuil} :")
        print(corr_filtre)

    return

#---------------------------------------------------------------------------------------------------------------------------------

def evaluate_pipeline_cv(model, X, y,
                         encoder_config,
                         ordinal_categories_map,
                         numerical_features,
                         scaler= 'passthrough',
                         oversampler=None,
                         n_splits=5,
                         k_best='all',
                         holdout_size=0.2,
                         seuil_decision=0.5,
                         optimiser_seuil_cv=True,
                         afficher_courbe_pr=True,
                         score_seuil_cv="score_pondere",
                         afficher_confusion_matrix=True):

    # Préprocessing + Modèle
    # Choix encoder catégoriel
    ordinal_cols = encoder_config["ordinal"]
    ordinal_categories = [ordinal_categories_map[col] for col in ordinal_cols]

    ordinal_enc = OrdinalEncoder(categories=ordinal_categories)

    preprocessor = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
            encoder_config["onehot"]),
            ("ordinal", ordinal_enc, ordinal_cols),
            ("target", TargetEncoder(random_state=42), encoder_config["target"]),
            ("num", scaler, numerical_features)
            ],
        )

    # Oversampler
    # "passthrough": pas d'oversampling
    # "randomoversampler": RandomOverSampler
    # "smote": SMOTE
    if oversampler is None or oversampler == "passthrough":
        oversampler_step = "passthrough"
    elif oversampler == "randomoversampler":
        oversampler_step = RandomOverSampler(random_state=42)
    elif oversampler == "smote":
        oversampler_step = SMOTE(random_state=42)

    # Pipeline (imblearn)
    steps = [("preprocessing", preprocessor)]
    if oversampler_step != "passthrough":
        steps.append(("oversampler", oversampler_step))
    steps += [
        ("selectkbest", SelectKBest(k=k_best)),
        ("model", model),
    ]
    pipe = Pipeline(steps)

    # Hold-out (identique entre appels + random_state)
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
        X,
        y,
        test_size=holdout_size,
        random_state=42,
        stratify=y,
    )

    # IsolationForest fit sur tout le train
    X_train_full = X_train_full.copy()
    y_train_full = y_train_full.copy()
    num_if_cols = X_train_full.select_dtypes(include=[np.number]).columns
    n_outliers_train = 0
    if len(num_if_cols) > 0:
        iso = IsolationForest(
            n_estimators=200,
            contamination=0.02,   # 2% d'outliers attendus
            random_state=42,
            n_jobs=-1,
        )
        labels_train = iso.fit_predict(X_train_full[num_if_cols])
        keep_mask = labels_train == 1  # 1 = normal, -1 = outlier
        n_outliers_train = int(np.sum(~keep_mask))
        X_train_full = X_train_full.loc[keep_mask].copy()
        y_train_full = y_train_full.loc[keep_mask].copy()

    if n_outliers_train > 0:
        print(f"IsolationForest: {n_outliers_train} outliers retirés du train.")

    # prédiction binaire uniquement
    if y_train_full.nunique() != 2:
        raise ValueError("Ce projet suppose une classification binaire (y doit contenir exactement 2 classes).")
    avg = "binary"

    # Cross-validation (stratifiée)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {}

    # OOF proba (classe positive) sur le train uniquement
    oof_proba_pos = pd.Series(index=X_train_full.index, dtype=float)

    for train_idx, test_idx in kf.split(X_train_full, y_train_full):
        X_train, X_test = X_train_full.iloc[train_idx].copy(), X_train_full.iloc[test_idx].copy()
        y_train, y_test = y_train_full.iloc[train_idx].copy(), y_train_full.iloc[test_idx].copy()

        p = clone(pipe)
        p.fit(X_train, y_train)

        proba_train_pos = p.predict_proba(X_train)[:, 1]
        proba_test_pos = p.predict_proba(X_test)[:, 1]

        y_pred_train = (proba_train_pos >= float(seuil_decision)).astype(int)
        y_pred_test = (proba_test_pos >= float(seuil_decision)).astype(int)

        # OOF
        oof_proba_pos.iloc[test_idx] = proba_test_pos

        # Métriques
        train_accuracy = float(accuracy_score(y_train, y_pred_train))
        test_accuracy = float(accuracy_score(y_test, y_pred_test))
        train_precision = float(precision_score(y_train, y_pred_train, average=avg, zero_division=0))
        test_precision = float(precision_score(y_test, y_pred_test, average=avg, zero_division=0))
        train_recall = float(recall_score(y_train, y_pred_train, average=avg, zero_division=0))
        test_recall = float(recall_score(y_test, y_pred_test, average=avg, zero_division=0))
        train_f1 = float(f1_score(y_train, y_pred_train, average=avg, zero_division=0))
        test_f1 = float(f1_score(y_test, y_pred_test, average=avg, zero_division=0))
        train_score_pondere = float(score_pondere(recall=train_recall, f1=train_f1, accuracy=train_accuracy))
        test_score_pondere = float(score_pondere(recall=test_recall, f1=test_f1, accuracy=test_accuracy))

        results.setdefault("train_accuracy", []).append(train_accuracy)
        results.setdefault("test_accuracy", []).append(test_accuracy)
        results.setdefault("train_precision", []).append(train_precision)
        results.setdefault("test_precision", []).append(test_precision)
        results.setdefault("train_recall", []).append(train_recall)
        results.setdefault("test_recall", []).append(test_recall)
        results.setdefault("train_f1", []).append(train_f1)
        results.setdefault("test_f1", []).append(test_f1)
        results.setdefault("train_score_pondere", []).append(train_score_pondere)
        results.setdefault("test_score_pondere", []).append(test_score_pondere)

        # ROC AUC (binaire)
        try:
            results.setdefault("train_roc_auc", []).append(roc_auc_score(y_train, proba_train_pos))
            results.setdefault("test_roc_auc", []).append(roc_auc_score(y_test, proba_test_pos))
        except ValueError:
            pass

    # Construction dynamique du tableau de synthèse (classification seulement)
    rows = []

    if "train_accuracy" in results:
        rows += [
            ("Accuracy", "train_accuracy", "test_accuracy"),
            ("Precision", "train_precision", "test_precision"),
            ("Recall", "train_recall", "test_recall"),
            ("F1", "train_f1", "test_f1"),
            ("Score_pondere", "train_score_pondere", "test_score_pondere"),
        ]
    if "train_roc_auc" in results:
        rows.append(("ROC_AUC", "train_roc_auc", "test_roc_auc"))

    summary = pd.DataFrame(
        {
            "metric": [m for (m, _, _) in rows],
            "train_mean": [float(np.mean(results[k_train])) for (_, k_train, _) in rows],
            "test_mean": [float(np.mean(results[k_test])) for (_, _, k_test) in rows],
            "train_std": [
                float(np.std(results[k_train], ddof=1)) if len(results[k_train]) > 1 else 0.0
                for (_, k_train, _) in rows
            ],
            "test_std": [
                float(np.std(results[k_test], ddof=1)) if len(results[k_test]) > 1 else 0.0
                for (_, _, k_test) in rows
            ],
        }
    )

    # Optimisation du seuil sur la CV (via proba out-of-fold)
    seuil_cv = float(seuil_decision)
    if optimiser_seuil_cv:
        if oof_proba_pos.isna().any():
            raise ValueError("Probas out-of-fold incomplètes : vérifie la boucle CV")

        seuil_cv, seuil_grid = _best_threshold_from_proba(
            y_true=y_train_full,
            y_proba_pos=oof_proba_pos.values,
            metric=score_seuil_cv,
            grid_size=101,
        )

        summary.attrs["seuil_decision"] = float(seuil_cv)
        summary.attrs["seuil_source"] = "cv_oof"
        summary.attrs["score_seuil_cv"] = str(score_seuil_cv)
        summary.attrs["seuil_grid"] = seuil_grid

        # Courbe précision-rappel (OOF)
        if afficher_courbe_pr:
            precision, recall, _ = precision_recall_curve(y_train_full, oof_proba_pos.values)

            # Point correspondant au seuil choisi
            y_pred_oof = (oof_proba_pos.values >= float(seuil_cv)).astype(int)
            p_sel = float(precision_score(y_train_full, y_pred_oof, zero_division=0))
            r_sel = float(recall_score(y_train_full, y_pred_oof, zero_division=0))

            plt.figure(figsize=(7, 5))
            plt.plot(recall, precision, label="PR (OOF CV)")
            plt.scatter([r_sel], [p_sel], color="red", zorder=3,
                        label=f"seuil={seuil_cv:.3f} (P={p_sel:.3f}, R={r_sel:.3f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Courbe précision-rappel (CV out-of-fold)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

    # Évaluation finale sur hold-out (1 seule fois, après sélection/validation)
    # (train déjà filtré via IsolationForest juste après le split)
    X_train_final = X_train_full
    y_train_final = y_train_full

    p_final = clone(pipe)
    p_final.fit(X_train_final, y_train_final)

    proba_holdout_pos = p_final.predict_proba(X_holdout)[:, 1]
    y_pred_holdout = (proba_holdout_pos >= float(seuil_cv)).astype(int)

    holdout_accuracy = float(accuracy_score(y_holdout, y_pred_holdout))
    holdout_precision = float(precision_score(y_holdout, y_pred_holdout, average=avg, zero_division=0))
    holdout_recall = float(recall_score(y_holdout, y_pred_holdout, average=avg, zero_division=0))
    holdout_f1 = float(f1_score(y_holdout, y_pred_holdout, average=avg, zero_division=0))
    holdout_score_pondere = float(score_pondere(recall=holdout_recall, f1=holdout_f1, accuracy=holdout_accuracy))
    holdout_roc_auc = float(roc_auc_score(y_holdout, proba_holdout_pos))

    holdout_metrics = {
        "Accuracy": holdout_accuracy,
        "Precision": holdout_precision,
        "Recall": holdout_recall,
        "F1": holdout_f1,
        "Score_pondere": holdout_score_pondere,
        "ROC_AUC": holdout_roc_auc,
    }

    summary["holdout"] = summary["metric"].map(holdout_metrics)

    if afficher_confusion_matrix:
        ConfusionMatrixDisplay.from_predictions(y_holdout, y_pred_holdout)
        plt.title("Matrice de confusion (hold-out)")
        plt.tight_layout()
        plt.show()

    return summary

#---------------------------------------------------------------------------------------------------------------------------------

def opti_pipeline(model, X, y,
                    search_space,
                    encoder_config,
                    ordinal_categories_map,
                    numerical_features,
                    n_splits=5,
                    k_best='all',
                    holdout_size=0.2,
                    seuil_decision=0.5,
                    scoring_label ="score_pondere",
                    optimiser_seuil_cv=True,
                    afficher_courbe_pr=True,
                    score_seuil_cv="score_pondere",
                    afficher_confusion_matrix=True):

    # Préprocessing + Modèle
    # Choix encoder catégoriel
    ordinal_cols = encoder_config["ordinal"]
    ordinal_categories = [ordinal_categories_map[col] for col in ordinal_cols]

    ordinal_enc = OrdinalEncoder(categories=ordinal_categories)

    preprocessor = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
            encoder_config["onehot"]),
            ("ordinal", ordinal_enc, ordinal_cols),
            ("target", TargetEncoder(random_state=42), encoder_config["target"]),
            ("num", "passthrough", numerical_features)
            ],
        )
    # Pipeline (imblearn)
    steps = [("preprocessing", preprocessor),
             ("selectkbest", SelectKBest(k=k_best)),
             ("model", model)]
    
    pipe = Pipeline(steps)

    # Hold-out
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
        X,
        y,
        test_size=holdout_size,
        random_state=42,
        stratify=y,
    )

    # IsolationForest fit sur tout le train
    X_train_full = X_train_full.copy()
    y_train_full = y_train_full.copy()
    num_if_cols = X_train_full.select_dtypes(include=[np.number]).columns
    n_outliers_train = 0
    if len(num_if_cols) > 0:
        iso = IsolationForest(
            n_estimators=50,
            contamination=0.02,
            random_state=42,
            n_jobs=-1,
        )
        labels_train = iso.fit_predict(X_train_full[num_if_cols])
        keep_mask = labels_train == 1
        n_outliers_train = int(np.sum(~keep_mask))
        X_train_full = X_train_full.loc[keep_mask].copy()
        y_train_full = y_train_full.loc[keep_mask].copy()

    if n_outliers_train > 0:
        print(f"IsolationForest: {n_outliers_train} outliers retirés du train (hold-out inchangé).")

    if scoring_label == "score_pondere":
        scoring = make_scorer(_score_pondere_from_pred, greater_is_better=True)

    # GridSearchCV avec CV stratifiée (sur train uniquement)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    bayes = BayesSearchCV(
        estimator=pipe,
        search_spaces=search_space,
        n_iter=200,              
        scoring=scoring,        
        cv=cv,
        n_jobs=-1,
        refit=True,
        random_state=42,)

    bayes.fit(X_train_full, y_train_full)

    best_estimator = bayes.best_estimator_
    best_params = bayes.best_params_
    best_score = bayes.best_score_

    print("\nBayesSearchCV terminé")
    print(f"Scoring: {scoring_label}")
    print(f"Meilleurs paramètres: {best_params}")
    print(f"Meilleur score: {best_score:.6f}")

    # Seuil décision optimisé via proba out-of-fold (sur train uniquement)
    seuil_cv = float(seuil_decision)
    oof_proba_pos = pd.Series(index=X_train_full.index, dtype=float)
    if optimiser_seuil_cv:
        for train_idx, test_idx in cv.split(X_train_full, y_train_full):
            X_tr = X_train_full.iloc[train_idx]
            y_tr = y_train_full.iloc[train_idx]
            X_te = X_train_full.iloc[test_idx]

            est = clone(best_estimator)
            est.fit(X_tr, y_tr)

            if not hasattr(est, "predict_proba"):
                raise ValueError(
                    "optimiser_seuil_cv=True requiert un estimateur avec predict_proba (classification binaire)."
                )

            oof_proba_pos.iloc[test_idx] = est.predict_proba(X_te)[:, 1]

        if oof_proba_pos.isna().any():
            raise ValueError("Probas out-of-fold incomplètes : vérifie la boucle CV")

        seuil_cv, seuil_grid = _best_threshold_from_proba(
            y_true=y_train_full,
            y_proba_pos=oof_proba_pos.values,
            metric=score_seuil_cv,
            grid_size=101,
        )

        # Courbe précision-rappel (OOF)
        if afficher_courbe_pr:
            precision, recall, _ = precision_recall_curve(y_train_full, oof_proba_pos.values)
            y_pred_oof = (oof_proba_pos.values >= float(seuil_cv)).astype(int)
            p_sel = float(precision_score(y_train_full, y_pred_oof, zero_division=0))
            r_sel = float(recall_score(y_train_full, y_pred_oof, zero_division=0))

            plt.figure(figsize=(7, 5))
            plt.plot(recall, precision, label="PR (OOF CV)")
            plt.scatter([r_sel], [p_sel], color="red", zorder=3,
                        label=f"seuil={seuil_cv:.3f} (P={p_sel:.3f}, R={r_sel:.3f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Courbe précision-rappel (OOF CV)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()
    else:
        seuil_grid = None

    # Évaluation hold-out avec le meilleur estimateur refit sur tout le train
    est_final = clone(best_estimator)
    est_final.fit(X_train_full, y_train_full)

    proba_holdout_pos = est_final.predict_proba(X_holdout)[:, 1]
    y_pred_holdout = (proba_holdout_pos >= float(seuil_cv)).astype(int)

    avg = "binary"
    holdout_accuracy = float(accuracy_score(y_holdout, y_pred_holdout))
    holdout_precision = float(precision_score(y_holdout, y_pred_holdout, average=avg, zero_division=0))
    holdout_recall = float(recall_score(y_holdout, y_pred_holdout, average=avg, zero_division=0))
    holdout_f1 = float(f1_score(y_holdout, y_pred_holdout, average=avg, zero_division=0))
    holdout_score_pondere = float(score_pondere(recall=holdout_recall, f1=holdout_f1, accuracy=holdout_accuracy))

    holdout_metrics = {
        "Accuracy": holdout_accuracy,
        "Precision": holdout_precision,
        "Recall": holdout_recall,
        "F1": holdout_f1,
        "Score_pondere": holdout_score_pondere,
    }

    holdout_metrics["ROC_AUC"] = float(roc_auc_score(y_holdout, proba_holdout_pos))

    summary = pd.DataFrame({
        "metric": list(holdout_metrics.keys()),
        "holdout": list(holdout_metrics.values()),
    })

    # Infos grid search en attributs
    summary.attrs["best_params"] = best_params
    summary.attrs["best_score"] = best_score
    summary.attrs["scoring"] = scoring_label
    summary.attrs["seuil_decision"] = float(seuil_cv)
    summary.attrs["seuil_source"] = "cv_oof" if optimiser_seuil_cv else "fixed"
    summary.attrs["score_seuil_cv"] = str(score_seuil_cv)
    summary.attrs["seuil_grid"] = seuil_grid

    if afficher_confusion_matrix:
        ConfusionMatrixDisplay.from_predictions(y_holdout, y_pred_holdout)
        plt.title("Matrice de confusion (hold-out)")
        plt.tight_layout()
        plt.show()

    return summary, bayes, X_holdout, y_holdout, X_train_full, y_train_full