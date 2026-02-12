import pandas as pd
from skopt.space import Integer, Real, Categorical
from sklearn.metrics import make_scorer
from src.utils import opti_pipeline
from imblearn.ensemble import BalancedRandomForestClassifier


eval_df = pd.read_csv('data/extrait_eval.csv')
sirh_df = pd.read_csv('data/extrait_sirh.csv')
sondage_df = pd.read_csv('data/extrait_sondage.csv')

# Arranger la colonne 'eval_number' de eval_df 
eval_df['eval_number'] = eval_df['eval_number'].str.replace('E_', '', regex=False).astype(int)

# Fusionner les DataFrames
merge_df = eval_df.merge(sirh_df,left_on='eval_number',right_on='id_employee',how='inner').merge(sondage_df,left_on='eval_number',right_on='code_sondage',how='inner')
merge_df.drop(columns=['eval_number','id_employee','code_sondage'],inplace=True)

# On transforme pourcentage "d'augementation du salaire precedent" en int
merge_df['augementation_salaire_precedente'] = merge_df['augementation_salaire_precedente'].str.replace('%','',regex=False).astype(int)

# Supprimer les colonnes sans information
colToDrop = ['nombre_heures_travailless','nombre_employee_sous_responsabilite','ayant_enfants', 'satisfaction_employee_environnement', 'satisfaction_employee_nature_travail', 'satisfaction_employee_equipe', 'satisfaction_employee_equilibre_pro_perso', 'note_evaluation_actuelle']
merge_df.drop(columns=colToDrop, inplace=True)

# On recupère les colonnes catégorielles et numériques
cat_cols = merge_df.select_dtypes(include=['object']).columns.tolist()
num_cols = merge_df.select_dtypes(include=['number']).columns.tolist()

# salaire moyen par niveau hiérarchique
salaire_moyen_par_niveau = (
    merge_df.groupby("niveau_hierarchique_poste")["revenu_mensuel"].mean()
)

# ajoute le salaire moyen du niveau de chaque employé
merge_df["salaire_moyen_niveau"] = merge_df["niveau_hierarchique_poste"].map(salaire_moyen_par_niveau)

# crée diff_salaire_vs_niveau
merge_df["diff_salaire_vs_niveau"] = merge_df["revenu_mensuel"] - merge_df["salaire_moyen_niveau"]
merge_df["diff_salaire_vs_niveau_pct"] = merge_df["diff_salaire_vs_niveau"] / merge_df["salaire_moyen_niveau"]
merge_df.drop(columns=['salaire_moyen_niveau', 'diff_salaire_vs_niveau'], inplace=True)

# crée ratio_salaire_anciennete
merge_df['annee_experience_totale'] = merge_df['annee_experience_totale'] + 1
merge_df["ratio_salaire_anciennete"] = merge_df["revenu_mensuel"] / merge_df["annee_experience_totale"]
num_cols = merge_df.select_dtypes(include=["number"]).columns.tolist()

target_col = 'a_quitte_l_entreprise'
X = merge_df.drop(columns=[target_col])
y = merge_df[target_col].map({"Non": 0, "Oui": 1}).astype(int)

#colonne à supprimer après matrice de corrélation
colToDrop = ['annees_dans_l_entreprise','annee_experience_totale', 'annes_sous_responsable_actuel', 'niveau_hierarchique_poste']
X.drop(columns=colToDrop, inplace=True)

num_features = X.select_dtypes(include=['number']).columns.tolist()
print(len(num_features))
print(num_features)

# dictionnaire pour determiner les encodages des features catégorielles
encoder_config = {
    "onehot": ["genre"],
    "ordinal": ["heure_supplementaires", "frequence_deplacement"],
    "target": ['domaine_etude', 'poste', 'departement', 'statut_marital']
}

ordinal_categories_map = {
    "heure_supplementaires": [
        "Non", 
        "Oui", 
    ],
    "frequence_deplacement": [
        "Aucun",
        "Occasionnel",
        "Frequent",
    ]
}

# Optimiser le random forest balanced classifier
search_space = {
    "model__n_estimators": Integer(100, 800),
    "model__max_depth": Integer(2, 50),
    "model__min_samples_split": Integer(2, 20),
    "model__min_samples_leaf": Integer(1, 15),
    "model__max_features": Real(0.2, 1.0),
    "model__bootstrap": Categorical([True, False]),
    "model__class_weight": Categorical(["balanced", "balanced_subsample"]),
}

cv = 5

summary, bayes, X_holdout, y_holdout, X_train_full, y_train_full = opti_pipeline(
    model=BalancedRandomForestClassifier(random_state=42),
    X=X,
    y=y,
    search_space=search_space,
    encoder_config=encoder_config,
    ordinal_categories_map=ordinal_categories_map,
    numerical_features=num_features,
    n_splits=cv,
    k_best='all',
    holdout_size=0.2,
    seuil_decision=0.50,
    scoring_label="score_pondere",
    optimiser_seuil_cv=False,
    score_seuil_cv="score_pondere",
    afficher_confusion_matrix=True,
)

estimator = bayes.best_estimator_