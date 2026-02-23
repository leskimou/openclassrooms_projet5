---
title: openclassrooms_projet5
sdk: docker
app_port: 7860
---

# openclassrooms_projet5

Ce projet consiste à déployer le modèle de classification pour déterminer si un employée reste dans l'entreprise ou non créer lors du porjet 4. Afin de l'utiliser de manière simple en entreprise.


## Vue d'ensemble

L'application expose :
- une API FastAPI pour la prédiction,
- une UI Gradio montée dans FastAPI,
- un mode d'exécution piloté par APP_MODE pour activer ou non les interactions base de données.

Endpoints principaux :
- GET /health : statut applicatif,
- POST /predict : prédiction sur un batch de records,
- /gradio : interface utilisateur Gradio.


## Modes d'exécution : Local vs Hugging Face

Le comportement est piloté par la variable APP_MODE.

- APP_MODE=local
	- active la base de données,
	- crée les tables api_requests et api_predictions au démarrage,
	- logge les requêtes et prédictions.

- APP_MODE=huggingface (ou demo)
	- désactive toute interaction base de données,
	- API et UI restent disponibles.

Ordre de résolution des variables d'environnement au démarrage :
1. variables déjà présentes dans l'environnement runtime (ex: Hugging Face Spaces Variables/Secrets),
2. fichier .env à la racine du projet (si présent),
3. sinon premier fichier trouvé dans confs/** : .env puis .env.* (ex: confs/dev/.env.dev).

Note : les variables déjà injectées par la plateforme ont priorité.

Important (local) : créez impérativement un fichier d'environnement avec APP_MODE=local, soit à la racine (.env), soit dans confs/.


## Variables d'environnement

Variables utilisées pour la base de données (mode local) :
- DB_HOST
- DB_PORT
- DB_NAME
- DB_USER
- DB_PASSWORD

Variables d'exécution :
- APP_MODE (local, huggingface ou demo)


## Lancement en local

Pré-requis :
- Python 3.12+
- dépendances installées via uv
- PostgreSQL accessible avec les credentials configurés

Exemple PowerShell :

1) créer un fichier d'environnement

Option A (recommandé) : .env à la racine

Option B : un fichier dans confs/ (ex: confs/dev/.env.dev)

Exemple minimal :

APP_MODE=local
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=change_me

2) lancer l'application

uv run python -m uvicorn main:app --reload --host 127.0.0.1 --port 7860

3) tester la santé

Invoke-RestMethod http://127.0.0.1:7860/health


## Déploiement Hugging Face (Docker Space)

- Le conteneur démarre avec uvicorn main:app sur le port 7860.
- Définir APP_MODE=huggingface dans les Variables du Space.
- Si APP_MODE n'est pas local, aucune opération DB n'est exécutée.


## Contrat API

### GET /health

Réponse attendue :

{
	"status": "ok"
}

### POST /predict

Entrée :
- payload JSON avec records (liste non vide d'objets)
- chaque record doit contenir exactement les features attendues par artifacts/input_schema.json

Sortie :
- proba_leave : liste de probabilités de départ
- label : liste de labels binaires (seuil 0.5)

Exemple de sortie :

{
	"proba_leave": [0.82],
	"label": [1]
}

Codes de réponse :
- 200 : succès
- 422 : payload invalide (feature manquante, type invalide, modalité inconnue, etc.)


## Tests et couverture

Commande standard du projet :

make test