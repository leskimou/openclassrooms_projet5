---
title: openclassrooms_projet5
sdk: docker
app_port: 7860
---




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



<!--  -->
<a id="readme-top"></a>
<!--

<!-- PROJECT LOGO -->
<br />
<div align="center">

<h3 align="center">API Modèle de décision</h3>

  <p align="center">
    Ce projet consiste à déployer le modèle de classification, créer lors du porjet 4, pour déterminer si um employé va quiter ou non l'entrerpise.
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Sommaire</summary>
  <ol>
    <li>
      <a href="#about-the-project">A propos du projet</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## A propos du projet

Scénario : Après avoir créer un modèle à la demande du département des ressources humaines afin détablir les causes d'attrition du personnel au sein de l'entreprise, et de prédire le risque d'un potentiel départ chez un employé. L'objective et de déployer un modèle de machine learning en production.

## Vue d'ensemble

L'application expose :
- une API FastAPI pour la prédiction,
- une UI Gradio montée dans FastAPI,
- un mode d'exécution piloté par APP_MODE pour activer ou non les interactions base de données.

Endpoints principaux :
- GET /health : statut applicatif,
- POST /predict : prédiction sur un batch de records,
- /gradio : interface utilisateur Gradio.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

Ce modèle peut s'utiliser de deux manières :

* En déport local
* Via HuggingFace

Le comportement est piloté par la variable APP_MODE.

- APP_MODE=local
	- active la base de données,
	- crée les tables api_requests et api_predictions au démarrage,
	- log des requêtes et prédictions.

- APP_MODE=huggingface (ou demo)
	- désactive toute interaction avec une base de données,
	- API et UI restent disponibles.

### Dépot local

En dépot local introduire un fichier .env (à la racine du projet ou dans confs) qui permettra de connecter l'API à une base de données PostgreSQL

Pré-requis :
- Python 3.12
- uv
- PostgreSQL accessible avec les credentials configurés

En dépot local vous pouvez introduire un fichier .env qui permettra de connecter l'API à une base de données PostgreSQL

1. Clone le repository
   ```sh
   git clone https://github.com/leskimou/openclassrooms_projet5.git
   ```

2. Installer les packages
   ```sh
   uv sync
   ```

3. Configurez votre fichier .env (en ajustant les variables selon les informations de votre BDD)
   ```.env
   DB_HOST=localhost
   DB_NAME=postgres
   DB_USER=postgres
   DB_PASSWORD=votremotdepasse
   DB_PORT=5432
   APP_MODE =local
   ```
4. Lancer l'application
   ```sh
   uv run python -m uvicorn main:app --reload --host 127.0.0.1 --port 7860
   ```
   
5. tester la santé
   ```sh
   Invoke-RestMethod http://127.0.0.1:7860/health
   ```

### Déploiement Hugging Face (Docker Space)

Vous pouvez tester l'API sur le Space HuggingFace en cliquant ici [huggingface-space]
- Le conteneur démarre avec uvicorn main:app sur le port 7860.
- APP_MODE=huggingface dans les Variables du Space.
- APP_MODE n'est pas local, aucune opération avec la base de données n'est exécutée.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
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

Exemple d'entrée :

```json
{
  "records": [
    {
      "note_evaluation_precedente": 3.8,
      "heure_supplementaires": "Oui",
      "augementation_salaire_precedente": 7,
      "age": 36,
      "genre": "M",
      "revenu_mensuel": 6200,
      "statut_marital": "Marié(e)",
      "departement": "Commercial",
      "poste": "Cadre Commercial",
      "nombre_experiences_precedentes": 3,
      "annees_dans_le_poste_actuel": 4,
      "nombre_participation_pee": 2,
      "nb_formations_suivies": 3,
      "distance_domicile_travail": 12,
      "niveau_education": 4,
      "domaine_etude": "Marketing",
      "frequence_deplacement": "Occasionnel",
      "annees_depuis_la_derniere_promotion": 1,
      "niveau_hierarchique_poste": 3,
      "annee_experience_totale": 11
    }
  ]
}
```

Sortie :
- proba_leave : liste de probabilités de départ
- label : liste de labels binaires (seuil 0.5)

Exemple de sortie :

```json
{
	"proba_leave": [0.82],
	"label": [1]
}
```

Codes de réponse :
- 200 : succès
- 422 : payload invalide (feature manquante, type invalide, modalité inconnue, etc.)













<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[huggingface-space]: https://huggingface.co/spaces/leskimou/openclassrooms_projet5
