---
title: openclassrooms_projet5
sdk: docker
app_port: 7860
---

# openclassrooms_projet5

Ce projet consiste à déployer le modèle de classification qu'un employée reste dans l'entreprise ou non créer lors du porjet 4. Afin de l'utiliser de manière simple en entreprise.


## Mode de lancement

- `APP_MODE=local` (par défaut) : initialise la base de données et active le logging des prédictions.
- `APP_MODE=demo` : désactive toute interaction avec la base de données (API + UI fonctionnent sans DB).

## Exposition API (Space Docker)

- API FastAPI exposée sur `/health` et `/predict`.
- UI Gradio montée sur `/gradio`.
- Le conteneur lance `uvicorn main:app` sur le port `7860`.



## Tests et couverture

make test