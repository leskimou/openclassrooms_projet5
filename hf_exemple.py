import requests

BASE_URL = "https://leskimou-openclassrooms-projet5.hf.space"

def main() -> None:
    payload = {
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

    # endpoints corrects pour un Space hosted
    health = requests.get(f"{BASE_URL}/health", timeout=30)
    print("health:", health.status_code, health.text)

    resp = requests.post(f"{BASE_URL}/predict", json=payload, timeout=60)
    resp.raise_for_status()
    print(resp.json())

if __name__ == "__main__":
    print("Exemple d'utilisation de l'API de prédiction")
    main()