import requests
from src.create_db import build_payload_from_bdd_row

BASE_URL = "http://127.0.0.1:7860"

def main() -> None:
    payload = build_payload_from_bdd_row(row_number=24)

    health = requests.get(f"{BASE_URL}/health", timeout=30)
    print("health:", health.status_code, health.text)

    resp = requests.post(f"{BASE_URL}/predict", json=payload, timeout=60)
    resp.raise_for_status()
    print(resp.json())

if __name__ == "__main__":
    print("Exemple d'utilisation de l'API de prédiction")
    main()