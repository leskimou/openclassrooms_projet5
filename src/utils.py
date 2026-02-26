from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
AVG_SALARY_BY_LEVEL_PATH = ARTIFACTS_DIR / "salaire_moyen_par_niveau_hierarchique.json"


def load_avg_salary_by_level(path: Path = AVG_SALARY_BY_LEVEL_PATH) -> dict[str, float]:
	payload = json.loads(path.read_text(encoding="utf-8"))
	return {str(level).strip(): float(value) for level, value in payload.items()}


AVG_SALARY_BY_LEVEL = load_avg_salary_by_level()


def normalize_level_key(value: Any) -> str:
	if isinstance(value, bool):
		raise ValueError("niveau_hierarchique_poste doit être un nombre valide")
	if isinstance(value, int):
		return str(value)
	if isinstance(value, float):
		return str(int(value)) if value.is_integer() else str(value)
	return str(value).strip()


def preprocess_record_for_model(
	record: dict[str, Any],
	avg_salary_by_level: dict[str, float] | None = None,
) -> dict[str, Any]:
	processed = dict(record)
	salary_map = avg_salary_by_level if avg_salary_by_level is not None else AVG_SALARY_BY_LEVEL

	level_value = float(processed["niveau_hierarchique_poste"])
	if level_value < 1 or level_value > 5:
		raise ValueError("niveau_hierarchique_poste doit être compris entre 1 et 5")

	level_key = normalize_level_key(processed["niveau_hierarchique_poste"])
	if level_key not in salary_map:
		raise ValueError(f"niveau_hierarchique_poste inconnu: {processed['niveau_hierarchique_poste']!r}")

	avg_salary = float(salary_map[level_key])
	if avg_salary == 0.0:
		raise ValueError("Salaire moyen par niveau invalide (division par zéro)")

	monthly_salary = float(processed["revenu_mensuel"])
	total_experience = float(processed["annee_experience_totale"])
	ratio_denominator = total_experience + 1.0
	if ratio_denominator <= 0.0:
		raise ValueError("annee_experience_totale doit être > -1")

	processed["diff_salaire_vs_niveau_pct"] = (monthly_salary - avg_salary) / avg_salary
	processed["ratio_salaire_anciennete"] = monthly_salary / ratio_denominator

	processed.pop("niveau_hierarchique_poste", None)
	processed.pop("annee_experience_totale", None)
	return processed

def load_env(root_dir: Path | None = None) -> None:
    """
    Cherche et charge le premier fichier .env trouvé dans l'arborescence du projet.
    Priorité :
    1. .env à la racine du projet
    2. confs/**/.env
    3. confs/**/.env.*
    """
    if root_dir is None:
        root_dir = Path(__file__).resolve().parent.parent  # remonte à la racine depuis src/

    root_env = root_dir / ".env"
    if root_env.exists():
        load_dotenv(dotenv_path=root_env, override=False)
        return

    confs_dir = root_dir / "confs"
    if confs_dir.exists():
        env_candidates = sorted(confs_dir.rglob(".env"))
        if env_candidates:
            load_dotenv(dotenv_path=env_candidates[0], override=False)
            return

        env_pattern_candidates = sorted(confs_dir.rglob(".env.*"))
        if env_pattern_candidates:
            load_dotenv(dotenv_path=env_pattern_candidates[0], override=False)
            return