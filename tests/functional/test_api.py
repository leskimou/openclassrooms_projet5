from __future__ import annotations

import os
from typing import Any

import pytest
from fastapi.testclient import TestClient

import main
from src.utils import load_env

TEST_API_KEY = "test-api-key"
AUTH_HEADERS = {"X-API-Key": TEST_API_KEY}


def _build_valid_record() -> dict[str, Any]:
	record: dict[str, Any] = {}
	for feature, spec in main.FEATURE_SPECS.items():
		expected_type = str(spec.get("type", "")).strip().lower()
		if expected_type == "number":
			record[feature] = 1.0
		elif expected_type == "category":
			choices = spec.get("choices", []) or []
			record[feature] = choices[0]
		else:
			raise AssertionError(f"Type de feature non supporté dans le schéma: {expected_type!r}")
	return record


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
	monkeypatch.setattr(main, "API_KEY", TEST_API_KEY)
	monkeypatch.setattr(main, "get_engine_from_env", lambda: None)
	monkeypatch.setattr(main, "init_api_logging_tables", lambda _engine: None)
	monkeypatch.setattr(main, "init_feature_tables_if_missing", lambda _engine: None)
	monkeypatch.setattr(
		main,
		"predict_with_artifact_model",
		lambda X, threshold=0.5: ([0.8 for _ in range(len(X))], [1 for _ in range(len(X))]),
	)

	with TestClient(main.app) as test_client:
		yield test_client


# ---------------------------------------------------------------------
# Tests API
# ---------------------------------------------------------------------

def test_health_returns_ok(client: TestClient) -> None:
	response = client.get("/health")

	assert response.status_code == 200
	assert response.json() == {"status": "ok"}


def test_predict_returns_proba_and_label(client: TestClient) -> None:
	payload = {"records": [_build_valid_record()]}

	response = client.post("/predict", json=payload, headers=AUTH_HEADERS)

	assert response.status_code == 200
	body = response.json()
	assert set(body.keys()) == {"proba_leave", "label"}
	assert len(body["proba_leave"]) == 1
	assert len(body["label"]) == 1
	assert isinstance(body["proba_leave"][0], float)
	assert body["label"][0] in (0, 1)


def test_predict_returns_401_without_api_key(client: TestClient) -> None:
	payload = {"records": [_build_valid_record()]}

	response = client.post("/predict", json=payload)

	assert response.status_code == 401


def test_predict_returns_401_with_wrong_api_key(client: TestClient) -> None:
	payload = {"records": [_build_valid_record()]}

	response = client.post("/predict", json=payload, headers={"X-API-Key": "wrong-api-key"})

	assert response.status_code == 401


def test_predict_returns_422_when_record_has_missing_key(client: TestClient) -> None:
	record = _build_valid_record()
	missing_feature = next(iter(main.REQUIRED_FEATURES_ORDER))
	record.pop(missing_feature)
	payload = {"records": [record]}

	response = client.post("/predict", json=payload, headers=AUTH_HEADERS)

	assert response.status_code == 422
	assert "features manquantes" in response.text


def test_predict_returns_422_when_feature_has_invalid_type(client: TestClient) -> None:
	record = _build_valid_record()
	numeric_feature = next(
		feature
		for feature, spec in main.FEATURE_SPECS.items()
		if str(spec.get("type", "")).strip().lower() == "number"
	)
	record[numeric_feature] = "not-a-number"
	payload = {"records": [record]}

	response = client.post("/predict", json=payload, headers=AUTH_HEADERS)

	assert response.status_code == 422
	assert "doit être un nombre" in response.text


# ---------------------------------------------------------------------
# Tests load_env (migré depuis main._resolve_env_file_path)
# ---------------------------------------------------------------------

def test_load_env_loads_root_env_if_present(tmp_path) -> None:
	root_env = tmp_path / ".env"
	root_env.write_text("TEST_VAR_ROOT=root\n", encoding="utf-8")

	load_env(root_dir=tmp_path)

	assert os.getenv("TEST_VAR_ROOT") == "root"


def test_load_env_does_nothing_if_no_env_file(tmp_path) -> None:
	# Ne doit pas lever d'exception
	load_env(root_dir=tmp_path)


def test_load_env_falls_back_to_confs_env(tmp_path) -> None:
	confs_env = tmp_path / "confs" / "dev" / ".env"
	confs_env.parent.mkdir(parents=True, exist_ok=True)
	confs_env.write_text("TEST_VAR_CONFS=confs\n", encoding="utf-8")

	load_env(root_dir=tmp_path)

	assert os.getenv("TEST_VAR_CONFS") == "confs"


def test_load_env_falls_back_to_confs_env_pattern(tmp_path) -> None:
	confs_env = tmp_path / "confs" / "dev" / ".env.dev"
	confs_env.parent.mkdir(parents=True, exist_ok=True)
	confs_env.write_text("TEST_VAR_PATTERN=pattern\n", encoding="utf-8")

	load_env(root_dir=tmp_path)

	assert os.getenv("TEST_VAR_PATTERN") == "pattern"