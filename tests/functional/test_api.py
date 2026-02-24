from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import main


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
	monkeypatch.setattr(main, "get_engine_from_env", lambda: None)
	monkeypatch.setattr(main, "init_api_logging_tables", lambda _engine: None)
	monkeypatch.setattr(
		main,
		"predict_with_artifact_model",
		lambda X, threshold=0.5: ([0.8 for _ in range(len(X))], [1 for _ in range(len(X))]),
	)

	with TestClient(main.app) as test_client:
		yield test_client


def test_health_returns_ok(client: TestClient) -> None:
	response = client.get("/health")

	assert response.status_code == 200
	assert response.json() == {"status": "ok"}


def test_predict_returns_proba_and_label(client: TestClient) -> None:
	payload = {"records": [_build_valid_record()]}

	response = client.post("/predict", json=payload)

	assert response.status_code == 200
	body = response.json()
	assert set(body.keys()) == {"proba_leave", "label"}
	assert len(body["proba_leave"]) == 1
	assert len(body["label"]) == 1
	assert isinstance(body["proba_leave"][0], float)
	assert body["label"][0] in (0, 1)


def test_predict_returns_422_when_record_has_missing_key(client: TestClient) -> None:
	record = _build_valid_record()
	missing_feature = next(iter(main.REQUIRED_FEATURES_ORDER))
	record.pop(missing_feature)
	payload = {"records": [record]}

	response = client.post("/predict", json=payload)

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

	response = client.post("/predict", json=payload)

	assert response.status_code == 422
	assert "doit être un nombre" in response.text


def test_resolve_env_file_path_returns_root_env_if_present(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
	monkeypatch.setattr(main, "ROOT_DIR", tmp_path)
	root_env = tmp_path / ".env"
	root_env.write_text("APP_MODE=local\n", encoding="utf-8")

	resolved = main._resolve_env_file_path()

	assert resolved == root_env


def test_resolve_env_file_path_returns_none_if_missing(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
	monkeypatch.setattr(main, "ROOT_DIR", tmp_path)

	resolved = main._resolve_env_file_path()

	assert resolved is None


def test_resolve_env_file_path_falls_back_to_confs_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
	monkeypatch.setattr(main, "ROOT_DIR", tmp_path)
	confs_env = tmp_path / "confs" / "dev" / ".env"
	confs_env.parent.mkdir(parents=True, exist_ok=True)
	confs_env.write_text("APP_MODE=local\n", encoding="utf-8")

	resolved = main._resolve_env_file_path()

	assert resolved == confs_env


def test_resolve_env_file_path_falls_back_to_confs_env_pattern(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
	monkeypatch.setattr(main, "ROOT_DIR", tmp_path)
	confs_env = tmp_path / "confs" / "dev" / ".env.dev"
	confs_env.parent.mkdir(parents=True, exist_ok=True)
	confs_env.write_text("APP_MODE=local\n", encoding="utf-8")

	resolved = main._resolve_env_file_path()

	assert resolved == confs_env
