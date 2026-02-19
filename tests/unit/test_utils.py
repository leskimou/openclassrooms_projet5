from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import utils


def test_load_avg_salary_by_level_reads_and_normalizes_keys(tmp_path: Path):
    payload = {" 1 ": 1000, 2: 2000.5}
    path = tmp_path / "avg_salary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = utils.load_avg_salary_by_level(path)

    assert result == {"1": 1000.0, "2": 2000.5}


@pytest.mark.parametrize(
    "value,expected",
    [
        (1, "1"),
        (1.0, "1"),
        (1.5, "1.5"),
        (" 2 ", "2"),
    ],
)
def test_normalize_level_key_valid_values(value, expected):
    assert utils.normalize_level_key(value) == expected


def test_normalize_level_key_rejects_bool():
    with pytest.raises(ValueError, match="nombre valide"):
        utils.normalize_level_key(True)


def test_preprocess_record_for_model_computes_expected_features():
    record = {
        "niveau_hierarchique_poste": 2,
        "revenu_mensuel": 3000,
        "annee_experience_totale": 4,
        "genre": "F",
    }
    avg_map = {"2": 2500.0}

    processed = utils.preprocess_record_for_model(record, avg_salary_by_level=avg_map)

    assert processed["diff_salaire_vs_niveau_pct"] == pytest.approx((3000.0 - 2500.0) / 2500.0)
    assert processed["ratio_salaire_anciennete"] == pytest.approx(3000.0 / 5.0)
    assert "niveau_hierarchique_poste" not in processed
    assert "annee_experience_totale" not in processed
    assert processed["genre"] == "F"


def test_preprocess_record_for_model_raises_for_unknown_level():
    record = {
        "niveau_hierarchique_poste": 3,
        "revenu_mensuel": 3000,
        "annee_experience_totale": 4,
    }

    with pytest.raises(ValueError, match="inconnu"):
        utils.preprocess_record_for_model(record, avg_salary_by_level={"1": 2000.0, "2": 3000.0})


@pytest.mark.parametrize("invalid_level", [0, 0.5, 5.1, 6])
def test_preprocess_record_for_model_raises_for_out_of_range_level(invalid_level):
    record = {
        "niveau_hierarchique_poste": invalid_level,
        "revenu_mensuel": 3000,
        "annee_experience_totale": 4,
    }

    with pytest.raises(ValueError, match="compris entre 1 et 5"):
        utils.preprocess_record_for_model(record, avg_salary_by_level={"1": 2000.0, "2": 3000.0, "3": 4000.0, "4": 5000.0, "5": 6000.0})


def test_preprocess_record_for_model_raises_for_invalid_experience():
    record = {
        "niveau_hierarchique_poste": 1,
        "revenu_mensuel": 3000,
        "annee_experience_totale": -1,
    }

    with pytest.raises(ValueError, match="> -1"):
        utils.preprocess_record_for_model(record, avg_salary_by_level={"1": 2000.0})


def test_preprocess_record_for_model_raises_when_avg_salary_is_zero():
    record = {
        "niveau_hierarchique_poste": 1,
        "revenu_mensuel": 3000,
        "annee_experience_totale": 4,
    }

    with pytest.raises(ValueError, match="division par zéro"):
        utils.preprocess_record_for_model(record, avg_salary_by_level={"1": 0.0})
