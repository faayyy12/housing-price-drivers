"""Checks against the real preserved deployment bundle; no database or fitting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from artifacts import ArtifactError, MODEL_DIR, SNAPSHOT_DIR, checked_path, load_estimator, load_snapshots, predict_quantiles, sha256
from model_features import FEATURE_COLS, category_lists, feature_frame, preprocess_population, quantile_diagnostics


@pytest.fixture(scope="module")
def bundle():
    population, dim, manifest = load_snapshots()
    models, metadata = load_estimator(manifest)
    return population, dim, manifest, models, metadata


def test_snapshot_population_and_district_reconciliation(bundle):
    population, dim, manifest, _, _ = bundle
    assert len(population) == 100828 and len(dim) == 41
    assert str(min(population.txn_date)) == "2014-05-21"
    assert str(max(population.txn_date)) == "2026-05-30"
    assert sum(d.year < 2023 for d in population.txn_date) == 104
    assert not {"case_id", "land_building_address", "note", "source_file", "loaded_at"} & set(population.columns)
    expected = population.groupby(["city_code", "district"]).price_per_ping.agg(["size", "median"])
    actual = dim.set_index(["city_code", "district"]).sort_index()
    assert expected["size"].tolist() == actual.txn_count.tolist()
    np.testing.assert_allclose(expected["median"], actual.median_price_per_ping, atol=1e-8, rtol=0)
    assert manifest["validation"]["every_exported_field_compared_to_database"]


def test_original_preprocessing_contract(bundle):
    population, _, manifest, _, metadata = bundle
    df = preprocess_population(population)
    assert len(df) == 98054 and len(population)-len(df) == 2774
    assert category_lists(df) == metadata["categories"]
    assert metadata["categories"]["district"][0] == "三峽區"
    assert len(metadata["categories"]["district"]) == 41
    assert len(metadata["categories"]["building_type"]) == 4
    assert df.months_since_start.min() == 0 and df.months_since_start.max() == 144
    assert df.txn_date.min().date().isoformat() == metadata["time"]["origin_date"] == "2014-05-21"
    retained = population.dropna(subset=["building_age", "total_floors"])
    missing_ratio = retained.floor_ratio.isna()
    assert missing_ratio.sum() == 2536
    assert (df.loc[missing_ratio[missing_ratio].index, "floor_ratio"] == 1.0).all()
    for column in ["has_management", "whole_building", "bundled_parking", "parking_price_is_zero"]:
        np.testing.assert_array_equal(df[column+"_int"], retained[column].astype(int))
    feature_frame(df, metadata["categories"])
    assert metadata["training_snapshot_sha256"] == manifest["tables"]["fct_transactions"]["sha256"]


def test_saved_model_roundtrip_regression_and_crossings(bundle):
    _, _, _, models, metadata = bundle
    path = checked_path(MODEL_DIR, metadata["prediction_checks"])
    probes = json.loads(path.read_text(encoding="utf-8"))
    rows = pd.DataFrame(probes["feature_rows"])
    predictions = predict_quantiles(models, metadata, rows)
    np.testing.assert_allclose(predictions, probes["expected_quantile_predictions_ntd_per_ping"], rtol=1e-12, atol=1e-6)
    assert quantile_diagnostics(predictions) == metadata["validation"]["serialization_probe"]
    assert metadata["validation"]["roundtrip_max_absolute_difference_ntd"] == 0
    assert metadata["validation"]["complete_modeling_population"]["crossing_rows"] == 309
    assert not metadata["validation"]["complete_modeling_population"]["predictions_reordered_or_clipped"]


def test_unknown_categories_and_snapshot_mismatch_fail_explicitly(bundle):
    population, _, manifest, models, metadata = bundle
    row = preprocess_population(population).iloc[:1].copy()
    row["district"] = "not a district"
    with pytest.raises(ValueError, match="Unknown or missing district"):
        predict_quantiles(models, metadata, row)
    modified = json.loads(json.dumps(manifest))
    modified["tables"]["fct_transactions"]["sha256"] = "0" * 64
    with pytest.raises(ArtifactError, match="different transaction snapshot"):
        load_estimator(modified)


def test_missing_and_corrupt_artifacts_fail_explicitly(tmp_path):
    with pytest.raises(ArtifactError, match="Cannot read required artifact"):
        load_snapshots(tmp_path)
    path = tmp_path / "model.txt"
    path.write_text("not a model", encoding="utf-8")
    with pytest.raises(ArtifactError, match="hash-mismatched"):
        checked_path(tmp_path, {"file": path.name, "sha256": "0"*64})


def test_evidence_hashes_survive_git_line_endings(tmp_path):
    windows = tmp_path / "windows.json"
    linux = tmp_path / "linux.json"
    windows.write_bytes(b'{\r\n  "evidence": true\r\n}\r\n')
    linux.write_bytes(b'{\n  "evidence": true\n}\n')
    assert sha256(windows) != sha256(linux)
    assert sha256(windows, normalize_newlines=True) == sha256(linux, normalize_newlines=True)


def test_historical_metrics_are_saved_notebook_evidence(bundle):
    _, _, _, _, metadata = bundle
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root/"notebooks/analysis.ipynb").read_text(encoding="utf-8"))
    outputs = "\n".join("".join(o.get("text", [])) for c in notebook["cells"] for o in c.get("outputs", []))
    for key in ["ols_full_sample_r2", "lightgbm_full_sample_r2", "ols_holdout_r2", "lightgbm_holdout_r2", "lightgbm_holdout_mae_ntd_per_ping"]:
        assert str(metadata["historical_evaluation"][key]) in outputs
