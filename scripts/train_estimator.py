"""Explicit offline reconstruction of the former app's three quantile models."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lightgbm as lgb
import numpy as np

from artifacts import MODEL_DIR, load_snapshots, predict_quantiles, sha256
from model_features import MODEL_PARAMS, QUANTILES, FEATURE_COLS, category_lists, feature_frame, preprocess_population, quantile_diagnostics


def main():
    if (MODEL_DIR / "metadata.json").exists() or list(MODEL_DIR.glob("*.txt")):
        raise FileExistsError("Models already exist. Preserve/version the existing bundle before reconstructing again.")
    population, _, manifest = load_snapshots()
    df = preprocess_population(population)
    categories = category_lists(df)
    X = feature_frame(df, categories)
    y = df["log_price_per_ping"].astype(float)
    assert len(X) == 98054 and np.isfinite(y).all()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "Newly reconstructed locally from the preserved snapshot; original VPS models are unavailable, so exact former predictions cannot be established",
        "training_snapshot_sha256": manifest["tables"]["fct_transactions"]["sha256"],
        "training_rows": len(X), "feature_order": FEATURE_COLS, "categories": categories,
        "time": {"origin_date": df["txn_date"].min().date().isoformat(), "origin_month": df["txn_date"].min().strftime("%Y-%m"), "latest_month_index": int(df["months_since_start"].max()), "formula": "(year - origin.year) * 12 + month - origin.month; days ignored", "prediction_rule": "Use latest_month_index, not the wall clock"},
        "missing_values": {"drop_training_rows_if_null": ["building_age", "total_floors"], "fill_floor_ratio": 1.0, "other_numeric_values": "No additional imputation; source has no other missing model inputs", "unknown_categories": "Reject"},
        "boolean_conversion": "has_management, whole_building, bundled_parking, parking_price_is_zero cast to int with _int suffix",
        "target": "log_price_per_ping (already natural-log transformed and winsorized upstream)",
        "inverse_transform": "numpy.exp; quantiles, so no mean-smearing correction",
        "quantiles": list(QUANTILES), "interval": "Nominal 5th-95th conditional quantiles; 90% coverage is not guaranteed or newly evaluated",
        "parameters": MODEL_PARAMS,
        "versions": {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "pyarrow"]},
        "python": platform.python_version(), "models": {},
        "historical_evaluation": {
            "source": "Preserved notebooks/analysis.ipynb, saved outputs; not the reconstructed quantile estimator",
            "target_scale": "natural log price per ping for R2; NTD/ping for MAE",
            "rows": 98054, "older_training_rows": 78443, "newer_test_rows": 19611,
            "ols_full_sample_r2": 0.7651160620122884,
            "lightgbm_full_sample_r2": 0.861692506345294,
            "ols_holdout_r2": 0.7459955440855299,
            "lightgbm_holdout_r2": 0.8437454623852827,
            "lightgbm_holdout_mae_ntd_per_ping": 74907.54197693172,
        },
    }
    # Fixed, privacy-safe feature-only sample: actual valid inputs covering every
    # district/type plus a deterministic broad sample. It contains no identifiers.
    indexes = sorted(set(df.groupby("district", sort=True).head(1).index)
                     | set(df.groupby("building_type", sort=True).head(1).index)
                     | set(df.sample(256, random_state=42).index))
    sample = df.loc[indexes, FEATURE_COLS].copy()
    probe = feature_frame(sample, categories)
    before, boosters, timings = [], {}, {}
    all_predictions = []
    for alpha in QUANTILES:
        started = time.perf_counter()
        model = lgb.LGBMRegressor(**MODEL_PARAMS, alpha=alpha)
        model.fit(X, y, categorical_feature=["district", "building_type"])
        timings[str(alpha)] = round(time.perf_counter()-started, 3)
        expected = np.exp(model.predict(probe))
        path = MODEL_DIR / f"quantile_{int(alpha*100):02d}.txt"
        model.booster_.save_model(str(path))
        restored = lgb.Booster(model_file=str(path))
        actual = np.exp(restored.predict(probe, num_threads=1, validate_features=True))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        boosters[alpha] = restored
        before.append(expected)
        all_predictions.append(np.exp(restored.predict(X, num_threads=1)))
        metadata["models"][str(alpha)] = {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path), "alpha": alpha, "effective_sklearn_parameters": model.get_params(), "native_parameters": model.booster_.params}
    expected = np.column_stack(before)
    actual = predict_quantiles(boosters, metadata, sample)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    metadata["validation"] = {
        "serialization_probe": quantile_diagnostics(actual),
        "roundtrip_max_absolute_difference_ntd": float(np.max(np.abs(actual-expected))),
        "complete_modeling_population": quantile_diagnostics(np.column_stack(all_predictions)),
        "training_seconds_by_quantile": timings,
        "no_new_out_of_sample_evaluation": True,
    }
    # Feature-only probes support independent round-trip regressions without retraining.
    probe_rows = json.loads(sample.to_json(orient="records", force_ascii=False))
    checks = {"feature_rows": probe_rows, "expected_quantile_predictions_ntd_per_ping": expected.tolist(), "quantile_order": list(QUANTILES), "tolerance": {"rtol": 1e-12, "atol_ntd": 1e-6}}
    checks_path = MODEL_DIR / "prediction_checks.json"
    checks_path.write_text(json.dumps(checks, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    metadata["prediction_checks"] = {"file": checks_path.name, "bytes": checks_path.stat().st_size, "sha256": sha256(checks_path)}
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(metadata["validation"], indent=2))


if __name__ == "__main__":
    main()
