"""Validated, repository-relative serving artifacts. Never opens a database."""

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from model_features import FEATURE_COLS, QUANTILES, category_lists, feature_frame, preprocess_population

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "data" / "snapshots"
MODEL_DIR = ROOT / "models" / "estimator"


class ArtifactError(RuntimeError):
    """A missing, corrupt, or incompatible deployment artifact."""


def sha256(path, normalize_newlines=False):
    content = Path(path).read_bytes()
    if normalize_newlines:
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"Cannot read required artifact {Path(path).name}. Restore the deployment bundle.") from exc


def checked_path(folder, info):
    path = Path(folder) / info["file"]
    if path.parent.resolve() != Path(folder).resolve():
        raise ArtifactError("Invalid artifact path in metadata.")
    if not path.is_file() or sha256(path) != info["sha256"]:
        raise ArtifactError(f"Missing or hash-mismatched artifact: {path.name}. Restore the deployment bundle.")
    return path


def load_snapshots(folder=SNAPSHOT_DIR):
    manifest = read_json(Path(folder) / "manifest.json")
    if manifest.get("format_version") != 1:
        raise ArtifactError("Unsupported snapshot manifest version.")
    frames = {}
    for name in ("fct_transactions", "dim_district"):
        info = manifest["tables"][name]
        path = checked_path(folder, info)
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            if [{"name": f.name, "type": str(f.type), "nullable": f.nullable} for f in table.schema] != info["schema"]:
                raise ArtifactError(f"Snapshot schema mismatch: {name}.")
            frames[name] = table.to_pandas()
        except (OSError, ValueError) as exc:
            raise ArtifactError(f"Cannot read snapshot: {name}.") from exc
        if len(frames[name]) != info["rows"]:
            raise ArtifactError(f"Snapshot row count mismatch: {name}.")
    population, dim = frames["fct_transactions"], frames["dim_district"]
    model_df = preprocess_population(population)
    if len(model_df) != manifest["modeling_population"]["rows"]:
        raise ArtifactError("Estimator population does not match the snapshot manifest.")
    if int(dim["txn_count"].sum()) != len(population):
        raise ArtifactError("District counts do not match the complete population.")
    for name, info in manifest["analysis_evidence"].items():
        path = ROOT / name
        if not path.is_file() or sha256(path, info.get("sha256_mode") == "normalize_crlf_to_lf") != info["sha256"]:
            raise ArtifactError(f"Analytical evidence changed or is missing: {name}.")
    return population, dim, manifest


def load_estimator(manifest, folder=MODEL_DIR):
    metadata = read_json(Path(folder) / "metadata.json")
    if metadata.get("format_version") != 1 or metadata["feature_order"] != FEATURE_COLS:
        raise ArtifactError("Unsupported estimator feature contract.")
    if metadata["training_snapshot_sha256"] != manifest["tables"]["fct_transactions"]["sha256"]:
        raise ArtifactError("Estimator was trained against a different transaction snapshot.")
    expected_population = manifest["modeling_population"]
    if (metadata["training_rows"] != expected_population["rows"]
            or metadata["time"]["origin_date"] != expected_population["time_origin"]
            or metadata["time"]["latest_month_index"] != expected_population["latest_month_index"]):
        raise ArtifactError("Estimator preprocessing metadata does not match the snapshot.")
    models = {}
    for alpha in QUANTILES:
        info = metadata["models"][str(alpha)]
        path = checked_path(folder, info)
        try:
            booster = lgb.Booster(model_file=str(path))
        except lgb.basic.LightGBMError as exc:
            raise ArtifactError(f"Cannot load estimator model {path.name}.") from exc
        if booster.feature_name() != FEATURE_COLS:
            raise ArtifactError("Native model feature order does not match metadata.")
        expected = [metadata["categories"][k] for k in ("district", "building_type")]
        if booster.pandas_categorical != expected:
            raise ArtifactError("Native model categories do not match metadata.")
        models[alpha] = booster
    return models, metadata


def predict_quantiles(models, metadata, rows):
    X = feature_frame(rows, metadata["categories"], metadata["feature_order"])
    # The original targets are logarithms; these are conditional quantiles, not means.
    result = np.column_stack([
        np.exp(models[a].predict(X, num_threads=1, validate_features=True)) for a in QUANTILES
    ])
    if not np.isfinite(result).all():
        raise ArtifactError("Estimator returned non-finite predictions.")
    return result
