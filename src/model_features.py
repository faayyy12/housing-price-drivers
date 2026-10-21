"""The original estimator's preprocessing contract; no database or training imports."""

import numpy as np
import pandas as pd

CONTINUOUS_COLS = [
    "building_age", "area_ping", "floor_ratio", "total_floors",
    "rooms", "halls", "baths", "months_since_start",
]
CONTROL_COLS = [
    "has_management_int", "whole_building_int",
    "bundled_parking_int", "parking_price_is_zero_int",
]
FEATURE_COLS = CONTINUOUS_COLS + CONTROL_COLS + ["district", "building_type"]
FLAG_COLUMNS = ["has_management", "whole_building", "bundled_parking", "parking_price_is_zero"]
QUANTILES = (0.05, 0.5, 0.95)
MODEL_PARAMS = dict(
    objective="quantile", n_estimators=300, max_depth=6,
    learning_rate=0.05, random_state=42, verbosity=-1,
)


def preprocess_population(population):
    """Match the former app, retaining row order and the pre-2023 observations."""
    df = population.dropna(subset=["building_age", "total_floors"]).copy()
    if df.empty:
        raise ValueError("The snapshot contains no usable estimator training rows.")
    df["floor_ratio"] = df["floor_ratio"].fillna(1.0)
    df["txn_date"] = pd.to_datetime(df["txn_date"])
    start = df["txn_date"].min()
    df["months_since_start"] = (
        (df["txn_date"].dt.year - start.year) * 12
        + df["txn_date"].dt.month - start.month
    )
    for column in FLAG_COLUMNS:
        df[column + "_int"] = df[column].astype(int)
    return df


def category_lists(df):
    return {name: sorted(df[name].unique().tolist()) for name in ("district", "building_type")}


def feature_frame(df, categories, feature_order=FEATURE_COLS):
    """Reject unknown categories instead of silently turning them into missing values."""
    result = df.loc[:, feature_order].copy()
    for name, values in categories.items():
        if result[name].isna().any() or not result[name].isin(values).all():
            raise ValueError(f"Unknown or missing {name} category in estimator input.")
        result[name] = pd.Categorical(result[name], categories=values)
    numeric = result[[c for c in feature_order if c not in categories]].astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Estimator numeric inputs must be finite after preprocessing.")
    return result


def quantile_diagnostics(predictions):
    predictions = np.asarray(predictions, dtype=float)
    return {
        "rows": len(predictions),
        "all_finite": bool(np.isfinite(predictions).all()),
        "crossing_rows": int(((predictions[:, 0] > predictions[:, 1])
                              | (predictions[:, 1] > predictions[:, 2])).sum()),
        "predictions_reordered_or_clipped": False,
    }
