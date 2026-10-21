"""Sanity bounds + winsorizing + quarantine routing.

Calls clean.clean_row for parsing/filtering, then enforces sanity bounds
beyond what clean.py checks (non-positive or absurd price/area), and
winsorizes legitimate-but-extreme price_per_ping values into
fct_transactions rather than dropping them, per the design doc's outlier
rule. Bounds below are set generously around the observed distribution
(price_per_ping ~75k-2.2M NTD/ping after the M3 price-per-ping formula fix
-- see clean.py compute_price_per_ping -- area_ping ~3.8-255 ping) so they
catch data-entry errors without rejecting genuine outliers.
"""

import logging
import math
from typing import Optional

import pandas as pd
from sqlalchemy.engine import Engine

import clean
import db

logger = logging.getLogger(__name__)

MIN_PRICE_PER_PING = 10_000
MAX_PRICE_PER_PING = 20_000_000
MIN_AREA_PING = 1
MAX_AREA_PING = 1_000

# Legitimate-but-extreme price_per_ping values are clipped (winsorized) at
# these percentiles rather than dropped (design doc outlier-handling rule).
WINSORIZE_LOWER_PCT = 0.01
WINSORIZE_UPPER_PCT = 0.99

FCT_COLUMNS = [
    "city_code",
    "case_id",
    "district",
    "txn_date",
    "price_per_ping",
    "log_price_per_ping",
    "building_age",
    "area_ping",
    "floor_level",
    "total_floors",
    "floor_ratio",
    "whole_building",
    "has_terrace_or_arcade",
    "building_type",
    "rooms",
    "halls",
    "baths",
    "has_management",
    "bundled_parking",
    "parking_price_is_zero",
]


def _sanity_check(cleaned: dict) -> Optional[str]:
    """Returns a reason_code if the cleaned row fails a sanity bound, else None."""
    ppp = cleaned.get("price_per_ping")
    area = cleaned.get("area_ping")
    if ppp is None or area is None:
        return "missing_price_or_area"
    if ppp <= 0 or area <= 0:
        return "non_positive_price_or_area"
    if not (MIN_PRICE_PER_PING <= ppp <= MAX_PRICE_PER_PING):
        return "absurd_price_per_ping"
    if not (MIN_AREA_PING <= area <= MAX_AREA_PING):
        return "absurd_area"
    return None


def process_batch(engine: Optional[Engine] = None) -> dict:
    """Reads every row from stg_transactions, runs it through clean.clean_row
    + sanity bounds, winsorizes price_per_ping, and writes
    fct_transactions / rej_transactions. Returns a run summary dict.
    """
    engine = engine or db.get_engine()
    stg = pd.read_sql("SELECT * FROM stg_transactions", engine)
    records = [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for row in stg.to_dict(orient="records")
    ]

    kept_rows = []
    rejections = []
    floor_parsed = 0
    floor_null = 0

    for row in records:
        city_code = row["city_code"]
        case_id = row["case_id"]
        cleaned = clean.clean_row(row)

        if not cleaned["keep"]:
            rejections.append(
                {
                    "city_code": city_code,
                    "case_id": case_id,
                    "reason_code": cleaned["reason_code"],
                    "detail": None,
                }
            )
            continue

        reason = _sanity_check(cleaned)
        if reason:
            rejections.append(
                {
                    "city_code": city_code,
                    "case_id": case_id,
                    "reason_code": reason,
                    "detail": None,
                }
            )
            continue

        # whole_building sales legitimately have no single floor_level --
        # don't count that as a parsing failure when computing the null rate.
        if cleaned["floor_level"] is None and not cleaned["whole_building"]:
            floor_null += 1
        else:
            floor_parsed += 1

        kept_rows.append(
            {
                "city_code": city_code,
                "case_id": case_id,
                "district": row.get("district"),
                **{
                    k: cleaned[k]
                    for k in FCT_COLUMNS
                    if k not in ("city_code", "case_id", "district")
                },
            }
        )

    fct_df = pd.DataFrame(kept_rows, columns=FCT_COLUMNS)

    if not fct_df.empty:
        lower = fct_df["price_per_ping"].quantile(WINSORIZE_LOWER_PCT)
        upper = fct_df["price_per_ping"].quantile(WINSORIZE_UPPER_PCT)
        fct_df["price_per_ping"] = fct_df["price_per_ping"].clip(lower, upper)
        fct_df["log_price_per_ping"] = fct_df["price_per_ping"].apply(
            lambda v: math.log(v) if v and v > 0 else None
        )

    n_upserted = db.upsert_dataframe(fct_df, "fct_transactions", ["city_code", "case_id"], engine)
    n_rejected = db.insert_rejections(rejections, engine)

    floor_total = floor_parsed + floor_null
    floor_null_rate = (floor_null / floor_total) if floor_total else 0.0

    summary = {
        "rows_seen": len(records),
        "kept": n_upserted,
        "rejected": n_rejected,
        "floor_null_rate": round(floor_null_rate, 4),
    }
    logger.info("quality summary: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = process_batch()
    print(result)
