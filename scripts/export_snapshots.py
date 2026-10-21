"""Export or reconcile the existing local database; never runs its preparation pipeline."""

import argparse
from datetime import datetime, timezone
from decimal import Decimal
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psycopg2
from dotenv import dotenv_values

from artifacts import SNAPSHOT_DIR, sha256
from model_features import category_lists, preprocess_population

FCT_SCHEMA = pa.schema([
    ("city_code", pa.string()), ("district", pa.string()), ("txn_date", pa.date32()),
    ("price_per_ping", pa.float64()), ("log_price_per_ping", pa.float64()),
    ("building_age", pa.float64()), ("area_ping", pa.float64()),
    ("floor_level", pa.int32()), ("total_floors", pa.int32()), ("floor_ratio", pa.float64()),
    ("whole_building", pa.bool_()), ("has_terrace_or_arcade", pa.bool_()),
    ("building_type", pa.string()), ("rooms", pa.int32()), ("halls", pa.int32()),
    ("baths", pa.int32()), ("has_management", pa.bool_()),
    ("bundled_parking", pa.bool_()), ("parking_price_is_zero", pa.bool_()),
])
DIM_SCHEMA = pa.schema([
    ("city_code", pa.string()), ("district", pa.string()), ("txn_count", pa.int32()),
    ("median_price_per_ping", pa.float64()),
])
SCHEMAS = {"fct_transactions": FCT_SCHEMA, "dim_district": DIM_SCHEMA}
EXPECTED_ROWS = {"fct_transactions": 100828, "dim_district": 41}


def connect_local():
    values = dotenv_values(ROOT / ".env")
    def setting(key, default):
        return os.environ.get(key, values.get(key) or default)
    if setting("DB_HOST", "localhost").lower() not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Export refused: DB_HOST must refer to local PostgreSQL.")
    # Pin the address: never resolve or contact a configured remote service.
    conn = psycopg2.connect(
        host="127.0.0.1", port=setting("DB_PORT", "5432"),
        user=setting("DB_USER", "postgres"), password=setting("DB_PASSWORD", ""),
        dbname=setting("DB_NAME", "housing_prices"), connect_timeout=5,
        options="-c default_transaction_read_only=on -c statement_timeout=60000",
        application_name="housing_snapshot_readonly",
    )
    conn.set_session(isolation_level="REPEATABLE READ", readonly=True)
    return conn


def table_from_database(cur, name):
    schema = SCHEMAS[name]
    cur.execute(f'SELECT {", ".join(schema.names)} FROM public.{name}')
    rows = cur.fetchall()
    if len(rows) != EXPECTED_ROWS[name]:
        raise ValueError(f"{name} has {len(rows)} rows; expected {EXPECTED_ROWS[name]}. Review a changed source explicitly.")
    columns = []
    decimal_errors = {}
    for i, field in enumerate(schema):
        values = [r[i] for r in rows]
        if pa.types.is_float64(field.type):
            decimal_errors[field.name] = float(max(
                (abs(Decimal.from_float(float(v)) - v) for v in values if isinstance(v, Decimal)),
                default=Decimal(0),
            ))
            values = [float(v) if v is not None else None for v in values]
        columns.append(pa.array(values, type=field.type))
    return pa.Table.from_arrays(columns, schema=schema), decimal_errors


def compare_values(left, right):
    """Compare every projected value as a multiset, allowing physical DB row reordering."""
    columns = left.columns.tolist()
    def canonical(df):
        return df.sort_values(columns, na_position="first").reset_index(drop=True)
    pd.testing.assert_frame_equal(canonical(left), canonical(right), check_exact=True)


def reconcile(cur, frames):
    fct, dim = frames["fct_transactions"], frames["dim_district"]
    model = preprocess_population(fct)
    assert len(model) == 98054, "Unexpected modeling population"
    assert sum(d.year < 2023 for d in fct["txn_date"]) == 104
    cur.execute("SELECT count(*), count(DISTINCT (city_code,case_id)), min(loaded_at), max(loaded_at) FROM public.fct_transactions")
    n, unique_n, first_loaded, last_loaded = cur.fetchone()
    assert n == unique_n == len(fct)
    cur.execute("""SELECT city_code,district,count(*),
        percentile_cont(0.5) WITHIN GROUP(ORDER BY price_per_ping)
        FROM public.fct_transactions GROUP BY city_code,district ORDER BY city_code,district""")
    db_agg = pd.DataFrame(cur.fetchall(), columns=DIM_SCHEMA.names)
    actual = dim.sort_values(["city_code", "district"]).reset_index(drop=True)
    # Python Unicode ordering and database collation may differ; sort both in Python.
    db_agg = db_agg.sort_values(["city_code", "district"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual[["city_code", "district", "txn_count"]],
        db_agg[["city_code", "district", "txn_count"]], check_dtype=False, check_exact=True,
    )
    median_delta = float(np.max(np.abs(actual["median_price_per_ping"] - db_agg["median_price_per_ping"])))
    assert median_delta < 1e-8
    snapshot_agg = fct.groupby(["city_code", "district"])["price_per_ping"].agg(["size", "median"]).reset_index()
    assert snapshot_agg["size"].tolist() == actual["txn_count"].tolist()
    assert np.allclose(snapshot_agg["median"], actual["median_price_per_ping"], atol=1e-8, rtol=0)
    return {
        "every_exported_field_compared_to_database": True,
        "source_transaction_keys_unique": True,
        "district_counts_match": True,
        "maximum_district_median_difference_ntd": median_delta,
        "pre_2023_rows_retained": 104,
        "source_loaded_at_range": [first_loaded.isoformat(), last_loaded.isoformat()],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Compare existing snapshots to the database without writing any files")
    args = parser.parse_args()
    conn = connect_local()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT current_setting('transaction_read_only'), current_database(), version()")
            read_only, database, server_version = cur.fetchone()
            assert read_only == "on"
            source, errors = {}, {}
            for name in SCHEMAS:
                source[name], errors[name] = table_from_database(cur, name)
            if not args.verify_only:
                if any((SNAPSHOT_DIR / name).exists() for name in ("manifest.json", "fct_transactions.parquet", "dim_district.parquet")):
                    raise FileExistsError("Snapshot artifacts already exist. Preserve/version the previous bundle before exporting again.")
                SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
                for name, table in source.items():
                    pq.write_table(table, SNAPSHOT_DIR / f"{name}.parquet", compression="zstd")
            frames = {}
            for name, table in source.items():
                frames[name] = pq.read_table(SNAPSHOT_DIR / f"{name}.parquet").to_pandas()
                compare_values(table.to_pandas(), frames[name])
            validation = reconcile(cur, frames)
            if args.verify_only:
                from artifacts import load_snapshots
                load_snapshots()
                print(json.dumps({"read_only": True, "validation": validation}, indent=2))
                return
            fct = frames["fct_transactions"]
            modeling = preprocess_population(fct)
            manifest = {
                "format_version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "provenance": {
                    "source": "Existing local PostgreSQL tables; no ingestion, cleaning, or refresh was run",
                    "schema": "public", "server_version": server_version.split(" on ", 1)[0],
                    "read_only": True, "isolation": "repeatable read",
                    "row_order": "Original SELECT physical order retained for reconstruction; not a stable transaction identifier",
                    "privacy": "Explicit allowlist excludes case_id, address, notes, source-file names and per-row load timestamps",
                    "numeric_representation": "PostgreSQL NUMERIC converted to IEEE float64, as in the original pandas read_sql model path; nullable integer/boolean/date types retained",
                },
                "coverage": {"min_date": str(min(fct["txn_date"])), "max_date": str(max(fct["txn_date"])), "pre_2023_rows": 104, "rows_from_2023": 100724},
                "modeling_population": {
                    "rows": len(modeling), "dropped_rows": len(fct)-len(modeling),
                    "drop_rule": "building_age or total_floors is null",
                    "time_origin": modeling["txn_date"].min().date().isoformat(),
                    "latest_month_index": int(modeling["months_since_start"].max()),
                    "category_counts": {k: len(v) for k,v in category_lists(modeling).items()},
                },
                "validation": validation,
                "tables": {},
                "analysis_evidence": {},
                "export_versions": {p: version(p) for p in ["pandas", "numpy", "pyarrow", "psycopg2-binary", "python-dotenv"]},
            }
            for name, table in source.items():
                p = SNAPSHOT_DIR / f"{name}.parquet"
                manifest["tables"][name] = {
                    "file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p), "rows": table.num_rows,
                    "schema": [{"name": f.name, "type": str(f.type), "nullable": f.nullable} for f in table.schema],
                    "source_schema": [], "max_decimal_to_float_error": errors[name],
                }
                cur.execute("SELECT column_name,data_type,is_nullable FROM information_schema.columns WHERE table_schema='public' AND table_name=%s ORDER BY ordinal_position", (name,))
                manifest["tables"][name]["source_schema"] = [dict(zip(["name","type","nullable"], r)) for r in cur.fetchall() if r[0] in SCHEMAS[name].names]
            for p in [ROOT/"notebooks/analysis.ipynb", *sorted((ROOT/"app/assets").glob("*"))]:
                if p.is_file():
                    normalize = p.suffix in (".json", ".ipynb")
                    manifest["analysis_evidence"][p.relative_to(ROOT).as_posix()] = {
                        "bytes_at_export": p.stat().st_size,
                        "sha256": sha256(p, normalize_newlines=normalize),
                        "sha256_mode": "normalize_crlf_to_lf" if normalize else "raw_bytes",
                    }
            (SNAPSHOT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            print(json.dumps({"exported_rows": EXPECTED_ROWS, "modeling_rows": len(modeling), "validation": validation}, indent=2))
    finally:
        conn.rollback()
        conn.close()


if __name__ == "__main__":
    main()
