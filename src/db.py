"""Connection + upsert helpers. SQL transform files (sql/*.sql) are run
directly via psql from scripts/refresh.sh, not through this module -- keeps
the "plain SQL, no ORM" design choice honest and avoids re-implementing a
statement splitter for multi-statement .sql files.
"""

from typing import Iterable, Optional, Sequence

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine, create_engine

import config

_engine: Optional[Engine] = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(config.DATABASE_URL)
    return _engine


def upsert_dataframe(
    df: pd.DataFrame,
    table: str,
    conflict_cols: Sequence[str],
    engine: Optional[Engine] = None,
) -> int:
    """Upsert rows from a DataFrame into `table` via INSERT ... ON CONFLICT DO
    UPDATE. Column names in `df` must match `table`'s columns exactly.
    """
    if df.empty:
        return 0
    engine = engine or get_engine()
    columns = list(df.columns)
    update_cols = [c for c in columns if c not in conflict_cols]

    col_list = ", ".join(columns)
    placeholders = ", ".join(f":{c}" for c in columns)
    conflict_list = ", ".join(conflict_cols)
    update_list = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)

    sql = text(
        f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict_list}) DO UPDATE SET {update_list}"
    )

    # Postgres NUMERIC has no NaN representation -- pandas NaN must become
    # SQL NULL. df.where(..., None) does NOT work here: float64 columns
    # can't hold Python None, so it silently reverts to NaN. Must scrub at
    # the record level instead, after to_dict() has produced plain objects.
    records = [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]
    with engine.begin() as conn:
        conn.execute(sql, records)
    return len(records)


def insert_rejections(rows: Iterable[dict], engine: Optional[Engine] = None) -> int:
    """Append rows into rej_transactions. Append-only by design (see
    01_schema.sql) -- not an upsert.
    """
    rows = list(rows)
    if not rows:
        return 0
    engine = engine or get_engine()
    sql = text(
        "INSERT INTO rej_transactions (city_code, case_id, reason_code, detail) "
        "VALUES (:city_code, :case_id, :reason_code, :detail)"
    )
    with engine.begin() as conn:
        conn.execute(sql, rows)
    return len(rows)
