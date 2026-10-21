"""Parse raw MOI CSVs into stg_transactions.

Applies the M0 recon findings (see M0_NOTES.md):
- 2 header rows: row 1 is the Chinese column name, row 2 is an English
  machine-translation header (not data) -- real data starts at row 3.
- Per-file encoding: the main data files are UTF-8-sig; falls back to Big5
  if that fails (manifest.csv was Big5, data files were not, but don't
  assume that holds for every future batch).
- Fails loudly if expected columns are missing (schema drift).
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy.engine import Engine

import config
import db

logger = logging.getLogger(__name__)

# Raw Chinese column name -> stg_transactions column name. Confirmed against
# the M0 manual download header row (see M0_NOTES.md).
COLUMN_MAP = {
    "編號": "case_id",
    "移轉編號": "transfer_seq",
    "鄉鎮市區": "district",
    "交易標的": "transaction_sign",
    "土地位置建物門牌": "land_building_address",
    "土地移轉總面積平方公尺": "land_area_sqm",
    "都市土地使用分區": "urban_land_use_zone",
    "非都市土地使用分區": "non_urban_land_use_zone",
    "非都市土地使用編定": "non_urban_land_use_designation",
    "交易年月日": "transaction_date_roc",
    "交易筆棟數": "transaction_unit_count",
    "移轉層次": "floor_raw",
    "總樓層數": "total_floors_raw",
    "建物型態": "building_type",
    "主要用途": "main_use",
    "主要建材": "main_material",
    "建築完成年月": "completion_date_roc",
    "建物移轉總面積平方公尺": "building_area_sqm",
    "建物現況格局-房": "rooms_raw",
    "建物現況格局-廳": "halls_raw",
    "建物現況格局-衛": "baths_raw",
    "建物現況格局-隔間": "has_partition_raw",
    "有無管理組織": "has_management_raw",
    "總價元": "total_price",
    "單價元平方公尺": "unit_price_sqm",
    "車位類別": "parking_type",
    "車位移轉總面積平方公尺": "parking_area_sqm",
    "車位總價元": "parking_price",
    "備註": "note",
    "主建物面積": "main_building_area_sqm",
    "附屬建物面積": "auxiliary_building_area_sqm",
    "陽台面積": "balcony_area_sqm",
    "電梯": "elevator_raw",
}

NUMERIC_COLUMNS = [
    "land_area_sqm",
    "building_area_sqm",
    "total_price",
    "unit_price_sqm",
    "parking_area_sqm",
    "parking_price",
    "main_building_area_sqm",
    "auxiliary_building_area_sqm",
    "balcony_area_sqm",
]


def read_raw_csv(path: Path) -> pd.DataFrame:
    """Read one MOI main-table CSV, skipping the Chinese+English header rows."""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", header=0, skiprows=[1], dtype=str)
    except UnicodeDecodeError:
        logger.warning("utf-8-sig failed for %s, falling back to Big5", path)
        df = pd.read_csv(path, encoding="big5", header=0, skiprows=[1], dtype=str)

    missing = set(COLUMN_MAP) - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name}: missing expected columns {missing} -- MOI schema may "
            "have drifted. Check schema-main.csv and re-confirm column names "
            "before proceeding."
        )
    return df


def to_stg_dataframe(raw: pd.DataFrame, city_code: str, source_file: str) -> pd.DataFrame:
    df = raw.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())].copy()
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["city_code"] = city_code
    df["source_file"] = source_file
    return df


def load_file(path: Path, city_code: str, engine: Optional[Engine] = None) -> int:
    raw = read_raw_csv(path)
    stg_df = to_stg_dataframe(raw, city_code, path.name)
    n = db.upsert_dataframe(stg_df, "stg_transactions", ["city_code", "case_id"], engine)
    logger.info("%s: %d rows upserted into stg_transactions", path.name, n)
    return n


def load_all(raw_dir: Path = config.RAW_DATA_DIR) -> int:
    total = 0
    for city_code in config.CITY_CODES:
        filename = config.MAIN_FILE_PATTERN.format(
            city_code=city_code, txn_type_code=config.TXN_TYPE_CODE
        )
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"expected raw file not found: {path}")
        total += load_file(path, city_code)
    return total


def load_backfill(seasons: list, raw_dir: Path = config.RAW_DATA_DIR) -> int:
    """Loads previously-downloaded historical season files (see
    ingest.download_backfill / ingest.season_local_filename) into
    stg_transactions. Overlapping case_ids across seasons upsert cleanly.
    """
    total = 0
    for season in seasons:
        for city_code in config.CITY_CODES:
            filename = f"{city_code}_lvr_land_{config.TXN_TYPE_CODE}_{season}.csv"
            path = raw_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"expected backfill file not found: {path}")
            total += load_file(path, city_code)
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load raw MOI CSVs into stg_transactions.")
    parser.add_argument(
        "--seasons",
        nargs=2,
        metavar=("START", "END"),
        help="Load a previously-downloaded historical season range, e.g. --seasons 112S3 115S1",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.seasons:
        start, end = args.seasons
        seasons = config.generate_seasons(start, end)
        n = load_backfill(seasons)
        print(f"loaded {n} rows total from {len(seasons)} backfill seasons into stg_transactions")
    else:
        n = load_all()
        print(f"loaded {n} rows total into stg_transactions")
