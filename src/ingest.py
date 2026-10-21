"""Download MOI 實價登錄 buy/sell batches into data/raw/.

Two confirmed endpoints, both returning a plain CSV directly (no ZIP):

- Current/live batch: https://plvr.land.moi.gov.tw/Download
  ?fileName={city}_lvr_land_a.csv (lowercase city + txn type)
- Historical quarterly batch: https://plvr.land.moi.gov.tw/DownloadSeason
  ?season={ROC_year}S{quarter}&fileName={CITY}_lvr_land_A.csv
  (uppercase city + txn type). Each season file is a cumulative snapshot
  (includes corrections/late filings from older periods), not an isolated
  quarter slice -- confirmed empirically (115S1 spans ROC 99-115, with the
  bulk in 114-115). Looping multiple quarters and upserting on
  (city_code, case_id) handles the resulting overlap correctly.
"""

import argparse
import logging
from pathlib import Path

import requests

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DOWNLOAD_BASE_URL = "https://plvr.land.moi.gov.tw/Download"
DOWNLOAD_SEASON_BASE_URL = "https://plvr.land.moi.gov.tw/DownloadSeason"


def _guard_not_zip(content: bytes, context: str) -> None:
    if content[:2] == b"PK":
        raise ValueError(
            f"{context}: response is a ZIP, not the expected plain CSV. "
            "MOI's response format may have changed -- investigate before "
            "treating the bytes as CSV."
        )


def download_main_file(city_code: str, dest_dir: Path = config.RAW_DATA_DIR) -> Path:
    """Download the current live buy/sell batch for one city. Verified endpoint (M0)."""
    filename = config.MAIN_FILE_PATTERN.format(
        city_code=city_code, txn_type_code=config.TXN_TYPE_CODE
    )
    url = f"{DOWNLOAD_BASE_URL}?fileName={filename}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    _guard_not_zip(resp.content, f"current batch ({filename})")

    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / filename
    out_path.write_bytes(resp.content)
    logger.info("downloaded %s (%d bytes) -> %s", filename, len(resp.content), out_path)
    return out_path


def download_current_batch(dest_dir: Path = config.RAW_DATA_DIR) -> list:
    return [download_main_file(code, dest_dir) for code in config.CITY_CODES]


def season_local_filename(city_code: str, season: str) -> str:
    """Local filename for a downloaded season batch, distinct per season so
    repeated backfill runs don't collide with each other or the current batch.
    """
    return f"{city_code}_lvr_land_{config.TXN_TYPE_CODE}_{season}.csv"


def download_season_file(
    city_code: str, season: str, dest_dir: Path = config.RAW_DATA_DIR
) -> Path:
    """Download one city's historical quarterly batch (confirmed: plain CSV,
    uppercase city + txn type in the remote filename).
    """
    remote_filename = f"{city_code.upper()}_lvr_land_{config.TXN_TYPE_CODE.upper()}.csv"
    url = f"{DOWNLOAD_SEASON_BASE_URL}?season={season}&fileName={remote_filename}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    _guard_not_zip(resp.content, f"season {season} ({remote_filename})")

    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / season_local_filename(city_code, season)
    out_path.write_bytes(resp.content)
    logger.info(
        "downloaded season %s city %s (%d bytes) -> %s",
        season, city_code, len(resp.content), out_path,
    )
    return out_path


def download_backfill(seasons: list, dest_dir: Path = config.RAW_DATA_DIR) -> list:
    results = []
    for season in seasons:
        for city_code in config.CITY_CODES:
            path = download_season_file(city_code, season, dest_dir)
            results.append((city_code, path))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MOI 實價登錄 buy/sell batches.")
    parser.add_argument(
        "--period",
        default="current",
        help="'current' downloads the live batch. Use --seasons for historical backfill.",
    )
    parser.add_argument(
        "--seasons",
        nargs=2,
        metavar=("START", "END"),
        help="Download a historical season range, e.g. --seasons 112S3 115S1",
    )
    args = parser.parse_args()

    if args.seasons:
        start, end = args.seasons
        seasons = config.generate_seasons(start, end)
        print(f"backfilling {len(seasons)} seasons: {seasons}")
        for city_code, path in download_backfill(seasons):
            print(f"saved {path}")
        return

    if args.period != "current":
        raise NotImplementedError(
            f"Unrecognized --period '{args.period}'. Use --period current for the "
            "live batch, or --seasons START END for historical backfill."
        )

    for path in download_current_batch():
        print(f"saved {path}")


if __name__ == "__main__":
    main()
