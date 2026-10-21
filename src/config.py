import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / os.getenv("RAW_DATA_DIR", "data/raw")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "housing_prices")

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# MOI 實價登錄 city codes — scope is Taipei + New Taipei only (design doc §3).
CITY_CODES = {
    "a": "Taipei City",  # 臺北市
    "f": "New Taipei City",  # 新北市
}

# "a" = 不動產買賣 (buy/sell). Rental/presale are out of scope for v1 (design doc §11).
TXN_TYPE_CODE = "a"

# Main-table filename pattern confirmed against the M0 manual download, e.g.
# a_lvr_land_a.csv, f_lvr_land_a.csv (see M0_NOTES.md).
MAIN_FILE_PATTERN = "{city_code}_lvr_land_{txn_type_code}.csv"


def generate_seasons(start: str, end: str) -> list:
    """Generates inclusive ROC season codes (e.g. "112S3", "112S4", "113S1", ...)
    from start to end, in order. Season format: "{ROC_year}S{quarter 1-4}".
    """
    def parse(s):
        year_str, q_str = s.split("S")
        return int(year_str), int(q_str)

    year, quarter = parse(start)
    end_year, end_quarter = parse(end)
    seasons = []
    while (year, quarter) <= (end_year, end_quarter):
        seasons.append(f"{year}S{quarter}")
        quarter += 1
        if quarter > 4:
            quarter = 1
            year += 1
    return seasons
