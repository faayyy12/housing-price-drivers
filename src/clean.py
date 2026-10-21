"""Pure cleaning/feature functions. No DB calls.

Every rule here is taken from the M0 recon findings (M0_NOTES.md), verified
against the real downloaded batch, not guessed:
- ROC dates are 7-digit YYYMMDD strings (transaction date and completion
  date both use this format).
- 移轉層次 / 總樓層數 use Chinese numerals + "層"; floor values are
  sometimes a comma-compound (e.g. "一層，平台，騎樓" or
  "地下二層，地下三層，地下四層") using a full-width comma "，".
- 單價元平方公尺 is already parking-adjusted by the source -- no manual
  subtraction needed; price_per_ping = unit_price_sqm / 0.3025 (NOT
  unit_price_sqm * 0.3025: that converts AREA sqm->ping, since ping is the
  bigger unit; PRICE per the bigger unit is correspondingly larger, so the
  price relationship inverts. A multiplicative bug here was caught and
  fixed in M3 -- see M2_notes.md "Bug found in M3".)
- Filters: keep only 房地(土地+建物) [+車位], 主要用途 = 住家用, and drop
  rows whose 備註 flags them as presale / related-party / a negotiated
  government acquisition (quarantined with a reason code, not silently
  dropped -- see quality.py).
"""

import math
import re
from datetime import date
from typing import Optional

SQM_TO_PING = 0.3025

CHINESE_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

_FLOOR_SUFFIX_RE = re.compile(r"^(.*?)層$")


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def chinese_numeral_to_int(s: str) -> Optional[int]:
    """Converts a Chinese numeral (e.g. "三", "十五", "二十二") to an int.
    Handles 0-99, which covers every floor count observed in the M0 batch
    (max seen: 二十四層 = 24).
    """
    s = (s or "").strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if "十" in s:
        tens_part, _, ones_part = s.partition("十")
        tens = CHINESE_DIGITS.get(tens_part, 1) if tens_part else 1
        ones = CHINESE_DIGITS.get(ones_part, 0) if ones_part else 0
        return tens * 10 + ones
    return CHINESE_DIGITS.get(s)


def roc_to_gregorian(roc_date_str: Optional[str]) -> Optional[date]:
    """Parses a 7-digit ROC date string (YYYMMDD) into a Gregorian date.
    Confirmed format against both transaction_date_roc and
    completion_date_roc in the M0 batch (uniformly 7 digits, no 5/6-digit
    variants observed).
    """
    s = (roc_date_str or "").strip()
    if not s or not s.isdigit() or len(s) < 5:
        return None
    day = int(s[-2:])
    month = int(s[-4:-2])
    roc_year = int(s[:-4])
    if roc_year <= 0 or not (1 <= month <= 12) or not (1 <= day <= 31):
        return None
    try:
        return date(roc_year + 1911, month, day)
    except ValueError:
        return None


def sqm_to_ping(sqm) -> Optional[float]:
    if _is_missing(sqm):
        return None
    return float(sqm) * SQM_TO_PING


def parse_floor(floor_raw: Optional[str]) -> dict:
    """Parses 移轉層次 per the M0-finalized rule:
    - "N層" -> floor_level = N
    - "N層，X[，Y...]" (comma-compound) -> floor_level = N, plus a generic
      has_extra_floor_descriptor flag for the trailing descriptor(s)
      (平台/騎樓/陽臺/夾層/見其他登記事項/other) -- one flag, not a flag
      per descriptor type (M0 decision).
    - "地下N層..." -> basement, excluded regardless of trailing tokens.
    - "全" -> whole_building flag, floor_level = None.
    - blank -> all None/False.
    No range format ("X層～Y層") was observed in the M0 batch; unparseable
    tokens fall back to floor_level = None rather than raising.
    """
    raw = (floor_raw or "").strip()
    if not raw:
        return {
            "floor_level": None,
            "whole_building": False,
            "is_basement": False,
            "has_extra_floor_descriptor": False,
        }
    if raw == "全":
        return {
            "floor_level": None,
            "whole_building": True,
            "is_basement": False,
            "has_extra_floor_descriptor": False,
        }

    tokens = raw.split("，")
    first = tokens[0].strip()
    has_extra = len(tokens) > 1

    if "地下" in first:
        return {
            "floor_level": None,
            "whole_building": False,
            "is_basement": True,
            "has_extra_floor_descriptor": has_extra,
        }

    match = _FLOOR_SUFFIX_RE.match(first)
    floor_level = chinese_numeral_to_int(match.group(1)) if match else None
    return {
        "floor_level": floor_level,
        "whole_building": False,
        "is_basement": False,
        "has_extra_floor_descriptor": has_extra,
    }


def parse_total_floors(total_floors_raw: Optional[str]) -> Optional[int]:
    raw = (total_floors_raw or "").strip()
    if not raw:
        return None
    match = _FLOOR_SUFFIX_RE.match(raw)
    if not match:
        return None
    return chinese_numeral_to_int(match.group(1))


def compute_floor_ratio(floor_level: Optional[int], total_floors: Optional[int]) -> Optional[float]:
    if floor_level is None or not total_floors:
        return None
    return floor_level / total_floors


def compute_building_age(txn_date: Optional[date], completion_date: Optional[date]) -> Optional[int]:
    """building_age = transaction year - completion year (design doc §6)."""
    if txn_date is None or completion_date is None:
        return None
    return txn_date.year - completion_date.year


def compute_price_per_ping(unit_price_sqm) -> Optional[float]:
    """price_per_ping = unit_price_sqm / 0.3025 (i.e. x3.3058). No manual
    parking subtraction -- M0 verified the source unit_price_sqm is already
    parking-adjusted.

    NOTE: this is the inverse of SQM_TO_PING, not SQM_TO_PING itself.
    SQM_TO_PING converts AREA sqm->ping (ping is bigger, so area_ping <
    area_sqm). Price per the bigger unit (ping) is correspondingly LARGER
    than price per the smaller unit (sqm), so the price conversion is the
    reciprocal of the area conversion. Multiplying by SQM_TO_PING here was
    a real bug (caught in M3 via an implausible dashboard estimate, verified
    against real 大安區 transactions and public price reports -- see
    M2_notes.md).
    """
    if _is_missing(unit_price_sqm):
        return None
    return float(unit_price_sqm) / SQM_TO_PING


def parse_has_management(raw: Optional[str]) -> Optional[bool]:
    r = (raw or "").strip()
    if r == "有":
        return True
    if r == "無":
        return False
    return None


def is_bundled_parking(transaction_sign: Optional[str]) -> bool:
    return (transaction_sign or "").strip() == "房地(土地+建物)+車位"


def is_parking_price_zero(parking_price) -> bool:
    if _is_missing(parking_price):
        return False
    return float(parking_price) == 0


def classify_transaction_type(transaction_sign: Optional[str]) -> Optional[str]:
    """Returns None if the row should be kept (房地, with or without 車位),
    otherwise a rejection reason_code.
    """
    sign = (transaction_sign or "").strip()
    if sign in ("房地(土地+建物)", "房地(土地+建物)+車位"):
        return None
    if sign == "土地":
        return "land_only"
    if sign == "車位":
        return "parking_only"
    if sign == "建物":
        return "building_only"
    return "unknown_transaction_sign"


def is_residential_use(main_use: Optional[str]) -> bool:
    return (main_use or "").strip() == "住家用"


def classify_note(note: Optional[str]) -> Optional[str]:
    """Checked in this priority order when multiple keywords match (M0):
    presale > government acquisition > related-party.
    """
    text = note or ""
    if "預售" in text:
        return "presale"
    if "協議價購" in text:
        return "govt_acquisition"
    if any(kw in text for kw in ("特殊關係", "親友", "關係人")):
        return "related_party"
    return None


def _parse_int(v) -> Optional[int]:
    if _is_missing(v):
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def clean_row(row: dict) -> dict:
    """Takes one stg_transactions row (as a dict) and returns either
    {"keep": False, "reason_code": ...} or {"keep": True, **derived_fct_fields}.

    Sanity bounds (zero/negative price or area, absurd price-per-ping) are
    NOT enforced here -- that's quality.py's job. This function only applies
    the scope/eligibility filters and feature derivations.
    """
    reason = classify_transaction_type(row.get("transaction_sign"))
    if reason:
        return {"keep": False, "reason_code": reason}

    if not is_residential_use(row.get("main_use")):
        return {"keep": False, "reason_code": "non_residential"}

    note_reason = classify_note(row.get("note"))
    if note_reason:
        return {"keep": False, "reason_code": note_reason}

    floor_info = parse_floor(row.get("floor_raw"))
    if floor_info["is_basement"]:
        return {"keep": False, "reason_code": "basement_floor"}

    txn_date = roc_to_gregorian(row.get("transaction_date_roc"))
    if txn_date is None:
        return {"keep": False, "reason_code": "unparseable_transaction_date"}

    completion_date = roc_to_gregorian(row.get("completion_date_roc"))
    building_age = compute_building_age(txn_date, completion_date)
    # Structural presale: completion year is after the transaction year, i.e.
    # the building did not yet exist when it was sold. These are presale /
    # new-construction sales that the 備註-keyword presale filter misses when
    # the note lacks the 預售 marker (M0/M2: ~75% have a blank note). A null
    # building_age (missing completion date) is NOT presale -- those are kept
    # as valid existing-home sales with unknown age.
    if building_age is not None and building_age < 0:
        return {"keep": False, "reason_code": "presale_structural"}

    total_floors = parse_total_floors(row.get("total_floors_raw"))
    area_ping = sqm_to_ping(row.get("building_area_sqm"))
    price_per_ping = compute_price_per_ping(row.get("unit_price_sqm"))
    log_price_per_ping = (
        math.log(price_per_ping) if price_per_ping and price_per_ping > 0 else None
    )
    bundled_parking = is_bundled_parking(row.get("transaction_sign"))

    return {
        "keep": True,
        "txn_date": txn_date,
        "price_per_ping": price_per_ping,
        "log_price_per_ping": log_price_per_ping,
        "building_age": building_age,
        "area_ping": area_ping,
        "floor_level": floor_info["floor_level"],
        "total_floors": total_floors,
        "floor_ratio": compute_floor_ratio(floor_info["floor_level"], total_floors),
        "whole_building": floor_info["whole_building"],
        "has_terrace_or_arcade": floor_info["has_extra_floor_descriptor"],
        "building_type": (row.get("building_type") or "").strip() or None,
        "rooms": _parse_int(row.get("rooms_raw")),
        "halls": _parse_int(row.get("halls_raw")),
        "baths": _parse_int(row.get("baths_raw")),
        "has_management": parse_has_management(row.get("has_management_raw")),
        "bundled_parking": bundled_parking,
        "parking_price_is_zero": bundled_parking
        and is_parking_price_zero(row.get("parking_price")),
    }
