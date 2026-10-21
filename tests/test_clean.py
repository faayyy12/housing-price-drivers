from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import clean
import load

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sample.csv"


# ---- roc_to_gregorian -------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1150518", date(2026, 5, 18)),
        ("0680618", date(1979, 6, 18)),
        ("", None),
        (None, None),
        ("0000000", None),  # ROC year 0 is invalid
        ("not-a-date", None),
    ],
)
def test_roc_to_gregorian(raw, expected):
    assert clean.roc_to_gregorian(raw) == expected


# ---- sqm_to_ping -------------------------------------------------------

def test_sqm_to_ping():
    assert clean.sqm_to_ping(100) == pytest.approx(30.25)


def test_sqm_to_ping_missing():
    assert clean.sqm_to_ping(None) is None
    assert clean.sqm_to_ping(float("nan")) is None


# ---- chinese_numeral_to_int --------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("三", 3),
        ("十", 10),
        ("十五", 15),
        ("二十", 20),
        ("二十二", 22),
        ("9", 9),
        ("", None),
    ],
)
def test_chinese_numeral_to_int(raw, expected):
    assert clean.chinese_numeral_to_int(raw) == expected


# ---- parse_floor [M7] --------------------------------------------------

def test_parse_floor_clean_value():
    assert clean.parse_floor("九層") == {
        "floor_level": 9,
        "whole_building": False,
        "is_basement": False,
        "has_extra_floor_descriptor": False,
    }


def test_parse_floor_comma_compound():
    result = clean.parse_floor("一層，平台，騎樓")
    assert result["floor_level"] == 1
    assert result["has_extra_floor_descriptor"] is True
    assert result["is_basement"] is False


def test_parse_floor_whole_building():
    result = clean.parse_floor("全")
    assert result["whole_building"] is True
    assert result["floor_level"] is None


def test_parse_floor_basement():
    result = clean.parse_floor("地下一層")
    assert result["is_basement"] is True
    assert result["floor_level"] is None


def test_parse_floor_multi_basement_compound():
    # real M0 example: comma-joined multiple basement levels
    result = clean.parse_floor("地下二層，地下三層，地下四層")
    assert result["is_basement"] is True
    assert result["has_extra_floor_descriptor"] is True


def test_parse_floor_blank():
    result = clean.parse_floor("")
    assert result == {
        "floor_level": None,
        "whole_building": False,
        "is_basement": False,
        "has_extra_floor_descriptor": False,
    }


# ---- compute_building_age ----------------------------------------------

def test_compute_building_age():
    txn = date(2026, 5, 18)
    completion = date(1979, 6, 18)
    assert clean.compute_building_age(txn, completion) == 2026 - 1979


def test_compute_building_age_missing():
    assert clean.compute_building_age(None, date(2000, 1, 1)) is None
    assert clean.compute_building_age(date(2000, 1, 1), None) is None


# ---- compute_price_per_ping [C1] ----------------------------------------

def test_compute_price_per_ping():
    # price_per_ping = unit_price_sqm / 0.3025 (NOT * 0.3025 -- that converts
    # area sqm->ping; price per the bigger unit is correspondingly larger,
    # so price inverts relative to the area conversion). No manual parking
    # subtraction needed (M0: source unit_price_sqm is already adjusted).
    assert clean.compute_price_per_ping(393997) == pytest.approx(1302469.42, rel=1e-6)


def test_compute_price_per_ping_missing():
    assert clean.compute_price_per_ping(None) is None


# ---- filters -------------------------------------------------------------

@pytest.mark.parametrize(
    "sign,expected_reason",
    [
        ("房地(土地+建物)", None),
        ("房地(土地+建物)+車位", None),
        ("土地", "land_only"),
        ("車位", "parking_only"),
        ("建物", "building_only"),
    ],
)
def test_classify_transaction_type(sign, expected_reason):
    assert clean.classify_transaction_type(sign) == expected_reason


def test_is_residential_use():
    assert clean.is_residential_use("住家用") is True
    assert clean.is_residential_use("商業用") is False
    assert clean.is_residential_use("") is False


@pytest.mark.parametrize(
    "note,expected_reason",
    [
        ("預售屋、或土地及建物分件登記案件；", "presale"),
        ("協議價購；", "govt_acquisition"),
        ("親友、員工、共有人或其他特殊關係間之交易；", "related_party"),
        ("陽台外推；其他增建；", None),
        ("", None),
        (None, None),
    ],
)
def test_classify_note(note, expected_reason):
    assert clean.classify_note(note) == expected_reason


def test_parse_has_management():
    assert clean.parse_has_management("有") is True
    assert clean.parse_has_management("無") is False
    assert clean.parse_has_management("") is None


def test_is_bundled_parking():
    assert clean.is_bundled_parking("房地(土地+建物)+車位") is True
    assert clean.is_bundled_parking("房地(土地+建物)") is False


def test_is_parking_price_zero():
    assert clean.is_parking_price_zero(0) is True
    assert clean.is_parking_price_zero(1700000) is False
    assert clean.is_parking_price_zero(None) is False


# ---- clean_row orchestration --------------------------------------------

def test_clean_row_keeps_valid_residential_sale():
    row = {
        "transaction_sign": "房地(土地+建物)",
        "main_use": "住家用",
        "note": "",
        "floor_raw": "三層",
        "total_floors_raw": "七層",
        "transaction_date_roc": "1150510",
        "completion_date_roc": "0900315",
        "building_area_sqm": 85.5,
        "unit_price_sqm": 175438,
        "has_management_raw": "有",
        "rooms_raw": "3",
        "halls_raw": "2",
        "baths_raw": "1",
        "building_type": "公寓(5樓含以下無電梯)",
        "parking_price": None,
    }
    result = clean.clean_row(row)
    assert result["keep"] is True
    assert result["txn_date"] == date(2026, 5, 10)
    assert result["building_age"] == 2026 - 2001  # ROC 90 + 1911 = 2001
    assert result["floor_level"] == 3
    assert result["total_floors"] == 7
    assert result["floor_ratio"] == pytest.approx(3 / 7)
    assert result["bundled_parking"] is False
    assert result["price_per_ping"] == pytest.approx(175438 / 0.3025)


@pytest.mark.parametrize(
    "overrides,expected_reason",
    [
        ({"transaction_sign": "土地"}, "land_only"),
        ({"main_use": "商業用"}, "non_residential"),
        ({"note": "預售屋、或土地及建物分件登記案件；"}, "presale"),
        ({"note": "親友、員工、共有人或其他特殊關係間之交易；"}, "related_party"),
        ({"floor_raw": "地下一層"}, "basement_floor"),
        ({"transaction_date_roc": ""}, "unparseable_transaction_date"),
    ],
)
def test_clean_row_rejects(overrides, expected_reason):
    row = {
        "transaction_sign": "房地(土地+建物)",
        "main_use": "住家用",
        "note": "",
        "floor_raw": "三層",
        "total_floors_raw": "七層",
        "transaction_date_roc": "1150510",
        "completion_date_roc": "0900315",
        "building_area_sqm": 85.5,
        "unit_price_sqm": 175438,
        "has_management_raw": "有",
    }
    row.update(overrides)
    result = clean.clean_row(row)
    assert result["keep"] is False
    assert result["reason_code"] == expected_reason


def test_clean_row_rejects_structural_presale():
    # completion (ROC 114 = 2025) is after transaction (ROC 105 = 2016):
    # building did not exist at sale -> presale, even with a blank note.
    row = {
        "transaction_sign": "房地(土地+建物)",
        "main_use": "住家用",
        "note": "",
        "floor_raw": "九層",
        "total_floors_raw": "十四層",
        "transaction_date_roc": "1050901",
        "completion_date_roc": "1140901",
        "building_area_sqm": 100.0,
        "unit_price_sqm": 200000,
        "has_management_raw": "有",
    }
    result = clean.clean_row(row)
    assert result["keep"] is False
    assert result["reason_code"] == "presale_structural"


def test_clean_row_keeps_null_age_missing_completion():
    # Missing completion date -> null building_age. NOT presale: kept as a
    # valid existing-home sale with unknown age (the model drops it later for
    # the missing feature, but the dashboard should still show it).
    row = {
        "transaction_sign": "房地(土地+建物)",
        "main_use": "住家用",
        "note": "",
        "floor_raw": "九層",
        "total_floors_raw": "十四層",
        "transaction_date_roc": "1150510",
        "completion_date_roc": "",
        "building_area_sqm": 100.0,
        "unit_price_sqm": 200000,
        "has_management_raw": "有",
    }
    result = clean.clean_row(row)
    assert result["keep"] is True
    assert result["building_age"] is None


def test_clean_row_parking_price_zero_flag():
    row = {
        "transaction_sign": "房地(土地+建物)+車位",
        "main_use": "住家用",
        "note": "",
        "floor_raw": "十層",
        "total_floors_raw": "十五層",
        "transaction_date_roc": "1150525",
        "completion_date_roc": "0850101",
        "building_area_sqm": 110.0,
        "unit_price_sqm": 181818,
        "has_management_raw": "無",
        "parking_price": 0,
    }
    result = clean.clean_row(row)
    assert result["keep"] is True
    assert result["bundled_parking"] is True
    assert result["parking_price_is_zero"] is True


# ---- end-to-end fixture: parse -> clean ---------------------------------

def test_fixture_csv_parses_and_cleans_end_to_end():
    raw = load.read_raw_csv(FIXTURE_PATH)
    stg_df = load.to_stg_dataframe(raw, "a", "sample.csv")
    records = [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for row in stg_df.to_dict(orient="records")
    ]
    results = {r["case_id"]: clean.clean_row(r) for r in records}

    assert results["FIXTURE0001"]["keep"] is True  # plain residential sale
    assert results["FIXTURE0002"]["keep"] is True  # bundled parking, comma-compound floor
    assert results["FIXTURE0002"]["has_terrace_or_arcade"] is True
    assert results["FIXTURE0003"]["keep"] is True  # bundled parking, price=0
    assert results["FIXTURE0003"]["parking_price_is_zero"] is True
    assert results["FIXTURE0004"]["reason_code"] == "land_only"
    assert results["FIXTURE0005"]["reason_code"] == "presale"
    assert results["FIXTURE0006"]["reason_code"] == "related_party"
    assert results["FIXTURE0007"]["reason_code"] == "non_residential"
    assert results["FIXTURE0008"]["reason_code"] == "basement_floor"
