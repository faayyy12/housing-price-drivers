"""Regression checks for presentation and validation; no preparation pipeline."""

import base64
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import streamlit as st
from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df
from streamlit.runtime.media_file_manager import MediaFileManager
from streamlit.testing.v1 import AppTest

import artifacts
from model_features import FEATURE_COLS

ROOT = Path(__file__).resolve().parents[1]
HIGH_RISE = "住宅大樓(11層含以上有電梯)"
MID_RISE = "華廈(10層含以下有電梯)"
WALK_UP = "公寓(5樓含以下無電梯)"
TOWNHOUSE = "透天厝"


def run_app():
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=30).run()
    assert not app.exception
    return app


def number(app, label):
    return next(x for x in app.number_input if x.label == label)


def test_shap_is_delivered_through_app_without_media_endpoint(monkeypatch):
    def no_media_request(*args, **kwargs):
        raise AssertionError("The SHAP chart must not depend on a separate media request")
    monkeypatch.setattr(MediaFileManager, "add", no_media_request)
    st.cache_data.clear()
    app = run_app()
    assert not app.error
    images = app.get("imgs")
    assert len(images) == 1 and len(images[0].proto.imgs) == 1
    uri = images[0].proto.imgs[0].url
    assert uri.startswith("data:image/png;base64,")
    content = base64.b64decode(uri.split(",", 1)[1], validate=True)
    assert content == (ROOT / "app/assets/shap_summary.png").read_bytes()
    with Image.open(io.BytesIO(content)) as image:
        assert image.size == (1186, 1055)
        image.verify()


@pytest.mark.parametrize("building_type,floors", [(HIGH_RISE, 5), (HIGH_RISE, 10), (MID_RISE, 11)])
def test_invalid_category_floor_combinations_block_predictions(monkeypatch, building_type, floors):
    def no_prediction(*args, **kwargs):
        raise AssertionError("Invalid floor/category combinations must not reach the model")
    monkeypatch.setattr(artifacts, "predict_quantiles", no_prediction)
    app = run_app()
    app.selectbox[1].set_value(building_type).run()
    number(app, "Total floors").set_value(floors).run()
    assert app.selectbox[1].value == building_type
    assert app.error and app.button[0].disabled
    unit = number(app, "Floor of unit")
    assert unit.value <= floors and unit.max == floors
    app.button[0].click().run()  # Also guards a stale/programmatic disabled click.
    assert not app.exception and len(app.metric) == 3


@pytest.mark.parametrize("building_type,floors,unusual", [
    (HIGH_RISE, 11, False), (MID_RISE, 3, False), (MID_RISE, 10, False),
    (WALK_UP, 5, False), (WALK_UP, 6, True), (TOWNHOUSE, 8, True),
])
def test_valid_and_unusual_floor_combinations(building_type, floors, unusual):
    app = run_app()
    app.selectbox[1].set_value(building_type).run()
    number(app, "Total floors").set_value(floors).run()
    assert not app.error and not app.button[0].disabled
    assert app.selectbox[1].value == building_type
    assert any("Unusual" in x.value for x in app.warning) == unusual
    app.button[0].click().run()
    assert not app.exception and not app.error and len(app.metric) == 7


def test_townhouse_defaults_and_unit_floor_bounds_survive_category_changes():
    app = run_app()
    assert number(app, "Total floors").value == 14
    assert number(app, "Floor of unit").value == 9
    app.selectbox[1].set_value(TOWNHOUSE).run()
    assert app.checkbox[1].value is True
    assert number(app, "Total floors").value == 3
    assert not any(x.label == "Floor of unit" for x in app.number_input)
    app.selectbox[1].set_value(MID_RISE).run()
    assert app.checkbox[1].value is False
    assert number(app, "Total floors").value == 8
    assert number(app, "Floor of unit").value <= 8


def test_bilingual_dropdowns_preserve_raw_model_categories(monkeypatch):
    calls = []
    original_predict = artifacts.predict_quantiles
    def capture(models, metadata, rows):
        calls.append((rows.copy(), metadata))
        return original_predict(models, metadata, rows)
    monkeypatch.setattr(artifacts, "predict_quantiles", capture)
    app = run_app()
    metadata = json.loads((ROOT / "models/estimator/metadata.json").read_text(encoding="utf-8"))
    assert "三峽區 (Sanxia)" in app.selectbox[0].options
    assert app.selectbox[1].options == [
        "住宅大樓 (High-rise)", "公寓 (Walk-up)", "華廈 (Mid-rise)", "透天厝 (Townhouse)",
    ]
    assert all(" (" in label for label in app.selectbox[0].options)
    app.selectbox[0].set_value("三峽區")
    app.selectbox[1].set_value(MID_RISE).run()
    app.button[0].click().run()
    assert not app.exception and len(calls) == 1
    rows, supplied_metadata = calls[0]
    assert rows.columns.tolist() == FEATURE_COLS
    assert rows.district.iloc[0] == "三峽區" and rows.building_type.iloc[0] == MID_RISE
    for column in ["district", "building_type"]:
        assert rows[column].cat.categories.tolist() == metadata["categories"][column]
    assert supplied_metadata["categories"] == metadata["categories"]


def test_table_currency_is_formatted_without_rounding_numeric_data():
    app = run_app()
    table = app.dataframe[0]
    values = table.value["Median sale price / ping (NT$)"]
    assert pd.api.types.is_numeric_dtype(values)
    assert np.any(values.to_numpy() % 1 != 0)
    formatted = convert_arrow_bytes_to_pandas_df(table.proto.arrow_data.styler.display_values)
    assert formatted["Median sale price / ping (NT$)"].tolist() == [f"NT${x:,.0f}" for x in values]
    assert all("," in value for value in formatted["Median sale price / ping (NT$)"])


@pytest.mark.parametrize("city,show_all,count,scope_text", [
    ("Both", False, 15, "15 highest-priced districts in Taipei City and New Taipei City"),
    ("Both", True, 41, "all 41 districts in Taipei City and New Taipei City"),
    ("Taipei City", False, 12, "all 12 available districts in Taipei City (fewer than 15)"),
    ("Taipei City", True, 12, "all 12 districts in Taipei City"),
    ("New Taipei City", False, 15, "15 highest-priced districts in New Taipei City"),
    ("New Taipei City", True, 29, "all 29 districts in New Taipei City"),
])
def test_comparison_scope_caption_matches_unchanged_selection(city, show_all, count, scope_text):
    app = run_app()
    app.radio[0].set_value(city)
    app.checkbox[0].set_value(show_all).run()
    assert not app.exception and not app.error
    assert any(scope_text in x.value for x in app.caption)
    _, dim, _ = artifacts.load_snapshots()
    if city != "Both":
        dim = dim[dim.city_code == {"Taipei City": "a", "New Taipei City": "f"}[city]]
    expected = dim.sort_values("median_price_per_ping", ascending=False)
    if not show_all:
        expected = expected.head(15)
    table = app.dataframe[0].value
    assert len(table) == count
    np.testing.assert_array_equal(table["Median sale price / ping (NT$)"], expected.median_price_per_ping)
    np.testing.assert_array_equal(table["Recorded sales"], expected.txn_count)
    assert table["District"].str.split(" (", regex=False).str[0].tolist() == expected.district.tolist()
    chart = json.loads(app.get("plotly_chart")[3].proto.spec)
    assert chart["data"][0]["x"] == table["District"].tolist()
    # The graph uses the same full-precision medians as the displayed table.
    np.testing.assert_array_equal(chart["data"][0]["y"], expected.median_price_per_ping)


def test_chart_selection_scope_and_responsive_coverage_are_explicit():
    app = run_app()
    assert any("10 highest and 10 lowest" in x.value and "middle 20" in x.value for x in app.caption)
    data = json.loads((ROOT / "app/assets/district_premium_data.json").read_text(encoding="utf-8"))
    chart = json.loads(app.get("plotly_chart")[0].proto.spec)
    expected = sorted(data, key=lambda row: row["effect_pct"], reverse=True)
    assert chart["data"][0]["x"] == [x["effect_pct"] for x in expected[:10] + expected[-10:]]
    assert app.metric[2].value == "2014/05 – 2026/05"
    assert any(x.proto.id.endswith("-snapshot-coverage") for x in app.get("flex_container"))
    css = next(x.value for x in app.markdown if ".st-key-snapshot-coverage" in x.value)
    assert "white-space: normal" in css and "text-overflow: clip" in css
    assert "clamp(1.125rem, 2vw, 1.625rem)" in css


def test_final_hover_labels_limit_precision_without_rounding_chart_data():
    """Check the exact frontend text, rather than just numeric format directives."""
    app = run_app()
    charts = [json.loads(x.proto.spec) for x in app.get("plotly_chart")]
    for chart in charts:
        for trace in chart["data"]:
            assert trace["hoverinfo"] == "text" and not trace.get("hovertemplate")
            assert len(trace["hovertext"]) == len(trace["x"])
            for label in trace["hovertext"]:
                assert "%{" not in label
                percentages = re.findall(r"[+-]?\d+(?:\.\d+)?%", label)
                assert all(re.fullmatch(r"[+-]?\d+\.\d{2}%", p) for p in percentages)
                if "NT$" in label:
                    assert re.fullmatch(r"\d{1,3}(?:,\d{3})*", label.rsplit("NT$", 1)[1])

    districts = charts[0]["data"][0]
    assert any("-59.50%" in label for label in districts["hovertext"])
    for raw, label in zip(districts["x"], districts["hovertext"]):
        assert label.endswith(f"{raw:+.2f}%")
    assert any(raw != round(raw, 2) for raw in districts["x"])

    management = charts[1]["data"][0]
    assert any("+6.37%" in label for label in management["hovertext"])
    for raw, custom, label in zip(management["x"], management["customdata"], management["hovertext"]):
        assert f"vs without: {raw:+.2f}%" in label
        assert f"{custom[1]:.2f}% with management" in label
        assert label.endswith(f"Model uncertainty: {custom[2]:+.2f}% to {custom[3]:+.2f}%")
    assert any(raw != round(raw, 2) for raw in management["x"])
