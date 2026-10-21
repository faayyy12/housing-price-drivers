"""Real Streamlit runtime checks with database, network and fitting forbidden."""

import builtins
import json
from pathlib import Path
import socket

import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st
import pytest
from streamlit.testing.v1 import AppTest

from artifacts import MODEL_DIR, load_estimator, load_snapshots

ROOT = Path(__file__).resolve().parents[1]


def test_all_views_filters_and_estimator_without_services(monkeypatch):
    attempted = []
    def forbidden(*args, **kwargs):
        attempted.append("database, network, or training call")
        raise AssertionError("Serving must not access a database/network or train a model")
    real_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"db", "config", "psycopg2", "sqlalchemy", "dotenv"}:
            attempted.append(name)
            raise AssertionError(f"Serving imported preparation-only module {name}")
        return real_import(name, *args, **kwargs)
    # Warm only third-party imports; Streamlit caches are explicitly cold below.
    monkeypatch.setattr(pd, "read_sql", forbidden)
    monkeypatch.setattr(lgb, "train", forbidden)
    monkeypatch.setattr(lgb.LGBMRegressor, "fit", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setenv("DB_HOST", "remote-database-must-never-be-used.invalid")
    monkeypatch.setenv("DB_PASSWORD", "not-a-real-password")
    st.cache_data.clear()
    st.cache_resource.clear()
    app = AppTest.from_file(str(ROOT/"app/streamlit_app.py"), default_timeout=30).run()
    assert not app.exception and not app.error
    assert [x.label for x in app.tabs] == ["Price Patterns", "District Comparison", "Price Estimator"]
    assert len(app.get("plotly_chart")) == 4
    assert app.metric[1].value == "100,828"
    assert len(app.metric) == 3
    assert app.metric[2].value == "2014/05 – 2026/05"
    evaluation = next(x for x in app.expander if x.label == "Model evaluation")
    assert any("**0.86**" in x.value and "**0.84**" in x.value for x in evaluation.markdown)
    assert len(app.dataframe[0].value) == 15
    app.checkbox[0].check().run()
    assert len(app.dataframe[0].value) == 41
    app.radio[0].set_value("Taipei City").run()
    assert len(app.dataframe[0].value) == 12
    app.radio[0].set_value("New Taipei City").run()
    assert len(app.dataframe[0].value) == 29
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert len(app.metric) == 7
    assert all(x.value.startswith("NT$") for x in app.metric[3:])
    assert [x.label for x in app.metric[3:6]] == [
        "Estimated price per ping", "Lower end of estimated range", "Upper end of estimated range",
    ]
    # Two currency signs are safe in plain text, which does not parse math.
    assert any(x.value.startswith("Model-generated total-price range: NT$") and x.value.count("NT$") == 2 for x in app.text)
    # Townhouse/whole-building and alternate city district path.
    app.selectbox[0].set_value("三峽區")
    app.selectbox[1].set_value("透天厝")
    app.checkbox[1].check()
    app.checkbox[2].uncheck()
    app.run()
    app.button[0].click().run()
    assert not app.exception and not app.error and len(app.metric) == 7
    assert not attempted


@pytest.mark.parametrize("predictions", [
    [500000.0, 400000.0, 450000.0],
    [300000.0, 500000.0, 400000.0],
    [500000.0, 450000.0, 400000.0],
])
def test_crossing_quantiles_are_flagged_without_reordering(predictions):
    # Exercise rendering with known crossing model outputs, while keeping actual
    # model loading/serialization covered in test_artifacts.
    source = (ROOT/"app/streamlit_app.py").read_text(encoding="utf-8")
    prefix = source.rsplit('if __name__ == "__main__":', 1)[0]
    test_source = f"__file__ = {str(ROOT/'app/streamlit_app.py')!r}\n" + prefix + '''
df, dim, manifest = load_dashboard_data()
models, metadata = load_quantile_models(manifest)
def predict_quantiles(*args):
    return [CROSSING_PREDICTIONS]
render_estimator(df, models, metadata)
'''.replace("CROSSING_PREDICTIONS", repr(predictions))
    crossing = AppTest.from_string(test_source, default_timeout=30).run()
    # __file__ in from_string points to Streamlit's temp dir, so use the real
    # repository paths in the test source rather than any current working dir.
    if crossing.exception:
        raise AssertionError(crossing.exception[0].message)
    crossing.button[0].click().run()
    assert not crossing.exception
    assert any("estimates are out of order" in x.value for x in crossing.warning)
    low, mid, high = predictions
    assert [x.value for x in crossing.metric[:3]] == [f"NT${mid:,.0f}", f"NT${low:,.0f}", f"NT${high:,.0f}"]
    captions = [x.value for x in crossing.caption]
    assert any("No total-price range is shown" in text for text in captions)
    assert not any("total-price range:" in x.value for x in list(crossing.text) + list(crossing.caption))
    assert any("real-world reliability has not been checked" in text for text in captions)


@pytest.mark.parametrize("loader", ["load_snapshots", "load_estimator"])
def test_missing_bundle_stops_app_with_actionable_error(monkeypatch, loader):
    import artifacts
    def missing(*args, **kwargs):
        raise artifacts.ArtifactError("Missing test artifact: restore the deployment bundle")
    st.cache_data.clear()
    st.cache_resource.clear()
    monkeypatch.setattr(artifacts, loader, missing)
    app = AppTest.from_file(str(ROOT/"app/streamlit_app.py"), default_timeout=30).run()
    assert not app.exception
    assert len(app.error) == 1 and "cannot open its saved data" in app.error[0].value
    assert any("Missing test artifact" in x.value for x in app.text)
    assert any("COMMUNITY_CLOUD.md" in x.value for x in app.markdown)
    assert not app.tabs and not app.metric
