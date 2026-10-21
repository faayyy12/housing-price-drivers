"""Serving-only AppTest smoke check; works without preparation/test packages.

Run with the environment containing only root requirements.txt. This is a real
Streamlit runtime test, not a browser rendering or external deployment test.
"""

import builtins
import base64
from importlib.util import find_spec
import json
import os
from pathlib import Path
import socket
import sys
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lightgbm as lgb
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest


def main():
    absent = {name: find_spec(name) is None for name in ["psycopg2", "sqlalchemy", "dotenv", "sklearn", "statsmodels", "shap", "matplotlib"]}
    attempted = []
    def forbidden(*args, **kwargs):
        attempted.append("database/network/training")
        raise AssertionError("The serving smoke check forbids database, network and training calls")
    real_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"db", "config", "psycopg2", "sqlalchemy", "dotenv"}:
            attempted.append(name)
            raise AssertionError(f"Preparation import from serving path: {name}")
        return real_import(name, *args, **kwargs)
    pd.read_sql = forbidden
    lgb.train = forbidden
    lgb.LGBMRegressor.fit = forbidden
    socket.socket.connect = forbidden
    socket.create_connection = forbidden
    builtins.__import__ = guarded_import
    os.environ["DB_HOST"] = "must-not-connect.invalid"
    st.cache_data.clear()
    st.cache_resource.clear()
    started = time.perf_counter()
    app = AppTest.from_file(str(ROOT/"app/streamlit_app.py"), default_timeout=30).run()
    cold = time.perf_counter()-started
    assert not app.exception and not app.error
    assert len(app.tabs) == 3 and len(app.get("plotly_chart")) == 4
    images = app.get("imgs")
    assert len(images) == 1
    image_uri = images[0].proto.imgs[0].url
    assert image_uri.startswith("data:image/png;base64,")
    assert base64.b64decode(image_uri.split(",", 1)[1], validate=True) == (ROOT/"app/assets/shap_summary.png").read_bytes()
    assert "三峽區 (Sanxia)" in app.selectbox[0].options
    assert "住宅大樓 (High-rise)" in app.selectbox[1].options
    assert app.metric[1].value == "100,828"
    assert app.metric[2].value == "2014/05 – 2026/05"
    started = time.perf_counter()
    app.button[0].click().run()
    warm_estimate = time.perf_counter()-started
    assert not app.exception and not app.error and len(app.metric) == 7
    default_predictions = [x.value for x in app.metric[3:]]
    app.checkbox[0].check().run()
    assert len(app.dataframe[0].value) == 41
    app.radio[0].set_value("Taipei City").run()
    assert len(app.dataframe[0].value) == 12
    app.radio[0].set_value("New Taipei City").run()
    assert len(app.dataframe[0].value) == 29
    assert not app.exception and not app.error and not attempted
    total_floors = next(x for x in app.number_input if x.label == "Total floors")
    total_floors.set_value(5).run()
    assert app.error and app.button[0].disabled
    assert app.selectbox[1].value == "住宅大樓(11層含以上有電梯)"
    next(x for x in app.number_input if x.label == "Total floors").set_value(14).run()
    assert not app.error and not app.exception and not app.button[0].disabled
    # Native-model regression probes are usable without scikit-learn.
    from artifacts import MODEL_DIR, checked_path, load_estimator, load_snapshots, predict_quantiles, read_json
    import numpy as np
    _, _, manifest = load_snapshots()
    models, metadata = load_estimator(manifest)
    checks = read_json(checked_path(MODEL_DIR, metadata["prediction_checks"]))
    predictions = predict_quantiles(models, metadata, pd.DataFrame(checks["feature_rows"]))
    np.testing.assert_allclose(predictions, checks["expected_quantile_predictions_ntd_per_ping"], rtol=1e-12, atol=1e-6)
    print(json.dumps({
        "passed": True, "views": 3, "plotly_charts": 4,
        "original_shap_png_delivered_inline": True, "invalid_high_rise_blocked": True,
        "city_filter_rows": [41, 12, 29], "serialization_probes": len(predictions),
        "preparation_modules_absent": absent,
        "cold_app_run_seconds_excluding_imports": round(cold, 3),
        "warm_estimator_rerun_seconds": round(warm_estimate, 3),
        "total_seconds_including_imports_and_checks": round(time.perf_counter()-STARTED, 3),
        "default_estimator_metrics": default_predictions,
        "database_network_training_attempts": attempted,
        "working_directory": str(Path.cwd()),
    }, indent=2))


if __name__ == "__main__":
    main()
