# Streamlit Community Cloud: fixed-snapshot dashboard

The snapshot dashboard is hosted at the owner's supplied
[live app URL](https://housing-price-drivers-elkgtgrynzsmuiyvzuavq9.streamlit.app/).
Use the settings below for deployment or redeployment. A successful local test
or Git push alone does not establish that the hosted app has rebuilt successfully.

## Deployment settings

| Setting | Value |
|---|---|
| GitHub repository | `faayyy12/housing-price-drivers` |
| Branch | `main` |
| Main file path | `app/streamlit_app.py` |
| Python version | **3.12**, explicitly selected in Advanced settings |
| Local reference interpreter | CPython **3.12.6**, Windows |
| Python dependencies | root `requirements.txt`, serving-only pinned versions |
| Debian dependency | root `packages.txt`: `libgomp1` for LightGBM's OpenMP runtime |
| Secrets | none |
| Refresh | none; the data ends **2026-05-30** |

Community Cloud searches the entrypoint directory, then repository root, for a
supported dependency specification. There is deliberately no `app/requirements.txt`;
root `requirements.txt` is selected. Files under `requirements/` are optional
offline preparation/development inputs, not automatically discovered app specs.
Use Python 3.12 explicitly rather than relying on the cloud default.

References: [dependency discovery](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
and [deployment/Python selection](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
The explicit protobuf pin stays below version 6, as advised by the
[Community Cloud compatibility notes](https://docs.streamlit.io/deploy/streamlit-community-cloud/status).
The `libgomp1` declaration covers the separately installed OpenMP runtime
described in [LightGBM 4.6.0's package instructions](https://github.com/lightgbm-org/LightGBM/blob/v4.6.0/python-package/README.rst).
Linux/Debian installation has not been executed on this Windows machine.

## Required deployment files

Keep these non-sensitive files in the reviewed Git change:

```text
requirements.txt
packages.txt
.gitattributes
.streamlit/config.toml
app/streamlit_app.py
app/assets/*_data.json
app/assets/*.png
src/artifacts.py
src/model_features.py
data/snapshots/fct_transactions.parquet
data/snapshots/dim_district.parquet
data/snapshots/manifest.json
models/estimator/quantile_05.txt
models/estimator/quantile_50.txt
models/estimator/quantile_95.txt
models/estimator/metadata.json
models/estimator/prediction_checks.json
notebooks/analysis.ipynb
```

The manifest hashes the preserved notebook and all original analysis assets as
evidence, so those files also remain in the bundle. Snapshot readers verify
schemas/counts and hashes; model readers verify model hashes, feature order,
categories and the training-snapshot association. The app fails visibly if the
bundle is incomplete or inconsistent. It never retrains or falls back to demo
data. Native model prediction uses one inference thread; this does not change
the fitted model parameters.

`.gitattributes` preserves generated artifact bytes across Git checkouts. The
original notebook and JSON evidence use explicitly LF-normalized hashes, so
Windows CRLF and Linux LF checkouts validate without altering that evidence.

The snapshots explicitly exclude transaction IDs, addresses, notes, original
source-file references and per-row load timestamps. `.env`, Streamlit secrets,
raw MOI files, private backups, internal milestone notes, environments and
temporary outputs are ignored. Do not add them to the deployment change.

The app does not import `src/db.py` or `src/config.py`. Neither PostgreSQL clients
nor database/Telegram credentials, MOI access, cron or VPS reverse proxies are
required. Keep the original pipeline source for offline preparation only.

## Local verification

Run from repository root with Python 3.12:

```bash
python -m pip install -r requirements/development.txt
python -m pytest -q -p no:cacheprovider tests
python scripts/export_snapshots.py --verify-only
```

The last command requires the existing **local** PostgreSQL database and private
`.env`; it enforces read-only repeatable-read access and changes no files or
tables. It compares every exported value with the source tables, verifies the
100,828/41 populations and checks district counts/medians. Serving tests forbid
database imports, SQL reads, network connections and model fitting while running
Streamlit's AppTest, changing filters and requesting estimates.

For a serving-only check, create a fresh local environment and install only
root `requirements.txt`, then run the smoke check and launch:

```bash
python scripts/verify_serving.py
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
```

The migration's 62-test baseline passed before publication review. The final
review's 13 focused artifact/serving checks passed, covering snapshot/model
integrity, saved prediction probes, all three views,
filters, missing-artifact errors and crossed-quantile display. For crossed cases,
the app preserves individual outputs but does not display a total-price interval.
Nominal 90% coverage remains unverified. A clean serving-only environment and a
local HTTP health check passed. Browser layout, Linux installation and hosted
execution remain untested. Internal working reports are kept locally, outside Git.

The SHAP chart uses the original PNG as an inline data URI accepted by the pinned
Streamlit version. This avoids a separate `/media` request and does not depend on
the working directory or a static-file copy. Regression tests decode the actual
`st.image` payload and compare it byte-for-byte with the protected asset. Other
UI checks cover category/floor validation, bilingual labels mapped to original
categories, full-precision numeric table values with formatted display, and the
coverage card's wrapping styles. AppTest does not verify visual layout: check
the full date range at desktop (about 1363 × 936) and narrower widths in a browser
after a hosted rebuild.

## Updating a snapshot later

The preserved population includes **104 pre-2023 records** and is not filtered
to match the old dashboard label. The 98,054-row model sample is separate from
the complete population. The stored district medians never become model inputs.

An update is an explicit offline process: preserve/version the old artifact
bundle, deliberately refresh/rebuild the local database if desired, export with
`scripts/export_snapshots.py`, train with `scripts/train_estimator.py`, reconcile
all hashes/populations and test again. The scripts refuse silent overwrites and
currently guard the inspected population sizes. A changed population needs
explicit review of those guards and of the historical chart/narrative alignment.
Do not combine refreshed estimator data with historical chart evidence without
labeling the different vintages or reviewing a separate analytical refresh.

## Estimator provenance

The quantile models were reconstructed locally from the dated snapshot using
the former app's preprocessing, features, parameters and seed. Original VPS
model artifacts are unavailable, so exact equivalence to the former VPS
predictions cannot be established. Their interval is nominal; independent
quantile crossings are disclosed and are never repaired silently. Historical
notebook metrics describe the old
analysis models, not new validation of the reconstructed estimator.
