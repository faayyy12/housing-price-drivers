# Offline pipeline and saved artifacts

The dashboard runs from included files. None of the database preparation,
downloads, exports or training below is needed to view it. For serving setup,
start with the [README](../README.md#run-locally); hosting configuration is in
the [Community Cloud guide](../deploy/COMMUNITY_CLOUD.md).

## Preparation environment

Use Python 3.12 in a separate environment, with PostgreSQL and its `psql` client
available for database work. From the repository root:

```bash
python -m venv .venv-preparation
```

Activate `.\.venv-preparation\Scripts\Activate.ps1` in Windows PowerShell, or
`source .venv-preparation/bin/activate` on macOS/Linux. Install offline dependencies:

```bash
python -m pip install -r requirements/preparation.txt
```

For the test suite, install the development requirements instead:

```bash
python -m pip install -r requirements/development.txt
python -m pytest -q -p no:cacheprovider tests
```

The development requirements include the preparation and serving dependencies.
The serving-only smoke check needs just root `requirements.txt`:

```bash
python scripts/verify_serving.py
```

## Prepare a database deliberately

Copy [`.env.example`](../.env.example) to a private `.env` and configure the
local database. Python scripts read it; `psql` does not automatically do so.
The commands below use example connection settings: replace the role, port and
database with your local configuration. Create the database first.

**`sql/01_schema.sql` drops the project tables. Use it only for an intentional
fresh rebuild, with any existing database backed up.** These commands modify
the selected database:

```bash
psql -h localhost -p 5432 -U YOUR_LOCAL_ROLE -d housing_prices -f sql/01_schema.sql
psql -h localhost -p 5432 -U YOUR_LOCAL_ROLE -d housing_prices -f sql/02_fct_transactions.sql
```

The first file creates staging, cleaned-sale, rejected-sale and district tables.
The second adds indexes and constraints. Complex parsing and transformations
are in [`src/clean.py`](../src/clean.py) and [`src/quality.py`](../src/quality.py),
not in a SQL transform query.

For the historical quarterly range used in preparation, these commands download
files and then load them into staging:

```bash
python src/ingest.py --seasons 112S3 115S1
python src/load.py --seasons 112S3 115S1
```

Skip downloads if the required raw files already exist. To load the two retained
current-batch files (`a_lvr_land_a.csv` and `f_lvr_land_a.csv`) and build the
cleaned tables and district summaries:

```bash
python src/load.py
python src/quality.py
psql -h localhost -p 5432 -U YOUR_LOCAL_ROLE -d housing_prices -f sql/03_dim_district.sql
```

The source's “current” batch changes over time. These preparation commands do
not promise to recreate the saved population from fresh downloads today.
[`scripts/refresh.sh`](../scripts/refresh.sh) is a Bash workflow that downloads
the current batch, loads staging, runs cleaning/quality checks and rebuilds the
district summaries. It changes the database; it does not update the dashboard's
saved snapshot or models. It prefers `.venv/bin/python`, otherwise an interpreter
on `PATH`. Ensure the selected interpreter has the preparation dependencies;
an existing `.venv` takes precedence over an activated environment. Optional alert
settings and earlier hosting procedures are archived in
[`deploy/DEPLOY.md`](../deploy/DEPLOY.md), not required for serving.

The [analysis notebook](../notebooks/analysis.ipynb) reads PostgreSQL and needs
a Jupyter environment with preparation dependencies. Its saved outputs can be
reviewed without rerunning it. Running its analysis/export cells fits models
and writes chart assets; do not rerun them as a dashboard startup step or as a
documentation check.

## Verify an existing snapshot: no file writes

With the existing local database and private connection settings available:

```bash
python scripts/export_snapshots.py --verify-only
```

This uses local-only, repeatable-read, read-only database access. It compares
every exported field with the saved Parquet files, reconciles district counts
and medians, and validates the saved bundle. It writes no snapshot files and
does not train a model.

## Export snapshots: reads the database, writes files

For an explicitly reviewed new bundle:

```bash
python scripts/export_snapshots.py
```

Database access is still read-only, but this command **writes Parquet files and
a manifest** under `data/snapshots/`. It refuses to overwrite the existing bundle.
Preserve/version the prior bundle before deliberately preparing a replacement.
The exporter guards the inspected population sizes of 100,828 sales and 41
districts; a changed population requires review of those checks.

PostgreSQL NUMERIC fields become float64, matching the pandas model input path.
The manifest records conversion errors; nullable integers, booleans and dates
have explicit Parquet types. Exported row order is retained for reconstruction.
Case IDs, addresses, notes, source-file names and per-row load timestamps are
excluded through an explicit column allowlist.

## Train the estimator: reads snapshots, writes model artifacts

This is a separate, deliberate operation in the preparation environment:

```bash
python scripts/train_estimator.py
```

It reads the validated snapshots, fits three LightGBM quantile models, and
**writes model files, metadata and prediction checks** under `models/estimator/`.
It does not require a database. It refuses to run when model files or metadata
already exist; preserve/version the prior bundle before preparing a replacement.
Training uses the exported row order and guards the 98,054-row modeling population.

The models use the saved preprocessing contract, feature order, categorical
values, parameters and seed. They are reconstructed models; the original
deployment artifacts are unavailable, so exact prediction equivalence cannot
be established. Serialization probes verify saved-model predictions, not
out-of-sample range coverage. See [findings and evaluation](FINDINGS.md).

## Saved bundle and integrity checks

| Location | Contents |
|---|---|
| `data/snapshots/fct_transactions.parquet` | All 100,828 cleaned sales, using the non-identifying export column allowlist. |
| `data/snapshots/dim_district.parquet` | 41 district sale-count and median-price summaries; display-only, never estimator features. |
| `data/snapshots/manifest.json` | Schemas, populations, dates, hashes, provenance, numeric representation and database reconciliation results. |
| `models/estimator/quantile_05.txt`, `quantile_50.txt`, `quantile_95.txt` | Native LightGBM models for the 5th, 50th and 95th conditional percentiles. |
| `models/estimator/metadata.json` | Features, categories, time origin, preprocessing rules, snapshot association, parameters, versions and diagnostics. |
| `models/estimator/prediction_checks.json` | Feature-only regression inputs and pre-serialization predictions, without transaction identifiers. |
| `notebooks/analysis.ipynb`, `app/assets/` | Preserved notebook, chart data and original analysis images, validated as analytical evidence. |

The cleaned population is distinct from the 98,054 modeling rows and from the
smaller chart summaries. The saved data retains 104 pre-2023 sales and the
existing cleaning and winsorization decisions. `src/artifacts.py` validates
snapshot/model hashes, schemas, counts, feature/category order and model/snapshot
association. Missing or inconsistent files produce a visible error, not demo data.
Generated artifacts have Git byte-preservation rules; notebook/JSON evidence
hashes normalize CRLF/LF line endings.

The dashboard does not automatically refresh or train. A new bundle requires
explicit preparation, reconciliation, model checks and a review of how new data
relates to the historical findings. Do not silently mix refreshed estimator data
with unchanged historical charts. Keep `.env`, credentials, raw records, internal
notes, backups and environments out of Git. Detailed deployment file requirements
and reconstruction context are in the [Community Cloud guide](../deploy/COMMUNITY_CLOUD.md).
