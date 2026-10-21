# Housing Price Drivers in Taipei & New Taipei

This historical project investigates housing prices in Taipei and New Taipei
using public Taiwan Ministry of the Interior (MOI) sale records. SQL, OLS
regression and LightGBM examine how recorded property characteristics relate
to **price per ping** (坪, about 3.3 m²).

**[Explore the live dashboard](https://housing-price-drivers-elkgtgrynzsmuiyvzuavq9.streamlit.app/)**

## Key findings

- **District:** OLS estimates prices per ping in 大安區 (Da'an) **209.3% higher
  than 三峽區 (Sanxia)** after accounting for recorded property differences.
  This is not a simple comparison of district medians.
- **Management:** adjusted prices with management
  are **10.3% lower for 公寓 (walk-ups)** and **8.8% lower for 透天厝 (townhouses)**
  in the analysis. High-rise comparisons are complicated by the younger
  buildings without management.
- **Age:** LightGBM's estimated pattern rises **27.6% for townhouses** and
  **8.8% for high-rises** between ages 40 and 50. This is a model-estimated
  relationship, not appreciation of an individual home.

These observational findings do not establish cause and effect.
See [detailed findings and evidence](docs/FINDINGS.md).

## Data and analytical approach

The saved dataset contains **100,828 cleaned residential sales**, dated
**2014-05-21–2026-05-30**. The models use **98,054 rows** with the required age
and floor details; the district summaries also include the other sales.

1. **Ingest and clean:** Python/pandas parse MOI CSVs, ROC dates and Chinese
   floor descriptions into PostgreSQL staging and cleaned-sale tables.
   Residential-sale filters separate rejected records; extreme prices are
   capped at the 1st/99th percentiles.
2. **Summarize with SQL:** district counts and medians describe recorded sales;
   medians are never model inputs.
3. **Model and evaluate:** the notebook uses statsmodels OLS, LightGBM, SHAP and
   partial-dependence plots. Separate temporal fits check performance on newer sales.
4. **Present:** Streamlit and Plotly turn saved results into an interactive dashboard.

## Dashboard features

- **Price Patterns:** explore adjusted district comparisons, management by
  building type, estimated age/floor patterns and feature importance.
- **District Comparison:** filter median sale prices per ping and sale counts
  by city. These summaries do not adjust for the mix of properties sold.
- **Price Estimator:** enter property details for an estimated median price per
  ping, a model-generated range and an area-based total. It assumes **May 2026**
  and no parking space. This is a teaching tool, not an appraisal.

## Model evaluation and limitations

The target is **natural-log price per ping**. R² measures model fit on that
scale; it is not percentage accuracy.

| Analysis model | R² on full-sample training data | R² in separate temporal evaluation |
|---|---:|---:|
| OLS | 0.77 | 0.75 |
| LightGBM squared-error regression | 0.86 | 0.84 |

The temporal evaluation fits separate models on **78,443 older sales** and
tests them on **19,611 newer sales**. These scores **do not evaluate the
reconstructed quantile models used by the Price Estimator**.

- **Estimator uncertainty:** its 5th–95th percentile range has unverified
  coverage; a 90% chance of containing a future sale price is not established.
  Quantiles cross for **309 of 98,054 modeling inputs**. The app warns, keeps
  individual estimates unchanged and omits the total-price range for those inputs.
- **Reconstruction:** the saved estimator was reconstructed from the preserved
  data. Original deployment model artifacts are unavailable, so exact prediction
  equivalence cannot be verified.
- **Data limits:** renovation, views and precise transport distance are missing.
  Parking included with a recorded price of zero can understate unit prices.
  The dataset does not refresh automatically, and historical evaluation does
  not establish future forecasting performance.

## Run locally

From the repository root, use **Python 3.12**:

```bash
python -m venv .venv-serving
```

Activate it with `.\.venv-serving\Scripts\Activate.ps1` in Windows PowerShell,
or `source .venv-serving/bin/activate` on macOS/Linux. Then run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

The dashboard uses included snapshots and models; no database, credentials or
training are needed. Linux hosts also need `libgomp1`; see the deployment guide.
Optional offline ingestion, database work, export and training have separate
dependencies and procedures in the [pipeline guide](docs/PIPELINE.md).

## Further documentation

- [Findings and methodology](docs/FINDINGS.md): detailed results, evaluation and caveats.
- [Analysis notebook](notebooks/analysis.ipynb): source code and saved analytical outputs.
- [Community Cloud deployment](deploy/COMMUNITY_CLOUD.md): serving configuration, dependencies and verification.
- [Offline pipeline and saved artifacts](docs/PIPELINE.md): database preparation, export, training and artifact contracts.
