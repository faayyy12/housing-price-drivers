# What Drives Housing Prices in Taipei and New Taipei

This observational analysis uses MOI registration data: **100,828 cleaned
residential transactions**, dated **2014-05-21–2026-05-30**, across Taipei City's
12 districts and New Taipei City's 29 districts. Of these, 100,724 date from
2023 onward and 104 are earlier registrations retained by the original pipeline.
The OLS and LightGBM analyses use the same **98,054 complete cases** after
excluding 2,774 rows missing building age or total floors. Whole-building sales
with missing floor ratio are retained with floor ratio imputed to 1.0.

## Location has the strongest recorded association

The historical OLS coefficient for 大安區 (Da'an) corresponds to **+209.336%
relative to 三峽區 (Sanxia)**, the omitted reference district, after controlling
for the other recorded features. That is about **3.09 times** the reference
district's fitted price per ping. It is not a comparison against the average or
cheapest district. The chart contains 40 coefficients; Sanxia's reference value
is zero and is omitted.

In the original notebook's OLS design matrix, pandas
`get_dummies(df['district'], prefix='district', drop_first=True)` omits the
first category in sorted order, 三峽區 (Sanxia); `sm.add_constant(X)` supplies
the intercept. Sanxia is therefore the encoding reference. Each percentage
compares another district with Sanxia, holding the other modeled features
constant. Changing the reference for the same OLS specification would change
the comparisons and intercept, not fitted predictions.

The historical LightGBM SHAP summary also ranks district first: mean absolute
SHAP is about 0.2880, versus 0.1111 for building age (about 2.59 times as large).
This is a comparison of model attribution magnitudes, not a causal decomposition.

### Reading the original SHAP image

Notebook cells 27–28 fit the original LightGBM squared-error model to
`log_price_per_ping`, then call `shap.TreeExplainer(lgbm_model)` with no
background data or output transformation. The default raw regression output
therefore remains natural-log price per ping. Contributions add to the
explainer's expected model output, using training-sample counts along the tree
paths for its background distribution. This is not a comparison with a literal
average listing or with Sanxia. The numerical baseline was not printed or
exported by the notebook and has not been reconstructed. See the
[TreeExplainer API](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
for its raw-output, background and additivity definitions.

Cell 45 exports the summary for 5,000 sampled rows. The image's x-axis values
are log-price contributions, not dollars or percentages. Pink/red and blue
encode higher and lower numerical feature values; district and building-type
rows are gray categorical values, not price rankings. The PNG is preserved;
the app explains its original labels alongside it.

## Management associations differ by building type

Pooling all building types gave a management-organization coefficient of about
−8.8%, but management status is strongly associated with building type. Within
each type, the historical adjusted associations are:

| Building type | Modeling rows | Managed share | Adjusted price association |
|---|---:|---:|---:|
| Townhouse (透天厝) | 2,525 | 42.6% | −8.8% |
| Walk-up (公寓) | 27,727 | 6.6% | −10.3% |
| Mid-rise (華廈) | 18,608 | 84.7% | −0.9% |
| High-rise (住宅大樓) | 49,194 | 98.3% | +6.4% |

The saved chart includes the OLS confidence intervals. Unmanaged high-rises are
predominantly very new buildings: median age about one year, with about 70% aged
two years or less. A linear age control may not capture their new-building
premium. These results do not establish that management raises or lowers prices;
they support reporting the stratified associations instead of one universal
management premium or discount.

The saved notebook's age check reports 840 unmanaged high-rises, with median
age one year and 69.6% aged two or less, versus median age 15 and 15.7% aged two
or less among managed high-rises. These records support the age imbalance, but
do not establish why a management group was absent. A claim that committees had
not formed yet goes beyond the recorded evidence.

## The saved age curve falls and then rises

The historical LightGBM partial-dependence curve reaches its minimum around age
40 before rising. Between ages 40 and 50, the preserved JSON values imply:

- Townhouses: **346,900.10 → 442,766.66 NTD/ping**, an increase of **27.635%**.
- High-rises: **430,692.84 → 468,776.72 NTD/ping**, an increase of **8.842%**.

These are model partial-dependence values, not repeated sales of the same home
as it aged. Land value or scarcity is a possible explanation, not an established
mechanism. OLS uses a linear age term and cannot independently demonstrate the
nonlinear reversal. Floor position has a smaller SHAP attribution than age or
district.

## Historical evaluation, separate from the serving estimator

The original notebook records the following results. R² is measured on **natural
log price per ping**; the MAE below is transformed back to **NTD/ping**.

| Historical model | Full-sample R² | Temporal-holdout R² | Holdout MAE, NTD/ping |
|---|---:|---:|---:|
| OLS | 0.765116 | 0.745996 | 97,362.78 |
| LightGBM squared-error regression | 0.861693 | 0.843745 | 74,907.54 |

The temporal split trains on 78,443 older transactions and tests on 19,611 newer
transactions. The boundary date is April 19, 2025; that date appears in both
partitions because the split is by ordered row count, not a strict date cutoff.
These checks support interpretation within this dataset; they do not establish
future forecasting performance or causal effects.

The dashboard's three LightGBM **quantile** models were reconstructed locally
from the preserved transaction snapshot using the former app's feature order,
preprocessing, hyperparameters and random seed. They are not the notebook's
squared-error model, and the historical R² figures must not be attributed to
them. Original VPS artifacts are unavailable, so exact former predictions cannot
be established. Their 5th–95th percentile range is a **nominal 90% interval**;
actual coverage has not received a new out-of-sample evaluation. The independent
quantiles cross for **309 of 98,054 modeling inputs**. Outputs remain unchanged,
and the dashboard warns when an entered input produces a crossing. Individual
quantiles remain visible, but no implied total-price interval is presented.

## Preserved evidence and corrections

The original `notebooks/analysis.ipynb`, `app/assets/*_data.json` and PNG images
remain unchanged. This text corrects the former reference-district wording,
coverage labels and claims that both models demonstrated the age reversal.
The notebook narrative's 28.6%/10.7% age changes are inconsistent with its own
saved outputs; the 27.635%/8.842% figures above come directly from the preserved
JSON. Historical screenshots and unused PNG chart labels predate these corrections.
No analysis was rerun or replacement finding manufactured for this migration.

The source data is observational, district-level location is coarse, the target
was winsorized at the 1st/99th percentiles, and zero-priced bundled parking can
understate some unit prices. The estimator is a teaching tool, not an appraisal.
The serving snapshot does not automatically refresh.

## Input wording and display precision

`src/load.py` maps `建物現況格局-房`, `建物現況格局-廳` and
`建物現況格局-衛` to `rooms`, `halls` and `baths`. The source's
[layout guide](https://olvr.land.moi.gov.tw/jsp/guide.jsp) uses 房/廳/衛 counts.
The app labels 廳 as living/dining areas to distinguish it from corridors;
it continues to use the single recorded count, without splitting it or changing
any model input. 房 remains the recorded room count, rather than being relabeled
as bedrooms without a more detailed source definition.

Currency labels and table cells show whole NT$ with thousands separators.
Analytical hover values use at most two decimals. Only display formatting
changes; stored values, prediction inputs, quantile outputs, crossing checks
and all underlying calculations retain their existing precision.
