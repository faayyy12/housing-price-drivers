"""Streamlit dashboard for the Taiwan housing price-driver analysis.

Three views: driver explorer, district comparison, price estimator. Reads
validated local snapshots and reconstructed native LightGBM models. The model NEVER reads
dim_district (district medians) as a feature; that table is dashboard
display only (see M0_NOTES.md [M6] and M2_notes.md leakage check).

Headline is the district premium (Q3 in the design doc / M2_notes.md) --
structural, largest SHAP. The management-organization finding (Q1) is shown
ONLY as the stratified, heterogeneous-by-building-type view; a single
pooled discount/premium number is never shown anywhere in this app -- that
oversimplification was explicitly identified and corrected in M2. (The
Q1/Q2/Q3 question numbering is internal shorthand tying this code back to
M2_notes.md -- it is deliberately never surfaced in the rendered UI, which
uses plain descriptive titles instead.)

The three Driver Explorer charts (district premium, management by type,
age/floor partial dependence) are interactive Plotly charts built from
JSON exported by notebooks/analysis.ipynb (app/assets/*_data.json) -- the
numbers are the notebook's exact output, not recomputed here. SHAP summary
stays a static PNG (Plotly has no native beeswarm equivalent worth
reimplementing for this).

Price estimator returns a prediction INTERVAL via LightGBM quantile
regression (alpha=0.05/0.5/0.95), not a point estimate. Quantile predictions
are transform-invariant under exp() (monotonic), so no smearing correction
is needed here -- unlike an OLS *mean* prediction, which would require one.
That's the reason this project uses the LightGBM-quantile option from the
two the design doc allowed, not OLS-smearing.
"""

import base64
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from artifacts import ArtifactError, load_snapshots, load_estimator, predict_quantiles, read_json  # noqa: E402
from model_features import FEATURE_COLS, preprocess_population  # noqa: E402

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

ACCENT = "#2563EB"
POSITIVE = "#16A34A"
NEGATIVE = "#DC2626"

# Standard romanized English names for Taipei City + New Taipei City
# districts. Display-only (never used in the model or joined into any
# query) -- lets a non-Chinese-reading reviewer follow the district charts
# without abandoning the decision to keep domain terms (district names,
# building types) in Chinese, since those are the source data's own terms.
DISTRICT_EN = {
    "大安區": "Da'an", "中正區": "Zhongzheng", "信義區": "Xinyi", "松山區": "Songshan",
    "大同區": "Datong", "中山區": "Zhongshan", "士林區": "Shilin", "北投區": "Beitou",
    "內湖區": "Neihu", "南港區": "Nangang", "文山區": "Wenshan", "萬華區": "Wanhua",
    "板橋區": "Banqiao", "三重區": "Sanchong", "中和區": "Zhonghe", "永和區": "Yonghe",
    "新莊區": "Xinzhuang", "新店區": "Xindian", "樹林區": "Shulin", "鶯歌區": "Yingge",
    "三峽區": "Sanxia", "淡水區": "Tamsui", "汐止區": "Xizhi", "瑞芳區": "Ruifang",
    "土城區": "Tucheng", "蘆洲區": "Luzhou", "五股區": "Wugu", "泰山區": "Taishan",
    "林口區": "Linkou", "深坑區": "Shenkeng", "石碇區": "Shiding", "坪林區": "Pinglin",
    "三芝區": "Sanzhi", "石門區": "Shimen", "八里區": "Bali", "平溪區": "Pingxi",
    "雙溪區": "Shuangxi", "貢寮區": "Gongliao", "金山區": "Jinshan", "萬里區": "Wanli",
    "烏來區": "Wulai",
}

# Short Chinese display labels for the four official building-type
# categories (Taiwan's 實價登錄 schema defines exactly these four -- see
# M0_NOTES.md -- trimmed here of the descriptive floor-count parenthetical
# for chart readability; the full category name is still used as the model
# feature value, unaffected).
BUILDING_TYPE_ORDER = [
    "住宅大樓(11層含以上有電梯)", "華廈(10層含以下有電梯)",
    "公寓(5樓含以下無電梯)", "透天厝",
]
BUILDING_TYPE_SHORT = {
    "住宅大樓(11層含以上有電梯)": "住宅大樓", "華廈(10層含以下有電梯)": "華廈",
    "公寓(5樓含以下無電梯)": "公寓", "透天厝": "透天厝",
}
BUILDING_TYPE_EN = {
    "住宅大樓(11層含以上有電梯)": "High-rise", "華廈(10層含以下有電梯)": "Mid-rise",
    "公寓(5樓含以下無電梯)": "Walk-up", "透天厝": "Townhouse",
}
BUILDING_TYPE_DEFAULT_FLOORS = {
    "住宅大樓(11層含以上有電梯)": 14, "華廈(10層含以下有電梯)": 8,
    "公寓(5樓含以下無電梯)": 5, "透天厝": 3,
}


def district_label(d: str) -> str:
    en = DISTRICT_EN.get(d)
    return f"{d} ({en})" if en else d


def building_type_label(value: str) -> str:
    label = BUILDING_TYPE_SHORT.get(value, value)
    english = BUILDING_TYPE_EN.get(value)
    return f"{label} ({english})" if english else label


def floor_consistency(building_type: str, total_floors: int):
    """Validate category definitions without rewriting a selected model value.

    The preserved source includes rare walk-ups above five floors and tall
    townhouses, so those get warnings. Elevator mid-rises include 2–5 floors;
    the category defines an upper limit of ten, not a lower limit of six.
    """
    if building_type == "住宅大樓(11層含以上有電梯)" and total_floors < 11:
        return "A high-rise (住宅大樓) must have at least 11 floors. Check Total floors or choose a different building type.", None
    if building_type == "華廈(10層含以下有電梯)" and total_floors > 10:
        return "A mid-rise (華廈) has an elevator and no more than 10 floors. Check Total floors or choose a different building type.", None
    if building_type == "公寓(5樓含以下無電梯)" and total_floors > 5:
        return None, "Unusual combination: a walk-up (公寓) normally has no elevator and no more than 5 floors. A few recorded sales exceed this. Check the building type and floor count before estimating."
    if building_type == "透天厝" and total_floors > 5:
        return None, "Unusual townhouse height: most recorded townhouses (透天厝) have 2–4 floors. Some are taller. Check the building type and floor count before estimating."
    return None, None


def fmt_ntd(x: float) -> str:
    """Whole New Taiwan dollars; multi-currency strings use st.text, not Markdown."""
    return f"NT${x:,.0f}"


st.set_page_config(page_title="Taipei & New Taipei Home Prices", layout="wide")

# Light CSS: hide Streamlit's default chrome, tighten padding, style metric
# cards. Kept deliberately small -- no JS, no layout hacks beyond this.
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    header[data-testid="stHeader"] {visibility: hidden; height: 0;}
    footer {visibility: hidden;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px;}
    /* light-dark follows the active Streamlit app's CSS color-scheme. */
    div[data-testid="stMetric"] {
        background: light-dark(#F8FAFC, #262730);
        border: 1px solid light-dark(#E2E8F0, #4B5563);
        border-radius: 10px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] p,
    .stMainBlockContainer [data-testid="stCaptionContainer"] p {
        color: light-dark(#4B5563, #D1D5DB);
        font-size: 0.9375rem;
    }
    .stMainBlockContainer [data-testid="stCaptionContainer"] {
        opacity: 1;
    }
    .st-key-summary-cards [data-testid="stMetric"] {
        min-height: 152px;
        height: 100%;
    }
    .st-key-summary-cards [data-testid="stMetricLabel"] {
        min-height: 2.75rem;
        align-items: flex-start;
    }
    .st-key-summary-cards [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"],
    .st-key-summary-cards [data-testid="stMetricLabel"] p {
        white-space: normal;
        overflow: visible;
        text-overflow: clip;
    }
    .st-key-summary-cards [data-testid="stMetricValue"] {
        font-size: clamp(1.5rem, 2.75vw, 2.25rem);
    }
    .st-key-snapshot-coverage [data-testid="stMetricValue"] {
        font-size: clamp(1.125rem, 2vw, 1.625rem);
    }
    .st-key-snapshot-coverage [data-testid="stMetricValue"],
    .st-key-snapshot-coverage [data-testid="stMetricValue"] > div,
    .st-key-snapshot-coverage [data-testid="stMetricValue"] p {
        white-space: normal;
        overflow: visible;
        text-overflow: clip;
        overflow-wrap: anywhere;
    }
    .st-key-district-heading a[href="#location-has-the-strongest-recorded-association"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_dashboard_data():
    population, dim, manifest = load_snapshots()
    return preprocess_population(population), dim, manifest


@st.cache_data
def load_finding_json(name: str):
    return read_json(ASSETS_DIR / name)


@st.cache_data
def load_shap_image() -> str:
    """Embed the original PNG, avoiding a separate session-scoped /media request.

    Streamlit 1.58 accepts data URIs directly in st.image. Reading bytes before
    encoding also makes path failures explicit rather than passing a path to a
    media backend. The original analytical asset is never re-encoded or changed.
    """
    path = ASSETS_DIR / "shap_summary.png"
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ArtifactError("The feature chart image cannot be opened (shap_summary.png).") from exc
    if not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ArtifactError("The feature chart image is damaged or has the wrong file type (shap_summary.png).")
    return "data:image/png;base64," + base64.b64encode(content).decode("ascii")


@st.cache_resource
def load_quantile_models(manifest: dict):
    """Load native models and validate their feature/category/snapshot contract."""
    return load_estimator(manifest)


def _plotly_layout(fig: go.Figure, title: str, height: int = 420, bottom_margin: int = 10) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=50, b=bottom_margin),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#1F2937", size=14),
        showlegend=False,
    )
    # automargin: Plotly's fixed margin (set above) won't expand for long
    # tick labels on its own -- without this, labels like "貢寮區 (Gongliao)"
    # get silently clipped at the edge of the plot area regardless of how
    # wide the container is. automargin makes Plotly grow the margin to fit.
    fig.update_xaxes(gridcolor="#E2E8F0", zerolinecolor="#94A3B8", automargin=True)
    fig.update_yaxes(gridcolor="#E2E8F0", automargin=True)
    return fig


def chart_district_premium(data: list, top_n: int = 10) -> go.Figure:
    sorted_desc = sorted(data, key=lambda d: d["effect_pct"], reverse=True)
    combined = sorted_desc[:top_n] + sorted_desc[-top_n:]
    labels = [district_label(d["district"]) for d in combined]
    values = [d["effect_pct"] for d in combined]
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        # Send final display text, avoiding client-side numeric-format fallbacks.
        hoverinfo="text",
        hovertext=[f"{label}<br>Price per ping vs Sanxia: {value:+.2f}%" for label, value in zip(labels, values)],
    ))
    fig.update_xaxes(title=dict(text="Price difference from 三峽區 (Sanxia), %", standoff=20))
    return _plotly_layout(
        fig, "Price per ping compared with 三峽區 (Sanxia)", height=680, bottom_margin=70,
    )


def chart_management_by_type(data: list) -> go.Figure:
    rows = {d["building_type"]: d for d in data}
    rows_ordered = [rows[k] for k in BUILDING_TYPE_ORDER]
    # Chinese primary + English alongside, same "中文 (English)" pattern as
    # district_label(). The English names ("Townhouse"/"Walk-up"/etc.) are
    # already sitting in each row's "label" field from the notebook export
    # -- no new data needed, just combined with BUILDING_TYPE_SHORT here.
    y_labels = [f"{BUILDING_TYPE_SHORT[k]} ({rows[k]['label']})" for k in BUILDING_TYPE_ORDER]
    values = [r["effect_pct"] for r in rows_ordered]
    ci_low = [r["ci_low_pct"] for r in rows_ordered]
    ci_high = [r["ci_high_pct"] for r in rows_ordered]
    err_low = [r["effect_pct"] - r["ci_low_pct"] for r in rows_ordered]
    err_high = [r["ci_high_pct"] - r["effect_pct"] for r in rows_ordered]
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in values]
    texts = [f"{r['effect_pct']:+.1f}%" for r in rows_ordered]

    fig = go.Figure(go.Bar(
        x=values, y=y_labels, orientation="h", marker_color=colors,
        error_x=dict(type="data", symmetric=False, array=err_high, arrayminus=err_low),
        customdata=[[r["n"], r["managed_pct"], r["ci_low_pct"], r["ci_high_pct"]] for r in rows_ordered],
        hoverinfo="text",
        hovertext=[
            f"{label}<br>Price per ping with management vs without: {r['effect_pct']:+.2f}%"
            f"<br>{r['n']:,} sales; {r['managed_pct']:.2f}% with management"
            f"<br>Model uncertainty: {r['ci_low_pct']:+.2f}% to {r['ci_high_pct']:+.2f}%"
            for label, r in zip(y_labels, rows_ordered)
        ],
    ))
    # Plotly's built-in textposition="outside" places labels just past the
    # BAR's end -- it doesn't know the error-bar (CI) whisker extends further,
    # so the label lands inside the whisker's reach and the line visually cuts
    # through the text. Position labels manually, past whichever is further
    # out: the bar or its CI edge (matches the matplotlib version's approach).
    span = max(abs(v) for v in ci_low + ci_high) or 1
    pad = span * 0.06
    for i, (v, lo, hi, txt) in enumerate(zip(values, ci_low, ci_high, texts)):
        x_pos = hi + pad if v >= 0 else lo - pad
        fig.add_annotation(
            x=x_pos, y=y_labels[i], text=txt, showarrow=False,
            xanchor="left" if v >= 0 else "right",
            font=dict(size=13, color="#1F2937"),
            # yshift: y is a categorical axis (bar labels), so a numeric
            # y-offset can't be added directly -- yshift is a pixel-based
            # nudge applied at render time, independent of the category
            # lookup, moving the label above the gridline instead of
            # sitting directly on it.
            yshift=10,
        )
    fig.update_xaxes(
        title=dict(text="Price difference: with management vs without (%)", standoff=28),
        range=[min(ci_low) - span * 0.55, max(ci_high) + span * 0.55],
    )
    return _plotly_layout(
        fig, "Price comparisons within each building type", height=440, bottom_margin=100,
    )


def chart_age_floor(data: dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Building age", "Floor level"))

    overall = data["overall"]
    townhouse = data["townhouse"]
    highrise = data["highrise"]
    floor = data["floor_ratio"]

    fig.add_trace(go.Scatter(
        x=[p["age"] for p in overall], y=[p["price_per_ping"] for p in overall],
        name="All building types", mode="lines+markers", line=dict(color=ACCENT),
        hoverinfo="text",
        hovertext=[f"All building types<br>Age: {p['age']:.0f} years<br>Estimated price per ping: {fmt_ntd(p['price_per_ping'])}" for p in overall],
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[p["age"] for p in townhouse], y=[p["price_per_ping"] for p in townhouse],
        name="透天厝 (Townhouse)", mode="lines+markers", line=dict(color="#F59E0B"),
        hoverinfo="text",
        hovertext=[f"透天厝 (Townhouse)<br>Age: {p['age']:.0f} years<br>Estimated price per ping: {fmt_ntd(p['price_per_ping'])}" for p in townhouse],
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[p["age"] for p in highrise], y=[p["price_per_ping"] for p in highrise],
        name="住宅大樓 (High-rise)", mode="lines+markers", line=dict(color=POSITIVE),
        hoverinfo="text",
        hovertext=[f"住宅大樓 (High-rise)<br>Age: {p['age']:.0f} years<br>Estimated price per ping: {fmt_ntd(p['price_per_ping'])}" for p in highrise],
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[p["floor_ratio"] for p in floor], y=[p["price_per_ping"] for p in floor],
        name="Floor level", mode="lines+markers", line=dict(color="#7C3AED"),
        hoverinfo="text",
        hovertext=[f"Unit floor / total floors: {p['floor_ratio']:.2f}<br>Estimated price per ping: {fmt_ntd(p['price_per_ping'])}" for p in floor],
    ), row=1, col=2)

    # This chart builds its own layout (subplots) rather than going through
    # _plotly_layout, so automargin/standoff must be set explicitly here too
    # -- otherwise the y-axis title overlaps its own tick labels, the same
    # clipping issue automargin fixes on the other two charts.
    fig.update_xaxes(title=dict(text="Building age (years)", standoff=18), row=1, col=1, gridcolor="#E2E8F0", automargin=True)
    fig.update_xaxes(title=dict(text="Unit floor / total floors", standoff=18), tickformat=".2f", row=1, col=2, gridcolor="#E2E8F0", automargin=True)
    fig.update_yaxes(title=dict(text="Estimated price per ping", standoff=25), tickprefix="NT$", tickformat=",.0f", row=1, col=1, gridcolor="#E2E8F0", automargin=True)
    fig.update_yaxes(title=dict(text="Estimated price per ping", standoff=25), tickprefix="NT$", tickformat=",.0f", row=1, col=2, gridcolor="#E2E8F0", automargin=True)
    fig.update_layout(
        height=460, margin=dict(l=10, r=10, t=60, b=110),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(color="#1F2937", size=14),
        legend=dict(orientation="h", yanchor="bottom", y=-0.38, x=0.15),
    )
    return fig


def chart_district_comparison(view: pd.DataFrame) -> go.Figure:
    """Vertical bar chart (districts along the x-axis) -- kept in the same
    orientation as the st.bar_chart it replaces, not switched to horizontal
    like the other three charts, since the page caption explicitly says
    "the Y axis of the chart is the median price per ping." Fixes the label
    truncation st.bar_chart had (Vega-Lite truncates rotated category labels
    with "..." when many long labels are packed into limited width, and
    exposes no automargin-equivalent control) using the same automargin
    technique already used elsewhere, plus fully vertical tick labels so
    even 41-district views (the "Both" city filter) stay readable.
    """
    labels = view["district_label"].tolist()
    values = view["median_price_per_ping"].tolist()
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=ACCENT,
        hoverinfo="text",
        hovertext=[f"{label}<br>Median sale price per ping: {fmt_ntd(value)}" for label, value in zip(labels, values)],
    ))
    fig.update_xaxes(tickangle=-90)
    fig.update_yaxes(title=dict(text="Median sale price per ping", standoff=20), tickprefix="NT$", tickformat=",.0f")
    return _plotly_layout(
        fig, "Median sale price per ping by district", height=560, bottom_margin=140,
    )


def render_building_type_guide():
    """Use the same source definitions beside both relevant views."""
    with st.expander("Building-type guide"):
        st.markdown(
            "Taiwan's government data uses four building categories:\n\n"
            "- **公寓 (Walk-up):** at most 5 floors without an elevator "
            "(rare source records exceed this).\n"
            "- **華廈 (Mid-rise):** an elevator building with at most 10 floors, "
            "including some below 6.\n"
            "- **住宅大樓 (High-rise):** 11 or more floors with an elevator.\n"
            "- **透天厝 (Townhouse):** a standalone, whole-building sale."
        )


def render_driver_explorer(district_data: list, mgmt_data: list, age_floor_data: dict):
    with st.container(key="district-heading"):
        # Preserve the old section URL while hiding only its permalink icon.
        st.subheader("Prices differ by district", anchor="location-has-the-strongest-recorded-association")
    st.markdown(
        "Compare price per ping with 三峽區 (Sanxia), after accounting for recorded "
        "differences such as age, size and building type. **Positive percentages mean "
        "higher prices; negative percentages mean lower prices.**"
    )
    # Keep the explicitly light chart surfaces and their matching text colors
    # together, including when the surrounding Streamlit theme is dark.
    st.plotly_chart(chart_district_premium(district_data), theme=None)
    st.caption(
        "To keep the chart readable, it shows the 10 highest and 10 lowest comparisons "
        "with Sanxia. The middle 20 districts are left out. Sanxia is also left out "
        "because its difference from itself is zero."
    )
    with st.expander("Why compare with 三峽區 (Sanxia)?"):
        st.markdown(
            "Sanxia is the starting point for comparing districts in this analysis. "
            "The software selected it automatically based on how district names were "
            "sorted—not because it is the cheapest or most typical district.\n\n"
            "The percentages compare each district with Sanxia after accounting for "
            "recorded property differences, such as age, size and building type. "
            "Choosing another starting district would change the percentages shown, "
            "but not the model’s estimated prices."
        )
    st.markdown(
        "大安區 (Da'an) has the largest price difference: **209.3% higher per ping than "
        "Sanxia**, or about **3.09 times** as much in this model. The comparison accounts "
        "for recorded property differences. It does not prove that location alone causes the gap."
    )

    st.divider()
    st.markdown("### How prices compare with and without building management")
    st.markdown(
        "Compare properties with and without a building management group (管理組織), "
        "within each building type. The comparison accounts for other recorded property "
        "differences. It does not show what would happen if a building added management."
    )
    render_building_type_guide()
    st.plotly_chart(chart_management_by_type(mgmt_data), theme=None)
    st.caption(
        "Positive means higher prices with management; negative means lower prices. "
        "Black lines show uncertainty in each comparison. Hover or tap for sale counts "
        "and the share with management."
    )
    st.markdown(
        "Prices with management are about **10.3% lower for 公寓 (walk-ups)** and "
        "**8.8% lower for 透天厝 (townhouses)** in this analysis. 華廈 (mid-rises) show "
        "little difference.\n\n"
        "Treat the 住宅大樓 (high-rise) comparison cautiously. Almost all high-rises "
        "have management. About **70% of those without it are two years old or newer**. "
        "Their newness may affect the comparison, even after accounting for age."
    )
    with st.expander("How to read the management comparison"):
        st.markdown(
            "Each building type is analyzed separately. A single number for all types "
            "would hide their different patterns. The black lines are the original model's "
            "95% confidence intervals: they describe statistical uncertainty under that "
            "model's assumptions, not a range of individual sale prices.\n\n"
            "Only 1.7% of high-rise sales lack management. Their median age is one year, "
            "compared with 15 years for managed high-rises. The model uses a straight-line "
            "age adjustment, which may not fully account for prices of very new homes. "
            "The records do not tell us why a building lacks a management group."
        )

    st.divider()
    st.markdown("### How age and floor level relate to price")
    st.markdown(
        "These lines show price patterns estimated by the model as age or floor level "
        "changes. They are not the sale history of one home."
    )
    st.plotly_chart(chart_age_floor(age_floor_data), theme=None)
    st.markdown(
        "The age pattern falls until about 40 years, then rises. Between ages 40 and 50, "
        "the saved estimates rise **27.6% for 透天厝 (townhouses)** and **8.8% for "
        "住宅大樓 (high-rises)**. This does **not** mean an individual home will gain value "
        "after turning 40. The data cannot establish why this pattern appears."
    )
    st.caption(
        "Floor level is shown as a share of the building's floors: 5th floor out of 10 = "
        "0.50, halfway up; 9th out of 10 = 0.90, near the top."
    )
    with st.expander("How these patterns were drawn"):
        st.markdown(
            "The original notebook changes one input at a time across a sample of "
            "properties, keeping their other recorded inputs fixed. It averages the "
            "model's log-price predictions, then converts the result back to dollars "
            "per ping. This is called partial dependence. These lines are not district "
            "sale-price averages or forecasts for an aging home.\n\n"
            "The lines come from the original LightGBM model. The separate OLS model "
            "uses a straight-line age term and cannot show this turn on its own."
        )

    st.divider()
    st.markdown("### Which details matter most to the model?")
    st.markdown(
        "District contributes most to this model's estimates, followed by building age."
    )
    with st.expander("Technical chart: how property details influence estimates"):
        st.markdown(
            "Each dot shows one detail's contribution for one sampled sale. Right means "
            "it pushes the estimate up; left means down. Rows near the top matter more overall."
        )
        with st.container(horizontal=True, horizontal_alignment="center", key="shap-display"):
            st.image(load_shap_image(), width=760)
        st.caption(
            "The horizontal numbers measure contributions to the model's estimate, not "
            "dollars or percentages. District colors do not rank districts by price. "
            "Use the fullscreen control to enlarge the original image."
        )
        st.markdown(
            "This original image is a SHAP summary of 5,000 sampled sales. "
            "**SHAP value** means a detail's contribution relative to the explanation's "
            "baseline prediction. Zero means no contribution from that detail. "
            "The baseline comes from the model's training records; it is not Sanxia "
            "or the price of an average listing. The scale is log price per ping, "
            "so a value of 0.50 does not mean 50% or 50 dollars.\n\n"
            "**Feature value** means the recorded value of a detail. For numerical "
            "details, pink/red means a higher value, and blue a lower value. "
            "For example, an older building has a higher age value. District and "
            "building type are categories, not cheap-to-expensive scales; their dots "
            "are gray in this image.\n\n"
            "Some original image labels are shortened:\n\n"
            "- **Rooms / Halls:** the recorded 房 / 廳 layout counts. Halls means "
            "living/dining areas, not corridors.\n"
            "- **Floor position:** unit floor divided by total floors.\n"
            "- **Time trend:** months since the first sale date in the data.\n"
            "- **Parking priced at zero:** parking is included in the sale, with a "
            "recorded parking price of zero; this does not establish that it was free.\n"
            "- **Parking bundled with sale:** the sale includes a parking space.\n"
            "- **Has management organization:** the recorded 管理組織 status.\n"
            "- **Whole-building sale:** the sale includes the entire building."
        )


def render_district_comparison(dim: pd.DataFrame):
    st.caption(
        "Compare recorded home sale prices per ping across districts. Each bar shows "
        "the median sale price per ping. The median is the middle price when sales are "
        "ordered from lowest to highest. "
        "These summaries do not account for differences in the properties sold, such "
        "as age, size or building type."
    )
    with st.expander("About these district summaries"):
        st.markdown(
            "The chart and table use the saved, cleaned sale records, including sales "
            "missing details needed by the models. They summarize what sold in each "
            "district; they do not estimate the price of a particular home.\n\n"
            "These district medians are display-only. They are not inputs to either "
            "model. Both models use district identity instead. Keeping the medians "
            "out avoids feeding a summary of the price being predicted back into the "
            "model, a problem called data leakage."
        )
    city_names = {"a": "Taipei City", "f": "New Taipei City"}
    dim = dim.copy()
    dim["city"] = dim["city_code"].map(city_names)
    dim["district_label"] = dim["district"].apply(district_label)

    city_choice = st.radio("City", ["Both", "Taipei City", "New Taipei City"], horizontal=True)
    st.caption("Choose both cities or compare districts within one city.")
    show_all = st.checkbox("Show all districts", value=False)

    view = dim if city_choice == "Both" else dim[dim["city"] == city_choice]
    view = view.sort_values("median_price_per_ping", ascending=False)
    if not show_all:
        view = view.head(15)

    scope = "Taipei City and New Taipei City" if city_choice == "Both" else city_choice
    if show_all:
        st.caption(f"Showing all {len(view)} districts in {scope}, from highest to lowest median sale price per ping.")
    elif len(view) < 15:
        st.caption(
            f"Showing all {len(view)} available districts in {scope} (fewer than 15), "
            "from highest to lowest median sale price per ping."
        )
    else:
        st.caption(f"Showing the 15 highest-priced districts in {scope}, ranked by median sale price per ping.")
    st.plotly_chart(chart_district_comparison(view), theme=None)
    st.dataframe(
        view[["city", "district_label", "median_price_per_ping", "txn_count"]]
        .rename(columns={
            "median_price_per_ping": "Median sale price / ping (NT$)",
            "txn_count": "Recorded sales",
            "district_label": "District", "city": "City",
        })
        .reset_index(drop=True)
        .style.format({"Median sale price / ping (NT$)": "NT${:,.0f}", "Recorded sales": "{:,.0f}"}),
        width="stretch",
    )


def render_estimator(df: pd.DataFrame, models, metadata: dict):
    st.warning(
        "This is a teaching tool, not a property appraisal. It uses recorded sales to "
        "estimate prices for properties with the details you enter. Do not use it to "
        "set a real home's asking price."
    )

    district_categories = metadata["categories"]["district"]
    type_categories = metadata["categories"]["building_type"]
    st.caption(
        "The model gives an estimated price per ping and an estimated range. The range's "
        "real-world reliability has not been checked against new sales. It is not "
        "a promise that a sale price will fall inside it."
    )
    with st.expander("How the estimates and range work"):
        st.markdown(
            "For the property details you enter, the model estimates a median price "
            "per ping and a range of possible prices per ping. The endpoints aim to "
            "describe lower and higher prices for properties with those details. "
            "They are not guaranteed minimum and maximum sale prices. The range's "
            "reliability on new sales has not been verified.\n\n"
            "Sometimes the separate estimates come out in the wrong order. The app "
            "then warns you and shows each estimate unchanged, without a total-price "
            "range. It does not sort or adjust the model's answers.\n\n"
            "Total price is estimated price per ping multiplied by the area you enter. "
            "It is a simple calculation, not a second model or a property appraisal."
        )
        st.markdown("**Technical details**")
        st.markdown(
            "Three separate models estimate the 5th, 50th and 95th conditional percentiles "
            "(quantiles); they do not look up district medians. "
            "The lower-to-upper range is sometimes called a nominal "
            "90% interval. That is a modeling target, not a verified 90% chance "
            "for a future sale. The estimator has no new out-of-sample coverage "
            "evaluation.\n\n"
            "Out-of-order estimates are called quantile crossing. This occurs for "
            "309 of the 98,054 saved modeling inputs."
        )

    # Sections are grouped by INPUT TYPE (location/type dropdowns, size/age
    # numbers, feature checkboxes), not topic -- per revision request. Visual
    # order (top to bottom) is Location & type -> Size & age -> Features, but
    # the underlying widgets have a dependency chain the other way (Features'
    # "whole-building" checkbox needs building_type from Location & type;
    # Size & age's total_floors default needs whole_building from Features).
    # st.container() reserves each section's visual slot up front, then the
    # sections are FILLED in dependency order (location -> features ->
    # size), independent of their reserved top-to-bottom screen position.
    location_section = st.container()
    size_section = st.container()
    features_section = st.container()

    with location_section:
        st.markdown("**Location and building type**")
        lcol1, lcol2 = st.columns(2)
        with lcol1:
            district = st.selectbox("District", district_categories, index=district_categories.index("大安區") if "大安區" in district_categories else 0, format_func=district_label, help="Choose the district where the property is located.")
        with lcol2:
            building_type = st.selectbox(
                "Building type", type_categories, format_func=building_type_label,
                help="Choose the recorded building type. Open the guide below if you are unsure.",
            )
        render_building_type_guide()

    with features_section:
        st.markdown("**What the sale includes**")
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            whole_building_default = building_type == "透天厝"
            whole_building = st.checkbox("Sale includes the whole building", value=whole_building_default, help="Usually selected for 透天厝 (townhouses). Leave unchecked for a single apartment or unit.")
        with fcol2:
            has_management = st.checkbox("Building has a management group (管理組織)", value=True, help="Use the property's recorded 管理組織 status. This refers to a group managing the building's shared affairs.")

    with size_section:
        st.markdown("**Size, age and layout**")
        scol1, scol2 = st.columns(2)
        with scol1:
            building_age = st.number_input("Building age (years)", min_value=0, max_value=100, value=15, help="Age to use for this estimate. The model's date assumption stays at May 2026.")
        with scol2:
            area_ping = st.number_input("Area (ping)", min_value=5.0, max_value=300.0, value=35.0, step=1.0, help="Use the building area recorded for the sale. One ping (坪) is about 3.3 square metres. This area is also used to calculate the estimated total price.")
        floor_col1, floor_col2 = st.columns(2)
        with floor_col1:
            if whole_building:
                total_floors = st.number_input("Total floors", min_value=1, max_value=60, value=3, help="Number of floors in the whole building, not the floor of one unit.")
                st.caption("No separate unit floor is needed for a whole-building sale.")
            else:
                total_floors = st.number_input("Total floors", min_value=1, max_value=60, value=BUILDING_TYPE_DEFAULT_FLOORS[building_type], help="Number of floors in the whole building, not the floor of one unit.")

        if whole_building:
            floor_ratio = 1.0
        else:
            with floor_col2:
                floor_level = st.number_input("Floor of unit", min_value=1, max_value=int(total_floors), value=min(9, int(total_floors)), help="The floor where this unit is located. For example, enter 5 for a fifth-floor apartment.")
            floor_ratio = floor_level / total_floors

        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            rooms = st.number_input("Rooms (房)", min_value=0, max_value=10, value=3, help="Use the recorded 房 room count, separate from the 廳 living/dining areas and bathrooms.")
        with rcol2:
            halls = st.number_input("Living/dining areas (廳)", min_value=0, max_value=5, value=2, help="Use the recorded 廳 count: living or dining areas, not hallways. The source has one combined count, not separate living-room and dining-room fields.")
        with rcol3:
            baths = st.number_input("Bathrooms (衛)", min_value=0, max_value=6, value=2, help="Use the recorded 衛 bathroom count.")

    validation_error, validation_warning = floor_consistency(building_type, total_floors)
    if validation_error:
        st.error(validation_error)
    elif validation_warning:
        st.warning(validation_warning)

    latest = pd.Timestamp(metadata["time"]["origin_date"]) + pd.DateOffset(months=metadata["time"]["latest_month_index"])
    st.caption(
        f"The estimate assumes a sale in {latest:%B %Y} and excludes a parking space."
    )

    estimate_clicked = st.button("Estimate price", type="primary", disabled=bool(validation_error))
    if estimate_clicked and not validation_error:
        latest_month = metadata["time"]["latest_month_index"]
        row = pd.DataFrame([{
            "building_age": building_age, "area_ping": area_ping, "floor_ratio": floor_ratio,
            "total_floors": total_floors, "rooms": rooms, "halls": halls, "baths": baths,
            "months_since_start": latest_month,
            "has_management_int": int(has_management), "whole_building_int": int(whole_building),
            "bundled_parking_int": 0, "parking_price_is_zero_int": 0,
            "district": district, "building_type": building_type,
        }])
        # Must match training's categorical dtype + categories exactly (same
        # order, same set) -- LightGBM raises if train/predict categories differ.
        row["district"] = pd.Categorical(row["district"], categories=district_categories)
        row["building_type"] = pd.Categorical(row["building_type"], categories=type_categories)
        row = row[FEATURE_COLS]

        low, mid, high = map(float, predict_quantiles(models, metadata, row)[0])
        crossed = low > mid or mid > high
        if crossed:
            st.warning(
                "The model's lower, middle and upper estimates are out of order for "
                "these inputs. They cannot form a usable range. Each estimate below "
                "is shown unchanged."
            )
        total_mid = mid * area_ping
        total_low = low * area_ping
        total_high = high * area_ping

        st.metric("Estimated price per ping", fmt_ntd(mid), help="The model's estimated median price per ping for the property details you entered.")
        st.caption("Both ends of the estimated range are prices per ping. The range's reliability on new sales has not been verified.")

        icol1, icol2 = st.columns(2)
        with icol1:
            st.metric("Lower end of estimated range", fmt_ntd(low), help="Price per ping at the lower end of the model's range (5th percentile). Its reliability on new sales has not been verified.")
        with icol2:
            st.metric("Upper end of estimated range", fmt_ntd(high), help="Price per ping at the upper end of the model's range (95th percentile). Its reliability on new sales has not been verified.")

        area_label = f"{area_ping:,.2f}".rstrip("0").rstrip(".")
        st.metric(f"Estimated total for {area_label} ping", fmt_ntd(total_mid), help="Estimated median price per ping multiplied by the area you entered. No parking-space price is added.")
        st.caption(f"Total price = estimated price per ping × {area_label} ping. This is a simple multiplication, not a separate valuation.")
        if crossed:
            st.caption("No total-price range is shown because the estimates are out of order.")
        else:
            # Plain text prevents two NT$ signs being parsed as a Markdown math pair.
            st.text(f"Model-generated total-price range: {fmt_ntd(total_low)} to {fmt_ntd(total_high)}")


def main():
    st.markdown(
        "<h1 style='margin-bottom:0;'>What Shapes Home Prices "
        "in Taipei &amp; New Taipei?</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='font-size:1.1rem; color:{ACCENT}; font-weight:600; margin-top:0;'>"
        "Location stands out: the model estimates prices per ping in 大安區 (Da'an) "
        "at about 3.09 times those in 三峽區 (Sanxia), after accounting for recorded "
        "property differences.</p>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Source: 內政部不動產交易實價查詢服務網 (Taiwan's public home sale records) "
        "· 1 坪 (ping) ≈ 3.3 m² / 35.6 sq ft"
    )

    try:
        df, dim, manifest = load_dashboard_data()
        models, metadata = load_quantile_models(manifest)
        district_data = load_finding_json("district_premium_data.json")
        mgmt_data = load_finding_json("management_by_type_data.json")
        age_floor_data = load_finding_json("age_floor_data.json")
    except (ArtifactError, ValueError, KeyError) as exc:
        st.error("The dashboard cannot open its saved data, models or charts.")
        st.info("Ask the dashboard owner to restore the saved files. More details are below.")
        with st.expander("Details for the dashboard owner"):
            st.text(str(exc))
            st.markdown("Restore the complete saved bundle described in `deploy/COMMUNITY_CLOUD.md`.")
        st.stop()
    n_fct = manifest["tables"]["fct_transactions"]["rows"]
    st.caption(
        "Sales through May 2026. This saved dataset does not update automatically."
    )
    with st.expander("About the sales data"):
        st.markdown(
            f"The saved data contains **{n_fct:,} cleaned residential sales**, dated "
            f"{manifest['coverage']['min_date']} to {manifest['coverage']['max_date']}. "
            f"The models use **{len(df):,} sales** with the required age and floor details. "
            "The other sales remain in the district summaries. There are 104 sales "
            "from before 2023; they have been kept. Prices were cleaned, and extreme "
            "values were capped at the 1st and 99th percentiles. District summaries "
            "also use these cleaned prices."
        )
    with st.expander("Model evaluation"):
        st.markdown("**How the models were checked**")
        st.markdown(
            "- **Pattern-analysis model:** the original LightGBM model used to explore "
            "price patterns scored "
            f"**{metadata['historical_evaluation']['lightgbm_full_sample_r2']:.2f}** "
            "on the same sales it learned from.\n"
            "- **Separate evaluation model:** another LightGBM model learned from older "
            "sales and scored "
            f"**{metadata['historical_evaluation']['lightgbm_holdout_r2']:.2f}** "
            "on newer sales kept aside for checking.\n"
            "- **Price Estimator:** this uses different, separately saved models. "
            "The scores above do not evaluate those models or the reliability of their ranges.\n\n"
            "The scores are R² on log price per ping: a "
            "measure of how well predictions track differences in that transformed "
            "price, compared with using one average value. They are not percentage "
            "accuracy and do not establish future performance."
        )
    with st.expander("What the data leaves out"):
        st.markdown(
            "The records omit details such as renovation, views and precise distance "
            "to transport. District summaries do not account for which kinds of "
            "properties sold there. The models can account only for recorded inputs; "
            "they cannot establish cause and effect or replace a property appraisal."
        )

    top_district = district_data[0]["district"]
    with st.container(key="summary-cards"):
        c1, c2, c3 = st.columns(3)
    c1.metric(
        "Da’an vs Sanxia: price per ping", f"+{district_data[0]['effect_pct']:.0f}%",
        help=(
            f"A model-adjusted historical comparison of {district_label(top_district)} "
            "with 三峽區 (Sanxia), from the original OLS analysis. It accounts for other "
            "recorded property details, including age, size and building type. It is "
            "not a simple comparison of district medians and does not prove cause and effect."
        ),
    )
    c2.metric("Recorded home sales", f"{n_fct:,}", help="Number of cleaned sales in this saved dataset. The models use a smaller set with the required property details; see About the sales data.")
    start_label = pd.Timestamp(manifest["coverage"]["min_date"]).strftime("%Y/%m")
    end_label = pd.Timestamp(manifest["coverage"]["max_date"]).strftime("%Y/%m")
    with c3.container(key="snapshot-coverage"):
        st.metric("Sales dates", f"{start_label} – {end_label}")

    st.write("")
    tab1, tab2, tab3 = st.tabs(["Price Patterns", "District Comparison", "Price Estimator"])
    with tab1:
        render_driver_explorer(district_data, mgmt_data, age_floor_data)
    with tab2:
        render_district_comparison(dim)
    with tab3:
        render_estimator(df, models, metadata)


if __name__ == "__main__":
    main()
