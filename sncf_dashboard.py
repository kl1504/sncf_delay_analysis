# TARDIS — SNCF Delay Prediction Platform (UI v2)
import inspect
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="TARDIS · SNCF Analytics",
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  DESIGN TOKENS
# ══════════════════════════════════════════════════════════════
BG, SURFACE, BORDER = "#0B0F19", "#111827", "#1F2937"
TEXT, MUTED = "#E5E7EB", "#9CA3AF"
PRIMARY, ACCENT = "#6366F1", "#38BDF8"
GOOD, WARN, BAD = "#34D399", "#FBBF24", "#F87171"
GRID = "rgba(148,163,184,.10)"

SEQ_BLUE = ["#1E2447", "#3730A3", "#6366F1", "#A5B4FC"]
SEQ_RED = ["#2A1620", "#7F1D1D", "#EF4444", "#FCA5A5"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

/* Éléments Streamlit par défaut masqués */
#MainMenu, footer, .stDeployButton,
[data-testid="stToolbar"], [data-testid="stDecoration"] {{ visibility: hidden; display: none; }}
header[data-testid="stHeader"] {{ background: transparent; }}

.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"] h5 {{
    font-size: .72rem; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; color: {MUTED}; margin: 18px 0 6px;
}}
[data-testid="stSidebar"] .stButton > button {{ width: 100%; }}

/* Cartes métriques */
[data-testid="stMetric"] {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 1px 2px rgba(0,0,0,.35), 0 10px 28px rgba(0,0,0,.18);
    transition: transform .15s ease, border-color .15s ease;
}}
[data-testid="stMetric"]:hover {{ transform: translateY(-2px); border-color: {PRIMARY}; }}
[data-testid="stMetricLabel"] p {{
    font-size: .74rem; font-weight: 500; letter-spacing: .05em;
    text-transform: uppercase; color: {MUTED};
}}
[data-testid="stMetricValue"] {{ font-size: 1.85rem; font-weight: 600; letter-spacing: -.02em; }}

/* Conteneurs à bordure = cartes de graphiques */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 14px; border-color: {BORDER}; background: {SURFACE};
    box-shadow: 0 1px 2px rgba(0,0,0,.3);
}}
.card-title {{ font-size: .95rem; font-weight: 600; color: {TEXT}; margin: 2px 0 0; }}
.card-sub   {{ font-size: .78rem; color: {MUTED}; margin: 2px 0 8px; }}

/* Bouton principal */
button[kind="primary"], button[data-testid="stBaseButton-primary"] {{
    background: linear-gradient(135deg, {PRIMARY}, #4F46E5); color: #fff; border: 0;
    border-radius: 10px; font-weight: 600; padding: .6rem 1rem;
    box-shadow: 0 4px 14px rgba(99,102,241,.35); transition: all .15s ease;
}}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {{
    transform: translateY(-1px); box-shadow: 0 8px 22px rgba(99,102,241,.5); filter: brightness(1.08);
}}

/* Onglets (style souligné) */
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ padding: 8px 14px; color: {MUTED}; font-weight: 500; }}
.stTabs [aria-selected="true"] {{ color: {TEXT}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {PRIMARY}; height: 2px; }}

/* Expanders */
[data-testid="stExpander"] {{ border: 1px solid {BORDER}; border-radius: 12px; }}

/* Pastille de statut */
.pill {{
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    font-size: .78rem; font-weight: 600; border: 1px solid;
}}
.pill-green  {{ color: {GOOD}; background: rgba(52,211,153,.10); border-color: rgba(52,211,153,.4); }}
.pill-orange {{ color: {WARN}; background: rgba(251,191,36,.10); border-color: rgba(251,191,36,.4); }}
.pill-red    {{ color: {BAD};  background: rgba(248,113,113,.10); border-color: rgba(248,113,113,.4); }}

.empty-state {{ text-align: center; padding: 48px 16px; color: {MUTED}; }}
.empty-state .big {{ font-size: 2.2rem; margin-bottom: 8px; }}

::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)

MONTH_ORDER = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_MAP = {d: i for i, d in enumerate(DAY_ORDER)}
MONTH_MAP = {m: i + 1 for i, m in enumerate(MONTH_ORDER)}

CAUSE_COLS = {
    "External causes": "Pct delay due to external causes",
    "Infrastructure": "Pct delay due to infrastructure",
    "Traffic management": "Pct delay due to traffic management",
    "Rolling stock": "Pct delay due to rolling stock",
    "Station management": "Pct delay due to station management and equipment reuse",
    "Passenger handling": "Pct delay due to passenger handling (crowding, disabled persons, connections)",
}
LABEL_MAP = {v: k for k, v in CAUSE_COLS.items()}

FEATURES = [
    "Departure station", "Arrival station",
    "Number of scheduled trains", "Number of cancelled trains",
    "Number of trains delayed at departure",
    "Average delay of late trains at departure",
    "Number of trains delayed > 15min", "Number of trains delayed > 30min",
    "Number of trains delayed > 60min",
    "Pct delay due to external causes", "Pct delay due to infrastructure",
    "Pct delay due to traffic management", "Pct delay due to rolling stock",
    "month", "year", "day_of_week",
]

DELAY = "Average delay of all trains at arrival"
CANCEL = "Number of cancelled trains"

# ══════════════════════════════════════════════════════════════
#  MISE EN PAGE
# ══════════════════════════════════════════════════════════════
BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=TEXT, size=12),
    margin=dict(l=8, r=8, t=16, b=8),
    hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font_color=TEXT),
    legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
)

_STRETCH = ({"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters
            else {"use_container_width": True})


def style(fig, h=320, **kw):
    """Applique le thème épuré : fond transparent, grille horizontale discrète."""
    fig.update_layout(**{**BASE_LAYOUT, "height": h, **kw})
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=BORDER, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)", tickfont=dict(color=MUTED))
    return fig


def show(fig):
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False}, **_STRETCH)


def card(title, sub=None):
    """Carte (conteneur à bordure) avec titre ; à utiliser avec `with`."""
    c = st.container(border=True)
    with c:
        html = f"<div class='card-title'>{title}</div>"
        if sub:
            html += f"<div class='card-sub'>{sub}</div>"
        st.markdown(html, unsafe_allow_html=True)
    return c


def status_of(pred):
    """Seuils de statut (logique métier inchangée)."""
    if pred < 5:
        return "On time", "pill-green", "✅"
    if pred < 15:
        return "Slight delay", "pill-orange", "⚠️"
    return "Major delay", "pill-red", "🚨"


# ══════════════════════════════════════════════════════════════
#  CHARGEMENT DES DONNÉES ET DU MODÈLE
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading data...")
def load_data():
    d = pd.read_csv("cleaned_dataset.csv")
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["Months"] = pd.Categorical(d["Months"], categories=MONTH_ORDER, ordered=True)
    d["Days"] = pd.Categorical(d["Days"], categories=DAY_ORDER, ordered=True)
    return d


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    try:
        bundle = joblib.load("model.joblib")
        return bundle["model"], bundle["station_encoder"]
    except Exception:
        return None, None


df = load_data()
model, station_encoder = load_model()


def slider(label, col, help_, is_float=False):
    """Slider borné par les valeurs min/max du dataset, valeur initiale = médiane."""
    cast = float if is_float else int
    return st.slider(label, cast(df[col].min()), cast(df[col].max()), cast(df[col].median()), help=help_)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:10px;padding:4px 0 8px'>
      <div style='font-size:1.7rem'>🚄</div>
      <div>
        <div style='font-size:1.15rem;font-weight:700;color:{TEXT};letter-spacing:-.01em'>TARDIS</div>
        <div style='font-size:.68rem;color:{MUTED};letter-spacing:.1em'>SNCF DELAY INTELLIGENCE</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("##### Route & date")
    station_dep = st.selectbox("Departure station", sorted(df["Departure station"].unique()),
                               help="Station of departure of the train.")
    station_arr = st.selectbox("Arrival station", sorted(df["Arrival station"].unique()),
                               help="Station of arrival of the train.")
    c_d, c_m = st.columns(2)
    pred_day = c_d.selectbox("Day", DAY_ORDER, help="Day of the week of travel.")
    pred_month = c_m.selectbox("Month", MONTH_ORDER, help="Month of travel.")
    pred_year = st.selectbox("Year", sorted(df["Years"].unique(), reverse=True), help="Year of travel.")

    st.markdown("##### Operational parameters")
    with st.expander("Traffic", expanded=False):
        n_scheduled = slider("Scheduled trains", "Number of scheduled trains",
                             "Number of trains planned on this route for the period.")
        n_cancelled = slider("Cancelled trains", "Number of cancelled trains",
                             "Number of trains cancelled on this route.")
    with st.expander("Delays", expanded=False):
        n_delayed_dep = slider("Delayed at departure", "Number of trains delayed at departure",
                               "Number of trains that left late.")
        avg_delay_dep = slider("Avg delay at departure (min)", "Average delay of late trains at departure",
                               "Average delay, in minutes, of the trains that left late.", is_float=True)
        n_15 = slider("Delayed > 15 min", "Number of trains delayed > 15min", "Trains with more than 15 min of delay.")
        n_30 = slider("Delayed > 30 min", "Number of trains delayed > 30min", "Trains with more than 30 min of delay.")
        n_60 = slider("Delayed > 60 min", "Number of trains delayed > 60min", "Trains with more than 60 min of delay.")
    with st.expander("Delay causes (%)", expanded=False):
        def pct_input(label, col, help_):
            return st.number_input(label, 0.0, 100.0, float(df[col].mean()), step=0.5, help=help_)
        pct_ext = pct_input("External", "Pct delay due to external causes", "Weather, incidents, third parties…")
        pct_inf = pct_input("Infrastructure", "Pct delay due to infrastructure", "Tracks, signalling, works…")
        pct_traf = pct_input("Traffic mgmt", "Pct delay due to traffic management", "Network regulation and scheduling.")
        pct_roll = pct_input("Rolling stock", "Pct delay due to rolling stock", "Train breakdowns and maintenance.")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    run_pred = st.button("Run prediction", type="primary", disabled=model is None,
                         help="Estimates the average arrival delay with the trained model.")
    if model is None:
        st.warning("model.joblib not found. Train and save the model first.")

    st.markdown("##### Analysis filters")
    with st.expander("Filter the charts", expanded=False):
        sel_service = st.multiselect("Service type", df["Service"].unique().tolist(),
                                     default=df["Service"].unique().tolist(),
                                     help="Applies to the analysis tabs only.")
        sel_years = st.select_slider("Year range", options=sorted(df["Years"].unique()),
                                     value=(int(df["Years"].min()), int(df["Years"].max())))
        sel_months = st.multiselect("Months", MONTH_ORDER, default=MONTH_ORDER)

# ══════════════════════════════════════════════════════════════
#  PRÉDICTION
# ══════════════════════════════════════════════════════════════
if run_pred:
    with st.spinner("Computing prediction…"):
        try:
            dep_enc = station_encoder.transform([station_dep.upper()])[0]
            arr_enc = station_encoder.transform([station_arr.upper()])[0]
        except ValueError as e:
            st.session_state["pred_error"] = str(e)
            st.session_state.pop("pred", None)
        else:
            input_df = pd.DataFrame([{
                "Departure station": dep_enc,
                "Arrival station": arr_enc,
                "Number of scheduled trains": n_scheduled,
                "Number of cancelled trains": n_cancelled,
                "Number of trains delayed at departure": n_delayed_dep,
                "Average delay of late trains at departure": avg_delay_dep,
                "Number of trains delayed > 15min": n_15,
                "Number of trains delayed > 30min": n_30,
                "Number of trains delayed > 60min": n_60,
                "Pct delay due to external causes": pct_ext,
                "Pct delay due to infrastructure": pct_inf,
                "Pct delay due to traffic management": pct_traf,
                "Pct delay due to rolling stock": pct_roll,
                "month": MONTH_MAP[pred_month],
                "year": pred_year,
                "day_of_week": DAY_MAP[pred_day],
            }])[FEATURES]
            st.session_state["pred"] = {
                "value": float(model.predict(input_df)[0]),
                "dep": station_dep, "arr": station_arr,
                "when": f"{pred_day}, {pred_month} {pred_year}",
            }
            st.session_state.pop("pred_error", None)

pred_state = st.session_state.get("pred")

# ══════════════════════════════════════════════════════════════
#  EN-TÊTE
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='margin-bottom:20px'>
  <h1 style='margin:0;font-size:1.7rem;font-weight:700;letter-spacing:-.02em;color:{TEXT}'>Delay Prediction</h1>
  <p style='color:{MUTED};margin:4px 0 0;font-size:.92rem'>
    Estimate arrival delays on the French high-speed rail network and explore historical performance.
  </p>
</div>""", unsafe_allow_html=True)

if "pred_error" in st.session_state:
    st.error(f"Encoding error: {st.session_state['pred_error']}")

# ══════════════════════════════════════════════════════════════
#  KPIs DE PRÉDICTION
# ══════════════════════════════════════════════════════════════
net_avg = df[DELAY].mean()
k1, k2, k3, k4 = st.columns(4)

if pred_state:
    pv = pred_state["value"]
    route_hist = df[(df["Departure station"] == pred_state["dep"]) &
                    (df["Arrival station"] == pred_state["arr"])]
    route_avg = route_hist[DELAY].mean() if len(route_hist) else np.nan
    hist = df[DELAY].dropna()
    percentile = (hist < pv).mean() * 100
    label, _, _ = status_of(pv)

    k1.metric("Estimated arrival delay", f"{pv:.1f} min", f"{pv - net_avg:+.1f} min vs network avg",
              delta_color="inverse", help="Model estimate of the average delay at arrival.")
    k2.metric("Predicted status", label, help="On time < 5 min · Slight delay 5–15 min · Major delay ≥ 15 min.")
    if np.isnan(route_avg):
        k3.metric("Route historical avg", "N/A", help="No historical data for this route.")
    else:
        k3.metric("Route historical avg", f"{route_avg:.1f} min", f"{pv - route_avg:+.1f} min vs prediction",
                  delta_color="inverse", help="Average arrival delay observed on this exact route.")
    k4.metric("Delay percentile", f"{percentile:.0f}th",
              help="Share of historical records with a smaller delay than the prediction.")
else:
    k1.metric("Estimated arrival delay", "—", help="Run a prediction from the sidebar.")
    k2.metric("Predicted status", "—")
    k3.metric("Route historical avg", "—")
    k4.metric("Delay percentile", "—")

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  DONNÉES FILTRÉES POUR L'ANALYSE
# ══════════════════════════════════════════════════════════════
mask = (df["Service"].isin(sel_service)
        & df["Years"].between(sel_years[0], sel_years[1])
        & df["Months"].isin(sel_months))
dff = df[mask].copy()


# ══════════════════════════════════════════════════════════════
#  ONGLET 1 — PRÉDICTION
# ══════════════════════════════════════════════════════════════
def tab_prediction():
    if model is None:
        st.warning("model.joblib not found. Run `tardis_model.ipynb` first to train and save the model.")
        return
    if not pred_state:
        st.markdown("""<div class='empty-state'><div class='big'>🎯</div>
            <b>No prediction yet</b><br>Set the route and parameters in the sidebar,
            then click <b>Run prediction</b>.</div>""", unsafe_allow_html=True)
        return

    pv = pred_state["value"]
    label, pill, icon = status_of(pv)
    st.markdown(
        f"<span class='pill {pill}'>{icon} {label}</span> &nbsp; "
        f"<span style='color:{MUTED};font-size:.9rem'>{pred_state['dep']} → {pred_state['arr']} · "
        f"{pred_state['when']}</span>", unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1.4])
    axis_max = max(30, int(np.ceil(pv * 1.2)))

    with left, card("Estimated delay", "Compared with the network average"):
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=pv,
            number=dict(suffix=" min", font=dict(color=TEXT, size=34)),
            delta=dict(reference=net_avg, valueformat=".1f",
                       increasing=dict(color=BAD), decreasing=dict(color=GOOD)),
            gauge=dict(
                axis=dict(range=[0, axis_max], tickcolor=MUTED),
                bar=dict(color=PRIMARY, thickness=0.28), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                steps=[dict(range=[0, 5], color="rgba(52,211,153,.18)"),
                       dict(range=[5, 15], color="rgba(251,191,36,.18)"),
                       dict(range=[15, axis_max], color="rgba(248,113,113,.18)")],
                threshold=dict(line=dict(color=TEXT, width=2), thickness=0.8, value=net_avg),
            )))
        show(style(fig, 300, margin=dict(l=20, r=20, t=30, b=10)))

    with right, card("Historical delay on this route", "Monthly average vs. your prediction"):
        rh = df[(df["Departure station"] == pred_state["dep"]) & (df["Arrival station"] == pred_state["arr"])]
        if len(rh) == 0:
            st.info("No historical data available for this specific route combination.")
        else:
            rm = rh.groupby("Months", observed=True)[DELAY].mean().reindex(MONTH_ORDER).reset_index()
            fig = px.bar(rm, x="Months", y=DELAY, color=DELAY, color_continuous_scale=SEQ_BLUE)
            fig.add_hline(y=pv, line_color=ACCENT, line_dash="dash",
                          annotation_text=f"Prediction: {pv:.1f} min", annotation_font_color=ACCENT)
            fig.update_traces(hovertemplate="%{x}: %{y:.1f} min<extra></extra>")
            show(style(fig, 300, coloraxis_showscale=False, yaxis_title="Avg delay (min)", xaxis_title=""))


# ══════════════════════════════════════════════════════════════
#  ONGLET 2 — VUE D'ENSEMBLE
# ══════════════════════════════════════════════════════════════
def tab_overview():
    total = int(dff["Number of scheduled trains"].sum())
    canc = int(dff[CANCEL].sum())
    cancel_rate = canc / total * 100 if total else 0
    on_time = (dff["Statut"] == "On Time").mean() * 100
    major = (dff["Statut"] == "Mayor delay").mean() * 100

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Scheduled trains", f"{total:,}")
    m2.metric("Avg arrival delay", f"{dff[DELAY].mean():.1f} min",
              f"{dff[DELAY].mean() - net_avg:+.1f} min vs all data", delta_color="inverse")
    m3.metric("Cancelled", f"{canc:,}", f"{cancel_rate:.1f}% rate", delta_color="off")
    m4.metric("On time", f"{on_time:.1f}%")
    m5.metric("Major delays", f"{major:.1f}%", "> 15 min", delta_color="off")
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1, card("Delay status distribution"):
        sc = dff["Statut"].value_counts().reset_index()
        sc.columns = ["Status", "Count"]
        fig = px.pie(sc, values="Count", names="Status", hole=0.62, color="Status",
                     color_discrete_map={"On Time": GOOD, "Slight delay": WARN, "Mayor delay": BAD})
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          marker=dict(line=dict(color=SURFACE, width=3)))
        show(style(fig, 300, showlegend=False))
    with c2, card("Average delay by service"):
        svc = dff.groupby("Service")[DELAY].mean().reset_index()
        fig = px.bar(svc, x="Service", y=DELAY, text=DELAY)
        fig.update_traces(marker_color=PRIMARY, texttemplate="%{text:.1f} min", textposition="outside")
        show(style(fig, 300, yaxis_title="Minutes", xaxis_title=""))

    with card("Yearly delay trend", "Average delay (area) and total cancellations (bars)"):
        yr = dff.groupby("Years").agg(avg_delay=(DELAY, "mean"), canc=(CANCEL, "sum")).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=yr["Years"], y=yr["canc"], name="Cancellations",
                             marker_color="rgba(248,113,113,.35)"), secondary_y=True)
        fig.add_trace(go.Scatter(x=yr["Years"], y=yr["avg_delay"], name="Avg delay (min)", mode="lines+markers",
                                 line=dict(color=ACCENT, width=3), fill="tozeroy",
                                 fillcolor="rgba(56,189,248,.12)", marker=dict(size=8)), secondary_y=False)
        fig.update_yaxes(title_text="Avg delay (min)", secondary_y=False)
        fig.update_yaxes(title_text="Cancellations", secondary_y=True, showgrid=False)
        show(style(fig, 330))

    with card("Arrival delay distribution"):
        fig = px.histogram(dff, x=DELAY, nbins=60, marginal="box", color_discrete_sequence=[PRIMARY])
        show(style(fig, 300, xaxis_title="Avg delay at arrival (min)", yaxis_title="Frequency"))


# ══════════════════════════════════════════════════════════════
#  ONGLET 3 — STATIONS
# ══════════════════════════════════════════════════════════════
def top_stations(col, scale, title):
    stats = dff.groupby(col)[DELAY].mean().nlargest(15).sort_values().reset_index()
    stats.columns = ["Station", "Avg Delay"]
    fig = px.bar(stats, x="Avg Delay", y="Station", orientation="h", text="Avg Delay",
                 color="Avg Delay", color_continuous_scale=scale)
    fig.update_traces(texttemplate="%{text:.1f} min", textposition="outside")
    with card(title):
        show(style(fig, 450, coloraxis_showscale=False, xaxis_title="Minutes", yaxis_title=""))


def tab_stations():
    c1, c2 = st.columns(2)
    with c1:
        top_stations("Departure station", SEQ_RED, "Top 15 departure stations · avg delay")
    with c2:
        top_stations("Arrival station", SEQ_BLUE, "Top 15 arrival stations · avg delay")

    with card("Most delayed routes (top 20)", "Routes with at least 5 records"):
        r = (dff.groupby(["Departure station", "Arrival station"])
                .agg(avg_delay=(DELAY, "mean"), count=("Date", "count")).reset_index())
        r = r[r["count"] >= 5].nlargest(20, "avg_delay")
        r["Route"] = r["Departure station"] + " → " + r["Arrival station"]
        fig = px.bar(r.sort_values("avg_delay"), x="avg_delay", y="Route", orientation="h",
                     color="avg_delay", color_continuous_scale=SEQ_RED, hover_data={"count": True, "avg_delay": ":.2f"})
        show(style(fig, 500, coloraxis_showscale=False, xaxis_title="Avg delay (min)", yaxis_title=""))

    with card("Top stations by cancellations"):
        can = dff.groupby("Departure station")[CANCEL].sum().nlargest(12).reset_index()
        can.columns = ["Station", "Cancellations"]
        fig = px.treemap(can, path=["Station"], values="Cancellations", color="Cancellations",
                         color_continuous_scale=SEQ_RED)
        show(style(fig, 380, coloraxis_showscale=False))


# ══════════════════════════════════════════════════════════════
#  ONGLET 4 — ANALYSE TEMPORELLE
# ══════════════════════════════════════════════════════════════
def tab_time():
    c1, c2 = st.columns(2)
    with c1, card("Average delay by day of week"):
        dd = dff.groupby("Days", observed=True)[DELAY].mean().reindex(DAY_ORDER).reset_index()
        dd.columns = ["Day", "Avg Delay"]
        colors = [BAD if v == dd["Avg Delay"].max() else PRIMARY for v in dd["Avg Delay"]]
        fig = go.Figure(go.Bar(x=dd["Day"], y=dd["Avg Delay"], marker_color=colors,
                               text=dd["Avg Delay"], texttemplate="%{text:.1f}", textposition="outside"))
        show(style(fig, 320, yaxis_title="Minutes"))
    with c2, card("Average delay by month"):
        md_ = dff.groupby("Months", observed=True)[DELAY].mean().reindex(MONTH_ORDER).reset_index()
        md_.columns = ["Month", "Avg Delay"]
        fig = px.line(md_, x="Month", y="Avg Delay", markers=True, line_shape="spline")
        fig.update_traces(line=dict(color=ACCENT, width=3), marker=dict(size=8),
                          fill="tozeroy", fillcolor="rgba(56,189,248,.10)")
        show(style(fig, 320, yaxis_title="Minutes", xaxis_title=""))

    with card("Monthly cancellations over time"):
        m = dff.groupby(["Years", "Months"], observed=True)[CANCEL].sum().reset_index()
        m["Period"] = pd.to_datetime(dict(year=m["Years"].astype(int),
                                          month=m["Months"].astype(str).map(MONTH_MAP), day=1))
        fig = px.area(m.sort_values("Period"), x="Period", y=CANCEL)
        fig.update_traces(line=dict(color=BAD, width=2), fillcolor="rgba(248,113,113,.15)")
        show(style(fig, 320, xaxis_title="", yaxis_title="Cancellations"))

    with card("Delay heatmap · day × month"):
        heat = (dff.groupby(["Days", "Months"], observed=True)[DELAY].mean()
                   .unstack("Months").reindex(index=DAY_ORDER, columns=MONTH_ORDER))
        fig = px.imshow(heat, color_continuous_scale=SEQ_BLUE, aspect="auto", text_auto=".1f",
                        labels=dict(color="Avg delay (min)"))
        show(style(fig, 350, coloraxis_colorbar=dict(title="min", thickness=10)))


# ══════════════════════════════════════════════════════════════
#  ONGLET 5 — CAUSES DES RETARDS
# ══════════════════════════════════════════════════════════════
def tab_causes():
    causes = {k: dff[v].mean() for k, v in CAUSE_COLS.items()}
    cdf = pd.DataFrame(list(causes.items()), columns=["Cause", "Pct"])

    c1, c2 = st.columns(2)
    with c1, card("Average cause breakdown"):
        fig = px.bar(cdf.sort_values("Pct"), x="Pct", y="Cause", orientation="h", text="Pct",
                     color="Pct", color_continuous_scale=SEQ_BLUE)
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        show(style(fig, 380, coloraxis_showscale=False, xaxis_title="Average %", yaxis_title=""))
    with c2, card("Cause profile · radar"):
        cats, vals = list(causes.keys()), list(causes.values())
        fig = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself",
                                        fillcolor="rgba(99,102,241,.22)", line=dict(color=PRIMARY, width=2.5)))
        fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                     radialaxis=dict(gridcolor=GRID, tickfont=dict(color=MUTED)),
                                     angularaxis=dict(gridcolor=GRID)))
        show(style(fig, 380, margin=dict(l=40, r=40, t=24, b=24)))

    with card("Delay causes evolution by year"):
        cy = dff.groupby("Years")[list(CAUSE_COLS.values())].mean().reset_index()
        cy = cy.melt(id_vars="Years", var_name="Cause", value_name="Pct")
        cy["Cause"] = cy["Cause"].map(LABEL_MAP)
        fig = px.line(cy, x="Years", y="Pct", color="Cause", markers=True, line_shape="spline",
                      color_discrete_sequence=[PRIMARY, ACCENT, GOOD, WARN, BAD, "#C084FC"])
        fig.update_traces(line_width=2.5, marker_size=7)
        show(style(fig, 380, yaxis_title="Average %", xaxis_title="Year", legend=dict(orientation="h", y=-0.2)))

    with card("Correlation · causes vs arrival delay"):
        corr = dff[list(CAUSE_COLS.values()) + [DELAY]].corr()[DELAY].drop(DELAY).reset_index()
        corr.columns = ["Feature", "Correlation"]
        corr["Feature"] = corr["Feature"].map(LABEL_MAP)
        corr = corr.sort_values("Correlation")
        fig = go.Figure(go.Bar(x=corr["Correlation"], y=corr["Feature"], orientation="h",
                               marker_color=[BAD if v > 0 else GOOD for v in corr["Correlation"]],
                               text=corr["Correlation"].round(3), textposition="outside"))
        show(style(fig, 320, xaxis_title="Pearson r", yaxis_title="", xaxis=dict(range=[-0.5, 0.5])))


# ══════════════════════════════════════════════════════════════
#  ONGLET 6 — MODÈLE & DONNÉES
# ══════════════════════════════════════════════════════════════
def tab_model():
    with card("Feature importance", "What the model relies on most to estimate a delay"):
        imp = getattr(model, "feature_importances_", None)
        if imp is None or len(imp) != len(FEATURES):
            st.info("Feature importance is not available for this model.")
        else:
            fi = pd.DataFrame({"Feature": FEATURES, "Importance": imp}).sort_values("Importance")
            fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale=SEQ_BLUE)
            show(style(fig, 460, coloraxis_showscale=False, xaxis_title="Importance", yaxis_title=""))

    with st.expander("How does the model work?"):
        st.markdown("""
- **Model**: Random Forest regressor trained on historical SNCF high-speed data.
- **Inputs**: route (encoded stations), operational counts, delay causes, and calendar features.
- **Output**: estimated average delay at arrival, in minutes.
- **Status thresholds**: on time < 5 min · slight delay 5–15 min · major delay ≥ 15 min.
- **Limits**: the estimate reflects historical patterns, not real-time conditions.
""")
    with st.expander("Raw dataset preview (first 50 rows)"):
        st.dataframe(dff.head(50).style.format(precision=2), height=350)


# ══════════════════════════════════════════════════════════════
#  ONGLETS
# ══════════════════════════════════════════════════════════════
tabs = st.tabs(["🎯 Prediction", "📊 Overview", "🗺️ Stations", "📅 Time", "⚙️ Causes", "🧠 Model & data"])

with tabs[0]:
    tab_prediction()

analysis = [(tabs[1], tab_overview), (tabs[2], tab_stations), (tabs[3], tab_time), (tabs[4], tab_causes)]
for tab, render in analysis:
    with tab:
        if dff.empty:
            st.info("No data matches the current filters. Adjust them in the sidebar (Analysis filters).")
        else:
            render()

with tabs[5]:
    tab_model()
