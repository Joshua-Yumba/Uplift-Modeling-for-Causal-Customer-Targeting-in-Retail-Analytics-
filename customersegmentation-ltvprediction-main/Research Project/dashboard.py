# dashboard.py
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard_utils import (
    FIGURES_DIR,
    RESULTS_DIR,
    compute_summary_metrics,
    format_currency,
    load_csv,
    load_pipeline_history,
)


# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Customer Intelligence Command Center",
    page_icon="Chart Increasing",
    layout="wide",
)

# -------------------------------------------------
#  THEME TOGGLE IN SIDEBAR
# -------------------------------------------------
st.sidebar.header("Command Center Controls")

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if st.sidebar.button("Switch to Light Theme" if st.session_state.theme == "dark" else "Switch to Dark Theme"):
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.rerun()

if st.session_state.theme == "light":
    st._config.set_option("theme.base", "light")
else:
    st._config.set_option("theme.base", "dark")

is_dark = st.session_state.theme == "dark"

BG_GRADIENT = (
    "linear-gradient(90deg, #1d2671, #c33764)" if is_dark
    else "linear-gradient(90deg, #3a86ff, #8338ec)"
)
TEXT_COLOR = "white" if is_dark else "#1a1a1a"
PLOT_TEMPLATE = "plotly_dark" if is_dark else "simple_white"
PRIMARY_COLOR = "#c33764"
CLUSTER_COLORS = px.colors.qualitative.Prism if is_dark else px.colors.qualitative.Set2


# ====================== DATA LOADING ======================
@st.cache_data(show_spinner=False)
def load_clv_data() -> pd.DataFrame:
    return load_csv("clv_predictions.csv")

@st.cache_data(show_spinner=False)
def load_segment_data() -> pd.DataFrame:
    return load_csv("segment_analysis.csv")

@st.cache_data(show_spinner=False)
def load_top_customers() -> pd.DataFrame:
    return load_csv("top_customers.csv")

@st.cache_data(show_spinner=False)
def load_churn_risk() -> pd.DataFrame:
    return load_csv("top_churn_risk.csv")

# --- NEW: NBO, UPLIFT, FORECAST ---
@st.cache_data(show_spinner=False)
def load_nbo_data() -> pd.DataFrame:
    path = RESULTS_DIR / "nbo_recommendations.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_uplift_data() -> pd.DataFrame:
    path = RESULTS_DIR / "uplift_results.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_forecast_data() -> pd.DataFrame:
    path = RESULTS_DIR / "forecast_results.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


clv_df = load_clv_data()
if clv_df.empty:
    st.warning("No analytics generated yet. Run the pipeline to populate the dashboard.")
    st.stop()


# ====================== RUN PIPELINE BUTTON ======================
if st.sidebar.button("Run Analytics Pipeline"):
    import subprocess, sys
    st.sidebar.info("Executing main.py… check terminal for progress.")
    subprocess.run([sys.executable, "main.py"], check=False)
    st.cache_data.clear()
    st.rerun()


# ====================== CLUSTER FILTER ======================
if "cluster" in clv_df.columns:
    cluster_options = ["All"] + sorted(clv_df["cluster"].astype(int).unique().tolist())
    cluster_filter = st.sidebar.selectbox("Focus on Cluster", cluster_options)
    if cluster_filter != "All":
        clv_df = clv_df[clv_df["cluster"].astype(int) == cluster_filter]


# ====================== HEADER ======================
st.markdown(
    f"""
    <div style="
        background: {BG_GRADIENT};
        padding: 32px;
        border-radius: 18px;
        color: {TEXT_COLOR};
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin-bottom: 2rem;
    ">
        <h1 style="margin:0; font-size:2.4rem;">Customer Intelligence Command Center</h1>
        <p style="margin:0.4rem 0 0; font-size:1.1rem; opacity:0.9;">
            Track city-level performance, segment behaviors, and lifetime value momentum in real time.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ====================== METRICS ======================
metrics = compute_summary_metrics(clv_df)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Cities Analysed", f"{metrics['total_cities']:,}")
col2.metric("Average CLV", format_currency(metrics["avg_clv"]))
col3.metric("Total Revenue", format_currency(metrics["total_revenue"]))
if metrics["avg_churn_probability"] is not None:
    col4.metric("Avg Churn Probability", f"{metrics['avg_churn_probability']*100:,.1f}%")
else:
    col4.metric("Avg Churn Probability", "N/A")

st.markdown("---")


# ====================== TABS (8 Total) ======================
tab_overview, tab_segments, tab_clv, tab_churn, tab_nbo, tab_uplift, tab_forecast, tab_diagnostics = st.tabs(
    ["Overview", "Segments", "CLV Insights", "Churn Risk", "Next-Best-Offer", "Uplift", "Forecast", "Diagnostics"]
)


# ====================== TAB: OVERVIEW ======================
with tab_overview:
    st.subheader("Momentum Snapshot")
    history = load_pipeline_history()
    if history:
        history_df = pd.DataFrame(history)
        history_df["run_timestamp"] = pd.to_datetime(history_df["run_timestamp"])
        timeline_fig = px.line(
            history_df,
            x="run_timestamp",
            y="avg_clv",
            markers=True,
            color_discrete_sequence=[PRIMARY_COLOR],
            title="Average CLV Over Runs",
            template=PLOT_TEMPLATE,
        )
        timeline_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=TEXT_COLOR if is_dark else "#2a2a2a"
        )
        st.plotly_chart(timeline_fig, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Latest Run", history_df["run_timestamp"].max().strftime("%d %b %Y %H:%M"))
        if "churn_auc" in history_df.columns:
            auc = history_df["churn_auc"].dropna()
            if not auc.empty:
                m2.metric("Latest Churn AUC", f"{auc.iloc[-1]:.3f}")
        if "ml_clv_r2" in history_df.columns:
            r2 = history_df["ml_clv_r2"].dropna()
            if not r2.empty:
                m3.metric("Latest ML CLV R²", f"{r2.iloc[-1]:.3f}")
    else:
        st.info("Run the analytics at least once to build a timeline.")

    st.markdown("### Highlights")
    c1, c2, c3 = st.columns(3)
    if "cluster" in clv_df.columns:
        top_cluster = clv_df.groupby("cluster")["CLV"].mean().sort_values(ascending=False).head(1)
        if not top_cluster.empty:
            c1.success(f"Cluster {int(top_cluster.index[0])} leads with avg CLV {format_currency(top_cluster.iloc[0])}.")
    top_city = clv_df.sort_values("CLV", ascending=False).head(1)
    if not top_city.empty:
        city = top_city.iloc[0]
        c2.info(f"{city['customer_id']} is top city with CLV {format_currency(city['CLV'])}.")
    if "churn_probability" in clv_df.columns:
        high_risk = (clv_df["churn_probability"] > 0.6).mean()
        c3.warning(f"{high_risk*100:,.1f}% of cities flagged as high churn risk (>60%).")


# ====================== TAB: SEGMENTS ======================
with tab_segments:
    st.subheader("Segment Explorer")
    seg_df = load_segment_data()
    if not seg_df.empty:
        st.dataframe(seg_df, use_container_width=True, height=280)
    else:
        st.info("Segment analysis not available.")

    if {"recency", "monetary_value", "CLV"}.issubset(clv_df.columns):
        src = clv_df.copy()
        src["CLV_size"] = src["CLV"].fillna(0).astype(float)
        scatter_fig = px.scatter(
            src,
            x="recency",
            y="monetary_value",
            size="CLV_size",
            color=src["cluster"].astype(str) if "cluster" in src.columns else None,
            hover_data={"customer_id": True, "frequency": True, "prob_alive": True},
            title="Segment Landscape",
            template=PLOT_TEMPLATE,
            color_discrete_sequence=CLUSTER_COLORS,
        )
        scatter_fig.update_layout(
            legend_title_text="Cluster",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=TEXT_COLOR if is_dark else "#2a2a2a"
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

    if "cluster" in clv_df.columns:
        pie_fig = px.pie(
            clv_df,
            names=clv_df["cluster"].astype(str),
            title="Cluster Mix",
            hole=0.45,
            template=PLOT_TEMPLATE,
            color_discrete_sequence=CLUSTER_COLORS,
        )
        pie_fig.update_layout(font_color=TEXT_COLOR if is_dark else "#2a2a2a")
        st.plotly_chart(pie_fig, use_container_width=True)


# ====================== TAB: CLV INSIGHTS ======================
with tab_clv:
    st.subheader("Value Trajectory")
    if "CLV" in clv_df.columns:
        hist_fig = px.histogram(
            clv_df,
            x="CLV",
            nbins=40,
            color=clv_df["cluster"].astype(str) if "cluster" in clv_df.columns else None,
            marginal="rug",
            opacity=0.75,
            title="CLV Distribution",
            template=PLOT_TEMPLATE,
            color_discrete_sequence=CLUSTER_COLORS,
        )
        hist_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=TEXT_COLOR if is_dark else "#2a2a2a"
        )
        st.plotly_chart(hist_fig, use_container_width=True)

    if {"CLV", "CLV_ML"}.issubset(clv_df.columns):
        box_fig = go.Figure()
        box_fig.add_trace(go.Box(y=clv_df["CLV"], name="Probabilistic CLV", marker_color="#1f77b4"))
        box_fig.add_trace(go.Box(y=clv_df["CLV_ML"], name="ML CLV", marker_color="#ff7f0e"))
        box_fig.update_layout(
            title="CLV Model Comparison",
            template=PLOT_TEMPLATE,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=TEXT_COLOR if is_dark else "#2a2a2a"
        )
        st.plotly_chart(box_fig, use_container_width=True)
        diff = (clv_df["CLV_ML"] - clv_df["CLV"]).mean()
        st.caption(f"Average ML lift vs probabilistic: {diff:,.1f}")


# ====================== TAB: CHURN RISK ======================
with tab_churn:
    st.subheader("Churn Radar")
    churn_df = load_churn_risk()
    if not churn_df.empty:
        st.dataframe(churn_df, use_container_width=True, height=280)

    if "churn_probability" in clv_df.columns:
        churn_hist = px.histogram(
            clv_df,
            x="churn_probability",
            nbins=30,
            color_discrete_sequence=[PRIMARY_COLOR],
            title="Churn Probability Spread",
            template=PLOT_TEMPLATE,
        )
        churn_hist.update_xaxes(title="Churn Probability", tickformat=".0%")
        churn_hist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=TEXT_COLOR if is_dark else "#2a2a2a"
        )
        st.plotly_chart(churn_hist, use_container_width=True)

        threshold = st.slider("Alert threshold", 0.0, 1.0, 0.6, 0.05)
        flagged = clv_df[clv_df["churn_probability"] >= threshold]
        st.write(f"Cities above threshold ({len(flagged)}):")
        st.dataframe(flagged[["customer_id", "cluster", "CLV", "churn_probability"]], use_container_width=True)
    else:
        st.info("Churn model disabled in the last run.")


# ====================== TAB: NEXT-BEST-OFFER ======================
with tab_nbo:
    st.subheader("Next-Best-Offer Engine")
    nbo_df = load_nbo_data()
    if not nbo_df.empty:
        nbo_df = nbo_df.copy()
        nbo_df["CLV"] = pd.to_numeric(nbo_df["CLV"], errors='coerce').fillna(0)
        nbo_df["offer_score"] = pd.to_numeric(nbo_df["offer_score"], errors='coerce').fillna(0)
        nbo_df = nbo_df[nbo_df["offer_score"] > 0]

        if nbo_df.empty:
            st.warning("No valid offer data.")
        else:
            st.dataframe(nbo_df.head(20), use_container_width=True)
            top10 = nbo_df.head(10)
            fig = px.bar(
                top10,
                x="customer_id",
                y="offer_score",
                color="recommended_product",
                title="Top 10 NBO Scores",
                template=PLOT_TEMPLATE,
                color_discrete_sequence=CLUSTER_COLORS
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Enable NBO in config.yaml and rerun pipeline.")


# ====================== TAB: UPLIFT ======================
with tab_uplift:
    st.subheader("Campaign Uplift Modeling")
    uplift_df = load_uplift_data()
    if not uplift_df.empty:
        uplift_df = uplift_df.copy()
        uplift_df["CLV"] = pd.to_numeric(uplift_df["CLV"], errors='coerce').fillna(0)
        uplift_df = uplift_df[uplift_df["CLV"] > 0]

        if uplift_df.empty:
            st.warning("No valid CLV data for uplift visualization.")
        else:
            st.metric("Avg Uplift", f"{uplift_df['uplift'].mean():.1f}")
            uplift_df["size_scaled"] = uplift_df["CLV"] / uplift_df["CLV"].max() * 30 + 5

            fig = px.scatter(
                uplift_df,
                x="churn_probability",
                y="uplift",
                size="size_scaled",
                color="treatment_group",
                hover_data=["customer_id"],
                title="Uplift vs Risk",
                template=PLOT_TEMPLATE,
                color_discrete_sequence=["#ff6b6b", "#4ecdc4"]
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=TEXT_COLOR if is_dark else "#2a2a2a"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Enable uplift in config.yaml and rerun pipeline.")


# ====================== TAB: FORECAST ======================
with tab_forecast:
    st.subheader("30-Day CLV Forecast")
    forecast_df = load_forecast_data()
    if not forecast_df.empty:
        # Clean forecast data
        forecast_df["historical_clv"] = pd.to_numeric(forecast_df["historical_clv"], errors='coerce').fillna(0)
        forecast_df["forecast_clv"] = pd.to_numeric(forecast_df["forecast_clv"], errors='coerce').fillna(0)

        city = st.selectbox("Select City", forecast_df["customer_id"].unique(), key="forecast_city")
        city_data = forecast_df[forecast_df["customer_id"] == city]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=city_data["date"], y=city_data["historical_clv"],
            name="Historical", line=dict(color="gray")
        ))
        fig.add_trace(go.Scatter(
            x=city_data["date"], y=city_data["forecast_clv"],
            name="Forecast", line=dict(dash="dot", color=PRIMARY_COLOR)
        ))
        fig.update_layout(
            title=f"CLV Forecast: {city}",
            template=PLOT_TEMPLATE,
            font_color=TEXT_COLOR if is_dark else "#2a2a2a"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write("**Next 30 Days:**", format_currency(city_data["forecast_clv"].sum()))
    else:
        st.info("Run forecasting module in pipeline.")


# ====================== TAB: DIAGNOSTICS ======================
with tab_diagnostics:
    st.subheader("Run Diagnostics")
    history = load_pipeline_history()
    if history:
        hist_df = pd.DataFrame(history)
        st.dataframe(hist_df.sort_values("run_timestamp", ascending=False), use_container_width=True)
    else:
        st.info("No run history found yet.")

    st.markdown("### Configuration Snapshot")
    from pathlib import Path
    config_path = Path("config/config.yaml")
    if config_path.exists():
        st.code(config_path.read_text(), language="yaml")