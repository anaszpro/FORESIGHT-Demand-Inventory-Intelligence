"""
app.py — Project FORESIGHT planning dashboard (D5)

Run: streamlit run app/app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

st.set_page_config(page_title="FORESIGHT — NorthBay Living Planning Dashboard", layout="wide")


@st.cache_data
def load_data():
    sales = pd.read_parquet(PROCESSED / "analysis_ready.parquet")
    risk = pd.read_parquet(PROCESSED / "risk_scored.parquet")
    forecast = pd.read_parquet(PROCESSED / "forecast_forward.parquet")
    sales["date"] = pd.to_datetime(sales["date"])
    return sales, risk, forecast


def empty_state(msg):
    st.info(msg)


try:
    sales, risk, forecast = load_data()
except FileNotFoundError:
    st.error(
        "Processed data not found. Run the pipeline first:\n\n"
        "```\npython src/pipeline.py\npython src/forecast.py\npython src/risk.py\n```"
    )
    st.stop()

st.title("📦 FORESIGHT — Demand & Inventory Intelligence")
st.caption("NorthBay Living · Planning Dashboard · Data Scientist: Zidio Internship engagement")

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")
categories = sorted(risk["category"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect("Category", categories, default=categories)

quadrants = sorted(risk["quadrant"].unique().tolist())
selected_quadrants = st.sidebar.multiselect("Risk quadrant", quadrants, default=quadrants)

sku_search = st.sidebar.text_input("Search SKU ID")

filtered_risk = risk[
    risk["category"].isin(selected_categories) & risk["quadrant"].isin(selected_quadrants)
]
if sku_search:
    filtered_risk = filtered_risk[filtered_risk["sku_id"].str.contains(sku_search, case=False)]

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("SKUs tracked", f"{len(risk):,}")
c2.metric("Sales at risk (stockouts)", f"₹{risk['sales_at_risk'].sum():,.0f}")
c3.metric("Capital locked (overstock)", f"₹{risk['capital_locked'].sum():,.0f}")
reorder_ct = (risk["quadrant"] == "Reorder Now").sum()
c4.metric("SKUs needing reorder now", f"{reorder_ct}")

st.divider()

# ---------------------------------------------------------------------------
# Decisioning grid
# ---------------------------------------------------------------------------
st.subheader("Decisioning view — stockout vs overstock risk")
if filtered_risk.empty:
    empty_state("No SKUs match the current filters. Try widening your selection.")
else:
    fig = px.scatter(
        filtered_risk,
        x="overstock_risk",
        y="stockout_risk",
        size="rupee_value_at_stake",
        color="quadrant",
        hover_data=["sku_id", "category", "forecast_total_units", "rupee_value_at_stake"],
        color_discrete_map={
            "Reorder Now": "#d62728",
            "Markdown / Clear": "#6a5acd",
            "Watch / Volatile": "#daa520",
            "Healthy": "#2ca02c",
        },
        labels={"overstock_risk": "Overstock risk →", "stockout_risk": "Stockout risk ↑"},
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Prioritised reorder / markdown list
# ---------------------------------------------------------------------------
st.subheader("Prioritised action list")
tab1, tab2, tab3 = st.tabs(["🔴 Reorder Now", "🟣 Markdown / Clear", "🟡 Watch / Volatile"])

def show_action_table(df, quadrant_name):
    q = df[df["quadrant"] == quadrant_name].sort_values("rupee_value_at_stake", ascending=False)
    if q.empty:
        empty_state(f"No SKUs currently flagged as '{quadrant_name}'.")
    else:
        st.dataframe(
            q[["sku_id", "category", "forecast_total_units", "on_hand_units",
               "on_order_units", "lead_time_days", "rupee_value_at_stake", "recommended_action"]],
            use_container_width=True,
            hide_index=True,
        )

with tab1:
    show_action_table(filtered_risk, "Reorder Now")
with tab2:
    show_action_table(filtered_risk, "Markdown / Clear")
with tab3:
    show_action_table(filtered_risk, "Watch / Volatile")

st.divider()

# ---------------------------------------------------------------------------
# SKU drill-down: forecast vs actual
# ---------------------------------------------------------------------------
st.subheader("SKU drill-down — forecast vs history")
sku_options = sorted(risk["sku_id"].unique().tolist())
default_sku = filtered_risk["sku_id"].iloc[0] if not filtered_risk.empty else sku_options[0]
picked_sku = st.selectbox("Choose a SKU", sku_options, index=sku_options.index(default_sku))

sku_hist = sales[sales["sku_id"] == picked_sku].sort_values("date").tail(120)
sku_fore = forecast[forecast["sku_id"] == picked_sku]
sku_risk_row = risk[risk["sku_id"] == picked_sku].iloc[0]

if sku_hist.empty:
    empty_state("No sales history available for this SKU.")
else:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sku_hist["date"], y=sku_hist["units_sold"],
                               mode="lines", name="Actual demand", line=dict(color="black")))
    if not sku_fore.empty:
        last_date = sku_hist["date"].max()
        f_row = sku_fore.iloc[0]
        fig2.add_trace(go.Scatter(
            x=[last_date, last_date + pd.Timedelta(days=f_row["forecast_horizon_days"])],
            y=[f_row["forecast_daily_mean"], f_row["forecast_daily_mean"]],
            mode="lines", name="Forecast (avg daily)", line=dict(color="royalblue", dash="dash"),
        ))
    fig2.update_layout(height=350, xaxis_title="Date", yaxis_title="Units / day")
    st.plotly_chart(fig2, use_container_width=True)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Forecast (next 8 wks)", f"{sku_risk_row['forecast_total_units']:.0f} units")
    col_b.metric("On-hand + on-order", f"{sku_risk_row['on_hand_units']:.0f} + {sku_risk_row['on_order_units']:.0f}")
    col_c.metric("Risk quadrant", sku_risk_row["quadrant"])
    st.caption(f"Recommended action: **{sku_risk_row['recommended_action']}**")

st.divider()
st.caption(
    "FORESIGHT · Zidio Development internship engagement for NorthBay Living · "
    "Forecast beats seasonal-naive baseline on rolling-origin backtest — see backtest_results.json."
)
