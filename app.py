"""Vello Install Base Simulator - Streamlit App."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta

from config import ALL_TEAMS, PROJECTION_YEARS, MONTE_CARLO_SIMULATIONS
from data_loader import load_all_data, compute_historical_defaults
from simulation import compute_run_rate_params, run_monte_carlo, run_scenario_simulation

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Vello Install Base Simulator",
    page_icon="📊",
    layout="wide",
)

st.title("Vello Install Base Simulator")
st.caption("Strategic planning tool — Monte Carlo projections of install base growth")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with st.spinner("Loading historical data from Snowflake..."):
    data = load_all_data()

ib_df = data["install_base"]
churns_df = data["churns"]

# Latest install base
latest_month = ib_df["REPORTING_MONTH"].max()
latest_ib = int(ib_df[ib_df["REPORTING_MONTH"] == latest_month]["VELLO_INSTALL_BASE"].values[0])

st.metric("Current Vello Install Base", f"{latest_ib:,}", f"As of {latest_month.strftime('%B %Y')}")

# ---------------------------------------------------------------------------
# Compute run rate baseline
# ---------------------------------------------------------------------------
run_rate_params = compute_run_rate_params(data)
defaults = compute_historical_defaults(data)

churn_mean = run_rate_params["churn_rate_mean"]
churn_std = run_rate_params["churn_rate_std"]

# Team-level run rate params (for baseline projection)
baseline_team_params = {
    k: v for k, v in run_rate_params.items() if k not in ("churn_rate_mean", "churn_rate_std")
}

# Run baseline projection
projection_months = PROJECTION_YEARS * 12
baseline_result = run_monte_carlo(
    starting_install_base=latest_ib,
    team_params=baseline_team_params,
    churn_rate_mean=churn_mean,
    churn_rate_std=churn_std,
    projection_months=projection_months,
)

# Build projection dates
projection_dates = [
    latest_month + relativedelta(months=i) for i in range(1, projection_months + 1)
]

# ---------------------------------------------------------------------------
# Scenario planner (dialog modal)
# ---------------------------------------------------------------------------

# Initialize scenario result in session state
if "scenario_result" not in st.session_state:
    st.session_state.scenario_result = None
if "scenario_name" not in st.session_state:
    st.session_state.scenario_name = ""


@st.dialog("Scenario Planner", width="large")
def scenario_planner_modal():
    """Modal dialog for creating scenario simulations."""
    st.markdown("Adjust parameters by team, then click **Run Simulation** to project the scenario.")

    scenario_name = st.text_input("Scenario Name", value="My Scenario", key="modal_scenario_name")

    # Global controls
    st.subheader("Global Settings")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        lead_growth = st.slider(
            "Monthly Lead/SQL Growth Rate (%)",
            min_value=-5.0, max_value=10.0, value=0.0, step=0.1,
            key="global_lead_growth",
            help="Compound monthly growth applied to SQLs across all teams",
        )
    with col_g2:
        churn_override = st.slider(
            "Monthly Churn Rate (%)",
            min_value=0.0, max_value=5.0,
            value=round(churn_mean * 100, 2), step=0.05,
            key="global_churn",
        )

    # Team tabs
    team_tabs = st.tabs(ALL_TEAMS)
    scenario_params = {}

    for tab, team in zip(team_tabs, ALL_TEAMS):
        with tab:
            d = defaults.get(team, {})
            has_attach = team in ("SOFTWARE SALES", "ESAM")

            st.markdown(f"**{team}** — adjust the levers below")
            c1, c2 = st.columns(2)

            with c1:
                mqls = st.number_input(
                    "MQLs / Month",
                    min_value=0, max_value=5000,
                    value=int(d.get("mqls_per_month", 100)),
                    step=10,
                    key=f"{team}_mqls",
                )
                mql_conv = st.slider(
                    "MQL → SQL Conversion Rate (%)",
                    min_value=0.0, max_value=100.0,
                    value=float(d.get("mql_conversion_rate", 10.0)),
                    step=0.5,
                    key=f"{team}_mql_conv",
                    help="Informational — SQLs are set independently since MQL-to-SQL mapping isn't 1:1",
                )
                sqls = st.number_input(
                    "SQLs / Month",
                    min_value=0, max_value=2000,
                    value=int(d.get("sqls_per_month", 50)),
                    step=5,
                    key=f"{team}_sqls",
                )

            with c2:
                sql_conv = st.slider(
                    "SQL Win Rate (%)",
                    min_value=0.0, max_value=100.0,
                    value=float(d.get("sql_conversion_rate", 20.0)),
                    step=0.5,
                    key=f"{team}_sql_conv",
                )
                tts = st.number_input(
                    "Time to Sale (days)",
                    min_value=1, max_value=365,
                    value=int(d.get("time_to_sale_days", 60)),
                    step=5,
                    key=f"{team}_tts",
                )
                tti = st.number_input(
                    "Time to Implement (days)",
                    min_value=1, max_value=365,
                    value=int(d.get("time_to_implement_days", 90)),
                    step=5,
                    key=f"{team}_tti",
                )

            attach = 100.0
            if has_attach:
                attach = st.slider(
                    "Vello Attach Rate (%)",
                    min_value=0.0, max_value=100.0,
                    value=float(d.get("vello_attach_rate", 30.0)),
                    step=1.0,
                    key=f"{team}_attach",
                    help="% of converted deals that include Vello",
                )

            # Show throughput preview
            monthly_wins = sqls * (sql_conv / 100.0) * (attach / 100.0)
            st.info(
                f"**Projected monthly Vello placements from {team}:** "
                f"{sqls} SQLs × {sql_conv:.1f}% win rate"
                + (f" × {attach:.1f}% attach" if has_attach else "")
                + f" = **{monthly_wins:.1f}** placements/month "
                f"(after ~{tts + tti} day lag)"
            )

            scenario_params[team] = {
                "sqls_per_month": sqls,
                "sql_conversion_rate": sql_conv,
                "time_to_sale_days": tts,
                "time_to_implement_days": tti,
                "vello_attach_rate": attach,
            }

    st.divider()

    if st.button("Run Simulation", type="primary", use_container_width=True):
        with st.spinner(f"Running {MONTE_CARLO_SIMULATIONS:,} simulations..."):
            result = run_scenario_simulation(
                starting_install_base=latest_ib,
                scenario_params=scenario_params,
                churn_rate_mean=churn_override / 100.0,
                churn_rate_std=churn_std,
                lead_growth_rate=lead_growth / 100.0,
                projection_months=projection_months,
            )
        st.session_state.scenario_result = result
        st.session_state.scenario_name = scenario_name
        st.success("Simulation complete! Close this dialog to see results on the chart.")
        st.rerun()


# Button to open modal
if st.button("Create Scenario", type="primary"):
    scenario_planner_modal()

# ---------------------------------------------------------------------------
# Build the chart
# ---------------------------------------------------------------------------
fig = go.Figure()

# 1) Actuals (historical install base)
fig.add_trace(go.Scatter(
    x=ib_df["REPORTING_MONTH"],
    y=ib_df["VELLO_INSTALL_BASE"],
    mode="lines+markers",
    name="Actuals",
    line=dict(color="#1f77b4", width=3),
    marker=dict(size=4),
))

# 2) Run Rate projection (median)
fig.add_trace(go.Scatter(
    x=projection_dates,
    y=baseline_result["median"],
    mode="lines",
    name="Run Rate (Median)",
    line=dict(color="#ff7f0e", width=2, dash="dash"),
))

# 3) Run Rate confidence interval
fig.add_trace(go.Scatter(
    x=projection_dates + projection_dates[::-1],
    y=np.concatenate([baseline_result["p95"], baseline_result["p5"][::-1]]).tolist(),
    fill="toself",
    fillcolor="rgba(255, 127, 14, 0.15)",
    line=dict(color="rgba(255, 127, 14, 0)"),
    name="Run Rate 90% CI",
    showlegend=True,
))

# 4) Scenario (if exists)
if st.session_state.scenario_result is not None:
    sr = st.session_state.scenario_result
    sname = st.session_state.scenario_name or "Scenario"

    fig.add_trace(go.Scatter(
        x=projection_dates,
        y=sr["median"],
        mode="lines",
        name=f"{sname} (Median)",
        line=dict(color="#2ca02c", width=2, dash="dot"),
    ))

    fig.add_trace(go.Scatter(
        x=projection_dates + projection_dates[::-1],
        y=np.concatenate([sr["p95"], sr["p5"][::-1]]).tolist(),
        fill="toself",
        fillcolor="rgba(44, 160, 44, 0.12)",
        line=dict(color="rgba(44, 160, 44, 0)"),
        name=f"{sname} 90% CI",
        showlegend=True,
    ))

# Layout
fig.update_layout(
    title="Vello Install Base — Actuals & Projections",
    xaxis_title="Month",
    yaxis_title="Install Base (Sites)",
    height=600,
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
    hovermode="x unified",
)

# Add a vertical line at the boundary between actuals and projection
fig.add_vline(
    x=latest_month,
    line_dash="dash",
    line_color="gray",
    annotation_text="Current",
    annotation_position="top",
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
st.subheader("Projection Summary")

summary_milestones = [12, 24, 36, 48, 60]  # months out
rows = []
for m in summary_milestones:
    if m <= projection_months:
        date_label = (latest_month + relativedelta(months=m)).strftime("%b %Y")
        row = {
            "Milestone": f"+{m} months ({date_label})",
            "Run Rate Median": f"{baseline_result['median'][m-1]:,.0f}",
            "Run Rate 5th %ile": f"{baseline_result['p5'][m-1]:,.0f}",
            "Run Rate 95th %ile": f"{baseline_result['p95'][m-1]:,.0f}",
        }
        if st.session_state.scenario_result is not None:
            sr = st.session_state.scenario_result
            sn = st.session_state.scenario_name or "Scenario"
            row[f"{sn} Median"] = f"{sr['median'][m-1]:,.0f}"
            row[f"{sn} 5th %ile"] = f"{sr['p5'][m-1]:,.0f}"
            row[f"{sn} 95th %ile"] = f"{sr['p95'][m-1]:,.0f}"
        rows.append(row)

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Historical data explorer (expandable)
# ---------------------------------------------------------------------------
with st.expander("Historical Data Explorer"):
    data_tab1, data_tab2, data_tab3, data_tab4, data_tab5 = st.tabs([
        "Install Base", "Churns", "SQLs Created", "Conversion Rates", "MQL Conversion"
    ])
    with data_tab1:
        st.dataframe(ib_df.sort_values("REPORTING_MONTH", ascending=False), use_container_width=True, hide_index=True)
    with data_tab2:
        st.dataframe(churns_df.sort_values("REPORTING_MONTH", ascending=False), use_container_width=True, hide_index=True)
    with data_tab3:
        st.dataframe(data["sqls_created"].sort_values(["TEAM_NAME", "MONTH"], ascending=[True, False]), use_container_width=True, hide_index=True)
    with data_tab4:
        conv_display = data["conversion_rates"].sort_values(
            ["TEAM_NAME", "REPORTING_MONTH"], ascending=[True, False]
        )
        st.dataframe(conv_display, use_container_width=True, hide_index=True)
    with data_tab5:
        st.dataframe(data["mql_conversion"].sort_values("MONTH", ascending=False), use_container_width=True, hide_index=True)
