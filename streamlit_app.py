"""Vello Install Base Simulator — single-file Streamlit app."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from snowflake.snowpark.context import get_active_session
from dateutil.relativedelta import relativedelta

# =============================================================================
# CONFIG
# =============================================================================
ALL_TEAMS = ["ISAS", "SOFTWARE SALES", "ESAM"]
LEAD_SOURCES = ["VDC", "Other Leads"]
PROJECTION_YEARS = 5
MONTE_CARLO_SIMULATIONS = 1000
CONFIDENCE_LEVEL = 0.90

# =============================================================================
# SQL QUERIES
# =============================================================================
SQLS_CREATED_QUERY = """
SELECT
    DATE_TRUNC('MONTH', OPPORTUNITY_CREATED_DATE)    AS MONTH,
    TEAM_NAME,
    CASE WHEN LEADSOURCE = 'IDEXX REFERRAL' THEN 'VDC' ELSE 'Other Leads' END AS LEAD_SOURCE_GROUP,
    COUNT(DISTINCT OPPORTUNITY_ID)                   AS SQLS_CREATED
FROM VSSANALYTICS_DB.SFDC.CDL_SALESFORCE
WHERE OPPORTUNITY_CREATED_DATE IS NOT NULL
  AND (
      (TEAM_NAME = 'ISAS' AND PRODUCT = 'Vello')
      OR (TEAM_NAME IN ('SOFTWARE SALES EAST', 'SOFTWARE SALES WEST')
          AND PRODUCT IN ('ezyVet', 'Neo', 'Vello'))
      OR (TEAM_NAME = 'ESAM'
          AND PRODUCT IN ('ezyVet', 'ezyVet Enterprise - GP', 'ezyVet Enterprise - Spec/ER'))
  )
GROUP BY 1, 2, 3
ORDER BY 2, 3, 1
"""

CONVERSION_RATES_QUERY = """
WITH opps AS (
    SELECT
        TEAM_NAME,
        CASE WHEN LEADSOURCE = 'IDEXX REFERRAL' THEN 'VDC' ELSE 'Other Leads' END AS LEAD_SOURCE_GROUP,
        OPPORTUNITY_ID,
        CURRENT_STAGE,
        OPPORTUNITY_CREATED_DATE,
        ACTUAL_CLOSE_DATE
    FROM VSSANALYTICS_DB.SFDC.CDL_SALESFORCE
    WHERE OPPORTUNITY_CREATED_DATE IS NOT NULL
      AND (
          (TEAM_NAME = 'ISAS' AND PRODUCT = 'Vello')
          OR (TEAM_NAME IN ('SOFTWARE SALES EAST', 'SOFTWARE SALES WEST')
              AND PRODUCT IN ('ezyVet', 'Neo', 'Vello'))
          OR (TEAM_NAME = 'ESAM'
              AND PRODUCT IN ('ezyVet', 'ezyVet Enterprise - GP', 'ezyVet Enterprise - Spec/ER'))
      )
)
SELECT
    DATE_TRUNC('MONTH', OPPORTUNITY_CREATED_DATE)                                           AS COHORT_MONTH,
    TEAM_NAME,
    LEAD_SOURCE_GROUP,
    COUNT(DISTINCT OPPORTUNITY_ID)                                                          AS TOTAL_OPPS,
    COUNT(DISTINCT CASE WHEN CURRENT_STAGE = 'WON' THEN OPPORTUNITY_ID END)                AS WINS,
    COUNT(DISTINCT CASE WHEN CURRENT_STAGE IN ('CANCELLED', 'LOST') THEN OPPORTUNITY_ID END) AS CLOSED_LOST,
    COUNT(DISTINCT CASE WHEN CURRENT_STAGE NOT IN ('WON', 'CANCELLED', 'LOST', 'CAG PARENT CLOSED')
        AND DATEDIFF('DAY', OPPORTUNITY_CREATED_DATE, CURRENT_DATE()) >= 180
        THEN OPPORTUNITY_ID END)                                                            AS OPEN_STALE,
    COUNT(DISTINCT CASE WHEN CURRENT_STAGE NOT IN ('WON', 'CANCELLED', 'LOST', 'CAG PARENT CLOSED')
        AND DATEDIFF('DAY', OPPORTUNITY_CREATED_DATE, CURRENT_DATE()) < 180
        THEN OPPORTUNITY_ID END)                                                            AS OPEN_ACTIVE
FROM opps
WHERE DATEDIFF('MONTH', OPPORTUNITY_CREATED_DATE, CURRENT_DATE()) >= 6
GROUP BY 1, 2, 3
ORDER BY 2, 3, 1
"""

VELLO_ATTACH_QUERY = """
WITH team_opps AS (
    SELECT DISTINCT
        DATE_TRUNC('MONTH', ACTUAL_CLOSE_DATE)                                              AS CLOSE_MONTH,
        TEAM_NAME,
        CASE WHEN LEADSOURCE = 'IDEXX REFERRAL' THEN 'VDC' ELSE 'Other Leads' END          AS LEAD_SOURCE_GROUP,
        OPPORTUNITY_ID
    FROM VSSANALYTICS_DB.SFDC.CDL_SALESFORCE
    WHERE CURRENT_STAGE = 'WON'
      AND ACTUAL_CLOSE_DATE IS NOT NULL
      AND (
          (TEAM_NAME IN ('SOFTWARE SALES EAST', 'SOFTWARE SALES WEST')
              AND PRODUCT IN ('ezyVet', 'Neo', 'Vello'))
          OR (TEAM_NAME = 'ESAM'
              AND PRODUCT IN ('ezyVet', 'ezyVet Enterprise - GP', 'ezyVet Enterprise - Spec/ER'))
      )
),
vello_flag AS (
    SELECT DISTINCT OPPORTUNITY_ID
    FROM VSSANALYTICS_DB.SFDC.CDL_SALESFORCE
    WHERE PRODUCT = 'Vello'
      AND CURRENT_STAGE = 'WON'
)
SELECT
    t.CLOSE_MONTH,
    t.TEAM_NAME,
    t.LEAD_SOURCE_GROUP,
    COUNT(DISTINCT t.OPPORTUNITY_ID)                                                        AS TOTAL_WON,
    COUNT(DISTINCT v.OPPORTUNITY_ID)                                                        AS VELLO_WON
FROM team_opps t
LEFT JOIN vello_flag v ON t.OPPORTUNITY_ID = v.OPPORTUNITY_ID
GROUP BY 1, 2, 3
ORDER BY 2, 3, 1
"""

TIME_TO_BOOKING_QUERY = """
SELECT
    DATE_TRUNC('MONTH', ACTUAL_CLOSE_DATE)                              AS MONTH,
    TEAM_NAME,
    CASE WHEN LEADSOURCE = 'IDEXX REFERRAL' THEN 'VDC' ELSE 'Other Leads' END AS LEAD_SOURCE_GROUP,
    COUNT(DISTINCT OPPORTUNITY_ID)                                      AS BOOKINGS,
    ROUND(AVG(DATEDIFF('DAY', OPPORTUNITY_CREATED_DATE, ACTUAL_CLOSE_DATE)), 1) AS AVG_DAYS_TO_BOOKING,
    ROUND(MEDIAN(DATEDIFF('DAY', OPPORTUNITY_CREATED_DATE, ACTUAL_CLOSE_DATE)), 1) AS MEDIAN_DAYS_TO_BOOKING
FROM VSSANALYTICS_DB.SFDC.CDL_SALESFORCE
WHERE CURRENT_STAGE = 'WON'
  AND ACTUAL_CLOSE_DATE IS NOT NULL
  AND OPPORTUNITY_CREATED_DATE IS NOT NULL
  AND (
      (TEAM_NAME = 'ISAS' AND PRODUCT = 'Vello')
      OR (TEAM_NAME IN ('SOFTWARE SALES EAST', 'SOFTWARE SALES WEST')
          AND PRODUCT IN ('ezyVet', 'Neo', 'Vello'))
      OR (TEAM_NAME = 'ESAM'
          AND PRODUCT IN ('ezyVet', 'ezyVet Enterprise - GP', 'ezyVet Enterprise - Spec/ER'))
  )
GROUP BY 1, 2, 3
ORDER BY 2, 3, 1
"""

TIME_TO_PLACEMENT_QUERY = """
SELECT
    DATE_TRUNC('MONTH', CF_GO_LIVE_DATE)                                AS MONTH,
    CASE
        WHEN CUSTOMER_TYPE = 'Enterprise' THEN 'ESAM'
        WHEN ONBOARDING_TYPE IN ('Conversion', 'Fresh') THEN 'SOFTWARE SALES'
        ELSE 'ISAS'
    END                                                                 AS TEAM_NAME,
    COUNT(DISTINCT PROJECT_ID)                                          AS PLACEMENTS,
    ROUND(AVG(DATEDIFF('DAY', CREATED_DATE, CF_GO_LIVE_DATE)), 1)      AS AVG_DAYS_TO_PLACEMENT,
    ROUND(MEDIAN(DATEDIFF('DAY', CREATED_DATE, CF_GO_LIVE_DATE)), 1)   AS MEDIAN_DAYS_TO_PLACEMENT
FROM VSSANALYTICS_DB.GUIDECX.CDL_ONBOARDING
WHERE CF_GO_LIVE_DATE IS NOT NULL
  AND CREATED_DATE IS NOT NULL
  AND STATUS NOT IN ('CANCELLED','ON-HOLD')
  AND VELLO_BOOLEAN = 1
GROUP BY 1, 2
ORDER BY 2, 1
"""

INSTALL_BASE_QUERY = """
SELECT
    REPORTING_MONTH,
    COUNT(DISTINCT SAP_ID)    AS VELLO_INSTALL_BASE
FROM VSSANALYTICS_DB.VSSANALYTICS.CDL_INSTALL_BASE
WHERE PRODUCT = 'Vello'
GROUP BY 1
ORDER BY 1
"""

CHURNS_QUERY = """
SELECT
    REPORTING_MONTH,
    COUNT(DISTINCT SAP_ID)    AS VELLO_CHURNS
FROM VSSANALYTICS_DB.OPERATIONS.CDL_ATTRITION
WHERE PRODUCT = 'Vello'
GROUP BY 1
ORDER BY 1
"""


# =============================================================================
# DATA LOADING
# =============================================================================
def get_session():
    return get_active_session()


def _run_query(query: str) -> pd.DataFrame:
    session = get_session()
    df = session.sql(query).to_pandas()
    df.columns = [c.upper() for c in df.columns]
    return df


TEAM_RENAME = {"SOFTWARE SALES EAST": "SOFTWARE SALES", "SOFTWARE SALES WEST": "SOFTWARE SALES"}


@st.cache_data(ttl=3600)
def load_sqls_created() -> pd.DataFrame:
    df = _run_query(SQLS_CREATED_QUERY)
    df["MONTH"] = pd.to_datetime(df["MONTH"])
    df["TEAM_NAME"] = df["TEAM_NAME"].replace(TEAM_RENAME)
    return df.groupby(["MONTH", "TEAM_NAME", "LEAD_SOURCE_GROUP"], as_index=False)["SQLS_CREATED"].sum()


@st.cache_data(ttl=3600)
def load_conversion_rates() -> pd.DataFrame:
    df = _run_query(CONVERSION_RATES_QUERY)
    df["COHORT_MONTH"] = pd.to_datetime(df["COHORT_MONTH"])
    df["TEAM_NAME"] = df["TEAM_NAME"].replace(TEAM_RENAME)
    combined = df.groupby(["COHORT_MONTH", "TEAM_NAME", "LEAD_SOURCE_GROUP"], as_index=False).agg(
        {"TOTAL_OPPS": "sum", "WINS": "sum", "CLOSED_LOST": "sum", "OPEN_STALE": "sum", "OPEN_ACTIVE": "sum"}
    )
    combined["RESOLVED_TOTAL"] = combined["WINS"] + combined["CLOSED_LOST"] + combined["OPEN_STALE"]
    combined["CONVERSION_RATE_PCT"] = (100.0 * combined["WINS"] / combined["RESOLVED_TOTAL"].replace(0, pd.NA)).round(1)
    return combined


@st.cache_data(ttl=3600)
def load_vello_attach() -> pd.DataFrame:
    df = _run_query(VELLO_ATTACH_QUERY)
    df["CLOSE_MONTH"] = pd.to_datetime(df["CLOSE_MONTH"])
    df["TEAM_NAME"] = df["TEAM_NAME"].replace(TEAM_RENAME)
    combined = df.groupby(["CLOSE_MONTH", "TEAM_NAME", "LEAD_SOURCE_GROUP"], as_index=False).agg(
        {"TOTAL_WON": "sum", "VELLO_WON": "sum"}
    )
    combined["VELLO_ATTACH_PCT"] = (100.0 * combined["VELLO_WON"] / combined["TOTAL_WON"].replace(0, pd.NA)).round(1)
    return combined


@st.cache_data(ttl=3600)
def load_time_to_booking() -> pd.DataFrame:
    df = _run_query(TIME_TO_BOOKING_QUERY)
    df["MONTH"] = pd.to_datetime(df["MONTH"])
    df["TEAM_NAME"] = df["TEAM_NAME"].replace(TEAM_RENAME)
    return df.groupby(["MONTH", "TEAM_NAME", "LEAD_SOURCE_GROUP"], as_index=False).agg(
        {"BOOKINGS": "sum", "AVG_DAYS_TO_BOOKING": "mean", "MEDIAN_DAYS_TO_BOOKING": "mean"}
    )


@st.cache_data(ttl=3600)
def load_time_to_placement() -> pd.DataFrame:
    df = _run_query(TIME_TO_PLACEMENT_QUERY)
    df["MONTH"] = pd.to_datetime(df["MONTH"])
    return df


@st.cache_data(ttl=3600)
def load_install_base() -> pd.DataFrame:
    df = _run_query(INSTALL_BASE_QUERY)
    df["REPORTING_MONTH"] = pd.to_datetime(df["REPORTING_MONTH"])
    return df


@st.cache_data(ttl=3600)
def load_churns() -> pd.DataFrame:
    df = _run_query(CHURNS_QUERY)
    df["REPORTING_MONTH"] = pd.to_datetime(df["REPORTING_MONTH"])
    return df


def load_all_data() -> dict:
    return {
        "sqls_created": load_sqls_created(),
        "conversion_rates": load_conversion_rates(),
        "vello_attach": load_vello_attach(),
        "time_to_booking": load_time_to_booking(),
        "time_to_placement": load_time_to_placement(),
        "install_base": load_install_base(),
        "churns": load_churns(),
    }


# =============================================================================
# HISTORICAL DEFAULTS HELPERS
# =============================================================================
def _safe_team_mean(df, team, col):
    subset = df[df["TEAM_NAME"] == team] if "TEAM_NAME" in df.columns else df
    if subset.empty or col not in subset.columns:
        return 0.0
    vals = subset[col].dropna()
    return vals.mean() if not vals.empty else 0.0


def _safe_team_monthly_total(df, team, month_col, value_col):
    """Average of monthly TOTALS (sum across lead sources per month, then average)."""
    subset = df[df["TEAM_NAME"] == team] if "TEAM_NAME" in df.columns else df
    if subset.empty or value_col not in subset.columns:
        return 0.0
    monthly = subset.groupby(month_col)[value_col].sum()
    return monthly.mean() if not monthly.empty else 0.0


def _safe_team_conv(df, team, n_cohorts=6):
    """Volume-weighted win rate across the most recent N cohort months for a team."""
    subset = df[df["TEAM_NAME"] == team]
    if subset.empty:
        return 0.0
    recent = subset["COHORT_MONTH"].drop_duplicates().nlargest(n_cohorts)
    subset = subset[subset["COHORT_MONTH"].isin(recent)]
    total_resolved = subset["RESOLVED_TOTAL"].sum()
    return (100.0 * subset["WINS"].sum() / total_resolved) if total_resolved > 0 else 0.0


def _safe_team_ls_monthly_total(df, team, lead_source, month_col, value_col):
    """Average monthly total for a specific team + lead source."""
    subset = df[(df["TEAM_NAME"] == team) & (df["LEAD_SOURCE_GROUP"] == lead_source)]
    if subset.empty or value_col not in subset.columns:
        return 0.0
    monthly = subset.groupby(month_col)[value_col].sum()
    return monthly.mean() if not monthly.empty else 0.0


def _safe_team_ls_conv(df, team, lead_source, n_cohorts=6):
    """Volume-weighted win rate across the most recent N cohort months for a team + lead source."""
    subset = df[(df["TEAM_NAME"] == team) & (df["LEAD_SOURCE_GROUP"] == lead_source)]
    if subset.empty:
        return 0.0
    recent = subset["COHORT_MONTH"].drop_duplicates().nlargest(n_cohorts)
    subset = subset[subset["COHORT_MONTH"].isin(recent)]
    total_resolved = subset["RESOLVED_TOTAL"].sum()
    return (100.0 * subset["WINS"].sum() / total_resolved) if total_resolved > 0 else 0.0


def _safe_team_ls_mean(df, team, lead_source, col):
    """Mean of a column for a specific team + lead source."""
    subset = df[(df["TEAM_NAME"] == team) & (df["LEAD_SOURCE_GROUP"] == lead_source)]
    if subset.empty or col not in subset.columns:
        return 0.0
    vals = subset[col].dropna()
    return vals.mean() if not vals.empty else 0.0


def _cohort_conv_std(df, team, lead_source, n_cohorts=6):
    """Std dev of per-cohort win rates for a team + lead source."""
    subset = df[(df["TEAM_NAME"] == team) & (df["LEAD_SOURCE_GROUP"] == lead_source)]
    if subset.empty:
        return 0.03
    recent = subset["COHORT_MONTH"].drop_duplicates().nlargest(n_cohorts)
    subset = subset[subset["COHORT_MONTH"].isin(recent)]
    cohort_rates = subset.groupby("COHORT_MONTH").apply(
        lambda g: g["WINS"].sum() / max(g["RESOLVED_TOTAL"].sum(), 1)
    )
    return max(cohort_rates.std(), 0.01) if len(cohort_rates) > 1 else 0.03


def _team_ls_attach(attach_df, team, lead_source, n_months=6):
    """Compute recent attach rate for a team + lead source from SFDC data."""
    subset = attach_df[(attach_df["TEAM_NAME"] == team) & (attach_df["LEAD_SOURCE_GROUP"] == lead_source)]
    if subset.empty:
        return 100.0
    recent = subset["CLOSE_MONTH"].drop_duplicates().nlargest(n_months)
    subset = subset[subset["CLOSE_MONTH"].isin(recent)]
    total = subset["TOTAL_WON"].sum()
    vello = subset["VELLO_WON"].sum()
    return round(100.0 * vello / total, 1) if total > 0 else 100.0


def compute_seasonality(ttb_df):
    """Compute monthly seasonal indices from historical booking volumes."""
    if ttb_df.empty:
        return {m: 1.0 for m in range(1, 13)}
    monthly = ttb_df.groupby("MONTH")["BOOKINGS"].sum().reset_index()
    monthly["CAL_MONTH"] = monthly["MONTH"].dt.month
    avg_by_cal = monthly.groupby("CAL_MONTH")["BOOKINGS"].mean()
    overall_avg = avg_by_cal.mean()
    if overall_avg == 0:
        return {m: 1.0 for m in range(1, 13)}
    indices = (avg_by_cal / overall_avg).to_dict()
    return {m: round(indices.get(m, 1.0), 3) for m in range(1, 13)}


def compute_historical_defaults(data):
    defaults = {}
    recent_months = 6

    sqls_df = data["sqls_created"]
    sqls_recent = sqls_df[sqls_df["MONTH"] >= sqls_df["MONTH"].max() - pd.DateOffset(months=recent_months)]
    conv_df = data["conversion_rates"]
    attach_df = data["vello_attach"]
    ttb_df = data["time_to_booking"]
    ttb_recent = ttb_df[ttb_df["MONTH"] >= ttb_df["MONTH"].max() - pd.DateOffset(months=recent_months)]
    ttp_df = data["time_to_placement"]
    ttp_recent = ttp_df[ttp_df["MONTH"] >= ttp_df["MONTH"].max() - pd.DateOffset(months=recent_months)]

    # Churn rate
    churns_df = data["churns"]
    ib_df = data["install_base"]
    if not churns_df.empty and not ib_df.empty:
        rc = churns_df[churns_df["REPORTING_MONTH"] >= churns_df["REPORTING_MONTH"].max() - pd.DateOffset(months=recent_months)]["VELLO_CHURNS"].mean()
        ri = ib_df[ib_df["REPORTING_MONTH"] >= ib_df["REPORTING_MONTH"].max() - pd.DateOffset(months=recent_months)]["VELLO_INSTALL_BASE"].mean()
        monthly_churn_rate = rc / ri if ri > 0 else 0.01
    else:
        monthly_churn_rate = 0.01

    for team in ALL_TEAMS:
        # Per-team time to implement from GuideCX
        team_ttp = ttp_recent[ttp_recent["TEAM_NAME"] == team] if "TEAM_NAME" in ttp_recent.columns else ttp_recent
        ttp_val = team_ttp["MEDIAN_DAYS_TO_PLACEMENT"].mean() if not team_ttp.empty else 90.0

        team_defaults = {
            "time_to_implement_days": round(ttp_val, 0),
            "monthly_churn_rate": round(monthly_churn_rate * 100, 2),
        }

        # Attach rate: ISAS is always 100% (Vello-only), others from SFDC data
        if team == "ISAS":
            team_defaults["vello_attach_rate"] = 100.0
        else:
            # Team-level blended attach across lead sources
            team_attach = attach_df[attach_df["TEAM_NAME"] == team]
            if not team_attach.empty:
                recent_close = team_attach["CLOSE_MONTH"].drop_duplicates().nlargest(recent_months)
                ra = team_attach[team_attach["CLOSE_MONTH"].isin(recent_close)]
                total = ra["TOTAL_WON"].sum()
                vello = ra["VELLO_WON"].sum()
                team_defaults["vello_attach_rate"] = round(100.0 * vello / total, 1) if total > 0 else 30.0
            else:
                team_defaults["vello_attach_rate"] = 30.0

        # Per lead-source defaults
        ls_defaults = {}
        for ls in LEAD_SOURCES:
            ls_defaults[ls] = {
                "sqls_per_month": round(_safe_team_ls_monthly_total(sqls_recent, team, ls, "MONTH", "SQLS_CREATED"), 0),
                "sql_conversion_rate": round(_safe_team_ls_conv(conv_df, team, ls), 1),
                "time_to_sale_days": round(_safe_team_ls_mean(ttb_recent, team, ls, "MEDIAN_DAYS_TO_BOOKING"), 0) or round(_safe_team_mean(ttb_recent, team, "MEDIAN_DAYS_TO_BOOKING"), 0),
            }
        team_defaults["lead_sources"] = ls_defaults
        defaults[team] = team_defaults

    return defaults


# =============================================================================
# SIMULATION ENGINE
# =============================================================================
def compute_run_rate_params(data):
    recent_months = 6
    params = {}
    sqls = data["sqls_created"]
    conv = data["conversion_rates"]
    attach_df = data["vello_attach"]
    ttb = data["time_to_booking"]
    ttp = data["time_to_placement"]
    ttp_recent = ttp[ttp["MONTH"] >= ttp["MONTH"].max() - pd.DateOffset(months=recent_months)]

    for team in ALL_TEAMS:
        # Per-team time to implement
        team_ttp = ttp_recent[ttp_recent["TEAM_NAME"] == team] if "TEAM_NAME" in ttp_recent.columns else ttp_recent
        ttp_impl_days = team_ttp["MEDIAN_DAYS_TO_PLACEMENT"].mean() if not team_ttp.empty else 90

        # Attach rate from SFDC (ISAS = 100%)
        if team == "ISAS":
            attach = 1.0
        else:
            attach = _team_ls_attach(attach_df, team, "VDC", recent_months) / 100.0  # placeholder, overridden per-ls below

        for ls in LEAD_SOURCES:
            ls_sqls = sqls[(sqls["TEAM_NAME"] == team) & (sqls["LEAD_SOURCE_GROUP"] == ls)]
            ls_recent = ls_sqls[ls_sqls["MONTH"] >= ls_sqls["MONTH"].max() - pd.DateOffset(months=recent_months)] if not ls_sqls.empty else ls_sqls
            monthly = ls_recent.groupby("MONTH")["SQLS_CREATED"].sum() if not ls_recent.empty else pd.Series(dtype=float)

            cr = _safe_team_ls_conv(conv, team, ls, n_cohorts=recent_months) / 100.0
            cr_std = _cohort_conv_std(conv, team, ls, n_cohorts=recent_months)

            ls_ttb = ttb[(ttb["TEAM_NAME"] == team) & (ttb["LEAD_SOURCE_GROUP"] == ls) & (ttb["MONTH"] >= ttb["MONTH"].max() - pd.DateOffset(months=recent_months))]

            # Per-lead-source attach for non-ISAS teams
            ls_attach = 1.0 if team == "ISAS" else _team_ls_attach(attach_df, team, ls, recent_months) / 100.0

            params[(team, ls)] = {
                "sqls_mean": monthly.mean() if not monthly.empty else 0,
                "sqls_std": max(monthly.std(), 1) if not monthly.empty else 1,
                "conv_rate": cr,
                "conv_rate_std": cr_std,
                "time_to_sale_days": ls_ttb["MEDIAN_DAYS_TO_BOOKING"].mean() if not ls_ttb.empty else 60,
                "time_to_implement_days": ttp_impl_days,
                "attach_rate": ls_attach,
            }

    # Churn
    churns = data["churns"]
    ib = data["install_base"]
    if not churns.empty and not ib.empty:
        rc = churns[churns["REPORTING_MONTH"] >= churns["REPORTING_MONTH"].max() - pd.DateOffset(months=recent_months)]
        ri = ib[ib["REPORTING_MONTH"] >= ib["REPORTING_MONTH"].max() - pd.DateOffset(months=recent_months)]
        params["churn_rate_mean"] = rc["VELLO_CHURNS"].mean() / max(ri["VELLO_INSTALL_BASE"].mean(), 1)
        params["churn_rate_std"] = max(rc["VELLO_CHURNS"].std() / max(ri["VELLO_INSTALL_BASE"].mean(), 1), 0.001)
    else:
        params["churn_rate_mean"] = 0.01
        params["churn_rate_std"] = 0.003

    return params


def run_monte_carlo(
    starting_install_base,
    team_params,
    churn_rate_mean,
    churn_rate_std,
    projection_months=PROJECTION_YEARS * 12,
    n_simulations=MONTE_CARLO_SIMULATIONS,
    growth_rates=None,
    seasonality=None,
    start_calendar_month=1,
):
    """
    growth_rates: dict keyed same as team_params → monthly growth rate per segment.
    seasonality: dict {1..12 → multiplier} for calendar month seasonality.
    start_calendar_month: the calendar month (1-12) of projection month 0.
    """
    rng = np.random.default_rng(42)
    results = np.zeros((n_simulations, projection_months))
    if growth_rates is None:
        growth_rates = {}
    if seasonality is None:
        seasonality = {m: 1.0 for m in range(1, 13)}

    for sim in range(n_simulations):
        ib = float(starting_install_base)
        pipeline = []

        # Pre-fill pipeline with in-flight deals
        for key, tp in team_params.items():
            gr = growth_rates.get(key, 0.0)
            sale_months = max(tp["time_to_sale_days"] / 30.0, 0.5)
            impl_months = max(tp["time_to_implement_days"] / 30.0, 0.5)
            total_lag = sale_months + impl_months
            for lag_month in range(int(np.ceil(total_lag))):
                growth_factor = (1 + gr) ** max(0, -lag_month)
                cal_month = ((start_calendar_month - 1 - lag_month) % 12) + 1
                seasonal = seasonality.get(cal_month, 1.0)
                s = max(rng.normal(tp["sqls_mean"] * growth_factor * seasonal, tp["sqls_std"]), 0)
                c = np.clip(rng.normal(tp["conv_rate"], tp.get("conv_rate_std", 0.03)), 0, 1)
                wins = s * c * tp["attach_rate"]
                go_live = total_lag - lag_month
                if go_live >= 0:
                    pipeline.append((int(np.round(go_live)), wins))

        for month in range(projection_months):
            monthly_new = 0
            cal_month = ((start_calendar_month + month) % 12) + 1

            for key, tp in team_params.items():
                gr = growth_rates.get(key, 0.0)
                growth_factor = (1 + gr) ** month
                seasonal = seasonality.get(cal_month, 1.0)
                s = max(rng.normal(tp["sqls_mean"] * growth_factor * seasonal, tp["sqls_std"]), 0)
                c = np.clip(rng.normal(tp["conv_rate"], tp.get("conv_rate_std", 0.03)), 0, 1)
                wins = s * c * tp["attach_rate"]
                sale_m = max(tp["time_to_sale_days"] / 30.0, 0.5)
                impl_m = max(tp["time_to_implement_days"] / 30.0, 0.5)
                pipeline.append((int(np.round(month + sale_m + impl_m)), wins))

            remaining = []
            for go_live, wins in pipeline:
                if go_live <= month:
                    monthly_new += wins
                else:
                    remaining.append((go_live, wins))
            pipeline = remaining

            churn_r = np.clip(rng.normal(churn_rate_mean, churn_rate_std), 0, 0.1)
            ib = max(ib + monthly_new - ib * churn_r, 0)
            results[sim, month] = ib

    alpha = (1 - CONFIDENCE_LEVEL) / 2
    return {
        "months": np.arange(1, projection_months + 1),
        "median": np.median(results, axis=0),
        "p5": np.percentile(results, alpha * 100, axis=0),
        "p95": np.percentile(results, (1 - alpha) * 100, axis=0),
        "mean": np.mean(results, axis=0),
    }


def run_scenario_simulation(starting_install_base, scenario_params, churn_rate_mean, churn_rate_std, growth_rates=None, seasonality=None, start_calendar_month=1, projection_months=PROJECTION_YEARS * 12, n_simulations=MONTE_CARLO_SIMULATIONS):
    """scenario_params is keyed by (team, lead_source) tuples."""
    team_params = {}
    for key, sp in scenario_params.items():
        team_params[key] = {
            "sqls_mean": sp["sqls_per_month"],
            "sqls_std": sp["sqls_per_month"] * 0.15,
            "conv_rate": sp["sql_conversion_rate"] / 100.0,
            "conv_rate_std": sp["sql_conversion_rate"] / 100.0 * 0.1,
            "time_to_sale_days": sp["time_to_sale_days"],
            "time_to_implement_days": sp["time_to_implement_days"],
            "attach_rate": sp.get("vello_attach_rate", 100.0) / 100.0,
        }
    return run_monte_carlo(starting_install_base, team_params, churn_rate_mean, churn_rate_std, projection_months, n_simulations, growth_rates, seasonality, start_calendar_month)


# =============================================================================
# STREAMLIT UI
# =============================================================================
st.set_page_config(page_title="Vello Install Base Simulator", page_icon="📊", layout="wide")
st.title("Vello Install Base Simulator")
st.caption("Strategic planning tool — Monte Carlo projections of install base growth")

with st.spinner("Loading historical data from Snowflake..."):
    data = load_all_data()

ib_df = data["install_base"]
latest_month = ib_df["REPORTING_MONTH"].max()
latest_ib = int(ib_df[ib_df["REPORTING_MONTH"] == latest_month]["VELLO_INSTALL_BASE"].values[0])

st.metric("Current Vello Install Base", f"{latest_ib:,}", f"As of {latest_month.strftime('%B %Y')}")

# Compute baselines
run_rate_params = compute_run_rate_params(data)
defaults = compute_historical_defaults(data)
seasonal_indices = compute_seasonality(data["time_to_booking"])
churn_mean = run_rate_params["churn_rate_mean"]
churn_std = run_rate_params["churn_rate_std"]
baseline_team_params = {k: v for k, v in run_rate_params.items() if k not in ("churn_rate_mean", "churn_rate_std") and isinstance(k, tuple)}
start_cal_month = latest_month.month

projection_months = PROJECTION_YEARS * 12
baseline_result = run_monte_carlo(latest_ib, baseline_team_params, churn_mean, churn_std, projection_months, seasonality=seasonal_indices, start_calendar_month=start_cal_month)
projection_dates = [latest_month + relativedelta(months=i) for i in range(1, projection_months + 1)]

# Session state for scenario
for _key in ("scenario_result", "scenario_name", "scenario_params", "scenario_churn", "scenario_growth_rates"):
    if _key not in st.session_state:
        st.session_state[_key] = None
if st.session_state.scenario_name is None:
    st.session_state.scenario_name = ""


@st.dialog("Scenario Planner", width="large")
def scenario_planner_modal():
    st.markdown("Adjust parameters by team, then click **Run Simulation** to project the scenario.")
    scenario_name = st.text_input("Scenario Name", value="My Scenario", key="modal_scenario_name")

    st.subheader("Global Settings")
    churn_multiplier = st.slider("Churn Multiplier", min_value=0.0, max_value=3.0, value=1.0, step=0.1, key="global_churn_mult", help=f"1.0x = current rate ({churn_mean*100:.2f}%/mo). 0x = no churn. 3x = triple churn.")

    team_tabs = st.tabs(ALL_TEAMS)
    scenario_params = {}
    growth_rates = {}

    for tab, team in zip(team_tabs, ALL_TEAMS):
        with tab:
            d = defaults.get(team, {})
            ls_defaults = d.get("lead_sources", {})
            st.markdown(f"**{team}** — adjust the levers below")

            # Team-level shared inputs
            shared_c1, shared_c2 = st.columns(2)
            with shared_c1:
                tti = st.number_input("Time to Implement (days)", min_value=1, max_value=2000, value=int(d.get("time_to_implement_days", 90)), step=5, key=f"{team}_tti")
            with shared_c2:
                attach = st.slider("Vello Attach Rate (%)", min_value=0.0, max_value=100.0, value=float(d.get("vello_attach_rate", 100.0)), step=1.0, key=f"{team}_attach", help="% of won deals that include Vello")

            st.divider()

            # Per lead-source inputs side by side
            ls_cols = st.columns(len(LEAD_SOURCES))
            total_monthly_wins = 0
            ls_details = []

            for col, ls in zip(ls_cols, LEAD_SOURCES):
                with col:
                    ls_d = ls_defaults.get(ls, {})
                    ls_label = "VDC (IDEXX Referrals)" if ls == "VDC" else "Other Leads"
                    st.markdown(f"**{ls_label}**")
                    sqls = st.number_input("SQLs / Month", min_value=0, max_value=2000, value=int(ls_d.get("sqls_per_month", 0)), step=5, key=f"{team}_{ls}_sqls")
                    sql_conv = st.slider("SQL Win Rate (%)", min_value=0.0, max_value=100.0, value=float(ls_d.get("sql_conversion_rate", 10.0)), step=0.5, key=f"{team}_{ls}_sql_conv")
                    tts = st.number_input("Time to Sale (days)", min_value=1, max_value=730, value=max(int(ls_d.get("time_to_sale_days", 60)), 1), step=5, key=f"{team}_{ls}_tts")
                    ls_growth = st.slider("SQL Growth (%/mo)", min_value=-5.0, max_value=10.0, value=0.0, step=0.1, key=f"{team}_{ls}_growth", help="Compound monthly growth rate for this segment's SQLs")

                    wins = sqls * (sql_conv / 100.0) * (attach / 100.0)
                    total_monthly_wins += wins
                    ls_details.append(f"{ls}: {sqls} SQLs x {sql_conv:.1f}%")

                    scenario_params[(team, ls)] = {
                        "sqls_per_month": sqls,
                        "sql_conversion_rate": sql_conv,
                        "time_to_sale_days": tts,
                        "time_to_implement_days": tti,
                        "vello_attach_rate": attach,
                    }
                    growth_rates[(team, ls)] = ls_growth / 100.0

            st.info(
                f"**Projected monthly Vello placements from {team}:** "
                + " + ".join(ls_details)
                + (f" x {attach:.1f}% attach" if attach < 100 else "")
                + f" = **{total_monthly_wins:.1f}** placements/month"
            )

    st.divider()
    if st.button("Run Simulation", type="primary", use_container_width=True):
        effective_churn = churn_mean * churn_multiplier
        with st.spinner(f"Running {MONTE_CARLO_SIMULATIONS:,} simulations..."):
            result = run_scenario_simulation(latest_ib, scenario_params, effective_churn, churn_std * churn_multiplier, growth_rates, seasonal_indices, start_cal_month, projection_months)
        st.session_state.scenario_result = result
        st.session_state.scenario_name = scenario_name
        st.session_state.scenario_params = scenario_params
        st.session_state.scenario_churn = effective_churn
        st.session_state.scenario_growth_rates = growth_rates
        st.success("Simulation complete! Close this dialog to see results on the chart.")
        st.rerun()


if st.button("Create Scenario", type="primary"):
    scenario_planner_modal()

# =============================================================================
# CHART
# =============================================================================
fig = go.Figure()

fig.add_trace(go.Scatter(x=ib_df["REPORTING_MONTH"], y=ib_df["VELLO_INSTALL_BASE"], mode="lines+markers", name="Actuals", line=dict(color="#1f77b4", width=3), marker=dict(size=4)))

fig.add_trace(go.Scatter(x=projection_dates, y=baseline_result["median"], mode="lines", name="Run Rate (Median)", line=dict(color="#ff7f0e", width=2, dash="dash")))

fig.add_trace(go.Scatter(
    x=projection_dates + projection_dates[::-1],
    y=np.concatenate([baseline_result["p95"], baseline_result["p5"][::-1]]).tolist(),
    fill="toself", fillcolor="rgba(255, 127, 14, 0.15)", line=dict(color="rgba(255, 127, 14, 0)"),
    name="Run Rate 90% CI", showlegend=True,
))

if st.session_state.scenario_result is not None:
    sr = st.session_state.scenario_result
    sname = st.session_state.scenario_name or "Scenario"
    fig.add_trace(go.Scatter(x=projection_dates, y=sr["median"], mode="lines", name=f"{sname} (Median)", line=dict(color="#2ca02c", width=2, dash="dot")))
    fig.add_trace(go.Scatter(
        x=projection_dates + projection_dates[::-1],
        y=np.concatenate([sr["p95"], sr["p5"][::-1]]).tolist(),
        fill="toself", fillcolor="rgba(44, 160, 44, 0.12)", line=dict(color="rgba(44, 160, 44, 0)"),
        name=f"{sname} 90% CI", showlegend=True,
    ))

fig.update_layout(
    title="Vello Install Base — Actuals & Projections", xaxis_title="Month", yaxis_title="Install Base (Sites)",
    height=600, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
)
fig.add_shape(type="line", x0=latest_month.isoformat(), x1=latest_month.isoformat(), y0=0, y1=1, yref="paper", line=dict(color="gray", dash="dash"))
fig.add_annotation(x=latest_month.isoformat(), y=1, yref="paper", text="Current", showarrow=False, yanchor="bottom")
st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# SUMMARY TABLE
# =============================================================================
st.subheader("Projection Summary")
rows = []
for m in [12, 24, 36, 48, 60]:
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

# =============================================================================
# MATH DECOMPOSITION
# =============================================================================
def _extract_row(p):
    """Normalize a param dict to (sqls, conv_rate_decimal, attach_decimal)."""
    sqls = p.get("sqls_mean", p.get("sqls_per_month", 0))
    cr = p.get("conv_rate", p.get("sql_conversion_rate", 0) / 100.0)
    attach = p.get("attach_rate", p.get("vello_attach_rate", 100.0) / 100.0)
    tts = p.get("time_to_sale_days", 60)
    tti = p.get("time_to_implement_days", 90)
    if cr > 1:
        cr = cr / 100.0
    if attach > 1:
        attach = attach / 100.0
    return sqls, cr, attach, tts, tti


def _build_math_breakdown(label, params, churn_rate, growth_rates, starting_ib):
    """Build a markdown string showing the funnel math behind a projection."""
    lines = []
    lines.append(f"### {label}")
    lines.append("")

    lines.append("| Team | Lead Source | SQLs/mo | Win Rate | Attach | Growth | Wins/mo |")
    lines.append("|------|------------|---------|----------|--------|--------|---------|")

    team_subtotals = {}
    total_gross = 0.0
    rows_data = []

    for key, p in sorted(params.items(), key=lambda x: x[0]):
        if not isinstance(key, tuple):
            continue
        team, ls = key
        sqls, cr, attach, tts, tti = _extract_row(p)
        wins = sqls * cr * attach
        total_gross += wins
        team_subtotals[team] = team_subtotals.get(team, 0) + wins
        gr = growth_rates.get(key, 0.0) if growth_rates else 0.0
        rows_data.append({"team": team, "ls": ls, "sqls": sqls, "cr": cr, "attach": attach, "wins": wins, "tts": tts, "tti": tti, "growth": gr})

        gr_str = f"{gr*100:+.1f}%" if gr != 0 else "—"
        lines.append(f"| {team} | {ls} | {sqls:.1f} | {cr*100:.1f}% | {attach*100:.0f}% | {gr_str} | **{wins:.1f}** |")

    lines.append("")
    lines.append("**Team subtotals:** " + " | ".join(f"{t} = {w:.1f}" for t, w in team_subtotals.items()))
    lines.append("")

    monthly_churn_count = starting_ib * churn_rate
    net_monthly = total_gross - monthly_churn_count
    lines.append(f"**Gross adds/month:** {total_gross:.1f} placements")
    lines.append(f"")
    lines.append(f"**Monthly churn:** {starting_ib:,} x {churn_rate*100:.2f}% = {monthly_churn_count:.1f} churns")
    lines.append(f"")
    lines.append(f"**Net monthly change:** {total_gross:.1f} - {monthly_churn_count:.1f} = **{net_monthly:+.1f}** sites/month")
    lines.append(f"")
    lines.append(f"**Annualized net adds (year 1, approx):** ~{net_monthly * 12:,.0f} sites")

    return "\n".join(lines), rows_data, total_gross, monthly_churn_count, net_monthly


def _generate_analysis(run_rate_data, scenario_data, starting_ib, churn_mean, scenario_churn):
    """Generate natural language analysis comparing run rate to scenario."""
    rr_rows, rr_gross, rr_churn_ct, rr_net = run_rate_data
    sc_rows, sc_gross, sc_churn_ct, sc_net = scenario_data

    lines = []
    lines.append("### What's Driving This Scenario")
    lines.append("")

    # 1. Overall direction
    if sc_net > 0 and rr_net <= 0:
        lines.append("This scenario **flips the install base from decline to growth**. "
                      f"The run rate loses ~{abs(rr_net):.0f} sites/month, but the scenario adds ~{sc_net:.0f} sites/month.")
    elif sc_net > rr_net:
        delta = sc_net - rr_net
        lines.append(f"This scenario **accelerates growth by {delta:+.0f} sites/month** vs the run rate.")
    elif sc_net < rr_net:
        delta = sc_net - rr_net
        lines.append(f"This scenario **slows growth by {abs(delta):.0f} sites/month** vs the run rate.")
    else:
        lines.append("This scenario produces roughly the **same trajectory** as the run rate.")
    lines.append("")

    # 2. Break down the biggest movers
    rr_map = {(r["team"], r["ls"]): r for r in rr_rows}
    sc_map = {(r["team"], r["ls"]): r for r in sc_rows}
    all_keys = set(rr_map.keys()) | set(sc_map.keys())

    deltas = []
    for k in all_keys:
        rr_w = rr_map.get(k, {}).get("wins", 0)
        sc_w = sc_map.get(k, {}).get("wins", 0)
        delta_w = sc_w - rr_w
        if abs(delta_w) > 0.05:
            rr_r = rr_map.get(k, {})
            sc_r = sc_map.get(k, {})
            reasons = []
            rr_sqls = rr_r.get("sqls", 0)
            sc_sqls = sc_r.get("sqls", 0)
            if sc_sqls != rr_sqls and rr_sqls > 0:
                pct = (sc_sqls - rr_sqls) / rr_sqls * 100
                reasons.append(f"SQLs {'up' if pct > 0 else 'down'} {abs(pct):.0f}% ({rr_sqls:.0f} -> {sc_sqls:.0f})")
            rr_cr = rr_r.get("cr", 0)
            sc_cr = sc_r.get("cr", 0)
            if abs(sc_cr - rr_cr) > 0.005:
                reasons.append(f"win rate {'up' if sc_cr > rr_cr else 'down'} ({rr_cr*100:.1f}% -> {sc_cr*100:.1f}%)")
            rr_att = rr_r.get("attach", 1)
            sc_att = sc_r.get("attach", 1)
            if abs(sc_att - rr_att) > 0.005:
                reasons.append(f"attach rate {'up' if sc_att > rr_att else 'down'} ({rr_att*100:.0f}% -> {sc_att*100:.0f}%)")
            sc_gr = sc_r.get("growth", 0)
            if sc_gr != 0:
                reasons.append(f"SQL growth {sc_gr*100:+.1f}%/mo")
            deltas.append((k, delta_w, reasons))

    deltas.sort(key=lambda x: abs(x[1]), reverse=True)

    if deltas:
        lines.append("**Key changes vs run rate:**")
        lines.append("")
        for (team, ls), delta_w, reasons in deltas:
            direction = "more" if delta_w > 0 else "fewer"
            reason_str = "; ".join(reasons) if reasons else "parameter changes"
            lines.append(f"- **{team} / {ls}:** {abs(delta_w):.1f} {direction} wins/month — {reason_str}")
        lines.append("")

    # 3. Churn comparison
    if abs(scenario_churn - churn_mean) > 0.0001:
        churn_delta = (scenario_churn - churn_mean) * starting_ib
        mult = scenario_churn / churn_mean if churn_mean > 0 else 1.0
        lines.append(f"**Churn** is {mult:.1f}x the historical rate ({scenario_churn*100:.2f}% vs {churn_mean*100:.2f}%), "
                      f"{'adding' if churn_delta > 0 else 'saving'} ~{abs(churn_delta):.0f} sites/month of churn.")
        lines.append("")

    # 4. Sustainability check
    if sc_net < 0:
        months_to_lose_10pct = int(0.1 * starting_ib / abs(sc_net)) if sc_net != 0 else 0
        lines.append(f"At this pace, the install base would shrink ~10% "
                      f"(lose ~{int(starting_ib * 0.1):,} sites) in roughly **{months_to_lose_10pct} months**.")
    elif sc_net > 0 and sc_gross > 0:
        top = max(sc_rows, key=lambda r: r["wins"])
        top_pct = top["wins"] / sc_gross * 100
        if top_pct > 50:
            lines.append(f"Growth is heavily concentrated: **{top['team']} / {top['ls']}** "
                          f"drives {top_pct:.0f}% of all gross adds ({top['wins']:.1f} of {sc_gross:.1f}/month).")

    return "\n".join(lines)


with st.expander("Growth Math Breakdown", expanded=st.session_state.scenario_result is not None):
    run_rate_col, scenario_col = st.columns(2)

    with run_rate_col:
        rr_md, rr_rows, rr_gross, rr_churn_ct, rr_net = _build_math_breakdown(
            "Run Rate", baseline_team_params, churn_mean, None, latest_ib,
        )
        st.markdown(rr_md)

    with scenario_col:
        if st.session_state.scenario_params is not None:
            sp = st.session_state.scenario_params
            sn = st.session_state.scenario_name or "Scenario"
            sc = st.session_state.scenario_churn or churn_mean
            sg = st.session_state.scenario_growth_rates or {}
            sc_md, sc_rows, sc_gross, sc_churn_ct, sc_net = _build_math_breakdown(
                sn, sp, sc, sg, latest_ib,
            )
            st.markdown(sc_md)
        else:
            st.info("Run a scenario to see its math breakdown here.")

    # Natural language analysis (only when scenario exists)
    if st.session_state.scenario_params is not None:
        st.divider()
        analysis = _generate_analysis(
            (rr_rows, rr_gross, rr_churn_ct, rr_net),
            (sc_rows, sc_gross, sc_churn_ct, sc_net),
            latest_ib, churn_mean,
            st.session_state.scenario_churn or churn_mean,
        )
        st.markdown(analysis)


# =============================================================================
# HISTORICAL DATA EXPLORER
# =============================================================================
with st.expander("Historical Data Explorer"):
    t1, t2, t3, t4, t5 = st.tabs(["Install Base", "Churns", "SQLs Created", "Conversion Rates", "Vello Attach"])
    with t1:
        st.dataframe(ib_df.sort_values("REPORTING_MONTH", ascending=False), use_container_width=True, hide_index=True)
    with t2:
        st.dataframe(data["churns"].sort_values("REPORTING_MONTH", ascending=False), use_container_width=True, hide_index=True)
    with t3:
        st.dataframe(data["sqls_created"].sort_values(["TEAM_NAME", "MONTH"], ascending=[True, False]), use_container_width=True, hide_index=True)
    with t4:
        st.dataframe(data["conversion_rates"].sort_values(["TEAM_NAME", "COHORT_MONTH"], ascending=[True, False]), use_container_width=True, hide_index=True)
    with t5:
        st.dataframe(data["vello_attach"].sort_values(["TEAM_NAME", "CLOSE_MONTH"], ascending=[True, False]), use_container_width=True, hide_index=True)
