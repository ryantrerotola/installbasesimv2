"""Vello Install Base Simulator — single-file Streamlit app."""

import json
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

BAYESIAN_K = 30
MIN_N_ATTACH = 10
CONV_RATE_FLOOR = 0.02
WINSORIZE_LOWER = 0.10
WINSORIZE_UPPER = 0.90

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
    combined["COHORT_MATURITY"] = np.where(
        combined["OPEN_ACTIVE"] > combined["RESOLVED_TOTAL"] * 0.25,
        "Immature",
        "Mature",
    )
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
    """Volume-weighted win rate across the most recent N mature cohort months for a team."""
    subset = df[df["TEAM_NAME"] == team]
    if "COHORT_MATURITY" in subset.columns:
        subset = subset[subset["COHORT_MATURITY"] == "Mature"]
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
    """Volume-weighted win rate across the most recent N mature cohort months for a team + lead source."""
    subset = df[(df["TEAM_NAME"] == team) & (df["LEAD_SOURCE_GROUP"] == lead_source)]
    if "COHORT_MATURITY" in subset.columns:
        subset = subset[subset["COHORT_MATURITY"] == "Mature"]
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
    """Std dev of per-cohort win rates for a team + lead source (mature cohorts only)."""
    subset = df[(df["TEAM_NAME"] == team) & (df["LEAD_SOURCE_GROUP"] == lead_source)]
    if "COHORT_MATURITY" in subset.columns:
        subset = subset[subset["COHORT_MATURITY"] == "Mature"]
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
def _winsorize(series, lower_pct=WINSORIZE_LOWER, upper_pct=WINSORIZE_UPPER):
    if len(series) < 3:
        return series
    lo = series.quantile(lower_pct)
    hi = series.quantile(upper_pct)
    return series.clip(lower=lo, upper=hi)


def _iqr_std(series):
    if len(series) < 3:
        return max(series.std(), 1) if len(series) > 1 else 1.0
    q75 = series.quantile(0.75)
    q25 = series.quantile(0.25)
    robust = (q75 - q25) / 1.35
    return max(robust, 1.0)


def _bayesian_conv_rate(observed_rate, n_obs, prior_rate, k=BAYESIAN_K):
    return (n_obs * observed_rate + k * prior_rate) / (n_obs + k)


def _team_overall_conv(conv_df, team, n_cohorts=6):
    subset = conv_df[conv_df["TEAM_NAME"] == team]
    if "COHORT_MATURITY" in subset.columns:
        subset = subset[subset["COHORT_MATURITY"] == "Mature"]
    if subset.empty:
        return 0.10
    recent = subset["COHORT_MONTH"].drop_duplicates().nlargest(n_cohorts)
    subset = subset[subset["COHORT_MONTH"].isin(recent)]
    total_resolved = subset["RESOLVED_TOTAL"].sum()
    return (subset["WINS"].sum() / total_resolved) if total_resolved > 0 else 0.10


def _robust_attach_rate(attach_df, team, lead_source, n_months=6):
    subset = attach_df[(attach_df["TEAM_NAME"] == team) & (attach_df["LEAD_SOURCE_GROUP"] == lead_source)]
    if subset.empty:
        team_all = attach_df[attach_df["TEAM_NAME"] == team]
        if team_all.empty:
            return 0.30
        recent = team_all["CLOSE_MONTH"].drop_duplicates().nlargest(n_months)
        team_all = team_all[team_all["CLOSE_MONTH"].isin(recent)]
        total = team_all["TOTAL_WON"].sum()
        vello = team_all["VELLO_WON"].sum()
        return vello / total if total > 0 else 0.30
    recent = subset["CLOSE_MONTH"].drop_duplicates().nlargest(n_months)
    subset = subset[subset["CLOSE_MONTH"].isin(recent)]
    total = subset["TOTAL_WON"].sum()
    vello = subset["VELLO_WON"].sum()
    if total < MIN_N_ATTACH:
        team_all = attach_df[attach_df["TEAM_NAME"] == team]
        recent_all = team_all["CLOSE_MONTH"].drop_duplicates().nlargest(n_months)
        team_all = team_all[team_all["CLOSE_MONTH"].isin(recent_all)]
        team_total = team_all["TOTAL_WON"].sum()
        team_vello = team_all["VELLO_WON"].sum()
        team_rate = team_vello / team_total if team_total > 0 else 0.30
        ls_rate = vello / total if total > 0 else team_rate
        blend_weight = total / MIN_N_ATTACH
        return blend_weight * ls_rate + (1 - blend_weight) * team_rate
    return vello / total if total > 0 else 0.30


def _compute_conv_ceiling(conv_df, n_cohorts=6):
    rates = []
    for team in ALL_TEAMS:
        for ls in LEAD_SOURCES:
            r = _safe_team_ls_conv(conv_df, team, ls, n_cohorts) / 100.0
            if r > 0:
                rates.append(r)
    if not rates:
        return 0.90
    return min(np.percentile(rates, 90), 0.95)


def compute_run_rate_params(data):
    recent_months = 6
    params = {}
    dq_flags = []
    sqls = data["sqls_created"]
    conv = data["conversion_rates"]
    attach_df = data["vello_attach"]
    ttb = data["time_to_booking"]
    ttp = data["time_to_placement"]
    ttp_recent = ttp[ttp["MONTH"] >= ttp["MONTH"].max() - pd.DateOffset(months=recent_months)]

    for team in ALL_TEAMS:
        team_ttp = ttp_recent[ttp_recent["TEAM_NAME"] == team] if "TEAM_NAME" in ttp_recent.columns else ttp_recent
        ttp_impl_days = team_ttp["MEDIAN_DAYS_TO_PLACEMENT"].mean() if not team_ttp.empty else 90

        for ls in LEAD_SOURCES:
            ls_sqls = sqls[(sqls["TEAM_NAME"] == team) & (sqls["LEAD_SOURCE_GROUP"] == ls)]
            ls_recent = ls_sqls[ls_sqls["MONTH"] >= ls_sqls["MONTH"].max() - pd.DateOffset(months=recent_months)] if not ls_sqls.empty else ls_sqls
            monthly = ls_recent.groupby("MONTH")["SQLS_CREATED"].sum() if not ls_recent.empty else pd.Series(dtype=float)

            if not monthly.empty and len(monthly) >= 3:
                raw_mean = monthly.mean()
                winsorized = _winsorize(monthly)
                sqls_mean = winsorized.mean()
                sqls_std = _iqr_std(winsorized)
                if abs(sqls_mean - raw_mean) / max(raw_mean, 1) > 0.10:
                    dq_flags.append({"stream": f"{team} / {ls}", "type": "SQLs Winsorized", "detail": f"Raw mean {raw_mean:.0f} -> adjusted {sqls_mean:.0f} (outliers capped at p10/p90)"})
            elif not monthly.empty:
                sqls_mean = monthly.mean()
                sqls_std = max(monthly.std(), 1) if len(monthly) > 1 else 1
                if len(monthly) <= 2:
                    dq_flags.append({"stream": f"{team} / {ls}", "type": "Low N (SQLs)", "detail": f"Only {len(monthly)} months of data"})
            else:
                sqls_mean = 0
                sqls_std = 1

            cr = _safe_team_ls_conv(conv, team, ls, n_cohorts=recent_months) / 100.0
            cr_std = _cohort_conv_std(conv, team, ls, n_cohorts=recent_months)

            subset = conv[(conv["TEAM_NAME"] == team) & (conv["LEAD_SOURCE_GROUP"] == ls)]
            if "COHORT_MATURITY" in subset.columns:
                subset = subset[subset["COHORT_MATURITY"] == "Mature"]
            recent_cohorts = subset["COHORT_MONTH"].drop_duplicates().nlargest(recent_months) if not subset.empty else pd.Series(dtype="datetime64[ns]")
            n_resolved = subset[subset["COHORT_MONTH"].isin(recent_cohorts)]["RESOLVED_TOTAL"].sum() if not subset.empty else 0

            if n_resolved < BAYESIAN_K:
                team_prior = _team_overall_conv(conv, team, n_cohorts=recent_months)
                adjusted_cr = _bayesian_conv_rate(cr, n_resolved, team_prior, k=BAYESIAN_K)
                if abs(adjusted_cr - cr) > 0.02:
                    dq_flags.append({"stream": f"{team} / {ls}", "type": "Conv Rate Adjusted (low N)", "detail": f"Raw {cr*100:.1f}% -> Bayesian {adjusted_cr*100:.1f}% (N={n_resolved}, prior={team_prior*100:.1f}%)"})
                    cr = adjusted_cr

            cr = max(cr, CONV_RATE_FLOOR)

            ls_ttb = ttb[(ttb["TEAM_NAME"] == team) & (ttb["LEAD_SOURCE_GROUP"] == ls) & (ttb["MONTH"] >= ttb["MONTH"].max() - pd.DateOffset(months=recent_months))]

            if team == "ISAS":
                ls_attach = 1.0
            else:
                raw_attach = _team_ls_attach(attach_df, team, ls, recent_months) / 100.0
                attach_subset = attach_df[(attach_df["TEAM_NAME"] == team) & (attach_df["LEAD_SOURCE_GROUP"] == ls)]
                recent_attach = attach_subset[attach_subset["CLOSE_MONTH"].isin(attach_subset["CLOSE_MONTH"].drop_duplicates().nlargest(recent_months))] if not attach_subset.empty else attach_subset
                attach_n = recent_attach["TOTAL_WON"].sum() if not recent_attach.empty else 0

                if attach_n < MIN_N_ATTACH:
                    ls_attach = _robust_attach_rate(attach_df, team, ls, recent_months)
                    if abs(ls_attach - raw_attach) > 0.02:
                        dq_flags.append({"stream": f"{team} / {ls}", "type": "Attach Rate Adjusted (low N)", "detail": f"Raw {raw_attach*100:.1f}% -> blended {ls_attach*100:.1f}% (N={attach_n}, min={MIN_N_ATTACH})"})
                else:
                    ls_attach = raw_attach

            params[(team, ls)] = {
                "sqls_mean": sqls_mean,
                "sqls_std": sqls_std,
                "conv_rate": cr,
                "conv_rate_std": cr_std,
                "time_to_sale_days": ls_ttb["MEDIAN_DAYS_TO_BOOKING"].mean() if not ls_ttb.empty else 60,
                "time_to_implement_days": ttp_impl_days,
                "attach_rate": ls_attach,
            }

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

    params["_dq_flags"] = dq_flags
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
        "raw": results,
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
baseline_team_params = {k: v for k, v in run_rate_params.items() if k not in ("churn_rate_mean", "churn_rate_std", "_dq_flags") and isinstance(k, tuple)}
dq_flags = run_rate_params.get("_dq_flags", [])
start_cal_month = latest_month.month

projection_months = PROJECTION_YEARS * 12
projection_dates = [latest_month + relativedelta(months=i) for i in range(1, projection_months + 1)]

# Cache baseline simulation in session state to avoid rerunning on every widget change
_baseline_cache_key = json.dumps({
    "ib": latest_ib, "churn": churn_mean, "churn_std": churn_std,
    "params": {str(k): v for k, v in baseline_team_params.items()},
    "seasonal": seasonal_indices, "cal_month": start_cal_month,
}, sort_keys=True)
if st.session_state.get("_baseline_cache_key") != _baseline_cache_key:
    baseline_result = run_monte_carlo(latest_ib, baseline_team_params, churn_mean, churn_std, projection_months, seasonality=seasonal_indices, start_calendar_month=start_cal_month)
    st.session_state._baseline_result = baseline_result
    st.session_state._baseline_cache_key = _baseline_cache_key
else:
    baseline_result = st.session_state._baseline_result

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
        st.session_state._scenario_just_ran = True
        st.rerun()


if st.button("Create Scenario", type="primary"):
    scenario_planner_modal()

# If scenario just ran inside dialog, force full app rerun to show results
if st.session_state.get("_scenario_just_ran"):
    del st.session_state._scenario_just_ran
    st.rerun()

tab_projections, tab_distribution, tab_goalsek = st.tabs(["Projections", "Monte Carlo Distribution", "Goal Seek"])

# =============================================================================
# TAB 1 — PROJECTIONS
# =============================================================================
with tab_projections:
    # CHART
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

    # SUMMARY TABLE
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
# TAB 2 — MONTE CARLO DISTRIBUTION
# =============================================================================
with tab_distribution:
    st.subheader("Simulation Distribution Analysis")
    st.caption(f"Explore the distribution of {MONTE_CARLO_SIMULATIONS:,} simulation outcomes at any point in the projection.")

    # Date selector — slider mapped to projection months
    dist_month_idx = st.slider(
        "Select projection month",
        min_value=1,
        max_value=projection_months,
        value=12,
        step=1,
        key="dist_month_slider",
        format="Month %d",
    )
    selected_date = latest_month + relativedelta(months=dist_month_idx)
    st.markdown(f"**Showing distributions for: {selected_date.strftime('%B %Y')}** (month {dist_month_idx} of {projection_months})")

    # Extract raw simulation values at selected month
    rr_sims = baseline_result["raw"][:, dist_month_idx - 1]
    has_scenario = st.session_state.scenario_result is not None
    sc_sims = st.session_state.scenario_result["raw"][:, dist_month_idx - 1] if has_scenario else None
    sc_name = st.session_state.scenario_name or "Scenario"

    # --- Histogram ---
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(
        x=rr_sims,
        name="Run Rate",
        marker_color="rgba(255, 127, 14, 0.6)",
        nbinsx=50,
    ))
    if has_scenario:
        hist_fig.add_trace(go.Histogram(
            x=sc_sims,
            name=sc_name,
            marker_color="rgba(44, 160, 44, 0.6)",
            nbinsx=50,
        ))
    hist_fig.update_layout(
        title=f"Install Base Distribution — {selected_date.strftime('%B %Y')}",
        xaxis_title="Install Base (Sites)",
        yaxis_title="Number of Simulations",
        barmode="overlay",
        height=450,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    # --- Summary Statistics Table ---
    st.subheader("Summary Statistics")
    percentiles = [5, 10, 25, 50, 75, 90, 95]

    def _build_stats(sims, label):
        stats = {"": label, "Mean": f"{np.mean(sims):,.0f}", "Std Dev": f"{np.std(sims):,.0f}"}
        for p in percentiles:
            stats[f"P{p}"] = f"{np.percentile(sims, p):,.0f}"
        return stats

    stats_rows = [_build_stats(rr_sims, "Run Rate")]
    if has_scenario:
        stats_rows.append(_build_stats(sc_sims, sc_name))
    st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

    # --- P(Scenario beats Run Rate) ---
    if has_scenario:
        pct_scenario_wins = (sc_sims > rr_sims).mean() * 100
        m1, m2, m3 = st.columns(3)
        m1.metric("P(Scenario > Run Rate)", f"{pct_scenario_wins:.1f}%")
        median_diff = np.median(sc_sims) - np.median(rr_sims)
        m2.metric("Median Difference", f"{median_diff:+,.0f} sites")
        mean_diff = np.mean(sc_sims) - np.mean(rr_sims)
        m3.metric("Mean Difference", f"{mean_diff:+,.0f} sites")

    # --- Box Plot ---
    st.subheader("Distribution Comparison")
    box_fig = go.Figure()
    box_fig.add_trace(go.Box(y=rr_sims, name="Run Rate", marker_color="#ff7f0e", boxmean="sd"))
    if has_scenario:
        box_fig.add_trace(go.Box(y=sc_sims, name=sc_name, marker_color="#2ca02c", boxmean="sd"))
    box_fig.update_layout(
        title=f"Box Plot — {selected_date.strftime('%B %Y')}",
        yaxis_title="Install Base (Sites)",
        height=400,
        template="plotly_white",
    )
    st.plotly_chart(box_fig, use_container_width=True)

# =============================================================================
# GOAL SEEK ENGINE
# =============================================================================
def _compute_sensitivity(baseline_params, churn_mean, churn_std, starting_ib, target_month,
                         seasonal_indices, start_cal_month, n_sims=200):
    """Compute marginal impact of each lever: how many sites does a 1% change add at target_month?"""
    base_result = run_monte_carlo(starting_ib, baseline_params, churn_mean, churn_std,
                                  target_month, n_sims, seasonality=seasonal_indices,
                                  start_calendar_month=start_cal_month)
    base_median = np.median(base_result["raw"][:, -1])

    sensitivities = []

    # Test each segment's levers
    for key, p in baseline_params.items():
        if not isinstance(key, tuple):
            continue
        team, ls = key

        # SQLs: +1% increase
        tweaked = {k: dict(v) if isinstance(v, dict) else v for k, v in baseline_params.items()}
        tweaked[key] = dict(p)
        tweaked[key]["sqls_mean"] = p["sqls_mean"] * 1.01
        tweaked[key]["sqls_std"] = p["sqls_std"] * 1.01
        r = run_monte_carlo(starting_ib, tweaked, churn_mean, churn_std, target_month, n_sims,
                            seasonality=seasonal_indices, start_calendar_month=start_cal_month)
        sql_impact = np.median(r["raw"][:, -1]) - base_median
        sensitivities.append({
            "lever": "SQLs/month", "team": team, "ls": ls,
            "current": p["sqls_mean"], "unit": "leads",
            "impact_per_pct": sql_impact, "key": key, "param": "sqls",
        })

        # Win rate: +1pp
        tweaked = {k: dict(v) if isinstance(v, dict) else v for k, v in baseline_params.items()}
        tweaked[key] = dict(p)
        tweaked[key]["conv_rate"] = p["conv_rate"] + 0.01
        r = run_monte_carlo(starting_ib, tweaked, churn_mean, churn_std, target_month, n_sims,
                            seasonality=seasonal_indices, start_calendar_month=start_cal_month)
        cr_impact = np.median(r["raw"][:, -1]) - base_median
        sensitivities.append({
            "lever": "Win Rate", "team": team, "ls": ls,
            "current": p["conv_rate"] * 100, "unit": "pp",
            "impact_per_pct": cr_impact, "key": key, "param": "conv_rate",
        })

        # Attach rate: +1pp (skip ISAS, always 100%)
        if team != "ISAS":
            tweaked = {k: dict(v) if isinstance(v, dict) else v for k, v in baseline_params.items()}
            tweaked[key] = dict(p)
            tweaked[key]["attach_rate"] = min(p["attach_rate"] + 0.01, 1.0)
            r = run_monte_carlo(starting_ib, tweaked, churn_mean, churn_std, target_month, n_sims,
                                seasonality=seasonal_indices, start_calendar_month=start_cal_month)
            att_impact = np.median(r["raw"][:, -1]) - base_median
            sensitivities.append({
                "lever": "Attach Rate", "team": team, "ls": ls,
                "current": p["attach_rate"] * 100, "unit": "pp",
                "impact_per_pct": att_impact, "key": key, "param": "attach_rate",
            })

    # Churn: -1pp (reducing churn = good)
    r = run_monte_carlo(starting_ib, baseline_params, churn_mean - 0.0001, churn_std, target_month, n_sims,
                        seasonality=seasonal_indices, start_calendar_month=start_cal_month)
    churn_impact = np.median(r["raw"][:, -1]) - base_median  # impact of -0.01pp
    sensitivities.append({
        "lever": "Churn Rate", "team": "ALL", "ls": "ALL",
        "current": churn_mean * 100, "unit": "pp reduction",
        "impact_per_pct": churn_impact * 100,  # scale to per-1pp
        "key": "churn", "param": "churn",
    })

    return base_median, sensitivities


def _goal_seek(baseline_params, churn_mean, churn_std, starting_ib, target_ib,
               target_month, seasonal_indices, start_cal_month, locked_levers=None,
               max_iterations=20, n_sims=200):
    """
    Find the path-of-least-resistance parameter changes to hit target_ib at target_month.

    Uses iterative gradient-based optimization with quadratic cost penalty:
    - Each lever's "cost" grows quadratically with % change (small changes are cheap, big ones expensive)
    - Distributes effort across many levers rather than concentrating on one
    - Respects locked_levers (set of lever identifiers to skip)
    """
    if locked_levers is None:
        locked_levers = set()

    # Get baseline and sensitivities
    base_median, sensitivities = _compute_sensitivity(
        baseline_params, churn_mean, churn_std, starting_ib, target_month,
        seasonal_indices, start_cal_month, n_sims,
    )

    gap = target_ib - base_median
    if abs(gap) < 1:
        return base_median, [], sensitivities, "on_target"

    # Filter out locked levers and zero-impact levers
    active = [s for s in sensitivities
              if (s["team"], s["ls"], s["lever"]) not in locked_levers
              and abs(s["impact_per_pct"]) > 0.01]

    if not active:
        return base_median, [], sensitivities, "no_levers"

    # Per-lever caps — max change the optimizer can recommend per lever.
    # Generous defaults that prevent absurd results (e.g. churn → 0%)
    # while still allowing meaningful optimization.
    MAX_CHANGE = {
        "sqls": 50.0,        # ±50% change in SQLs
        "conv_rate": 15.0,   # ±15pp change in win rate
        "attach_rate": 25.0, # ±25pp change in attach rate
        "churn": 0.75,       # ±0.75pp change in churn (~50% of typical rate)
    }

    total_sensitivity = sum(abs(s["impact_per_pct"]) for s in active)
    if total_sensitivity == 0:
        return base_median, [], sensitivities, "no_impact"

    recommendations = []
    remaining_gap = gap
    allocated = {id(s): 0.0 for s in active}

    for iteration in range(max_iterations):
        if abs(remaining_gap) < 1:
            break

        # Weight by sensitivity / (1 + normalized_usage^2) — quadratic penalty
        # normalized_usage = how much of this lever's budget we've used (0-1)
        weights = []
        for s in active:
            max_chg = MAX_CHANGE.get(s["param"], 10.0)
            usage = abs(allocated[id(s)]) / max_chg  # 0..1 fraction of budget used
            if usage >= 1.0:
                continue  # lever is maxed out
            penalty = 1.0 + (usage ** 2) * 8.0  # penalty as lever approaches cap
            direction = 1.0 if (remaining_gap > 0) == (s["impact_per_pct"] > 0) else -1.0
            effective_weight = abs(s["impact_per_pct"]) / penalty
            weights.append((s, effective_weight, direction, max_chg))

        total_weight = sum(w for _, w, _, _ in weights)
        if total_weight == 0:
            break

        # Allocate a step — constant fraction of remaining gap per iteration
        step_fraction = 0.4
        step_gap = remaining_gap * step_fraction

        for s, w, direction, max_chg in weights:
            share = (w / total_weight) * abs(step_gap)
            pct_change = share / abs(s["impact_per_pct"]) if abs(s["impact_per_pct"]) > 0.01 else 0
            # Clamp to remaining headroom for this lever
            headroom = max_chg - abs(allocated[id(s)])
            pct_change = min(pct_change, max(headroom, 0))
            allocated[id(s)] += pct_change * direction

        # Re-estimate remaining gap
        filled = sum(allocated[id(s)] * s["impact_per_pct"] for s in active)
        remaining_gap = gap - filled

    # Build recommendations from allocations
    for s in active:
        change = allocated[id(s)]
        if abs(change) < 0.01:
            continue

        if s["param"] == "sqls":
            new_val = s["current"] * (1 + change / 100.0)
            pct_change = change
            rec = {
                "team": s["team"], "ls": s["ls"], "lever": s["lever"],
                "current": round(s["current"], 1), "recommended": round(max(new_val, 0), 1),
                "change": f"{pct_change:+.1f}%", "change_raw": abs(pct_change),
                "impact": round(abs(change * s["impact_per_pct"]), 0),
                "unit": "leads/mo",
            }
        elif s["param"] == "conv_rate":
            new_val = s["current"] + change
            rec = {
                "team": s["team"], "ls": s["ls"], "lever": s["lever"],
                "current": round(s["current"], 1), "recommended": round(min(max(new_val, 0), 100), 1),
                "change": f"{change:+.1f}pp", "change_raw": abs(change),
                "impact": round(abs(change * s["impact_per_pct"]), 0),
                "unit": "%",
            }
        elif s["param"] == "attach_rate":
            new_val = s["current"] + change
            rec = {
                "team": s["team"], "ls": s["ls"], "lever": s["lever"],
                "current": round(s["current"], 1), "recommended": round(min(max(new_val, 0), 100), 1),
                "change": f"{change:+.1f}pp", "change_raw": abs(change),
                "impact": round(abs(change * s["impact_per_pct"]), 0),
                "unit": "%",
            }
        elif s["param"] == "churn":
            # positive allocation = reduce churn (since impact_per_pct measures benefit of reduction)
            new_val = s["current"] - change
            rec = {
                "team": s["team"], "ls": s["ls"], "lever": s["lever"],
                "current": round(s["current"], 3), "recommended": round(max(new_val, 0), 3),
                "change": f"{-change:+.3f}pp", "change_raw": abs(change),
                "impact": round(abs(change * s["impact_per_pct"]), 0),
                "unit": "%",
            }
        else:
            continue

        recommendations.append(rec)

    # Sort by change magnitude (smallest first — path of least resistance)
    recommendations.sort(key=lambda r: r["change_raw"])

    projected = base_median + sum(allocated[id(s)] * s["impact_per_pct"] for s in active)

    # Determine feasibility
    if abs(projected - target_ib) / max(target_ib, 1) < 0.02:
        status = "achievable"
    elif abs(projected - target_ib) / max(target_ib, 1) < 0.10:
        status = "stretch"
    else:
        status = "difficult"

    return projected, recommendations, sensitivities, status


def _build_goalsek_params(recommendations, baseline_params, churn_mean):
    """Build modified params dict from goal seek recommendations for validation simulation."""
    params = {k: dict(v) if isinstance(v, dict) else v for k, v in baseline_params.items()}
    new_churn = churn_mean

    for rec in recommendations:
        if rec["lever"] == "Churn Rate":
            new_churn = rec["recommended"] / 100.0
            continue

        key = (rec["team"], rec["ls"])
        if key not in params:
            continue
        p = dict(params[key])

        if rec["lever"] == "SQLs/month":
            p["sqls_mean"] = rec["recommended"]
            p["sqls_std"] = rec["recommended"] * 0.15
        elif rec["lever"] == "Win Rate":
            p["conv_rate"] = rec["recommended"] / 100.0
        elif rec["lever"] == "Attach Rate":
            p["attach_rate"] = rec["recommended"] / 100.0

        params[key] = p

    return params, new_churn


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


# --- Growth Math Breakdown (inside Projections tab context) ---
with tab_projections:
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

    if dq_flags:
        with st.expander(f"Data Quality Flags ({len(dq_flags)} adjustments)", expanded=True):
            st.caption("Automatic adjustments applied to handle outliers and low-sample-size issues.")
            flag_rows = [{"Stream": f["stream"], "Adjustment": f["type"], "Details": f["detail"]} for f in dq_flags]
            st.dataframe(pd.DataFrame(flag_rows), use_container_width=True, hide_index=True)
            st.markdown(
                "**Methods:** "
                f"Winsorization at p{int(WINSORIZE_LOWER*100)}/p{int(WINSORIZE_UPPER*100)} for SQL volumes, "
                f"Bayesian shrinkage (k={BAYESIAN_K}) for conversion rates, "
                f"Min-N threshold ({MIN_N_ATTACH} deals) for attach rates, "
                f"Floor at {CONV_RATE_FLOOR*100:.0f}%/ceiling at 90th pctile, "
                "IQR-based robust std dev for Monte Carlo"
            )

    with st.expander("Historical Data Explorer"):
        ht1, ht2, ht3, ht4, ht5 = st.tabs(["Install Base", "Churns", "SQLs Created", "Conversion Rates", "Vello Attach"])
        with ht1:
            st.dataframe(ib_df.sort_values("REPORTING_MONTH", ascending=False), use_container_width=True, hide_index=True)
        with ht2:
            st.dataframe(data["churns"].sort_values("REPORTING_MONTH", ascending=False), use_container_width=True, hide_index=True)
        with ht3:
            st.dataframe(data["sqls_created"].sort_values(["TEAM_NAME", "MONTH"], ascending=[True, False]), use_container_width=True, hide_index=True)
        with ht4:
            st.caption("Immature cohorts (>25% of opps still open) are shown but excluded from run rate calculations.")
            st.dataframe(data["conversion_rates"].sort_values(["TEAM_NAME", "COHORT_MONTH"], ascending=[True, False]), use_container_width=True, hide_index=True)
        with ht5:
            st.dataframe(data["vello_attach"].sort_values(["TEAM_NAME", "CLOSE_MONTH"], ascending=[True, False]), use_container_width=True, hide_index=True)

# =============================================================================
# TAB 3 — GOAL SEEK
# =============================================================================
with tab_goalsek:
    st.subheader("Goal Seek — Path of Least Resistance")
    st.caption(
        "Set a target install base and date. The optimizer finds the smallest, "
        "most distributed changes across all levers to hit your goal."
    )

    # --- Inputs ---
    gs_col1, gs_col2 = st.columns(2)
    with gs_col1:
        gs_target = st.number_input(
            "Target Install Base",
            min_value=latest_ib,
            max_value=50000,
            value=latest_ib + 500,
            step=50,
            key="gs_target",
        )
    with gs_col2:
        gs_month_options = {
            (latest_month + relativedelta(months=m)).strftime("%b %Y"): m
            for m in range(1, projection_months + 1)
        }
        gs_date_label = st.selectbox("Achieve by", options=list(gs_month_options.keys()), index=11, key="gs_date")
        gs_target_month = gs_month_options[gs_date_label]

    # --- Constraints: lock levers ---
    with st.expander("Constraints — lock levers you can't change"):
        st.caption("Check any levers you want the optimizer to leave alone.")
        locked = set()
        constraint_cols = st.columns(len(ALL_TEAMS) + 1)

        for ci, team in enumerate(ALL_TEAMS):
            with constraint_cols[ci]:
                st.markdown(f"**{team}**")
                for ls in LEAD_SOURCES:
                    if st.checkbox(f"Lock SQLs ({ls})", key=f"lock_{team}_{ls}_sqls"):
                        locked.add((team, ls, "SQLs/month"))
                    if st.checkbox(f"Lock Win Rate ({ls})", key=f"lock_{team}_{ls}_cr"):
                        locked.add((team, ls, "Win Rate"))
                    if team != "ISAS":
                        if st.checkbox(f"Lock Attach ({ls})", key=f"lock_{team}_{ls}_att"):
                            locked.add((team, ls, "Attach Rate"))

        with constraint_cols[-1]:
            st.markdown("**Global**")
            if st.checkbox("Lock Churn Rate", key="lock_churn"):
                locked.add(("ALL", "ALL", "Churn Rate"))

    # --- Run optimizer ---
    if st.button("Find Optimal Path", type="primary", use_container_width=True, key="gs_run"):
        with st.spinner(f"Running optimizer ({MONTE_CARLO_SIMULATIONS:,} sims per iteration)..."):
            projected, recommendations, sensitivities, status = _goal_seek(
                baseline_team_params, churn_mean, churn_std, starting_ib=latest_ib,
                target_ib=gs_target, target_month=gs_target_month,
                seasonal_indices=seasonal_indices, start_cal_month=start_cal_month,
                locked_levers=locked, max_iterations=30, n_sims=200,
            )
            # Run validation simulation now (not on every rerun)
            val_result = None
            if recommendations:
                rec_params, rec_churn = _build_goalsek_params(recommendations, baseline_team_params, churn_mean)
                val_result = run_monte_carlo(
                    latest_ib, rec_params, rec_churn, churn_std,
                    projection_months, MONTE_CARLO_SIMULATIONS,
                    seasonality=seasonal_indices, start_calendar_month=start_cal_month,
                )
        st.session_state.gs_result = {
            "projected": projected,
            "recommendations": recommendations,
            "sensitivities": sensitivities,
            "status": status,
            "target": gs_target,
            "target_month": gs_target_month,
            "date_label": gs_date_label,
            "val_result": val_result,
        }

    # --- Display results ---
    if "gs_result" not in st.session_state:
        st.session_state.gs_result = None

    if st.session_state.gs_result is not None:
        gsr = st.session_state.gs_result
        projected = gsr["projected"]
        recommendations = gsr["recommendations"]
        sensitivities = gsr["sensitivities"]
        status = gsr["status"]
        target = gsr["target"]
        target_month = gsr["target_month"]

        # Feasibility verdict
        st.divider()
        gap = target - np.median(baseline_result["raw"][:, min(target_month, projection_months) - 1])
        run_rate_at_target = np.median(baseline_result["raw"][:, min(target_month, projection_months) - 1])

        if status == "on_target":
            st.success(f"You're already on track! Run rate projects **{run_rate_at_target:,.0f}** sites by {gsr['date_label']}.")
        elif status == "achievable":
            st.success(
                f"**Achievable.** With small, distributed changes you can reach **{target:,}** sites by {gsr['date_label']}. "
                f"Run rate alone would reach {run_rate_at_target:,.0f}. Gap: **{gap:+,.0f}** sites."
            )
        elif status == "stretch":
            st.warning(
                f"**Stretch goal.** The optimizer gets to ~{projected:,.0f} sites (target: {target:,}). "
                f"Run rate alone: {run_rate_at_target:,.0f}. You'll need aggressive execution across multiple levers."
            )
        else:
            st.error(
                f"**Very difficult.** Even spreading effort across all levers, the optimizer reaches ~{projected:,.0f} "
                f"(target: {target:,}). Consider extending the timeline or revising the target."
            )

        # Gap analysis
        st.subheader("Gap Analysis")
        gap_c1, gap_c2, gap_c3 = st.columns(3)
        gap_c1.metric("Run Rate Projection", f"{run_rate_at_target:,.0f}", help=f"Median at {gsr['date_label']}")
        gap_c2.metric("Target", f"{target:,}")
        gap_c3.metric("Gap to Close", f"{gap:+,.0f} sites")

        net_monthly_rr = gap / max(target_month, 1)
        st.markdown(f"You need **{net_monthly_rr:+,.1f} additional net sites/month** beyond run rate to close this gap over {target_month} months.")

        # Recommendations table
        if recommendations:
            st.subheader("Recommended Changes")
            st.caption("Sorted by size of change (smallest first) — path of least resistance.")

            rec_rows = []
            for rec in recommendations:
                label = f"{rec['team']} / {rec['ls']}" if rec['team'] != 'ALL' else "All Teams"
                rec_rows.append({
                    "Segment": label,
                    "Lever": rec["lever"],
                    "Current": f"{rec['current']:.1f}{rec['unit']}",
                    "Recommended": f"{rec['recommended']:.1f}{rec['unit']}",
                    "Change": rec["change"],
                    "Est. Impact": f"+{rec['impact']:.0f} sites",
                })
            st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)

            # Visual: change magnitudes
            change_fig = go.Figure()
            labels = [f"{r['Segment']}\n{r['Lever']}" for r in rec_rows]
            changes = [r["change_raw"] for r in recommendations]
            colors = ["#2ca02c" if c < 3 else "#ff7f0e" if c < 8 else "#d62728" for c in changes]
            change_fig.add_trace(go.Bar(
                x=labels, y=changes, marker_color=colors,
                text=[r["change"] for r in recommendations], textposition="outside",
            ))
            change_fig.update_layout(
                title="Change Magnitude by Lever (smaller = easier)",
                yaxis_title="Magnitude of Change",
                height=350, template="plotly_white",
                showlegend=False,
            )
            st.plotly_chart(change_fig, use_container_width=True)
        else:
            st.info("No parameter changes needed — you're on target or all levers are locked.")

        # Sensitivity ranking
        st.subheader("Sensitivity Ranking")
        st.caption("Which levers move the needle the most? Impact of a 1% or 1pp change.")
        sens_rows = []
        for s in sorted(sensitivities, key=lambda x: abs(x["impact_per_pct"]), reverse=True):
            label = f"{s['team']} / {s['ls']}" if s['team'] != 'ALL' else "All Teams"
            sens_rows.append({
                "Segment": label,
                "Lever": s["lever"],
                "Current Value": f"{s['current']:.1f}",
                "Impact per 1% / 1pp": f"{s['impact_per_pct']:+.1f} sites",
            })
        st.dataframe(pd.DataFrame(sens_rows), use_container_width=True, hide_index=True)

        # Validation simulation — use cached result from button click
        val_result = gsr.get("val_result")
        if recommendations and val_result is not None:
            st.subheader("Validation — Recommended vs Run Rate")

            val_fig = go.Figure()
            val_fig.add_trace(go.Scatter(
                x=projection_dates, y=baseline_result["median"], mode="lines",
                name="Run Rate (Median)", line=dict(color="#ff7f0e", width=2, dash="dash"),
            ))
            val_fig.add_trace(go.Scatter(
                x=projection_dates + projection_dates[::-1],
                y=np.concatenate([baseline_result["p95"], baseline_result["p5"][::-1]]).tolist(),
                fill="toself", fillcolor="rgba(255, 127, 14, 0.10)",
                line=dict(color="rgba(255, 127, 14, 0)"), name="Run Rate 90% CI", showlegend=True,
            ))
            val_fig.add_trace(go.Scatter(
                x=projection_dates, y=val_result["median"], mode="lines",
                name="Goal Seek (Median)", line=dict(color="#9467bd", width=2),
            ))
            val_fig.add_trace(go.Scatter(
                x=projection_dates + projection_dates[::-1],
                y=np.concatenate([val_result["p95"], val_result["p5"][::-1]]).tolist(),
                fill="toself", fillcolor="rgba(148, 103, 189, 0.12)",
                line=dict(color="rgba(148, 103, 189, 0)"), name="Goal Seek 90% CI", showlegend=True,
            ))

            # Target line
            target_date = latest_month + relativedelta(months=target_month)
            val_fig.add_shape(
                type="line", x0=projection_dates[0].isoformat(), x1=projection_dates[-1].isoformat(),
                y0=target, y1=target, line=dict(color="red", dash="dot", width=1),
            )
            val_fig.add_annotation(
                x=target_date.isoformat(), y=target, text=f"Target: {target:,}",
                showarrow=True, arrowhead=2, ax=40, ay=-30,
                font=dict(color="red"),
            )
            val_fig.update_layout(
                title="Goal Seek Projection vs Run Rate",
                xaxis_title="Month", yaxis_title="Install Base (Sites)",
                height=500, template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
            )
            st.plotly_chart(val_fig, use_container_width=True)

            # P(hitting target) metric
            val_at_target = val_result["raw"][:, min(target_month, projection_months) - 1]
            pct_hit = (val_at_target >= target).mean() * 100
            rr_at_target = baseline_result["raw"][:, min(target_month, projection_months) - 1]
            rr_pct_hit = (rr_at_target >= target).mean() * 100
            pm1, pm2 = st.columns(2)
            pm1.metric("P(Run Rate hits target)", f"{rr_pct_hit:.0f}%")
            pm2.metric("P(Goal Seek hits target)", f"{pct_hit:.0f}%")
