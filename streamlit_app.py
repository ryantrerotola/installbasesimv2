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
WITH months AS (
    SELECT DATEADD('MONTH', SEQ4(), '2020-01-01')::DATE AS REPORTING_MONTH
    FROM TABLE(GENERATOR(ROWCOUNT => 200))
    WHERE DATEADD('MONTH', SEQ4(), '2020-01-01') <= DATE_TRUNC('MONTH', CURRENT_DATE())
),
opps AS (
    SELECT
        TEAM_NAME,
        CASE WHEN LEADSOURCE = 'IDEXX REFERRAL' THEN 'VDC' ELSE 'Other Leads' END AS LEAD_SOURCE_GROUP,
        OPPORTUNITY_ID,
        CURRENT_STAGE,
        OPPORTUNITY_CREATED_DATE,
        ACTUAL_CLOSE_DATE
    FROM VSSANALYTICS_DB.SFDC.CDL_SALESFORCE
    WHERE (
          (TEAM_NAME = 'ISAS' AND PRODUCT = 'Vello')
          OR (TEAM_NAME IN ('SOFTWARE SALES EAST', 'SOFTWARE SALES WEST')
              AND PRODUCT IN ('ezyVet', 'Neo', 'Vello'))
          OR (TEAM_NAME = 'ESAM'
              AND PRODUCT IN ('ezyVet', 'ezyVet Enterprise - GP', 'ezyVet Enterprise - Spec/ER'))
      )
),
monthly_conv AS (
    SELECT
        m.REPORTING_MONTH,
        o.TEAM_NAME,
        o.LEAD_SOURCE_GROUP,
        COUNT(DISTINCT CASE WHEN o.CURRENT_STAGE = 'WON' THEN o.OPPORTUNITY_ID END)        AS WINS,
        COUNT(DISTINCT CASE WHEN o.CURRENT_STAGE = 'CANCELLED' THEN o.OPPORTUNITY_ID END)  AS CANCELS,
        COUNT(DISTINCT CASE WHEN o.CURRENT_STAGE = 'LOST' THEN o.OPPORTUNITY_ID END)       AS LOST,
        COUNT(DISTINCT CASE WHEN o.CURRENT_STAGE NOT IN ('WON','CANCELLED','LOST','CAG PARENT CLOSED')
            AND DATEDIFF('DAY', o.OPPORTUNITY_CREATED_DATE, LAST_DAY(m.REPORTING_MONTH)) >= 180
            THEN o.OPPORTUNITY_ID END)                                                     AS OPEN_180_PLUS,
        COUNT(DISTINCT CASE WHEN o.CURRENT_STAGE NOT IN ('WON','CANCELLED','LOST','CAG PARENT CLOSED')
            AND DATEDIFF('DAY', o.OPPORTUNITY_CREATED_DATE, LAST_DAY(m.REPORTING_MONTH)) < 180
            THEN o.OPPORTUNITY_ID END)                                                     AS OPEN_ACTIVE
    FROM months m
    JOIN opps o
      ON (o.OPPORTUNITY_CREATED_DATE >= DATEADD('MONTH', -18, m.REPORTING_MONTH)
          AND o.OPPORTUNITY_CREATED_DATE < DATEADD('MONTH', 1, m.REPORTING_MONTH))
      OR (o.ACTUAL_CLOSE_DATE >= DATEADD('MONTH', -18, m.REPORTING_MONTH)
          AND o.ACTUAL_CLOSE_DATE < DATEADD('MONTH', 1, m.REPORTING_MONTH))
    GROUP BY 1, 2, 3
)
SELECT
    REPORTING_MONTH,
    TEAM_NAME,
    LEAD_SOURCE_GROUP,
    WINS,
    CANCELS,
    LOST,
    OPEN_180_PLUS,
    OPEN_ACTIVE,
    WINS + CANCELS + LOST + OPEN_180_PLUS                                                  AS RESOLVED_TOTAL,
    ROUND(100.0 * WINS / NULLIF(WINS + CANCELS + LOST + OPEN_180_PLUS, 0), 1)             AS CONVERSION_RATE_PCT
FROM monthly_conv
ORDER BY 2, 3, 1
"""

VELLO_PLACEMENTS_QUERY = """
SELECT
    DATE_TRUNC('MONTH', CF_GO_LIVE_DATE)    AS MONTH,
    COUNT(DISTINCT PROJECT_ID)              AS VELLO_PLACEMENTS
FROM VSSANALYTICS_DB.GUIDECX.CDL_ONBOARDING
WHERE VELLO_BOOLEAN = 1
  AND CF_GO_LIVE_DATE IS NOT NULL
  AND STATUS NOT IN ('CANCELLED','ON_HOLD')
GROUP BY 1
ORDER BY 1
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
    CF_PRODUCT                                                          AS PRODUCT,
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

MQL_CONVERSION_QUERY = """
SELECT
    DATE_TRUNC('MONTH', CONTACT_RECENT_MQL_DATE)    AS MONTH,
    'MQL'                                           AS LEAD_SOURCE,
    CASE
        WHEN FLATTENED_PRODUCT IN ('ezyVet', 'Neo') THEN 'ezyVet + Neo'
        WHEN FLATTENED_PRODUCT = 'Vello' THEN 'Vello'
    END                                             AS PRODUCT_GROUP,
    COUNT(DISTINCT HUBSPOT_CONTACT_ID)              AS LEADS,
    COUNT(DISTINCT OPPORTUNITY_ID)                   AS OPPS_CREATED,
    ROUND(100.0 * COUNT(DISTINCT OPPORTUNITY_ID)
        / NULLIF(COUNT(DISTINCT HUBSPOT_CONTACT_ID), 0), 1) AS LEAD_TO_OPP_CONVERSION_PCT
FROM VSSANALYTICS_DB.STRATPLAN_ANALYSIS.HUBSPOT_TO_SALESFORCE_MAPPING
WHERE IS_MQL = TRUE
  AND FLATTENED_PRODUCT IN ('ezyVet', 'Neo', 'Vello')
  AND CONTACT_RECENT_MQL_DATE IS NOT NULL
GROUP BY 1, 2, 3
UNION ALL
SELECT
    DATE_TRUNC('MONTH', CREATEDDATE)                AS MONTH,
    'VDC'                                           AS LEAD_SOURCE,
    CASE
        WHEN PRODUCT IN ('ezyVet', 'Neo') THEN 'ezyVet + Neo'
        WHEN PRODUCT = 'Vello' THEN 'Vello'
    END                                             AS PRODUCT_GROUP,
    COUNT(DISTINCT ID)                               AS LEADS,
    NULL                                             AS OPPS_CREATED,
    NULL                                             AS LEAD_TO_OPP_CONVERSION_PCT
FROM VSSANALYTICS_DB.SFDC.SF_VDC_LEADS
WHERE PRODUCT IN ('ezyVet', 'Neo', 'Vello')
  AND CREATEDDATE IS NOT NULL
GROUP BY 1, 2, 3
ORDER BY 3, 2, 1
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
    df["REPORTING_MONTH"] = pd.to_datetime(df["REPORTING_MONTH"])
    df["TEAM_NAME"] = df["TEAM_NAME"].replace(TEAM_RENAME)
    combined = df.groupby(["REPORTING_MONTH", "TEAM_NAME", "LEAD_SOURCE_GROUP"], as_index=False).agg(
        {"WINS": "sum", "CANCELS": "sum", "LOST": "sum", "OPEN_180_PLUS": "sum", "OPEN_ACTIVE": "sum"}
    )
    combined["RESOLVED_TOTAL"] = combined["WINS"] + combined["CANCELS"] + combined["LOST"] + combined["OPEN_180_PLUS"]
    combined["CONVERSION_RATE_PCT"] = (100.0 * combined["WINS"] / combined["RESOLVED_TOTAL"].replace(0, pd.NA)).round(1)
    return combined


@st.cache_data(ttl=3600)
def load_vello_placements() -> pd.DataFrame:
    df = _run_query(VELLO_PLACEMENTS_QUERY)
    df["MONTH"] = pd.to_datetime(df["MONTH"])
    return df


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


@st.cache_data(ttl=3600)
def load_mql_conversion() -> pd.DataFrame:
    df = _run_query(MQL_CONVERSION_QUERY)
    df["MONTH"] = pd.to_datetime(df["MONTH"])
    return df


def load_all_data() -> dict:
    return {
        "sqls_created": load_sqls_created(),
        "conversion_rates": load_conversion_rates(),
        "vello_placements": load_vello_placements(),
        "time_to_booking": load_time_to_booking(),
        "time_to_placement": load_time_to_placement(),
        "install_base": load_install_base(),
        "churns": load_churns(),
        "mql_conversion": load_mql_conversion(),
    }


# =============================================================================
# HISTORICAL DEFAULTS HELPERS
# =============================================================================
def _safe_mean(df, product_group, col):
    subset = df[df["PRODUCT_GROUP"] == product_group] if "PRODUCT_GROUP" in df.columns else df
    if subset.empty or col not in subset.columns:
        return 0.0
    vals = subset[col].dropna()
    return vals.mean() if not vals.empty else 0.0


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


def _safe_team_conv(df, team):
    subset = df[df["TEAM_NAME"] == team]
    if subset.empty:
        return 0.0
    total_resolved = subset["RESOLVED_TOTAL"].sum()
    return (100.0 * subset["WINS"].sum() / total_resolved) if total_resolved > 0 else 0.0


def _safe_team_ls_monthly_total(df, team, lead_source, month_col, value_col):
    """Average monthly total for a specific team + lead source."""
    subset = df[(df["TEAM_NAME"] == team) & (df["LEAD_SOURCE_GROUP"] == lead_source)]
    if subset.empty or value_col not in subset.columns:
        return 0.0
    monthly = subset.groupby(month_col)[value_col].sum()
    return monthly.mean() if not monthly.empty else 0.0


def _safe_team_ls_conv(df, team, lead_source):
    """Conversion rate for a specific team + lead source."""
    subset = df[(df["TEAM_NAME"] == team) & (df["LEAD_SOURCE_GROUP"] == lead_source)]
    if subset.empty:
        return 0.0
    total_resolved = subset["RESOLVED_TOTAL"].sum()
    return (100.0 * subset["WINS"].sum() / total_resolved) if total_resolved > 0 else 0.0


def _safe_team_ls_mean(df, team, lead_source, col):
    """Mean of a column for a specific team + lead source."""
    subset = df[(df["TEAM_NAME"] == team) & (df["LEAD_SOURCE_GROUP"] == lead_source)]
    if subset.empty or col not in subset.columns:
        return 0.0
    vals = subset[col].dropna()
    return vals.mean() if not vals.empty else 0.0


def compute_historical_defaults(data):
    defaults = {}
    recent_months = 6

    mql_df = data["mql_conversion"]
    mql_recent = mql_df[mql_df["MONTH"] >= mql_df["MONTH"].max() - pd.DateOffset(months=recent_months)]
    sqls_df = data["sqls_created"]
    sqls_recent = sqls_df[sqls_df["MONTH"] >= sqls_df["MONTH"].max() - pd.DateOffset(months=recent_months)]
    conv_df = data["conversion_rates"]
    conv_latest = conv_df[conv_df["REPORTING_MONTH"] == conv_df["REPORTING_MONTH"].max()]
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

    # Vello attach rate approximation
    placements_df = data["vello_placements"]
    ss_wins = conv_latest[conv_latest["TEAM_NAME"] == "SOFTWARE SALES"]["WINS"].sum()
    recent_vello = placements_df[placements_df["MONTH"] >= placements_df["MONTH"].max() - pd.DateOffset(months=recent_months)]["VELLO_PLACEMENTS"].mean() if not placements_df.empty else 0
    ss_monthly_wins = ss_wins / 18 if ss_wins > 0 else 1
    vello_attach = min((recent_vello / ss_monthly_wins) * 100, 100) if ss_monthly_wins > 0 else 30.0

    ttp_median = ttp_recent["MEDIAN_DAYS_TO_PLACEMENT"].mean() if not ttp_recent.empty else 90.0

    # MQL mappings per team (informational only)
    mql_map = {
        "ISAS": {"product_group": "Vello"},
        "SOFTWARE SALES": {"product_group": "ezyVet + Neo"},
        "ESAM": {"product_group": "ezyVet + Neo"},
    }
    attach_map = {
        "SOFTWARE SALES": round(vello_attach, 1),
        "ESAM": 100.0,
    }

    for team in ALL_TEAMS:
        pg = mql_map[team]["product_group"]
        has_attach = team in attach_map
        team_defaults = {
            "mqls_per_month": round(_safe_mean(mql_recent, pg, "LEADS"), 0),
            "mql_conversion_rate": round(_safe_mean(mql_recent, pg, "LEAD_TO_OPP_CONVERSION_PCT"), 1),
            "time_to_implement_days": round(ttp_median if not ttp_recent.empty else (90.0 if team == "ISAS" else 120.0), 0),
            "monthly_churn_rate": round(monthly_churn_rate * 100, 2),
        }
        if has_attach:
            team_defaults["vello_attach_rate"] = attach_map[team]

        # Per lead-source defaults
        ls_defaults = {}
        for ls in LEAD_SOURCES:
            ls_defaults[ls] = {
                "sqls_per_month": round(_safe_team_ls_monthly_total(sqls_recent, team, ls, "MONTH", "SQLS_CREATED"), 0),
                "sql_conversion_rate": round(_safe_team_ls_conv(conv_latest, team, ls), 1),
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
    latest_month = conv["REPORTING_MONTH"].max()
    ttb = data["time_to_booking"]
    ttp = data["time_to_placement"]
    ttp_recent = ttp[ttp["MONTH"] >= ttp["MONTH"].max() - pd.DateOffset(months=recent_months)]
    placements = data["vello_placements"]
    ttp_impl_days = ttp_recent["MEDIAN_DAYS_TO_PLACEMENT"].mean() if not ttp_recent.empty else 90

    for team in ALL_TEAMS:
        # Compute team-level attach rate (shared across lead sources)
        team_sqls_all = sqls[sqls["TEAM_NAME"] == team]
        team_recent_all = team_sqls_all[team_sqls_all["MONTH"] >= team_sqls_all["MONTH"].max() - pd.DateOffset(months=recent_months)]
        monthly_sqls_all = team_recent_all.groupby("MONTH")["SQLS_CREATED"].sum()
        team_conv_all = conv[(conv["TEAM_NAME"] == team) & (conv["REPORTING_MONTH"] == latest_month)]
        conv_rate_all = team_conv_all["WINS"].sum() / max(team_conv_all["RESOLVED_TOTAL"].sum(), 1)

        if team == "SOFTWARE SALES":
            plc_recent = placements[placements["MONTH"] >= placements["MONTH"].max() - pd.DateOffset(months=recent_months)] if not placements.empty else placements
            monthly_plc = plc_recent["VELLO_PLACEMENTS"].mean() if not plc_recent.empty else 0
            ss_monthly_wins = (conv_rate_all * monthly_sqls_all.mean()) if not monthly_sqls_all.empty else 1
            attach = min(monthly_plc / max(ss_monthly_wins, 1), 1.0)
        elif team == "ISAS":
            attach = 1.0
        else:
            attach = 1.0

        for ls in LEAD_SOURCES:
            ls_sqls = sqls[(sqls["TEAM_NAME"] == team) & (sqls["LEAD_SOURCE_GROUP"] == ls)]
            ls_recent = ls_sqls[ls_sqls["MONTH"] >= ls_sqls["MONTH"].max() - pd.DateOffset(months=recent_months)] if not ls_sqls.empty else ls_sqls
            monthly = ls_recent.groupby("MONTH")["SQLS_CREATED"].sum() if not ls_recent.empty else pd.Series(dtype=float)

            ls_conv = conv[(conv["TEAM_NAME"] == team) & (conv["LEAD_SOURCE_GROUP"] == ls) & (conv["REPORTING_MONTH"] == latest_month)]
            cr = ls_conv["WINS"].sum() / max(ls_conv["RESOLVED_TOTAL"].sum(), 1)

            ls_ttb = ttb[(ttb["TEAM_NAME"] == team) & (ttb["LEAD_SOURCE_GROUP"] == ls) & (ttb["MONTH"] >= ttb["MONTH"].max() - pd.DateOffset(months=recent_months))]

            params[(team, ls)] = {
                "sqls_mean": monthly.mean() if not monthly.empty else 0,
                "sqls_std": max(monthly.std(), 1) if not monthly.empty else 1,
                "conv_rate": cr,
                "conv_rate_std": 0.03,
                "time_to_sale_days": ls_ttb["MEDIAN_DAYS_TO_BOOKING"].mean() if not ls_ttb.empty else 60,
                "time_to_implement_days": ttp_impl_days,
                "attach_rate": attach,
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
    lead_growth_rate=0.0,
):
    rng = np.random.default_rng(42)
    results = np.zeros((n_simulations, projection_months))

    for sim in range(n_simulations):
        ib = float(starting_install_base)
        pipeline = []

        # Pre-fill pipeline with in-flight deals
        for team_name, tp in team_params.items():
            sale_months = max(tp["time_to_sale_days"] / 30.0, 0.5)
            impl_months = max(tp["time_to_implement_days"] / 30.0, 0.5)
            total_lag = sale_months + impl_months
            for lag_month in range(int(np.ceil(total_lag))):
                growth_factor = (1 + lead_growth_rate) ** max(0, -lag_month)
                s = max(rng.normal(tp["sqls_mean"] * growth_factor, tp["sqls_std"]), 0)
                c = np.clip(rng.normal(tp["conv_rate"], tp.get("conv_rate_std", 0.03)), 0, 1)
                wins = s * c * tp["attach_rate"]
                go_live = total_lag - lag_month
                if go_live >= 0:
                    pipeline.append((int(np.round(go_live)), wins))

        for month in range(projection_months):
            growth_factor = (1 + lead_growth_rate) ** month
            monthly_new = 0

            for team_name, tp in team_params.items():
                s = max(rng.normal(tp["sqls_mean"] * growth_factor, tp["sqls_std"]), 0)
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


def run_scenario_simulation(starting_install_base, scenario_params, churn_rate_mean, churn_rate_std, lead_growth_rate=0.0, projection_months=PROJECTION_YEARS * 12, n_simulations=MONTE_CARLO_SIMULATIONS):
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
    return run_monte_carlo(starting_install_base, team_params, churn_rate_mean, churn_rate_std, projection_months, n_simulations, lead_growth_rate)


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
churn_mean = run_rate_params["churn_rate_mean"]
churn_std = run_rate_params["churn_rate_std"]
baseline_team_params = {k: v for k, v in run_rate_params.items() if k not in ("churn_rate_mean", "churn_rate_std") and isinstance(k, tuple)}

projection_months = PROJECTION_YEARS * 12
baseline_result = run_monte_carlo(latest_ib, baseline_team_params, churn_mean, churn_std, projection_months)
projection_dates = [latest_month + relativedelta(months=i) for i in range(1, projection_months + 1)]

# Session state for scenario
if "scenario_result" not in st.session_state:
    st.session_state.scenario_result = None
if "scenario_name" not in st.session_state:
    st.session_state.scenario_name = ""


@st.dialog("Scenario Planner", width="large")
def scenario_planner_modal():
    st.markdown("Adjust parameters by team, then click **Run Simulation** to project the scenario.")
    scenario_name = st.text_input("Scenario Name", value="My Scenario", key="modal_scenario_name")

    st.subheader("Global Settings")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        lead_growth = st.slider("Monthly Lead/SQL Growth Rate (%)", min_value=-5.0, max_value=10.0, value=0.0, step=0.1, key="global_lead_growth", help="Compound monthly growth applied to SQLs across all teams")
    with col_g2:
        churn_override = st.slider("Monthly Churn Rate (%)", min_value=0.0, max_value=5.0, value=round(churn_mean * 100, 2), step=0.05, key="global_churn")

    team_tabs = st.tabs(ALL_TEAMS)
    scenario_params = {}

    for tab, team in zip(team_tabs, ALL_TEAMS):
        with tab:
            d = defaults.get(team, {})
            ls_defaults = d.get("lead_sources", {})
            has_attach = team in ("SOFTWARE SALES", "ESAM")
            st.markdown(f"**{team}** — adjust the levers below")

            # Team-level shared inputs
            shared_c1, shared_c2, shared_c3 = st.columns(3)
            with shared_c1:
                mqls = st.number_input("MQLs / Month", min_value=0, max_value=5000, value=int(d.get("mqls_per_month", 100)), step=10, key=f"{team}_mqls")
                mql_conv = st.slider("MQL → SQL Conv (%)", min_value=0.0, max_value=100.0, value=float(d.get("mql_conversion_rate", 10.0)), step=0.5, key=f"{team}_mql_conv", help="Informational — SQLs are set independently since MQL-to-SQL mapping isn't 1:1")
            with shared_c2:
                tti = st.number_input("Time to Implement (days)", min_value=1, max_value=2000, value=int(d.get("time_to_implement_days", 90)), step=5, key=f"{team}_tti")
            with shared_c3:
                attach = 100.0
                if has_attach:
                    attach = st.slider("Vello Attach Rate (%)", min_value=0.0, max_value=100.0, value=float(d.get("vello_attach_rate", 30.0)), step=1.0, key=f"{team}_attach", help="% of converted deals that include Vello")

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

                    wins = sqls * (sql_conv / 100.0) * (attach / 100.0)
                    total_monthly_wins += wins
                    ls_details.append(f"{ls}: {sqls} SQLs × {sql_conv:.1f}%")

                    scenario_params[(team, ls)] = {
                        "sqls_per_month": sqls,
                        "sql_conversion_rate": sql_conv,
                        "time_to_sale_days": tts,
                        "time_to_implement_days": tti,
                        "vello_attach_rate": attach,
                    }

            st.info(
                f"**Projected monthly Vello placements from {team}:** "
                + " + ".join(ls_details)
                + (f" × {attach:.1f}% attach" if has_attach else "")
                + f" = **{total_monthly_wins:.1f}** placements/month"
            )

    st.divider()
    if st.button("Run Simulation", type="primary", use_container_width=True):
        with st.spinner(f"Running {MONTE_CARLO_SIMULATIONS:,} simulations..."):
            result = run_scenario_simulation(latest_ib, scenario_params, churn_override / 100.0, churn_std, lead_growth / 100.0, projection_months)
        st.session_state.scenario_result = result
        st.session_state.scenario_name = scenario_name
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
# HISTORICAL DATA EXPLORER
# =============================================================================
with st.expander("Historical Data Explorer"):
    t1, t2, t3, t4, t5 = st.tabs(["Install Base", "Churns", "SQLs Created", "Conversion Rates", "MQL Conversion"])
    with t1:
        st.dataframe(ib_df.sort_values("REPORTING_MONTH", ascending=False), use_container_width=True, hide_index=True)
    with t2:
        st.dataframe(data["churns"].sort_values("REPORTING_MONTH", ascending=False), use_container_width=True, hide_index=True)
    with t3:
        st.dataframe(data["sqls_created"].sort_values(["TEAM_NAME", "MONTH"], ascending=[True, False]), use_container_width=True, hide_index=True)
    with t4:
        st.dataframe(data["conversion_rates"].sort_values(["TEAM_NAME", "REPORTING_MONTH"], ascending=[True, False]), use_container_width=True, hide_index=True)
    with t5:
        st.dataframe(data["mql_conversion"].sort_values("MONTH", ascending=False), use_container_width=True, hide_index=True)
