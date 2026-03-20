"""Data loading layer - pulls historical data from Snowflake and caches it."""

import streamlit as st
import pandas as pd
import snowflake.connector
from queries import (
    SQLS_CREATED_QUERY,
    CONVERSION_RATES_QUERY,
    VELLO_PLACEMENTS_QUERY,
    TIME_TO_BOOKING_QUERY,
    TIME_TO_PLACEMENT_QUERY,
    INSTALL_BASE_QUERY,
    CHURNS_QUERY,
    MQL_CONVERSION_QUERY,
)


def get_snowflake_connection():
    """Create a Snowflake connection using Streamlit secrets."""
    return snowflake.connector.connect(
        account=st.secrets["snowflake"]["account"],
        user=st.secrets["snowflake"]["user"],
        password=st.secrets["snowflake"]["password"],
        warehouse=st.secrets["snowflake"]["warehouse"],
        database=st.secrets["snowflake"]["database"],
        schema=st.secrets["snowflake"]["schema"],
        role=st.secrets["snowflake"]["role"],
    )


def _run_query(query: str) -> pd.DataFrame:
    """Execute a query and return a DataFrame."""
    conn = get_snowflake_connection()
    try:
        df = pd.read_sql(query, conn)
        df.columns = [c.upper() for c in df.columns]
        return df
    finally:
        conn.close()


@st.cache_data(ttl=3600)
def load_sqls_created() -> pd.DataFrame:
    """Monthly SQLs created by team and lead source."""
    df = _run_query(SQLS_CREATED_QUERY)
    df["MONTH"] = pd.to_datetime(df["MONTH"])
    # Combine SOFTWARE SALES EAST/WEST
    df["TEAM_NAME"] = df["TEAM_NAME"].replace(
        {"SOFTWARE SALES EAST": "SOFTWARE SALES", "SOFTWARE SALES WEST": "SOFTWARE SALES"}
    )
    df = df.groupby(["MONTH", "TEAM_NAME", "LEAD_SOURCE_GROUP"], as_index=False)[
        "SQLS_CREATED"
    ].sum()
    return df


@st.cache_data(ttl=3600)
def load_conversion_rates() -> pd.DataFrame:
    """Rolling 18-month conversion rates by team and lead source."""
    df = _run_query(CONVERSION_RATES_QUERY)
    df["REPORTING_MONTH"] = pd.to_datetime(df["REPORTING_MONTH"])
    # Combine SOFTWARE SALES EAST/WEST
    df["TEAM_NAME"] = df["TEAM_NAME"].replace(
        {"SOFTWARE SALES EAST": "SOFTWARE SALES", "SOFTWARE SALES WEST": "SOFTWARE SALES"}
    )
    combined = (
        df.groupby(["REPORTING_MONTH", "TEAM_NAME", "LEAD_SOURCE_GROUP"], as_index=False)
        .agg(
            {
                "WINS": "sum",
                "CANCELS": "sum",
                "LOST": "sum",
                "OPEN_180_PLUS": "sum",
                "OPEN_ACTIVE": "sum",
            }
        )
    )
    combined["RESOLVED_TOTAL"] = (
        combined["WINS"] + combined["CANCELS"] + combined["LOST"] + combined["OPEN_180_PLUS"]
    )
    combined["CONVERSION_RATE_PCT"] = (
        100.0 * combined["WINS"] / combined["RESOLVED_TOTAL"].replace(0, pd.NA)
    ).round(1)
    return combined


@st.cache_data(ttl=3600)
def load_vello_placements() -> pd.DataFrame:
    """Monthly Vello placements from GuideCX."""
    df = _run_query(VELLO_PLACEMENTS_QUERY)
    df["MONTH"] = pd.to_datetime(df["MONTH"])
    return df


@st.cache_data(ttl=3600)
def load_time_to_booking() -> pd.DataFrame:
    """Time to booking by team and lead source."""
    df = _run_query(TIME_TO_BOOKING_QUERY)
    df["MONTH"] = pd.to_datetime(df["MONTH"])
    df["TEAM_NAME"] = df["TEAM_NAME"].replace(
        {"SOFTWARE SALES EAST": "SOFTWARE SALES", "SOFTWARE SALES WEST": "SOFTWARE SALES"}
    )
    combined = (
        df.groupby(["MONTH", "TEAM_NAME", "LEAD_SOURCE_GROUP"], as_index=False)
        .agg(
            {
                "BOOKINGS": "sum",
                "AVG_DAYS_TO_BOOKING": "mean",
                "MEDIAN_DAYS_TO_BOOKING": "mean",
            }
        )
    )
    return combined


@st.cache_data(ttl=3600)
def load_time_to_placement() -> pd.DataFrame:
    """Time to placement from GuideCX."""
    df = _run_query(TIME_TO_PLACEMENT_QUERY)
    df["MONTH"] = pd.to_datetime(df["MONTH"])
    return df


@st.cache_data(ttl=3600)
def load_install_base() -> pd.DataFrame:
    """Monthly Vello install base."""
    df = _run_query(INSTALL_BASE_QUERY)
    df["REPORTING_MONTH"] = pd.to_datetime(df["REPORTING_MONTH"])
    return df


@st.cache_data(ttl=3600)
def load_churns() -> pd.DataFrame:
    """Monthly Vello churns."""
    df = _run_query(CHURNS_QUERY)
    df["REPORTING_MONTH"] = pd.to_datetime(df["REPORTING_MONTH"])
    return df


@st.cache_data(ttl=3600)
def load_mql_conversion() -> pd.DataFrame:
    """MQL and VDC lead-to-opportunity conversion rates."""
    df = _run_query(MQL_CONVERSION_QUERY)
    df["MONTH"] = pd.to_datetime(df["MONTH"])
    return df


def load_all_data() -> dict:
    """Load all datasets and return as a dictionary."""
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


def compute_historical_defaults(data: dict) -> dict:
    """Compute default parameter values from historical data for each team.

    Returns a nested dict: defaults[team][param] = value.
    Uses recent 6-month averages where possible.
    """
    defaults = {}
    recent_months = 6

    # --- MQL data ---
    mql_df = data["mql_conversion"]
    mql_df_recent = mql_df[mql_df["MONTH"] >= mql_df["MONTH"].max() - pd.DateOffset(months=recent_months)]

    # --- SQLs data ---
    sqls_df = data["sqls_created"]
    sqls_recent = sqls_df[sqls_df["MONTH"] >= sqls_df["MONTH"].max() - pd.DateOffset(months=recent_months)]

    # --- Conversion rates (use latest month) ---
    conv_df = data["conversion_rates"]
    latest_conv_month = conv_df["REPORTING_MONTH"].max()
    conv_latest = conv_df[conv_df["REPORTING_MONTH"] == latest_conv_month]

    # --- Time to booking ---
    ttb_df = data["time_to_booking"]
    ttb_recent = ttb_df[ttb_df["MONTH"] >= ttb_df["MONTH"].max() - pd.DateOffset(months=recent_months)]

    # --- Time to placement ---
    ttp_df = data["time_to_placement"]
    ttp_recent = ttp_df[ttp_df["MONTH"] >= ttp_df["MONTH"].max() - pd.DateOffset(months=recent_months)]

    # --- Churn rate ---
    churns_df = data["churns"]
    ib_df = data["install_base"]
    if not churns_df.empty and not ib_df.empty:
        recent_churns = churns_df[
            churns_df["REPORTING_MONTH"] >= churns_df["REPORTING_MONTH"].max() - pd.DateOffset(months=recent_months)
        ]["VELLO_CHURNS"].mean()
        recent_ib = ib_df[
            ib_df["REPORTING_MONTH"] >= ib_df["REPORTING_MONTH"].max() - pd.DateOffset(months=recent_months)
        ]["VELLO_INSTALL_BASE"].mean()
        monthly_churn_rate = recent_churns / recent_ib if recent_ib > 0 else 0.01
    else:
        monthly_churn_rate = 0.01

    # --- Vello attach rate for SOFTWARE SALES ---
    placements_df = data["vello_placements"]
    # Approximate attach rate: Vello placements / total Won opps for SS
    ss_wins = conv_latest[conv_latest["TEAM_NAME"] == "SOFTWARE SALES"]["WINS"].sum()
    recent_vello = placements_df[
        placements_df["MONTH"] >= placements_df["MONTH"].max() - pd.DateOffset(months=recent_months)
    ]["VELLO_PLACEMENTS"].mean() if not placements_df.empty else 0

    # ISAS
    isas_mqls = _safe_mean(mql_df_recent, "Vello", "LEADS")
    isas_mql_conv = _safe_mean(mql_df_recent, "Vello", "LEAD_TO_OPP_CONVERSION_PCT")
    isas_sqls = _safe_team_mean(sqls_recent, "ISAS", "SQLS_CREATED")
    isas_sql_conv = _safe_team_conv(conv_latest, "ISAS")
    isas_ttb = _safe_team_mean(ttb_recent, "ISAS", "MEDIAN_DAYS_TO_BOOKING")
    isas_ttp = ttp_recent["MEDIAN_DAYS_TO_PLACEMENT"].mean() if not ttp_recent.empty else 90.0

    defaults["ISAS"] = {
        "mqls_per_month": round(isas_mqls, 0),
        "mql_conversion_rate": round(isas_mql_conv, 1),
        "sqls_per_month": round(isas_sqls, 0),
        "sql_conversion_rate": round(isas_sql_conv, 1),
        "time_to_sale_days": round(isas_ttb, 0),
        "time_to_implement_days": round(isas_ttp, 0),
        "monthly_churn_rate": round(monthly_churn_rate * 100, 2),
    }

    # SOFTWARE SALES (combined East/West)
    ss_mqls = _safe_mean(mql_df_recent, "ezyVet + Neo", "LEADS")
    ss_mqls += _safe_mean(mql_df_recent, "Vello", "LEADS")  # they also get Vello MQLs
    ss_mql_conv = _safe_mean(mql_df_recent, "ezyVet + Neo", "LEAD_TO_OPP_CONVERSION_PCT")
    ss_sqls = _safe_team_mean(sqls_recent, "SOFTWARE SALES", "SQLS_CREATED")
    ss_sql_conv = _safe_team_conv(conv_latest, "SOFTWARE SALES")
    ss_ttb = _safe_team_mean(ttb_recent, "SOFTWARE SALES", "MEDIAN_DAYS_TO_BOOKING")
    ss_ttp = ttp_recent["MEDIAN_DAYS_TO_PLACEMENT"].mean() if not ttp_recent.empty else 120.0

    # Attach rate: approximate from vello placements vs total SS wins
    ss_monthly_wins = ss_wins / max(CONVERSION_ROLLING_MONTHS_USED, 1) if ss_wins > 0 else 1
    vello_attach = min((recent_vello / ss_monthly_wins) * 100, 100) if ss_monthly_wins > 0 else 30.0

    defaults["SOFTWARE SALES"] = {
        "mqls_per_month": round(ss_mqls, 0),
        "mql_conversion_rate": round(ss_mql_conv, 1),
        "sqls_per_month": round(ss_sqls, 0),
        "sql_conversion_rate": round(ss_sql_conv, 1),
        "vello_attach_rate": round(vello_attach, 1),
        "time_to_sale_days": round(ss_ttb, 0),
        "time_to_implement_days": round(ss_ttp, 0),
        "monthly_churn_rate": round(monthly_churn_rate * 100, 2),
    }

    # ESAM
    esam_mqls = _safe_mean(mql_df_recent, "ezyVet + Neo", "LEADS")
    esam_mql_conv = _safe_mean(mql_df_recent, "ezyVet + Neo", "LEAD_TO_OPP_CONVERSION_PCT")
    esam_sqls = _safe_team_mean(sqls_recent, "ESAM", "SQLS_CREATED")
    esam_sql_conv = _safe_team_conv(conv_latest, "ESAM")
    esam_ttb = _safe_team_mean(ttb_recent, "ESAM", "MEDIAN_DAYS_TO_BOOKING")
    esam_ttp = ttp_recent["MEDIAN_DAYS_TO_PLACEMENT"].mean() if not ttp_recent.empty else 120.0

    defaults["ESAM"] = {
        "mqls_per_month": round(esam_mqls, 0),
        "mql_conversion_rate": round(esam_mql_conv, 1),
        "sqls_per_month": round(esam_sqls, 0),
        "sql_conversion_rate": round(esam_sql_conv, 1),
        "vello_attach_rate": 100.0,  # ESAM: same attach rate logic as SS
        "time_to_sale_days": round(esam_ttb, 0),
        "time_to_implement_days": round(esam_ttp, 0),
        "monthly_churn_rate": round(monthly_churn_rate * 100, 2),
    }

    return defaults


# Used as a fallback for attach rate calc
CONVERSION_ROLLING_MONTHS_USED = 18


def _safe_mean(df: pd.DataFrame, product_group: str, col: str) -> float:
    """Safely compute mean for a product group."""
    subset = df[df["PRODUCT_GROUP"] == product_group] if "PRODUCT_GROUP" in df.columns else df
    if subset.empty or col not in subset.columns:
        return 0.0
    vals = subset[col].dropna()
    return vals.mean() if not vals.empty else 0.0


def _safe_team_mean(df: pd.DataFrame, team: str, col: str) -> float:
    """Safely compute mean for a team."""
    subset = df[df["TEAM_NAME"] == team] if "TEAM_NAME" in df.columns else df
    if subset.empty or col not in subset.columns:
        return 0.0
    vals = subset[col].dropna()
    return vals.mean() if not vals.empty else 0.0


def _safe_team_conv(df: pd.DataFrame, team: str) -> float:
    """Get weighted average conversion rate for a team across lead sources."""
    subset = df[df["TEAM_NAME"] == team]
    if subset.empty:
        return 0.0
    total_resolved = subset["RESOLVED_TOTAL"].sum()
    if total_resolved == 0:
        return 0.0
    return (100.0 * subset["WINS"].sum() / total_resolved)
