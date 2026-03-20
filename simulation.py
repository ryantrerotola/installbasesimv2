"""Monte Carlo simulation engine for install base projections."""

import numpy as np
import pandas as pd
from config import MONTE_CARLO_SIMULATIONS, CONFIDENCE_LEVEL, PROJECTION_YEARS


def compute_run_rate_params(data: dict) -> dict:
    """Extract recent run-rate parameters from historical data for baseline projection.

    Returns parameters representing the current trajectory (means + std devs)
    for each team, used as the baseline "run rate" projection.
    """
    recent_months = 6
    params = {}

    # --- ISAS ---
    sqls = data["sqls_created"]
    isas_sqls = sqls[sqls["TEAM_NAME"] == "ISAS"]
    isas_recent = isas_sqls[isas_sqls["MONTH"] >= isas_sqls["MONTH"].max() - pd.DateOffset(months=recent_months)]
    isas_monthly_sqls = isas_recent.groupby("MONTH")["SQLS_CREATED"].sum()

    conv = data["conversion_rates"]
    latest_month = conv["REPORTING_MONTH"].max()
    isas_conv = conv[(conv["TEAM_NAME"] == "ISAS") & (conv["REPORTING_MONTH"] == latest_month)]
    isas_conv_rate = (isas_conv["WINS"].sum() / max(isas_conv["RESOLVED_TOTAL"].sum(), 1))

    ttb = data["time_to_booking"]
    isas_ttb = ttb[
        (ttb["TEAM_NAME"] == "ISAS")
        & (ttb["MONTH"] >= ttb["MONTH"].max() - pd.DateOffset(months=recent_months))
    ]

    ttp = data["time_to_placement"]
    ttp_recent = ttp[ttp["MONTH"] >= ttp["MONTH"].max() - pd.DateOffset(months=recent_months)]

    params["ISAS"] = {
        "sqls_mean": isas_monthly_sqls.mean() if not isas_monthly_sqls.empty else 0,
        "sqls_std": max(isas_monthly_sqls.std(), 1) if not isas_monthly_sqls.empty else 1,
        "conv_rate": isas_conv_rate,
        "conv_rate_std": 0.03,  # approximate variability
        "time_to_sale_days": isas_ttb["MEDIAN_DAYS_TO_BOOKING"].mean() if not isas_ttb.empty else 60,
        "time_to_implement_days": ttp_recent["MEDIAN_DAYS_TO_PLACEMENT"].mean() if not ttp_recent.empty else 90,
        "attach_rate": 1.0,  # ISAS is all Vello
    }

    # --- SOFTWARE SALES ---
    ss_sqls = sqls[sqls["TEAM_NAME"] == "SOFTWARE SALES"]
    ss_recent = ss_sqls[ss_sqls["MONTH"] >= ss_sqls["MONTH"].max() - pd.DateOffset(months=recent_months)]
    ss_monthly_sqls = ss_recent.groupby("MONTH")["SQLS_CREATED"].sum()

    ss_conv = conv[(conv["TEAM_NAME"] == "SOFTWARE SALES") & (conv["REPORTING_MONTH"] == latest_month)]
    ss_conv_rate = (ss_conv["WINS"].sum() / max(ss_conv["RESOLVED_TOTAL"].sum(), 1))

    ss_ttb = ttb[
        (ttb["TEAM_NAME"] == "SOFTWARE SALES")
        & (ttb["MONTH"] >= ttb["MONTH"].max() - pd.DateOffset(months=recent_months))
    ]

    # Vello attach rate from placements data
    placements = data["vello_placements"]
    plc_recent = placements[
        placements["MONTH"] >= placements["MONTH"].max() - pd.DateOffset(months=recent_months)
    ] if not placements.empty else placements
    monthly_placements = plc_recent["VELLO_PLACEMENTS"].mean() if not plc_recent.empty else 0
    ss_monthly_wins = (ss_conv_rate * ss_monthly_sqls.mean()) if not ss_monthly_sqls.empty else 1
    vello_attach = min(monthly_placements / max(ss_monthly_wins, 1), 1.0)

    params["SOFTWARE SALES"] = {
        "sqls_mean": ss_monthly_sqls.mean() if not ss_monthly_sqls.empty else 0,
        "sqls_std": max(ss_monthly_sqls.std(), 1) if not ss_monthly_sqls.empty else 1,
        "conv_rate": ss_conv_rate,
        "conv_rate_std": 0.03,
        "time_to_sale_days": ss_ttb["MEDIAN_DAYS_TO_BOOKING"].mean() if not ss_ttb.empty else 90,
        "time_to_implement_days": ttp_recent["MEDIAN_DAYS_TO_PLACEMENT"].mean() if not ttp_recent.empty else 120,
        "attach_rate": vello_attach,
    }

    # --- ESAM ---
    esam_sqls = sqls[sqls["TEAM_NAME"] == "ESAM"]
    esam_recent = esam_sqls[esam_sqls["MONTH"] >= esam_sqls["MONTH"].max() - pd.DateOffset(months=recent_months)]
    esam_monthly_sqls = esam_recent.groupby("MONTH")["SQLS_CREATED"].sum()

    esam_conv = conv[(conv["TEAM_NAME"] == "ESAM") & (conv["REPORTING_MONTH"] == latest_month)]
    esam_conv_rate = (esam_conv["WINS"].sum() / max(esam_conv["RESOLVED_TOTAL"].sum(), 1))

    esam_ttb = ttb[
        (ttb["TEAM_NAME"] == "ESAM")
        & (ttb["MONTH"] >= ttb["MONTH"].max() - pd.DateOffset(months=recent_months))
    ]

    params["ESAM"] = {
        "sqls_mean": esam_monthly_sqls.mean() if not esam_monthly_sqls.empty else 0,
        "sqls_std": max(esam_monthly_sqls.std(), 1) if not esam_monthly_sqls.empty else 1,
        "conv_rate": esam_conv_rate,
        "conv_rate_std": 0.03,
        "time_to_sale_days": esam_ttb["MEDIAN_DAYS_TO_BOOKING"].mean() if not esam_ttb.empty else 90,
        "time_to_implement_days": ttp_recent["MEDIAN_DAYS_TO_PLACEMENT"].mean() if not ttp_recent.empty else 120,
        "attach_rate": 1.0,  # ESAM is all ezyVet/Vello
    }

    # --- Churn ---
    churns = data["churns"]
    ib = data["install_base"]
    if not churns.empty and not ib.empty:
        recent_churns = churns[
            churns["REPORTING_MONTH"] >= churns["REPORTING_MONTH"].max() - pd.DateOffset(months=recent_months)
        ]
        recent_ib = ib[
            ib["REPORTING_MONTH"] >= ib["REPORTING_MONTH"].max() - pd.DateOffset(months=recent_months)
        ]
        params["churn_rate_mean"] = (
            recent_churns["VELLO_CHURNS"].mean() / max(recent_ib["VELLO_INSTALL_BASE"].mean(), 1)
        )
        params["churn_rate_std"] = max(
            recent_churns["VELLO_CHURNS"].std() / max(recent_ib["VELLO_INSTALL_BASE"].mean(), 1),
            0.001,
        )
    else:
        params["churn_rate_mean"] = 0.01
        params["churn_rate_std"] = 0.003

    return params


def run_monte_carlo(
    starting_install_base: int,
    team_params: dict,
    churn_rate_mean: float,
    churn_rate_std: float,
    projection_months: int = PROJECTION_YEARS * 12,
    n_simulations: int = MONTE_CARLO_SIMULATIONS,
    lead_growth_rate: float = 0.0,
) -> dict:
    """Run Monte Carlo simulation for install base projection.

    Args:
        starting_install_base: Current install base count.
        team_params: Dict of team -> {sqls_mean, sqls_std, conv_rate, conv_rate_std,
                     time_to_sale_days, time_to_implement_days, attach_rate}.
        churn_rate_mean: Monthly churn rate (as decimal, e.g. 0.01 = 1%).
        churn_rate_std: Std dev of monthly churn rate.
        projection_months: Number of months to project.
        n_simulations: Number of Monte Carlo iterations.
        lead_growth_rate: Monthly compound growth rate for leads/SQLs (e.g. 0.02 = 2%).

    Returns:
        Dict with 'months', 'median', 'p5', 'p95', 'mean' arrays.
    """
    rng = np.random.default_rng(42)
    results = np.zeros((n_simulations, projection_months))

    for sim in range(n_simulations):
        ib = float(starting_install_base)

        # Build a pipeline of deals in progress at start
        # Each team contributes deals that are in sales or implementation
        pipeline = []
        for team_name, tp in team_params.items():
            sale_months = max(tp["time_to_sale_days"] / 30.0, 0.5)
            impl_months = max(tp["time_to_implement_days"] / 30.0, 0.5)
            total_lag = sale_months + impl_months

            # Pre-fill pipeline: deals created in prior months that haven't gone live yet
            for lag_month in range(int(np.ceil(total_lag))):
                growth_factor = (1 + lead_growth_rate) ** max(0, -lag_month)
                sqls = max(rng.normal(tp["sqls_mean"] * growth_factor, tp["sqls_std"]), 0)
                conv = np.clip(rng.normal(tp["conv_rate"], tp.get("conv_rate_std", 0.03)), 0, 1)
                wins = sqls * conv * tp["attach_rate"]
                # These deals go live at month (total_lag - lag_month)
                go_live_month = total_lag - lag_month
                if go_live_month >= 0:
                    pipeline.append((int(np.round(go_live_month)), wins))

        for month in range(projection_months):
            # New SQLs created this month (with growth)
            growth_factor = (1 + lead_growth_rate) ** month
            monthly_new_placements = 0

            for team_name, tp in team_params.items():
                sqls = max(rng.normal(tp["sqls_mean"] * growth_factor, tp["sqls_std"]), 0)
                conv = np.clip(rng.normal(tp["conv_rate"], tp.get("conv_rate_std", 0.03)), 0, 1)
                wins = sqls * conv * tp["attach_rate"]

                # These wins go live after time_to_sale + time_to_implement
                sale_months = max(tp["time_to_sale_days"] / 30.0, 0.5)
                impl_months = max(tp["time_to_implement_days"] / 30.0, 0.5)
                go_live_month = month + sale_months + impl_months
                pipeline.append((int(np.round(go_live_month)), wins))

            # Process pipeline: add placements that go live this month
            remaining_pipeline = []
            for go_live, wins in pipeline:
                if go_live <= month:
                    monthly_new_placements += wins
                else:
                    remaining_pipeline.append((go_live, wins))
            pipeline = remaining_pipeline

            # Churn
            churn_rate = np.clip(rng.normal(churn_rate_mean, churn_rate_std), 0, 0.1)
            monthly_churns = ib * churn_rate

            # Update install base
            ib = ib + monthly_new_placements - monthly_churns
            ib = max(ib, 0)
            results[sim, month] = ib

    # Compute statistics
    alpha = (1 - CONFIDENCE_LEVEL) / 2
    return {
        "months": np.arange(1, projection_months + 1),
        "median": np.median(results, axis=0),
        "p5": np.percentile(results, alpha * 100, axis=0),
        "p95": np.percentile(results, (1 - alpha) * 100, axis=0),
        "mean": np.mean(results, axis=0),
    }


def run_scenario_simulation(
    starting_install_base: int,
    scenario_params: dict,
    churn_rate_mean: float,
    churn_rate_std: float,
    lead_growth_rate: float = 0.0,
    projection_months: int = PROJECTION_YEARS * 12,
    n_simulations: int = MONTE_CARLO_SIMULATIONS,
) -> dict:
    """Run Monte Carlo simulation for a user-defined scenario.

    Args:
        starting_install_base: Current install base count.
        scenario_params: Dict of team -> {sqls_per_month, sql_conversion_rate,
                         time_to_sale_days, time_to_implement_days, attach_rate (optional)}.
                         These are the user-adjusted values from the scenario planner.
        churn_rate_mean: Monthly churn rate (decimal).
        churn_rate_std: Std dev of monthly churn rate.
        lead_growth_rate: Monthly compound growth rate for leads/SQLs.
        projection_months: Number of months to project.
        n_simulations: Number of Monte Carlo iterations.

    Returns:
        Dict with 'months', 'median', 'p5', 'p95', 'mean' arrays.
    """
    # Convert scenario_params into the team_params format expected by run_monte_carlo
    team_params = {}
    for team_name, sp in scenario_params.items():
        team_params[team_name] = {
            "sqls_mean": sp["sqls_per_month"],
            "sqls_std": sp["sqls_per_month"] * 0.15,  # 15% variability
            "conv_rate": sp["sql_conversion_rate"] / 100.0,
            "conv_rate_std": sp["sql_conversion_rate"] / 100.0 * 0.1,  # 10% relative variability
            "time_to_sale_days": sp["time_to_sale_days"],
            "time_to_implement_days": sp["time_to_implement_days"],
            "attach_rate": sp.get("vello_attach_rate", 100.0) / 100.0,
        }

    return run_monte_carlo(
        starting_install_base=starting_install_base,
        team_params=team_params,
        churn_rate_mean=churn_rate_mean,
        churn_rate_std=churn_rate_std,
        projection_months=projection_months,
        n_simulations=n_simulations,
        lead_growth_rate=lead_growth_rate,
    )
