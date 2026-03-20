"""Configuration and constants for the Install Base Simulator."""

# Teams
TEAM_ISAS = "ISAS"
TEAM_SS_EAST = "SOFTWARE SALES EAST"
TEAM_SS_WEST = "SOFTWARE SALES WEST"
TEAM_ESAM = "ESAM"

# Combined reporting label for East/West
TEAM_SS_COMBINED = "SOFTWARE SALES"

# Team groupings
ALL_TEAMS = [TEAM_ISAS, TEAM_SS_COMBINED, TEAM_ESAM]

# Products by team
ISAS_PRODUCTS = ["Vello"]
SS_PRODUCTS = ["ezyVet", "Neo", "Vello"]
ESAM_PRODUCTS = ["ezyVet", "ezyVet Enterprise - GP", "ezyVet Enterprise - Spec/ER"]

# Simulation parameters
PROJECTION_YEARS = 5
MONTE_CARLO_SIMULATIONS = 1000
CONFIDENCE_LEVEL = 0.90  # 90% confidence interval

# Rolling window for conversion rate calculation
CONVERSION_ROLLING_MONTHS = 18

# Snowflake connection - set via Streamlit secrets or environment
SNOWFLAKE_CONFIG_KEYS = [
    "account",
    "user",
    "password",
    "warehouse",
    "database",
    "schema",
    "role",
]
