from pathlib import Path
import warnings
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

HISTORICAL_WINDOW_DAYS = 90

REVIEW_PERIOD_DAYS = 7

DEFAULT_ORDERING_COST = 50.0

DEFAULT_HOLDING_RATE = 0.20


# ABC service-level policy
SERVICE_LEVELS = {
    "A": 0.98,
    "B": 0.95,
    "C": 0.90
}


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


EXTERNAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
)


TABLE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "tables"
)


FIGURE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)


EXTERNAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT FILES
# ============================================================

FORECAST_FILE = (
    PROCESSED_DIR
    / "stage8_final_test_predictions.parquet"
)


CLEAN_DATA_FILE = (
    PROCESSED_DIR
    / "m5_ca1_dev_clean.parquet"
)


REAL_INVENTORY_FILE = (
    EXTERNAL_DIR
    / "inventory_snapshot.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

SIMULATED_INVENTORY_FILE = (
    EXTERNAL_DIR
    / "inventory_snapshot_simulated.csv"
)


INVENTORY_TEMPLATE_FILE = (
    EXTERNAL_DIR
    / "inventory_snapshot_template.csv"
)


FINAL_OUTPUT_PARQUET = (
    PROCESSED_DIR
    / "stage9_inventory_optimization.parquet"
)


FINAL_OUTPUT_CSV = (
    TABLE_DIR
    / "stage9_inventory_optimization.csv"
)


SUMMARY_FILE = (
    TABLE_DIR
    / "stage9_inventory_summary.csv"
)


RISK_SUMMARY_FILE = (
    TABLE_DIR
    / "stage9_stockout_risk_summary.csv"
)


ABC_XYZ_FILE = (
    TABLE_DIR
    / "stage9_abc_xyz_summary.csv"
)


# ============================================================
# START
# ============================================================

print("=" * 90)

print(
    "STAGE 9 - INVENTORY OPTIMIZATION "
    "AND STOCKOUT RISK INTELLIGENCE"
)

print("=" * 90)


# ============================================================
# STEP 1 - CHECK INPUT FILES
# ============================================================

for file in [
    FORECAST_FILE,
    CLEAN_DATA_FILE
]:

    if not file.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n"
            f"{file}\n\n"
            "Complete Stage 8 first."
        )


# ============================================================
# STEP 2 - LOAD FORECASTS
# ============================================================

print("\nLoading Stage 8 forecasts...")


forecast = pd.read_parquet(
    FORECAST_FILE
)


forecast["date"] = pd.to_datetime(
    forecast["date"]
)


forecast = forecast.sort_values(
    [
        "store_id",
        "item_id",
        "date"
    ]
).reset_index(drop=True)


print("\nForecast shape:")

print(
    forecast.shape
)


print("\nForecast period:")

print(
    forecast["date"].min(),
    "to",
    forecast["date"].max()
)


print("\nUnique products:")

print(
    forecast["item_id"].nunique()
)


# ============================================================
# STEP 3 - LOAD CLEAN HISTORICAL DATA
# ============================================================

print("\nLoading historical data...")


historical = pd.read_parquet(
    CLEAN_DATA_FILE
)


historical["date"] = pd.to_datetime(
    historical["date"]
)


test_start_date = (
    forecast["date"].min()
)


# IMPORTANT:
# Only use history before the final test period.
#
# We do not use actual future test demand
# in inventory calculations.

historical = historical[
    historical["date"]
    <
    test_start_date
].copy()


historical = historical.sort_values(
    [
        "store_id",
        "item_id",
        "date"
    ]
)


print("\nHistorical data ends:")

print(
    historical["date"].max()
)


# ============================================================
# STEP 4 - PRODUCT METADATA
# ============================================================

print("\n" + "=" * 90)

print(
    "STEP 4 - PREPARING PRODUCT METADATA"
)

print("=" * 90)


metadata = (

    historical

    .sort_values("date")

    .groupby(
        [
            "store_id",
            "item_id"
        ],
        observed=True
    )

    .tail(1)

    [
        [
            "store_id",
            "item_id",
            "dept_id",
            "cat_id",
            "state_id",
            "sell_price"
        ]
    ]

    .copy()
)


metadata = metadata.rename(
    columns={
        "sell_price":
        "current_sell_price"
    }
)


# ============================================================
# STEP 5 - FORECAST SUMMARY
# ============================================================

print("\n" + "=" * 90)

print(
    "STEP 5 - SUMMARIZING FORECAST DEMAND"
)

print("=" * 90)


forecast_summary = (

    forecast

    .groupby(
        [
            "store_id",
            "item_id"
        ],
        observed=True
    )

    .agg(

        forecast_total_28=(
            "prediction",
            "sum"
        ),

        forecast_avg_daily=(
            "prediction",
            "mean"
        ),

        forecast_max_daily=(
            "prediction",
            "max"
        ),

        forecast_std_daily=(
            "prediction",
            "std"
        )
    )

    .reset_index()
)


forecast_summary[
    "forecast_std_daily"
] = (

    forecast_summary[
        "forecast_std_daily"
    ]

    .fillna(0)
)


forecast_summary[
    "annual_demand_estimate"
] = (

    forecast_summary[
        "forecast_avg_daily"
    ]

    * 365
)


# ============================================================
# STEP 6 - HISTORICAL DEMAND STATISTICS
# ============================================================

print("\n" + "=" * 90)

print(
    "STEP 6 - CALCULATING HISTORICAL "
    "DEMAND UNCERTAINTY"
)

print("=" * 90)


history_cutoff = (

    historical["date"].max()

    -
    pd.Timedelta(
        days=
        HISTORICAL_WINDOW_DAYS
        -
        1
    )
)


recent_history = historical[
    historical["date"]
    >=
    history_cutoff
].copy()


historical_stats = (

    recent_history

    .groupby(
        [
            "store_id",
            "item_id"
        ],
        observed=True
    )

    .agg(

        historical_avg_daily=(
            "sales",
            "mean"
        ),

        historical_std_daily=(
            "sales",
            "std"
        ),

        historical_zero_rate=(
            "zero_sales",
            "mean"
        ),

        historical_max_daily=(
            "sales",
            "max"
        )
    )

    .reset_index()
)


historical_stats[
    "historical_std_daily"
] = (

    historical_stats[
        "historical_std_daily"
    ]

    .fillna(0)
)


historical_stats[
    "demand_cv"
] = (

    historical_stats[
        "historical_std_daily"
    ]

    /

    (
        historical_stats[
            "historical_avg_daily"
        ]

        + 1e-6
    )
)


# ============================================================
# STEP 7 - BUILD BASE INVENTORY TABLE
# ============================================================

inventory = (

    forecast_summary

    .merge(
        historical_stats,
        on=[
            "store_id",
            "item_id"
        ],
        how="left"
    )

    .merge(
        metadata,
        on=[
            "store_id",
            "item_id"
        ],
        how="left"
    )
)


inventory[
    "historical_avg_daily"
] = (

    inventory[
        "historical_avg_daily"
    ]

    .fillna(
        inventory[
            "forecast_avg_daily"
        ]
    )
)


inventory[
    "historical_std_daily"
] = (

    inventory[
        "historical_std_daily"
    ]

    .fillna(
        inventory[
            "forecast_std_daily"
        ]
    )
)


inventory[
    "historical_zero_rate"
] = (

    inventory[
        "historical_zero_rate"
    ]

    .fillna(0)
)


inventory[
    "demand_cv"
] = (

    inventory[
        "demand_cv"
    ]

    .replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    .fillna(0)
)


# ============================================================
# STEP 8 - ABC CLASSIFICATION
# ============================================================

print("\n" + "=" * 90)

print(
    "STEP 8 - ABC INVENTORY CLASSIFICATION"
)

print("=" * 90)


inventory[
    "estimated_annual_revenue"
] = (

    inventory[
        "annual_demand_estimate"
    ]

    *
    inventory[
        "current_sell_price"
    ]
)


inventory = inventory.sort_values(
    "estimated_annual_revenue",
    ascending=False
).reset_index(drop=True)


total_revenue = (

    inventory[
        "estimated_annual_revenue"
    ]
    .sum()
)


if total_revenue > 0:

    inventory[
        "revenue_cumulative_share"
    ] = (

        inventory[
            "estimated_annual_revenue"
        ]
        .cumsum()

        /
        total_revenue
    )

else:

    inventory[
        "revenue_cumulative_share"
    ] = 0


def abc_classification(
    cumulative_share
):

    if cumulative_share <= 0.80:

        return "A"

    elif cumulative_share <= 0.95:

        return "B"

    return "C"


inventory[
    "abc_class"
] = (

    inventory[
        "revenue_cumulative_share"
    ]

    .apply(
        abc_classification
    )
)


print("\nABC counts:")

print(
    inventory[
        "abc_class"
    ]
    .value_counts()
)


# ============================================================
# STEP 9 - XYZ CLASSIFICATION
# ============================================================

print("\n" + "=" * 90)

print(
    "STEP 9 - XYZ DEMAND VARIABILITY "
    "CLASSIFICATION"
)

print("=" * 90)


def xyz_classification(
    cv
):

    if cv < 0.5:

        return "X"

    elif cv < 1.0:

        return "Y"

    return "Z"


inventory[
    "xyz_class"
] = (

    inventory[
        "demand_cv"
    ]

    .apply(
        xyz_classification
    )
)


inventory[
    "abc_xyz_class"
] = (

    inventory[
        "abc_class"
    ]

    +

    inventory[
        "xyz_class"
    ]
)


print("\nABC-XYZ classes:")

print(
    inventory[
        "abc_xyz_class"
    ]
    .value_counts()
)


# ============================================================
# STEP 10 - SERVICE LEVEL
# ============================================================

inventory[
    "service_level"
] = (

    inventory[
        "abc_class"
    ]

    .map(
        SERVICE_LEVELS
    )
)


inventory[
    "z_score"
] = (

    inventory[
        "service_level"
    ]

    .apply(
        norm.ppf
    )
)


# ============================================================
# STEP 11 - CREATE INVENTORY INPUT TEMPLATE
# ============================================================

template = inventory[
    [
        "store_id",
        "item_id"
    ]
].copy()


template[
    "current_inventory"
] = np.nan


template[
    "incoming_inventory"
] = np.nan


template[
    "lead_time_days"
] = np.nan


template[
    "ordering_cost"
] = DEFAULT_ORDERING_COST


template[
    "holding_rate"
] = DEFAULT_HOLDING_RATE


template.to_csv(
    INVENTORY_TEMPLATE_FILE,
    index=False
)


# ============================================================
# STEP 12 - LOAD REAL OR SIMULATED INVENTORY
# ============================================================

print("\n" + "=" * 90)

print(
    "STEP 12 - INVENTORY SNAPSHOT"
)

print("=" * 90)


if REAL_INVENTORY_FILE.exists():

    print(
        "\nUsing REAL inventory snapshot:"
    )

    print(
        REAL_INVENTORY_FILE
    )


    inventory_snapshot = pd.read_csv(
        REAL_INVENTORY_FILE
    )


    required_inventory_columns = [
        "store_id",
        "item_id",
        "current_inventory",
        "incoming_inventory",
        "lead_time_days",
        "ordering_cost",
        "holding_rate"
    ]


    missing_columns = [

        column

        for column
        in required_inventory_columns

        if column
        not in
        inventory_snapshot.columns
    ]


    if missing_columns:

        raise ValueError(
            "Inventory file is missing "
            f"columns: {missing_columns}"
        )


    inventory_source = "REAL"


else:

    print(
        "\nNo real inventory file found."
    )

    print(
        "Creating a deterministic "
        "SIMULATED inventory scenario."
    )


    rng = np.random.default_rng(
        RANDOM_STATE
    )


    # forecast_avg_daily is needed temporarily
    # for generating simulated inventory.
    inventory_snapshot = inventory[
        [
            "store_id",
            "item_id",
            "forecast_avg_daily"
        ]
    ].copy()


    inventory_snapshot[
        "simulated_days_of_cover"
    ] = rng.integers(
        low=2,
        high=15,
        size=len(inventory_snapshot)
    )


    inventory_snapshot[
        "current_inventory"
    ] = np.ceil(

        inventory_snapshot[
            "forecast_avg_daily"
        ]

        *

        inventory_snapshot[
            "simulated_days_of_cover"
        ]

    ).astype(int)


    incoming_days = rng.choice(
        [
            0,
            0,
            0,
            2,
            4
        ],
        size=len(inventory_snapshot)
    )


    inventory_snapshot[
        "incoming_inventory"
    ] = np.ceil(

        inventory_snapshot[
            "forecast_avg_daily"
        ]

        *

        incoming_days

    ).astype(int)


    inventory_snapshot[
        "lead_time_days"
    ] = rng.integers(
        low=3,
        high=11,
        size=len(inventory_snapshot)
    )


    inventory_snapshot[
        "ordering_cost"
    ] = DEFAULT_ORDERING_COST


    inventory_snapshot[
        "holding_rate"
    ] = DEFAULT_HOLDING_RATE


    inventory_snapshot.to_csv(
        SIMULATED_INVENTORY_FILE,
        index=False
    )


    inventory_source = "SIMULATED"


# ============================================================
# IMPORTANT FIX
# ============================================================

# forecast_avg_daily already exists in the main inventory
# dataframe. Drop the temporary copy before merging.

if "forecast_avg_daily" in inventory_snapshot.columns:

    inventory_snapshot = inventory_snapshot.drop(
        columns=[
            "forecast_avg_daily"
        ]
    )


# ============================================================
# MERGE INVENTORY INFORMATION
# ============================================================

inventory = inventory.merge(

    inventory_snapshot,

    on=[
        "store_id",
        "item_id"
    ],

    how="left",

    validate="one_to_one"
)


inventory[
    "inventory_source"
] = inventory_source


# ============================================================
# STEP 13 - FORECAST LOOKUP
# ============================================================

print("\n" + "=" * 90)

print(
    "STEP 13 - CALCULATING LEAD-TIME "
    "FORECAST DEMAND"
)

print("=" * 90)


forecast_lookup = {}


for key, group in forecast.groupby(
    [
        "store_id",
        "item_id"
    ],
    observed=True
):

    group = group.sort_values(
        "date"
    )


    forecast_lookup[
        key
    ] = (
        group[
            "prediction"
        ]
        .astype(float)
        .tolist()
    )


def lead_time_forecast(
    row
):

    key = (
        row["store_id"],
        row["item_id"]
    )


    values = forecast_lookup.get(
        key,
        []
    )


    lead_time = int(
        row[
            "lead_time_days"
        ]
    )


    if not values:

        return 0.0


    if lead_time <= len(values):

        return float(
            np.sum(
                values[
                    :lead_time
                ]
            )
        )


    # In case lead time exceeds forecast horizon,
    # extrapolate using average forecast demand.

    available_forecast = float(
        np.sum(values)
    )


    extra_days = (
        lead_time
        -
        len(values)
    )


    extra_demand = (
        extra_days
        *
        np.mean(values)
    )


    return (
        available_forecast
        +
        extra_demand
    )


inventory[
    "lead_time_demand_forecast"
] = (

    inventory.apply(
        lead_time_forecast,
        axis=1
    )
)


# ============================================================
# STEP 14 - LEAD-TIME UNCERTAINTY
# ============================================================

inventory[
    "lead_time_demand_std"
] = (

    inventory[
        "historical_std_daily"
    ]

    *

    np.sqrt(
        inventory[
            "lead_time_days"
        ]
    )
)


# ============================================================
# STEP 15 - SAFETY STOCK
# ============================================================

print("\n" + "=" * 90)

print(
    "STEP 15 - CALCULATING SAFETY STOCK"
)

print("=" * 90)


inventory[
    "safety_stock"
] = (

    inventory[
        "z_score"
    ]

    *

    inventory[
        "lead_time_demand_std"
    ]
)


inventory[
    "safety_stock"
] = (

    np.ceil(
        inventory[
            "safety_stock"
        ]
    )

    .clip(
        lower=0
    )
)


# ============================================================
# STEP 16 - REORDER POINT
# ============================================================

inventory[
    "reorder_point"
] = (

    inventory[
        "lead_time_demand_forecast"
    ]

    +

    inventory[
        "safety_stock"
    ]
)


inventory[
    "reorder_point"
] = np.ceil(
    inventory[
        "reorder_point"
    ]
)


# ============================================================
# STEP 17 - INVENTORY POSITION
# ============================================================

inventory[
    "inventory_position"
] = (

    inventory[
        "current_inventory"
    ]

    +

    inventory[
        "incoming_inventory"
    ]
)


# ============================================================
# STEP 18 - STOCKOUT PROBABILITY
# ============================================================

print("\n" + "=" * 90)

print(
    "STEP 18 - CALCULATING "
    "STOCKOUT PROBABILITY"
)

print("=" * 90)


def stockout_probability(
    row
):

    mean_demand = (
        row[
            "lead_time_demand_forecast"
        ]
    )


    demand_std = (
        row[
            "lead_time_demand_std"
        ]
    )


    inventory_position = (
        row[
            "inventory_position"
        ]
    )


    if demand_std <= 1e-8:

        if (
            inventory_position
            <
            mean_demand
        ):

            return 1.0

        return 0.0


    z = (

        inventory_position
        -
        mean_demand

    ) / demand_std


    probability = (
        1
        -
        norm.cdf(z)
    )


    return float(
        np.clip(
            probability,
            0,
            1
        )
    )


inventory[
    "stockout_probability"
] = (

    inventory.apply(
        stockout_probability,
        axis=1
    )
)


inventory[
    "stockout_probability_pct"
] = (

    inventory[
        "stockout_probability"
    ]

    * 100
)


# ============================================================
# STEP 19 - RISK LEVEL
# ============================================================

def risk_level(
    probability
):

    if probability < 0.20:

        return "LOW"

    elif probability < 0.50:

        return "MEDIUM"

    return "HIGH"


inventory[
    "stockout_risk"
] = (

    inventory[
        "stockout_probability"
    ]

    .apply(
        risk_level
    )
)


# ============================================================
# STEP 20 - ECONOMIC ORDER QUANTITY
# ============================================================

print("\n" + "=" * 90)

print(
    "STEP 20 - CALCULATING EOQ"
)

print("=" * 90)


inventory[
    "annual_holding_cost_per_unit"
] = (

    inventory[
        "current_sell_price"
    ]

    *

    inventory[
        "holding_rate"
    ]
)


inventory[
    "annual_holding_cost_per_unit"
] = (

    inventory[
        "annual_holding_cost_per_unit"
    ]

    .replace(
        0,
        np.nan
    )
)


inventory[
    "eoq"
] = np.sqrt(

    (
        2
        *
        inventory[
            "annual_demand_estimate"
        ]
        *
        inventory[
            "ordering_cost"
        ]
    )

    /

    inventory[
        "annual_holding_cost_per_unit"
    ]
)


inventory[
    "eoq"
] = (

    inventory[
        "eoq"
    ]

    .replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    .fillna(0)
)


inventory[
    "eoq"
] = np.ceil(
    inventory[
        "eoq"
    ]
)


# ============================================================
# STEP 21 - ORDER-UP-TO LEVEL
# ============================================================

inventory[
    "review_period_demand"
] = (

    inventory[
        "forecast_avg_daily"
    ]

    *
    REVIEW_PERIOD_DAYS
)


inventory[
    "order_up_to_level"
] = (

    inventory[
        "reorder_point"
    ]

    +

    inventory[
        "review_period_demand"
    ]
)


inventory[
    "order_up_to_level"
] = np.ceil(
    inventory[
        "order_up_to_level"
    ]
)


# ============================================================
# STEP 22 - REORDER DECISION
# ============================================================

inventory[
    "reorder_required"
] = (

    inventory[
        "inventory_position"
    ]

    <=

    inventory[
        "reorder_point"
    ]
)


# ============================================================
# STEP 23 - RECOMMENDED ORDER QUANTITY
# ============================================================

def recommended_order(
    row
):

    if not row[
        "reorder_required"
    ]:

        return 0


    shortage_replenishment = (

        row[
            "order_up_to_level"
        ]

        -

        row[
            "inventory_position"
        ]
    )


    quantity = max(

        row[
            "eoq"
        ],

        shortage_replenishment,

        0
    )


    return int(
        math.ceil(
            quantity
        )
    )


inventory[
    "recommended_order_qty"
] = (

    inventory.apply(
        recommended_order,
        axis=1
    )
)


# ============================================================
# STEP 24 - DAYS OF COVER
# ============================================================

inventory[
    "days_of_cover"
] = (

    inventory[
        "inventory_position"
    ]

    /

    (
        inventory[
            "forecast_avg_daily"
        ]

        + 1e-6
    )
)


# ============================================================
# STEP 25 - ACTION RECOMMENDATION
# ============================================================

def action_recommendation(
    row
):

    if (
        row[
            "stockout_risk"
        ]
        ==
        "HIGH"
    ):

        return "URGENT REPLENISHMENT"


    if (
        row[
            "stockout_risk"
        ]
        ==
        "MEDIUM"
    ):

        return "REVIEW AND REORDER"


    if row[
        "reorder_required"
    ]:

        return "REORDER"


    return "NO ACTION"


inventory[
    "recommended_action"
] = (

    inventory.apply(
        action_recommendation,
        axis=1
    )
)


# ============================================================
# STEP 26 - PRIORITY SCORE
# ============================================================

# Priority combines:
# - stockout probability
# - ABC importance
# - demand volatility

abc_weight = {
    "A": 3,
    "B": 2,
    "C": 1
}


inventory[
    "abc_weight"
] = (

    inventory[
        "abc_class"
    ]

    .map(
        abc_weight
    )
)


inventory[
    "priority_score"
] = (

    inventory[
        "stockout_probability"
    ]
    *
    60

    +

    inventory[
        "abc_weight"
    ]
    *
    10

    +

    np.minimum(
        inventory[
            "demand_cv"
        ],
        3
    )
    *
    10
)


inventory = inventory.sort_values(
    "priority_score",
    ascending=False
).reset_index(drop=True)


# ============================================================
# STEP 27 - SUMMARY
# ============================================================

print("\n" + "=" * 90)

print(
    "STEP 27 - INVENTORY OPTIMIZATION SUMMARY"
)

print("=" * 90)


total_items = len(
    inventory
)


high_risk_items = (

    inventory[
        "stockout_risk"
    ]

    .eq(
        "HIGH"
    )

    .sum()
)


medium_risk_items = (

    inventory[
        "stockout_risk"
    ]

    .eq(
        "MEDIUM"
    )

    .sum()
)


low_risk_items = (

    inventory[
        "stockout_risk"
    ]

    .eq(
        "LOW"
    )

    .sum()
)


reorder_items = (

    inventory[
        "reorder_required"
    ]

    .sum()
)


total_order_units = (

    inventory[
        "recommended_order_qty"
    ]

    .sum()
)


summary = pd.DataFrame(
    {
        "metric": [

            "inventory_source",

            "total_products",

            "high_risk_products",

            "medium_risk_products",

            "low_risk_products",

            "products_requiring_reorder",

            "total_recommended_order_units",

            "average_stockout_probability_pct",

            "average_days_of_cover"
        ],

        "value": [

            inventory_source,

            total_items,

            high_risk_items,

            medium_risk_items,

            low_risk_items,

            reorder_items,

            total_order_units,

            round(
                inventory[
                    "stockout_probability_pct"
                ].mean(),
                2
            ),

            round(
                inventory[
                    "days_of_cover"
                ].mean(),
                2
            )
        ]
    }
)


print(
    summary
)


# ============================================================
# STEP 28 - SAVE RESULTS
# ============================================================

inventory.to_parquet(
    FINAL_OUTPUT_PARQUET,
    index=False
)


inventory.to_csv(
    FINAL_OUTPUT_CSV,
    index=False
)


summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# STEP 29 - RISK SUMMARY
# ============================================================

risk_summary = (

    inventory

    .groupby(
        "stockout_risk",
        observed=True
    )

    .agg(

        products=(
            "item_id",
            "count"
        ),

        average_stockout_probability=(
            "stockout_probability_pct",
            "mean"
        ),

        total_recommended_order_qty=(
            "recommended_order_qty",
            "sum"
        ),

        average_days_of_cover=(
            "days_of_cover",
            "mean"
        )
    )

    .reset_index()
)


risk_summary.to_csv(
    RISK_SUMMARY_FILE,
    index=False
)


# ============================================================
# STEP 30 - ABC XYZ SUMMARY
# ============================================================

abc_xyz_summary = (

    inventory

    .groupby(
        "abc_xyz_class",
        observed=True
    )

    .agg(

        products=(
            "item_id",
            "count"
        ),

        average_daily_forecast=(
            "forecast_avg_daily",
            "mean"
        ),

        average_stockout_probability=(
            "stockout_probability_pct",
            "mean"
        ),

        total_order_quantity=(
            "recommended_order_qty",
            "sum"
        )
    )

    .reset_index()
)


abc_xyz_summary.to_csv(
    ABC_XYZ_FILE,
    index=False
)


# ============================================================
# STEP 31 - STOCKOUT RISK CHART
# ============================================================

risk_order = [
    "LOW",
    "MEDIUM",
    "HIGH"
]


risk_counts = (

    inventory[
        "stockout_risk"
    ]

    .value_counts()

    .reindex(
        risk_order,
        fill_value=0
    )
)


plt.figure(
    figsize=(8, 6)
)


plt.bar(
    risk_counts.index,
    risk_counts.values
)


plt.title(
    "Stockout Risk Distribution"
)


plt.xlabel(
    "Risk Level"
)


plt.ylabel(
    "Number of Products"
)


plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    / "29_stockout_risk_distribution.png",
    dpi=300
)


plt.close()


# ============================================================
# STEP 32 - TOP REPLENISHMENT ITEMS
# ============================================================

top_replenishment = (

    inventory[
        inventory[
            "recommended_order_qty"
        ]
        >
        0
    ]

    .nlargest(
        15,
        "recommended_order_qty"
    )
)


if not top_replenishment.empty:

    plt.figure(
        figsize=(11, 8)
    )


    plot_data = (

        top_replenishment

        .sort_values(
            "recommended_order_qty"
        )
    )


    plt.barh(

        plot_data[
            "item_id"
        ],

        plot_data[
            "recommended_order_qty"
        ]
    )


    plt.title(
        "Top 15 Recommended "
        "Replenishment Quantities"
    )


    plt.xlabel(
        "Recommended Order Quantity"
    )


    plt.ylabel(
        "Product"
    )


    plt.tight_layout()


    plt.savefig(
        FIGURE_DIR
        / "30_top_replenishment_items.png",
        dpi=300
    )


    plt.close()


# ============================================================
# STEP 33 - TOP PRIORITY PRODUCTS
# ============================================================

top_priority_columns = [

    "store_id",

    "item_id",

    "cat_id",

    "abc_xyz_class",

    "forecast_total_28",

    "current_inventory",

    "incoming_inventory",

    "lead_time_days",

    "safety_stock",

    "reorder_point",

    "inventory_position",

    "stockout_probability_pct",

    "stockout_risk",

    "eoq",

    "recommended_order_qty",

    "recommended_action",

    "priority_score"
]


print("\nTop 20 priority products:")


print(

    inventory[
        top_priority_columns
    ]

    .head(20)

    .to_string(
        index=False
    )
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 90)

print(
    "STAGE 9 COMPLETED SUCCESSFULLY"
)

print("=" * 90)


print(
    "\nInventory source:"
)

print(
    inventory_source
)


print(
    "\nProducts analyzed:"
)

print(
    total_items
)


print(
    "\nHigh stockout risk:"
)

print(
    high_risk_items
)


print(
    "\nProducts requiring reorder:"
)

print(
    reorder_items
)


print(
    "\nTotal recommended order units:"
)

print(
    total_order_units
)


print(
    "\nFinal optimization dataset:"
)

print(
    FINAL_OUTPUT_CSV
)