from pathlib import Path
import pandas as pd
import numpy as np
import json
import gc


# ============================================================
# CONFIGURATION
# ============================================================

FORECAST_HORIZON = 28

VALIDATION_DAYS = 28

TEST_DAYS = 28


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

INPUT_FILE = (
    PROCESSED_DIR
    / "m5_ca1_dev_clean.parquet"
)

FEATURE_FILE = (
    PROCESSED_DIR
    / "m5_ca1_features.parquet"
)

TRAIN_FILE = (
    PROCESSED_DIR
    / "m5_ca1_train.parquet"
)

VALID_FILE = (
    PROCESSED_DIR
    / "m5_ca1_validation.parquet"
)

TEST_FILE = (
    PROCESSED_DIR
    / "m5_ca1_test.parquet"
)

FEATURE_LIST_FILE = (
    PROCESSED_DIR
    / "feature_columns.json"
)


# ============================================================
# START
# ============================================================

print("=" * 80)
print("STAGE 5 - TIME SERIES FEATURE ENGINEERING")
print("=" * 80)


# ============================================================
# STEP 1 - LOAD DATA
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"""
Clean dataset not found:

{INPUT_FILE}

Run Stage 3 first.
"""
    )


print("\nLoading clean dataset...")

df = pd.read_parquet(
    INPUT_FILE
)


df["date"] = pd.to_datetime(
    df["date"]
)


print("\nDataset loaded.")

print("\nShape:")
print(df.shape)

print("\nDate range:")

print(
    df["date"].min(),
    "to",
    df["date"].max()
)


print("\nUnique products:")

print(
    df["item_id"].nunique()
)


# ============================================================
# STEP 2 - SORT DATA
# ============================================================

print("\n" + "=" * 80)
print("STEP 2 - SORTING TIME SERIES")
print("=" * 80)


df = df.sort_values(
    [
        "store_id",
        "item_id",
        "date"
    ]
).reset_index(drop=True)


GROUP_COLS = [
    "store_id",
    "item_id"
]


# ============================================================
# STEP 3 - DATE FEATURES
# ============================================================

print("\n" + "=" * 80)
print("STEP 3 - CREATING DATE FEATURES")
print("=" * 80)


df["day_of_week"] = (
    df["date"]
    .dt.dayofweek
    .astype("int8")
)


df["day_of_month"] = (
    df["date"]
    .dt.day
    .astype("int8")
)


df["week_of_year"] = (
    df["date"]
    .dt.isocalendar()
    .week
    .astype("int16")
)


df["month_num"] = (
    df["date"]
    .dt.month
    .astype("int8")
)


df["quarter"] = (
    df["date"]
    .dt.quarter
    .astype("int8")
)


df["year_num"] = (
    df["date"]
    .dt.year
    .astype("int16")
)


df["is_weekend"] = (
    df["day_of_week"]
    .isin([5, 6])
    .astype("int8")
)


df["is_month_start"] = (
    df["date"]
    .dt.is_month_start
    .astype("int8")
)


df["is_month_end"] = (
    df["date"]
    .dt.is_month_end
    .astype("int8")
)


print("Date features created.")


# ============================================================
# STEP 4 - CYCLICAL DATE FEATURES
# ============================================================

print("\n" + "=" * 80)
print("STEP 4 - CREATING CYCLICAL FEATURES")
print("=" * 80)


# Day-of-week is cyclical.
# Sunday and Monday should be considered close.

df["dow_sin"] = np.sin(
    2
    * np.pi
    * df["day_of_week"]
    / 7
)

df["dow_cos"] = np.cos(
    2
    * np.pi
    * df["day_of_week"]
    / 7
)


df["month_sin"] = np.sin(
    2
    * np.pi
    * df["month_num"]
    / 12
)

df["month_cos"] = np.cos(
    2
    * np.pi
    * df["month_num"]
    / 12
)


# ============================================================
# STEP 5 - CREATE SALES LAG FEATURES
# ============================================================

print("\n" + "=" * 80)
print("STEP 5 - CREATING SALES LAGS")
print("=" * 80)


lag_days = [
    1,
    7,
    14,
    28,
    56
]


for lag in lag_days:

    print(
        f"Creating lag_{lag}..."
    )

    df[
        f"lag_{lag}"
    ] = (
        df.groupby(
            GROUP_COLS,
            observed=True
        )["sales"]
        .shift(lag)
    )


# ============================================================
# STEP 6 - ROLLING DEMAND FEATURES
# ============================================================

print("\n" + "=" * 80)
print("STEP 6 - CREATING ROLLING FEATURES")
print("=" * 80)


# IMPORTANT:
#
# Shift by 1 before rolling.
#
# This ensures today's sales are NOT used
# to predict today's sales.


grouped_sales = (
    df.groupby(
        GROUP_COLS,
        observed=True
    )["sales"]
)


shifted_sales = (
    grouped_sales
    .shift(1)
)


rolling_windows = [
    7,
    14,
    28,
    56
]


for window in rolling_windows:

    print(
        f"Creating rolling features: {window} days"
    )

    df[
        f"rolling_mean_{window}"
    ] = (
        shifted_sales
        .groupby(
            [
                df["store_id"],
                df["item_id"]
            ],
            observed=True
        )
        .transform(
            lambda x:
            x.rolling(
                window=window,
                min_periods=1
            ).mean()
        )
    )


    df[
        f"rolling_std_{window}"
    ] = (
        shifted_sales
        .groupby(
            [
                df["store_id"],
                df["item_id"]
            ],
            observed=True
        )
        .transform(
            lambda x:
            x.rolling(
                window=window,
                min_periods=2
            ).std()
        )
    )


# ============================================================
# STEP 7 - ROLLING MIN/MAX
# ============================================================

print("\n" + "=" * 80)
print("STEP 7 - CREATING ROLLING MIN/MAX")
print("=" * 80)


for window in [
    7,
    28
]:

    df[
        f"rolling_min_{window}"
    ] = (
        shifted_sales
        .groupby(
            [
                df["store_id"],
                df["item_id"]
            ],
            observed=True
        )
        .transform(
            lambda x:
            x.rolling(
                window=window,
                min_periods=1
            ).min()
        )
    )


    df[
        f"rolling_max_{window}"
    ] = (
        shifted_sales
        .groupby(
            [
                df["store_id"],
                df["item_id"]
            ],
            observed=True
        )
        .transform(
            lambda x:
            x.rolling(
                window=window,
                min_periods=1
            ).max()
        )
    )


# ============================================================
# STEP 8 - ZERO-DEMAND ROLLING FEATURE
# ============================================================

print("\n" + "=" * 80)
print("STEP 8 - CREATING INTERMITTENT DEMAND FEATURE")
print("=" * 80)


previous_zero_sales = (
    df.groupby(
        GROUP_COLS,
        observed=True
    )["zero_sales"]
    .shift(1)
)


df[
    "zero_sales_rate_28"
] = (
    previous_zero_sales
    .groupby(
        [
            df["store_id"],
            df["item_id"]
        ],
        observed=True
    )
    .transform(
        lambda x:
        x.rolling(
            window=28,
            min_periods=1
        ).mean()
    )
)


# ============================================================
# STEP 9 - PRICE FEATURES
# ============================================================

print("\n" + "=" * 80)
print("STEP 9 - CREATING PRICE FEATURES")
print("=" * 80)


previous_price = (
    df.groupby(
        GROUP_COLS,
        observed=True
    )["sell_price"]
    .shift(1)
)


df[
    "price_change"
] = (
    df["sell_price"]
    -
    previous_price
)


df[
    "price_change_pct"
] = (
    df["price_change"]
    /
    previous_price.replace(
        0,
        np.nan
    )
)


# Replace first price-change observation with 0.

df[
    "price_change"
] = (
    df["price_change"]
    .fillna(0)
)


df[
    "price_change_pct"
] = (
    df["price_change_pct"]
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
# STEP 10 - PRICE RELATIVE TO PRODUCT HISTORY
# ============================================================

print("\n" + "=" * 80)
print("STEP 10 - CREATING RELATIVE PRICE FEATURES")
print("=" * 80)


historical_price_mean = (
    df.groupby(
        GROUP_COLS,
        observed=True
    )["sell_price"]
    .transform(
        lambda x:
        x.expanding()
        .mean()
        .shift(1)
    )
)


df[
    "price_vs_historical_mean"
] = (
    df["sell_price"]
    /
    historical_price_mean
)


df[
    "price_vs_historical_mean"
] = (
    df[
        "price_vs_historical_mean"
    ]
    .replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )
    .fillna(1)
)


# ============================================================
# STEP 11 - PRODUCT AGE / DAYS SINCE LAUNCH
# ============================================================

print("\n" + "=" * 80)
print("STEP 11 - CREATING PRODUCT AGE")
print("=" * 80)


if (
    "first_available_date"
    in df.columns
):

    df[
        "first_available_date"
    ] = pd.to_datetime(
        df[
            "first_available_date"
        ]
    )

    df[
        "days_since_launch"
    ] = (
        df["date"]
        -
        df[
            "first_available_date"
        ]
    ).dt.days


else:

    first_date = (
        df.groupby(
            GROUP_COLS,
            observed=True
        )["date"]
        .transform("min")
    )

    df[
        "days_since_launch"
    ] = (
        df["date"]
        -
        first_date
    ).dt.days


df[
    "days_since_launch"
] = (
    df[
        "days_since_launch"
    ]
    .clip(lower=0)
    .astype("int32")
)


# ============================================================
# STEP 12 - DEMAND TREND FEATURES
# ============================================================

print("\n" + "=" * 80)
print("STEP 12 - CREATING DEMAND TREND FEATURES")
print("=" * 80)


df[
    "demand_trend_7_28"
] = (
    df[
        "rolling_mean_7"
    ]
    -
    df[
        "rolling_mean_28"
    ]
)


df[
    "demand_ratio_7_28"
] = (
    df[
        "rolling_mean_7"
    ]
    /
    (
        df[
            "rolling_mean_28"
        ]
        + 1e-6
    )
)


# ============================================================
# STEP 13 - VOLATILITY FEATURE
# ============================================================

print("\n" + "=" * 80)
print("STEP 13 - CREATING DEMAND VOLATILITY")
print("=" * 80)


df[
    "demand_cv_28"
] = (
    df[
        "rolling_std_28"
    ]
    /
    (
        df[
            "rolling_mean_28"
        ]
        + 1e-6
    )
)


df[
    "demand_cv_28"
] = (
    df[
        "demand_cv_28"
    ]
    .replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )
)


# ============================================================
# STEP 14 - CLEAN FEATURE INFINITIES
# ============================================================

print("\n" + "=" * 80)
print("STEP 14 - CLEANING FEATURE VALUES")
print("=" * 80)


df = df.replace(
    [
        np.inf,
        -np.inf
    ],
    np.nan
)


# ============================================================
# STEP 15 - REMOVE INITIAL ROWS WITHOUT FULL HISTORY
# ============================================================

print("\n" + "=" * 80)
print("STEP 15 - REMOVING ROWS WITHOUT REQUIRED HISTORY")
print("=" * 80)


rows_before = len(df)


# lag_28 is our minimum essential
# forecasting history.

df = df.dropna(
    subset=[
        "lag_28",
        "rolling_mean_28"
    ]
).copy()


rows_after = len(df)


print(
    "\nRows removed due to insufficient history:"
)

print(
    rows_before
    -
    rows_after
)


# Fill standard deviation features where
# limited history caused NaN.

std_columns = [
    column
    for column in df.columns
    if column.startswith(
        "rolling_std_"
    )
]


for column in std_columns:

    df[column] = (
        df[column]
        .fillna(0)
    )


df[
    "demand_cv_28"
] = (
    df[
        "demand_cv_28"
    ]
    .fillna(0)
)


# ============================================================
# STEP 16 - OPTIMIZE FEATURE DTYPES
# ============================================================

print("\n" + "=" * 80)
print("STEP 16 - OPTIMIZING DATA TYPES")
print("=" * 80)


float_columns = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "lag_56",

    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_mean_56",

    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28",
    "rolling_std_56",

    "rolling_min_7",
    "rolling_min_28",

    "rolling_max_7",
    "rolling_max_28",

    "zero_sales_rate_28",

    "price_change",
    "price_change_pct",

    "price_vs_historical_mean",

    "demand_trend_7_28",
    "demand_ratio_7_28",
    "demand_cv_28",

    "dow_sin",
    "dow_cos",

    "month_sin",
    "month_cos"
]


for column in float_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            downcast="float"
        )


# ============================================================
# STEP 17 - FINAL FEATURE LIST
# ============================================================

print("\n" + "=" * 80)
print("STEP 17 - DEFINING MODEL FEATURES")
print("=" * 80)


TARGET = "sales"


feature_columns = [

    # Historical demand
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "lag_56",

    # Rolling means
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_mean_56",

    # Rolling volatility
    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28",
    "rolling_std_56",

    # Rolling range
    "rolling_min_7",
    "rolling_min_28",
    "rolling_max_7",
    "rolling_max_28",

    # Intermittent demand
    "zero_sales_rate_28",

    # Price
    "sell_price",
    "price_change",
    "price_change_pct",
    "price_vs_historical_mean",

    # Calendar
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month_num",
    "quarter",
    "year_num",
    "is_weekend",
    "is_month_start",
    "is_month_end",

    # Cyclical
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",

    # External / event
    "snap",
    "has_event",

    # Product lifecycle
    "days_since_launch",

    # Demand dynamics
    "demand_trend_7_28",
    "demand_ratio_7_28",
    "demand_cv_28"
]


print(
    "\nNumber of numerical model features:"
)

print(
    len(feature_columns)
)


# ============================================================
# STEP 18 - SAVE FEATURE METADATA
# ============================================================

feature_metadata = {

    "target": TARGET,

    "numerical_features":
        feature_columns,

    "categorical_features": [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id"
    ],

    "forecast_horizon":
        FORECAST_HORIZON
}


with open(
    FEATURE_LIST_FILE,
    "w"
) as file:

    json.dump(
        feature_metadata,
        file,
        indent=4
    )


# ============================================================
# STEP 19 - CHECK FEATURE MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("STEP 19 - FEATURE QUALITY CHECK")
print("=" * 80)


feature_missing = (
    df[
        feature_columns
    ]
    .isnull()
    .sum()
    .sort_values(
        ascending=False
    )
)


print(
    "\nMissing values in model features:"
)

print(
    feature_missing[
        feature_missing > 0
    ]
)


# ============================================================
# STEP 20 - SAVE COMPLETE FEATURE DATASET
# ============================================================

print("\n" + "=" * 80)
print("STEP 20 - SAVING FEATURE DATASET")
print("=" * 80)


df.to_parquet(
    FEATURE_FILE,
    index=False
)


print(
    "\nFeature dataset saved:"
)

print(
    FEATURE_FILE
)


# ============================================================
# STEP 21 - TIME-BASED SPLIT
# ============================================================

print("\n" + "=" * 80)
print("STEP 21 - CREATING TIME-BASED SPLITS")
print("=" * 80)


max_date = (
    df["date"].max()
)


test_start_date = (
    max_date
    -
    pd.Timedelta(
        days=TEST_DAYS - 1
    )
)


validation_end_date = (
    test_start_date
    -
    pd.Timedelta(
        days=1
    )
)


validation_start_date = (
    validation_end_date
    -
    pd.Timedelta(
        days=VALIDATION_DAYS - 1
    )
)


print(
    "\nTraining period:"
)

print(
    df["date"].min(),
    "to",
    validation_start_date
    -
    pd.Timedelta(days=1)
)


print(
    "\nValidation period:"
)

print(
    validation_start_date,
    "to",
    validation_end_date
)


print(
    "\nTest period:"
)

print(
    test_start_date,
    "to",
    max_date
)


# ============================================================
# STEP 22 - CREATE SPLIT DATASETS
# ============================================================

train = df[
    df["date"]
    <
    validation_start_date
].copy()


validation = df[
    (
        df["date"]
        >= validation_start_date
    )
    &
    (
        df["date"]
        <= validation_end_date
    )
].copy()


test = df[
    df["date"]
    >= test_start_date
].copy()


# ============================================================
# STEP 23 - VERIFY SPLIT
# ============================================================

print("\nTraining shape:")
print(train.shape)

print("\nValidation shape:")
print(validation.shape)

print("\nTest shape:")
print(test.shape)


print(
    "\nTrain last date:"
)

print(
    train["date"].max()
)


print(
    "\nValidation first date:"
)

print(
    validation["date"].min()
)


print(
    "\nValidation last date:"
)

print(
    validation["date"].max()
)


print(
    "\nTest first date:"
)

print(
    test["date"].min()
)


# ============================================================
# STEP 24 - SAVE SPLITS
# ============================================================

print("\n" + "=" * 80)
print("STEP 24 - SAVING TRAIN / VALIDATION / TEST")
print("=" * 80)


train.to_parquet(
    TRAIN_FILE,
    index=False
)

validation.to_parquet(
    VALID_FILE,
    index=False
)

test.to_parquet(
    TEST_FILE,
    index=False
)


print("\nFiles saved successfully:")

print(
    TRAIN_FILE
)

print(
    VALID_FILE
)

print(
    TEST_FILE
)


# ============================================================
# STEP 25 - MEMORY SUMMARY
# ============================================================

feature_memory_mb = (
    df.memory_usage(
        deep=True
    ).sum()
    /
    1024**2
)


print(
    f"\nFeature dataset memory: "
    f"{feature_memory_mb:.2f} MB"
)


# ============================================================
# STEP 26 - SAMPLE FEATURES
# ============================================================

print("\n" + "=" * 80)
print("FEATURE SAMPLE")
print("=" * 80)


sample_columns = [

    "date",
    "item_id",
    "sales",

    "lag_1",
    "lag_7",
    "lag_28",

    "rolling_mean_7",
    "rolling_mean_28",

    "sell_price",
    "price_change_pct",

    "snap",
    "has_event",

    "days_since_launch"
]


print(
    df[
        sample_columns
    ].head(20)
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("STAGE 5 COMPLETED SUCCESSFULLY")
print("=" * 80)


del train
del validation
del test
del df

gc.collect()