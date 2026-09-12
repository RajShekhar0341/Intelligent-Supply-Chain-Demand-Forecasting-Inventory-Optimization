from pathlib import Path
import pandas as pd
import numpy as np
import gc


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = (
    PROCESSED_DIR
    / "m5_ca1_dev_integrated.parquet"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "m5_ca1_dev_clean.parquet"
)

QUALITY_REPORT_FILE = (
    PROCESSED_DIR
    / "stage3_data_quality_report.csv"
)


# ============================================================
# START
# ============================================================

print("=" * 80)
print("STAGE 3 - DATA CLEANING AND QUALITY ENGINEERING")
print("=" * 80)


# ============================================================
# STEP 1 - CHECK INPUT FILE
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"""
Stage 2 processed file not found:

{INPUT_FILE}

Run Stage 2 first.
"""
    )


print("\nLoading dataset...")

data = pd.read_parquet(INPUT_FILE)

original_rows = len(data)

print("\nDataset loaded successfully.")

print("\nOriginal shape:")
print(data.shape)

print("\nColumns:")
print(data.columns.tolist())


# ============================================================
# STEP 2 - CHECK REQUIRED COLUMNS
# ============================================================

print("\n" + "=" * 80)
print("STEP 2 - CHECKING REQUIRED COLUMNS")
print("=" * 80)

required_columns = [
    "date",
    "item_id",
    "store_id",
    "state_id",
    "sales",
    "sell_price",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap"
]

missing_required_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_required_columns:

    raise ValueError(
        f"Missing required columns: "
        f"{missing_required_columns}"
    )

print("\nAll required columns are available.")


# ============================================================
# STEP 3 - STANDARDIZE DATE
# ============================================================

print("\n" + "=" * 80)
print("STEP 3 - STANDARDIZING DATE")
print("=" * 80)

data["date"] = pd.to_datetime(
    data["date"],
    errors="coerce"
)

invalid_dates = data["date"].isna().sum()

print("\nInvalid dates:")
print(invalid_dates)

if invalid_dates > 0:

    data = data[
        data["date"].notna()
    ].copy()


# ============================================================
# STEP 4 - REMOVE MISSING KEY IDENTIFIERS
# ============================================================

print("\n" + "=" * 80)
print("STEP 4 - CHECKING KEY IDENTIFIERS")
print("=" * 80)

key_columns = [
    "date",
    "item_id",
    "store_id"
]

missing_key_rows = (
    data[key_columns]
    .isnull()
    .any(axis=1)
    .sum()
)

print("\nRows with missing key identifiers:")
print(missing_key_rows)

data = data.dropna(
    subset=key_columns
).copy()


# ============================================================
# STEP 5 - DUPLICATE CHECK
# ============================================================

print("\n" + "=" * 80)
print("STEP 5 - REMOVING DUPLICATES")
print("=" * 80)

duplicate_subset = [
    "date",
    "item_id",
    "store_id"
]

duplicate_count = data.duplicated(
    subset=duplicate_subset
).sum()

print("\nDuplicate rows found:")
print(duplicate_count)

if duplicate_count > 0:

    data = data.drop_duplicates(
        subset=duplicate_subset,
        keep="first"
    ).copy()

print("\nRemaining duplicates:")

print(
    data.duplicated(
        subset=duplicate_subset
    ).sum()
)


# ============================================================
# STEP 6 - CLEAN SALES COLUMN
# ============================================================

print("\n" + "=" * 80)
print("STEP 6 - CLEANING SALES VALUES")
print("=" * 80)

data["sales"] = pd.to_numeric(
    data["sales"],
    errors="coerce"
)

missing_sales = data["sales"].isna().sum()

negative_sales = (
    data["sales"] < 0
).sum()

print("\nMissing sales:")
print(missing_sales)

print("\nNegative sales:")
print(negative_sales)


# Negative demand is invalid.

data.loc[
    data["sales"] < 0,
    "sales"
] = np.nan


sales_rows_before = len(data)

data = data.dropna(
    subset=["sales"]
).copy()

invalid_sales_removed = (
    sales_rows_before
    - len(data)
)


print("\nInvalid sales rows removed:")
print(invalid_sales_removed)


# Downcast to save memory

data["sales"] = pd.to_numeric(
    data["sales"],
    downcast="integer"
)


# ============================================================
# STEP 7 - CLEAN SELL PRICE
# ============================================================

print("\n" + "=" * 80)
print("STEP 7 - CHECKING SELL PRICE")
print("=" * 80)

data["sell_price"] = pd.to_numeric(
    data["sell_price"],
    errors="coerce"
)


# Save information about whether price
# was originally missing.

data["price_missing_original"] = (
    data["sell_price"].isna()
).astype("int8")


invalid_price_count = (
    data["sell_price"] <= 0
).sum()


print("\nInvalid prices <= 0:")
print(invalid_price_count)


# Price cannot be negative or zero.

data.loc[
    data["sell_price"] <= 0,
    "sell_price"
] = np.nan


print("\nMissing prices before cleaning:")

print(
    data["sell_price"]
    .isna()
    .sum()
)


# ============================================================
# STEP 8 - IDENTIFY PRODUCT AVAILABILITY
# ============================================================

print("\n" + "=" * 80)
print("STEP 8 - IDENTIFYING PRODUCT AVAILABILITY")
print("=" * 80)


# Find first date on which each product
# has a valid selling price.

first_price_date = (
    data[
        data["sell_price"].notna()
    ]
    .groupby(
        [
            "store_id",
            "item_id"
        ]
    )["date"]
    .min()
    .reset_index()
    .rename(
        columns={
            "date":
            "first_available_date"
        }
    )
)


data = data.merge(
    first_price_date,
    how="left",
    on=[
        "store_id",
        "item_id"
    ]
)


products_without_price = (
    data[
        "first_available_date"
    ]
    .isna()
    .sum()
)


print(
    "\nRows belonging to products "
    "with no price information:"
)

print(products_without_price)


# ============================================================
# STEP 9 - REMOVE PRE-LAUNCH PERIOD
# ============================================================

print("\n" + "=" * 80)
print("STEP 9 - REMOVING PRE-LAUNCH PRODUCT HISTORY")
print("=" * 80)


# Rows before the first recorded product price
# usually represent periods before the product
# was available for sale.

prelaunch_mask = (
    data["first_available_date"].notna()
    &
    (
        data["date"]
        <
        data["first_available_date"]
    )
)

prelaunch_rows = prelaunch_mask.sum()


print("\nPre-launch rows:")
print(prelaunch_rows)


# Remove products that never had a price.

never_available_mask = (
    data["first_available_date"]
    .isna()
)

never_available_rows = (
    never_available_mask.sum()
)


data = data[
    ~prelaunch_mask
    &
    ~never_available_mask
].copy()


print("\nRows remaining after availability cleaning:")
print(len(data))


# ============================================================
# STEP 10 - SORT TIME SERIES
# ============================================================

print("\n" + "=" * 80)
print("STEP 10 - SORTING TIME SERIES")
print("=" * 80)

data = data.sort_values(
    by=[
        "store_id",
        "item_id",
        "date"
    ]
).reset_index(drop=True)


# ============================================================
# STEP 11 - FORWARD FILL SELL PRICE
# ============================================================

print("\n" + "=" * 80)
print("STEP 11 - FORWARD FILLING PRICE GAPS")
print("=" * 80)


price_missing_before_ffill = (
    data["sell_price"]
    .isna()
    .sum()
)


# IMPORTANT:
# Forward-fill only.
#
# We do NOT back-fill because back-filling
# would use future information.

data["sell_price"] = (
    data
    .groupby(
        [
            "store_id",
            "item_id"
        ],
        observed=True
    )["sell_price"]
    .ffill()
)


price_missing_after_ffill = (
    data["sell_price"]
    .isna()
    .sum()
)


print("\nMissing price before forward fill:")
print(price_missing_before_ffill)

print("\nMissing price after forward fill:")
print(price_missing_after_ffill)


# If anything still has no price,
# remove it.

price_rows_before_drop = len(data)

data = data.dropna(
    subset=["sell_price"]
).copy()

remaining_price_rows_removed = (
    price_rows_before_drop
    - len(data)
)


print(
    "\nRemaining rows removed "
    "because price is unavailable:"
)

print(remaining_price_rows_removed)


# ============================================================
# STEP 12 - CLEAN EVENT VARIABLES
# ============================================================

print("\n" + "=" * 80)
print("STEP 12 - CLEANING EVENT VARIABLES")
print("=" * 80)


event_columns = [
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2"
]


for column in event_columns:

    data[column] = (
        data[column]
        .fillna("No_Event")
    )


# Create a simple data-quality indicator.
# This is not our main ML feature engineering yet.

data["has_event"] = (
    (
        data["event_name_1"]
        != "No_Event"
    )
    |
    (
        data["event_name_2"]
        != "No_Event"
    )
).astype("int8")


print("\nEvent columns cleaned.")


# ============================================================
# STEP 13 - CLEAN SNAP
# ============================================================

print("\n" + "=" * 80)
print("STEP 13 - CLEANING SNAP FEATURE")
print("=" * 80)

data["snap"] = pd.to_numeric(
    data["snap"],
    errors="coerce"
)


missing_snap = (
    data["snap"]
    .isna()
    .sum()
)


print("\nMissing SNAP values:")
print(missing_snap)


# SNAP should normally be 0 or 1.
# Missing values are treated as no SNAP event.

data["snap"] = (
    data["snap"]
    .fillna(0)
    .astype("int8")
)


# ============================================================
# STEP 14 - ZERO SALES ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("STEP 14 - ANALYZING ZERO SALES")
print("=" * 80)


# DO NOT remove zero sales.
# Zero demand is meaningful information.

data["zero_sales"] = (
    data["sales"] == 0
).astype("int8")


zero_sales_count = (
    data["zero_sales"].sum()
)

zero_sales_percentage = (
    zero_sales_count
    / len(data)
    * 100
)


print("\nZero-sales observations:")
print(zero_sales_count)

print(
    f"\nZero-sales percentage: "
    f"{zero_sales_percentage:.2f}%"
)


# ============================================================
# STEP 15 - CHECK SALES DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("STEP 15 - SALES DISTRIBUTION")
print("=" * 80)

print(
    data["sales"]
    .describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
)


# IMPORTANT:
# We are intentionally NOT deleting high-sales
# values yet.
#
# Large sales spikes can be caused by:
# holidays
# promotions
# seasonality
# real demand changes.
#
# We will investigate them during EDA.


# ============================================================
# STEP 16 - CLEAN BASIC TEXT COLUMNS
# ============================================================

print("\n" + "=" * 80)
print("STEP 16 - OPTIMIZING CATEGORICAL VARIABLES")
print("=" * 80)


categorical_columns = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "weekday",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2"
]


for column in categorical_columns:

    if column in data.columns:

        data[column] = (
            data[column]
            .astype("category")
        )


# ============================================================
# STEP 17 - FINAL DUPLICATE CHECK
# ============================================================

print("\n" + "=" * 80)
print("STEP 17 - FINAL DUPLICATE CHECK")
print("=" * 80)

final_duplicates = (
    data.duplicated(
        subset=[
            "date",
            "item_id",
            "store_id"
        ]
    ).sum()
)

print("\nFinal duplicate count:")
print(final_duplicates)


# ============================================================
# STEP 18 - FINAL MISSING VALUE REPORT
# ============================================================

print("\n" + "=" * 80)
print("STEP 18 - FINAL MISSING VALUE REPORT")
print("=" * 80)


missing_report = (
    data.isnull()
    .sum()
    .sort_values(
        ascending=False
    )
)


print(
    missing_report[
        missing_report > 0
    ]
)


# ============================================================
# STEP 19 - FINAL DATA QUALITY CHECKS
# ============================================================

print("\n" + "=" * 80)
print("STEP 19 - FINAL QUALITY CHECKS")
print("=" * 80)


print("\nFinal shape:")
print(data.shape)


print("\nDate range:")

print(
    data["date"].min(),
    "to",
    data["date"].max()
)


print("\nUnique products:")

print(
    data["item_id"]
    .nunique()
)


print("\nUnique stores:")

print(
    data["store_id"]
    .nunique()
)


print("\nUnique categories:")

print(
    data["cat_id"]
    .nunique()
)


print("\nTotal sales:")

print(
    data["sales"]
    .sum()
)


print("\nAverage sales:")

print(
    data["sales"]
    .mean()
)


print("\nMinimum price:")

print(
    data["sell_price"]
    .min()
)


print("\nMaximum price:")

print(
    data["sell_price"]
    .max()
)


# ============================================================
# STEP 20 - MEMORY USAGE
# ============================================================

memory_mb = (
    data.memory_usage(
        deep=True
    ).sum()
    / (1024 ** 2)
)


print("\nDataset memory usage:")

print(
    f"{memory_mb:.2f} MB"
)


# ============================================================
# STEP 21 - SAVE DATA QUALITY REPORT
# ============================================================

print("\n" + "=" * 80)
print("STEP 21 - SAVING QUALITY REPORT")
print("=" * 80)


quality_report = pd.DataFrame(
    {
        "metric": [
            "original_rows",
            "missing_key_rows",
            "duplicates_removed",
            "invalid_sales_removed",
            "invalid_prices_detected",
            "prelaunch_rows_removed",
            "never_available_rows_removed",
            "price_missing_before_ffill",
            "price_missing_after_ffill",
            "remaining_price_rows_removed",
            "final_rows",
            "unique_items",
            "zero_sales_count",
            "zero_sales_percentage",
            "memory_mb"
        ],

        "value": [
            original_rows,
            missing_key_rows,
            duplicate_count,
            invalid_sales_removed,
            invalid_price_count,
            prelaunch_rows,
            never_available_rows,
            price_missing_before_ffill,
            price_missing_after_ffill,
            remaining_price_rows_removed,
            len(data),
            data["item_id"].nunique(),
            zero_sales_count,
            round(
                zero_sales_percentage,
                2
            ),
            round(
                memory_mb,
                2
            )
        ]
    }
)


quality_report.to_csv(
    QUALITY_REPORT_FILE,
    index=False
)


print("\nQuality report saved to:")

print(
    QUALITY_REPORT_FILE
)


# ============================================================
# STEP 22 - SAVE CLEAN DATASET
# ============================================================

print("\n" + "=" * 80)
print("STEP 22 - SAVING CLEAN DATASET")
print("=" * 80)


data.to_parquet(
    OUTPUT_FILE,
    index=False
)


print("\nClean dataset saved successfully:")

print(
    OUTPUT_FILE
)


# ============================================================
# STEP 23 - FINAL SAMPLE
# ============================================================

print("\n" + "=" * 80)
print("CLEAN DATA SAMPLE")
print("=" * 80)


sample_columns = [
    "date",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "sales",
    "sell_price",
    "event_name_1",
    "event_type_1",
    "snap",
    "has_event",
    "zero_sales"
]


print(
    data[
        sample_columns
    ].head(20)
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("STAGE 3 COMPLETED SUCCESSFULLY")
print("=" * 80)


del data

gc.collect()