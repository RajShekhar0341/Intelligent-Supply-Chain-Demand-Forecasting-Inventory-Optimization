from pathlib import Path
import pandas as pd
import numpy as np
import gc
import time


# ============================================================
# CONFIGURATION
# ============================================================

STORE_ID = "CA_1"

# For development/testing.
# Later we will increase this.
MAX_ITEMS = 300

CHUNK_SIZE = 1000


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 80)
print("STAGE 2 - DATA UNDERSTANDING AND INTEGRATION")
print("=" * 80)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nRaw data directory:")
print(RAW_DATA_DIR)


# ============================================================
# SELECT SALES FILE
# ============================================================

evaluation_file = RAW_DATA_DIR / "sales_train_evaluation.csv"
validation_file = RAW_DATA_DIR / "sales_train_validation.csv"

if evaluation_file.exists():

    SALES_FILE = evaluation_file
    print("\nUsing: sales_train_evaluation.csv")

elif validation_file.exists():

    SALES_FILE = validation_file
    print("\nUsing: sales_train_validation.csv")

else:

    raise FileNotFoundError(
        "No sales training file found inside data/raw/"
    )


CALENDAR_FILE = RAW_DATA_DIR / "calendar.csv"
PRICES_FILE = RAW_DATA_DIR / "sell_prices.csv"


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

for file in [CALENDAR_FILE, PRICES_FILE]:

    if not file.exists():

        raise FileNotFoundError(
            f"Required file not found: {file}"
        )


# ============================================================
# STEP 1 - LOAD CALENDAR DATA
# ============================================================

print("\n" + "=" * 80)
print("STEP 1 - LOADING CALENDAR DATA")
print("=" * 80)

calendar = pd.read_csv(
    CALENDAR_FILE,
    parse_dates=["date"]
)

print("\nCalendar shape:")
print(calendar.shape)

print("\nCalendar columns:")
print(calendar.columns.tolist())

print("\nFirst 5 calendar rows:")
print(calendar.head())

print("\nCalendar date range:")
print(calendar["date"].min(), "to", calendar["date"].max())


# ============================================================
# STEP 2 - CALENDAR MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("STEP 2 - CALENDAR MISSING VALUES")
print("=" * 80)

calendar_missing = (
    calendar.isnull()
    .sum()
    .sort_values(ascending=False)
)

print(calendar_missing[calendar_missing > 0])


# ============================================================
# STEP 3 - LOAD SALES DATA IN CHUNKS
# ============================================================

print("\n" + "=" * 80)
print(f"STEP 3 - SELECTING {MAX_ITEMS} ITEMS FROM STORE {STORE_ID}")
print("=" * 80)

selected_chunks = []

items_collected = 0

reader = pd.read_csv(
    SALES_FILE,
    chunksize=CHUNK_SIZE
)

for chunk_number, chunk in enumerate(reader, start=1):

    print(
        f"Reading chunk {chunk_number}...",
        end="\r"
    )

    store_chunk = chunk[
        chunk["store_id"] == STORE_ID
    ]

    if len(store_chunk) > 0:

        remaining = MAX_ITEMS - items_collected

        store_chunk = store_chunk.head(remaining)

        selected_chunks.append(store_chunk)

        items_collected += len(store_chunk)

    if items_collected >= MAX_ITEMS:
        break


if not selected_chunks:

    raise ValueError(
        f"No data found for store {STORE_ID}"
    )


sales_wide = pd.concat(
    selected_chunks,
    ignore_index=True
)

del selected_chunks
gc.collect()


print("\n\nSelected sales shape:")
print(sales_wide.shape)

print("\nNumber of selected products:")
print(sales_wide["item_id"].nunique())

print("\nStore:")
print(sales_wide["store_id"].unique())

print("\nCategories:")
print(sales_wide["cat_id"].value_counts())


# ============================================================
# STEP 4 - UNDERSTAND WIDE SALES DATA
# ============================================================

print("\n" + "=" * 80)
print("STEP 4 - UNDERSTANDING SALES DATA")
print("=" * 80)

id_columns = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id"
]

day_columns = [
    column
    for column in sales_wide.columns
    if column.startswith("d_")
]


print("\nIdentifier columns:")
print(id_columns)

print("\nNumber of daily sales columns:")
print(len(day_columns))

print("\nFirst day column:")
print(day_columns[0])

print("\nLast day column:")
print(day_columns[-1])


# ============================================================
# STEP 5 - CONVERT WIDE DATA TO LONG FORMAT
# ============================================================

print("\n" + "=" * 80)
print("STEP 5 - CONVERTING WIDE DATA TO LONG FORMAT")
print("=" * 80)

start_time = time.time()

sales_long = sales_wide.melt(
    id_vars=id_columns,
    value_vars=day_columns,
    var_name="d",
    value_name="sales"
)


# Reduce memory usage

sales_long["sales"] = pd.to_numeric(
    sales_long["sales"],
    downcast="integer"
)


elapsed = time.time() - start_time

print("\nConversion completed.")

print(f"Time taken: {elapsed:.2f} seconds")

print("\nLong dataset shape:")
print(sales_long.shape)

print("\nFirst 5 rows:")
print(sales_long.head())


del sales_wide
gc.collect()


# ============================================================
# STEP 6 - SELECT REQUIRED CALENDAR FEATURES
# ============================================================

print("\n" + "=" * 80)
print("STEP 6 - PREPARING CALENDAR FEATURES")
print("=" * 80)

calendar_columns = [
    "date",
    "wm_yr_wk",
    "weekday",
    "wday",
    "month",
    "year",
    "d",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI"
]

calendar_selected = calendar[
    calendar_columns
].copy()


# ============================================================
# STEP 7 - MERGE SALES WITH CALENDAR
# ============================================================

print("\n" + "=" * 80)
print("STEP 7 - MERGING SALES WITH CALENDAR")
print("=" * 80)

data = sales_long.merge(
    calendar_selected,
    how="left",
    on="d"
)

print("\nShape after calendar merge:")
print(data.shape)


del sales_long
del calendar_selected

gc.collect()


# ============================================================
# STEP 8 - CREATE SINGLE SNAP FEATURE
# ============================================================

print("\n" + "=" * 80)
print("STEP 8 - CREATING SNAP FEATURE")
print("=" * 80)

state = data["state_id"].iloc[0]

snap_column = f"snap_{state}"

if snap_column in data.columns:

    data["snap"] = data[snap_column]

else:

    data["snap"] = 0


data.drop(
    columns=[
        "snap_CA",
        "snap_TX",
        "snap_WI"
    ],
    inplace=True
)


print(f"\nUsing SNAP column: {snap_column}")


# ============================================================
# STEP 9 - LOAD AND FILTER SELL PRICES
# ============================================================

print("\n" + "=" * 80)
print("STEP 9 - LOADING SELL PRICES")
print("=" * 80)

selected_items = data["item_id"].unique()

price_chunks = []

price_reader = pd.read_csv(
    PRICES_FILE,
    chunksize=100000
)

for chunk in price_reader:

    filtered = chunk[
        (chunk["store_id"] == STORE_ID)
        &
        (chunk["item_id"].isin(selected_items))
    ]

    if len(filtered) > 0:

        price_chunks.append(filtered)


prices = pd.concat(
    price_chunks,
    ignore_index=True
)

del price_chunks
gc.collect()


print("\nFiltered price data shape:")
print(prices.shape)

print("\nPrice sample:")
print(prices.head())


# ============================================================
# STEP 10 - MERGE SELL PRICES
# ============================================================

print("\n" + "=" * 80)
print("STEP 10 - MERGING SELL PRICES")
print("=" * 80)

data = data.merge(
    prices,
    how="left",
    on=[
        "store_id",
        "item_id",
        "wm_yr_wk"
    ]
)


print("\nFinal merged shape:")
print(data.shape)


del prices
gc.collect()


# ============================================================
# STEP 11 - SORT DATA
# ============================================================

data.sort_values(
    by=[
        "store_id",
        "item_id",
        "date"
    ],
    inplace=True
)

data.reset_index(
    drop=True,
    inplace=True
)


# ============================================================
# STEP 12 - BASIC DATA UNDERSTANDING
# ============================================================

print("\n" + "=" * 80)
print("STEP 12 - FINAL DATASET UNDERSTANDING")
print("=" * 80)

print("\nDataset shape:")
print(data.shape)

print("\nDate range:")
print(
    data["date"].min(),
    "to",
    data["date"].max()
)

print("\nUnique products:")
print(data["item_id"].nunique())

print("\nUnique departments:")
print(data["dept_id"].nunique())

print("\nUnique categories:")
print(data["cat_id"].nunique())

print("\nTotal sales:")
print(data["sales"].sum())

print("\nAverage daily sales:")
print(data["sales"].mean())

print("\nMaximum daily sales:")
print(data["sales"].max())


# ============================================================
# STEP 13 - DATA TYPES
# ============================================================

print("\n" + "=" * 80)
print("STEP 13 - DATA TYPES")
print("=" * 80)

print(data.dtypes)


# ============================================================
# STEP 14 - MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("STEP 14 - MISSING VALUES")
print("=" * 80)

missing_values = (
    data.isnull()
    .sum()
    .sort_values(ascending=False)
)

print(
    missing_values[
        missing_values > 0
    ]
)


# ============================================================
# STEP 15 - DUPLICATES
# ============================================================

print("\n" + "=" * 80)
print("STEP 15 - DUPLICATE CHECK")
print("=" * 80)

duplicates = data.duplicated(
    subset=[
        "date",
        "item_id",
        "store_id"
    ]
).sum()

print("\nDuplicate item-store-date rows:")
print(duplicates)


# ============================================================
# STEP 16 - MEMORY USAGE
# ============================================================

memory_mb = (
    data.memory_usage(
        deep=True
    ).sum()
    / 1024**2
)

print("\nApproximate dataset memory:")
print(f"{memory_mb:.2f} MB")


# ============================================================
# STEP 17 - SAVE PROCESSED DATASET
# ============================================================

print("\n" + "=" * 80)
print("STEP 17 - SAVING PROCESSED DATASET")
print("=" * 80)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "m5_ca1_dev_integrated.parquet"
)

data.to_parquet(
    OUTPUT_FILE,
    index=False
)


print("\nSaved successfully:")
print(OUTPUT_FILE)


# ============================================================
# FINAL SAMPLE
# ============================================================

print("\n" + "=" * 80)
print("FINAL DATA SAMPLE")
print("=" * 80)

display_columns = [
    "date",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "sales",
    "sell_price",
    "event_name_1",
    "event_type_1",
    "snap"
]

print(
    data[
        display_columns
    ].head(20)
)


print("\n" + "=" * 80)
print("STAGE 2 COMPLETED SUCCESSFULLY")
print("=" * 80)