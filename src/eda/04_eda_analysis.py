from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "m5_ca1_dev_clean.parquet"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "tables"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# START
# ============================================================

print("=" * 80)
print("STAGE 4 - EXPLORATORY DATA ANALYSIS")
print("=" * 80)


# ============================================================
# STEP 1 - LOAD CLEAN DATA
# ============================================================

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"Clean dataset not found:\n{DATA_FILE}\n"
        "Run Stage 3 first."
    )


print("\nLoading clean dataset...")

df = pd.read_parquet(DATA_FILE)

df["date"] = pd.to_datetime(
    df["date"]
)


print("\nDataset loaded successfully.")

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

print("\nCategories:")
print(
    df["cat_id"].unique()
)


# ============================================================
# STEP 2 - BASIC STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("BASIC SALES STATISTICS")
print("=" * 80)

sales_stats = df["sales"].describe(
    percentiles=[
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99
    ]
)

print(sales_stats)

sales_stats.to_csv(
    TABLE_DIR
    / "sales_statistics.csv"
)


# ============================================================
# STEP 3 - DAILY SALES TREND
# ============================================================

print("\nCreating daily sales trend...")

daily_sales = (
    df.groupby(
        "date",
        observed=True
    )["sales"]
    .sum()
    .reset_index()
)


daily_sales.to_csv(
    TABLE_DIR
    / "daily_sales.csv",
    index=False
)


plt.figure(
    figsize=(14, 6)
)

plt.plot(
    daily_sales["date"],
    daily_sales["sales"]
)

plt.title(
    "Overall Daily Sales Trend"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Total Units Sold"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "01_daily_sales_trend.png",
    dpi=300
)

plt.close()


# ============================================================
# STEP 4 - 28-DAY ROLLING SALES TREND
# ============================================================

daily_sales[
    "rolling_28"
] = (
    daily_sales["sales"]
    .rolling(
        window=28
    )
    .mean()
)


plt.figure(
    figsize=(14, 6)
)

plt.plot(
    daily_sales["date"],
    daily_sales["sales"],
    alpha=0.35,
    label="Daily Sales"
)

plt.plot(
    daily_sales["date"],
    daily_sales["rolling_28"],
    linewidth=2,
    label="28-Day Rolling Average"
)

plt.title(
    "Daily Sales with 28-Day Rolling Average"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Units Sold"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "02_rolling_28_day_sales.png",
    dpi=300
)

plt.close()


# ============================================================
# STEP 5 - MONTHLY SALES TREND
# ============================================================

print("Creating monthly sales analysis...")


monthly_sales = (
    df.set_index("date")
    .resample("ME")["sales"]
    .sum()
    .reset_index()
)


monthly_sales.to_csv(
    TABLE_DIR
    / "monthly_sales.csv",
    index=False
)


plt.figure(
    figsize=(14, 6)
)

plt.plot(
    monthly_sales["date"],
    monthly_sales["sales"],
    marker="o",
    markersize=3
)

plt.title(
    "Monthly Sales Trend"
)

plt.xlabel(
    "Month"
)

plt.ylabel(
    "Total Units Sold"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "03_monthly_sales_trend.png",
    dpi=300
)

plt.close()


# ============================================================
# STEP 6 - CATEGORY SALES
# ============================================================

print("Analyzing categories...")


category_sales = (
    df.groupby(
        "cat_id",
        observed=True
    )["sales"]
    .sum()
    .sort_values(
        ascending=False
    )
)


category_sales.to_csv(
    TABLE_DIR
    / "category_sales.csv"
)


plt.figure(
    figsize=(10, 6)
)

category_sales.plot(
    kind="bar"
)

plt.title(
    "Total Sales by Product Category"
)

plt.xlabel(
    "Category"
)

plt.ylabel(
    "Units Sold"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "04_category_sales.png",
    dpi=300
)

plt.close()


# ============================================================
# STEP 7 - DEPARTMENT SALES
# ============================================================

print("Analyzing departments...")


department_sales = (
    df.groupby(
        "dept_id",
        observed=True
    )["sales"]
    .sum()
    .sort_values(
        ascending=False
    )
)


department_sales.to_csv(
    TABLE_DIR
    / "department_sales.csv"
)


plt.figure(
    figsize=(12, 6)
)

department_sales.plot(
    kind="bar"
)

plt.title(
    "Total Sales by Department"
)

plt.xlabel(
    "Department"
)

plt.ylabel(
    "Units Sold"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "05_department_sales.png",
    dpi=300
)

plt.close()


# ============================================================
# STEP 8 - WEEKDAY DEMAND
# ============================================================

print("Analyzing weekday demand...")


weekday_sales = (
    df.groupby(
        "weekday",
        observed=True
    )["sales"]
    .mean()
)


weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]


weekday_sales = weekday_sales.reindex(
    weekday_order
)


weekday_sales.to_csv(
    TABLE_DIR
    / "weekday_average_sales.csv"
)


plt.figure(
    figsize=(10, 6)
)

weekday_sales.plot(
    kind="bar"
)

plt.title(
    "Average Sales by Day of Week"
)

plt.xlabel(
    "Day"
)

plt.ylabel(
    "Average Units Sold"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "06_weekday_sales.png",
    dpi=300
)

plt.close()


# ============================================================
# STEP 9 - MONTH OF YEAR SEASONALITY
# ============================================================

print("Analyzing monthly seasonality...")


month_sales = (
    df.groupby(
        "month",
        observed=True
    )["sales"]
    .mean()
)


month_sales.to_csv(
    TABLE_DIR
    / "month_average_sales.csv"
)


plt.figure(
    figsize=(10, 6)
)

month_sales.plot(
    kind="bar"
)

plt.title(
    "Average Demand by Month"
)

plt.xlabel(
    "Month"
)

plt.ylabel(
    "Average Units Sold"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "07_month_seasonality.png",
    dpi=300
)

plt.close()


# ============================================================
# STEP 10 - TOP 10 PRODUCTS
# ============================================================

print("Finding top selling products...")


top_products = (
    df.groupby(
        "item_id",
        observed=True
    )["sales"]
    .sum()
    .sort_values(
        ascending=False
    )
    .head(10)
)


top_products.to_csv(
    TABLE_DIR
    / "top_10_products.csv"
)


plt.figure(
    figsize=(12, 6)
)

top_products.sort_values().plot(
    kind="barh"
)

plt.title(
    "Top 10 Products by Total Sales"
)

plt.xlabel(
    "Units Sold"
)

plt.ylabel(
    "Product"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "08_top_10_products.png",
    dpi=300
)

plt.close()


# ============================================================
# STEP 11 - ZERO SALES ANALYSIS
# ============================================================

print("Analyzing zero-demand observations...")


zero_percentage = (
    df["zero_sales"]
    .mean()
    * 100
)


print(
    f"\nOverall zero-sales percentage: "
    f"{zero_percentage:.2f}%"
)


zero_by_category = (
    df.groupby(
        "cat_id",
        observed=True
    )["zero_sales"]
    .mean()
    * 100
)


zero_by_category.to_csv(
    TABLE_DIR
    / "zero_sales_by_category.csv"
)


plt.figure(
    figsize=(10, 6)
)

zero_by_category.plot(
    kind="bar"
)

plt.title(
    "Zero-Sales Percentage by Category"
)

plt.xlabel(
    "Category"
)

plt.ylabel(
    "Zero Sales (%)"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "09_zero_sales_by_category.png",
    dpi=300
)

plt.close()


# ============================================================
# STEP 12 - SNAP IMPACT
# ============================================================

print("Analyzing SNAP impact...")


snap_analysis = (
    df.groupby(
        "snap",
        observed=True
    )["sales"]
    .agg(
        [
            "mean",
            "median",
            "sum"
        ]
    )
)


snap_analysis.to_csv(
    TABLE_DIR
    / "snap_sales_analysis.csv"
)


print("\nSNAP analysis:")

print(
    snap_analysis
)


snap_mean = (
    df.groupby(
        "snap",
        observed=True
    )["sales"]
    .mean()
)


plt.figure(
    figsize=(8, 6)
)

snap_mean.plot(
    kind="bar"
)

plt.title(
    "Average Sales: SNAP vs Non-SNAP Days"
)

plt.xlabel(
    "SNAP Indicator"
)

plt.ylabel(
    "Average Units Sold"
)

plt.xticks(
    ticks=[0, 1],
    labels=[
        "No SNAP",
        "SNAP"
    ],
    rotation=0
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "10_snap_impact.png",
    dpi=300
)

plt.close()


# ============================================================
# STEP 13 - EVENT IMPACT
# ============================================================

print("Analyzing event impact...")


event_analysis = (
    df.groupby(
        "has_event",
        observed=True
    )["sales"]
    .agg(
        [
            "mean",
            "median",
            "sum"
        ]
    )
)


event_analysis.to_csv(
    TABLE_DIR
    / "event_sales_analysis.csv"
)


event_mean = (
    df.groupby(
        "has_event",
        observed=True
    )["sales"]
    .mean()
)


plt.figure(
    figsize=(8, 6)
)

event_mean.plot(
    kind="bar"
)

plt.title(
    "Average Sales: Event vs Non-Event Days"
)

plt.xlabel(
    "Event Indicator"
)

plt.ylabel(
    "Average Units Sold"
)

plt.xticks(
    ticks=[0, 1],
    labels=[
        "No Event",
        "Event"
    ],
    rotation=0
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "11_event_impact.png",
    dpi=300
)

plt.close()


# ============================================================
# STEP 14 - INDIVIDUAL EVENT ANALYSIS
# ============================================================

events_only = df[
    df["event_name_1"]
    != "No_Event"
].copy()


if not events_only.empty:

    event_name_sales = (
        events_only.groupby(
            "event_name_1",
            observed=True
        )["sales"]
        .mean()
        .sort_values(
            ascending=False
        )
        .head(15)
    )


    event_name_sales.to_csv(
        TABLE_DIR
        / "top_events_by_average_sales.csv"
    )


    plt.figure(
        figsize=(12, 7)
    )

    event_name_sales.sort_values().plot(
        kind="barh"
    )

    plt.title(
        "Top Events by Average Product Demand"
    )

    plt.xlabel(
        "Average Units Sold"
    )

    plt.ylabel(
        "Event"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "12_event_level_sales.png",
        dpi=300
    )

    plt.close()


# ============================================================
# STEP 15 - PRICE DISTRIBUTION
# ============================================================

print("Analyzing product prices...")


plt.figure(
    figsize=(10, 6)
)

plt.hist(
    df["sell_price"],
    bins=40,
    edgecolor="black",
    alpha=0.7
)

plt.title(
    "Distribution of Product Selling Prices"
)

plt.xlabel(
    "Selling Price"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "13_price_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# STEP 16 - PRICE VS SALES
# ============================================================

print("Analyzing price-demand relationship...")


# Sampling prevents the scatter plot from becoming
# extremely dense.

sample_size = min(
    20000,
    len(df)
)

price_sample = df.sample(
    n=sample_size,
    random_state=42
)


plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    price_sample["sell_price"],
    price_sample["sales"],
    alpha=0.2,
    s=10
)

plt.title(
    "Selling Price vs Daily Sales"
)

plt.xlabel(
    "Selling Price"
)

plt.ylabel(
    "Units Sold"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "14_price_vs_sales.png",
    dpi=300
)

plt.close()


# ============================================================
# STEP 17 - DEMAND VARIABILITY BY PRODUCT
# ============================================================

print("Analyzing product demand variability...")


product_variability = (
    df.groupby(
        "item_id",
        observed=True
    )["sales"]
    .agg(
        [
            "mean",
            "std",
            "sum"
        ]
    )
)


product_variability[
    "coefficient_of_variation"
] = (
    product_variability["std"]
    /
    product_variability["mean"]
    .replace(
        0,
        np.nan
    )
)


product_variability = (
    product_variability
    .sort_values(
        "coefficient_of_variation",
        ascending=False
    )
)


product_variability.to_csv(
    TABLE_DIR
    / "product_demand_variability.csv"
)


highest_variability = (
    product_variability
    .dropna()
    .head(15)
)


plt.figure(
    figsize=(12, 7)
)

highest_variability[
    "coefficient_of_variation"
].sort_values().plot(
    kind="barh"
)

plt.title(
    "Products with Highest Demand Variability"
)

plt.xlabel(
    "Coefficient of Variation"
)

plt.ylabel(
    "Product"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "15_product_demand_variability.png",
    dpi=300
)

plt.close()


# ============================================================
# STEP 18 - SEASONAL DECOMPOSITION
# ============================================================

print("Performing weekly seasonal decomposition...")


daily_series = (
    daily_sales
    .set_index("date")["sales"]
    .asfreq("D")
)


daily_series = (
    daily_series
    .interpolate()
    .ffill()
    .bfill()
)


if len(daily_series) >= 14:

    decomposition = seasonal_decompose(
        daily_series,
        model="additive",
        period=7
    )


    fig = decomposition.plot()

    fig.set_size_inches(
        14,
        10
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "16_weekly_seasonal_decomposition.png",
        dpi=300
    )

    plt.close(fig)


# ============================================================
# STEP 19 - YEARLY SALES
# ============================================================

print("Analyzing yearly sales...")


yearly_sales = (
    df.groupby(
        "year",
        observed=True
    )["sales"]
    .sum()
)


yearly_sales.to_csv(
    TABLE_DIR
    / "yearly_sales.csv"
)


plt.figure(
    figsize=(10, 6)
)

yearly_sales.plot(
    kind="bar"
)

plt.title(
    "Total Sales by Year"
)

plt.xlabel(
    "Year"
)

plt.ylabel(
    "Units Sold"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "17_yearly_sales.png",
    dpi=300
)

plt.close()


# ============================================================
# STEP 20 - FINAL EDA SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("EDA SUMMARY")
print("=" * 80)


best_category = (
    category_sales.index[0]
)


best_product = (
    top_products.index[0]
)


highest_weekday = (
    weekday_sales.idxmax()
)


highest_month = (
    month_sales.idxmax()
)


print(
    f"\nHighest selling category: "
    f"{best_category}"
)

print(
    f"Highest selling product: "
    f"{best_product}"
)

print(
    f"Highest average demand weekday: "
    f"{highest_weekday}"
)

print(
    f"Highest average demand month: "
    f"{highest_month}"
)

print(
    f"Zero-sales percentage: "
    f"{zero_percentage:.2f}%"
)


print("\nCharts saved to:")

print(
    FIGURE_DIR
)


print("\nSummary tables saved to:")

print(
    TABLE_DIR
)


print("\n" + "=" * 80)
print("STAGE 4 COMPLETED SUCCESSFULLY")
print("=" * 80)