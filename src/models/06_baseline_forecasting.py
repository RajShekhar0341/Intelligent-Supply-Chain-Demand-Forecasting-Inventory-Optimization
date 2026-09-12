from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

FORECAST_HORIZON = 28


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

REPORT_TABLE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "tables"
)

REPORT_FIGURE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

REPORT_TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TRAIN_FILE = (
    PROCESSED_DIR
    / "m5_ca1_train.parquet"
)

VALID_FILE = (
    PROCESSED_DIR
    / "m5_ca1_validation.parquet"
)

PREDICTION_FILE = (
    PROCESSED_DIR
    / "baseline_validation_predictions.parquet"
)

METRICS_FILE = (
    REPORT_TABLE_DIR
    / "baseline_model_metrics.csv"
)

ITEM_METRICS_FILE = (
    REPORT_TABLE_DIR
    / "baseline_item_metrics.csv"
)


# ============================================================
# START
# ============================================================

print("=" * 80)
print("STAGE 6 - BASELINE DEMAND FORECASTING")
print("=" * 80)


# ============================================================
# STEP 1 - CHECK FILES
# ============================================================

for file in [
    TRAIN_FILE,
    VALID_FILE
]:

    if not file.exists():

        raise FileNotFoundError(
            f"Required file missing:\n{file}\n"
            "Run Stage 5 first."
        )


# ============================================================
# STEP 2 - LOAD DATA
# ============================================================

print("\nLoading training data...")

train = pd.read_parquet(
    TRAIN_FILE
)


print("Loading validation data...")

validation = pd.read_parquet(
    VALID_FILE
)


train["date"] = pd.to_datetime(
    train["date"]
)

validation["date"] = pd.to_datetime(
    validation["date"]
)


train = train.sort_values(
    [
        "store_id",
        "item_id",
        "date"
    ]
).reset_index(drop=True)


validation = validation.sort_values(
    [
        "store_id",
        "item_id",
        "date"
    ]
).reset_index(drop=True)


print("\nTraining shape:")
print(train.shape)

print("\nValidation shape:")
print(validation.shape)


print("\nTraining period:")

print(
    train["date"].min(),
    "to",
    train["date"].max()
)


print("\nValidation period:")

print(
    validation["date"].min(),
    "to",
    validation["date"].max()
)


# ============================================================
# STEP 3 - VERIFY NO DATE LEAKAGE
# ============================================================

print("\n" + "=" * 80)
print("STEP 3 - VERIFYING TIME SPLIT")
print("=" * 80)


train_max_date = train["date"].max()

validation_min_date = validation["date"].min()


if train_max_date >= validation_min_date:

    raise ValueError(
        "Training and validation periods overlap."
    )


print("\nTime split is valid.")

print(
    "Last training date:",
    train_max_date
)

print(
    "First validation date:",
    validation_min_date
)


# ============================================================
# CREATE FORECAST DATAFRAME
# ============================================================

forecast = validation[
    [
        "date",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "sales"
    ]
].copy()


forecast = forecast.rename(
    columns={
        "sales":
        "actual_sales"
    }
)


KEYS = [
    "store_id",
    "item_id"
]


# ============================================================
# STEP 4 - BASELINE 1: LAST OBSERVED VALUE
# ============================================================

print("\n" + "=" * 80)
print("BASELINE 1 - LAST OBSERVED VALUE")
print("=" * 80)


last_observation = (
    train
    .sort_values("date")
    .groupby(
        KEYS,
        observed=True
    )
    .tail(1)
    [
        KEYS
        + ["sales"]
    ]
)


last_observation = (
    last_observation
    .rename(
        columns={
            "sales":
            "pred_last_value"
        }
    )
)


forecast = forecast.merge(
    last_observation,
    how="left",
    on=KEYS
)


print(
    "\nLast-value forecasts created."
)


# ============================================================
# STEP 5 - BASELINE 2: 28-DAY SEASONAL NAIVE
# ============================================================

print("\n" + "=" * 80)
print("BASELINE 2 - 28-DAY SEASONAL NAIVE")
print("=" * 80)


seasonal_history = train[
    [
        "date",
        "store_id",
        "item_id",
        "sales"
    ]
].copy()


# The historical observation becomes the forecast
# exactly 28 days later.

seasonal_history[
    "date"
] = (
    seasonal_history["date"]
    +
    pd.Timedelta(
        days=FORECAST_HORIZON
    )
)


seasonal_history = (
    seasonal_history
    .rename(
        columns={
            "sales":
            "pred_seasonal_28"
        }
    )
)


forecast = forecast.merge(
    seasonal_history,
    how="left",
    on=[
        "date",
        "store_id",
        "item_id"
    ]
)


# If a seasonal observation is unavailable,
# use last observed demand as fallback.

forecast[
    "pred_seasonal_28"
] = (
    forecast[
        "pred_seasonal_28"
    ]
    .fillna(
        forecast[
            "pred_last_value"
        ]
    )
)


print(
    "\n28-day seasonal forecasts created."
)


# ============================================================
# STEP 6 - BASELINE 3: LAST 28-DAY MEAN
# ============================================================

print("\n" + "=" * 80)
print("BASELINE 3 - LAST 28-DAY MEAN")
print("=" * 80)


last_28_train = (
    train
    .groupby(
        KEYS,
        observed=True,
        group_keys=False
    )
    .tail(
        FORECAST_HORIZON
    )
)


mean_28 = (
    last_28_train
    .groupby(
        KEYS,
        observed=True
    )["sales"]
    .mean()
    .reset_index()
)


mean_28 = mean_28.rename(
    columns={
        "sales":
        "pred_mean_28"
    }
)


forecast = forecast.merge(
    mean_28,
    how="left",
    on=KEYS
)


forecast[
    "pred_mean_28"
] = (
    forecast[
        "pred_mean_28"
    ]
    .fillna(
        forecast[
            "pred_last_value"
        ]
    )
)


print(
    "\n28-day mean forecasts created."
)


# ============================================================
# STEP 7 - BASELINE 4: WEEKDAY DEMAND MEAN
# ============================================================

print("\n" + "=" * 80)
print("BASELINE 4 - WEEKDAY DEMAND MEAN")
print("=" * 80)


# Use only the most recent 56 training days.

recent_train = (
    train
    .groupby(
        KEYS,
        observed=True,
        group_keys=False
    )
    .tail(56)
    .copy()
)


recent_train[
    "forecast_day_of_week"
] = (
    recent_train["date"]
    .dt.dayofweek
)


weekday_history = (
    recent_train
    .groupby(
        KEYS
        + ["forecast_day_of_week"],
        observed=True
    )["sales"]
    .mean()
    .reset_index()
)


weekday_history = (
    weekday_history
    .rename(
        columns={
            "sales":
            "pred_weekday_mean"
        }
    )
)


forecast[
    "forecast_day_of_week"
] = (
    forecast["date"]
    .dt.dayofweek
)


forecast = forecast.merge(
    weekday_history,
    how="left",
    on=(
        KEYS
        + ["forecast_day_of_week"]
    )
)


forecast[
    "pred_weekday_mean"
] = (
    forecast[
        "pred_weekday_mean"
    ]
    .fillna(
        forecast[
            "pred_mean_28"
        ]
    )
)


print(
    "\nWeekday forecasts created."
)


# ============================================================
# STEP 8 - CLIP NEGATIVE FORECASTS
# ============================================================

print("\n" + "=" * 80)
print("STEP 8 - CLEANING PREDICTIONS")
print("=" * 80)


prediction_columns = [
    "pred_last_value",
    "pred_seasonal_28",
    "pred_mean_28",
    "pred_weekday_mean"
]


for column in prediction_columns:

    forecast[column] = (
        pd.to_numeric(
            forecast[column],
            errors="coerce"
        )
        .fillna(0)
        .clip(lower=0)
    )


# ============================================================
# STEP 9 - METRIC FUNCTIONS
# ============================================================

print("\n" + "=" * 80)
print("STEP 9 - EVALUATING BASELINES")
print("=" * 80)


def mae(y_true, y_pred):

    return np.mean(
        np.abs(
            y_true
            -
            y_pred
        )
    )


def rmse(y_true, y_pred):

    return np.sqrt(
        np.mean(
            (
                y_true
                -
                y_pred
            ) ** 2
        )
    )


def wape(y_true, y_pred):

    denominator = np.sum(
        np.abs(
            y_true
        )
    )

    if denominator == 0:
        return np.nan

    return (
        np.sum(
            np.abs(
                y_true
                -
                y_pred
            )
        )
        /
        denominator
        *
        100
    )


def rmsle(y_true, y_pred):

    y_true = np.clip(
        y_true,
        0,
        None
    )

    y_pred = np.clip(
        y_pred,
        0,
        None
    )

    return np.sqrt(
        np.mean(
            (
                np.log1p(y_pred)
                -
                np.log1p(y_true)
            ) ** 2
        )
    )


def forecast_bias(
    y_true,
    y_pred
):

    return np.mean(
        y_pred
        -
        y_true
    )


# ============================================================
# STEP 10 - CALCULATE OVERALL METRICS
# ============================================================

model_mapping = {

    "Last Value":
        "pred_last_value",

    "Seasonal Naive 28":
        "pred_seasonal_28",

    "28-Day Mean":
        "pred_mean_28",

    "Weekday Mean":
        "pred_weekday_mean"
}


actual = (
    forecast[
        "actual_sales"
    ]
    .to_numpy()
)


metrics_results = []


for model_name, prediction_column in model_mapping.items():

    prediction = (
        forecast[
            prediction_column
        ]
        .to_numpy()
    )


    model_mae = mae(
        actual,
        prediction
    )

    model_rmse = rmse(
        actual,
        prediction
    )

    model_wape = wape(
        actual,
        prediction
    )

    model_rmsle = rmsle(
        actual,
        prediction
    )

    model_bias = forecast_bias(
        actual,
        prediction
    )


    metrics_results.append(
        {
            "model":
                model_name,

            "MAE":
                model_mae,

            "RMSE":
                model_rmse,

            "WAPE_percent":
                model_wape,

            "RMSLE":
                model_rmsle,

            "Forecast_Bias":
                model_bias
        }
    )


metrics_df = pd.DataFrame(
    metrics_results
)


metrics_df = (
    metrics_df
    .sort_values(
        "WAPE_percent"
    )
    .reset_index(
        drop=True
    )
)


print("\nBaseline Model Results:")

print(
    metrics_df.round(4)
)


# ============================================================
# STEP 11 - SELECT BEST BASELINE
# ============================================================

best_model = (
    metrics_df
    .iloc[0]["model"]
)


best_wape = (
    metrics_df
    .iloc[0][
        "WAPE_percent"
    ]
)


print(
    "\nBest baseline model:"
)

print(
    best_model
)


print(
    "\nBest baseline WAPE:"
)

print(
    f"{best_wape:.2f}%"
)


# ============================================================
# STEP 12 - ITEM-LEVEL METRICS
# ============================================================

print("\n" + "=" * 80)
print("STEP 12 - ITEM-LEVEL PERFORMANCE")
print("=" * 80)


item_metric_rows = []


for (
    store_id,
    item_id
), item_data in forecast.groupby(
    KEYS,
    observed=True
):

    y_true = (
        item_data[
            "actual_sales"
        ]
        .to_numpy()
    )


    for model_name, column in model_mapping.items():

        y_pred = (
            item_data[
                column
            ]
            .to_numpy()
        )


        item_metric_rows.append(
            {
                "store_id":
                    store_id,

                "item_id":
                    item_id,

                "model":
                    model_name,

                "MAE":
                    mae(
                        y_true,
                        y_pred
                    ),

                "RMSE":
                    rmse(
                        y_true,
                        y_pred
                    ),

                "WAPE_percent":
                    wape(
                        y_true,
                        y_pred
                    )
            }
        )


item_metrics = pd.DataFrame(
    item_metric_rows
)


# ============================================================
# STEP 13 - SAVE RESULTS
# ============================================================

print("\n" + "=" * 80)
print("STEP 13 - SAVING RESULTS")
print("=" * 80)


forecast.to_parquet(
    PREDICTION_FILE,
    index=False
)


metrics_df.to_csv(
    METRICS_FILE,
    index=False
)


item_metrics.to_csv(
    ITEM_METRICS_FILE,
    index=False
)


print("\nPrediction file:")

print(
    PREDICTION_FILE
)


print("\nMetrics file:")

print(
    METRICS_FILE
)


print("\nItem metrics file:")

print(
    ITEM_METRICS_FILE
)


# ============================================================
# STEP 14 - MODEL COMPARISON GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plot_data = (
    metrics_df
    .sort_values(
        "WAPE_percent",
        ascending=False
    )
)


plt.barh(
    plot_data["model"],
    plot_data["WAPE_percent"]
)


plt.title(
    "Baseline Forecast Comparison"
)

plt.xlabel(
    "WAPE (%) - Lower is Better"
)

plt.ylabel(
    "Forecasting Method"
)


plt.tight_layout()


plt.savefig(
    REPORT_FIGURE_DIR
    / "18_baseline_model_comparison.png",
    dpi=300
)


plt.close()


# ============================================================
# STEP 15 - AGGREGATED FORECAST VS ACTUAL
# ============================================================

daily_forecast = (
    forecast
    .groupby(
        "date",
        observed=True
    )
    [
        [
            "actual_sales",
            "pred_last_value",
            "pred_seasonal_28",
            "pred_mean_28",
            "pred_weekday_mean"
        ]
    ]
    .sum()
    .reset_index()
)


plt.figure(
    figsize=(14, 7)
)


plt.plot(
    daily_forecast["date"],
    daily_forecast["actual_sales"],
    marker="o",
    label="Actual"
)


plt.plot(
    daily_forecast["date"],
    daily_forecast["pred_seasonal_28"],
    label="Seasonal Naive 28"
)


plt.plot(
    daily_forecast["date"],
    daily_forecast["pred_mean_28"],
    label="28-Day Mean"
)


plt.plot(
    daily_forecast["date"],
    daily_forecast["pred_weekday_mean"],
    label="Weekday Mean"
)


plt.title(
    "Actual vs Baseline Demand Forecast"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Total Units Sold"
)


plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    REPORT_FIGURE_DIR
    / "19_baseline_actual_vs_forecast.png",
    dpi=300
)


plt.close()


# ============================================================
# STEP 16 - FORECAST ERROR
# ============================================================

best_model_column = {
    "Last Value":
        "pred_last_value",

    "Seasonal Naive 28":
        "pred_seasonal_28",

    "28-Day Mean":
        "pred_mean_28",

    "Weekday Mean":
        "pred_weekday_mean"

}[best_model]


forecast[
    "best_baseline_prediction"
] = (
    forecast[
        best_model_column
    ]
)


forecast[
    "forecast_error"
] = (
    forecast[
        "actual_sales"
    ]
    -
    forecast[
        "best_baseline_prediction"
    ]
)


forecast[
    "absolute_error"
] = (
    forecast[
        "forecast_error"
    ]
    .abs()
)


# Save again with best-model columns.

forecast.to_parquet(
    PREDICTION_FILE,
    index=False
)


# ============================================================
# STEP 17 - ERROR BY CATEGORY
# ============================================================

category_error = (
    forecast
    .groupby(
        "cat_id",
        observed=True
    )
    .agg(
        actual_sales=(
            "actual_sales",
            "sum"
        ),

        forecast_sales=(
            "best_baseline_prediction",
            "sum"
        ),

        absolute_error=(
            "absolute_error",
            "sum"
        )
    )
)


category_error[
    "WAPE_percent"
] = (
    category_error[
        "absolute_error"
    ]
    /
    category_error[
        "actual_sales"
    ]
    .replace(
        0,
        np.nan
    )
    *
    100
)


category_error.to_csv(
    REPORT_TABLE_DIR
    / "baseline_category_error.csv"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("STAGE 6 SUMMARY")
print("=" * 80)


print(
    f"\nNumber of validation observations: "
    f"{len(forecast):,}"
)


print(
    f"\nForecast horizon: "
    f"{FORECAST_HORIZON} days"
)


print(
    f"\nBest baseline model: "
    f"{best_model}"
)


print(
    f"\nBest baseline WAPE: "
    f"{best_wape:.2f}%"
)


print(
    "\nThe test dataset has NOT been used."
)


print(
    "\nStage 6 completed successfully."
)


print("\n" + "=" * 80)
print("STAGE 6 COMPLETED SUCCESSFULLY")
print("=" * 80)