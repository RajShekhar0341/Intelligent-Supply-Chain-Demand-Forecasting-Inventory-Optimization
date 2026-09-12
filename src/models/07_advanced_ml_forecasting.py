from pathlib import Path
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

import lightgbm as lgb
from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

FORECAST_HORIZON = 28

EARLY_STOPPING_ROUNDS = 60


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
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


MODEL_DIR.mkdir(
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


TRAIN_FILE = (
    PROCESSED_DIR
    / "m5_ca1_train.parquet"
)

VALID_FILE = (
    PROCESSED_DIR
    / "m5_ca1_validation.parquet"
)

BASELINE_METRICS_FILE = (
    TABLE_DIR
    / "baseline_model_metrics.csv"
)

OUTPUT_PREDICTIONS = (
    PROCESSED_DIR
    / "advanced_ml_validation_predictions.parquet"
)

ADVANCED_METRICS_FILE = (
    TABLE_DIR
    / "advanced_ml_metrics.csv"
)

COMPARISON_FILE = (
    TABLE_DIR
    / "baseline_vs_advanced_models.csv"
)


# ============================================================
# START
# ============================================================

print("=" * 80)
print("STAGE 7 - ADVANCED MACHINE LEARNING FORECASTING")
print("=" * 80)


# ============================================================
# STEP 1 - LOAD DATA
# ============================================================

if not TRAIN_FILE.exists():
    raise FileNotFoundError(
        f"Training file not found: {TRAIN_FILE}"
    )

if not VALID_FILE.exists():
    raise FileNotFoundError(
        f"Validation file not found: {VALID_FILE}"
    )


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


print("\nTrain shape:")
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
# STEP 2 - VERIFY TIME SPLIT
# ============================================================

print("\n" + "=" * 80)
print("STEP 2 - VERIFYING TIME SPLIT")
print("=" * 80)


if (
    train["date"].max()
    >=
    validation["date"].min()
):

    raise ValueError(
        "Train and validation periods overlap."
    )


print("Time-based split is valid.")


# ============================================================
# STEP 3 - IDENTIFIER ENCODING
# ============================================================

print("\n" + "=" * 80)
print("STEP 3 - ENCODING PRODUCT IDENTIFIERS")
print("=" * 80)


identifier_columns = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id"
]


encoding_maps = {}


for column in identifier_columns:

    unique_values = (
        train[column]
        .astype(str)
        .unique()
        .tolist()
    )

    encoding_maps[column] = {
        value: index
        for index, value
        in enumerate(unique_values)
    }

    train[
        f"{column}_code"
    ] = (
        train[column]
        .astype(str)
        .map(
            encoding_maps[column]
        )
        .fillna(-1)
        .astype("int32")
    )


# Save mappings

with open(
    MODEL_DIR
    / "identifier_encodings.json",
    "w"
) as file:

    json.dump(
        encoding_maps,
        file,
        indent=4
    )


# ============================================================
# STEP 4 - DEFINE MODEL FEATURES
# ============================================================

print("\n" + "=" * 80)
print("STEP 4 - DEFINING FEATURES")
print("=" * 80)


MODEL_FEATURES = [

    # Product identity

    "item_id_code",
    "dept_id_code",
    "cat_id_code",
    "store_id_code",
    "state_id_code",

    # Lag features

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

    # Rolling standard deviation

    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28",
    "rolling_std_56",

    # Rolling min/max

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

    # Cyclical calendar

    "dow_sin",
    "dow_cos",

    "month_sin",
    "month_cos",

    # External effects

    "snap",
    "has_event",

    # Product lifecycle

    "days_since_launch",

    # Demand dynamics

    "demand_trend_7_28",
    "demand_ratio_7_28",
    "demand_cv_28"
]


TARGET = "sales"


print(
    "\nTotal model features:",
    len(MODEL_FEATURES)
)


# ============================================================
# STEP 5 - INTERNAL EARLY-STOPPING SPLIT
# ============================================================

print("\n" + "=" * 80)
print("STEP 5 - INTERNAL TRAINING SPLIT")
print("=" * 80)


internal_validation_start = (
    train["date"].max()
    -
    pd.Timedelta(
        days=27
    )
)


fit_data = train[
    train["date"]
    <
    internal_validation_start
].copy()


early_stop_data = train[
    train["date"]
    >=
    internal_validation_start
].copy()


print("\nModel fitting period:")

print(
    fit_data["date"].min(),
    "to",
    fit_data["date"].max()
)


print("\nEarly stopping period:")

print(
    early_stop_data["date"].min(),
    "to",
    early_stop_data["date"].max()
)


X_fit = (
    fit_data[
        MODEL_FEATURES
    ]
    .astype("float32")
)


y_fit = (
    fit_data[
        TARGET
    ]
    .astype("float32")
)


X_early = (
    early_stop_data[
        MODEL_FEATURES
    ]
    .astype("float32")
)


y_early = (
    early_stop_data[
        TARGET
    ]
    .astype("float32")
)


X_full = (
    train[
        MODEL_FEATURES
    ]
    .astype("float32")
)


y_full = (
    train[
        TARGET
    ]
    .astype("float32")
)


# ============================================================
# STEP 6 - TRAIN XGBOOST WITH EARLY STOPPING
# ============================================================

print("\n" + "=" * 80)
print("STEP 6 - TRAINING XGBOOST")
print("=" * 80)


xgb_tuning_model = XGBRegressor(

    objective="reg:squarederror",

    n_estimators=1500,

    learning_rate=0.03,

    max_depth=8,

    min_child_weight=5,

    subsample=0.8,

    colsample_bytree=0.8,

    reg_alpha=0.05,

    reg_lambda=1.0,

    tree_method="hist",

    eval_metric="rmse",

    early_stopping_rounds=
        EARLY_STOPPING_ROUNDS,

    random_state=RANDOM_STATE,

    n_jobs=-1
)


xgb_tuning_model.fit(

    X_fit,
    y_fit,

    eval_set=[
        (
            X_early,
            y_early
        )
    ],

    verbose=False
)


if hasattr(
    xgb_tuning_model,
    "best_iteration"
):

    xgb_best_estimators = (
        xgb_tuning_model.best_iteration
        + 1
    )

else:

    xgb_best_estimators = 700


print(
    "\nBest XGBoost trees:",
    xgb_best_estimators
)


# Retrain on entire training history

xgb_model = XGBRegressor(

    objective="reg:squarederror",

    n_estimators=
        xgb_best_estimators,

    learning_rate=0.03,

    max_depth=8,

    min_child_weight=5,

    subsample=0.8,

    colsample_bytree=0.8,

    reg_alpha=0.05,

    reg_lambda=1.0,

    tree_method="hist",

    eval_metric="rmse",

    random_state=
        RANDOM_STATE,

    n_jobs=-1
)


xgb_model.fit(
    X_full,
    y_full,
    verbose=False
)


xgb_model.save_model(
    MODEL_DIR
    / "xgboost_demand_model.json"
)


# ============================================================
# STEP 7 - TRAIN LIGHTGBM
# ============================================================

print("\n" + "=" * 80)
print("STEP 7 - TRAINING LIGHTGBM")
print("=" * 80)


lgb_tuning_model = LGBMRegressor(

    objective="regression",

    n_estimators=2000,

    learning_rate=0.03,

    num_leaves=63,

    max_depth=-1,

    min_child_samples=30,

    subsample=0.8,

    colsample_bytree=0.8,

    reg_alpha=0.05,

    reg_lambda=0.1,

    random_state=
        RANDOM_STATE,

    n_jobs=-1,

    verbosity=-1
)


lgb_tuning_model.fit(

    X_fit,
    y_fit,

    eval_set=[
        (
            X_early,
            y_early
        )
    ],

    eval_metric="rmse",

    callbacks=[

        lgb.early_stopping(
            EARLY_STOPPING_ROUNDS,
            verbose=False
        ),

        lgb.log_evaluation(0)
    ]
)


if (
    lgb_tuning_model.best_iteration_
    is not None
    and
    lgb_tuning_model.best_iteration_ > 0
):

    lgb_best_estimators = (
        lgb_tuning_model
        .best_iteration_
    )

else:

    lgb_best_estimators = 700


print(
    "\nBest LightGBM trees:",
    lgb_best_estimators
)


lgb_model = LGBMRegressor(

    objective="regression",

    n_estimators=
        lgb_best_estimators,

    learning_rate=0.03,

    num_leaves=63,

    max_depth=-1,

    min_child_samples=30,

    subsample=0.8,

    colsample_bytree=0.8,

    reg_alpha=0.05,

    reg_lambda=0.1,

    random_state=
        RANDOM_STATE,

    n_jobs=-1,

    verbosity=-1
)


lgb_model.fit(
    X_full,
    y_full
)


lgb_model.booster_.save_model(
    str(
        MODEL_DIR
        / "lightgbm_demand_model.txt"
    )
)


# ============================================================
# STEP 8 - TRAIN CATBOOST
# ============================================================

print("\n" + "=" * 80)
print("STEP 8 - TRAINING CATBOOST")
print("=" * 80)


cat_tuning_model = CatBoostRegressor(

    iterations=1800,

    learning_rate=0.03,

    depth=8,

    loss_function="RMSE",

    eval_metric="RMSE",

    l2_leaf_reg=5,

    random_strength=1,

    random_seed=
        RANDOM_STATE,

    verbose=False,

    allow_writing_files=False
)


cat_tuning_model.fit(

    X_fit,
    y_fit,

    eval_set=(
        X_early,
        y_early
    ),

    early_stopping_rounds=
        EARLY_STOPPING_ROUNDS,

    use_best_model=True,

    verbose=False
)


cat_best_iteration = (
    cat_tuning_model
    .get_best_iteration()
)


if cat_best_iteration >= 0:

    cat_best_iterations = (
        cat_best_iteration
        + 1
    )

else:

    cat_best_iterations = 700


print(
    "\nBest CatBoost trees:",
    cat_best_iterations
)


cat_model = CatBoostRegressor(

    iterations=
        cat_best_iterations,

    learning_rate=0.03,

    depth=8,

    loss_function="RMSE",

    l2_leaf_reg=5,

    random_strength=1,

    random_seed=
        RANDOM_STATE,

    verbose=False,

    allow_writing_files=False
)


cat_model.fit(
    X_full,
    y_full,
    verbose=False
)


cat_model.save_model(
    str(
        MODEL_DIR
        / "catboost_demand_model.cbm"
    )
)


# ============================================================
# STEP 9 - HELPER FUNCTIONS FOR RECURSIVE FEATURES
# ============================================================

print("\n" + "=" * 80)
print("STEP 9 - PREPARING RECURSIVE FORECAST ENGINE")
print("=" * 80)


GROUP_KEYS = [
    "store_id",
    "item_id"
]


def lag_value(
    history,
    lag
):

    if len(history) >= lag:
        return float(
            history[-lag]
        )

    return np.nan


def rolling_mean(
    history,
    window
):

    if len(history) == 0:
        return np.nan

    values = history[
        -min(
            window,
            len(history)
        ):
    ]

    return float(
        np.mean(values)
    )


def rolling_std(
    history,
    window
):

    values = history[
        -min(
            window,
            len(history)
        ):
    ]

    if len(values) < 2:
        return 0.0

    return float(
        np.std(
            values,
            ddof=1
        )
    )


def rolling_min(
    history,
    window
):

    if len(history) == 0:
        return np.nan

    return float(
        np.min(
            history[-window:]
        )
    )


def rolling_max(
    history,
    window
):

    if len(history) == 0:
        return np.nan

    return float(
        np.max(
            history[-window:]
        )
    )


def zero_rate(
    history,
    window=28
):

    if len(history) == 0:
        return 0.0

    values = np.asarray(
        history[-window:],
        dtype=float
    )

    return float(
        np.mean(
            values < 0.5
        )
    )


# ============================================================
# STEP 10 - RECURSIVE FORECAST FUNCTION
# ============================================================

def recursive_forecast(
    model,
    model_name
):

    print(
        f"\nGenerating recursive "
        f"28-day forecast: {model_name}"
    )


    # --------------------------------------------------------
    # INITIAL SALES HISTORY
    # --------------------------------------------------------

    sales_history = {}

    price_history = {}


    for key, group in train.groupby(
        GROUP_KEYS,
        observed=True
    ):

        group = group.sort_values(
            "date"
        )


        sales_history[
            key
        ] = (
            group["sales"]
            .astype(float)
            .tolist()
        )


        price_history[
            key
        ] = (
            group["sell_price"]
            .astype(float)
            .tolist()
        )


    forecast_results = []


    forecast_dates = sorted(
        validation[
            "date"
        ].unique()
    )


    for forecast_date in forecast_dates:

        current_day = validation[
            validation["date"]
            ==
            forecast_date
        ].copy()


        feature_rows = []

        row_keys = []


        for _, row in current_day.iterrows():

            key = (
                row["store_id"],
                row["item_id"]
            )


            history = (
                sales_history.get(
                    key,
                    []
                )
            )


            prices = (
                price_history.get(
                    key,
                    []
                )
            )


            # -----------------------------------------------
            # SALES LAGS
            # -----------------------------------------------

            lag_1 = lag_value(
                history,
                1
            )

            lag_7 = lag_value(
                history,
                7
            )

            lag_14 = lag_value(
                history,
                14
            )

            lag_28 = lag_value(
                history,
                28
            )

            lag_56 = lag_value(
                history,
                56
            )


            # -----------------------------------------------
            # ROLLING STATISTICS
            # -----------------------------------------------

            rm7 = rolling_mean(
                history,
                7
            )

            rm14 = rolling_mean(
                history,
                14
            )

            rm28 = rolling_mean(
                history,
                28
            )

            rm56 = rolling_mean(
                history,
                56
            )


            rs7 = rolling_std(
                history,
                7
            )

            rs14 = rolling_std(
                history,
                14
            )

            rs28 = rolling_std(
                history,
                28
            )

            rs56 = rolling_std(
                history,
                56
            )


            rmin7 = rolling_min(
                history,
                7
            )

            rmin28 = rolling_min(
                history,
                28
            )


            rmax7 = rolling_max(
                history,
                7
            )

            rmax28 = rolling_max(
                history,
                28
            )


            zr28 = zero_rate(
                history,
                28
            )


            # -----------------------------------------------
            # PRICE FEATURES
            # -----------------------------------------------

            current_price = float(
                row[
                    "sell_price"
                ]
            )


            if len(prices) > 0:

                previous_price = (
                    prices[-1]
                )

                historical_price_mean = (
                    np.mean(
                        prices
                    )
                )

            else:

                previous_price = (
                    current_price
                )

                historical_price_mean = (
                    current_price
                )


            price_change = (
                current_price
                -
                previous_price
            )


            if previous_price != 0:

                price_change_pct = (
                    price_change
                    /
                    previous_price
                )

            else:

                price_change_pct = 0


            if historical_price_mean > 0:

                price_relative = (
                    current_price
                    /
                    historical_price_mean
                )

            else:

                price_relative = 1


            # -----------------------------------------------
            # DATE FEATURES
            # -----------------------------------------------

            current_date = pd.Timestamp(
                forecast_date
            )


            day_of_week = (
                current_date.dayofweek
            )

            day_of_month = (
                current_date.day
            )

            week_of_year = int(
                current_date
                .isocalendar()
                .week
            )

            month_num = (
                current_date.month
            )

            quarter = (
                current_date.quarter
            )

            year_num = (
                current_date.year
            )


            is_weekend = int(
                day_of_week
                in [5, 6]
            )

            is_month_start = int(
                current_date
                .is_month_start
            )

            is_month_end = int(
                current_date
                .is_month_end
            )


            dow_sin = np.sin(
                2
                *
                np.pi
                *
                day_of_week
                /
                7
            )


            dow_cos = np.cos(
                2
                *
                np.pi
                *
                day_of_week
                /
                7
            )


            month_sin = np.sin(
                2
                *
                np.pi
                *
                month_num
                /
                12
            )


            month_cos = np.cos(
                2
                *
                np.pi
                *
                month_num
                /
                12
            )


            # -----------------------------------------------
            # DEMAND DYNAMICS
            # -----------------------------------------------

            demand_trend = (
                rm7
                -
                rm28
            )


            demand_ratio = (
                rm7
                /
                (
                    rm28
                    +
                    1e-6
                )
            )


            demand_cv = (
                rs28
                /
                (
                    rm28
                    +
                    1e-6
                )
            )


            # -----------------------------------------------
            # ENCODE IDENTIFIERS
            # -----------------------------------------------

            item_code = (
                encoding_maps[
                    "item_id"
                ]
                .get(
                    str(
                        row[
                            "item_id"
                        ]
                    ),
                    -1
                )
            )


            dept_code = (
                encoding_maps[
                    "dept_id"
                ]
                .get(
                    str(
                        row[
                            "dept_id"
                        ]
                    ),
                    -1
                )
            )


            cat_code = (
                encoding_maps[
                    "cat_id"
                ]
                .get(
                    str(
                        row[
                            "cat_id"
                        ]
                    ),
                    -1
                )
            )


            store_code = (
                encoding_maps[
                    "store_id"
                ]
                .get(
                    str(
                        row[
                            "store_id"
                        ]
                    ),
                    -1
                )
            )


            state_code = (
                encoding_maps[
                    "state_id"
                ]
                .get(
                    str(
                        row[
                            "state_id"
                        ]
                    ),
                    -1
                )
            )


            # -----------------------------------------------
            # CREATE FEATURE ROW
            # -----------------------------------------------

            features = {

                "item_id_code":
                    item_code,

                "dept_id_code":
                    dept_code,

                "cat_id_code":
                    cat_code,

                "store_id_code":
                    store_code,

                "state_id_code":
                    state_code,

                "lag_1":
                    lag_1,

                "lag_7":
                    lag_7,

                "lag_14":
                    lag_14,

                "lag_28":
                    lag_28,

                "lag_56":
                    lag_56,

                "rolling_mean_7":
                    rm7,

                "rolling_mean_14":
                    rm14,

                "rolling_mean_28":
                    rm28,

                "rolling_mean_56":
                    rm56,

                "rolling_std_7":
                    rs7,

                "rolling_std_14":
                    rs14,

                "rolling_std_28":
                    rs28,

                "rolling_std_56":
                    rs56,

                "rolling_min_7":
                    rmin7,

                "rolling_min_28":
                    rmin28,

                "rolling_max_7":
                    rmax7,

                "rolling_max_28":
                    rmax28,

                "zero_sales_rate_28":
                    zr28,

                "sell_price":
                    current_price,

                "price_change":
                    price_change,

                "price_change_pct":
                    price_change_pct,

                "price_vs_historical_mean":
                    price_relative,

                "day_of_week":
                    day_of_week,

                "day_of_month":
                    day_of_month,

                "week_of_year":
                    week_of_year,

                "month_num":
                    month_num,

                "quarter":
                    quarter,

                "year_num":
                    year_num,

                "is_weekend":
                    is_weekend,

                "is_month_start":
                    is_month_start,

                "is_month_end":
                    is_month_end,

                "dow_sin":
                    dow_sin,

                "dow_cos":
                    dow_cos,

                "month_sin":
                    month_sin,

                "month_cos":
                    month_cos,

                "snap":
                    float(
                        row["snap"]
                    ),

                "has_event":
                    float(
                        row["has_event"]
                    ),

                "days_since_launch":
                    float(
                        row[
                            "days_since_launch"
                        ]
                    ),

                "demand_trend_7_28":
                    demand_trend,

                "demand_ratio_7_28":
                    demand_ratio,

                "demand_cv_28":
                    demand_cv
            }


            feature_rows.append(
                features
            )

            row_keys.append(
                key
            )


        # ----------------------------------------------------
        # PREDICT ALL ITEMS FOR CURRENT DAY TOGETHER
        # ----------------------------------------------------

        X_day = pd.DataFrame(
            feature_rows
        )[
            MODEL_FEATURES
        ].astype(
            "float32"
        )


        predictions = model.predict(
            X_day
        )


        predictions = np.asarray(
            predictions,
            dtype=float
        )


        # Demand cannot be negative.

        predictions = np.clip(
            predictions,
            0,
            None
        )


        # ----------------------------------------------------
        # UPDATE HISTORY USING PREDICTIONS
        # ----------------------------------------------------

        for (
            (_, row),
            key,
            prediction
        ) in zip(

            current_day.iterrows(),
            row_keys,
            predictions
        ):

            sales_history.setdefault(
                key,
                []
            ).append(
                float(
                    prediction
                )
            )


            price_history.setdefault(
                key,
                []
            ).append(
                float(
                    row[
                        "sell_price"
                    ]
                )
            )


            forecast_results.append(
                {

                    "date":
                        row["date"],

                    "store_id":
                        row["store_id"],

                    "item_id":
                        row["item_id"],

                    "actual_sales":
                        float(
                            row["sales"]
                        ),

                    "prediction":
                        float(
                            prediction
                        )
                }
            )


    result = pd.DataFrame(
        forecast_results
    )


    print(
        f"{model_name} recursive "
        f"forecast completed."
    )


    return result


# ============================================================
# STEP 11 - GENERATE RECURSIVE FORECASTS
# ============================================================

print("\n" + "=" * 80)
print("STEP 11 - GENERATING 28-DAY FORECASTS")
print("=" * 80)


xgb_forecast = recursive_forecast(
    xgb_model,
    "XGBoost"
)


lgb_forecast = recursive_forecast(
    lgb_model,
    "LightGBM"
)


cat_forecast = recursive_forecast(
    cat_model,
    "CatBoost"
)


# ============================================================
# STEP 12 - COMBINE FORECASTS
# ============================================================

prediction_df = (
    xgb_forecast[
        [
            "date",
            "store_id",
            "item_id",
            "actual_sales",
            "prediction"
        ]
    ]
    .rename(
        columns={
            "prediction":
            "pred_xgboost"
        }
    )
)


prediction_df = prediction_df.merge(

    lgb_forecast[
        [
            "date",
            "store_id",
            "item_id",
            "prediction"
        ]
    ].rename(
        columns={
            "prediction":
            "pred_lightgbm"
        }
    ),

    on=[
        "date",
        "store_id",
        "item_id"
    ],

    how="left"
)


prediction_df = prediction_df.merge(

    cat_forecast[
        [
            "date",
            "store_id",
            "item_id",
            "prediction"
        ]
    ].rename(
        columns={
            "prediction":
            "pred_catboost"
        }
    ),

    on=[
        "date",
        "store_id",
        "item_id"
    ],

    how="left"
)


# ============================================================
# STEP 13 - METRIC FUNCTIONS
# ============================================================

def mae(
    y_true,
    y_pred
):

    return np.mean(
        np.abs(
            y_true
            -
            y_pred
        )
    )


def rmse(
    y_true,
    y_pred
):

    return np.sqrt(
        np.mean(
            (
                y_true
                -
                y_pred
            ) ** 2
        )
    )


def wape(
    y_true,
    y_pred
):

    denominator = (
        np.sum(
            np.abs(
                y_true
            )
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


def rmsle(
    y_true,
    y_pred
):

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
                np.log1p(
                    y_pred
                )
                -
                np.log1p(
                    y_true
                )
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
# STEP 14 - EVALUATE MODELS
# ============================================================

print("\n" + "=" * 80)
print("STEP 14 - MODEL EVALUATION")
print("=" * 80)


actual = (
    prediction_df[
        "actual_sales"
    ]
    .to_numpy()
)


model_predictions = {

    "XGBoost":
        "pred_xgboost",

    "LightGBM":
        "pred_lightgbm",

    "CatBoost":
        "pred_catboost"
}


metric_rows = []


for (
    model_name,
    column
) in model_predictions.items():

    predicted = (
        prediction_df[
            column
        ]
        .to_numpy()
    )


    metric_rows.append(
        {

            "model":
                model_name,

            "MAE":
                mae(
                    actual,
                    predicted
                ),

            "RMSE":
                rmse(
                    actual,
                    predicted
                ),

            "WAPE_percent":
                wape(
                    actual,
                    predicted
                ),

            "RMSLE":
                rmsle(
                    actual,
                    predicted
                ),

            "Forecast_Bias":
                forecast_bias(
                    actual,
                    predicted
                )
        }
    )


advanced_metrics = (
    pd.DataFrame(
        metric_rows
    )
    .sort_values(
        "WAPE_percent"
    )
    .reset_index(
        drop=True
    )
)


print("\nAdvanced ML Results:")

print(
    advanced_metrics.round(4)
)


# ============================================================
# STEP 15 - BEST ADVANCED MODEL
# ============================================================

best_model_name = (
    advanced_metrics
    .iloc[0]["model"]
)


best_model_wape = (
    advanced_metrics
    .iloc[0]["WAPE_percent"]
)


print(
    "\nBest advanced model:"
)

print(
    best_model_name
)


print(
    "\nBest advanced WAPE:"
)

print(
    f"{best_model_wape:.2f}%"
)


# ============================================================
# STEP 16 - SAVE PREDICTIONS AND METRICS
# ============================================================

prediction_df.to_parquet(
    OUTPUT_PREDICTIONS,
    index=False
)


advanced_metrics.to_csv(
    ADVANCED_METRICS_FILE,
    index=False
)


# ============================================================
# STEP 17 - COMPARE WITH BASELINES
# ============================================================

print("\n" + "=" * 80)
print("STEP 17 - BASELINE VS ADVANCED ML")
print("=" * 80)


if BASELINE_METRICS_FILE.exists():

    baseline_metrics = (
        pd.read_csv(
            BASELINE_METRICS_FILE
        )
    )


    baseline_metrics[
        "model_type"
    ] = "Baseline"


    advanced_comparison = (
        advanced_metrics.copy()
    )


    advanced_comparison[
        "model_type"
    ] = "Advanced ML"


    comparison = pd.concat(
        [
            baseline_metrics,
            advanced_comparison
        ],
        ignore_index=True
    )


else:

    print(
        "\nBaseline metrics not found."
    )

    comparison = (
        advanced_metrics.copy()
    )

    comparison[
        "model_type"
    ] = "Advanced ML"


comparison = (
    comparison
    .sort_values(
        "WAPE_percent"
    )
    .reset_index(
        drop=True
    )
)


comparison.to_csv(
    COMPARISON_FILE,
    index=False
)


print("\nOverall model ranking:")

print(
    comparison[
        [
            "model",
            "model_type",
            "MAE",
            "RMSE",
            "WAPE_percent",
            "Forecast_Bias"
        ]
    ].round(4)
)


# ============================================================
# STEP 18 - FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 80)
print("STEP 18 - FEATURE IMPORTANCE")
print("=" * 80)


importance_tables = {}


xgb_importance = pd.DataFrame(
    {
        "feature":
            MODEL_FEATURES,

        "importance":
            xgb_model
            .feature_importances_
    }
).sort_values(
    "importance",
    ascending=False
)


lgb_importance = pd.DataFrame(
    {
        "feature":
            MODEL_FEATURES,

        "importance":
            lgb_model
            .feature_importances_
    }
).sort_values(
    "importance",
    ascending=False
)


cat_importance = pd.DataFrame(
    {
        "feature":
            MODEL_FEATURES,

        "importance":
            cat_model
            .get_feature_importance()
    }
).sort_values(
    "importance",
    ascending=False
)


importance_tables[
    "XGBoost"
] = xgb_importance

importance_tables[
    "LightGBM"
] = lgb_importance

importance_tables[
    "CatBoost"
] = cat_importance


xgb_importance.to_csv(
    TABLE_DIR
    / "xgboost_feature_importance.csv",
    index=False
)


lgb_importance.to_csv(
    TABLE_DIR
    / "lightgbm_feature_importance.csv",
    index=False
)


cat_importance.to_csv(
    TABLE_DIR
    / "catboost_feature_importance.csv",
    index=False
)


best_importance = (
    importance_tables[
        best_model_name
    ]
    .head(20)
    .sort_values(
        "importance"
    )
)


plt.figure(
    figsize=(11, 8)
)


plt.barh(
    best_importance[
        "feature"
    ],
    best_importance[
        "importance"
    ]
)


plt.title(
    f"Top 20 Features - {best_model_name}"
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    / "20_best_ml_feature_importance.png",
    dpi=300
)


plt.close()


# ============================================================
# STEP 19 - MODEL COMPARISON CHART
# ============================================================

comparison_plot = (
    comparison
    .sort_values(
        "WAPE_percent",
        ascending=False
    )
)


plt.figure(
    figsize=(11, 7)
)


plt.barh(
    comparison_plot[
        "model"
    ],
    comparison_plot[
        "WAPE_percent"
    ]
)


plt.title(
    "Baseline vs Advanced ML Forecasting"
)

plt.xlabel(
    "WAPE (%) - Lower is Better"
)

plt.ylabel(
    "Model"
)

plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    / "21_baseline_vs_ml_comparison.png",
    dpi=300
)


plt.close()


# ============================================================
# STEP 20 - ACTUAL VS BEST MODEL
# ============================================================

best_prediction_column = {

    "XGBoost":
        "pred_xgboost",

    "LightGBM":
        "pred_lightgbm",

    "CatBoost":
        "pred_catboost"

}[
    best_model_name
]


daily_comparison = (

    prediction_df
    .groupby(
        "date",
        observed=True
    )
    [
        [
            "actual_sales",
            best_prediction_column
        ]
    ]
    .sum()
    .reset_index()
)


plt.figure(
    figsize=(14, 7)
)


plt.plot(

    daily_comparison[
        "date"
    ],

    daily_comparison[
        "actual_sales"
    ],

    marker="o",

    label="Actual Demand"
)


plt.plot(

    daily_comparison[
        "date"
    ],

    daily_comparison[
        best_prediction_column
    ],

    marker="o",

    label=
        f"{best_model_name} Forecast"
)


plt.title(
    f"Actual vs {best_model_name} "
    f"28-Day Recursive Forecast"
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
    FIGURE_DIR
    / "22_best_ml_actual_vs_forecast.png",
    dpi=300
)


plt.close()


# ============================================================
# STEP 21 - IMPROVEMENT OVER BEST BASELINE
# ============================================================

if BASELINE_METRICS_FILE.exists():

    baseline_metrics = (
        pd.read_csv(
            BASELINE_METRICS_FILE
        )
    )


    best_baseline_wape = (
        baseline_metrics[
            "WAPE_percent"
        ].min()
    )


    improvement = (

        (
            best_baseline_wape
            -
            best_model_wape
        )

        /

        best_baseline_wape

        *

        100
    )


    print(
        "\nBest baseline WAPE:"
    )

    print(
        f"{best_baseline_wape:.2f}%"
    )


    print(
        "\nBest ML WAPE:"
    )

    print(
        f"{best_model_wape:.2f}%"
    )


    print(
        "\nRelative WAPE improvement:"
    )

    print(
        f"{improvement:.2f}%"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("STAGE 7 SUMMARY")
print("=" * 80)


print(
    "\nModels trained:"
)

print(
    "XGBoost"
)

print(
    "LightGBM"
)

print(
    "CatBoost"
)


print(
    "\nBest model:"
)

print(
    best_model_name
)


print(
    "\nBest model WAPE:"
)

print(
    f"{best_model_wape:.2f}%"
)


print(
    "\nValidation method:"
)

print(
    "28-day recursive forecasting"
)


print(
    "\nFuture validation sales "
    "were NOT used as lag features."
)


print(
    "\nThe final test dataset "
    "has NOT been touched."
)


print("\n" + "=" * 80)
print("STAGE 7 COMPLETED SUCCESSFULLY")
print("=" * 80)