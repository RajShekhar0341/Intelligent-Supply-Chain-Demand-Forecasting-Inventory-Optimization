from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import optuna
import shap

from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

FORECAST_HORIZON = 28

# Start small on a normal laptop.
# Once everything works, increase to 25-40.
N_TRIALS = 10

N_CV_FOLDS = 2

# Use the most recent 2 years for tuning model fitting.
# Historical data before this can still be used as lag history.
TUNING_LOOKBACK_DAYS = 730

SHAP_SAMPLE_SIZE = 2000


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


for directory in [
    MODEL_DIR,
    TABLE_DIR,
    FIGURE_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


TRAIN_FILE = (
    PROCESSED_DIR
    / "m5_ca1_train.parquet"
)

VALIDATION_FILE = (
    PROCESSED_DIR
    / "m5_ca1_validation.parquet"
)

TEST_FILE = (
    PROCESSED_DIR
    / "m5_ca1_test.parquet"
)

STAGE7_METRICS_FILE = (
    TABLE_DIR
    / "advanced_ml_metrics.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

OPTUNA_TRIALS_FILE = (
    TABLE_DIR
    / "stage8_optuna_trials.csv"
)

BEST_PARAMS_FILE = (
    MODEL_DIR
    / "stage8_best_parameters.json"
)

VALIDATION_METRICS_FILE = (
    TABLE_DIR
    / "stage8_tuned_validation_metrics.csv"
)

VALIDATION_PREDICTIONS_FILE = (
    PROCESSED_DIR
    / "stage8_tuned_validation_predictions.parquet"
)

TEST_METRICS_FILE = (
    TABLE_DIR
    / "stage8_final_test_metrics.csv"
)

TEST_PREDICTIONS_FILE = (
    PROCESSED_DIR
    / "stage8_final_test_predictions.parquet"
)

SHAP_IMPORTANCE_FILE = (
    TABLE_DIR
    / "stage8_shap_global_importance.csv"
)

SHAP_LOCAL_FILE = (
    TABLE_DIR
    / "stage8_shap_local_explanation.csv"
)

METADATA_FILE = (
    MODEL_DIR
    / "stage8_final_model_metadata.json"
)


# ============================================================
# START
# ============================================================

print("=" * 85)
print("STAGE 8 - MODEL OPTIMIZATION, EXPLAINABILITY AND FINAL TEST")
print("=" * 85)


# ============================================================
# STEP 1 - CHECK FILES
# ============================================================

required_files = [
    TRAIN_FILE,
    VALIDATION_FILE,
    TEST_FILE,
    STAGE7_METRICS_FILE
]


for file in required_files:

    if not file.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{file}\n\n"
            "Complete Stage 7 before running Stage 8."
        )


# ============================================================
# STEP 2 - LOAD DATA
# ============================================================

print("\nLoading datasets...")


train = pd.read_parquet(
    TRAIN_FILE
)

validation = pd.read_parquet(
    VALIDATION_FILE
)

test = pd.read_parquet(
    TEST_FILE
)


for dataframe in [
    train,
    validation,
    test
]:

    dataframe["date"] = pd.to_datetime(
        dataframe["date"]
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


test = test.sort_values(
    [
        "store_id",
        "item_id",
        "date"
    ]
).reset_index(drop=True)


print("\nTraining:")
print(train.shape)

print(
    train["date"].min(),
    "to",
    train["date"].max()
)


print("\nValidation:")
print(validation.shape)

print(
    validation["date"].min(),
    "to",
    validation["date"].max()
)


print("\nTest:")
print(test.shape)

print(
    test["date"].min(),
    "to",
    test["date"].max()
)


# ============================================================
# STEP 3 - IDENTIFY STAGE 7 WINNER
# ============================================================

print("\n" + "=" * 85)
print("STEP 3 - READING STAGE 7 WINNING MODEL")
print("=" * 85)


stage7_metrics = pd.read_csv(
    STAGE7_METRICS_FILE
)


stage7_metrics = stage7_metrics.sort_values(
    "WAPE_percent"
)


BEST_MODEL_NAME = (
    stage7_metrics.iloc[0]["model"]
)


STAGE7_BEST_WAPE = float(
    stage7_metrics.iloc[0][
        "WAPE_percent"
    ]
)


print(
    "\nStage 7 winner:",
    BEST_MODEL_NAME
)

print(
    "Stage 7 validation WAPE:",
    f"{STAGE7_BEST_WAPE:.2f}%"
)


if BEST_MODEL_NAME not in [
    "XGBoost",
    "LightGBM",
    "CatBoost"
]:

    raise ValueError(
        f"Unsupported Stage 7 winner: "
        f"{BEST_MODEL_NAME}"
    )


# ============================================================
# STEP 4 - ENCODE IDENTIFIERS
# ============================================================

print("\n" + "=" * 85)
print("STEP 4 - PREPARING IDENTIFIER ENCODINGS")
print("=" * 85)


ID_COLUMNS = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id"
]


encoding_maps = {}


for column in ID_COLUMNS:

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


def apply_encodings(df):

    result = df.copy()

    for column in ID_COLUMNS:

        result[
            f"{column}_code"
        ] = (
            result[column]
            .astype(str)
            .map(
                encoding_maps[
                    column
                ]
            )
            .fillna(-1)
            .astype("int32")
        )

    return result


train = apply_encodings(
    train
)

validation = apply_encodings(
    validation
)

test = apply_encodings(
    test
)


# ============================================================
# STEP 5 - MODEL FEATURES
# ============================================================

MODEL_FEATURES = [

    # Identifiers

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

    # Rolling volatility

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

    # Cyclical

    "dow_sin",
    "dow_cos",

    "month_sin",
    "month_cos",

    # External

    "snap",
    "has_event",

    # Product age

    "days_since_launch",

    # Demand dynamics

    "demand_trend_7_28",
    "demand_ratio_7_28",
    "demand_cv_28"
]


TARGET = "sales"


print(
    "\nNumber of model features:",
    len(MODEL_FEATURES)
)


# ============================================================
# STEP 6 - METRIC FUNCTIONS
# ============================================================

def mae(y_true, y_pred):

    return float(
        np.mean(
            np.abs(
                y_true - y_pred
            )
        )
    )


def rmse(y_true, y_pred):

    return float(
        np.sqrt(
            np.mean(
                (
                    y_true
                    - y_pred
                ) ** 2
            )
        )
    )


def wape(y_true, y_pred):

    denominator = np.sum(
        np.abs(y_true)
    )

    if denominator == 0:

        return np.nan

    return float(
        np.sum(
            np.abs(
                y_true
                - y_pred
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

    return float(
        np.sqrt(
            np.mean(
                (
                    np.log1p(y_pred)
                    -
                    np.log1p(y_true)
                ) ** 2
            )
        )
    )


def forecast_bias(
    y_true,
    y_pred
):

    return float(
        np.mean(
            y_pred
            -
            y_true
        )
    )


# ============================================================
# STEP 7 - MODEL BUILDER
# ============================================================

def build_model(
    model_name,
    params
):

    if model_name == "XGBoost":

        model = XGBRegressor(

            objective=
                "reg:squarederror",

            tree_method="hist",

            eval_metric="rmse",

            random_state=
                RANDOM_STATE,

            n_jobs=-1,

            **params
        )


    elif model_name == "LightGBM":

        model = LGBMRegressor(

            objective=
                "regression",

            random_state=
                RANDOM_STATE,

            n_jobs=-1,

            verbosity=-1,

            **params
        )


    elif model_name == "CatBoost":

        model = CatBoostRegressor(

            loss_function=
                "RMSE",

            random_seed=
                RANDOM_STATE,

            verbose=False,

            allow_writing_files=
                False,

            thread_count=-1,

            **params
        )


    else:

        raise ValueError(
            f"Unknown model: "
            f"{model_name}"
        )


    return model


# ============================================================
# STEP 8 - OPTUNA SEARCH SPACE
# ============================================================

def get_trial_parameters(
    trial,
    model_name
):

    if model_name == "XGBoost":

        return {

            "n_estimators":
                trial.suggest_int(
                    "n_estimators",
                    300,
                    1100,
                    step=100
                ),

            "learning_rate":
                trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.10,
                    log=True
                ),

            "max_depth":
                trial.suggest_int(
                    "max_depth",
                    4,
                    10
                ),

            "min_child_weight":
                trial.suggest_int(
                    "min_child_weight",
                    1,
                    20
                ),

            "subsample":
                trial.suggest_float(
                    "subsample",
                    0.65,
                    1.0
                ),

            "colsample_bytree":
                trial.suggest_float(
                    "colsample_bytree",
                    0.65,
                    1.0
                ),

            "reg_alpha":
                trial.suggest_float(
                    "reg_alpha",
                    1e-4,
                    2.0,
                    log=True
                ),

            "reg_lambda":
                trial.suggest_float(
                    "reg_lambda",
                    1e-3,
                    10.0,
                    log=True
                )
        }


    elif model_name == "LightGBM":

        return {

            "n_estimators":
                trial.suggest_int(
                    "n_estimators",
                    300,
                    1100,
                    step=100
                ),

            "learning_rate":
                trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.10,
                    log=True
                ),

            "num_leaves":
                trial.suggest_int(
                    "num_leaves",
                    25,
                    127
                ),

            "max_depth":
                trial.suggest_int(
                    "max_depth",
                    5,
                    12
                ),

            "min_child_samples":
                trial.suggest_int(
                    "min_child_samples",
                    10,
                    100
                ),

            "subsample":
                trial.suggest_float(
                    "subsample",
                    0.65,
                    1.0
                ),

            "colsample_bytree":
                trial.suggest_float(
                    "colsample_bytree",
                    0.65,
                    1.0
                ),

            "reg_alpha":
                trial.suggest_float(
                    "reg_alpha",
                    1e-4,
                    2.0,
                    log=True
                ),

            "reg_lambda":
                trial.suggest_float(
                    "reg_lambda",
                    1e-3,
                    10.0,
                    log=True
                )
        }


    elif model_name == "CatBoost":

        return {

            "iterations":
                trial.suggest_int(
                    "iterations",
                    300,
                    1100,
                    step=100
                ),

            "learning_rate":
                trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.10,
                    log=True
                ),

            "depth":
                trial.suggest_int(
                    "depth",
                    5,
                    10
                ),

            "l2_leaf_reg":
                trial.suggest_float(
                    "l2_leaf_reg",
                    1.0,
                    20.0,
                    log=True
                ),

            "random_strength":
                trial.suggest_float(
                    "random_strength",
                    0.0,
                    2.0
                )
        }


    raise ValueError(
        model_name
    )


# ============================================================
# STEP 9 - RECURSIVE FEATURE HELPERS
# ============================================================

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


def historical_mean(
    history,
    window
):

    if not history:

        return 0.0

    values = history[
        -min(
            window,
            len(history)
        ):
    ]

    return float(
        np.mean(values)
    )


def historical_std(
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


def historical_min(
    history,
    window
):

    if not history:

        return 0.0

    return float(
        np.min(
            history[-window:]
        )
    )


def historical_max(
    history,
    window
):

    if not history:

        return 0.0

    return float(
        np.max(
            history[-window:]
        )
    )


def zero_rate(
    history,
    window=28
):

    if not history:

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
# STEP 10 - RECURSIVE FORECAST ENGINE
# ============================================================

def recursive_forecast(
    model,
    history_df,
    future_df
):

    history_df = history_df.sort_values(
        [
            "store_id",
            "item_id",
            "date"
        ]
    )


    future_df = future_df.sort_values(
        [
            "date",
            "store_id",
            "item_id"
        ]
    )


    sales_history = {}

    price_history = {}


    for key, group in history_df.groupby(
        GROUP_KEYS,
        observed=True
    ):

        group = group.sort_values(
            "date"
        )


        sales_history[key] = (
            group["sales"]
            .astype(float)
            .tolist()
        )


        price_history[key] = (
            group["sell_price"]
            .astype(float)
            .tolist()
        )


    results = []


    forecast_dates = sorted(
        future_df[
            "date"
        ].unique()
    )


    for forecast_date in forecast_dates:

        current_day = future_df[
            future_df["date"]
            ==
            forecast_date
        ].copy()


        feature_rows = []

        key_rows = []


        for _, row in current_day.iterrows():

            key = (
                row["store_id"],
                row["item_id"]
            )


            history = (
                sales_history
                .get(
                    key,
                    []
                )
            )


            prices = (
                price_history
                .get(
                    key,
                    []
                )
            )


            # -----------------------------------------------
            # LAGS
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
            # ROLLING FEATURES
            # -----------------------------------------------

            rm7 = historical_mean(
                history,
                7
            )

            rm14 = historical_mean(
                history,
                14
            )

            rm28 = historical_mean(
                history,
                28
            )

            rm56 = historical_mean(
                history,
                56
            )


            rs7 = historical_std(
                history,
                7
            )

            rs14 = historical_std(
                history,
                14
            )

            rs28 = historical_std(
                history,
                28
            )

            rs56 = historical_std(
                history,
                56
            )


            rmin7 = historical_min(
                history,
                7
            )

            rmin28 = historical_min(
                history,
                28
            )


            rmax7 = historical_max(
                history,
                7
            )

            rmax28 = historical_max(
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
                row["sell_price"]
            )


            if prices:

                previous_price = float(
                    prices[-1]
                )

                history_price_mean = float(
                    np.mean(prices)
                )

            else:

                previous_price = (
                    current_price
                )

                history_price_mean = (
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

                price_change_pct = 0.0


            if history_price_mean > 0:

                price_relative = (
                    current_price
                    /
                    history_price_mean
                )

            else:

                price_relative = 1.0


            # -----------------------------------------------
            # CALENDAR
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


            features = {

                "item_id_code":
                    row[
                        "item_id_code"
                    ],

                "dept_id_code":
                    row[
                        "dept_id_code"
                    ],

                "cat_id_code":
                    row[
                        "cat_id_code"
                    ],

                "store_id_code":
                    row[
                        "store_id_code"
                    ],

                "state_id_code":
                    row[
                        "state_id_code"
                    ],


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

            key_rows.append(
                key
            )


        X_day = pd.DataFrame(
            feature_rows
        )


        X_day = (
            X_day[
                MODEL_FEATURES
            ]
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .fillna(0)
            .astype(
                "float32"
            )
        )


        predictions = model.predict(
            X_day
        )


        predictions = np.clip(
            np.asarray(
                predictions,
                dtype=float
            ),
            0,
            None
        )


        # -----------------------------------------------
        # UPDATE HISTORY WITH PREDICTIONS
        # -----------------------------------------------

        for (
            (_, row),
            key,
            prediction
        ) in zip(
            current_day.iterrows(),
            key_rows,
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


            results.append(
                {

                    "date":
                        row["date"],

                    "store_id":
                        row[
                            "store_id"
                        ],

                    "item_id":
                        row[
                            "item_id"
                        ],

                    "actual_sales":
                        float(
                            row[
                                "sales"
                            ]
                        ),

                    "prediction":
                        float(
                            prediction
                        )
                }
            )


    return pd.DataFrame(
        results
    )


# ============================================================
# STEP 11 - CREATE ROLLING CV FOLDS
# ============================================================

print("\n" + "=" * 85)
print("STEP 11 - BUILDING ROLLING TIME-SERIES FOLDS")
print("=" * 85)


def create_cv_folds(
    data,
    number_of_folds,
    horizon
):

    folds = []

    max_date = (
        data["date"].max()
    )


    for fold_number in range(
        number_of_folds,
        0,
        -1
    ):

        validation_end = (
            max_date
            -
            pd.Timedelta(
                days=(
                    fold_number
                    -
                    1
                )
                *
                horizon
            )
        )


        validation_start = (
            validation_end
            -
            pd.Timedelta(
                days=
                    horizon
                    -
                    1
            )
        )


        train_end = (
            validation_start
            -
            pd.Timedelta(
                days=1
            )
        )


        folds.append(
            {

                "train_end":
                    train_end,

                "validation_start":
                    validation_start,

                "validation_end":
                    validation_end
            }
        )


    return folds


cv_folds = create_cv_folds(
    train,
    N_CV_FOLDS,
    FORECAST_HORIZON
)


for index, fold in enumerate(
    cv_folds,
    start=1
):

    print(
        f"\nFold {index}:"
    )

    print(
        "Train through:",
        fold[
            "train_end"
        ]
    )

    print(
        "Forecast:",
        fold[
            "validation_start"
        ],
        "to",
        fold[
            "validation_end"
        ]
    )


# ============================================================
# STEP 12 - OPTUNA OBJECTIVE
# ============================================================

print("\n" + "=" * 85)
print("STEP 12 - STARTING OPTUNA OPTIMIZATION")
print("=" * 85)


def objective(trial):

    params = get_trial_parameters(
        trial,
        BEST_MODEL_NAME
    )


    fold_scores = []


    for fold_index, fold in enumerate(
        cv_folds
    ):

        history_source = train[
            train["date"]
            <=
            fold["train_end"]
        ].copy()


        tuning_start = (
            fold["train_end"]
            -
            pd.Timedelta(
                days=
                    TUNING_LOOKBACK_DAYS
            )
        )


        fitting_data = history_source[
            history_source["date"]
            >=
            tuning_start
        ].copy()


        fold_validation = train[
            (
                train["date"]
                >=
                fold[
                    "validation_start"
                ]
            )
            &
            (
                train["date"]
                <=
                fold[
                    "validation_end"
                ]
            )
        ].copy()


        X_fit = (
            fitting_data[
                MODEL_FEATURES
            ]
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .fillna(0)
            .astype(
                "float32"
            )
        )


        y_fit = (
            fitting_data[
                TARGET
            ]
            .astype(
                "float32"
            )
        )


        model = build_model(
            BEST_MODEL_NAME,
            params
        )


        model.fit(
            X_fit,
            y_fit
        )


        fold_predictions = (
            recursive_forecast(
                model,
                history_source,
                fold_validation
            )
        )


        score = wape(
            fold_predictions[
                "actual_sales"
            ].to_numpy(),

            fold_predictions[
                "prediction"
            ].to_numpy()
        )


        fold_scores.append(
            score
        )


        running_score = float(
            np.mean(
                fold_scores
            )
        )


        trial.report(
            running_score,
            step=fold_index
        )


        if trial.should_prune():

            raise optuna.TrialPruned()


    return float(
        np.mean(
            fold_scores
        )
    )


study = optuna.create_study(

    direction="minimize",

    sampler=TPESampler(
        seed=
            RANDOM_STATE
    ),

    pruner=MedianPruner(
        n_startup_trials=4,
        n_warmup_steps=1
    ),

    study_name=(
        "supply_chain_demand_"
        "optimization"
    )
)


study.optimize(
    objective,
    n_trials=N_TRIALS
)


# ============================================================
# STEP 13 - SAVE OPTUNA RESULTS
# ============================================================

trials_df = (
    study.trials_dataframe()
)


trials_df.to_csv(
    OPTUNA_TRIALS_FILE,
    index=False
)


best_params = (
    study.best_trial.params
)


print(
    "\nBest model:",
    BEST_MODEL_NAME
)


print(
    "\nBest CV WAPE:",
    f"{study.best_value:.2f}%"
)


print(
    "\nBest parameters:"
)


for key, value in best_params.items():

    print(
        f"{key}: {value}"
    )


with open(
    BEST_PARAMS_FILE,
    "w"
) as file:

    json.dump(
        {
            "model":
                BEST_MODEL_NAME,

            "best_cv_wape":
                study.best_value,

            "parameters":
                best_params
        },
        file,
        indent=4
    )


# ============================================================
# STEP 14 - OPTIMIZATION HISTORY GRAPH
# ============================================================

completed_trials = trials_df[
    trials_df[
        "state"
    ]
    ==
    "COMPLETE"
].copy()


if not completed_trials.empty:

    plt.figure(
        figsize=(11, 6)
    )


    plt.plot(

        completed_trials[
            "number"
        ],

        completed_trials[
            "value"
        ],

        marker="o"
    )


    plt.title(
        "Optuna Hyperparameter Optimization"
    )

    plt.xlabel(
        "Trial"
    )

    plt.ylabel(
        "Rolling-CV WAPE (%)"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plt.savefig(
        FIGURE_DIR
        / "23_optuna_optimization_history.png",
        dpi=300
    )


    plt.close()


# ============================================================
# STEP 15 - TRAIN TUNED MODEL ON FULL TRAIN SET
# ============================================================

print("\n" + "=" * 85)
print("STEP 15 - VALIDATING THE TUNED MODEL")
print("=" * 85)


X_train = (

    train[
        MODEL_FEATURES
    ]

    .replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    .fillna(0)

    .astype(
        "float32"
    )
)


y_train = (
    train[
        TARGET
    ]
    .astype(
        "float32"
    )
)


tuned_validation_model = (
    build_model(
        BEST_MODEL_NAME,
        best_params
    )
)


tuned_validation_model.fit(
    X_train,
    y_train
)


validation_predictions = (
    recursive_forecast(
        tuned_validation_model,
        train,
        validation
    )
)


validation_actual = (
    validation_predictions[
        "actual_sales"
    ]
    .to_numpy()
)


validation_predicted = (
    validation_predictions[
        "prediction"
    ]
    .to_numpy()
)


validation_metrics = {

    "model":
        BEST_MODEL_NAME,

    "MAE":
        mae(
            validation_actual,
            validation_predicted
        ),

    "RMSE":
        rmse(
            validation_actual,
            validation_predicted
        ),

    "WAPE_percent":
        wape(
            validation_actual,
            validation_predicted
        ),

    "RMSLE":
        rmsle(
            validation_actual,
            validation_predicted
        ),

    "Forecast_Bias":
        forecast_bias(
            validation_actual,
            validation_predicted
        )
}


validation_metrics_df = pd.DataFrame(
    [
        validation_metrics
    ]
)


validation_metrics_df.to_csv(
    VALIDATION_METRICS_FILE,
    index=False
)


validation_predictions.to_parquet(
    VALIDATION_PREDICTIONS_FILE,
    index=False
)


TUNED_VALIDATION_WAPE = (
    validation_metrics[
        "WAPE_percent"
    ]
)


print(
    "\nStage 7 WAPE:",
    f"{STAGE7_BEST_WAPE:.2f}%"
)


print(
    "Tuned validation WAPE:",
    f"{TUNED_VALIDATION_WAPE:.2f}%"
)


validation_improvement = (

    (
        STAGE7_BEST_WAPE
        -
        TUNED_VALIDATION_WAPE
    )

    /

    STAGE7_BEST_WAPE

    *

    100
)


print(
    "\nRelative improvement:",
    f"{validation_improvement:.2f}%"
)


# ============================================================
# STEP 16 - STAGE 7 VS TUNED MODEL CHART
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.bar(

    [
        "Stage 7",
        "Stage 8 Tuned"
    ],

    [
        STAGE7_BEST_WAPE,
        TUNED_VALIDATION_WAPE
    ]
)


plt.ylabel(
    "Validation WAPE (%)"
)

plt.title(
    f"{BEST_MODEL_NAME}: "
    f"Before vs After Optimization"
)

plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    / "24_before_after_tuning.png",
    dpi=300
)


plt.close()


# ============================================================
# STEP 17 - LOCK MODEL
# ============================================================

print("\n" + "=" * 85)
print("STEP 17 - LOCKING FINAL MODEL")
print("=" * 85)


# Validation period is now historical.
# We can train on Train + Validation
# before evaluating the untouched test set.

final_training_data = pd.concat(
    [
        train,
        validation
    ],
    ignore_index=True
)


final_training_data = (
    final_training_data
    .sort_values(
        [
            "store_id",
            "item_id",
            "date"
        ]
    )
    .reset_index(
        drop=True
    )
)


X_final_train = (

    final_training_data[
        MODEL_FEATURES
    ]

    .replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    .fillna(0)

    .astype(
        "float32"
    )
)


y_final_train = (
    final_training_data[
        TARGET
    ]
    .astype(
        "float32"
    )
)


final_model = build_model(
    BEST_MODEL_NAME,
    best_params
)


final_model.fit(
    X_final_train,
    y_final_train
)


# ============================================================
# STEP 18 - SAVE FINAL MODEL
# ============================================================

if BEST_MODEL_NAME == "XGBoost":

    FINAL_MODEL_FILE = (
        MODEL_DIR
        / "final_optimized_xgboost.json"
    )

    final_model.save_model(
        FINAL_MODEL_FILE
    )


elif BEST_MODEL_NAME == "LightGBM":

    FINAL_MODEL_FILE = (
        MODEL_DIR
        / "final_optimized_lightgbm.txt"
    )

    final_model.booster_.save_model(
        str(
            FINAL_MODEL_FILE
        )
    )


else:

    FINAL_MODEL_FILE = (
        MODEL_DIR
        / "final_optimized_catboost.cbm"
    )

    final_model.save_model(
        str(
            FINAL_MODEL_FILE
        )
    )


print(
    "\nFinal model saved:"
)

print(
    FINAL_MODEL_FILE
)


# ============================================================
# STEP 19 - SHAP EXPLAINABILITY
# ============================================================

print("\n" + "=" * 85)
print("STEP 19 - GENERATING SHAP EXPLANATIONS")
print("=" * 85)


shap_sample_size = min(
    SHAP_SAMPLE_SIZE,
    len(X_final_train)
)


X_shap = (
    X_final_train.sample(
        n=shap_sample_size,
        random_state=
            RANDOM_STATE
    )
)


print(
    "\nSHAP sample size:",
    shap_sample_size
)


explainer = shap.TreeExplainer(
    final_model
)


explanation = explainer(
    X_shap
)


shap_values = (
    explanation.values
)


if shap_values.ndim > 2:

    shap_values = np.squeeze(
        shap_values
    )


# ============================================================
# STEP 20 - GLOBAL SHAP IMPORTANCE
# ============================================================

global_shap = pd.DataFrame(
    {

        "feature":
            MODEL_FEATURES,

        "mean_abs_shap":
            np.abs(
                shap_values
            ).mean(
                axis=0
            )
    }
)


global_shap = (
    global_shap
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
)


global_shap.to_csv(
    SHAP_IMPORTANCE_FILE,
    index=False
)


print(
    "\nTop 15 SHAP features:"
)

print(
    global_shap.head(15)
)


# ============================================================
# STEP 21 - SHAP SUMMARY PLOT
# ============================================================

shap.summary_plot(

    shap_values,

    X_shap,

    feature_names=
        MODEL_FEATURES,

    show=False,

    max_display=20
)


plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    / "25_shap_summary_plot.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# STEP 22 - SHAP BAR IMPORTANCE
# ============================================================

top_shap = (
    global_shap
    .head(20)
    .sort_values(
        "mean_abs_shap"
    )
)


plt.figure(
    figsize=(11, 8)
)


plt.barh(

    top_shap[
        "feature"
    ],

    top_shap[
        "mean_abs_shap"
    ]
)


plt.xlabel(
    "Mean |SHAP Value|"
)

plt.ylabel(
    "Feature"
)

plt.title(
    f"Global SHAP Importance - "
    f"{BEST_MODEL_NAME}"
)

plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    / "26_shap_global_importance.png",
    dpi=300
)


plt.close()


# ============================================================
# STEP 23 - LOCAL EXPLANATION
# ============================================================

local_index = 0


local_shap = pd.DataFrame(
    {

        "feature":
            MODEL_FEATURES,

        "feature_value":
            X_shap
            .iloc[
                local_index
            ]
            .values,

        "shap_value":
            shap_values[
                local_index
            ]
    }
)


local_shap[
    "absolute_shap"
] = (
    local_shap[
        "shap_value"
    ]
    .abs()
)


local_shap = (
    local_shap
    .sort_values(
        "absolute_shap",
        ascending=False
    )
)


local_shap.to_csv(
    SHAP_LOCAL_FILE,
    index=False
)


local_plot = (
    local_shap
    .head(15)
    .sort_values(
        "shap_value"
    )
)


plt.figure(
    figsize=(11, 8)
)


plt.barh(

    local_plot[
        "feature"
    ],

    local_plot[
        "shap_value"
    ]
)


plt.xlabel(
    "SHAP Contribution"
)

plt.title(
    "Example Forecast Explanation"
)

plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    / "27_shap_local_explanation.png",
    dpi=300
)


plt.close()


# ============================================================
# STEP 24 - FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 85)
print("STEP 24 - FINAL UNTOUCHED TEST EVALUATION")
print("=" * 85)


print(
    "\nIMPORTANT:"
)

print(
    "Hyperparameters are now locked."
)

print(
    "Test results will NOT be used "
    "to modify the model."
)


test_predictions = (
    recursive_forecast(
        final_model,
        final_training_data,
        test
    )
)


test_actual = (
    test_predictions[
        "actual_sales"
    ]
    .to_numpy()
)


test_predicted = (
    test_predictions[
        "prediction"
    ]
    .to_numpy()
)


test_metrics = {

    "model":
        BEST_MODEL_NAME,

    "MAE":
        mae(
            test_actual,
            test_predicted
        ),

    "RMSE":
        rmse(
            test_actual,
            test_predicted
        ),

    "WAPE_percent":
        wape(
            test_actual,
            test_predicted
        ),

    "RMSLE":
        rmsle(
            test_actual,
            test_predicted
        ),

    "Forecast_Bias":
        forecast_bias(
            test_actual,
            test_predicted
        )
}


test_metrics_df = pd.DataFrame(
    [
        test_metrics
    ]
)


test_metrics_df.to_csv(
    TEST_METRICS_FILE,
    index=False
)


test_predictions.to_parquet(
    TEST_PREDICTIONS_FILE,
    index=False
)


print(
    "\nFINAL TEST RESULTS"
)

print(
    test_metrics_df.round(4)
)


# ============================================================
# STEP 25 - FINAL ACTUAL VS FORECAST CHART
# ============================================================

daily_test = (

    test_predictions
    .groupby(
        "date",
        observed=True
    )
    [
        [
            "actual_sales",
            "prediction"
        ]
    ]
    .sum()
    .reset_index()
)


plt.figure(
    figsize=(14, 7)
)


plt.plot(

    daily_test[
        "date"
    ],

    daily_test[
        "actual_sales"
    ],

    marker="o",

    label="Actual Demand"
)


plt.plot(

    daily_test[
        "date"
    ],

    daily_test[
        "prediction"
    ],

    marker="o",

    label=
        f"{BEST_MODEL_NAME} Forecast"
)


plt.title(
    f"Final Test: Actual vs "
    f"{BEST_MODEL_NAME} Forecast"
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
    / "28_final_test_actual_vs_forecast.png",
    dpi=300
)


plt.close()


# ============================================================
# STEP 26 - SAVE FINAL METADATA
# ============================================================

metadata = {

    "model":
        BEST_MODEL_NAME,

    "best_parameters":
        best_params,

    "optuna_trials":
        N_TRIALS,

    "cv_folds":
        N_CV_FOLDS,

    "forecast_horizon_days":
        FORECAST_HORIZON,

    "stage7_validation_wape":
        STAGE7_BEST_WAPE,

    "stage8_validation_wape":
        TUNED_VALIDATION_WAPE,

    "final_test_metrics":
        test_metrics,

    "training_start":
        str(
            final_training_data[
                "date"
            ].min()
        ),

    "training_end":
        str(
            final_training_data[
                "date"
            ].max()
        ),

    "test_start":
        str(
            test[
                "date"
            ].min()
        ),

    "test_end":
        str(
            test[
                "date"
            ].max()
        ),

    "model_file":
        str(
            FINAL_MODEL_FILE
        )
}


with open(
    METADATA_FILE,
    "w"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 85)
print("STAGE 8 FINAL SUMMARY")
print("=" * 85)


print(
    "\nOptimized model:"
)

print(
    BEST_MODEL_NAME
)


print(
    "\nOptuna trials:"
)

print(
    N_TRIALS
)


print(
    "\nRolling CV folds:"
)

print(
    N_CV_FOLDS
)


print(
    "\nStage 7 validation WAPE:"
)

print(
    f"{STAGE7_BEST_WAPE:.2f}%"
)


print(
    "\nStage 8 tuned validation WAPE:"
)

print(
    f"{TUNED_VALIDATION_WAPE:.2f}%"
)


print(
    "\nFinal test WAPE:"
)

print(
    f"{test_metrics['WAPE_percent']:.2f}%"
)


print(
    "\nFinal test RMSE:"
)

print(
    f"{test_metrics['RMSE']:.4f}"
)


print(
    "\nFinal test MAE:"
)

print(
    f"{test_metrics['MAE']:.4f}"
)


print(
    "\nFinal forecast bias:"
)

print(
    f"{test_metrics['Forecast_Bias']:.4f}"
)


print(
    "\nSHAP explanations generated successfully."
)


print(
    "\nFinal model locked and saved."
)


print("\n" + "=" * 85)
print("STAGE 8 COMPLETED SUCCESSFULLY")
print("=" * 85)