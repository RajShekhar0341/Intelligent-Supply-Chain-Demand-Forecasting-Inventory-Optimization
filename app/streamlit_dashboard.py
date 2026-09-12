from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Supply Chain Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "tables"
)


INVENTORY_FILE = (
    PROCESSED_DIR
    / "stage9_inventory_optimization.parquet"
)

FORECAST_FILE = (
    PROCESSED_DIR
    / "stage8_final_test_predictions.parquet"
)

TEST_METRICS_FILE = (
    TABLE_DIR
    / "stage8_final_test_metrics.csv"
)

INVENTORY_SUMMARY_FILE = (
    TABLE_DIR
    / "stage9_inventory_summary.csv"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .sub-title {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .risk-high {
        padding: 10px;
        border-radius: 8px;
        font-weight: bold;
        background-color: rgba(255,0,0,0.08);
    }

    .risk-medium {
        padding: 10px;
        border-radius: 8px;
        font-weight: bold;
        background-color: rgba(255,165,0,0.10);
    }

    .risk-low {
        padding: 10px;
        border-radius: 8px;
        font-weight: bold;
        background-color: rgba(0,128,0,0.08);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data
def load_inventory_data():

    if not INVENTORY_FILE.exists():

        return None

    data = pd.read_parquet(
        INVENTORY_FILE
    )

    return data


@st.cache_data
def load_forecast_data():

    if not FORECAST_FILE.exists():

        return None

    data = pd.read_parquet(
        FORECAST_FILE
    )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    return data


@st.cache_data
def load_test_metrics():

    if not TEST_METRICS_FILE.exists():

        return None

    return pd.read_csv(
        TEST_METRICS_FILE
    )


@st.cache_data
def load_inventory_summary():

    if not INVENTORY_SUMMARY_FILE.exists():

        return None

    return pd.read_csv(
        INVENTORY_SUMMARY_FILE
    )


# ============================================================
# LOAD DATA
# ============================================================

inventory = load_inventory_data()

forecast = load_forecast_data()

test_metrics = load_test_metrics()

inventory_summary = load_inventory_summary()


# ============================================================
# VALIDATION
# ============================================================

if inventory is None:

    st.error(
        "Stage 9 inventory file was not found. "
        "Run Stage 9 first."
    )

    st.stop()


if forecast is None:

    st.error(
        "Stage 8 forecast file was not found. "
        "Run Stage 8 first."
    )

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    '📦 Intelligent Supply Chain '
    'Forecasting & Inventory Optimization'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="sub-title">'
    'Machine Learning • Demand Forecasting • '
    'Stockout Risk • Safety Stock • '
    'Reorder Optimization • Explainable AI'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Dashboard Filters"
)


stores = sorted(
    inventory[
        "store_id"
    ]
    .dropna()
    .astype(str)
    .unique()
)


selected_store = st.sidebar.selectbox(
    "Store",
    stores
)


store_inventory = inventory[
    inventory[
        "store_id"
    ].astype(str)
    ==
    selected_store
].copy()


categories = sorted(
    store_inventory[
        "cat_id"
    ]
    .dropna()
    .astype(str)
    .unique()
)


selected_categories = (
    st.sidebar.multiselect(
        "Product Category",
        categories,
        default=categories
    )
)


if selected_categories:

    filtered_inventory = (
        store_inventory[
            store_inventory[
                "cat_id"
            ]
            .astype(str)
            .isin(
                selected_categories
            )
        ]
        .copy()
    )

else:

    filtered_inventory = (
        store_inventory.copy()
    )


risk_options = [
    "LOW",
    "MEDIUM",
    "HIGH"
]


selected_risks = (
    st.sidebar.multiselect(
        "Stockout Risk",
        risk_options,
        default=risk_options
    )
)


if selected_risks:

    filtered_inventory = (
        filtered_inventory[
            filtered_inventory[
                "stockout_risk"
            ]
            .isin(
                selected_risks
            )
        ]
    )


st.sidebar.markdown("---")


st.sidebar.caption(
    f"Products displayed: "
    f"{len(filtered_inventory):,}"
)


if (
    "inventory_source"
    in inventory.columns
):

    source_value = str(
        inventory[
            "inventory_source"
        ]
        .iloc[0]
    )

    st.sidebar.info(
        f"Inventory source: "
        f"{source_value}"
    )


# ============================================================
# DASHBOARD TABS
# ============================================================

tabs = st.tabs(
    [
        "🏠 Executive Overview",
        "📈 Demand Forecast",
        "⚠️ Stockout Risk",
        "📦 Inventory Optimization",
        "🔍 Product Intelligence",
        "🧮 ABC / XYZ Analysis",
        "📋 Data Explorer"
    ]
)


# ============================================================
# TAB 1 - EXECUTIVE OVERVIEW
# ============================================================

with tabs[0]:

    st.header(
        "Executive Supply Chain Overview"
    )


    total_products = (
        filtered_inventory[
            "item_id"
        ]
        .nunique()
    )


    total_forecast = (
        filtered_inventory[
            "forecast_total_28"
        ]
        .sum()
    )


    reorder_products = int(
        filtered_inventory[
            "reorder_required"
        ]
        .sum()
    )


    high_risk_products = int(

        filtered_inventory[
            "stockout_risk"
        ]

        .eq(
            "HIGH"
        )

        .sum()
    )


    total_order_quantity = (

        filtered_inventory[
            "recommended_order_qty"
        ]
        .sum()
    )


    avg_stockout_probability = (

        filtered_inventory[
            "stockout_probability_pct"
        ]
        .mean()
    )


    col1, col2, col3 = st.columns(3)

    col4, col5, col6 = st.columns(3)


    col1.metric(
        "Products",
        f"{total_products:,}"
    )


    col2.metric(
        "28-Day Forecast",
        f"{total_forecast:,.0f} units"
    )


    col3.metric(
        "High-Risk Products",
        f"{high_risk_products:,}"
    )


    col4.metric(
        "Products to Reorder",
        f"{reorder_products:,}"
    )


    col5.metric(
        "Recommended Order",
        f"{total_order_quantity:,.0f} units"
    )


    col6.metric(
        "Avg Stockout Probability",
        f"{avg_stockout_probability:.1f}%"
    )


    st.markdown("---")


    left, right = st.columns(2)


    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    with left:

        st.subheader(
            "Stockout Risk Distribution"
        )


        risk_counts = (

            filtered_inventory[
                "stockout_risk"
            ]

            .value_counts()

            .rename_axis(
                "Risk"
            )

            .reset_index(
                name="Products"
            )
        )


        fig = px.bar(

            risk_counts,

            x="Risk",

            y="Products",

            text_auto=True,

            title=(
                "Products by Stockout Risk"
            )
        )


        fig.update_layout(
            xaxis_title="Risk Level",
            yaxis_title="Products"
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


    # --------------------------------------------------------
    # ABC CLASS
    # --------------------------------------------------------

    with right:

        st.subheader(
            "Inventory Value Classification"
        )


        abc_counts = (

            filtered_inventory[
                "abc_class"
            ]

            .value_counts()

            .rename_axis(
                "ABC Class"
            )

            .reset_index(
                name="Products"
            )
        )


        fig = px.pie(

            abc_counts,

            names="ABC Class",

            values="Products",

            hole=0.45,

            title="ABC Inventory Mix"
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


    # --------------------------------------------------------
    # PRIORITY PRODUCTS
    # --------------------------------------------------------

    st.subheader(
        "Top Management Priorities"
    )


    priority_columns = [
        "item_id",
        "cat_id",
        "abc_xyz_class",
        "forecast_total_28",
        "inventory_position",
        "reorder_point",
        "stockout_probability_pct",
        "stockout_risk",
        "recommended_order_qty",
        "recommended_action",
        "priority_score"
    ]


    available_priority_columns = [

        column

        for column
        in priority_columns

        if column
        in filtered_inventory.columns
    ]


    priority_table = (

        filtered_inventory[
            available_priority_columns
        ]

        .sort_values(
            "priority_score",
            ascending=False
        )

        .head(20)
    )


    st.dataframe(
        priority_table,
        width="stretch",
        hide_index=True
    )


# ============================================================
# TAB 2 - DEMAND FORECAST
# ============================================================

with tabs[1]:

    st.header(
        "Demand Forecast Performance"
    )


    selected_store_forecast = forecast[
        forecast[
            "store_id"
        ].astype(str)
        ==
        selected_store
    ].copy()


    daily_forecast = (

        selected_store_forecast

        .groupby(
            "date"
        )

        .agg(

            actual_sales=(
                "actual_sales",
                "sum"
            ),

            predicted_sales=(
                "prediction",
                "sum"
            )
        )

        .reset_index()
    )


    fig = go.Figure()


    fig.add_trace(

        go.Scatter(

            x=daily_forecast[
                "date"
            ],

            y=daily_forecast[
                "actual_sales"
            ],

            mode="lines+markers",

            name="Actual Demand"
        )
    )


    fig.add_trace(

        go.Scatter(

            x=daily_forecast[
                "date"
            ],

            y=daily_forecast[
                "predicted_sales"
            ],

            mode="lines+markers",

            name="Forecast Demand"
        )
    )


    fig.update_layout(

        title=(
            "Actual vs Forecast Demand"
        ),

        xaxis_title="Date",

        yaxis_title="Units Sold",

        hovermode="x unified"
    )


    st.plotly_chart(
        fig,
        width="stretch"
    )


    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    if (
        test_metrics
        is not None
        and
        len(test_metrics) > 0
    ):

        st.subheader(
            "Final Forecast Model Performance"
        )


        row = test_metrics.iloc[0]


        c1, c2, c3, c4, c5 = (
            st.columns(5)
        )


        if "MAE" in row:

            c1.metric(
                "MAE",
                f"{row['MAE']:.3f}"
            )


        if "RMSE" in row:

            c2.metric(
                "RMSE",
                f"{row['RMSE']:.3f}"
            )


        if "WAPE_percent" in row:

            c3.metric(
                "WAPE",
                f"{row['WAPE_percent']:.2f}%"
            )


        if "RMSLE" in row:

            c4.metric(
                "RMSLE",
                f"{row['RMSLE']:.3f}"
            )


        if "Forecast_Bias" in row:

            c5.metric(
                "Forecast Bias",
                f"{row['Forecast_Bias']:.3f}"
            )


    # --------------------------------------------------------
    # DAILY FORECAST ERROR
    # --------------------------------------------------------

    daily_forecast[
        "error"
    ] = (

        daily_forecast[
            "actual_sales"
        ]

        -

        daily_forecast[
            "predicted_sales"
        ]
    )


    error_fig = px.bar(

        daily_forecast,

        x="date",

        y="error",

        title="Daily Forecast Error"
    )


    error_fig.update_layout(

        xaxis_title="Date",

        yaxis_title=(
            "Actual - Forecast"
        )
    )


    st.plotly_chart(
        error_fig,
        width="stretch"
    )


# ============================================================
# TAB 3 - STOCKOUT RISK
# ============================================================

with tabs[2]:

    st.header(
        "Stockout Risk Intelligence"
    )


    high_risk = filtered_inventory[
        filtered_inventory[
            "stockout_risk"
        ]
        ==
        "HIGH"
    ].copy()


    high_risk = high_risk.sort_values(
        "stockout_probability_pct",
        ascending=False
    )


    st.subheader(
        "High-Risk Products"
    )


    if len(high_risk) == 0:

        st.success(
            "No high-risk products "
            "under the selected filters."
        )

    else:

        columns = [
            "item_id",
            "cat_id",
            "abc_xyz_class",
            "lead_time_days",
            "forecast_avg_daily",
            "inventory_position",
            "lead_time_demand_forecast",
            "safety_stock",
            "reorder_point",
            "stockout_probability_pct",
            "recommended_order_qty",
            "recommended_action"
        ]


        available = [

            column

            for column
            in columns

            if column
            in high_risk.columns
        ]


        st.dataframe(

            high_risk[
                available
            ],

            width="stretch",

            hide_index=True
        )


    st.subheader(
        "Stockout Probability by Product"
    )


    risk_chart_data = (

        filtered_inventory

        .nlargest(
            25,
            "stockout_probability_pct"
        )

        .sort_values(
            "stockout_probability_pct"
        )
    )


    fig = px.bar(

        risk_chart_data,

        x="stockout_probability_pct",

        y="item_id",

        orientation="h",

        title=(
            "Top 25 Products by "
            "Stockout Probability"
        ),

        hover_data=[
            "abc_xyz_class",
            "inventory_position",
            "reorder_point"
        ]
    )


    fig.update_layout(

        xaxis_title=(
            "Stockout Probability (%)"
        ),

        yaxis_title="Product"
    )


    st.plotly_chart(
        fig,
        width="stretch"
    )


# ============================================================
# TAB 4 - INVENTORY OPTIMIZATION
# ============================================================

with tabs[3]:

    st.header(
        "Inventory Optimization"
    )


    reorder_df = filtered_inventory[
        filtered_inventory[
            "recommended_order_qty"
        ]
        >
        0
    ].copy()


    total_reorder_units = (

        reorder_df[
            "recommended_order_qty"
        ]
        .sum()
    )


    average_safety_stock = (

        filtered_inventory[
            "safety_stock"
        ]
        .mean()
    )


    average_reorder_point = (

        filtered_inventory[
            "reorder_point"
        ]
        .mean()
    )


    average_eoq = (

        filtered_inventory[
            "eoq"
        ]
        .mean()
    )


    c1, c2, c3, c4 = (
        st.columns(4)
    )


    c1.metric(
        "Products Requiring Reorder",
        f"{len(reorder_df):,}"
    )


    c2.metric(
        "Total Replenishment",
        f"{total_reorder_units:,.0f}"
    )


    c3.metric(
        "Average Safety Stock",
        f"{average_safety_stock:,.1f}"
    )


    c4.metric(
        "Average EOQ",
        f"{average_eoq:,.1f}"
    )


    st.subheader(
        "Top Replenishment Recommendations"
    )


    top_orders = (

        reorder_df

        .nlargest(
            25,
            "recommended_order_qty"
        )

        .sort_values(
            "recommended_order_qty"
        )
    )


    if not top_orders.empty:

        fig = px.bar(

            top_orders,

            x="recommended_order_qty",

            y="item_id",

            orientation="h",

            title=(
                "Recommended Order Quantity"
            ),

            hover_data=[
                "current_inventory",
                "incoming_inventory",
                "reorder_point",
                "eoq",
                "stockout_risk"
            ]
        )


        fig.update_layout(

            xaxis_title=(
                "Recommended Order Quantity"
            ),

            yaxis_title="Product"
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


    st.subheader(
        "Inventory Decision Table"
    )


    decision_columns = [
        "item_id",
        "abc_xyz_class",
        "current_inventory",
        "incoming_inventory",
        "inventory_position",
        "lead_time_days",
        "safety_stock",
        "reorder_point",
        "eoq",
        "days_of_cover",
        "stockout_risk",
        "recommended_order_qty",
        "recommended_action"
    ]


    available_columns = [

        column

        for column
        in decision_columns

        if column
        in filtered_inventory.columns
    ]


    decision_table = (

        filtered_inventory[
            available_columns
        ]

        .sort_values(
            "recommended_order_qty",
            ascending=False
        )
    )


    st.dataframe(
        decision_table,
        width="stretch",
        hide_index=True
    )


# ============================================================
# TAB 5 - PRODUCT INTELLIGENCE
# ============================================================

with tabs[4]:

    st.header(
        "Product-Level Intelligence"
    )


    products = sorted(
        filtered_inventory[
            "item_id"
        ]
        .astype(str)
        .unique()
    )


    if products:

        selected_product = (
            st.selectbox(
                "Select Product",
                products
            )
        )


        product = (

            filtered_inventory[
                filtered_inventory[
                    "item_id"
                ].astype(str)
                ==
                selected_product
            ]

            .iloc[0]
        )


        st.subheader(
            f"Product: {selected_product}"
        )


        row1 = st.columns(5)


        row1[0].metric(
            "28-Day Forecast",
            f"{product['forecast_total_28']:.1f}"
        )


        row1[1].metric(
            "Current Inventory",
            f"{product['current_inventory']:.0f}"
        )


        row1[2].metric(
            "Safety Stock",
            f"{product['safety_stock']:.0f}"
        )


        row1[3].metric(
            "Reorder Point",
            f"{product['reorder_point']:.0f}"
        )


        row1[4].metric(
            "Order Quantity",
            f"{product['recommended_order_qty']:.0f}"
        )


        row2 = st.columns(5)


        row2[0].metric(
            "Stockout Probability",
            f"{product['stockout_probability_pct']:.1f}%"
        )


        row2[1].metric(
            "Risk Level",
            str(
                product[
                    "stockout_risk"
                ]
            )
        )


        row2[2].metric(
            "ABC / XYZ",
            str(
                product[
                    "abc_xyz_class"
                ]
            )
        )


        row2[3].metric(
            "Lead Time",
            f"{product['lead_time_days']:.0f} days"
        )


        row2[4].metric(
            "Days of Cover",
            f"{product['days_of_cover']:.1f}"
        )


        # ----------------------------------------------------
        # PRODUCT FORECAST
        # ----------------------------------------------------

        product_forecast = forecast[
            (
                forecast[
                    "store_id"
                ].astype(str)
                ==
                selected_store
            )
            &
            (
                forecast[
                    "item_id"
                ].astype(str)
                ==
                selected_product
            )
        ].copy()


        if not product_forecast.empty:

            st.subheader(
                "Product Demand Forecast"
            )


            fig = go.Figure()


            fig.add_trace(

                go.Scatter(

                    x=product_forecast[
                        "date"
                    ],

                    y=product_forecast[
                        "actual_sales"
                    ],

                    name="Actual",

                    mode="lines+markers"
                )
            )


            fig.add_trace(

                go.Scatter(

                    x=product_forecast[
                        "date"
                    ],

                    y=product_forecast[
                        "prediction"
                    ],

                    name="Forecast",

                    mode="lines+markers"
                )
            )


            fig.update_layout(

                xaxis_title="Date",

                yaxis_title="Demand",

                title=(
                    f"Actual vs Forecast: "
                    f"{selected_product}"
                ),

                hovermode="x unified"
            )


            st.plotly_chart(
                fig,
                width="stretch"
            )


        # ----------------------------------------------------
        # INVENTORY LEVEL GRAPH
        # ----------------------------------------------------

        st.subheader(
            "Inventory Decision Levels"
        )


        inventory_levels = pd.DataFrame(
            {
                "Metric": [
                    "Inventory Position",
                    "Lead-Time Demand",
                    "Safety Stock",
                    "Reorder Point",
                    "EOQ",
                    "Order-Up-To Level"
                ],

                "Units": [
                    product[
                        "inventory_position"
                    ],

                    product[
                        "lead_time_demand_forecast"
                    ],

                    product[
                        "safety_stock"
                    ],

                    product[
                        "reorder_point"
                    ],

                    product[
                        "eoq"
                    ],

                    product[
                        "order_up_to_level"
                    ]
                ]
            }
        )


        fig = px.bar(

            inventory_levels,

            x="Metric",

            y="Units",

            text_auto=".1f",

            title=(
                "Inventory Optimization Levels"
            )
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


        st.subheader(
            "Recommended Action"
        )


        action = str(
            product[
                "recommended_action"
            ]
        )


        risk = str(
            product[
                "stockout_risk"
            ]
        )


        if risk == "HIGH":

            st.error(
                f"{action}: Order approximately "
                f"{product['recommended_order_qty']:.0f} "
                f"units."
            )


        elif risk == "MEDIUM":

            st.warning(
                f"{action}: Monitor inventory and "
                f"consider ordering "
                f"{product['recommended_order_qty']:.0f} "
                f"units."
            )


        else:

            st.success(
                f"{action}. Current inventory position "
                f"is {product['inventory_position']:.0f} "
                f"units."
            )


# ============================================================
# TAB 6 - ABC / XYZ ANALYSIS
# ============================================================

with tabs[5]:

    st.header(
        "ABC / XYZ Inventory Segmentation"
    )


    matrix = pd.crosstab(

        filtered_inventory[
            "abc_class"
        ],

        filtered_inventory[
            "xyz_class"
        ]
    )


    matrix = matrix.reindex(
        index=[
            "A",
            "B",
            "C"
        ],
        columns=[
            "X",
            "Y",
            "Z"
        ],
        fill_value=0
    )


    fig = px.imshow(

        matrix,

        text_auto=True,

        aspect="auto",

        labels={
            "x":
                "Demand Variability (XYZ)",

            "y":
                "Business Value (ABC)",

            "color":
                "Products"
        },

        title=(
            "ABC / XYZ Product Matrix"
        )
    )


    st.plotly_chart(
        fig,
        width="stretch"
    )


    st.subheader(
        "ABC / XYZ Segment Details"
    )


    segment_summary = (

        filtered_inventory

        .groupby(
            "abc_xyz_class",
            observed=True
        )

        .agg(

            Products=(
                "item_id",
                "count"
            ),

            Forecast_Demand=(
                "forecast_total_28",
                "sum"
            ),

            Avg_Stockout_Probability=(
                "stockout_probability_pct",
                "mean"
            ),

            Recommended_Order_Qty=(
                "recommended_order_qty",
                "sum"
            )
        )

        .reset_index()

        .sort_values(
            "Avg_Stockout_Probability",
            ascending=False
        )
    )


    st.dataframe(
        segment_summary,
        width="stretch",
        hide_index=True
    )


    st.info(
        """
        AX products are high-value with stable demand.
        AZ products are high-value but highly uncertain and
        typically deserve the closest monitoring.
        """
    )


# ============================================================
# TAB 7 - DATA EXPLORER
# ============================================================

with tabs[6]:

    st.header(
        "Inventory Decision Data Explorer"
    )


    search_text = st.text_input(
        "Search Product ID"
    )


    explorer_data = (
        filtered_inventory.copy()
    )


    if search_text:

        explorer_data = explorer_data[
            explorer_data[
                "item_id"
            ]
            .astype(str)
            .str.contains(
                search_text,
                case=False,
                na=False
            )
        ]


    st.dataframe(
        explorer_data,
        width="stretch",
        hide_index=True
    )


    csv_data = (
        explorer_data
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )


    st.download_button(
        label=(
            "Download Inventory "
            "Recommendations CSV"
        ),

        data=csv_data,

        file_name=(
            "inventory_recommendations.csv"
        ),

        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")


st.caption(
    "AI-Powered Supply Chain Demand Forecasting "
    "& Inventory Optimization System | "
    "Python • Machine Learning • XGBoost/LightGBM/"
    "CatBoost • Optuna • SHAP • Streamlit"
)