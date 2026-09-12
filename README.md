# Intelligent Supply Chain Demand Forecasting & Inventory Optimization

An end-to-end Data Science and Machine Learning project for forecasting retail demand, estimating stockout risk, calculating safety stock and reorder points, optimizing replenishment quantities, and delivering supply-chain insights through an interactive Streamlit dashboard.

This project uses the **M5 Forecasting dataset** and combines time-series forecasting, machine learning, explainable AI, inventory analytics, and dashboard development into a complete decision-support system.

---

## Project Overview

Retail and supply-chain businesses need to answer several important questions:

* How much demand should we expect in the next few days?
* Which products are likely to run out of stock?
* How much safety stock should we maintain?
* When should inventory be reordered?
* How many units should be ordered?
* Which products require immediate management attention?

This project addresses those problems using a complete Python-based Data Science pipeline.

The system predicts future demand at the **Product × Store × Date** level and converts those forecasts into inventory-management recommendations.

---

# Project Architecture

```text
M5 Retail Dataset
        │
        ▼
Data Validation
        │
        ▼
Data Integration
        │
        ├── Historical Sales
        ├── Calendar
        ├── Product Prices
        ├── Events / Holidays
        └── SNAP Information
        │
        ▼
Data Cleaning
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Engineering
        │
        ├── Lag Features
        ├── Rolling Statistics
        ├── Calendar Features
        ├── Price Features
        ├── Event Features
        ├── Demand Volatility
        └── Product Lifecycle Features
        │
        ▼
Baseline Forecasting
        │
        ├── Last Value
        ├── Seasonal Naive
        ├── 28-Day Mean
        └── Weekday Mean
        │
        ▼
Advanced Machine Learning
        │
        ├── XGBoost
        ├── LightGBM
        └── CatBoost
        │
        ▼
Recursive 28-Day Forecasting
        │
        ▼
Hyperparameter Optimization
        │
        └── Optuna
        │
        ▼
Explainable AI
        │
        └── SHAP
        │
        ▼
Final Demand Forecast
        │
        ▼
Inventory Intelligence
        │
        ├── ABC Classification
        ├── XYZ Classification
        ├── Safety Stock
        ├── Reorder Point
        ├── Stockout Probability
        ├── EOQ
        └── Recommended Order Quantity
        │
        ▼
Streamlit Dashboard
        │
        ▼
Supply Chain Decision Support System
```

---

# Key Features

## Demand Forecasting

The forecasting system predicts future product-level demand using historical sales patterns and contextual features.

Forecast horizons supported:

```text
7-Day Forecast
14-Day Forecast
28-Day Forecast
```

The primary project evaluation focuses on a **28-day forecasting horizon**.

---

## Time-Series Feature Engineering

The project creates several advanced forecasting features.

### Lag Features

```text
lag_1
lag_7
lag_14
lag_28
lag_56
```

These capture recent and seasonal demand behaviour.

### Rolling Demand Features

```text
rolling_mean_7
rolling_mean_14
rolling_mean_28
rolling_mean_56

rolling_std_7
rolling_std_14
rolling_std_28
rolling_std_56

rolling_min_7
rolling_min_28

rolling_max_7
rolling_max_28
```

### Demand Dynamics

```text
demand_trend_7_28
demand_ratio_7_28
demand_cv_28
zero_sales_rate_28
```

### Price Features

```text
sell_price
price_change
price_change_pct
price_vs_historical_mean
```

### Calendar Features

```text
day_of_week
day_of_month
week_of_year
month
quarter
year
is_weekend
is_month_start
is_month_end
```

### Cyclical Encoding

```text
dow_sin
dow_cos

month_sin
month_cos
```

### External Factors

```text
SNAP
Events
Holidays
```

---

# Dataset

This project uses the:

## M5 Forecasting - Accuracy Dataset

Dataset source:

**Kaggle M5 Forecasting – Accuracy**

https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Main files used:

```text
calendar.csv
sell_prices.csv
sales_train_validation.csv
sales_train_evaluation.csv
sample_submission.csv
```

The dataset includes:

* Historical daily sales
* Product information
* Store information
* State information
* Selling prices
* Calendar data
* Events
* Holidays
* SNAP program indicators

---

# Important Dataset Note

The M5 dataset does **not** provide complete warehouse inventory information such as:

* Current stock-on-hand
* Supplier lead time
* Incoming purchase orders
* Ordering cost
* Inventory holding cost

Therefore, the inventory optimization module supports two modes:

### Real Inventory Mode

Provide:

```text
data/external/inventory_snapshot.csv
```

### Simulation Mode

If no real inventory file exists, the system generates a clearly identified simulated inventory scenario.

Simulated inventory values are used only to demonstrate the inventory optimization pipeline and are not treated as original M5 data.

---

# Technology Stack

## Programming

```text
Python
```

## Data Processing

```text
Pandas
NumPy
PyArrow
```

## Visualization

```text
Matplotlib
Plotly
```

## Statistical Analysis

```text
SciPy
Statsmodels
```

## Machine Learning

```text
Scikit-learn
XGBoost
LightGBM
CatBoost
```

## Hyperparameter Optimization

```text
Optuna
```

## Explainable AI

```text
SHAP
```

## Dashboard

```text
Streamlit
```

## Storage

```text
CSV
Parquet
JSON
```

## Development

```text
VS Code
Git
GitHub
```

---

# Project Structure

```text
Supply_Chain_DS_Project/
│
├── app/
│   └── streamlit_dashboard.py
│
├── data/
│   │
│   ├── raw/
│   │   ├── calendar.csv
│   │   ├── sell_prices.csv
│   │   ├── sales_train_validation.csv
│   │   ├── sales_train_evaluation.csv
│   │   └── sample_submission.csv
│   │
│   ├── processed/
│   │   ├── m5_ca1_dev_integrated.parquet
│   │   ├── m5_ca1_dev_clean.parquet
│   │   ├── m5_ca1_features.parquet
│   │   ├── m5_ca1_train.parquet
│   │   ├── m5_ca1_validation.parquet
│   │   ├── m5_ca1_test.parquet
│   │   ├── baseline_validation_predictions.parquet
│   │   ├── advanced_ml_validation_predictions.parquet
│   │   ├── stage8_final_test_predictions.parquet
│   │   └── stage9_inventory_optimization.parquet
│   │
│   └── external/
│       ├── inventory_snapshot_template.csv
│       ├── inventory_snapshot_simulated.csv
│       └── inventory_snapshot.csv
│
├── models/
│   ├── xgboost_demand_model.json
│   ├── lightgbm_demand_model.txt
│   ├── catboost_demand_model.cbm
│   ├── identifier_encodings.json
│   ├── stage8_best_parameters.json
│   ├── stage8_final_model_metadata.json
│   └── final_optimized_model.*
│
├── reports/
│   │
│   ├── figures/
│   │   ├── 01_daily_sales_trend.png
│   │   ├── 02_rolling_28_day_sales.png
│   │   ├── 03_monthly_sales_trend.png
│   │   ├── ...
│   │   ├── 21_baseline_vs_ml_comparison.png
│   │   ├── 25_shap_summary_plot.png
│   │   ├── 28_final_test_actual_vs_forecast.png
│   │   ├── 29_stockout_risk_distribution.png
│   │   └── 30_top_replenishment_items.png
│   │
│   └── tables/
│       ├── baseline_model_metrics.csv
│       ├── advanced_ml_metrics.csv
│       ├── baseline_vs_advanced_models.csv
│       ├── stage8_final_test_metrics.csv
│       ├── stage8_shap_global_importance.csv
│       ├── stage9_inventory_optimization.csv
│       └── stage9_inventory_summary.csv
│
├── src/
│   │
│   ├── data/
│   │   ├── 01_validate_dataset.py
│   │   ├── 02_build_integrated_dataset.py
│   │   └── 03_clean_dataset.py
│   │
│   ├── eda/
│   │   └── 04_eda_analysis.py
│   │
│   ├── features/
│   │   └── 05_feature_engineering.py
│   │
│   ├── models/
│   │   ├── 06_baseline_forecasting.py
│   │   ├── 07_advanced_ml_forecasting.py
│   │   └── 08_model_optimization_explainability.py
│   │
│   └── optimization/
│       └── 09_inventory_optimization.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# Project Execution Pipeline

The project is designed to be executed sequentially.

## Stage 1 — Dataset Validation

Run:

```bash
python src/data/01_validate_dataset.py
```

Purpose:

* Validate M5 files
* Inspect columns
* Check file availability
* Inspect dataset structure

---

## Stage 2 — Data Integration

Run:

```bash
python src/data/02_build_integrated_dataset.py
```

Purpose:

* Convert sales data from wide to long format
* Merge sales with calendar
* Merge selling prices
* Add SNAP information
* Create modeling-ready base dataset

Output:

```text
data/processed/m5_ca1_dev_integrated.parquet
```

---

## Stage 3 — Data Cleaning

Run:

```bash
python src/data/03_clean_dataset.py
```

Tasks:

* Handle missing values
* Remove duplicate records
* Validate sales values
* Remove invalid prices
* Detect pre-launch product periods
* Forward-fill appropriate price gaps
* Preserve meaningful zero-demand observations

Output:

```text
data/processed/m5_ca1_dev_clean.parquet
```

---

## Stage 4 — Exploratory Data Analysis

Run:

```bash
python src/eda/04_eda_analysis.py
```

Analysis includes:

* Daily sales trend
* Rolling demand trend
* Monthly seasonality
* Weekday patterns
* Category performance
* Department performance
* Top products
* Zero-demand behaviour
* SNAP impact
* Event impact
* Price distribution
* Price-demand relationships
* Demand volatility
* Seasonal decomposition

Outputs are saved in:

```text
reports/figures/
reports/tables/
```

---

## Stage 5 — Feature Engineering

Run:

```bash
python src/features/05_feature_engineering.py
```

Creates:

* Lag features
* Rolling averages
* Rolling standard deviations
* Price-change features
* Calendar variables
* Cyclical encoding
* Product age
* Demand trends
* Demand volatility
* Intermittent-demand indicators

The data is split chronologically into:

```text
Training
Validation
Test
```

The project does **not** use random train/test splitting for forecasting.

Outputs:

```text
m5_ca1_train.parquet
m5_ca1_validation.parquet
m5_ca1_test.parquet
```

---

# Stage 6 — Baseline Forecasting

Run:

```bash
python src/models/06_baseline_forecasting.py
```

Baseline models:

```text
Last Value
Seasonal Naive 28
28-Day Mean
Weekday Mean
```

Metrics:

```text
MAE
RMSE
WAPE
RMSLE
Forecast Bias
```

These models provide a benchmark for evaluating advanced machine-learning models.

---

# Stage 7 — Advanced Machine Learning Forecasting

Run:

```bash
python src/models/07_advanced_ml_forecasting.py
```

Models trained:

```text
XGBoost
LightGBM
CatBoost
```

The project uses **recursive multi-step forecasting**.

Instead of using actual sales inside the future validation period, predicted demand is recursively added back into historical demand when generating later lag and rolling features.

This prevents future target leakage.

---

# Stage 8 — Hyperparameter Optimization & Explainable AI

Run:

```bash
python src/models/08_model_optimization_explainability.py
```

Stage 8 performs:

```text
Best Stage 7 Model
        ↓
Optuna Optimization
        ↓
Rolling Time-Series Validation
        ↓
Optimized Model
        ↓
SHAP Explainability
        ↓
Final Model Lock
        ↓
Train + Validation
        ↓
Untouched Test Evaluation
```

The final test dataset is used only after model selection and hyperparameter optimization are complete.

---

# Explainable AI

The project uses **SHAP** to explain model predictions.

Global explainability answers:

> Which features have the largest overall influence on demand?

Potential important variables include:

```text
lag_28
lag_7
rolling_mean_28
rolling_mean_7
sell_price
zero_sales_rate_28
SNAP
events
days_since_launch
```

Local explainability answers:

> Why did the model predict this amount for this specific product?

SHAP results are stored in:

```text
reports/tables/stage8_shap_global_importance.csv
```

and visualizations are stored in:

```text
reports/figures/
```

---

# Stage 9 — Inventory Optimization

Run:

```bash
python src/optimization/09_inventory_optimization.py
```

This stage converts forecasts into actionable supply-chain decisions.

---

## ABC Analysis

Products are ranked according to estimated annual sales value.

```text
A = High-value products
B = Medium-value products
C = Lower-value products
```

Different service-level targets can be assigned:

```text
A → 98%
B → 95%
C → 90%
```

---

## XYZ Analysis

Products are also classified according to demand variability.

```text
X = Stable demand
Y = Moderately variable demand
Z = Highly variable demand
```

Classification is based on the coefficient of variation:

```text
CV = Demand Standard Deviation / Average Demand
```

The resulting combinations include:

```text
AX
AY
AZ
BX
BY
BZ
CX
CY
CZ
```

For example:

```text
AZ
```

means:

```text
High business value
+
Highly uncertain demand
```

and may require closer inventory monitoring.

---

# Safety Stock

Safety stock protects the business against demand uncertainty during supplier lead time.

Conceptually:

```text
Safety Stock
=
Service Level Factor
×
Demand Uncertainty
×
sqrt(Lead Time)
```

---

# Reorder Point

The system calculates:

```text
Reorder Point
=
Forecast Demand During Lead Time
+
Safety Stock
```

When inventory position drops below the reorder point, the system recommends replenishment.

---

# Stockout Risk

The system estimates the probability that lead-time demand will exceed available inventory.

Outputs:

```text
LOW
MEDIUM
HIGH
```

Example:

```text
Product                 FOODS_1_001
Stockout Probability    82.4%
Risk                     HIGH
Recommended Action       URGENT REPLENISHMENT
```

---

# Economic Order Quantity

EOQ is used to balance ordering cost and holding cost.

```text
EOQ = sqrt((2 × Annual Demand × Ordering Cost) / Holding Cost)
```

The final recommended quantity considers:

* EOQ
* Current inventory
* Incoming stock
* Reorder point
* Forecast demand
* Review-period demand

---

# Stage 10 — Streamlit Dashboard

Run:

```bash
python -m streamlit run app/streamlit_dashboard.py
```

The dashboard provides seven interactive sections.

### Executive Overview

Displays:

* Number of products
* 28-day demand forecast
* High-risk products
* Products requiring reorder
* Recommended replenishment quantity
* Average stockout probability

### Demand Forecast

Displays:

* Actual vs predicted demand
* MAE
* RMSE
* WAPE
* RMSLE
* Forecast Bias
* Daily forecast errors

### Stockout Risk

Displays:

* High-risk products
* Stockout probabilities
* Lead-time demand
* Safety stock
* Reorder recommendations

### Inventory Optimization

Displays:

* Current stock
* Incoming inventory
* Inventory position
* Safety stock
* Reorder points
* EOQ
* Recommended order quantities
* Inventory actions

### Product Intelligence

Allows users to select a product and view:

```text
28-Day Forecast
Current Inventory
Safety Stock
Reorder Point
Recommended Order
Stockout Probability
Risk Level
ABC / XYZ Class
Lead Time
Days of Cover
Actual vs Forecast
```

### ABC / XYZ Analysis

Displays a matrix of:

```text
        X     Y     Z

A       AX    AY    AZ
B       BX    BY    BZ
C       CX    CY    CZ
```

### Data Explorer

Provides:

* Product search
* Risk filtering
* Category filtering
* Inventory table
* CSV export

---

# Installation

## Clone Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project:

```bash
cd Supply_Chain_DS_Project
```

---

## Create Virtual Environment

Windows:

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

For PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Complete Execution Order

Run the project in this order:

```bash
python src/data/01_validate_dataset.py
```

```bash
python src/data/02_build_integrated_dataset.py
```

```bash
python src/data/03_clean_dataset.py
```

```bash
python src/eda/04_eda_analysis.py
```

```bash
python src/features/05_feature_engineering.py
```

```bash
python src/models/06_baseline_forecasting.py
```

```bash
python src/models/07_advanced_ml_forecasting.py
```

```bash
python src/models/08_model_optimization_explainability.py
```

```bash
python src/optimization/09_inventory_optimization.py
```

Finally:

```bash
python -m streamlit run app/streamlit_dashboard.py
```

---

# Model Evaluation

The forecasting models are evaluated using:

## MAE

Measures average absolute forecast error.

Lower is better.

## RMSE

Penalizes large forecasting errors more strongly.

Lower is better.

## WAPE

Weighted Absolute Percentage Error is one of the primary project metrics.

```text
WAPE
=
Total Absolute Forecast Error
/
Total Actual Demand
× 100
```

Lower is better.

## RMSLE

Useful when products operate at significantly different demand levels.

## Forecast Bias

Measures systematic overforecasting or underforecasting.

```text
Positive Bias
→ Overforecasting

Negative Bias
→ Underforecasting
```

---

# Final Model Results

Update this section with the actual Stage 8 results generated on your machine.

```text
Winning Model:                <XGBoost / LightGBM / CatBoost>

Validation MAE:               <value>

Validation RMSE:              <value>

Validation WAPE:              <value> %

Final Test MAE:               <value>

Final Test RMSE:              <value>

Final Test WAPE:              <value> %

Final RMSLE:                  <value>

Forecast Bias:                <value>
```

Results can be found in:

```text
reports/tables/stage8_final_test_metrics.csv
```

---

# Business Outputs

The final system provides:

```text
Expected Demand

7-Day Forecast

14-Day Forecast

28-Day Forecast

Stockout Probability

Stockout Risk Level

Safety Stock

Reorder Point

Economic Order Quantity

Recommended Order Quantity

ABC Classification

XYZ Classification

Management Priority Score

Inventory Action Recommendation
```

---

# Example Decision Output

```text
Product                    FOODS_1_001

28-Day Forecast             165 units

Average Daily Demand        5.9 units

Current Inventory           32 units

Incoming Inventory          10 units

Inventory Position          42 units

Lead Time                   7 days

Safety Stock                18 units

Reorder Point               59 units

Stockout Probability        81%

Risk Level                  HIGH

ABC / XYZ Class             AZ

EOQ                         95 units

Recommended Order           110 units

Recommended Action          URGENT REPLENISHMENT
```

---

# Data Leakage Prevention

Special care is taken to prevent time-series target leakage.

Examples:

```text
Rolling features use only previous observations.

Validation dates always occur after training dates.

Future actual demand is not used when recursively forecasting.

Predicted demand is used to construct later future lag values.

The final test set remains untouched during model development.
```

This makes the evaluation more representative of a real production forecasting environment.

---

# Key Data Science Concepts Demonstrated

This project demonstrates:

```text
Data Cleaning

Data Integration

Exploratory Data Analysis

Feature Engineering

Time-Series Forecasting

Machine Learning

Gradient Boosting

Recursive Forecasting

Time-Based Validation

Hyperparameter Optimization

Explainable AI

Inventory Analytics

Stockout Risk Modeling

ABC Analysis

XYZ Analysis

Safety Stock Optimization

EOQ

Reorder Point Calculation

Dashboard Development

Business Decision Intelligence
```

---

# Why This Project Is Resume-Relevant

Unlike a simple sales prediction project, this project connects Machine Learning predictions directly with supply-chain business decisions.

It demonstrates the complete workflow:

```text
Raw Data
   ↓
Data Engineering
   ↓
Statistical Analysis
   ↓
Machine Learning
   ↓
Forecasting
   ↓
Model Optimization
   ↓
Explainable AI
   ↓
Inventory Optimization
   ↓
Business Recommendation
   ↓
Interactive Deployment
```

This makes the project suitable for roles such as:

```text
Data Science Intern

Junior Data Scientist

Machine Learning Intern

Data Analyst

Supply Chain Data Analyst

Business Intelligence Analyst

ML Engineer Intern
```

---

# Future Improvements

Possible future extensions include:

### Full M5 Scale

The development version may use a subset of products from one store.

The pipeline can later be expanded to:

```text
All products
All stores
All states
```

### Hierarchical Forecasting

Forecast and reconcile demand across:

```text
State
Store
Category
Department
Product
```

### Deep Learning

Possible models:

```text
LSTM
GRU
Temporal Fusion Transformer
N-BEATS
DeepAR
```

### Probabilistic Forecasting

Instead of point forecasts, estimate:

```text
P10
P50
P90
```

demand intervals.

### Dynamic Safety Stock

Use forecast-specific uncertainty rather than only historical demand volatility.

### ERP Integration

Connect directly with:

```text
SAP
Oracle
ERPNext
Odoo
Warehouse Management Systems
```

### Real Inventory Data

Replace simulated inventory information with actual:

```text
Stock-on-hand

Purchase orders

Supplier lead times

Order costs

Holding costs
```

### MLOps

Add:

```text
MLflow

Docker 

FastAPI

Model Monitoring

Data Drift Detection

Scheduled Retraining
```

### Cloud Deployment

Deploy using:

```text
AWS

Azure

Google Cloud

Streamlit Community Cloud
```

---

# GitHub Notes

Large raw M5 files should normally not be uploaded directly to GitHub.

Recommended `.gitignore` entries:

```gitignore
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/

data/raw/
data/processed/

*.parquet

.env
```

Users can download the dataset separately from Kaggle and place it inside:

```text
data/raw/
```

---

# Responsible Interpretation

This project is intended for educational and portfolio purposes.

Inventory recommendations based on simulated inventory inputs should not be treated as actual business recommendations.

For production use, the system should be connected to verified:

* Inventory balances
* Purchase orders
* Supplier lead times
* Procurement costs
* Service-level policies
* Warehouse constraints

---

# Project Summary

This project implements an advanced end-to-end supply-chain Data Science system combining:

```text
Retail Demand Forecasting
+
Machine Learning
+
Time-Series Feature Engineering
+
Recursive Forecasting
+
Optuna Optimization
+
SHAP Explainability
+
Stockout Risk Estimation
+
Safety Stock
+
Reorder Point
+
EOQ
+
ABC / XYZ Analysis
+
Inventory Optimization
+
Streamlit Dashboard
```

The final objective is not simply to predict sales, but to convert those predictions into actionable inventory and replenishment decisions.

---

# Author

**Raj Shekhar**

Data Science / Machine Learning Project

---

# Acknowledgements

* Kaggle
* M5 Forecasting Competition
* Walmart retail dataset contributors
* Python open-source Data Science community

---

# License

This project is intended for educational, academic, and portfolio use.

Add your preferred open-source license before public distribution, for example:

```text
MIT License
```

---

⭐ If you found this project useful, consider starring the repository.
