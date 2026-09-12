import pandas as pd

df = pd.read_parquet(
    "data/processed/m5_ca1_dev_integrated.parquet"
)

print(df.shape)

print(df.head())

print(df.columns.tolist())