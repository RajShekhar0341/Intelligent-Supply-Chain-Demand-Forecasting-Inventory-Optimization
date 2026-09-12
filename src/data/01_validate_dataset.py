from pathlib import Path
import pandas as pd


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"


print("=" * 70)
print("M5 SUPPLY CHAIN PROJECT - DATASET VALIDATION")
print("=" * 70)

print(f"\nProject root : {PROJECT_ROOT}")
print(f"Dataset path: {DATA_DIR}")


# --------------------------------------------------
# EXPECTED FILES
# --------------------------------------------------

expected_files = [
    "calendar.csv",
    "sell_prices.csv",
    "sales_train_validation.csv",
    "sales_train_evaluation.csv",
    "sample_submission.csv",
]


print("\n" + "=" * 70)
print("1. CHECKING DATASET FILES")
print("=" * 70)


available_files = []

for filename in expected_files:

    file_path = DATA_DIR / filename

    if file_path.exists():

        size_mb = file_path.stat().st_size / (1024 * 1024)

        print(f"[FOUND]   {filename:<35} {size_mb:>10.2f} MB")

        available_files.append(filename)

    else:

        print(f"[MISSING] {filename}")


# --------------------------------------------------
# INSPECT CSV FILES
# --------------------------------------------------

print("\n" + "=" * 70)
print("2. INSPECTING DATASET STRUCTURE")
print("=" * 70)


for filename in available_files:

    file_path = DATA_DIR / filename

    print("\n" + "-" * 70)
    print(f"FILE: {filename}")
    print("-" * 70)

    try:

        # Only load a few rows at this stage.
        # This prevents unnecessary memory usage.
        df_sample = pd.read_csv(file_path, nrows=5)

        print("\nColumns:")

        for column in df_sample.columns:
            print(f"  - {column}")

        print("\nSample shape:")

        print(df_sample.shape)

        print("\nFirst 5 rows:")

        print(df_sample.head())

    except Exception as error:

        print(f"Error reading {filename}: {error}")


print("\n" + "=" * 70)
print("DATASET VALIDATION COMPLETED")
print("=" * 70)