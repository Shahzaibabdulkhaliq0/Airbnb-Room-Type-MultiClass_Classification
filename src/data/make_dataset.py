from pathlib import Path
import os
import boto3
import pandas as pd
import yaml

# ===== LOGGER START =====
# Logger: the project logger, see src/logger.py
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from src.logger import get_logger
logger = get_logger(__name__)
# ===== LOGGER END =====

# Define Project Root Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Read the settings from params.yaml
with open(PROJECT_ROOT / "params.yaml") as file:
    params = yaml.safe_load(file)["make_dataset"]

# Setup Path For S3 Bucket Before Fetch Dataset from S3 Bucket
raw_path = PROJECT_ROOT / "data" / "raw" / "AB_NYC_2019.csv"
raw_path.parent.mkdir(parents=True, exist_ok=True)

# S3 Bucket Setup
S3 = boto3.client("s3")
S3.download_file(params["S3_BUCKET"], params["S3_KEY"], str(raw_path))
print("Downloaded from S3 to:", raw_path)
logger.info("Downloaded raw data from S3 to %s", raw_path)

# Load Dataset
df = pd.read_csv(raw_path, low_memory=False)

# Drop Columns
df = df.drop(columns=params["DROP_COLUMNS"])

# Remove Duplicates
df = df.drop_duplicates()

# 'reviews per month' Columns Missing Values Convert into 0
df["reviews_per_month"] = df["reviews_per_month"].fillna(params["FILL_REVIEWS_PER_MONTH"])

# NOTE: outlier caps are NOT computed here. They are computed on the TRAIN split
# in build_features.py, so test data never influences the cleaning (no leakage).

# Save the cleaned data
clean_path = PROJECT_ROOT / "data" / "interim" / "cleaned_data.csv"
clean_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(clean_path, index=False)

print("Cleaned shape:", df.shape)
logger.info("Cleaned data saved, shape=%s", df.shape)
print("All steps executed successfully!")
