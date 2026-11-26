from pathlib import Path
import json
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# ===== LOGGER START =====
# Logger: the project logger, see src/logger.py
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from src.logger import get_logger
logger = get_logger(__name__)
# ===== LOGGER END =====

# The project folder is two folders above this file
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Read the settings from params.yaml
with open(PROJECT_ROOT / "params.yaml") as file:
    params = yaml.safe_load(file)["build_features"]

# Load the cleaned data from make_dataset.py
clean_path = PROJECT_ROOT / "data" / "interim" / "cleaned_data.csv"
df = pd.read_csv(clean_path, low_memory=False)

# Define target
TARGET = params["TARGET"]
X = df.drop(columns=[TARGET])
y = df[TARGET]
print("X Shape:", X.shape)
print("y Shape:", y.shape)

# Train-Test Split
# stratify=y keeps the class ratios the same in both parts (the target is imbalanced)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=params["TEST_SIZE"],
    random_state=params["RANDOM_STATE"],
    stratify=y,
)

# Join inputs and target again, so each file holds everything in one table
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

# Baseline stats from the TRAIN split (before clipping).
# check_drift.py compares new data with these numbers later.
baseline = {
    "price_mean": float(train["price"].mean()),
    "room_type_share": {str(k): float(v) for k, v in train[TARGET].value_counts(normalize=True).items()},
}
reports_folder = PROJECT_ROOT / "reports"
reports_folder.mkdir(parents=True, exist_ok=True)
with open(reports_folder / "baseline_stats.json", "w") as file:
    json.dump(baseline, file, indent=4)

# Outlier caps are computed on the TRAIN split only (no test leakage)
price_cap = train["price"].quantile(params["PRICE_CAP_QUANTILE"])
nights_cap = train["minimum_nights"].quantile(params["MIN_NIGHTS_CAP_QUANTILE"])

# The same train caps are applied to both train and test
train["price"] = train["price"].clip(upper=price_cap)
train["minimum_nights"] = train["minimum_nights"].clip(upper=nights_cap)
test["price"] = test["price"].clip(upper=price_cap)
test["minimum_nights"] = test["minimum_nights"].clip(upper=nights_cap)

# Save the caps, predict_model.py and the API need the same numbers
caps_path = PROJECT_ROOT / "models" / "caps.json"
caps_path.parent.mkdir(parents=True, exist_ok=True)
with open(caps_path, "w") as file:
    json.dump({"price_cap": float(price_cap), "nights_cap": float(nights_cap)}, file, indent=4)

print("Train Shape :", train.shape)
print("Test  Shape :", test.shape)
print("Caps (from train):", {"price_cap": float(price_cap), "nights_cap": float(nights_cap)})

# Save the train and test data
Processed_Folder = PROJECT_ROOT / "data" / "processed"
Processed_Folder.mkdir(parents=True, exist_ok=True)
train.to_csv(Processed_Folder / "train.csv", index=False)
test.to_csv(Processed_Folder / "test.csv", index=False)
print("Saved train.csv and test.csv in:", Processed_Folder)
logger.info("Saved train/test split, train=%s test=%s", train.shape, test.shape)
logger.info("Caps from train: price_cap=%s nights_cap=%s", float(price_cap), float(nights_cap))