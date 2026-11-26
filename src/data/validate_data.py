from pathlib import Path
import json
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "params.yaml") as file:
    params = yaml.safe_load(file)["validate_data"]

df = pd.read_csv(PROJECT_ROOT / "data" / "interim" / "cleaned_data.csv", low_memory=False)

# Stop early if a required column is missing, the checks below need these columns
missing = [col for col in params["REQUIRED_COLUMNS"] if col not in df.columns]
assert not missing, f"Missing columns: {missing}"

# Each check is True (passed) or False (failed)
checks = {
    "min_rows": bool(len(df) >= params["MIN_ROWS"]),
    "price_not_negative": bool(df["price"].min() >= params["MIN_PRICE"]),
    "availability_in_range": bool(df["availability_365"].between(params["MIN_AVAILABILITY"], params["MAX_AVAILABILITY"]).all()),
    "room_type_allowed": bool(df["room_type"].isin(params["ALLOWED_ROOM_TYPES"]).all()),
    "null_share_ok": bool(df.isnull().mean().max() <= params["MAX_NULL_SHARE"]),
}
all_passed = all(checks.values())

# Save the report first, so you can read it even when a check fails
report = {"checks": checks, "all_passed": all_passed}
report_path = PROJECT_ROOT / "reports" / "data_validation.json"
report_path.parent.mkdir(parents=True, exist_ok=True)
with open(report_path, "w") as file:
    json.dump(report, file, indent=4)

print(report)
logger.info("Data validation: %s", report)

# assert stops the pipeline (dvc repro, retrain.yml) when any check fails
assert all_passed, f"Data validation FAILED: {checks}"
print("Data validation passed") 