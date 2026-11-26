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
    params = yaml.safe_load(file)["check_drift"]

# Baseline = numbers saved from the training split (build_features.py)
with open(PROJECT_ROOT / "reports" / "baseline_stats.json") as file:
    baseline = json.load(file)

# New data = the cleaned data from make_dataset.py
df = pd.read_csv(PROJECT_ROOT / "data" / "interim" / "cleaned_data.csv", low_memory=False)

# Check 1: change in average price, as a fraction of the baseline average
new_price_mean = float(df["price"].mean())
price_change = abs(new_price_mean - baseline["price_mean"]) / baseline["price_mean"]

# Check 2: change in the share of each room type (largest change counts)
new_shares = df["room_type"].value_counts(normalize=True)
share_changes = {}
for room_type, old_share in baseline["room_type_share"].items():
    share_changes[room_type] = abs(float(new_shares.get(room_type, 0.0)) - old_share)
max_share_change = max(share_changes.values())

# Drift is a WARNING, not an error. The pipeline keeps running so you can read the report first.
drift_detected = bool(
    price_change > params["MAX_PRICE_MEAN_CHANGE"]
    or max_share_change > params["MAX_ROOM_TYPE_SHARE_CHANGE"]
)

report = {
    "new_price_mean": new_price_mean,
    "baseline_price_mean": baseline["price_mean"],
    "price_mean_change": float(price_change),
    "room_type_share_change": share_changes,
    "drift_detected": drift_detected,
}
report_path = PROJECT_ROOT / "reports" / "drift_report.json"
report_path.parent.mkdir(parents=True, exist_ok=True)
with open(report_path, "w") as file:
    json.dump(report, file, indent=4)

print(report)
logger.info("Drift check: drift_detected=%s price_change=%.3f max_share_change=%.3f",
            drift_detected, price_change, max_share_change)