# Tests for the data checks.
# validate_data.py must STOP on bad data. check_drift.py only WARNS.
# Each test copies the scripts into a temp folder, makes a small fake dataset, and runs the script.
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room"]


def make_data(rows=1200):
    # Same columns the scripts expect. These values pass every check.
    return pd.DataFrame({
        "latitude": [40.7] * rows,
        "longitude": [-73.9] * rows,
        "price": [100.0 + (i % 50) for i in range(rows)],
        "minimum_nights": [1 + (i % 5) for i in range(rows)],
        "number_of_reviews": [i % 20 for i in range(rows)],
        "reviews_per_month": [1.0] * rows,
        "calculated_host_listings_count": [1] * rows,
        "availability_365": [100 + (i % 200) for i in range(rows)],
        "room_type": [ROOM_TYPES[i % 3] for i in range(rows)],
        "neighbourhood_group": ["Brooklyn"] * rows,
        "neighbourhood": ["Williamsburg"] * rows,
    })


def make_project(tmp_path, frame):
    # Copy params and src into a temp folder that acts as the project root
    shutil.copy(REPO / "params.yaml", tmp_path / "params.yaml")
    shutil.copytree(REPO / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "data" / "interim").mkdir(parents=True)
    (tmp_path / "reports").mkdir()
    frame.to_csv(tmp_path / "data" / "interim" / "cleaned_data.csv", index=False)
    return tmp_path


def write_baseline(project, frame):
    # Same kind of baseline that build_features.py saves
    baseline = {
        "price_mean": float(frame["price"].mean()),
        "room_type_share": {str(k): float(v) for k, v in frame["room_type"].value_counts(normalize=True).items()},
    }
    (project / "reports" / "baseline_stats.json").write_text(json.dumps(baseline))


def run_script(project, script):
    return subprocess.run(
        [sys.executable, str(project / "src" / "data" / script)],
        capture_output=True, text=True, cwd=project,
    )


def read_report(project, name):
    return json.loads((project / "reports" / name).read_text())


# ---------- validate_data.py ----------

def test_validation_passes_on_good_data(tmp_path):
    project = make_project(tmp_path, make_data())
    result = run_script(project, "validate_data.py")
    assert result.returncode == 0, result.stderr
    assert read_report(project, "data_validation.json")["all_passed"] is True


def test_validation_fails_on_negative_price(tmp_path):
    frame = make_data()
    frame.loc[0, "price"] = -5
    project = make_project(tmp_path, frame)
    result = run_script(project, "validate_data.py")
    assert result.returncode != 0
    assert read_report(project, "data_validation.json")["checks"]["price_not_negative"] is False


def test_validation_fails_on_unknown_room_type(tmp_path):
    frame = make_data()
    frame.loc[0, "room_type"] = "Castle"
    project = make_project(tmp_path, frame)
    result = run_script(project, "validate_data.py")
    assert result.returncode != 0
    assert read_report(project, "data_validation.json")["checks"]["room_type_allowed"] is False


def test_validation_fails_on_missing_column(tmp_path):
    project = make_project(tmp_path, make_data().drop(columns=["price"]))
    result = run_script(project, "validate_data.py")
    assert result.returncode != 0
    assert "Missing columns" in result.stderr


def test_validation_fails_on_too_few_rows(tmp_path):
    project = make_project(tmp_path, make_data(rows=100))
    result = run_script(project, "validate_data.py")
    assert result.returncode != 0
    assert read_report(project, "data_validation.json")["checks"]["min_rows"] is False


# ---------- check_drift.py ----------

def test_drift_not_detected_on_same_data(tmp_path):
    frame = make_data()
    project = make_project(tmp_path, frame)
    write_baseline(project, frame)
    result = run_script(project, "check_drift.py")
    assert result.returncode == 0, result.stderr
    assert read_report(project, "drift_report.json")["drift_detected"] is False


def test_drift_detected_but_pipeline_keeps_running(tmp_path):
    base = make_data()
    project = make_project(tmp_path, base)
    write_baseline(project, base)
    # New data with prices doubled: big drift
    shifted = make_data()
    shifted["price"] = shifted["price"] * 2
    shifted.to_csv(project / "data" / "interim" / "cleaned_data.csv", index=False)
    result = run_script(project, "check_drift.py")
    # Warning only: the script must still exit with 0
    assert result.returncode == 0, result.stderr
    assert read_report(project, "drift_report.json")["drift_detected"] is True