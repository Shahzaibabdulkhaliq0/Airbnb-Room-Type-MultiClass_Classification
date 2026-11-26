# Silence the "unknown categories" warning in this script AND in the sklearn worker processes.
# This must be set BEFORE sklearn is imported.
import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import sys

# MLflow prints emoji run links; Windows consoles (cp1252) cannot encode them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

# Path builds file paths that work on any computer
from pathlib import Path

# json: saves the best parameters and run info
import json

# joblib: loads and saves the pipeline
import joblib

# pandas: load table data
import pandas as pd

# yaml: reads params.yaml
import yaml

# mlflow: experiment tracking on DagsHub
import mlflow
import mlflow.sklearn

# Step 10: RandomizedSearchCV tries random combinations and keeps the best
from sklearn.model_selection import RandomizedSearchCV

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ===== LOGGER START =====
# Logger: the project logger, see src/logger.py
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from src.logger import get_logger
logger = get_logger(__name__)
# ===== LOGGER END ======

# The project folder is two folders above this file
project_root = Path(__file__).resolve().parents[2]

# Read the settings from params.yaml
with open(project_root / "params.yaml") as file:
    params = yaml.safe_load(file)
tune_params = params["tune_model"]
mlflow_params = params["mlflow"]

# MLflow + DagsHub setup (same credentials as train_model.py)
from dotenv import load_dotenv
# override=True prevents a stale token exported in the shell from winning over .env
load_dotenv(project_root / ".env", override=True)
dagshub_token = os.environ.get("DAGSHUB_TOKEN")
assert dagshub_token, "DAGSHUB_TOKEN environment variable is not set"
os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USERNAME"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
mlflow.set_tracking_uri(mlflow_params["TRACKING_URI"])
try:
    mlflow.set_experiment(mlflow_params["EXPERIMENT_NAME"])
except mlflow.exceptions.MlflowException as exc:
    if "401" in str(exc):
        raise SystemExit(
            "DagsHub rejected the MLflow credentials. Check DAGSHUB_USERNAME and "
            "DAGSHUB_TOKEN in .env; if needed, create a fresh token at "
            "https://dagshub.com/user/settings/tokens."
        ) from exc
    raise

# Folders
models_dir = project_root / "models"
reports_dir = project_root / "reports"
models_dir.mkdir(parents=True, exist_ok=True)
reports_dir.mkdir(parents=True, exist_ok=True)

# Load the training data only
train = pd.read_csv(project_root / "data" / "processed" / "train.csv")
X_train = train.drop(columns=[tune_params["TARGET"]])
y_train = train[tune_params["TARGET"]]

# Which model won in train_model.py
best_model_name = (models_dir / "best_model_name.txt").read_text()
print("Tuning:", best_model_name)

# Load the untrained pipeline of the winner
pipeline = joblib.load(models_dir / "untrained_pipeline.joblib")

# Search space for the winner (params.yaml: param_grids)
param_dist = tune_params["PARAM_GRIDS"][best_model_name]

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=tune_params["N_ITER"],
    cv=tune_params["CV_FOLDS"],
    scoring=tune_params["SCORING"],          # f1_macro: classes are imbalanced
    random_state=tune_params["RANDOM_STATE"],
    n_jobs=-1,
)

random_search.fit(X_train, y_train)

# The best pipeline, already trained on the full training data
best_model = random_search.best_estimator_
best_params = random_search.best_params_
best_cv_macro_f1 = random_search.best_score_

print("Best Parameters :", best_params)
print("Best CV Macro-F1:", best_cv_macro_f1)
logger.info("Tuning done: best_cv_macro_f1=%s", best_cv_macro_f1)

# Save local files FIRST, so they exist even if MLflow fails
joblib.dump(best_model, models_dir / "best_model.joblib")
with open(reports_dir / "best_params.json", "w") as file:
    json.dump({"model": best_model_name, "best_params": best_params, "best_cv_macro_f1": best_cv_macro_f1}, file, indent=4, default=str)

# Log to MLflow and save the run id for evaluate_model.py
with mlflow.start_run(run_name="tune_model") as run:
    mlflow.log_param("selected_model", best_model_name)
    mlflow.log_params({key: str(value) for key, value in best_params.items()})
    mlflow.log_metric("best_cv_macro_f1", float(best_cv_macro_f1))
    mlflow.log_artifact(str(reports_dir / "best_params.json"))
    model_info = mlflow.sklearn.log_model(
        best_model,
        name="model",
        skops_trusted_types=["sklearn.tree._tree.Tree", "numpy.dtype"],
    )
    run_id = run.info.run_id

with open(reports_dir / "run_info.json", "w") as file:
    json.dump({"run_id": run_id, "model_path": "model", "model_uri": model_info.model_uri}, file, indent=4)

print("Final pipeline saved to:", models_dir / "best_model.joblib")
print("MLflow run_id:", run_id)
logger.info("Tuned model logged to MLflow run %s", run_id)