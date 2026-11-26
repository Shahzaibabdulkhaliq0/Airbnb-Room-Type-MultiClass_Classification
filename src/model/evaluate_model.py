# Silence the "unknown categories" warning (unseen neighbourhoods in the test data)
import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import sys

# MLflow prints emoji run links; Windows consoles (cp1252) cannot encode them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

# Path builds file paths that work on any computer
from pathlib import Path

# json: saves the metrics and reads the run info
import json

# joblib: loads the final pipeline
import joblib

# pandas: load table data
import pandas as pd

# yaml: reads params.yaml
import yaml

# matplotlib: draws the confusion matrix
import matplotlib
# "Agg" saves plots as files without opening windows
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# seaborn: heatmap
import seaborn as sns

# mlflow: tracking and model registry
import mlflow
from mlflow.tracking import MlflowClient

# Step 11: final metrics
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report

# ===== LOGGER START =====
# Logger: the project logger, see src/logger.py
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from src.logger import get_logger
logger = get_logger(__name__)
# ===== LOGGER END =====

# The project folder is two folders above this file
project_root = Path(__file__).resolve().parents[2]

# Read the settings from params.yaml
with open(project_root / "params.yaml") as file:
    params = yaml.safe_load(file)
eval_params = params["evaluate_model"]
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
figures_dir = reports_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# Load the TEST data. It is used only here, never for model selection.
test = pd.read_csv(project_root / "data" / "processed" / "test.csv")
X_test = test.drop(columns=[eval_params["TARGET"]])
y_test = test[eval_params["TARGET"]]

# Load the final tuned pipeline
model = joblib.load(models_dir / "best_model.joblib")

# Predict on the untouched test set
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print("Test Accuracy :", round(accuracy, 4))
print("Test Macro-F1 :", round(macro_f1, 4))
logger.info("Test metrics: accuracy=%.4f macro_f1=%.4f", accuracy, macro_f1)
print(classification_report(y_test, y_pred))

# First 10 real rooms next to the predictions
comparison = pd.DataFrame({"Actual Room": y_test.values[:10], "Predicted Room": y_pred[:10]})
print(comparison)

metrics = {"test_accuracy": float(accuracy), "test_macro_f1": float(macro_f1)}
with open(reports_dir / "metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

# Confusion matrix (counts)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(figures_dir / "confusion_matrix.png", bbox_inches="tight")
plt.close()

# Log metrics and the plot to the SAME MLflow run that tuned the model
with open(reports_dir / "run_info.json") as file:
    run_info = json.load(file)

with mlflow.start_run(run_id=run_info["run_id"]):
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(str(figures_dir / "confusion_matrix.png"))

# Model registry: register as my_model and move it to Staging
model_name = eval_params["REGISTERED_MODEL_NAME"]
model_uri = run_info["model_uri"]
version = mlflow.register_model(model_uri, model_name)
MlflowClient().transition_model_version_stage(
    name=model_name,
    version=version.version,
    stage="Staging",
)
print(f"{model_name} v{version.version} registered in Staging")
logger.info("%s v%s registered in Staging", model_name, version.version)