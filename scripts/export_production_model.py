# Downloads the Production model from the registry and saves it for Docker
# Run: python scripts/export_production_model.py

# os: reads DagsHub username and token
import os

# Path builds file paths that work on any computer
from pathlib import Path

# joblib: saves the model as a file for the API
import joblib

# yaml: reads params.yaml
import yaml

# dotenv: loads DAGSHUB_USERNAME and DAGSHUB_TOKEN from .env
from dotenv import load_dotenv

# mlflow: model registry
import mlflow
from mlflow.tracking import MlflowClient

# ===== LOGGER START =====
# Logger: the project logger, see src/logger.py
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from src.logger import get_logger
logger = get_logger(__name__)
# ===== LOGGER END =====

# The project folder is one folder above this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Same DagsHub login as the other scripts
load_dotenv(PROJECT_ROOT / ".env", override=True)
os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USERNAME"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]

# Read the settings from params.yaml
with open(PROJECT_ROOT / "params.yaml") as file:
    params = yaml.safe_load(file)
mlflow.set_tracking_uri(params["mlflow"]["TRACKING_URI"])
model_name = params["evaluate_model"]["REGISTERED_MODEL_NAME"]

# Find the Production version (there must be exactly one)
client = MlflowClient()
production_versions = [
    version for version in client.search_model_versions(f"name='{model_name}'")
    if version.current_stage == "Production"
]
assert len(production_versions) == 1, "Expected exactly one Production version"
version_number = production_versions[0].version

# Load the Production pipeline from the registry and save it locally
model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
models_dir = PROJECT_ROOT / "models"
models_dir.mkdir(exist_ok=True)
joblib.dump(model, models_dir / "best_model.joblib")
(models_dir / "production_version.txt").write_text(version_number)

print(f"Exported {model_name} v{version_number} to models/best_model.joblib")
logger.info("Exported %s v%s to models/best_model.joblib", model_name, version_number)