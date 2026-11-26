import os
from pathlib import Path
import joblib
import pandas as pd
import yaml
import mlflow

os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import warnings
warnings.filterwarnings("ignore")

# Step 8: preprocessing tools
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Step 9: the four models from the notebook
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Pipeline and cross-validation
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

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
train_params = params["train_model"]
mlflow_params = params["mlflow"]


from dotenv import load_dotenv

# override=True prevents a stale token exported in the shell from winning over .env
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

# MLflow + DagsHub setup: username plus an access token used as the password.
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

# Load ONLY training data. test.csv is reserved for the final evaluation.
train = pd.read_csv(project_root / "data" / "processed" / "train.csv")
X_train = train.drop(columns=[train_params["TARGET"]])
y_train = train[train_params["TARGET"]]

# ---------- Step 8: Preprocessing (from notebook) ----------
numerical_columns = [
    "latitude", "longitude", "price", "minimum_nights", "number_of_reviews",
    "reviews_per_month", "calculated_host_listings_count", "availability_365",
]
categorical_columns = ["neighbourhood_group", "neighbourhood"]

# Numeric: fill missing with median, then scale
numerical_pipeline = Pipeline(steps=[
    ("Imputer", SimpleImputer(strategy="median")),
    ("Scaler", StandardScaler()),
])

# Categorical: fill missing with most frequent, then one-hot encode
categorical_pipeline = Pipeline(steps=[
    ("Imputer", SimpleImputer(strategy="most_frequent")),
    ("Encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
])

# Combine both pipelines into one preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("Numerical Pipeline", numerical_pipeline, numerical_columns),
    ("Categorical Pipeline", categorical_pipeline, categorical_columns),
])

# ---------- Step 9: Multiple algorithms (from notebook) ----------
# class_weight="balanced" gives the rare class more attention.
# GradientBoosting has no class_weight option, so it is left as it is.
RANDOM_STATE = train_params["RANDOM_STATE"]
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE)
}

# Compare every model with the SAME preprocessor, using CV on TRAINING data only
results = []
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", clone(preprocessor)),
        ("classifier", clone(model)),
    ])

    # Accuracy alone can mislead on imbalanced classes, so macro-F1 is the main score
    accuracy = cross_val_score(pipeline, X_train, y_train, cv=train_params["CV_FOLDS"], scoring="accuracy", n_jobs=1)
    macro_f1 = cross_val_score(pipeline, X_train, y_train, cv=train_params["CV_FOLDS"], scoring="f1_macro", n_jobs=1)

    results.append({
        "Model": name,
        "CV Accuracy Mean": accuracy.mean(),
        "CV Accuracy Std": accuracy.std(),
        "CV Macro-F1 Mean": macro_f1.mean(),
        "CV Macro-F1 Std": macro_f1.std(),
    })

# Highest macro-F1 first
comparison_df = pd.DataFrame(results).sort_values(by="CV Macro-F1 Mean", ascending=False)
print(comparison_df.to_string(index=False))

# Automatic selection: the first row is the best model
best_model_name = comparison_df["Model"].iloc[0]
print("Best model:", best_model_name)
logger.info("Best model by train CV macro-F1: %s", best_model_name)

comparison_df.to_csv(reports_dir / "model_comparison.csv", index=False)

# Log the comparison to MLflow
with mlflow.start_run(run_name="model_comparison"):
    mlflow.log_param("CV_FOLDS", train_params["CV_FOLDS"])
    mlflow.log_param("selected_model", best_model_name)
    for row in results:
        key = row["Model"].lower().replace(" ", "_")
        mlflow.log_metric(f"{key}_cv_macro_f1", float(row["CV Macro-F1 Mean"]))
        mlflow.log_metric(f"{key}_cv_accuracy", float(row["CV Accuracy Mean"]))
    mlflow.log_artifact(str(reports_dir / "model_comparison.csv"))

# Save the winner's name, and an UNTRAINED copy of its full pipeline for tune_model.py
(models_dir / "best_model_name.txt").write_text(best_model_name)
untrained_pipeline = Pipeline(steps=[
    ("preprocessor", clone(preprocessor)),
    ("classifier", clone(models[best_model_name])),
])
joblib.dump(untrained_pipeline, models_dir / "untrained_pipeline.joblib")

print("Saved the best model name and untrained pipeline in:", models_dir)
logger.info("Saved untrained pipeline for %s", best_model_name)