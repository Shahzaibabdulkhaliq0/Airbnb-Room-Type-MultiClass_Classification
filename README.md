# NYC Airbnb Room Type Classification: End-to-End MLOps Project

[![CI Pipeline](https://github.com/Shahzaibabdulkhaliq0/Airbnb-Room-Type-MultiClass_Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/Shahzaibabdulkhaliq0/Airbnb-Room-Type-MultiClass_Classification/actions/workflows/ci.yml)

An end-to-end MLOps project that predicts the **room type** of NYC Airbnb listings (`Entire home/apt`, `Private room`, `Shared room`). The project covers data versioning, a DVC pipeline, experiment tracking and a Model Registry with a quality gate, a FastAPI REST API with an HTML frontend, Docker, CI/CD with GitHub Actions, deployment on AWS EKS (Kubernetes), and monitoring with Prometheus and Grafana.

**Video walkthrough:** [Watch on YouTube](https://www.youtube.com/watch?v=u5y4Pfc6Pbs)

---

## Table of Contents

1. [Project Highlights](#project-highlights)
2. [Architecture](#architecture)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [ML Pipeline (DVC)](#ml-pipeline-dvc)
6. [Model Training, Tuning and Selection](#model-training-tuning-and-selection)
7. [Experiment Tracking and Model Registry](#experiment-tracking-and-model-registry)
8. [Data Validation and Drift Detection](#data-validation-and-drift-detection)
9. [FastAPI REST API](#fastapi-rest-api)
10. [Testing](#testing)
11. [Docker](#docker)
12. [CI/CD with GitHub Actions](#cicd-with-github-actions)
13. [Retraining Workflow](#retraining-workflow)
14. [Deployment on AWS EKS](#deployment-on-aws-eks)
15. [Monitoring with Prometheus and Grafana](#monitoring-with-prometheus-and-grafana)
16. [Results](#results)
17. [How to Run Locally](#how-to-run-locally)
18. [Configuration and Secrets](#configuration-and-secrets)
19. [Known Limitations and Next Steps](#known-limitations-and-next-steps)

---

## Project Highlights

- **Data versioning** of the raw dataset with **DVC**, stored in **AWS S3**
- **Reproducible 7-stage DVC pipeline**: data cleaning, data validation, feature engineering, training, tuning, evaluation and drift check
- **Data leakage prevention**: the drift baseline comes from the training split only, and preprocessing runs inside cross-validation
- **Model comparison** across 4 classifiers with cross-validation and class imbalance handling
- **Hyperparameter tuning** with `RandomizedSearchCV`
- **Experiment tracking** and **Model Registry** with **MLflow** (hosted on DagsHub)
- **Quality gate**: a model is promoted from Staging to Production only if it passes accuracy and macro-F1 thresholds
- **FastAPI REST API** with input validation (Pydantic), health checks, and an HTML frontend
- **Containerized** with **Docker** and pushed to **AWS ECR**
- **CI/CD** with **GitHub Actions**: tests, model quality gate, image build, smoke test, push to ECR, deploy to **AWS EKS**
- **Kubernetes (EKS)** deployment with 2 replicas, readiness and liveness probes, and a LoadBalancer service
- **Monitoring** with **Prometheus** and **Grafana**, using custom metrics exposed by the API
- **Retraining workflow** that can be started from the GitHub Actions tab, with an option to force a full rerun
- **26 pytest tests** covering the API, data checks, logging, the quality gate and promotion rules

---

## Architecture

```text
                 ┌──────────────┐
   Raw CSV ────► │  AWS S3      │  (DVC remote)
                 └──────┬───────┘
                        │ dvc pull
                        ▼
  ┌──────────────────────────────────────────────────┐
  │ DVC pipeline                                     │
  │ make_dataset → validate_data → build_features →  │
  │ train_model → tune_model → evaluate_model →      │
  │ check_drift                                      │
  └─────────────────────┬────────────────────────────┘
                        │ log runs, register model
                        ▼
  ┌──────────────────────────────────────────────────┐
  │ MLflow on DagsHub: experiments + Model Registry  │
  │ my_model: Staging → Production (quality gate)    │
  └─────────────────────┬────────────────────────────┘
                        │ export Production model
                        ▼
  ┌──────────────────────────────────────────────────┐
  │ GitHub Actions (CI/CD)                           │
  │ test → build → smoke test → push to ECR → deploy │
  └─────────────────────┬────────────────────────────┘
                        ▼
  ┌──────────────────────────────────────────────────┐
  │ AWS EKS (Kubernetes)                             │
  │ FastAPI pods (2 replicas) + LoadBalancer         │
  │ /metrics → Prometheus → Grafana                  │
  └──────────────────────────────────────────────────┘
```

---

## Tech Stack

| Area | Tools |
|---|---|
| Language and ML | Python, pandas, NumPy, scikit-learn, joblib |
| MLOps | DVC, MLflow, DagsHub, Model Registry, experiment tracking |
| Data | AWS S3, data validation, data drift detection |
| API | FastAPI, Pydantic, Uvicorn, HTML frontend |
| Containers | Docker, AWS ECR |
| Orchestration | Kubernetes (AWS EKS) |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus, Grafana, prometheus-client |
| Testing | pytest, httpx |
| Cloud | AWS (S3, ECR, EKS, IAM) |

---

## Project Structure

```text
.
├── .github/workflows/
│   ├── ci.yml                  # Tests, image build, ECR push, EKS deploy
│   └── retrain.yml             # Manual retraining workflow
├── api/
│   ├── main.py                 # FastAPI app: /, /health, /predict, /metrics
│   └── index.html              # Frontend page served by FastAPI
├── src/
│   ├── data/
│   │   ├── make_dataset.py     # Load from S3, clean, split
│   │   ├── validate_data.py    # Data validation checks
│   │   └── check_drift.py      # Drift check against the training baseline
│   ├── features/
│   │   └── build_features.py   # Feature engineering, outlier capping, baseline stats
│   ├── model/
│   │   ├── train_model.py      # Model comparison with cross-validation
│   │   ├── tune_model.py       # RandomizedSearchCV
│   │   ├── evaluate_model.py   # Test metrics, confusion matrix, Model Registry (Staging)
│   │   ├── promotion_rules.py  # Quality gate and promotion rules
│   │   └── predict_model.py
│   └── logger/                 # Project logger
├── scripts/
│   ├── export_production_model.py  # Download the Production model for Docker
│   └── promote_model.py            # Promote the best Staging model to Production
├── tests/                      # pytest tests (API, data checks, logger, promotion, model quality)
├── notebooks/                  # Exploratory notebook
├── dvc.yaml                    # DVC pipeline stages
├── params.yaml                 # All pipeline settings and thresholds
├── Dockerfile
├── deployment.yaml             # Kubernetes Deployment and Service
├── requirements.txt            # Pipeline dependencies
├── requirements-api.txt        # API runtime dependencies
└── requirements-dev.txt        # Test dependencies
```

---

## ML Pipeline (DVC)

The pipeline is defined in `dvc.yaml` and every setting lives in `params.yaml`.

| Stage | What it does | Output |
|---|---|---|
| `make_dataset` | Loads the CSV from S3, drops unused columns, cleans missing values | `data/processed/` |
| `validate_data` | Checks required columns, row count, price and availability ranges, allowed room types, and null share. Fails the pipeline on error | `reports/data_validation.json` |
| `build_features` | Feature engineering, outlier capping (quantile based), train/test split. Saves baseline statistics from the training split only | `models/`, `reports/baseline_stats.json` |
| `train_model` | Compares Logistic Regression, Decision Tree, Random Forest and Gradient Boosting with cross-validation | MLflow run |
| `tune_model` | `RandomizedSearchCV` on the best model family | MLflow run, best model |
| `evaluate_model` | Test accuracy, macro-F1, confusion matrix. Registers the model as `my_model` in Staging | MLflow Model Registry |
| `check_drift` | Compares current data with the baseline (average price and room type shares). Warning only | `reports/drift_report.json` |

Run the whole pipeline:

```bash
dvc repro
```

---

## Model Training, Tuning and Selection

- **Preprocessing pipeline:** `ColumnTransformer` with `SimpleImputer`, `StandardScaler` for numeric columns, and `OneHotEncoder` for categorical columns, inside a scikit-learn `Pipeline`.
- **Data leakage prevention:** the drift baseline is saved from the training split only, and preprocessing runs inside cross-validation, so each fold is fitted on its own training data.
- **Class imbalance:** `class_weight="balanced"` for the models that support it.
- **Model comparison:** 4 algorithms, each scored with cross-validated accuracy and macro-F1.
- **Hyperparameter tuning:** `RandomizedSearchCV` with `f1_macro` as the scoring metric, configured in `params.yaml`.
- **Selection:** the best model is chosen from the cross-validation results, and then evaluated on the held-out test set.

---

## Experiment Tracking and Model Registry

- Every run logs parameters, cross-validation metrics, test metrics and the confusion matrix to **MLflow** (hosted on **DagsHub**).
- The best model is registered as **`my_model`** and moved to **Staging**.
- `scripts/promote_model.py` moves a Staging model to **Production** only if it passes the quality gate and beats the current Production model.
- The CI pipeline and the Docker image always use the **Production** model.

**Quality gate** (from `params.yaml`):

| Metric | Minimum |
|---|---|
| Test accuracy | 0.70 |
| Test macro-F1 | 0.55 |

A failed gate never promotes a model, and the current Production model stays in place.

---

## Data Validation and Drift Detection

**Data validation** (`src/data/validate_data.py`) runs before training and stops the pipeline on bad data:
- Required columns are present
- Minimum number of rows
- Price is not negative
- Availability is between 0 and 365 days
- `room_type` has only the allowed values
- Share of missing values is below the limit

**Drift detection** (`src/data/check_drift.py`) compares the current data with the training baseline:
- Change in average price
- Change in room type shares

It writes a report to `reports/drift_report.json`. Drift is reported as a **warning only**, so the pipeline keeps running. Thresholds are set in `params.yaml`.

---

## FastAPI REST API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | HTML frontend for entering a listing and seeing the prediction |
| `GET` | `/health` | Health check, also used by Kubernetes probes |
| `POST` | `/predict` | Predicts the room type and class probabilities |
| `GET` | `/metrics` | Prometheus metrics |

**Input validation:** the `/predict` request body is validated with Pydantic. For example, price and minimum nights must be 0 or more, and availability must be between 0 and 365.

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "neighbourhood_group": "Manhattan",
    "neighbourhood": "Harlem",
    "latitude": 40.80,
    "longitude": -73.94,
    "price": 120,
    "minimum_nights": 2,
    "number_of_reviews": 25,
    "reviews_per_month": 1.5,
    "calculated_host_listings_count": 1,
    "availability_365": 200
  }'
```

**Example response** (illustrative values, not a real model output):

```json
{
  "room_type": "Private room",
  "probabilities": {
    "Entire home/apt": 0.21,
    "Private room": 0.70,
    "Shared room": 0.09
  }
}
```

The same caps used in training (price and minimum nights) are applied at prediction time.

**Custom metrics** exposed at `/metrics`:
- `predictions_total`: number of predictions, labelled by room type
- `predict_latency_seconds`: histogram of prediction time

---

## Testing

```bash
pytest -q --ignore=tests/test_model_quality.py   # Fast tests (API, data checks, logger, promotion rules)
pytest -q tests/test_model_quality.py             # Quality gate on the real Production model (needs DagsHub access)
```

| Test file | Tests | What it covers |
|---|---|---|
| `test_api.py` | 6 | Health, predictions, validation errors, price cap, metrics endpoint |
| `test_data_checks.py` | 7 | Validation pass and fail cases, drift detected and not detected |
| `test_logger.py` | 2 | Logger setup |
| `test_promotion_rules.py` | 10 | Quality gate limits, promotion and keeping Production |
| `test_model_quality.py` | 1 | Production model meets the thresholds |

---

## Docker

The image is built from `python:3.14.10`. It contains the API code, the `src/` logger, and the Production model files.

```bash
docker build -t airbnb-api .
docker run -d -p 8000:8000 --name airbnb-api airbnb-api
```

Then open `http://localhost:8000` (frontend) or `http://localhost:8000/docs` (Swagger UI).

---

## CI/CD with GitHub Actions

**`ci.yml`** runs on every pull request to `main`, and on every push to `main`:

1. **test:** installs dependencies, pulls test data from S3 with DVC, runs the API tests and the model quality gate
2. **build-and-push:** exports the Production model from the registry, builds the Docker image, runs a smoke test on `/health`, and pushes the image to ECR, tagged with the commit SHA and `latest`
3. **deploy:** updates `deployment.yaml` with the new image, applies it to EKS, and waits for the rollout to finish

Pushes to `main` deploy automatically. The workflow can also be started by hand from the Actions tab with the `deploy` option.

---

## Retraining Workflow

**`retrain.yml`** is a manual workflow (Actions → **Retrain model** → **Run workflow**):

1. Pulls the data from S3
2. Runs `dvc repro` (with `force` option to rerun every stage)
3. Promotes the best Staging model if it passes the gate
4. Pushes the new DVC outputs to S3
5. Commits the new `dvc.lock` and reports to the repo
6. Starts `ci.yml` with `deploy=true`, which builds, pushes and deploys the new model

There is no automatic schedule, so retraining never runs by accident.

---

## Deployment on AWS EKS

- **Cluster:** `airbnb-api-cluster` (eu-north-1), managed node group with 2 nodes
- **Deployment:** 2 replicas, resource requests and limits, readiness probe and liveness probe on `/health`
- **Service:** `LoadBalancer` on port 80, forwarding to port 8000 in the pods
- **Image:** pulled from AWS ECR, tagged with the commit SHA for rollback

Useful commands:

```bash
kubectl get deployment airbnb-api
kubectl rollout status deployment/airbnb-api
kubectl rollout history deployment/airbnb-api
```

---

## Monitoring with Prometheus and Grafana

- Prometheus is installed on the cluster with the `kube-prometheus-stack` Helm chart.
- Each pod has Prometheus scrape annotations, and a `ServiceMonitor` in the `monitoring` namespace tells Prometheus to scrape `/metrics`.
- Grafana runs on the cluster and reads from Prometheus.

**Grafana dashboard:**

![Grafana dashboard](docs/images/grafana-dashboard.png)

**Useful PromQL queries:**

```promql
sum by (room_type) (predictions_total)
rate(predictions_total[5m])
histogram_quantile(0.95, rate(predict_latency_seconds_bucket[5m]))
```

---

## Results

| Metric | Value |
|---|---|
| Test accuracy | 85.5% (0.8553) |
| Test macro-F1 | 0.762 (0.7624) |
| Selected model | [best model name] |
| Quality gate | Passed: accuracy 0.855 ≥ 0.70 and macro-F1 0.762 ≥ 0.55 |

**Confusion matrix:**

![Confusion matrix](docs/images/confusion-matrix.png)

**MLflow Model Registry:**

![MLflow Model Registry](docs/images/mlflow-registry.png)

---

## How to Run Locally

**Prerequisites:** Python 3.14, Git, DVC, Docker (optional), AWS credentials with access to the S3 bucket, and DagsHub credentials.

```bash
# 1. Clone the repo
git clone https://github.com/Shahzaibabdulkhaliq0/Airbnb-Room-Type-MultiClass_Classification.git
cd Airbnb-Room-Type-MultiClass_Classification

# 2. Install dependencies
pip install -r requirements-dev.txt

# 3. Pull the data and the saved outputs from S3
dvc pull

# 4. Run the pipeline (optional, rebuilds the model)
dvc repro

# 5. Run the tests
pytest -q --ignore=tests/test_model_quality.py

# 6. Start the API
uvicorn api.main:app --reload --port 8000
```

Open `http://localhost:8000`.

---

## Configuration and Secrets

- All settings and thresholds are in **`params.yaml`**.
- Secrets are **never** stored in code or in `params.yaml`. They are read from environment variables:
  - `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`: MLflow on DagsHub
  - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`: AWS access (for DVC and CI)
- In GitHub Actions, these are stored as **repository secrets**.

---

## Known Limitations and Next Steps

- Drift detection is basic (average price and room type shares) and only gives a warning. Next: per-feature drift tests and alerting.
- Grafana alerts are not configured yet.
- Infrastructure is created with `eksctl` and CLI commands. Next: move it to Terraform.
- No feature store yet. Next: add a feature store for consistent training and serving features.
- The model is a classical ML model (scikit-learn). Next: add a batch inference job.

---

## Author

**Shahzaib Abdul Khaliq**
[LinkedIn](https://www.linkedin.com/in/shahzaibabdulkhaliq/) | [GitHub](https://github.com/Shahzaibabdulkhaliq0)
