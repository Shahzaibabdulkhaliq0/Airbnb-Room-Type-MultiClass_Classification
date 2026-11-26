# FastAPI app: serves the final model from models/best_model.joblib

# This line hides only UserWarning messages (for example "unknown categories") from the logs.
# It must run BEFORE sklearn is imported.
import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

# json: reads the cap values saved by make_dataset.py
import json

# Path builds file paths that work on any computer
from pathlib import Path

# joblib: loads the trained pipeline
import joblib

# pandas: makes the one-row table the model needs
import pandas as pd

# FastAPI: web framework. Field: adds rules to the input (for example, price >= 0)
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ===== LOGGER START =====
# Logger: the project logger, see src/logger.py
from src.logger import get_logger
logger = get_logger(__name__)
# ===== LOGGER END =====

# prometheus_client: counts predictions and measures how long they take, for Prometheus/Grafana
from prometheus_client import Counter, Histogram, make_asgi_app

# The project folder is one folder above this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Use the env variables if they are set (tests and Docker), otherwise the default files
MODEL_PATH = Path(os.environ.get("MODEL_PATH", PROJECT_ROOT / "models" / "best_model.joblib"))
CAPS_PATH = Path(os.environ.get("CAPS_PATH", PROJECT_ROOT / "models" / "caps.json"))

# Load the model and caps ONCE when the server starts, not on every request
model = joblib.load(MODEL_PATH)
with open(CAPS_PATH) as file:
    caps = json.load(file)

# ===== LOGGER START =====
logger.info("Model loaded from %s, caps=%s", MODEL_PATH, caps)
# ===== LOGGER END =====
# Metrics: how many predictions per room type, and how long each prediction takes
PREDICTIONS_TOTAL = Counter("predictions_total", "Total predictions", ["room_type"])
PREDICT_LATENCY = Histogram("predict_latency_seconds", "Time taken by one /predict request")

app = FastAPI(title="NYC Airbnb room type API")

# Expose the metrics at /metrics (Prometheus scrapes this URL)
app.mount("/metrics", make_asgi_app())


# The input the client must send. These are the same 10 columns the model was trained on.
class Listing(BaseModel):
    neighbourhood_group: str
    neighbourhood: str
    latitude: float
    longitude: float
    price: float = Field(ge=0)
    minimum_nights: float = Field(ge=0)
    number_of_reviews: float = Field(ge=0)
    reviews_per_month: float = Field(ge=0)
    calculated_host_listings_count: float = Field(ge=0)
    availability_365: float = Field(ge=0, le=365)


# Front end page: returns only api/index.html (not the other files in this folder)
@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).resolve().parent / "index.html")


# Health check: the server is up and tells which model file it loaded.
# Kubernetes uses this URL for its readiness and liveness probes.
@app.get("/health")
def health():
    return {"status": "ok", "model_file": MODEL_PATH.name}


# Prediction: apply the SAME caps as training, then predict the room type
@app.post("/predict")
def predict(listing: Listing):
    # Measure how long this request takes
    with PREDICT_LATENCY.time():
        row = listing.model_dump()

        # Cap the outliers, exactly like make_dataset.py did
        row["price"] = min(row["price"], caps["price_cap"])
        row["minimum_nights"] = min(row["minimum_nights"], caps["nights_cap"])

        # The model expects a table, not a dictionary
        listing_df = pd.DataFrame([row])

        prediction = str(model.predict(listing_df)[0])
        probabilities = model.predict_proba(listing_df)[0]

    # Count this prediction under its room type
    PREDICTIONS_TOTAL.labels(room_type=prediction).inc()

    # ===== LOGGER START =====
    logger.info("Prediction: room_type=%s", prediction)
    # ===== LOGGER END =====

    return {
        "room_type": prediction,
        "probabilities": {str(room): float(prob) for room, prob in zip(model.classes_, probabilities)},
    }