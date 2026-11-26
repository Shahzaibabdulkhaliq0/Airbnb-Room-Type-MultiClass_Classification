# This file runs BEFORE the tests. It builds a small fake model and caps,
# so the API tests do not need S3, DagsHub or the real trained model.
import os
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Same column lists as train_model.py
numerical_columns = [
    "latitude", "longitude", "price", "minimum_nights", "number_of_reviews",
    "reviews_per_month", "calculated_host_listings_count", "availability_365",
]
categorical_columns = ["neighbourhood_group", "neighbourhood"]

# Make 300 fake listings with the same columns
rng = np.random.default_rng(42)
rows = 300
fake = pd.DataFrame({
    "neighbourhood_group": rng.choice(["Manhattan", "Brooklyn"], rows),
    "neighbourhood": rng.choice(["Midtown", "Harlem", "Astoria"], rows),
    "latitude": rng.uniform(40.5, 40.9, rows),
    "longitude": rng.uniform(-74.2, -73.7, rows),
    "price": rng.uniform(20, 500, rows),
    "minimum_nights": rng.integers(1, 60, rows),
    "number_of_reviews": rng.integers(0, 100, rows),
    "reviews_per_month": rng.uniform(0, 5, rows),
    "calculated_host_listings_count": rng.integers(1, 10, rows),
    "availability_365": rng.integers(0, 366, rows),
})
fake_labels = rng.choice(["Entire home/apt", "Private room", "Shared room"], rows)

# Same preprocessing as train_model.py
preprocessor = ColumnTransformer(transformers=[
    ("Numerical Pipeline", Pipeline(steps=[
        ("Imputer", SimpleImputer(strategy="median")),
        ("Scaler", StandardScaler()),
    ]), numerical_columns),
    ("Categorical Pipeline", Pipeline(steps=[
        ("Imputer", SimpleImputer(strategy="most_frequent")),
        ("Encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
    ]), categorical_columns),
])
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])
pipeline.fit(fake, fake_labels)

# Save the fake model and caps into a temporary folder
test_dir = Path(tempfile.mkdtemp())
model_file = test_dir / "best_model.joblib"
caps_file = test_dir / "caps.json"
joblib.dump(pipeline, model_file)
with open(caps_file, "w") as file:
    json.dump({"price_cap": 300.0, "nights_cap": 30.0}, file)

# api/main.py reads these paths when it is imported, so set them first
os.environ["MODEL_PATH"] = str(model_file)
os.environ["CAPS_PATH"] = str(caps_file)