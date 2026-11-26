# API tests: run with "pytest" from the project folder
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

# One valid listing, same format as predict_model.py
valid_listing = {
    "neighbourhood_group": "Manhattan",
    "neighbourhood": "Midtown",
    "latitude": 40.75362,
    "longitude": -73.98377,
    "price": 225,
    "minimum_nights": 1,
    "number_of_reviews": 45,
    "reviews_per_month": 0.38,
    "calculated_host_listings_count": 2,
    "availability_365": 355,
}

ALLOWED_ROOMS = {"Entire home/apt", "Private room", "Shared room"}


# The server is up and reports the model file
def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# A valid listing gets a known room type and probabilities that add up to 1
def test_predict_returns_valid_room_type():
    response = client.post("/predict", json=valid_listing)
    assert response.status_code == 200
    body = response.json()
    assert body["room_type"] in ALLOWED_ROOMS
    assert set(body["probabilities"].keys()) == ALLOWED_ROOMS
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6


# Negative price is rejected by the input rules
def test_negative_price_rejected():
    bad_listing = dict(valid_listing, price=-10)
    response = client.post("/predict", json=bad_listing)
    assert response.status_code == 422


# A missing column is rejected
def test_missing_column_rejected():
    bad_listing = {key: value for key, value in valid_listing.items() if key != "neighbourhood"}
    response = client.post("/predict", json=bad_listing)
    assert response.status_code == 422


# A huge price gets capped to the saved cap (300.0 in the test caps),
# so it must give the same answer as a listing priced exactly at the cap
def test_price_cap_is_applied():
    huge = client.post("/predict", json=dict(valid_listing, price=1_000_000)).json()
    at_cap = client.post("/predict", json=dict(valid_listing, price=300.0)).json()
    assert huge == at_cap

# Metrics endpoint: Prometheus can read it and it contains our custom counter
def test_metrics_endpoint():
    client.post("/predict", json=valid_listing)
    response = client.get("/metrics/")
    assert response.status_code == 200
    assert "predictions_total" in response.text
