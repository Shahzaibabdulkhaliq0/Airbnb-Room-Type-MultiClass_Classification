FROM python:3.14.10

# All commands below run inside /app in the container
WORKDIR /app

# Install the API packages first, so Docker can cache this layer
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# The API code
COPY api/ api/

# The whole src folder: the logger (src/logger.py or src/logger/ package) used by the API
COPY src/ src/

# The final model files
COPY models/best_model.joblib models/caps.json models/

# The API listens on port 8000
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]