# docker/fastapi/Dockerfile
# ---------------------------------------------------------------------------
# PURPOSE: Builds the image for the Churn Prediction FastAPI service.
#          Pushed to Docker Hub as measureapp/demo_ai_api:<version>.
#          Deployed to Kubernetes via k8s/api/fastapi-depl.yaml.
#
# BUILD COMMAND (from repo root):
#   docker build -t fastapi-service -f docker/fastapi/Dockerfile .
#   # Or via Makefile:
#   make build-fastapi
#
# KEY DESIGN DECISIONS:
#   - python:3.9-slim keeps the image small (~130 MB vs ~900 MB for full Python).
#   - Model is NOT baked into the image; it is downloaded from MinIO at
#     startup.  This means the same image works with any model version — you
#     only need to push a new Docker image when application code changes.
#   - --no-cache-dir reduces image size by not storing pip's download cache.
# ---------------------------------------------------------------------------

# Use an official Python slim image to minimise the attack surface and image size.
FROM python:3.9-slim

# Install system packages required by some Python dependencies.
# curl is used in liveness/readiness checks; libssl-dev is needed by cryptography packages.
# `apt-get clean` removes cached package files to keep the layer small.
RUN apt-get update && apt-get install -y \
    curl \
    libssl-dev \
    && apt-get clean

# All subsequent commands run from /app inside the container.
WORKDIR /app

# Copy only the requirements file first.  Docker caches layers; if requirements.txt
# hasn't changed, the expensive `pip install` step is skipped on subsequent builds.
COPY requirements/requirements.app.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the application source code.
# The model (models/best_model/model.pkl) is intentionally excluded — it is
# downloaded from MinIO at runtime by create_service.py, so model updates
# don't require rebuilding or redeploying the container image.
COPY application /app/application
# COPY models/best_model /app/models/best_model  ← only enable for offline testing

# Inform Docker (and humans) which port this service listens on.
# The actual binding is done by Uvicorn via the CMD below.
EXPOSE 8000

# Start Uvicorn with the application factory.
# --factory tells Uvicorn to call create_app() to obtain the app instance.
# --host 0.0.0.0 makes the server reachable from outside the container.
CMD ["uvicorn", "application.src.create_app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
