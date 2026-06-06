# docker/mlserver/Dockerfile
# ---------------------------------------------------------------------------
# PURPOSE: Builds the MLServer + MLflow tracking server image.
#          This single container serves two roles:
#            1. MLServer (port 8080): Seldon's model-serving runtime that
#               exposes a V2 Inference Protocol REST API for sklearn models.
#            2. MLflow UI (port 5000): The experiment tracking web interface
#               where you can compare runs, view metrics, and inspect artefacts.
#          Pushed to Docker Hub as measureapp/mlserver:<version>.
#          Deployed via k8s/mlserver/mlserver.yaml.
#
# BUILD COMMAND (from the docker/mlserver/ directory):
#   docker build -t mlserver -f docker/mlserver/Dockerfile .
#   # Or via Makefile:
#   make build-mlserver
#
# ⚠️  SECURITY WARNING:
#   AWS credentials are baked in as ENV defaults for demo convenience.
#   In production NEVER put secrets in a Dockerfile.  Use Kubernetes
#   Secrets and mount them as environment variables instead.
# ---------------------------------------------------------------------------

# python:3.11-slim — lightweight image; MLServer requires Python 3.8+.
FROM python:3.11-slim

# Install all required Python packages in a single RUN layer to minimise
# image size.  mlserver-sklearn is the runtime plugin that makes MLServer
# understand sklearn Pipeline model artefacts.
RUN pip install --no-cache-dir \
    mlserver \
    mlserver-sklearn \
    boto3 \
    mlflow

WORKDIR /app

# ⚠️  These ENV values are overridden by the Kubernetes Deployment's env: section.
# They are provided here only so the image works in a standalone `docker run`.
ENV MODEL_NAME=best-model          # Logical name MLServer uses for the model
ENV MODEL_URI=/app/models/best_model  # Path where MLServer looks for model files

# ⚠️  DEMO ONLY — replace with Kubernetes Secrets in production.
ENV AWS_ACCESS_KEY_ID=qgJC73MaeLiIsNx1hBWE
ENV AWS_SECRET_ACCESS_KEY=rP8zK6AUFvB5Y0LtxUQEPmgSGFHXT2shgCkap1Y0

# Expose the two service ports:
#   8080 → MLServer model API (V2 inference protocol)
#   5000 → MLflow tracking UI
EXPOSE 8080 5000

# Copy and make the startup script executable.
# entrypoint.sh launches both MLServer and the MLflow UI in the same container.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# The entrypoint script starts MLServer in the background and the MLflow UI
# in the foreground.  This co-location is fine for local/demo use; in
# production you would run them as separate deployments.
ENTRYPOINT ["/entrypoint.sh"]
