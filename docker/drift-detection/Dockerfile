# docker/drift-detection/Dockerfile
# ---------------------------------------------------------------------------
# PURPOSE: Standalone drift-detection image — an alternative deployment
#          strategy where drift detection runs as a dedicated container
#          rather than being called via the FastAPI /detect-drift endpoint.
#
# NOTE: In the current pipeline architecture, drift detection is triggered
#       by the Kubernetes CronJob (k8s/drift/drift-job.yaml) which calls
#       FastAPI's /detect-drift endpoint.  This Dockerfile exists as an
#       alternative approach if you want to run the Evidently logic
#       independently without FastAPI.
#
# KEY DESIGN DECISIONS:
#   - Uses continuumio/miniconda3 for reliable Evidently AI installation
#     (some Evidently dependencies have complex C extension requirements).
#   - All environment variables default to the in-cluster MinIO service DNS.
#     Override them via the Kubernetes Job's env: section at runtime.
# ---------------------------------------------------------------------------

FROM continuumio/miniconda3

RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    && apt-get clean

WORKDIR /app

# Copy the minimal requirements for drift detection only (pandas, evidently, boto3).
COPY requirements/requirements.drift.txt /app/requirements.txt

# Create an isolated Conda environment to avoid dependency conflicts.
RUN conda create -n drift-env python=3.9 -y && \
    conda run -n drift-env pip install --no-cache-dir -r /app/requirements.txt && \
    conda clean -a

# Make drift-env the active Python for all subsequent commands.
ENV PATH="/opt/conda/envs/drift-env/bin:$PATH"

# Copy only the drift detection script — keeps the image lean.
COPY scripts/drift_detection.py /app/scripts/

# ------------------------------------------------------------------ #
# RUNTIME CONFIGURATION                                                #
# ------------------------------------------------------------------ #
# These defaults point to in-cluster Kubernetes service DNS names.
# Override via the container's env: block in your Job/CronJob manifest.

ENV MINIO_URL=http://minio.minio.svc.cluster.local:9000

# MinIO default credentials (demo only — use Kubernetes Secrets in production).
ENV AWS_ACCESS_KEY_ID=minioadmin
ENV AWS_SECRET_ACCESS_KEY=minioadmin

# MinIO bucket containing the training reference snapshot.
ENV REFERENCE_DATA_BUCKET=models

# MinIO bucket containing the current production data snapshot.
ENV CURRENT_DATA_BUCKET=data

# Argo Events EventSource webhook URL — called when drift is detected.
ENV WEBHOOK_URL=http://argo-events-service.argo-events.svc.cluster.local:12000/drift

# Run the drift detection script.  It compares reference vs. current data,
# generates a report, and fires the webhook if drift exceeds the threshold.
CMD ["python", "/app/scripts/drift_detection.py"]
