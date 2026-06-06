# docker/model/Dockerfile
# ---------------------------------------------------------------------------
# PURPOSE: Builds the container image that executes the full ML pipeline:
#          data preparation → model training → model evaluation.
#          Pushed to Docker Hub as measureapp/ml_job:<version>.
#          Used by both:
#            - k8s/model/model-train-job.yaml  (initial one-off training Job)
#            - k8s/argo/workflow-template.yaml  (automated retrain via Argo)
#
# BUILD COMMAND (from repo root):
#   docker build -t retrain-job -f docker/model/Dockerfile .
#   # Or via Makefile:
#   make build-retrain
#
# KEY DESIGN DECISIONS:
#   - Uses continuumio/miniconda3 as the base because some ML packages
#     (XGBoost, scikit-learn) benefit from Conda's pre-compiled binaries
#     and avoid C/C++ compile-time dependencies.
#   - A dedicated Conda environment (train-env) isolates dependencies and
#     avoids polluting the base image's Python installation.
#   - Raw data is baked into the image so the training Job is fully
#     self-contained.  For large datasets, swap this for a download step.
#   - MLflow and MinIO URLs are provided as build-time ENV defaults but
#     are always overridden by the Kubernetes Job's env: section at runtime.
# ---------------------------------------------------------------------------

# continuumio/miniconda3 provides the `conda` command pre-installed.
FROM continuumio/miniconda3

# Build tools required by some Python C-extension packages (e.g. numpy, XGBoost).
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    && apt-get clean

WORKDIR /app

# Copy the training requirements file separately from the rest of the code
# so Docker can cache the pip install layer and skip it on code-only changes.
COPY requirements/requirements.retrain.txt /app/requirements.txt

# Create a clean Conda environment (python 3.9) and install all dependencies.
# `conda clean -a` removes package caches to reduce final image size.
RUN conda create -n train-env python=3.9 -y && \
    conda run -n train-env pip install --no-cache-dir -r /app/requirements.txt && \
    conda clean -a

# Add the Conda environment's bin directory to PATH so all subsequent
# RUN / CMD / ENTRYPOINT instructions use the correct Python/pip.
ENV PATH="/opt/conda/envs/train-env/bin:$PATH"

# Copy the training scripts, YAML config, and raw data into the image.
# data/raw/telco_churn.csv (~1 MB) is baked in so the Job is self-contained.
COPY scripts/  /app/scripts/
COPY config/   /app/config/
COPY data/raw/ /app/data/raw/

# Default MLflow tracking URI — points to the mlserver Service in Kubernetes.
# Override at runtime via the Job's env: section (MLFLOW_TRACKING_URI).
ENV MLFLOW_TRACKING_URI=http://mlserver-service.mlserver.svc.cluster.local:5000

# Default MinIO URL — points to the minio Service in Kubernetes.
# Override at runtime via the Job's env: section (MINIO_URL).
ENV MINIO_URL=http://minio.minio.svc.cluster.local:9000

# Default command: run the three-stage pipeline in sequence.
# Each stage must succeed before the next starts (shell && chaining).
# For retraining only, the Argo WorkflowTemplate overrides this with:
#   python /app/scripts/retrain_model.py
CMD ["bash", "-c", \
     "python /app/scripts/run_data_preparation.py \
        --data-path /app/data/raw/telco_churn.csv \
        --config-path /app/config/process.yaml \
        --output-dir /app/data/processed && \
      python /app/scripts/model_training.py && \
      python /app/scripts/evaluate_model.py"]
