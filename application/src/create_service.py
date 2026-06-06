# create_service.py
# ---------------------------------------------------------------------------
# PURPOSE: Utility module responsible for loading the trained sklearn Pipeline
#          into memory so the FastAPI application can serve predictions.
#
# FLOW:
#   1. On FastAPI startup (create_app.py), load_model() is called.
#   2. load_model() checks whether model.pkl already exists locally.
#        - If yes → load it directly with joblib.
#        - If no  → download it from MinIO first, then load.
#   3. The loaded pipeline is stored in app.state.model so every request
#      handler can access it without re-loading it on every call.
#   4. When /reload-model is called (after retraining), the same load_model()
#      function is invoked again to hot-swap the in-memory pipeline with the
#      newly uploaded version — zero-downtime model update.
#
# LEARNING NOTE:
#   Storing the model in `app.state` (FastAPI's built-in application state
#   object) is the idiomatic way to share heavy resources (models, DB
#   connections) across request handlers without using globals.
# ---------------------------------------------------------------------------

import os
import joblib
import boto3
from botocore.exceptions import NoCredentialsError

# The model path is configurable via an environment variable, making it easy
# to mount a Kubernetes volume and override the default local path.
# In Kubernetes (fastapi-depl.yaml) this is set to /app/models/best_model/model.pkl.
BEST_MODEL_PATH = os.getenv("BEST_MODEL_PATH", "models/best_model/model.pkl")


def download_model_from_minio():
    """
    Pull model.pkl from the MinIO `models` bucket and save it to BEST_MODEL_PATH.

    Called automatically by load_model() when the file is not found locally —
    useful for cold starts (first pod launch or after a node restart).
    """
    # In Kubernetes, MINIO_URL points to the in-cluster Service DNS name.
    # Locally it defaults to localhost:9000 (where MinIO is port-forwarded).
    minio_url = os.getenv("MINIO_URL", "http://minio-service.minio.svc.cluster.local:9000")
    minio_access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    minio_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    bucket_name = "models"
    model_key   = "best_model/model.pkl"   # canonical key used by all training jobs

    s3_client = boto3.client(
        "s3",
        endpoint_url=minio_url,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
    )

    try:
        # Create parent directory if it doesn't exist (e.g. /app/models/best_model/)
        os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
        s3_client.download_file(bucket_name, model_key, BEST_MODEL_PATH)
        print(f"Model downloaded successfully from MinIO to {BEST_MODEL_PATH}")
    except NoCredentialsError:
        raise RuntimeError("Invalid MinIO credentials")
    except Exception as e:
        raise RuntimeError(f"Failed to download model from MinIO: {e}")


def load_model():
    """
    Load the sklearn Pipeline from disk (downloading from MinIO if needed).

    Returns:
        The deserialised sklearn Pipeline object (preprocessor + classifier).
    Raises:
        RuntimeError: if the model cannot be found or loaded.
    """
    # Check for a local copy first — avoids a network round-trip on warm restarts
    # and after the model has already been downloaded to the pod's filesystem.
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Model not found locally at {BEST_MODEL_PATH}. Attempting to download from MinIO.")
        download_model_from_minio()

    try:
        # joblib.load reads the serialised sklearn Pipeline including all fitted
        # transformers and the classifier, making it immediately usable.
        model = joblib.load(BEST_MODEL_PATH)
        print(f"Model loaded successfully from {BEST_MODEL_PATH}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load the model. Reason: {e}")
