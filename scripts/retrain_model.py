# retrain_model.py
# ---------------------------------------------------------------------------
# PURPOSE: Automated retraining worker — this script is the heart of the
#          self-healing feedback loop.  It is executed inside a Kubernetes
#          Job/WorkflowTemplate container (see k8s/argo/workflow-template.yaml
#          and k8s/model/model-train-job.yaml).
#
# FLOW:
#   1. Download processed datasets from MinIO (data bucket).
#   2. Ensure reference_data.csv is available (download from models bucket or
#      fall back to a local copy from a previous training run).
#   3. Run Evidently AI drift detection between reference and current data.
#   4. If drift_share > 0.3, trigger a full retrain cycle:
#        a. Call train_and_save_model() from model_training.py.
#        b. Upload the new model.pkl to the `models` MinIO bucket.
#        c. Upload updated reference_data.csv and processed splits.
#   5. The Argo Workflow's second step (notify-reload) will then POST to
#      FastAPI's /reload-model endpoint, hot-swapping the in-memory model
#      without restarting the pod.
#
# LEARNING NOTES:
#   - The `drift_share` metric is the fraction of features where Evidently
#     detects a statistically significant distribution shift.  A value of 0
#     means no drift; 1.0 means every feature drifted.
#
#   - The 0.3 threshold is deliberately low for this demo so you can trigger
#     retraining quickly.  In production you might use 0.1–0.2 and combine
#     it with model performance degradation signals.
#
#   - By re-uploading the trained model to the same MinIO key
#     (best_model/model.pkl), the FastAPI pod can reload it without any
#     configuration change — "convention over configuration" at work.
#
#   - This script is triggered by Argo Events, not a human.  The event chain
#     is: CronJob (drift-job.yaml) → FastAPI /detect-drift → Evidently →
#     webhook POST to Argo Events EventSource → Sensor → WorkflowTemplate.
# ---------------------------------------------------------------------------

import os
import sys
import pandas as pd
import boto3
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from botocore.exceptions import NoCredentialsError

# Add the project root directory to the PYTHONPATH so that local imports work
# both when running directly (python scripts/retrain_model.py) and inside the
# Docker container (/app/scripts/retrain_model.py).
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from scripts.model_training import train_and_save_model, upload_model_to_minio, upload_reference_data_to_minio

# ------------------------------------------------------------------ #
# MODULE-LEVEL S3/MINIO CLIENT                                         #
# ------------------------------------------------------------------ #
# Initialised once at import time so all helper functions reuse the
# same connection pool.  Credentials are read from environment variables
# which are injected by Kubernetes (see workflow-template.yaml env: section).
S3_CLIENT = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_URL", "http://localhost:9000"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
)


def download_from_s3(bucket, key, local_path):
    """
    Download a single file from a MinIO bucket to a local path.

    Args:
        bucket     (str): Source MinIO bucket name.
        key        (str): Object key (path inside the bucket).
        local_path (str): Destination path on the local filesystem.
    """
    # Ensure the local directory exists before attempting the download
    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)

    try:
        S3_CLIENT.download_file(bucket, key, local_path)
        print(f"Downloaded {key} from bucket {bucket} to {local_path}")
    except NoCredentialsError:
        raise RuntimeError("Invalid MinIO/S3 credentials")
    except Exception as e:
        raise RuntimeError(f"Failed to download {key} from bucket {bucket}: {e}")


def upload_to_s3(local_path, bucket, key):
    """
    Upload a single local file to a MinIO bucket.

    Args:
        local_path (str): Source file path on the local filesystem.
        bucket     (str): Destination MinIO bucket name.
        key        (str): Object key (path) the file will be stored under.
    """
    try:
        S3_CLIENT.upload_file(local_path, bucket, key)
        print(f"Uploaded {local_path} to bucket {bucket} with key {key}")
    except Exception as e:
        raise RuntimeError(f"Failed to upload {local_path} to bucket {bucket}: {e}")


def assess_drift_for_retraining(reference_data_path, current_data_path):
    """
    Run Evidently drift detection between the reference (training) distribution
    and the current (production) data distribution.

    Args:
        reference_data_path (str): Path to reference_data.csv (baseline / training split).
        current_data_path   (str): Path to the current dataset (e.g. X_test.csv or live data).

    Returns:
        bool: True if drift exceeds the retraining threshold, False otherwise.
    """
    reference_data = pd.read_csv(reference_data_path)
    current_data   = pd.read_csv(current_data_path)

    # Build and run the drift report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    # report.as_dict() returns the full results tree; we extract the
    # aggregated `drift_share` (0–1) from the first metric entry.
    drift_score = report.as_dict()['metrics'][0]['result']['drift_share']
    print(f"Drift Share: {drift_score}")

    # Save the HTML report for human inspection / audit trail
    drift_report_path = os.path.join("models/evaluation_reports", "retraining_drift_report.html")
    os.makedirs(os.path.dirname(drift_report_path), exist_ok=True)
    report.save_html(drift_report_path)

    # Decision threshold — lowered to 0.3 for demo purposes so that the
    # simulated drift (via FastAPI /simulate-drift) triggers retraining.
    # In production, consider 0.1–0.15 combined with AUC degradation.
    if drift_score > 0.3:
        print("Significant dataset drift detected. Proceeding with retraining.")
        return True
    else:
        print("No significant drift detected. Skipping retraining.")
        return False


def train_and_save_model_with_upload(data_dir, best_model_dir, model_bucket, data_bucket):
    """
    Orchestrates the full retrain → persist → upload cycle.

    After training completes locally:
      - The new model.pkl is pushed to the `models` bucket under the same key
        (best_model/model.pkl), overwriting the previous version.
      - The updated reference_data.csv and all processed splits are also
        re-uploaded so downstream jobs have fresh baseline data.

    Args:
        data_dir       (str): Local directory with processed splits.
        best_model_dir (str): Local directory where model.pkl is saved.
        model_bucket   (str): MinIO bucket for model artefacts ("models").
        data_bucket    (str): MinIO bucket for processed data ("data").
    """
    os.makedirs(best_model_dir, exist_ok=True)

    # Retrain the model using the downloaded processed data
    train_and_save_model(data_dir, best_model_dir)

    # Overwrite the old model in MinIO so FastAPI's /reload-model picks it up
    model_uri = upload_model_to_minio(best_model_dir, model_bucket, "best_model")
    print(f"Model uploaded to: {model_uri}")

    # Update the reference snapshot in the models bucket
    reference_data_path = os.path.join(data_dir, "reference_data.csv")
    upload_to_s3(reference_data_path, model_bucket, "reference_data.csv")
    print(f"Reference data uploaded to MinIO: s3://{model_bucket}/reference_data.csv")

    # Mirror all processed splits back to the data bucket under the canonical
    # path `app/data/processed/` so future retrain jobs can find them.
    for filename in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv", "X_val.csv", "y_val.csv"]:
        local_path = os.path.join(data_dir, filename)
        upload_to_s3(local_path, data_bucket, f"app/data/processed/{filename}")
        print(f"Processed data {filename} uploaded to MinIO: s3://{data_bucket}/app/data/processed/{filename}")


def ensure_local_directories():
    """
    Create required local directories if they do not yet exist.
    Called at container startup to avoid FileNotFoundError during downloads.
    """
    data_processed_dir = "data/processed"
    models_dir         = "models/best_model"

    os.makedirs(data_processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print(f"Directories ensured: {data_processed_dir}, {models_dir}")


def prepare_reference_data(data_dir, model_bucket):
    """
    Guarantee reference_data.csv is available locally before drift assessment.

    Priority:
      1. Already exists locally → use it.
      2. Download from MinIO models bucket.
      3. Fallback: write a placeholder (signals a setup issue — investigate).

    Args:
        data_dir     (str): Local directory to place reference_data.csv.
        model_bucket (str): MinIO bucket to attempt download from.
    Returns:
        str: Absolute path to reference_data.csv.
    """
    reference_data_path = os.path.join(data_dir, "reference_data.csv")

    if os.path.exists(reference_data_path):
        print(f"Reference data already exists locally at {reference_data_path}")
        return reference_data_path

    try:
        download_from_s3(model_bucket, "reference_data.csv", reference_data_path)
        print(f"Reference data downloaded from MinIO to {reference_data_path}")
    except Exception as e:
        # Fallback: generate a minimal placeholder so the script doesn't crash.
        # This situation should not occur in a healthy pipeline; investigate why
        # data_preparation.py didn't upload reference_data.csv to MinIO.
        print(f"Failed to download reference data: {e}")
        print("Generating reference_data.csv locally as fallback...")
        pd.DataFrame({"example_column": [1, 2, 3]}).to_csv(reference_data_path, index=False)

    return reference_data_path


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # ENTRY POINT — executed by the Argo WorkflowTemplate container        #
    # ------------------------------------------------------------------ #

    # Step 0: create required local directories in the container filesystem
    ensure_local_directories()

    DATA_DIR       = "data/processed"
    BEST_MODEL_DIR = "models/best_model"
    MODEL_BUCKET   = "models"
    DATA_BUCKET    = "data"

    # Step 1: ensure reference_data.csv is available (baseline for drift check)
    reference_data_path = prepare_reference_data(DATA_DIR, MODEL_BUCKET)

    # Step 2: download all processed splits from MinIO so the training script
    # can read them from a known local path
    datasets = [
        ("app/data/processed/X_train.csv", "X_train.csv"),
        ("app/data/processed/X_test.csv",  "X_test.csv"),
        ("app/data/processed/y_train.csv", "y_train.csv"),
        ("app/data/processed/y_test.csv",  "y_test.csv"),
        ("app/data/processed/X_val.csv",   "X_val.csv"),
        ("app/data/processed/y_val.csv",   "y_val.csv"),
    ]
    for remote_path, local_filename in datasets:
        download_from_s3(DATA_BUCKET, remote_path, os.path.join(DATA_DIR, local_filename))

    # Step 3: use X_test as the "current" data to compare against the reference.
    # In production this would be replaced by data collected from real API calls.
    current_data_local = os.path.join(DATA_DIR, "X_test.csv")

    # Step 4: assess drift and retrain only if the threshold is exceeded
    if assess_drift_for_retraining(reference_data_path, current_data_local):
        print("Starting retraining process...")
        train_and_save_model_with_upload(DATA_DIR, BEST_MODEL_DIR, MODEL_BUCKET, DATA_BUCKET)
        print("Retraining completed successfully.")
    else:
        print("Retraining skipped due to no significant drift.")
