# drift_detection.py
# ---------------------------------------------------------------------------
# PURPOSE: Core drift-detection logic used by the FastAPI /detect-drift
#          endpoint.  It downloads the reference and current datasets from
#          MinIO, runs Evidently AI's DataDriftPreset, and returns the
#          drift_share score plus the path to the saved HTML report.
#
# HOW IT FITS IN THE PIPELINE:
#   - A Kubernetes CronJob (k8s/drift/drift-job.yaml) calls POST /detect-drift
#     every 10 minutes.
#   - detect_drift_and_generate_report() is invoked.
#   - If drift_score > 0.3, create_app.py fires a webhook POST to the Argo
#     Events EventSource, which triggers the retrain WorkflowTemplate.
#
# DATA SOURCES (MinIO):
#   - Reference data : models/reference_data.csv   (training-set snapshot)
#   - Current data   : data/app/data/processed/X_test.csv
#     NOTE: In production, replace X_test.csv with data collected from live
#           API calls stored to MinIO by the predict endpoint.
#
# LEARNING NOTES:
#   - /tmp is writable inside any container without volume mounts — good for
#     ephemeral artefacts like downloaded CSVs and report HTML files.
#   - Schema validation (checking for missing columns) is critical: if the
#     drift-simulation endpoint drops a column, Evidently would crash without
#     a clear error message.  This guard surfaces the issue early.
#   - Evidently's DataDriftPreset selects the appropriate statistical test per
#     column type automatically (Wasserstein for numerics, chi-square for cats).
# ---------------------------------------------------------------------------

import pandas as pd
import requests
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# ------------------------------------------------------------------ #
# CONFIGURATION — read from environment variables                      #
# ------------------------------------------------------------------ #
# All defaults assume a locally-running MinIO on the default port.
# In Kubernetes these are overridden by the pod's env: section.
MINIO_URL             = os.getenv("MINIO_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

# Reference data: the training snapshot uploaded by model_training.py
REFERENCE_DATA_BUCKET = "models"
REFERENCE_DATA_KEY    = "reference_data.csv"

# Current data: latest production data (X_test used as a proxy here).
# In a real system this would be replaced with inference-request logs.
CURRENT_DATA_BUCKET = "data"
CURRENT_DATA_KEY    = "app/data/processed/X_test.csv"

# Webhook URL for Argo Events — POSTed to when drift is detected.
# In Kubernetes this resolves to the EventSource Service inside the cluster.
webhook_url = os.getenv(
    "WEBHOOK_URL",
    "http://drift-detection-eventsource-svc.argo-events.svc.cluster.local:12000/drift-detected"
)


def detect_drift_and_generate_report():
    """
    Full drift-detection pipeline:
      1. Download reference and current datasets from MinIO.
      2. Align dtypes and validate schema consistency.
      3. Run Evidently DataDriftPreset.
      4. Save the HTML report to /tmp.
      5. Return (drift_score, html_report_path).

    Returns:
        tuple: (drift_score: float, html_report_path: str)
    Raises:
        ValueError: if the current dataset is missing columns present in reference.
        RuntimeError: if MinIO download fails.
    """
    try:
        # ------------------------------------------------------------------ #
        # 1. DOWNLOAD DATASETS FROM MINIO                                     #
        # ------------------------------------------------------------------ #
        # Files are written to /tmp (always writable in containers) and then
        # read into DataFrames.  download_from_minio returns the DataFrame
        # directly after downloading.
        reference_data_path = "/tmp/reference_data.csv"
        reference_data = download_from_minio(REFERENCE_DATA_BUCKET, REFERENCE_DATA_KEY, reference_data_path)

        current_data_path = "/tmp/current_data.csv"
        current_data = download_from_minio(CURRENT_DATA_BUCKET, CURRENT_DATA_KEY, current_data_path)

        # ------------------------------------------------------------------ #
        # 2. ALIGN DTYPES                                                     #
        # ------------------------------------------------------------------ #
        # After CSV round-trips, column dtypes may differ (e.g. an int column
        # may become float).  Evidently requires matching dtypes to apply
        # the correct statistical test per column.
        for col in reference_data.columns:
            if col in current_data:
                expected_dtype = reference_data[col].dtype
                current_data[col] = current_data[col].astype(expected_dtype)

        # ------------------------------------------------------------------ #
        # 3. VALIDATE SCHEMA                                                  #
        # ------------------------------------------------------------------ #
        # If /simulate-drift dropped a column (drop_column type), Evidently
        # would fail silently or crash.  Raise early with a descriptive error.
        missing_columns = set(reference_data.columns) - set(current_data.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in input data: {missing_columns}")

        # ------------------------------------------------------------------ #
        # 4. RUN EVIDENTLY DRIFT DETECTION                                    #
        # ------------------------------------------------------------------ #
        # DataDriftPreset computes per-feature drift p-values and aggregates
        # them into a single drift_share metric (fraction of drifted features).
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data)

        # ------------------------------------------------------------------ #
        # 5. PERSIST HTML REPORT                                              #
        # ------------------------------------------------------------------ #
        # The /drift-report FastAPI endpoint serves this file back to the user.
        html_report_path = "/tmp/data_drift_report.html"
        report.save_html(html_report_path)
        print(f"Drift report saved to {html_report_path}")

        # Extract the scalar drift_share from the nested results dict.
        # metrics[0] is DataDriftPreset; result.drift_share is 0–1.
        drift_metrics = report.as_dict()
        drift_score = drift_metrics["metrics"][0]["result"]["drift_share"]
        print(f"Drift Score: {drift_score}")

        return drift_score, html_report_path

    except Exception as e:
        print(f"Error during drift detection: {e}")
        raise


def download_from_minio(bucket_name, object_name, local_path):
    """
    Download a file from MinIO and return it as a pandas DataFrame.

    Args:
        bucket_name (str): MinIO bucket to download from.
        object_name (str): Object key (path inside the bucket).
        local_path  (str): Local path where the file will be saved.

    Returns:
        pd.DataFrame: The downloaded CSV file as a DataFrame.
    """
    # Lazy imports keep module-level startup fast and avoid circular imports.
    import boto3
    from botocore.exceptions import NoCredentialsError

    s3_client = boto3.client(
        "s3",
        endpoint_url=MINIO_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    try:
        s3_client.download_file(bucket_name, object_name, local_path)
        print(f"Downloaded {object_name} from MinIO bucket {bucket_name} to {local_path}")
        # Read and return immediately — caller gets a DataFrame, not a file path.
        return pd.read_csv(local_path)
    except NoCredentialsError:
        raise RuntimeError("Invalid MinIO credentials")
    except Exception as e:
        raise RuntimeError(f"Failed to download {object_name} from MinIO: {e}")

