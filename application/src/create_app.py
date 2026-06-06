# create_app.py
# ---------------------------------------------------------------------------
# PURPOSE: FastAPI application factory.  Uses the "factory pattern" so that
#          the app is created on demand (useful for testing — each test can
#          spin up a fresh app instance).
#
# ENDPOINTS EXPOSED:
#   POST /api/predict/       — churn probability for a single customer
#   POST /simulate-drift     — artificially corrupt current data in MinIO
#   POST /detect-drift       — run Evidently, fire webhook if drift > 0.3
#   GET  /drift-report       — serve the latest drift HTML report
#   GET  /metrics            — Prometheus metrics scrape endpoint
#   GET  /health             — liveness probe (Kubernetes health check)
#   POST /reload-model       — hot-swap the in-memory model after retraining
#
# LEARNING NOTES:
#   - "Application factory pattern": `create_app()` returns the FastAPI
#     instance instead of creating it at module level.  This is the same
#     pattern used by Flask and Django for testability.  Uvicorn's --factory
#     flag supports this: `uvicorn application.src.create_app:create_app --factory`
#
#   - Prometheus metrics (prometheus_client):
#       · Gauge  — a single numerical value that can go up or down (drift score).
#       · Counter — monotonically increasing count (number of drift events).
#     The /metrics endpoint returns data in the Prometheus text format which
#     a Prometheus server can scrape and Grafana can visualise.
#
#   - Argo Events webhook: when drift is detected, FastAPI fires a raw HTTP
#     POST to the Argo Events EventSource Service.  The EventSource translates
#     it into an Argo event, and the Sensor triggers the WorkflowTemplate.
#     This decouples the application from the orchestration layer.
# ---------------------------------------------------------------------------

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel, Field
import boto3
import pandas as pd
from botocore.exceptions import NoCredentialsError
from fastapi.logger import logger
import logging
import requests

import os
import sys

# Ensure the project root is on PYTHONPATH so application.src.* imports work
# whether this is run locally (from repo root) or inside a Docker container.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from application.src.drift_detection import detect_drift_and_generate_report
from application.src.create_service import load_model
from application.src.predict import router as predict_router

from prometheus_client import Gauge, Counter, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------ #
# CONFIGURATION                                                        #
# ------------------------------------------------------------------ #
MINIO_URL             = os.getenv("MINIO_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
MODEL_BUCKET          = "models"
MODEL_KEY_PREFIX      = "best_model"
CURRENT_DATA_BUCKET   = "data"
CURRENT_DATA_KEY      = "app/data/processed/X_test.csv"

# Webhook URL used to notify Argo Events when drift is detected.
# In Kubernetes this resolves to the EventSource Service (port 12000).
WEBHOOK_URL = os.getenv(
    "WEBHOOK_URL",
    "http://argo-events-service.argo-events.svc.cluster.local:12000/drift"
)

# ------------------------------------------------------------------ #
# PROMETHEUS METRICS                                                   #
# ------------------------------------------------------------------ #
# These are module-level objects; prometheus_client registers them
# globally so the /metrics endpoint can collect them from any request.
data_drift_score = Gauge("data_drift_score", "Latest data drift score")
drift_detected   = Counter("drift_detected", "Number of times drift was detected")

# Module-level MinIO boto3 client reused across all request handlers.
minio_client = boto3.client(
    "s3",
    endpoint_url=MINIO_URL,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# ------------------------------------------------------------------ #
# REQUEST SCHEMA                                                       #
# ------------------------------------------------------------------ #
class DriftSimulationInput(BaseModel):
    drift_type: str = Field(
        ...,
        example="numerical_shift",
        description="Type of drift to simulate. Options: 'numerical_shift', 'category_mismatch', 'drop_column'",
    )


# ------------------------------------------------------------------ #
# HELPERS                                                              #
# ------------------------------------------------------------------ #
def fetch_latest_model_info():
    """
    List all objects in the `models` bucket under the `best_model` prefix
    and return the key of the most recently modified object.

    Returns:
        str: S3 key of the latest model file.
    """
    try:
        response = minio_client.list_objects_v2(Bucket=MODEL_BUCKET, Prefix=MODEL_KEY_PREFIX)
        if 'Contents' not in response:
            raise RuntimeError("No models found in MinIO bucket.")

        # Sort by LastModified and pick the most recent artefact.
        latest_model = max(response['Contents'], key=lambda obj: obj['LastModified'])
        return latest_model['Key']
    except NoCredentialsError:
        raise RuntimeError("Invalid MinIO credentials")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch latest model info: {e}")


def load_model_from_minio():
    """
    Orchestrate model loading: look up the latest key in MinIO, then
    delegate to create_service.load_model() which handles the actual
    download + joblib deserialisation.

    Returns:
        The loaded sklearn Pipeline object.
    """
    latest_model_key = fetch_latest_model_info()
    # Build a /tmp path for the downloaded file (container-safe temp dir).
    local_model_path = f"/tmp/{latest_model_key.split('/')[-1]}"

    try:
        minio_client.download_file(MODEL_BUCKET, latest_model_key, local_model_path)
        print(f"Loaded model from MinIO: {latest_model_key}")
        # load_model() in create_service.py reads from BEST_MODEL_PATH env var
        # (or models/best_model/model.pkl) — it handles the final joblib.load.
        return load_model()
    except Exception as e:
        raise RuntimeError(f"Failed to load model from MinIO: {e}")


# ------------------------------------------------------------------ #
# APPLICATION FACTORY                                                  #
# ------------------------------------------------------------------ #
def create_app() -> FastAPI:
    """
    Build and return the configured FastAPI application instance.

    Called by Uvicorn via: uvicorn application.src.create_app:create_app --factory
    """
    app = FastAPI(title="Churn Prediction API", version="1.0.0")

    # Load the model into app.state at startup so all request handlers can
    # access it via `request.app.state.model` without re-loading on every call.
    app.state.model = load_model_from_minio()

    # Mount the prediction router under /api so all its routes become
    # /api/predict/, etc.  The `tags` argument groups them in the Swagger UI.
    app.include_router(predict_router, prefix="/api", tags=["predictions"])

    # ------------------------------------------------------------------ #
    # ENDPOINT: POST /simulate-drift                                       #
    # ------------------------------------------------------------------ #
    @app.post("/simulate-drift")
    def simulate_drift(input: DriftSimulationInput):
        """
        Artificially inject distribution shift into the current dataset stored
        in MinIO.  Used to test the full automated retraining loop without
        waiting for real-world data drift.

        Drift Types:
        - numerical_shift:  Multiply MonthlyCharges by 1.5 (shifts the numeric distribution).
        - category_mismatch: Replace a known PaymentMethod value with an unknown one ("Crypto").
        - drop_column:       Remove the 'Contract' column entirely (schema drift).

        After calling this endpoint, hit POST /detect-drift to measure the drift
        and (if > 0.3) trigger the Argo retraining workflow automatically.
        """
        drift_type = input.drift_type
        try:
            # Download the current dataset that drift detection uses as its
            # "production" data snapshot.
            current_data_path = "/tmp/current_data.csv"
            minio_client.download_file(CURRENT_DATA_BUCKET, CURRENT_DATA_KEY, current_data_path)
            data = pd.read_csv(current_data_path)

            # Apply the requested drift mutation
            if drift_type == "numerical_shift":
                # Shift the MonthlyCharges distribution upward by 50 %
                data["MonthlyCharges"] *= 1.5
            elif drift_type == "category_mismatch":
                # Replace a valid category with one never seen during training
                data["PaymentMethod"] = data["PaymentMethod"].replace("Electronic check", "Crypto")
            elif drift_type == "drop_column":
                # Remove a column entirely to simulate schema drift
                if "Contract" in data.columns:
                    data.drop(columns=["Contract"], inplace=True)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown drift type: {drift_type}")

            # Write the mutated dataset back to MinIO, overwriting the original.
            # The next drift detection run will compare against this modified data.
            drifted_data_path = "/tmp/current_data.csv"
            data.to_csv(drifted_data_path, index=False)
            minio_client.upload_file(drifted_data_path, CURRENT_DATA_BUCKET, CURRENT_DATA_KEY)

            return {"status": "success", "message": "Drift simulated"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ------------------------------------------------------------------ #
    # ENDPOINT: POST /detect-drift                                         #
    # ------------------------------------------------------------------ #
    @app.post("/detect-drift")
    def detect_drift_endpoint():
        """
        Run Evidently drift detection, update Prometheus metrics, and
        conditionally fire the Argo Events webhook to trigger retraining.

        Called automatically by the CronJob in k8s/drift/drift-job.yaml
        every 10 minutes.  Can also be called manually for testing.
        """
        try:
            logger.info("Starting drift detection process.")

            # Core drift detection — downloads data from MinIO and runs Evidently
            score, html_report_path = detect_drift_and_generate_report()
            logger.info(f"Drift detection completed. Drift score: {score}. Report path: {html_report_path}")

            # Update the Prometheus Gauge so Grafana dashboards stay current
            data_drift_score.set(score)
            logger.info("Updated Prometheus metric: data_drift_score.")

            if score > 0.3:
                # Increment the drift event counter (monotonic — never resets)
                drift_detected.inc()
                logger.info("Drift detected. Incremented Prometheus metric: drift_detected.")

                # Fire webhook to Argo Events EventSource.
                # The EventSource (webhook-eventsource.yaml) listens on port
                # 12000 at /drift-detected and converts this POST into an Argo
                # event, which the Sensor (drift-detection-sensor.yaml) picks up
                # and uses to submit a WorkflowTemplate run.
                webhook_url = os.getenv(
                    "WEBHOOK_URL",
                    "http://drift-detection-eventsource-svc.argo-events.svc.cluster.local:12000/drift-detected"
                )
                response = requests.post(webhook_url, json={"drift_score": score})
                logger.info(f"Webhook sent to {webhook_url}. Response: {response.status_code} - {response.text}")

            return {
                "status": "success",
                "drift_score": score,
                "report_path": html_report_path
            }
        except Exception as e:
            logger.error(f"Error during drift detection: {str(e)}", exc_info=True)
            return {"status": "error", "message": str(e)}

    # ------------------------------------------------------------------ #
    # ENDPOINT: GET /drift-report                                          #
    # ------------------------------------------------------------------ #
    @app.get("/drift-report")
    def get_drift_report():
        """
        Serve the Evidently HTML report generated by the last /detect-drift
        call.  Open in a browser for an interactive visual breakdown of which
        features drifted and by how much.
        """
        html_report_path = "/tmp/data_drift_report.html"
        if os.path.exists(html_report_path):
            return FileResponse(html_report_path, media_type="text/html")
        return {"status": "error", "message": "Report not found"}

    # ------------------------------------------------------------------ #
    # ENDPOINT: GET /metrics                                               #
    # ------------------------------------------------------------------ #
    @app.get("/metrics")
    def metrics():
        """
        Expose prometheus_client metrics in the Prometheus text exposition
        format.  Configure a Prometheus scrape job to poll this endpoint and
        you can graph drift_score and drift_detected over time in Grafana.
        """
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # ------------------------------------------------------------------ #
    # ENDPOINT: GET /health                                                #
    # ------------------------------------------------------------------ #
    @app.get("/health")
    def health_check():
        """
        Liveness probe used by Kubernetes.  Returns HTTP 200 when the
        application is running.  Configure this under livenessProbe in the
        Deployment spec to enable automatic pod restarts on failure.
        """
        return {"status": "healthy"}

    # ------------------------------------------------------------------ #
    # ENDPOINT: POST /reload-model                                         #
    # ------------------------------------------------------------------ #
    @app.post("/reload-model")
    def reload_model():
        """
        Hot-swap the in-memory model by downloading the latest version from
        MinIO and replacing app.state.model.

        Called automatically by the Argo WorkflowTemplate's `notify-reload`
        step after a successful retrain cycle — no pod restart required.
        """
        try:
            app.state.model = load_model_from_minio()
            return {"status": "success", "message": "Model reloaded successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return app
