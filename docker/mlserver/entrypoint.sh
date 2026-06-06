#!/bin/bash
# entrypoint.sh
# ---------------------------------------------------------------------------
# PURPOSE: Start both MLServer and the MLflow UI inside the same container.
#
# WHY TWO PROCESSES?
#   MLServer serves the model via the V2 Inference Protocol (port 8080).
#   MLflow UI lets you inspect experiment runs and artefacts (port 5000).
#   Co-locating them in one pod keeps the Kubernetes setup simple for demos.
#   In production, these would be separate Deployments.
#
# PROCESS MANAGEMENT:
#   `mlserver start . &` launches MLServer in the background (&).
#   `mlflow ui ...` runs in the foreground.  When the foreground process
#   ends, the container exits (and Kubernetes restarts it if needed).
# ---------------------------------------------------------------------------

# Start MLServer in the background — reads model configuration from
# the current directory (.).  The model artefacts and settings file
# (model-settings.json) must exist at /app/models/best_model/.
mlserver start . &

# Start the MLflow tracking UI in the foreground.
# --host 0.0.0.0 makes it reachable from outside the container.
# --port 5000 matches the Service definition in k8s/mlserver/mlserver.yaml.
mlflow ui --host 0.0.0.0 --port 5000
