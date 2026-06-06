# run_data_preparation.py
# ---------------------------------------------------------------------------
# PURPOSE: CLI entry point that wires together command-line arguments and the
#          preprocess_data() function defined in data_preparation.py.
#
# USAGE (local):
#   python scripts/run_data_preparation.py \
#       --data-path   data/raw/telco_churn.csv \
#       --config-path config/process.yaml \
#       --output-dir  data/processed
#
#   Or via the Makefile shortcut:
#       make run-data-preparation
#
# USAGE (inside Kubernetes Job — see k8s/model/model-train-job.yaml):
#   python /app/scripts/run_data_preparation.py \
#       --data-path /app/data/raw/telco_churn.csv \
#       --config-path /app/config/process.yaml \
#       --output-dir /app/data/processed
#
# LEARNING NOTE:
#   Separating the CLI wrapper (this file) from the library function
#   (data_preparation.py) follows the "single responsibility" principle:
#   the library stays importable and testable without side-effects, while
#   this script handles all argument parsing and top-level orchestration.
# ---------------------------------------------------------------------------

import argparse
import os
import sys

# Ensure the project root is on the Python path so that
# `from scripts.data_preparation import ...` resolves correctly regardless
# of the current working directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from scripts.data_preparation import preprocess_data

def main():
    # ------------------------------------------------------------------ #
    # ARGUMENT PARSING                                                     #
    # ------------------------------------------------------------------ #
    parser = argparse.ArgumentParser(description="Run data preparation.")

    # Required: path to the raw input CSV.
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the raw data CSV file."
    )

    # Optional: YAML config that lists feature columns and split ratios.
    # Defaulting to the repo's checked-in config keeps things reproducible.
    parser.add_argument(
        "--config-path",
        type=str,
        default="config/process.yaml",
        help="Path to the YAML configuration file."
    )

    # Optional: where to write processed CSVs.  This directory is also
    # mirrored into MinIO, so keep it consistent with the training script.
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed data."
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # RUN THE PREPROCESSING PIPELINE                                       #
    # ------------------------------------------------------------------ #
    # preprocess_data() returns a dict of DataFrames; we only use it here
    # for reporting — the real artefacts are the CSVs written to disk and
    # the files uploaded to MinIO.
    datasets = preprocess_data(args.data_path, args.config_path, args.output_dir)

    # ------------------------------------------------------------------ #
    # SUMMARY LOG                                                          #
    # ------------------------------------------------------------------ #
    print("Data preparation completed successfully.")
    print(f"Processed data saved in: {args.output_dir}")
    print(f"Training set shape:   {datasets['X_train'].shape}")
    print(f"Validation set shape: {datasets['X_val'].shape}")
    print(f"Test set shape:       {datasets['X_test'].shape}")

if __name__ == "__main__":
    main()

