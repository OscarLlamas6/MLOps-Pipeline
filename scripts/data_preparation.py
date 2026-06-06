# data_preparation.py
# ---------------------------------------------------------------------------
# PURPOSE: Ingests the raw Telco Churn CSV, cleans and engineers features,
#          splits the dataset into train / validation / test sets, persists
#          every split locally, and uploads them (plus a "reference" snapshot
#          used later by the drift-detection step) to MinIO.
#
# LEARNING NOTES:
#   - MinIO is an S3-compatible object-storage server that we run locally
#     (or in Kubernetes).  boto3 — Amazon's Python SDK — speaks to it via
#     its S3-compatible API, so no MinIO-specific client is needed.
#   - "Reference data" is the training-set snapshot that Evidently AI will
#     compare future/production data against to detect statistical drift.
#   - The split strategy is stratified: each split keeps the same class ratio
#     as the original dataset, preventing skewed evaluation sets.
# ---------------------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml
import boto3


def upload_file_to_minio(local_file_path, bucket_name, object_name):
    """
    Upload a local file to a MinIO bucket using the S3-compatible API.

    Args:
        local_file_path (str): Path on the local filesystem to the file to upload.
        bucket_name     (str): Target bucket name inside MinIO (created if missing).
        object_name     (str): Key (path) the file will have inside the bucket.
    """
    # Read MinIO connection details from environment variables.
    # Defaults work for a locally-running MinIO with its default credentials.
    # In Kubernetes these are injected via the Deployment's `env:` section.
    minio_url = os.getenv("MINIO_URL", "http://localhost:9000")
    minio_access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    minio_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

    # boto3.client with a custom `endpoint_url` redirects all S3 calls to MinIO
    # instead of AWS.  The API surface is identical.
    s3_client = boto3.client(
        "s3",
        endpoint_url=minio_url,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
    )
    try:
        # Check whether the bucket already exists; create it if not.
        # head_bucket raises a ClientError (404) when the bucket is absent.
        try:
            s3_client.head_bucket(Bucket=bucket_name)
        except s3_client.exceptions.ClientError:
            print(f"Bucket {bucket_name} does not exist. Creating it...")
            s3_client.create_bucket(Bucket=bucket_name)

        # Upload file — boto3 handles chunking for large files automatically.
        print(f"Uploading {local_file_path} to MinIO bucket {bucket_name} with key {object_name}...")
        s3_client.upload_file(local_file_path, bucket_name, object_name)
        print(f"Upload successful: {object_name}")
    except Exception as e:
        raise RuntimeError(f"Failed to upload file to MinIO: {e}")


def preprocess_data(data_path, config_path, output_dir="data/processed", bucket_name="data"):
    """
    Full data-preparation pipeline: load → clean → engineer → split → persist → upload.

    Args:
        data_path   (str): Path to the raw CSV file (e.g. data/raw/telco_churn.csv).
        config_path (str): Path to the YAML preprocessing config (config/process.yaml).
        output_dir  (str): Local directory where processed CSVs will be saved.
        bucket_name (str): MinIO bucket where the processed files are uploaded.

    Returns:
        dict: Keys are X_train, y_train, X_val, y_val, X_test, y_test as DataFrames.
    """
    # ------------------------------------------------------------------ #
    # 1. LOAD RAW DATA                                                     #
    # ------------------------------------------------------------------ #
    df = pd.read_csv(data_path)

    # ------------------------------------------------------------------ #
    # 2. BASIC CLEANING                                                    #
    # ------------------------------------------------------------------ #
    # Remove exact duplicate rows — they add no information and inflate
    # training-set size artificially.
    df.drop_duplicates(inplace=True)

    # Drop rows with any NaN values.  For a production pipeline you would
    # use imputation strategies instead, but this keeps things simple.
    df.dropna(inplace=True)

    # The raw CSV stores TotalCharges as a string (some entries are " ").
    # pd.to_numeric with errors="coerce" turns unparseable strings into NaN,
    # which we then fill with the column median to avoid dropping rows.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # ------------------------------------------------------------------ #
    # 3. FEATURE ENGINEERING                                               #
    # ------------------------------------------------------------------ #
    # Convert the continuous `tenure` (months) into an ordinal bin.
    # This gives the model an easier-to-learn grouping and mirrors how a
    # business analyst would naturally think about customer lifecycle stages.
    df['Tenure_Bin'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 36, 48, 60, 72],
        labels=["0-1 yr", "1-2 yrs", "2-3 yrs", "3-4 yrs", "4-5 yrs", "5-6 yrs"]
    )

    # ------------------------------------------------------------------ #
    # 4. LOAD PREPROCESSING CONFIGURATION                                  #
    # ------------------------------------------------------------------ #
    # config/process.yaml drives which columns are numeric vs categorical
    # and controls the train/test split ratios.  Externalising this to YAML
    # makes it easy to change without touching code (DataOps best practice).
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    numeric_features = config["preprocessing"]["numeric_features"]
    categorical_features = config["preprocessing"]["categorical_features"]
    target_column = config["preprocessing"]["target_column"]      # "Churn"

    # ------------------------------------------------------------------ #
    # 5. BUILD FEATURE MATRIX (X) AND LABEL VECTOR (y)                    #
    # ------------------------------------------------------------------ #
    # X contains only the columns we want to feed the model.
    X = df[numeric_features + categorical_features]

    # y is binary: 1 = customer churned, 0 = retained.
    # The raw label is "Yes"/"No" — we map it to integers here so
    # scikit-learn classifiers accept it without extra transformation.
    y = df[target_column].apply(lambda x: 1 if x == "Yes" else 0)

    # Log preprocessing steps
    print(f"Number of rows after cleaning: {len(df)}")
    print(f"Columns used for training: {numeric_features + categorical_features}")
    print("Data preparation completed successfully.")

    # ------------------------------------------------------------------ #
    # 6. STRATIFIED TRAIN / VALIDATION / TEST SPLIT                        #
    # ------------------------------------------------------------------ #
    # Step 1 — carve off `test_size` (30 %) of the total data.
    # `stratify=y` ensures the churn ratio is preserved in every split.
    train_test_split_params = config["preprocessing"]["train_test_split"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=train_test_split_params["test_size"],       # 0.3 → 70 % train
        random_state=train_test_split_params["random_state"], # reproducibility seed
        stratify=y
    )

    # Step 2 — split the held-out 30 % equally into validation and test.
    # Final proportions: 70 % train | 15 % validation | 15 % test.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=train_test_split_params["random_state"],
        stratify=y_temp
    )

    # ------------------------------------------------------------------ #
    # 7. PERSIST LOCALLY AND UPLOAD TO MINIO                               #
    # ------------------------------------------------------------------ #
    os.makedirs(output_dir, exist_ok=True)

    datasets = {
        "X_train.csv": X_train,
        "y_train.csv": y_train,
        "X_val.csv":   X_val,
        "y_val.csv":   y_val,
        "X_test.csv":  X_test,
        "y_test.csv":  y_test,
    }

    for filename, dataset in datasets.items():
        local_path = os.path.join(output_dir, filename)
        dataset.to_csv(local_path, index=False)
        # Mirror the same relative path structure inside the MinIO bucket
        # so consumers can locate files predictably (e.g. data/processed/X_train.csv).
        upload_file_to_minio(local_path, bucket_name, f"{output_dir}/{filename}")

    # ------------------------------------------------------------------ #
    # 8. SAVE REFERENCE DATA FOR DRIFT DETECTION                           #
    # ------------------------------------------------------------------ #
    # reference_data.csv is a copy of X_train — this is the "baseline"
    # distribution that Evidently AI will use to measure drift later.
    # Storing it in MinIO means the drift-detection container can fetch it
    # without coupling to the training container's filesystem.
    reference_data_path = os.path.join(output_dir, "reference_data.csv")
    X_train.to_csv(reference_data_path, index=False)
    upload_file_to_minio(reference_data_path, bucket_name, f"{output_dir}/reference_data.csv")
    print(f"Reference data uploaded successfully: {reference_data_path}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test
    }


