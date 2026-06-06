# model_training.py
# ---------------------------------------------------------------------------
# PURPOSE: Trains two classifiers (RandomForest and XGBoost) on the processed
#          Telco Churn data, tracks every run with MLflow, selects the best
#          model by ROC-AUC, serialises it with joblib, and uploads both the
#          model artefact and the reference dataset to MinIO.
#
# LEARNING NOTES:
#   - MLflow Tracking: each model run is wrapped in `mlflow.start_run()`.
#     Within that context, we log hyper-parameters (log_param), metrics
#     (log_metric), and the whole pipeline (log_model including its signature).
#     You can inspect all runs at http://127.0.0.1:8080 after starting the
#     MLflow server locally.
#
#   - sklearn Pipeline: chaining the ColumnTransformer (preprocessing) and
#     the classifier into a single Pipeline object ensures that the exact
#     same transformation steps are applied at inference time, preventing
#     training-serving skew.
#
#   - ColumnTransformer applies different preprocessing to numeric vs
#     categorical columns simultaneously:
#       · StandardScaler normalises numeric columns (mean=0, std=1).
#       · OneHotEncoder encodes categorical strings as sparse binary vectors.
#
#   - ROC-AUC (Area Under the ROC Curve) is the selection metric.  A score
#     of 1.0 is perfect; 0.5 is random.  It is threshold-independent and
#     works well for imbalanced binary classification problems like churn.
#
#   - joblib is preferred over pickle for scikit-learn models because it
#     handles numpy arrays more efficiently (memory-mapped files).
# ---------------------------------------------------------------------------

import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import boto3
from botocore.exceptions import NoCredentialsError


def upload_reference_data_to_minio(data_dir, bucket_name, object_name, save_to_models_bucket=False):
    """
    Upload reference_data.csv to MinIO so the drift-detection container can
    retrieve it later without accessing the training container's filesystem.

    Args:
        data_dir              (str):  Local directory containing reference_data.csv.
        bucket_name           (str):  Target MinIO bucket (overridden if save_to_models_bucket=True).
        object_name           (str):  S3 key the file will be stored under.
        save_to_models_bucket (bool): If True, force upload to the "models" bucket.
    Returns:
        str: s3://bucket/key URI of the uploaded file.
    """
    minio_url = os.getenv("MINIO_URL", "http://localhost:9000")
    minio_access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    minio_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

    # Use the 'models' bucket if specified
    if save_to_models_bucket:
        bucket_name = "models"
        reference_data_path = os.path.join("models", object_name)  # Use the models path
    else:
        reference_data_path = os.path.join(data_dir, object_name)  # Use data_dir path

    s3_client = boto3.client(
        "s3",
        endpoint_url=minio_url,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
    )

    # Always resolve to the actual reference_data.csv path in data_dir
    reference_data_path = os.path.join(data_dir, "reference_data.csv")
    try:
        # Ensure bucket exists; create if absent
        try:
            s3_client.head_bucket(Bucket=bucket_name)
        except s3_client.exceptions.ClientError:
            print(f"Bucket {bucket_name} does not exist. Creating it...")
            s3_client.create_bucket(Bucket=bucket_name)

        print(f"Uploading {reference_data_path} to MinIO bucket {bucket_name} with key {object_name}...")
        s3_client.upload_file(reference_data_path, bucket_name, object_name)
        print("Reference data upload successful!")

        return f"s3://{bucket_name}/{object_name}"
    except Exception as e:
        raise RuntimeError(f"Failed to upload reference data to MinIO: {e}")


def upload_model_to_minio(model_dir, bucket_name, model_name):
    """
    Upload the serialised model (model.pkl) to MinIO so the FastAPI service
    and the retraining container can download it at runtime.

    Args:
        model_dir   (str): Local directory containing model.pkl.
        bucket_name (str): Target MinIO bucket (typically "models").
        model_name  (str): Prefix/folder name inside the bucket.
    Returns:
        str: s3://bucket/key URI of the uploaded model.
    """
    minio_url = os.getenv("MINIO_URL", "http://localhost:9000")
    minio_access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    minio_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

    s3_client = boto3.client(
        "s3",
        endpoint_url=minio_url,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
    )

    try:
        model_path = os.path.join(model_dir, "model.pkl")
        # Store under <model_name>/model.pkl so multiple model versions can
        # coexist under different prefix names in the same bucket.
        s3_key = f"{model_name}/model.pkl"

        # Ensure bucket exists; differentiate credential errors from bucket errors
        try:
            s3_client.head_bucket(Bucket=bucket_name)
        except NoCredentialsError:
            raise RuntimeError("Invalid MinIO credentials")
        except s3_client.exceptions.ClientError:
            print(f"Bucket {bucket_name} does not exist. Creating it...")
            s3_client.create_bucket(Bucket=bucket_name)

        print(f"Uploading {model_path} to MinIO bucket {bucket_name} with key {s3_key}...")
        s3_client.upload_file(model_path, bucket_name, s3_key)
        print("Upload successful!")

        return f"s3://{bucket_name}/{s3_key}"
    except Exception as e:
        raise RuntimeError(f"Failed to upload model to MinIO: {e}")


def train_and_save_model(data_dir, best_model_dir):
    """
    Core training loop: loads processed splits, defines preprocessing +
    classifier pipelines, trains each model with MLflow tracking, picks the
    winner by validation AUC, and serialises it locally.

    Args:
        data_dir       (str): Directory containing the processed CSV splits.
        best_model_dir (str): Directory where the winning model.pkl is saved.
    """
    # ------------------------------------------------------------------ #
    # 1. LOAD PROCESSED DATASETS                                           #
    # ------------------------------------------------------------------ #
    # These files were produced by data_preparation.py and uploaded to
    # MinIO; in the Kubernetes Job they are downloaded first, then this
    # script reads them from a local /app/data/processed path.
    X_train_path = os.path.join(data_dir, "X_train.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    X_val_path   = os.path.join(data_dir, "X_val.csv")
    y_val_path   = os.path.join(data_dir, "y_val.csv")

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    X_val   = pd.read_csv(X_val_path)
    y_val   = pd.read_csv(y_val_path)

    # ------------------------------------------------------------------ #
    # 2. DEFINE PREPROCESSING PIPELINE                                     #
    # ------------------------------------------------------------------ #
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [
        "gender", "Partner", "Dependents", "PhoneService",
        "InternetService", "Contract", "PaymentMethod", "Tenure_Bin"
    ]

    # StandardScaler: subtracts mean, divides by std.  Required for
    # distance-based algorithms; also helps tree-based models converge faster.
    numeric_transformer = StandardScaler()

    # OneHotEncoder: converts each category to a binary indicator column.
    # handle_unknown="ignore" prevents errors when the validation/test set
    # contains a category not seen during training.
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # ColumnTransformer applies each transformer to its designated columns,
    # concatenating the results into a single feature matrix.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,    numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # ------------------------------------------------------------------ #
    # 3. CONFIGURE MLFLOW EXPERIMENT                                       #
    # ------------------------------------------------------------------ #
    # MLFLOW_TRACKING_URI can be overridden via env var.  Locally it points
    # to the server started with `mlflow server --host 127.0.0.1 --port 8080`.
    # In Kubernetes it points to the mlserver Service (port 5000).
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080"))

    # All runs go under the "Churn_Prediction" experiment so they are grouped
    # and comparable in the MLflow UI.
    mlflow.set_experiment("Churn_Prediction")

    # ------------------------------------------------------------------ #
    # 4. DEFINE CANDIDATE MODELS                                           #
    # ------------------------------------------------------------------ #
    # Each model is paired with the same preprocessor inside a Pipeline.
    # Adding more models here is easy — just extend this dict.
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost":      XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
    }

    best_model = None
    best_auc   = 0

    # ------------------------------------------------------------------ #
    # 5. TRAIN, EVALUATE, AND LOG EACH MODEL                               #
    # ------------------------------------------------------------------ #
    for model_name, model in models.items():
        # Each `with mlflow.start_run()` block creates a new run row in the
        # MLflow UI with its own parameters, metrics, and artefacts.
        with mlflow.start_run(run_name=model_name):

            # Build the end-to-end pipeline: preprocessing → classifier
            clf_pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier",   model)
            ])

            # y_train is a single-column DataFrame; .values.ravel() flattens
            # it to a 1-D array as expected by sklearn estimators.
            clf_pipeline.fit(X_train, y_train.values.ravel())

            # Predict class probabilities on the validation set.
            # [:, 1] selects the probability for the positive class (churn=1).
            y_val_proba = clf_pipeline.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_val_proba)

            # -- MLflow logging --
            mlflow.log_param("model", model_name)    # what algorithm was used
            mlflow.log_metric("auc", auc)            # validation ROC-AUC

            # infer_signature extracts the input/output schema from actual
            # data; this is stored alongside the model in MLflow and enables
            # input validation at serving time.
            signature = infer_signature(X_train, clf_pipeline.predict(X_train))
            mlflow.sklearn.log_model(
                clf_pipeline,
                artifact_path="model",
                input_example=X_train[:5],    # first 5 rows as a usage example
                signature=signature,
            )

            # Track the model with the highest validation AUC
            if auc > best_auc:
                best_auc   = auc
                best_model = clf_pipeline

    # ------------------------------------------------------------------ #
    # 6. SERIALISE THE WINNING MODEL                                       #
    # ------------------------------------------------------------------ #
    best_model_path = os.path.join(best_model_dir, "model.pkl")
    os.makedirs(best_model_dir, exist_ok=True)

    # joblib.dump serialises the full Pipeline (preprocessing + classifier)
    # so inference code only needs to call pipeline.predict_proba(X).
    with open(best_model_path, "wb") as f:
        joblib.dump(best_model, f)

    print(f"Best Model AUC: {best_auc}")
    print(f"Best model saved to {best_model_path}")


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # ENTRY POINT — runs the full train → upload sequence                  #
    # ------------------------------------------------------------------ #
    DATA_DIR       = "data/processed"
    BEST_MODEL_DIR = "models/best_model"
    BUCKET_NAME    = "models"
    MODEL_NAME     = "best_model"

    # Step 1: train locally and save model.pkl
    model_path = train_and_save_model(DATA_DIR, BEST_MODEL_DIR)

    # Step 2: push the serialised model to MinIO (makes it available to
    # the FastAPI service running in Kubernetes)
    model_uri = upload_model_to_minio(BEST_MODEL_DIR, BUCKET_NAME, MODEL_NAME)
    print(f"Model available at: {model_uri}")

    # Step 3: push the reference snapshot so drift detection can compare
    # future data distributions against the training distribution
    REFERENCE_OBJECT_NAME = "reference_data.csv"
    reference_uri = upload_reference_data_to_minio(DATA_DIR, BUCKET_NAME, REFERENCE_OBJECT_NAME)
    print(f"Reference data available at: {reference_uri}")
    