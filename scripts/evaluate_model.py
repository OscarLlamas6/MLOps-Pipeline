# evaluate_model.py
# ---------------------------------------------------------------------------
# PURPOSE: Post-training evaluation step. Loads the best saved model and the
#          held-out test split, computes classification metrics (ROC-AUC and
#          full classification report), and runs a first-pass drift check
#          comparing the test set against the training reference data.
#
# LEARNING NOTES:
#   - ROC-AUC measures the model's ability to discriminate between classes
#     across all decision thresholds — higher is better (1.0 = perfect).
#
#   - classification_report prints precision, recall, F1-score and support
#     per class, plus macro/weighted averages.  For imbalanced datasets
#     (churn is typically ~26 % positive) look at F1 and recall for class 1.
#
#   - Evidently AI (https://evidentlyai.com) compares two DataFrames using
#     statistical tests (e.g. Wasserstein distance for numerics, chi-square
#     for categoricals) and produces an interactive HTML report.
#     DataDriftPreset bundles these tests as a preset pack.
#
#   - Running drift detection here (post-training) gives you an early
#     sanity check: ideally test-set distribution should NOT drift from
#     training data, so the drift score should be low (< 0.2).
#     Any score > 0.3 would warrant investigation.
#
# ARTEFACTS PRODUCED:
#   models/evaluation_reports/classification_report.txt
#   models/evaluation_reports/data_drift_report.html
# ---------------------------------------------------------------------------

import os
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, classification_report
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def evaluate_model(model_path, data_dir):
    """
    Load the trained model, evaluate it on the test set, and generate reports.

    Args:
        model_path (str): Path to the serialised model file (model.pkl).
        data_dir   (str): Directory containing X_test.csv, y_test.csv, X_train.csv.
    """
    # ------------------------------------------------------------------ #
    # 1. LOAD MODEL                                                        #
    # ------------------------------------------------------------------ #
    # joblib.load deserialises the full sklearn Pipeline (preprocessor +
    # classifier), making it immediately usable for prediction.
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

    # ------------------------------------------------------------------ #
    # 2. LOAD TEST DATA                                                    #
    # ------------------------------------------------------------------ #
    # X_test and y_test are the held-out splits created during data
    # preparation (15 % of the original dataset, stratified).
    X_test_path = os.path.join(data_dir, "X_test.csv")
    y_test_path = os.path.join(data_dir, "y_test.csv")
    try:
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path)
    except Exception as e:
        raise RuntimeError(f"Error loading test data: {e}")

    # ------------------------------------------------------------------ #
    # 3. COMPUTE METRICS                                                   #
    # ------------------------------------------------------------------ #
    # predict_proba returns [[P(0), P(1)], ...]; [:, 1] gives churn prob.
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC on test set: {auc}")

    # Convert probabilities to binary predictions using 0.5 as decision threshold.
    # In production you may tune this threshold to balance precision vs recall.
    y_pred = (y_pred_proba > 0.5).astype(int)
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    # ------------------------------------------------------------------ #
    # 4. SAVE CLASSIFICATION REPORT                                        #
    # ------------------------------------------------------------------ #
    report_dir = "models/evaluation_reports"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"ROC AUC: {auc}\n\n")
        f.write(report)
    print(f"Evaluation report saved to {report_path}")

    # ------------------------------------------------------------------ #
    # 5. DATA DRIFT DETECTION (BASELINE CHECK)                             #
    # ------------------------------------------------------------------ #
    # Compare X_test (current) against X_train (reference) to verify the
    # test set's feature distributions haven't drifted from training.
    # This is also used as a smoke test: if there IS drift here it usually
    # indicates a data preparation bug (e.g. leakage or wrong split).
    reference_data_path = os.path.join(data_dir, "X_train.csv")
    reference_data = pd.read_csv(reference_data_path)

    # DataDriftPreset applies per-column statistical tests and aggregates
    # a drift_share (fraction of columns deemed drifted).
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_data, current_data=X_test)

    # Save as a self-contained interactive HTML — open in any browser.
    drift_report_path = os.path.join(report_dir, "data_drift_report.html")
    drift_report.save_html(drift_report_path)
    print(f"Data Drift Report saved to {drift_report_path}")


if __name__ == "__main__":
    MODEL_PATH = "models/best_model/model.pkl"
    DATA_DIR   = "data/processed"
    print("Evaluating the best model...")
    evaluate_model(MODEL_PATH, DATA_DIR)


