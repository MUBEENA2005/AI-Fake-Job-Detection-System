"""
==================================================================
Day 3: Machine Learning Model Development
Project: AI-Based Fake Job Detection and Verification System
==================================================================

WHAT THIS SCRIPT DOES:
Loads the feature artifacts already produced by Day 2
(src/preprocessing.py), reconstructs the final feature matrix,
trains three candidate classifiers using the EXACT train/test split
saved during Day 2 (no new random split), evaluates them with a
focus on the fraudulent class, selects the best model, and saves
the model, its metadata, a comparison table, and a confusion-matrix
image.

WHAT THIS SCRIPT DOES NOT DO:
- It does not modify src/preprocessing.py or the raw CSV.
- It does not build the Flask app, a database, or a prediction UI.
- It does not apply SMOTE (class-weighted models are used instead).
- It does not invent any numbers — every metric printed and saved
  comes from actually running scikit-learn on the real data.

WHY WE REUSE THE DAY 2 TRAIN/TEST SPLIT (NOT A NEW RANDOM SPLIT):
The TF-IDF vectorizer and the categorical encoders in Day 2 were
already fitted using only the "train" rows of that saved split. If
we now split the data differently here, rows that were used to FIT
those transformers could end up in today's "test" set — leaking
information into evaluation. Reusing the identical indices keeps
the whole pipeline leakage-free from Day 2 through Day 3.
==================================================================
"""

import os
import json

import numpy as np
import pandas as pd
from scipy import sparse
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless/script use
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# PROJECT PATHS (relative — same convention as Day 1 and Day 2)
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

TFIDF_PATH = os.path.join(PROCESSED_DIR, "tfidf_features.npz")
STRUCTURED_PATH = os.path.join(PROCESSED_DIR, "structured_features.csv")
TARGET_PATH = os.path.join(PROCESSED_DIR, "target.csv")
SPLIT_PATH = os.path.join(PROCESSED_DIR, "train_test_indices.joblib")

TARGET_COL = "fraudulent"
RANDOM_STATE = 42


# ==================================================================
# STEP 1 — LOAD DAY 2 ARTIFACTS
# ==================================================================
def load_processed_artifacts():
    """
    WHAT: Loads the four artifacts produced by Day 2's
          src/preprocessing.py: the TF-IDF matrix, the structured
          feature table, the target labels, and the saved
          train/test row indices.
    WHY:  Day 3 must build directly on Day 2's output rather than
          recomputing anything, so that both stages stay perfectly
          consistent (same rows, same features, same split).
    SAFETY: Each file is checked for existence first, with a clear
          error message pointing back to Day 2 if something is
          missing — this is much friendlier than a raw
          FileNotFoundError deep inside the script.
    """
    required_files = {
        "TF-IDF features": TFIDF_PATH,
        "Structured features": STRUCTURED_PATH,
        "Target labels": TARGET_PATH,
        "Train/test indices": SPLIT_PATH,
    }
    for label, path in required_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} not found at: {path}\n"
                f"Please run 'python src/preprocessing.py' (Day 2) first."
            )

    tfidf_matrix = sparse.load_npz(TFIDF_PATH)
    structured_df = pd.read_csv(STRUCTURED_PATH)
    target_df = pd.read_csv(TARGET_PATH)
    split_data = joblib.load(SPLIT_PATH)
    train_idx = split_data["train_idx"]
    test_idx = split_data["test_idx"]

    # ------------------------------------------------------------
    # KNOWN DAY 2 ARTIFACT QUIRK (documented here, preprocessing.py
    # itself is NOT modified, per Day 3 instructions):
    # The one-hot encoded columns in structured_features.csv were
    # built from a pandas "sparse" DataFrame. When that sparse
    # DataFrame was written to CSV, its zero/fill values were saved
    # as BLANK cells rather than literal "0" characters, so
    # pd.read_csv loads them back as NaN. A blank one-hot cell
    # unambiguously means "this category was not present for that
    # row", i.e. the value is 0 — it is not genuinely missing data.
    # We therefore safely fill these with 0 here before modeling.
    # (The frequency-encoded, binary, and indicator columns from
    # Day 2 were already fully populated, so this fillna only ever
    # affects the one-hot columns in practice.)
    # ------------------------------------------------------------
    n_nan_before = structured_df.isna().sum().sum()
    structured_df = structured_df.fillna(0)
    print(f"[Step 1] Filled {n_nan_before} blank one-hot cells in structured "
          f"features with 0 (see comment in load_processed_artifacts()).")

    print(f"[Step 1] Loaded TF-IDF matrix: {tfidf_matrix.shape}")
    print(f"[Step 1] Loaded structured features: {structured_df.shape}")
    print(f"[Step 1] Loaded target labels: {target_df.shape}")
    print(f"[Step 1] Loaded split: train={len(train_idx)}, test={len(test_idx)}")

    return tfidf_matrix, structured_df, target_df, train_idx, test_idx


# ==================================================================
# STEP 2 — INTEGRITY / LEAKAGE CHECKS
# ==================================================================
def run_integrity_checks(tfidf_matrix, structured_df, target_df, train_idx, test_idx):
    """
    WHAT: A set of assertions that must ALL pass before any model is
          trained. If any check fails, the script stops immediately
          with a clear error rather than silently producing
          misleading results.
    WHY EACH CHECK MATTERS:
      1. Row counts must match across every artifact — otherwise
         feature row i would not correspond to target row i.
      2. job_id order must match between structured features and
         target — this is the real proof that rows are aligned
         (matching shapes alone is not enough).
      3. Train and test indices must not overlap — any overlap would
         mean the model is evaluated on data it also trained on.
      4. Train + test indices together must cover every row exactly
         once — otherwise some rows are silently dropped or
         duplicated.
      5. The target column must not appear among the structured
         feature columns — this is the classic "target leakage" bug
         where the answer accidentally becomes an input.
    """
    n_tfidf = tfidf_matrix.shape[0]
    n_struct = structured_df.shape[0]
    n_target = target_df.shape[0]

    assert n_tfidf == n_struct == n_target, (
        f"Row count mismatch! tfidf={n_tfidf}, structured={n_struct}, target={n_target}"
    )

    assert (structured_df["job_id"].values == target_df["job_id"].values).all(), (
        "job_id order does not match between structured_features.csv and target.csv! "
        "Rows are not aligned."
    )

    overlap = set(train_idx) & set(test_idx)
    assert len(overlap) == 0, f"Train and test indices overlap! {len(overlap)} rows shared."

    all_idx = set(train_idx) | set(test_idx)
    assert len(all_idx) == n_target, (
        f"Train+test indices do not cover all rows! "
        f"Covered {len(all_idx)} of {n_target} rows."
    )

    assert TARGET_COL not in structured_df.columns, (
        f"Target column '{TARGET_COL}' found inside structured features — target leakage!"
    )

    print("[Step 2] All integrity checks passed: row alignment OK, "
          "no train/test overlap, all rows accounted for, no target leakage.")


# ==================================================================
# STEP 3 — BUILD THE FINAL FEATURE MATRIX
# ==================================================================
def build_feature_matrix(tfidf_matrix, structured_df):
    """
    WHAT: Horizontally combines the TF-IDF matrix (3,000 text
          features) with the structured feature table (81 numeric
          features, e.g. binary flags, missingness flags, encoded
          categories) into ONE feature matrix.
    WHY SPARSE, NOT DENSE: The TF-IDF matrix is a SciPy sparse
          matrix — most of its values are zero (a given job posting
          only uses a small fraction of the 3,000-word vocabulary).
          Converting it to a regular dense NumPy array would use far
          more memory for no benefit. `scipy.sparse.hstack` combines
          the sparse TF-IDF matrix with the (also sparsified)
          structured features while staying in sparse format, which
          scikit-learn's classifiers can consume directly.
    """
    # job_id is an identifier, not a predictive feature — drop it here.
    structured_values = structured_df.drop(columns=["job_id"]).values.astype(np.float64)
    structured_sparse = sparse.csr_matrix(structured_values)

    feature_matrix = sparse.hstack([tfidf_matrix, structured_sparse]).tocsr()

    feature_names = list(structured_df.drop(columns=["job_id"]).columns)
    print(f"[Step 3] Combined feature matrix built: {feature_matrix.shape} "
          f"({tfidf_matrix.shape[1]} TF-IDF + {structured_sparse.shape[1]} structured)")
    return feature_matrix, feature_names


# ==================================================================
# STEP 4 — APPLY THE DAY 2 TRAIN/TEST SPLIT
# ==================================================================
def split_using_saved_indices(feature_matrix, target_df, train_idx, test_idx):
    """
    WHAT: Slices the feature matrix and target labels using the
          EXACT row indices saved during Day 2 — no new split is
          created here.
    """
    y = target_df[TARGET_COL].values

    X_train = feature_matrix[train_idx]
    X_test = feature_matrix[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    print(f"[Step 4] Applied Day 2 split -> X_train: {X_train.shape}, "
          f"X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ==================================================================
# STEP 5 — TRAIN CANDIDATE MODELS
# ==================================================================
def train_candidate_models(X_train, y_train):
    """
    WHAT: Trains three different classifiers, all using
          class_weight='balanced' to handle the 95.16%/4.84% class
          imbalance WITHOUT resampling (no SMOTE), as required.
    WHY class_weight='balanced' INSTEAD OF SMOTE HERE: SMOTE
          generates synthetic minority-class samples by interpolating
          between existing points in feature space. Doing that
          directly on a 3,000-dimensional sparse TF-IDF matrix is
          expensive, can produce nonsensical "synthetic text vectors",
          and is unnecessary as a first attempt. class_weight='balanced'
          instead simply makes the model penalize mistakes on the
          minority (fraudulent) class more heavily during training —
          a simpler, cheaper, and well-supported first approach.

    MODELS AND WHY THESE THREE:
      - LogisticRegression: a simple, fast, highly interpretable
        linear baseline — a natural first model for any binary
        classification problem.
      - RandomForestClassifier: a non-linear ensemble model that can
        capture more complex interactions between structured
        features and text features than a linear model can.
      - LinearSVC: a linear Support Vector Machine, which often
        performs very well on high-dimensional sparse text data
        such as TF-IDF matrices.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Linear SVM (LinearSVC)": LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            dual=False,       # recommended when n_samples > n_features; works with sparse input
            max_iter=5000,
        ),
    }

    fitted_models = {}
    for name, model in models.items():
        print(f"[Step 5] Training {name} ...")
        model.fit(X_train, y_train)
        fitted_models[name] = model
        print(f"[Step 5] Finished training {name}.")

    return fitted_models


# ==================================================================
# STEP 6 — EVALUATE EACH MODEL
# ==================================================================
def evaluate_model(model, X_test, y_test):
    """
    WHAT: Computes accuracy, and precision/recall/F1 for BOTH classes
          (with the fraudulent class, label=1, singled out since that
          is the class we actually care about catching), plus the
          confusion matrix.
    WHY REPORT THE FRAUDULENT CLASS SEPARATELY: With 95.16% of
          postings genuine, a model could reach ~95% "accuracy" by
          predicting "genuine" for everything and never catching a
          single fraudulent posting. Overall accuracy alone would
          hide this failure completely — the fraudulent-class
          recall and F1-score are what actually tell us whether the
          model is useful for this project's real goal.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    # precision/recall/f1 per class: index 0 = genuine, index 1 = fraudulent
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1], zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    results = {
        "accuracy": accuracy,
        "precision_genuine": precision[0],
        "recall_genuine": recall[0],
        "f1_genuine": f1[0],
        "precision_fraudulent": precision[1],
        "recall_fraudulent": recall[1],
        "f1_fraudulent": f1[1],
        "confusion_matrix": cm,
        "y_pred": y_pred,
    }
    return results


# ==================================================================
# STEP 7 — COMPARE MODELS AND SELECT THE BEST ONE
# ==================================================================
def build_comparison_table(all_results: dict) -> pd.DataFrame:
    """
    WHAT: Builds a tidy comparison table across all trained models.
    """
    rows = []
    for name, res in all_results.items():
        rows.append({
            "Model": name,
            "Accuracy": res["accuracy"],
            "Precision": res["precision_fraudulent"],
            "Recall": res["recall_fraudulent"],
            "F1-score": res["f1_fraudulent"],
            "Precision (genuine)": res["precision_genuine"],
            "Recall (genuine)": res["recall_genuine"],
            "F1-score (genuine)": res["f1_genuine"],
        })
    comparison_df = pd.DataFrame(rows)
    return comparison_df


def select_best_model(comparison_df: pd.DataFrame) -> str:
    """
    WHAT: Picks the best model primarily by fraudulent-class
          F1-score, using fraudulent-class recall as a tiebreaker.
    WHY NOT ACCURACY: As explained in evaluate_model(), accuracy is
          misleading on this imbalanced dataset. F1-score balances
          precision and recall for the class we actually care about
          detecting (fraudulent postings), which is why it is the
          primary selection criterion here, as required.
    """
    sorted_df = comparison_df.sort_values(
        by=["F1-score", "Recall"], ascending=[False, False]
    )
    best_model_name = sorted_df.iloc[0]["Model"]
    print(f"[Step 7] Model ranking (by fraudulent-class F1-score, then recall):")
    print(sorted_df[["Model", "F1-score", "Recall", "Accuracy"]].to_string(index=False))
    return best_model_name


# ==================================================================
# STEP 8 — SAVE OUTPUTS
# ==================================================================
def save_comparison_csv(comparison_df: pd.DataFrame):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, "model_comparison.csv")
    comparison_df.to_csv(path, index=False)
    print(f"[Step 8] Model comparison table saved to: {path}")


def save_best_model(model, model_name: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(model, path)
    print(f"[Step 8] Best model ('{model_name}') saved to: {path}")
    return path


def save_metadata(
    model_name: str,
    n_features: int,
    n_train: int,
    n_test: int,
    metrics: dict,
    model_path: str,
):
    os.makedirs(MODELS_DIR, exist_ok=True)
    metadata = {
        "model_name": model_name,
        "feature_count": n_features,
        "training_row_count": n_train,
        "test_row_count": n_test,
        "evaluation_metrics": {
            "accuracy": metrics["accuracy"],
            "precision_fraudulent": metrics["precision_fraudulent"],
            "recall_fraudulent": metrics["recall_fraudulent"],
            "f1_fraudulent": metrics["f1_fraudulent"],
            "precision_genuine": metrics["precision_genuine"],
            "recall_genuine": metrics["recall_genuine"],
            "f1_genuine": metrics["f1_genuine"],
        },
        "random_state": RANDOM_STATE,
        "class_imbalance_handling": "class_weight='balanced' (no resampling/SMOTE applied)",
        "model_file": os.path.relpath(model_path, PROJECT_ROOT),
    }
    path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Step 8] Model metadata saved to: {path}")


def save_confusion_matrix_plot(cm: np.ndarray, model_name: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Genuine (0)", "Fraudulent (1)"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    path = os.path.join(MODELS_DIR, "best_model_confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Step 8] Confusion matrix plot saved to: {path}")


# ==================================================================
# MAIN PIPELINE
# ==================================================================
def main():
    print("=" * 70)
    print("DAY 3 — MACHINE LEARNING MODEL DEVELOPMENT")
    print("=" * 70)

    # Step 1: Load Day 2 artifacts
    tfidf_matrix, structured_df, target_df, train_idx, test_idx = load_processed_artifacts()

    # Step 2: Integrity / leakage checks (must pass before anything else)
    run_integrity_checks(tfidf_matrix, structured_df, target_df, train_idx, test_idx)

    # Step 3: Build the combined feature matrix (kept sparse throughout)
    feature_matrix, feature_names = build_feature_matrix(tfidf_matrix, structured_df)

    # Step 4: Apply the EXACT Day 2 train/test split
    X_train, X_test, y_train, y_test = split_using_saved_indices(
        feature_matrix, target_df, train_idx, test_idx
    )

    # Step 5: Train all three candidate models
    fitted_models = train_candidate_models(X_train, y_train)

    # Step 6: Evaluate every model
    all_results = {}
    for name, model in fitted_models.items():
        print(f"[Step 6] Evaluating {name} ...")
        all_results[name] = evaluate_model(model, X_test, y_test)

    # Step 7: Compare and select the best model
    comparison_df = build_comparison_table(all_results)
    best_model_name = select_best_model(comparison_df)
    best_model = fitted_models[best_model_name]
    best_results = all_results[best_model_name]

    # Step 8: Save everything
    save_comparison_csv(comparison_df)
    model_path = save_best_model(best_model, best_model_name)
    save_metadata(
        model_name=best_model_name,
        n_features=feature_matrix.shape[1],
        n_train=X_train.shape[0],
        n_test=X_test.shape[0],
        metrics=best_results,
        model_path=model_path,
    )
    save_confusion_matrix_plot(best_results["confusion_matrix"], best_model_name)

    # ==============================================================
    # FINAL SUMMARY (printed to terminal)
    # ==============================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Full feature matrix shape:        {feature_matrix.shape}")
    print(f"  - TF-IDF features:               {tfidf_matrix.shape[1]}")
    print(f"  - Structured features:           {feature_matrix.shape[1] - tfidf_matrix.shape[1]}")
    print(f"Training rows:                     {X_train.shape[0]}")
    print(f"Test rows:                         {X_test.shape[0]}")
    print("\nModel comparison (sorted by fraudulent-class F1-score):")
    print(comparison_df.sort_values(by="F1-score", ascending=False).to_string(index=False))
    print(f"\nSelected best model:               {best_model_name}")
    print(f"  Fraudulent-class precision:      {best_results['precision_fraudulent']:.4f}")
    print(f"  Fraudulent-class recall:         {best_results['recall_fraudulent']:.4f}")
    print(f"  Fraudulent-class F1-score:       {best_results['f1_fraudulent']:.4f}")
    print(f"  Overall accuracy:                {best_results['accuracy']:.4f}")
    print(f"\nSaved model path:                  {model_path}")
    print("=" * 70)
    print("DAY 3 MODEL DEVELOPMENT COMPLETE — Flask app, database, and "
          "explainability work have NOT been started.")
    print("=" * 70)


if __name__ == "__main__":
    main()
