"""
==================================================================
Day 4: Prediction on New Job Postings
Project: AI-Based Fake Job Detection and Verification System
==================================================================

WHAT THIS SCRIPT DOES:
Takes a single NEW job posting (as a Python dictionary) and produces
a fraud-risk prediction, using ONLY the artifacts already fitted and
saved during Day 2 (preprocessing) and Day 3 (modeling). It does not
retrain anything, does not touch the raw dataset, and does not
build Flask or a database — this is a plain, testable Python
function that Flask can call directly in a later stage.

WHY WE IMPORT FROM preprocessing.py INSTEAD OF COPYING ITS LOGIC:
Day 2's text-cleaning rules and column lists (TEXT_COLUMNS,
BINARY_COLUMNS, CATEGORICAL_COLUMNS, etc.) must be applied to a new
job posting in EXACTLY the same way they were applied when the
model was trained — otherwise the features would not match what the
model learned from. Re-typing that logic here would risk it quietly
drifting out of sync with preprocessing.py over time. Importing the
constants and the `clean_single_text` function directly from
preprocessing.py guarantees they can never disagree. This is a
read-only import — preprocessing.py itself is NOT modified, and
running `python src/prediction.py` does NOT re-run preprocessing's
pipeline (its main() is guarded by `if __name__ == "__main__":`).

HIGH-LEVEL FLOW OF predict_job():
    1. Validate the incoming job posting dictionary.
    2. Rebuild the same 3,000 TF-IDF text features (Day 2 rules).
    3. Rebuild the same 81 structured features (Day 2 rules).
    4. Combine them into one 3,081-feature row, in the exact order
       the model was trained on.
    5. Run the Day 3 best_model to get a prediction.
    6. Convert the model's decision_function output into a
       0-100 "risk score" for human-friendly interpretation
       (explicitly NOT a calibrated probability).
    7. Compute simple, explainable rule-based risk indicators.
    8. Return everything in one clear, structured result.
==================================================================
"""

import os
import sys
import json

import numpy as np
import pandas as pd
from scipy import sparse
import joblib

# ------------------------------------------------------------------
# Import shared definitions from preprocessing.py (read-only import;
# preprocessing.py is NOT modified, and its main() does not run here
# because it is guarded by `if __name__ == "__main__":`).
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from preprocessing import (         # noqa: E402  (import after sys.path setup is intentional)
    TEXT_COLUMNS,
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    ONE_HOT_COLUMNS,
    FREQUENCY_ENCODE_COLUMNS,
    clean_single_text,
    URL_PATTERN,
    EMAIL_PATTERN,
)

# ------------------------------------------------------------------
# PROJECT PATHS (relative — same convention as Day 1-3 scripts)
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

TFIDF_VECTORIZER_PATH = os.path.join(PROCESSED_DIR, "tfidf_vectorizer.joblib")
ONEHOT_ENCODER_PATH = os.path.join(PROCESSED_DIR, "onehot_encoder.joblib")
FREQ_MAPS_PATH = os.path.join(PROCESSED_DIR, "frequency_encoding_maps.joblib")
PREPROCESSING_METADATA_PATH = os.path.join(PROCESSED_DIR, "preprocessing_metadata.json")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
MODEL_METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# Required fields: a job posting must at least have SOME title and
# description text for the prediction to be meaningful (TF-IDF on
# completely empty text carries no signal). All other fields are
# optional and are treated as "missing" (same rule as Day 2) if
# absent, exactly like a blank cell would have been in the raw CSV.
REQUIRED_TEXT_FIELDS = ["title", "description"]

# Simple, explainable threshold for the "unusually short description"
# indicator (Day 2 also computed description_length as a feature —
# this is a separate, human-readable rule for the indicator list).
SHORT_DESCRIPTION_THRESHOLD = 50  # characters, after cleaning

# Risk-level cut points, as specified for this project.
RISK_LEVEL_THRESHOLDS = {"Low Risk": (0, 29), "Medium Risk": (30, 59), "High Risk": (60, 100)}


# ==================================================================
# STEP 1 — LOAD ARTIFACTS (Day 2 + Day 3, never retrained here)
# ==================================================================
def load_artifacts() -> dict:
    """
    WHAT: Loads the fitted TF-IDF vectorizer, OneHotEncoder,
          frequency-encoding maps, and the trained best_model, plus
          the exact structured-feature column order recorded during
          Day 2 (so we can guarantee the new posting's features line
          up with what the model expects).
    WHY CHECK EACH FILE EXPLICITLY: A clear, specific error message
          ("run preprocessing.py first") is far more useful to a
          final-year student debugging this than a generic
          FileNotFoundError buried inside joblib's internals.
    """
    required_files = {
        "TF-IDF vectorizer (Day 2)": TFIDF_VECTORIZER_PATH,
        "One-hot encoder (Day 2)": ONEHOT_ENCODER_PATH,
        "Frequency encoding maps (Day 2)": FREQ_MAPS_PATH,
        "Preprocessing metadata (Day 2)": PREPROCESSING_METADATA_PATH,
        "Best trained model (Day 3)": BEST_MODEL_PATH,
        "Model metadata (Day 3)": MODEL_METADATA_PATH,
    }
    for label, path in required_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} not found at: {path}\n"
                f"Make sure Day 2 (src/preprocessing.py) and Day 3 "
                f"(src/model_training.py) have both been run successfully "
                f"before using prediction.py."
            )

    tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    one_hot_encoder = joblib.load(ONEHOT_ENCODER_PATH)
    freq_maps = joblib.load(FREQ_MAPS_PATH)
    model = joblib.load(BEST_MODEL_PATH)

    with open(PREPROCESSING_METADATA_PATH) as f:
        preprocessing_metadata = json.load(f)
    structured_feature_columns = preprocessing_metadata["structured_feature_columns"]

    with open(MODEL_METADATA_PATH) as f:
        model_metadata = json.load(f)
    expected_feature_count = model_metadata["feature_count"]
    model_name = model_metadata["model_name"]

    return {
        "tfidf_vectorizer": tfidf_vectorizer,
        "one_hot_encoder": one_hot_encoder,
        "freq_maps": freq_maps,
        "model": model,
        "structured_feature_columns": structured_feature_columns,
        "expected_feature_count": expected_feature_count,
        "model_name": model_name,
    }


# ==================================================================
# STEP 2 — VALIDATE THE INCOMING JOB POSTING
# ==================================================================
def validate_job_data(job_data: dict) -> None:
    """
    WHAT: Basic, strong input validation before any processing.
    WHY:  Failing early with a clear message is much safer than
          letting a malformed input silently produce a meaningless
          prediction.
    """
    if not isinstance(job_data, dict):
        raise TypeError(f"job_data must be a dict, got {type(job_data).__name__}.")

    missing_required = [
        field for field in REQUIRED_TEXT_FIELDS
        if not str(job_data.get(field, "")).strip()
    ]
    if missing_required:
        raise ValueError(
            f"Missing required field(s): {missing_required}. "
            f"A job posting needs at least a non-empty 'title' and "
            f"'description' to generate a meaningful prediction."
        )

    for col in BINARY_COLUMNS:
        if col in job_data and job_data[col] is not None:
            value = job_data[col]
            if value not in (0, 1, True, False):
                raise ValueError(
                    f"Field '{col}' must be 0, 1, True, or False (got {value!r})."
                )


# ==================================================================
# STEP 3 — REBUILD FEATURES FOR ONE NEW JOB POSTING
# ==================================================================
def _is_missing(value) -> bool:
    """A field counts as 'missing' if it's absent, None, or NaN — the
    same idea as pandas' isna() used on the raw CSV in Day 2. An
    explicitly provided empty string is treated as 'provided but
    blank', matching how an already-empty text cell behaves."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def build_feature_vector(job_data: dict, artifacts: dict) -> sparse.csr_matrix:
    """
    WHAT: Converts one job posting dictionary into the exact
          3,081-feature sparse row the model expects, applying the
          identical rules Day 2 used on the full training dataset.
    """
    # ---- Missingness flags (computed BEFORE filling, same as Day 2) ----
    missingness = {
        "missing_company_profile": int(_is_missing(job_data.get("company_profile"))),
        "missing_salary_range": int(_is_missing(job_data.get("salary_range"))),
        "missing_requirements": int(_is_missing(job_data.get("requirements"))),
        "missing_benefits": int(_is_missing(job_data.get("benefits"))),
        "missing_department": int(_is_missing(job_data.get("department"))),
    }

    # ---- Text columns: fill missing with "", then clean (Day 2 rule) ----
    cleaned_text = {}
    for col in TEXT_COLUMNS:
        raw_value = job_data.get(col)
        raw_value = "" if _is_missing(raw_value) else str(raw_value)
        cleaned_text[col] = clean_single_text(raw_value)

    combined_text = (
        cleaned_text["title"] + " . " +
        cleaned_text["company_profile"] + " . " +
        cleaned_text["description"] + " . " +
        cleaned_text["requirements"] + " . " +
        cleaned_text["benefits"]
    )

    # ---- TF-IDF transform using the ALREADY-FITTED Day 2 vectorizer ----
    tfidf_vector = artifacts["tfidf_vectorizer"].transform([combined_text])  # shape (1, 3000)

    # ---- Binary columns: fill missing with 0 (Day 2 rule) ----
    binary_values = {}
    for col in BINARY_COLUMNS:
        raw_value = job_data.get(col)
        binary_values[col] = 0 if _is_missing(raw_value) else int(raw_value)

    # ---- Categorical columns: fill missing with "Unknown" (Day 2 rule) ----
    categorical_values = {}
    for col in CATEGORICAL_COLUMNS:
        raw_value = job_data.get(col)
        categorical_values[col] = "Unknown" if _is_missing(raw_value) else str(raw_value)

    # ---- Text-derived indicators (same definitions as Day 2) ----
    text_for_scan = (
        cleaned_text["description"] + " " +
        cleaned_text["company_profile"] + " " +
        cleaned_text["requirements"]
    )
    text_indicators = {
        "description_length": len(cleaned_text["description"]),
        "contains_url": int(bool(URL_PATTERN.search(text_for_scan))),
        "contains_email": int(bool(EMAIL_PATTERN.search(text_for_scan))),
    }

    # ---- Frequency encoding (uses the Day 2 fitted lookup tables) ----
    freq_values = {}
    for col in FREQUENCY_ENCODE_COLUMNS:
        freq_map = artifacts["freq_maps"][col]
        category = categorical_values[col]
        # Unseen category (never seen during Day 2 training) -> 0.0,
        # exactly matching Day 2's `.map(freq_map).fillna(0.0)` rule.
        freq_values[f"{col}_freq"] = float(freq_map.get(category, 0.0))

    # ---- One-hot encoding (uses the Day 2 fitted encoder) ----
    one_hot_input = pd.DataFrame(
        [[categorical_values[col] for col in ONE_HOT_COLUMNS]],
        columns=ONE_HOT_COLUMNS,
    )
    one_hot_matrix = artifacts["one_hot_encoder"].transform(one_hot_input)  # shape (1, 66)
    one_hot_cols = artifacts["one_hot_encoder"].get_feature_names_out(ONE_HOT_COLUMNS)
    one_hot_values = dict(zip(one_hot_cols, np.asarray(one_hot_matrix.todense()).ravel()))

    # ---- Assemble all 81 structured features in the EXACT saved order ----
    all_structured_values = {}
    all_structured_values.update({col: binary_values[col] for col in BINARY_COLUMNS})
    all_structured_values.update(missingness)
    all_structured_values.update(text_indicators)
    all_structured_values.update(freq_values)
    all_structured_values.update(one_hot_values)

    structured_columns = artifacts["structured_feature_columns"]
    try:
        structured_row = [all_structured_values[col] for col in structured_columns]
    except KeyError as missing_col:
        raise ValueError(
            f"Could not build structured feature {missing_col} — this usually "
            f"means preprocessing.py's feature definitions have changed since "
            f"this model was trained. Re-run Day 2 and Day 3 to regenerate "
            f"consistent artifacts."
        )
    structured_sparse = sparse.csr_matrix(np.array(structured_row, dtype=np.float64).reshape(1, -1))

    # ---- Combine TF-IDF + structured, in the SAME order used in Day 3 ----
    feature_vector = sparse.hstack([tfidf_vector, structured_sparse]).tocsr()

    if feature_vector.shape[1] != artifacts["expected_feature_count"]:
        raise ValueError(
            f"Feature dimension mismatch: built {feature_vector.shape[1]} features, "
            f"but the trained model expects {artifacts['expected_feature_count']}. "
            f"This usually means the Day 2 artifacts and the Day 3 model were not "
            f"produced from the same preprocessing run."
        )

    return feature_vector, missingness, text_indicators, binary_values, cleaned_text


# ==================================================================
# STEP 4 — RISK SCORE (explicitly NOT a calibrated probability)
# ==================================================================
def decision_score_to_risk_score(decision_value: float) -> float:
    """
    WHAT: Converts the LinearSVC decision_function output (an
          unbounded real number representing signed distance from
          the model's decision boundary) into a 0-100 score.

    IMPORTANT — READ BEFORE USING THIS ELSEWHERE:
    LinearSVC does NOT provide predict_proba(). The number returned
    by decision_function() is NOT a probability — it has no
    statistical calibration guaranteeing that, say, a score of 80
    means "80% chance of fraud". It is simply how far (and in which
    direction) the posting falls from the model's decision boundary.

    We apply a logistic (sigmoid) squashing function purely to
    produce a bounded, monotonic, human-readable 0-100 scale for
    display purposes:
        risk_score = 100 / (1 + e^(-decision_value))
    A decision_value of 0 (exactly on the boundary) maps to 50. More
    positive values (further on the "fraudulent" side) approach 100;
    more negative values (further on the "genuine" side) approach 0.
    This is a MODEL-DERIVED RISK SCORE for interpretation only —
    NOT a calibrated probability of fraud.
    """
    squashed = 1.0 / (1.0 + np.exp(-decision_value))
    return round(float(squashed) * 100, 2)


def risk_score_to_level(risk_score: float) -> str:
    """Maps the 0-100 risk score to a Low/Medium/High label using the
    project's specified cut points: 0-29 Low, 30-59 Medium, 60-100 High."""
    if risk_score <= 29:
        return "Low Risk"
    elif risk_score <= 59:
        return "Medium Risk"
    else:
        return "High Risk"


# ==================================================================
# STEP 5 — EXPLAINABLE RULE-BASED RISK INDICATORS
# ==================================================================
def compute_risk_indicators(missingness: dict, text_indicators: dict, binary_values: dict,
                             cleaned_text: dict) -> list:
    """
    WHAT: Produces a plain-English list of rule-based signals detected
          in this specific posting.
    WHY RULE-BASED, SEPARATE FROM THE MODEL: These are simple,
          transparent checks anyone can verify by eye (unlike the
          3,081-dimensional model, which is not easily inspectable
          feature-by-feature). They give a human reviewer a
          starting point for understanding WHY a posting might look
          suspicious.
    IMPORTANT: Every indicator here is a SIGNAL, not proof. Plenty
          of genuine postings are missing a salary range or lack a
          company logo. These should never be presented to an end
          user as confirmation of fraud.
    """
    indicators = []

    if missingness["missing_company_profile"]:
        indicators.append("Missing company profile")
    if missingness["missing_salary_range"]:
        indicators.append("Missing salary information")
    if missingness["missing_requirements"]:
        indicators.append("Missing job requirements")
    if missingness["missing_benefits"]:
        indicators.append("Missing benefits information")
    if missingness["missing_department"]:
        indicators.append("Missing department information")
    if text_indicators["contains_email"]:
        indicators.append("Contains an email address in the posting text")
    if text_indicators["contains_url"]:
        indicators.append("Contains a URL in the posting text")
    if len(cleaned_text["description"]) < SHORT_DESCRIPTION_THRESHOLD:
        indicators.append(
            f"Unusually short job description (under {SHORT_DESCRIPTION_THRESHOLD} characters)"
        )
    if binary_values.get("telecommuting", 0) == 1:
        indicators.append("Listed as a telecommuting/remote position")
    if binary_values.get("has_company_logo", 0) == 0:
        indicators.append("No company logo provided")
    if binary_values.get("has_questions", 0) == 0:
        indicators.append("No screening questions included")

    return indicators


# ==================================================================
# MAIN PUBLIC FUNCTION
# ==================================================================
def predict_job(job_data: dict, artifacts: dict = None) -> dict:
    """
    WHAT: The main entry point for this script. Takes one job
          posting (as a dict) and returns a full prediction result.

    ARGS:
        job_data:  A dict describing one job posting. Recognized keys
                   (all optional except 'title' and 'description'):
                     title, company_profile, description, requirements,
                     benefits (text);
                     telecommuting, has_company_logo, has_questions
                     (0/1 or True/False);
                     employment_type, required_experience,
                     required_education, industry, function, location,
                     department, salary_range (free-text categories).
        artifacts: Optional pre-loaded artifacts dict (from
                   load_artifacts()). If not provided, this function
                   loads them itself — pass your own if you are calling
                   predict_job() many times in a row (e.g. from Flask
                   later) to avoid reloading the model on every call.

    RETURNS: A dict with:
        predicted_class   : 0 (genuine) or 1 (fraudulent)
        predicted_label   : "Genuine" or "Fraudulent"
        risk_score         : 0-100 model-derived risk score (NOT a probability)
        risk_level          : "Low Risk" / "Medium Risk" / "High Risk"
        risk_indicators    : list of plain-English rule-based signals
        model_name          : name of the model used
        disclaimer          : reminder that this is not a calibrated probability
    """
    validate_job_data(job_data)

    if artifacts is None:
        artifacts = load_artifacts()

    feature_vector, missingness, text_indicators, binary_values, cleaned_text = \
        build_feature_vector(job_data, artifacts)

    model = artifacts["model"]
    predicted_class = int(model.predict(feature_vector)[0])
    decision_value = float(model.decision_function(feature_vector)[0])

    risk_score = decision_score_to_risk_score(decision_value)
    risk_level = risk_score_to_level(risk_score)
    risk_indicators = compute_risk_indicators(missingness, text_indicators, binary_values, cleaned_text)

    return {
        "predicted_class": predicted_class,
        "predicted_label": "Fraudulent" if predicted_class == 1 else "Genuine",
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_indicators": risk_indicators,
        "model_name": artifacts["model_name"],
        "disclaimer": (
            "risk_score is a model-derived, sigmoid-squashed transform of the "
            "SVM decision function for human interpretation only — it is NOT a "
            "calibrated probability of fraud. risk_indicators are rule-based "
            "signals, not proof of fraud."
        ),
    }


# ==================================================================
# MAIN TEST BLOCK
# ==================================================================
if __name__ == "__main__":
    sample_job_posting = {
        "title": "Data Entry Clerk - Work From Home",
        "company_profile": None,  # missing on purpose, to demonstrate a missingness flag
        "description": "Earn money fast! No experience needed. Contact us now.",
        "requirements": None,
        "benefits": None,
        "telecommuting": 1,
        "has_company_logo": 0,
        "has_questions": 0,
        "employment_type": "Full-time",
        "required_experience": "Entry level",
        "required_education": "Unspecified",
        "industry": "Unknown",
        "function": "Unknown",
        "location": "Unknown",
        "department": None,
        "salary_range": None,
    }

    print("=" * 70)
    print("DAY 4 — SAMPLE PREDICTION TEST")
    print("=" * 70)

    result = predict_job(sample_job_posting)

    print(f"\nModel used:            {result['model_name']}")
    print(f"\nModel Prediction:      {result['predicted_label']}")
    print(f"Risk Score:            {result['risk_score']} / 100")
    print(f"Risk Level:            {result['risk_level']}")
    print(f"\nRisk Indicators detected ({len(result['risk_indicators'])}):")
    if result["risk_indicators"]:
        for indicator in result["risk_indicators"]:
            print(f"  - {indicator}")
    else:
        print("  (none detected)")
    print(f"\nDisclaimer: {result['disclaimer']}")
    print("=" * 70)
