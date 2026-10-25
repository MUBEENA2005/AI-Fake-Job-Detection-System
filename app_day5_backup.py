"""
==================================================================
Day 5: Flask Web Application
Project: AI-Based Fake Job Detection and Verification System
==================================================================

WHAT THIS FILE DOES:
A thin Flask web layer around the ALREADY-TRAINED prediction engine
in src/prediction.py. This file contains NO machine-learning logic
of its own — no TF-IDF, no encoding, no risk-score math. Its only
job is:
    1. Show an HTML form (GET /)
    2. Collect what the user typed and call predict_job() (POST /predict)
    3. Show the real, returned result on a result page

WHAT THIS FILE DELIBERATELY DOES NOT DO:
    - It does not retrain or modify the model.
    - It does not recompute TF-IDF, structured features, or the risk
      score — all of that already happens inside predict_job().
    - It does not store submitted postings anywhere (no database).
==================================================================
"""

import os
import sys

from flask import Flask, render_template, request

# ------------------------------------------------------------------
# Make sure the project root (and therefore "src") is importable
# regardless of which directory `python app.py` is launched from.
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prediction import predict_job, load_artifacts  # noqa: E402  (import after sys.path setup is intentional)

app = Flask(__name__)

# ------------------------------------------------------------------
# Load the Day 2 / Day 3 artifacts (TF-IDF vectorizer, encoders, the
# trained model) ONCE when the server starts, instead of on every
# request. predict_job() supports this via its optional `artifacts`
# argument (see src/prediction.py). If loading fails — e.g. Day 2/3
# haven't been run yet — the app still starts and the form still
# loads; only the /predict route shows a friendly error until the
# missing files exist.
# ------------------------------------------------------------------
try:
    ARTIFACTS = load_artifacts()
    ARTIFACTS_ERROR = None
    print("ML artifacts loaded successfully. Model:", ARTIFACTS["model_name"])
except Exception as exc:  # noqa: BLE001 - deliberately broad: this is a startup safety net
    ARTIFACTS = None
    ARTIFACTS_ERROR = str(exc)
    print("WARNING: Could not load ML artifacts at startup.")
    print(ARTIFACTS_ERROR)

# The exact field names predict_job() expects (see src/prediction.py).
# app.py must not invent alternate names for these.
TEXT_FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]
CATEGORY_FIELDS = [
    "location", "department", "salary_range", "employment_type",
    "required_experience", "required_education", "industry", "function",
]
BINARY_FIELDS = ["telecommuting", "has_company_logo", "has_questions"]

# Dropdown choices, taken directly from the categories the OneHotEncoder
# was actually fitted on during Day 2 — not invented values. Users can
# also leave these blank; a blank value is treated as missing (see
# build_job_data_from_form below), which predict_job() maps to "Unknown",
# the same way Day 2 handled genuinely missing categories.
EMPLOYMENT_TYPE_OPTIONS = ["Full-time", "Part-time", "Contract", "Temporary", "Other"]
REQUIRED_EXPERIENCE_OPTIONS = [
    "Internship", "Entry level", "Associate", "Mid-Senior level",
    "Director", "Executive", "Not Applicable",
]
REQUIRED_EDUCATION_OPTIONS = [
    "Some High School Coursework", "High School or equivalent", "Vocational",
    "Vocational - HS Diploma", "Vocational - Degree", "Certification",
    "Some College Coursework Completed", "Associate Degree", "Bachelor's Degree",
    "Master's Degree", "Professional", "Doctorate", "Unspecified",
]
FUNCTION_OPTIONS = [
    "Accounting/Auditing", "Administrative", "Advertising", "Art/Creative",
    "Business Analyst", "Business Development", "Consulting", "Customer Service",
    "Data Analyst", "Design", "Distribution", "Education", "Engineering",
    "Finance", "Financial Analyst", "General Business", "Health Care Provider",
    "Human Resources", "Information Technology", "Legal", "Management",
    "Manufacturing", "Marketing", "Other", "Product Management", "Production",
    "Project Management", "Public Relations", "Purchasing", "Quality Assurance",
    "Research", "Sales", "Science", "Strategy/Planning", "Supply Chain",
    "Training", "Writing/Editing",
]

DROPDOWN_OPTIONS = {
    "employment_type": EMPLOYMENT_TYPE_OPTIONS,
    "required_experience": REQUIRED_EXPERIENCE_OPTIONS,
    "required_education": REQUIRED_EDUCATION_OPTIONS,
    "function": FUNCTION_OPTIONS,
}


def build_job_data_from_form(form) -> dict:
    """
    WHAT: Converts the submitted HTML form into the dictionary shape
          predict_job() expects.

    IMPORTANT INTEGRATION DECISION (why blank fields become None,
    not ""): predict_job() treats a field as "missing" only if it is
    absent or None (see prediction.py's _is_missing helper) — an
    empty string is treated as "provided but blank". When a person
    leaves a field blank in this web form, the correct interpretation
    is "missing", exactly like a blank cell in the original training
    CSV, so that indicators such as "Missing company profile" trigger
    the same way they did during Day 2/3. We therefore convert
    blank/whitespace-only form values to None here, in app.py, WITHOUT
    changing anything inside prediction.py itself.
    """
    job_data = {}

    for field in TEXT_FIELDS + CATEGORY_FIELDS:
        value = form.get(field, "").strip()
        job_data[field] = value if value else None

    # HTML checkboxes only appear in submitted form data when checked,
    # so a missing key means "unchecked" -> 0; a present key (normally
    # "on") means "checked" -> 1.
    for field in BINARY_FIELDS:
        job_data[field] = 1 if form.get(field) else 0

    return job_data


@app.route("/", methods=["GET"])
def index():
    """Renders the empty job-analysis form."""
    return render_template(
        "index.html",
        error=None,
        form_data={},
        dropdown_options=DROPDOWN_OPTIONS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Collects the submitted form, validates it, calls the EXISTING
    predict_job() function, and shows the real result. No prediction
    logic lives here — this route only adapts HTTP form data into the
    dict predict_job() already knows how to consume.
    """
    title = request.form.get("title", "").strip()
    description = request.form.get("description", "").strip()

    # Friendly, specific validation messages. The form is re-rendered
    # with whatever the user already typed (form_data=request.form)
    # so they don't have to start over.
    if not title:
        return render_template(
            "index.html",
            error="Job title is required.",
            form_data=request.form,
            dropdown_options=DROPDOWN_OPTIONS,
        )
    if not description:
        return render_template(
            "index.html",
            error="Job description is required.",
            form_data=request.form,
            dropdown_options=DROPDOWN_OPTIONS,
        )

    if ARTIFACTS is None:
        return render_template(
            "index.html",
            error=(
                "The prediction engine is not available right now (ML "
                "artifacts failed to load at startup). Make sure Day 2 "
                "(src/preprocessing.py) and Day 3 (src/model_training.py) "
                "have both been run successfully, then restart the app."
            ),
            form_data=request.form,
            dropdown_options=DROPDOWN_OPTIONS,
        )

    job_data = build_job_data_from_form(request.form)

    try:
        result = predict_job(job_data, artifacts=ARTIFACTS)
    except Exception as exc:  # noqa: BLE001 - never show a raw traceback to the user
        print("ERROR while predicting:", repr(exc))
        return render_template(
            "index.html",
            error="Something went wrong while analyzing this posting. Please check your inputs and try again.",
            form_data=request.form,
            dropdown_options=DROPDOWN_OPTIONS,
        )

    return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(debug=False)
