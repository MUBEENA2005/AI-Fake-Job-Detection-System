"""
==================================================================
AI-Based Fake Job Detection and Verification System
Flask Web Application
==================================================================

WORKFLOW:

    Open Application
          ↓
       Login
          ↓
    Job Analysis Form
          ↓
     ML Prediction
          ↓
  Company Verification
          ↓
      Result Page
          ↓
   SQLite Analysis History

Registration is available from the Login page.

IMPORTANT:
- User registration/login is handled through SQLite.
- Passwords are stored as secure hashes.
- ML logic remains inside src/prediction.py.
- Company verification remains inside src/company_verification.py.
- Database logic remains inside src/database.py.
==================================================================
"""

import os
import sys
from functools import wraps

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
)

from dotenv import load_dotenv


# ==================================================================
# PROJECT ROOT
# ==================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==================================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================================

load_dotenv()


# ==================================================================
# IMPORT EXISTING PROJECT MODULES
# ==================================================================

from src.prediction import (  # noqa: E402
    predict_job,
    load_artifacts,
)

from src.company_verification import (  # noqa: E402
    verify_company,
)

from src.database import (  # noqa: E402
    initialize_database,
    register_user,
    verify_user,
    get_user_by_id,
    save_analysis,
    get_analysis_history,
)


# ==================================================================
# FLASK APPLICATION
# ==================================================================

app = Flask(__name__)


# ==================================================================
# SECRET KEY
# ==================================================================

app.secret_key = os.getenv(
    "SECRET_KEY",
    "local-development-secret-key-change-before-production"
)


# ==================================================================
# DATABASE INITIALIZATION
# ==================================================================

try:

    initialize_database()

    print(
        "Database initialized successfully at "
        "database/fake_job_detection.db"
    )

except Exception as exc:

    print(
        "WARNING: Could not initialize the database."
    )

    print(
        repr(exc)
    )


# ==================================================================
# ML ARTIFACT LOADING
# ==================================================================

try:

    ARTIFACTS = load_artifacts()

    ARTIFACTS_ERROR = None

    print(
        "ML artifacts loaded successfully. Model:",
        ARTIFACTS["model_name"]
    )

except Exception as exc:

    ARTIFACTS = None

    ARTIFACTS_ERROR = str(exc)

    print(
        "WARNING: Could not load ML artifacts at startup."
    )

    print(
        ARTIFACTS_ERROR
    )


# ==================================================================
# FIELD DEFINITIONS
# ==================================================================

TEXT_FIELDS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]


CATEGORY_FIELDS = [
    "location",
    "department",
    "salary_range",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]


BINARY_FIELDS = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
]


COMPANY_VERIFICATION_FIELDS = [
    "company_name",
    "company_website",
    "application_email",
]


# ==================================================================
# DROPDOWN OPTIONS
# ==================================================================

EMPLOYMENT_TYPE_OPTIONS = [
    "Full-time",
    "Part-time",
    "Contract",
    "Temporary",
    "Other",
]


REQUIRED_EXPERIENCE_OPTIONS = [
    "Internship",
    "Entry level",
    "Associate",
    "Mid-Senior level",
    "Director",
    "Executive",
    "Not Applicable",
]


REQUIRED_EDUCATION_OPTIONS = [
    "Some High School Coursework",
    "High School or equivalent",
    "Vocational",
    "Vocational - HS Diploma",
    "Vocational - Degree",
    "Certification",
    "Some College Coursework Completed",
    "Associate Degree",
    "Bachelor's Degree",
    "Master's Degree",
    "Professional",
    "Doctorate",
    "Unspecified",
]


FUNCTION_OPTIONS = [
    "Accounting/Auditing",
    "Administrative",
    "Advertising",
    "Art/Creative",
    "Business Analyst",
    "Business Development",
    "Consulting",
    "Customer Service",
    "Data Analyst",
    "Design",
    "Distribution",
    "Education",
    "Engineering",
    "Finance",
    "Financial Analyst",
    "General Business",
    "Health Care Provider",
    "Human Resources",
    "Information Technology",
    "Legal",
    "Management",
    "Manufacturing",
    "Marketing",
    "Other",
    "Product Management",
    "Production",
    "Project Management",
    "Public Relations",
    "Purchasing",
    "Quality Assurance",
    "Research",
    "Sales",
    "Science",
    "Strategy/Planning",
    "Supply Chain",
    "Training",
    "Writing/Editing",
]


DROPDOWN_OPTIONS = {
    "employment_type": EMPLOYMENT_TYPE_OPTIONS,
    "required_experience": REQUIRED_EXPERIENCE_OPTIONS,
    "required_education": REQUIRED_EDUCATION_OPTIONS,
    "function": FUNCTION_OPTIONS,
}


# ==================================================================
# LOGIN REQUIRED DECORATOR
# ==================================================================

def login_required(view_function):
    """
    Protects pages that require a logged-in user.
    """

    @wraps(view_function)
    def wrapped_view(*args, **kwargs):

        user_id = session.get("user_id")

        if not user_id:

            flash(
                "Please login to access the Fake Job Detection System.",
                "warning"
            )

            return redirect(
                url_for("login")
            )

        user = get_user_by_id(user_id)

        if user is None:

            session.clear()

            flash(
                "Your account could not be found. Please login again.",
                "warning"
            )

            return redirect(
                url_for("login")
            )

        return view_function(
            *args,
            **kwargs
        )

    return wrapped_view


# ==================================================================
# ROOT ROUTE
# ==================================================================

@app.route("/")
def home():
    """
    Application entry point.

    If the user is not logged in:
        show Login page.

    If the user is already logged in:
        go to the Job Analysis page.
    """

    if session.get("user_id"):

        return redirect(
            url_for("index")
        )

    return redirect(
        url_for("login")
    )


# ==================================================================
# REGISTER ROUTE
# ==================================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():
    """
    Creates a new user account.
    """

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        # ----------------------------------------------------------
        # BASIC VALIDATION
        # ----------------------------------------------------------

        if not username or not email or not password:

            flash(
                "Please fill in all required fields.",
                "warning"
            )

            return render_template(
                "register.html"
            )

        # ----------------------------------------------------------
        # PASSWORD CONFIRMATION
        # ----------------------------------------------------------

        if password != confirm_password:

            flash(
                "Passwords do not match.",
                "warning"
            )

            return render_template(
                "register.html"
            )

        # ----------------------------------------------------------
        # PASSWORD LENGTH
        # ----------------------------------------------------------

        if len(password) < 6:

            flash(
                "Password must contain at least 6 characters.",
                "warning"
            )

            return render_template(
                "register.html"
            )

        # ----------------------------------------------------------
        # CREATE USER
        # ----------------------------------------------------------

        user_id = register_user(
            username,
            email,
            password
        )

        if user_id is None:

            flash(
                "Username or email already exists. "
                "Please choose another one.",
                "warning"
            )

            return render_template(
                "register.html"
            )

        # ----------------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------------

        flash(
            "Account created successfully. "
            "Please login using your username or email.",
            "success"
        )

        return redirect(
            url_for("login")
        )

    return render_template(
        "register.html"
    )


# ==================================================================
# LOGIN ROUTE
# ==================================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():
    """
    Login page.

    User can login using:
        - username
        - email
    """

    if request.method == "POST":

        login_value = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        # ----------------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------------

        if not login_value or not password:

            return render_template(
                "login.html",
                error="Please enter your username/email and password."
            )

        # ----------------------------------------------------------
        # VERIFY USER
        # ----------------------------------------------------------

        user = verify_user(
            login_value,
            password
        )

        if user is None:

            return render_template(
                "login.html",
                error="Invalid username/email or password. Please try again."
            )

        # ----------------------------------------------------------
        # LOGIN SUCCESS
        # ----------------------------------------------------------

        session.clear()

        session["user_id"] = user["id"]
        session["username"] = user["username"]
        session["email"] = user["email"]

        flash(
            "Login successful. Welcome, "
            + user["username"]
            + "!",
            "success"
        )

        return redirect(
            url_for("index")
        )

    # --------------------------------------------------------------
    # IMPORTANT:
    # Do NOT redirect GET /login based on an existing session.
    #
    # This allows the Login page to actually open when the user
    # clicks Login.
    # --------------------------------------------------------------

    return render_template(
        "login.html",
        error=None
    )


# ==================================================================
# LOGOUT
# ==================================================================

@app.route("/logout")
def logout():
    """
    Clears the login session.
    """

    session.clear()

    flash(
        "You have been logged out successfully.",
        "success"
    )

    return redirect(
        url_for("login")
    )


# ==================================================================
# BUILD JOB DATA
# ==================================================================

def build_job_data_from_form(form) -> dict:
    """
    Converts HTML form data into the format expected by predict_job().
    """

    job_data = {}

    # --------------------------------------------------------------
    # TEXT AND CATEGORY FIELDS
    # --------------------------------------------------------------

    for field in TEXT_FIELDS + CATEGORY_FIELDS:

        value = form.get(
            field,
            ""
        ).strip()

        job_data[field] = (
            value
            if value
            else None
        )

    # --------------------------------------------------------------
    # CHECKBOX FIELDS
    # --------------------------------------------------------------

    for field in BINARY_FIELDS:

        job_data[field] = (
            1
            if form.get(field)
            else 0
        )

    return job_data


# ==================================================================
# JOB ANALYSIS PAGE
# ==================================================================

@app.route(
    "/index",
    methods=["GET"]
)
@login_required
def index():
    """
    Displays the protected job analysis page.
    """

    return render_template(
        "index.html",
        error=None,
        form_data={},
        dropdown_options=DROPDOWN_OPTIONS,
    )


# ==================================================================
# PREDICT
# ==================================================================

@app.route(
    "/predict",
    methods=["POST"]
)
@login_required
def predict():
    """
    Performs:

        1. ML prediction
        2. Company verification
        3. Database storage
        4. Result display
    """

    title = request.form.get(
        "title",
        ""
    ).strip()

    description = request.form.get(
        "description",
        ""
    ).strip()

    # --------------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # ML ARTIFACT CHECK
    # --------------------------------------------------------------

    if ARTIFACTS is None:

        return render_template(
            "index.html",
            error=(
                "The prediction engine is not available right now. "
                "Please check the ML artifacts and restart the application."
            ),
            form_data=request.form,
            dropdown_options=DROPDOWN_OPTIONS,
        )

    # --------------------------------------------------------------
    # BUILD JOB DATA
    # --------------------------------------------------------------

    job_data = build_job_data_from_form(
        request.form
    )

    # --------------------------------------------------------------
    # ML PREDICTION
    # --------------------------------------------------------------

    try:

        result = predict_job(
            job_data,
            artifacts=ARTIFACTS
        )

    except Exception as exc:

        print(
            "ERROR while predicting:",
            repr(exc)
        )

        return render_template(
            "index.html",
            error=(
                "Something went wrong while analyzing this posting. "
                "Please check your inputs and try again."
            ),
            form_data=request.form,
            dropdown_options=DROPDOWN_OPTIONS,
        )

    # --------------------------------------------------------------
    # COMPANY VERIFICATION
    # --------------------------------------------------------------

    company_name = (
        request.form.get(
            "company_name",
            ""
        ).strip()
        or None
    )

    company_website = (
        request.form.get(
            "company_website",
            ""
        ).strip()
        or None
    )

    application_email = (
        request.form.get(
            "application_email",
            ""
        ).strip()
        or None
    )

    try:

        verification_result = verify_company(
            company_name,
            company_website,
            application_email,
        )

    except Exception as exc:

        print(
            "ERROR during company verification:",
            repr(exc)
        )

        verification_result = {
            "company_name_provided": False,
            "website_provided": False,
            "website_valid_structure": False,
            "website_domain": None,
            "email_provided": False,
            "email_valid_structure": False,
            "email_domain": None,
            "is_free_email_provider": False,
            "domain_comparison": "not_comparable",
            "verification_score": 0,
            "verification_level": "Low Risk",
            "verification_findings": [
                "Company verification could not be completed "
                "due to an internal error."
            ],
            "disclaimer": (
                "Company verification could not be completed "
                "for this submission."
            ),
        }

    # --------------------------------------------------------------
    # SAVE ANALYSIS
    # --------------------------------------------------------------

    try:

        save_analysis(
            job_title=title,
            predicted_label=result["predicted_label"],
            ml_risk_score=result["risk_score"],
            ml_risk_level=result["risk_level"],
            company_name=company_name,
            company_website=company_website,
            application_email=application_email,
            verification_score=verification_result[
                "verification_score"
            ],
            verification_level=verification_result[
                "verification_level"
            ],
            domain_comparison=verification_result[
                "domain_comparison"
            ],
        )

    except Exception as exc:

        print(
            "ERROR while saving analysis to database:",
            repr(exc)
        )

    # --------------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------------

    return render_template(
        "result.html",
        result=result,
        verification_result=verification_result,
    )


# ==================================================================
# HISTORY
# ==================================================================

@app.route("/history")
@login_required
def history():
    """
    Displays previously saved analyses.
    """

    try:

        records = get_analysis_history()

    except Exception as exc:

        print(
            "ERROR while loading history:",
            repr(exc)
        )

        records = []

        flash(
            "Analysis history could not be loaded.",
            "warning"
        )

    return render_template(
        "history.html",
        history=records,
    )


# ==================================================================
# RUN APPLICATION
# ==================================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )