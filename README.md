# AI-Based Fake Job Detection and Verification System

A final-year engineering capstone project that combines a trained
machine learning model with rule-based company verification checks to
help job seekers spot potentially fraudulent job postings.

---

## Problem Statement

Online job platforms host a large volume of postings, and a small but
consistent share of them are fraudulent — designed to harvest personal
information, extract upfront fees, or otherwise scam applicants.
Fraudulent postings often share subtle patterns (missing company
details, vague descriptions, mismatched contact information) that are
tedious for a person to check manually across many listings. This
project automates that first pass of screening.

## Objective

Build a complete, working, end-to-end system that:
1. Learns to distinguish genuine from fraudulent job postings from a
   real, labeled dataset.
2. Gives a job seeker a plain-language risk assessment for any posting
   they paste in.
3. Independently checks the supplied company name, website, and email
   for simple, explainable warning signs.
4. Keeps a local history of past analyses.
5. Presents all of this through a clean, understandable web interface.

**Important limitation, stated up front:** this system provides an
automated **risk assessment**, not a verdict. It cannot guarantee that
any specific posting is genuinely fraudulent or genuinely legitimate.
Every score it produces should be treated as one input among several,
not a final decision.

---

## Key Features

- Machine learning–based fraud risk scoring (0–100) with a Low /
  Medium / High risk level and plain-English reasons
- Independent, rule-based company verification (name, website, email)
- A history of past analyses stored locally in SQLite
- A clean, responsive Flask web interface — no page needs JavaScript
  to function
- Every score is clearly labeled as an automated assessment, never as
  a guarantee or a calibrated probability

---

## System Workflow

```
Home page
  → Enter job details (and, optionally, company details)
  → Click "Analyze Job Posting"
  → ML model scores the posting text and metadata
  → Company verification checks the supplied company details
  → Both results are combined on one result page (never merged into
    a single score)
  → The analysis summary is saved to the local history
  → From the result page: analyze another job, or view history
```

### Architecture (viva-ready summary)

```
Browser
   │
   ▼
Flask app.py  ──────────────► adapts HTTP form data only
   │                           (no ML/verification logic lives here)
   ├──► src/prediction.py ───► loads fitted TF-IDF vectorizer,
   │        │                  encoders, and the trained Linear SVM;
   │        │                  rebuilds features for the new posting;
   │        ▼                  returns prediction + risk score/level
   │    (uses artifacts saved by preprocessing.py / model_training.py,
   │     never retrains anything at request time)
   │
   ├──► src/company_verification.py ─► structural checks only
   │        │                          (no web requests, no APIs)
   │        ▼                          returns verification score/level
   │
   └──► src/database.py ──────────────► saves a short summary to
                                         SQLite (analysis_history)

Result page ◄── both results displayed side by side, never combined
History page ◄── src/database.py's get_analysis_history()
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3 |
| Web framework | Flask |
| ML library | scikit-learn (TF-IDF, OneHotEncoder, LinearSVC, LogisticRegression, RandomForestClassifier) |
| Data handling | pandas, NumPy, SciPy (sparse matrices) |
| Model/artifact persistence | joblib |
| Evaluation visualization | matplotlib |
| Database | SQLite via Python's built-in `sqlite3` module — no ORM |
| Frontend | Jinja2 templates, hand-written CSS (Inter font with a full system fallback, no CDN, no framework) |

No unnecessary dependencies were added — `requirements.txt` lists only
packages actually imported somewhere in the codebase.

---

## Dataset

- Source file: `data/raw/fake_job_postings.csv`
- **17,880** job postings, **18** columns
- Target column: `fraudulent` (0 = genuine, 1 = fraudulent)
- Class balance: **17,014 genuine (95.16%)** vs **866 fraudulent
  (4.84%)** — a significantly imbalanced dataset, which directly
  shaped the modeling and evaluation approach below
- Full column-by-column statistics (missing values, data types,
  cardinality) are in [`data/DATASET_INFO.md`](data/DATASET_INFO.md)

---

## Machine Learning Approach

### Preprocessing (`src/preprocessing.py`)
- Missing values handled explicitly per column type: text → empty
  string, categorical → `"Unknown"`, binary → `0`. No rows or columns
  are dropped, since the missing-salary/missing-department columns
  alone would have discarded 60–84% of the dataset.
- Text is conservatively cleaned (lowercased, HTML stripped,
  whitespace collapsed) — numbers, URLs, emails, and currency symbols
  are deliberately preserved as potential fraud signals.
- `title`, `company_profile`, `description`, `requirements`, and
  `benefits` are combined into one `combined_text` field.

### TF-IDF (text features)
- `max_features=3000`, `ngram_range=(1, 2)` (unigrams + bigrams),
  English stop words removed
- Fitted **only on the training split** and then applied to the full
  dataset, to prevent leakage into the features used for evaluation

### Structured features
- Low-cardinality categorical columns (`employment_type`,
  `required_experience`, `required_education`, `function`) are
  one-hot encoded
- High-cardinality columns (`industry`, `location`, `department`,
  `salary_range`) are frequency-encoded instead, to avoid an
  unmanageable number of one-hot columns
- Missingness itself is kept as a feature (`missing_company_profile`,
  `missing_salary_range`, etc.), plus a few simple text-derived
  indicators (`description_length`, `contains_url`, `contains_email`)
- **Total: 3,000 TF-IDF features + 81 structured features = 3,081
  features**

### Model comparison (`src/model_training.py`)
Three models were trained on an 80/20 stratified split (14,304 train
/ 3,576 test rows, `random_state=42`), all using `class_weight="balanced"`
instead of resampling (no SMOTE) to handle the class imbalance:

| Model | Accuracy | Precision (fraud) | Recall (fraud) | F1 (fraud) |
|---|---|---|---|---|
| Logistic Regression | 0.937 | 0.427 | 0.879 | 0.575 |
| Random Forest | 0.979 | 1.000 | 0.572 | 0.728 |
| **Linear SVM (LinearSVC)** | **0.980** | **0.750** | **0.867** | **0.804** |

**Linear SVM was selected**, primarily on fraudulent-class F1-score
(not overall accuracy, which is misleading on a 95/5 imbalanced
dataset — a model predicting "genuine" for everything would already
score ~95% accuracy while catching zero fraud).

### Risk scoring (`src/prediction.py`)
`LinearSVC` has no `predict_proba()`. Its `decision_function()` output
(signed distance from the decision boundary) is passed through a
sigmoid squash into a 0–100 scale for display purposes only:

```
risk_score = 100 / (1 + e^(-decision_value))
```

This is explicitly documented and labeled everywhere in the UI as a
**model-derived risk score, not a calibrated probability**.

**Risk levels:** 0–29 = Low Risk · 30–59 = Medium Risk · 60–100 = High Risk

---

## Company Verification (`src/company_verification.py`)

A completely separate, rule-based module — no web scraping, no
external API calls, no network access. Given a company name, website,
and application email, it checks:

- Whether each field was provided
- Basic structural validity of the website URL and email address
- Whether the email uses a free/public provider (Gmail, Yahoo, etc.)
- Whether the website domain and email domain match

Each signal contributes explainable risk points to a separate 0–100
**verification risk score** (same Low/Medium/High thresholds as the ML
score, for a consistent experience) — never merged with the ML score.

---

## Analysis History (`src/database.py`)

- SQLite database at `database/fake_job_detection.db`, accessed only
  through Python's built-in `sqlite3` — no ORM, no extra dependency
- Table `analysis_history` stores a short **summary** of each
  analysis (job title, company info, prediction, both scores, domain
  comparison) — never the full job description, and never TF-IDF
  vectors or other large model data
- All queries use parameterized `?` placeholders — no string-built SQL
- If saving to the database ever fails, the error is logged to the
  terminal and the result page is still shown; a database problem
  never blocks the user from seeing their analysis

---

## Web Application (`app.py`)

Three routes, all thin adapters with no ML/verification/SQL logic of
their own:

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Shows the job analysis form |
| `/predict` | POST | Validates input, calls `predict_job()` and `verify_company()`, saves a summary, shows the result |
| `/history` | GET | Shows every saved analysis, newest first |

The ML artifacts and the database are both initialized **once** at
server startup, not on every request.

---

## Project Structure

```
AI-Fake-Job-Detection-System/
├── app.py                        # Flask web application
├── requirements.txt
├── README.md
├── data/
│   ├── raw/fake_job_postings.csv
│   ├── processed/                # TF-IDF matrix, encoders, split indices
│   └── DATASET_INFO.md
├── models/
│   ├── best_model.joblib
│   ├── model_metadata.json
│   └── best_model_confusion_matrix.png
├── database/
│   └── fake_job_detection.db     # created automatically on first run
├── src/
│   ├── data_exploration.py       # Day 1
│   ├── preprocessing.py          # Day 2
│   ├── model_training.py         # Day 3
│   ├── prediction.py             # Day 4
│   ├── company_verification.py   # Day 5
│   └── database.py               # Day 6
├── templates/                    # Day 7
│   ├── index.html
│   ├── result.html
│   └── history.html
└── static/
    └── style.css
```

---

## How to Install

```bash
# from the project root
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run

The ML pipeline only needs to be run once (or whenever the raw dataset
changes) — its output is saved to disk and reused by the web app:

```bash
python src/preprocessing.py     # builds data/processed/ (TF-IDF, encoders, split)
python src/model_training.py    # trains and saves models/best_model.joblib
python app.py                   # starts the web app
```

Then open **http://127.0.0.1:5000** in a browser.

## How to Test

Standalone module tests (no Flask needed):
```bash
python src/data_exploration.py
python src/company_verification.py
python src/database.py
```

Full application testing checklist — with `python app.py` running:

| # | Scenario | Expected result |
|---|---|---|
| 1 | Complete posting, matching website/email domains | ML result + verification both shown, domains reported as a match, record saved to history |
| 2 | Company uses Gmail, no website | ML still runs; verification flags the free email provider |
| 3 | Website and email domains differ | Domain comparison clearly shown as a mismatch |
| 4 | All company fields left blank | ML prediction still works; app does not crash; verification shows all fields as "not provided" |
| 5 | History with multiple records | Newest analysis appears first |
| 6 | Empty history | Friendly "No analysis history available." message |
| 7 | Missing job title or description | Friendly validation message, no crash, previously entered fields preserved |

## Example Workflow

1. Open the home page, read the short explanation of what the system does
2. Paste in a job title and description (required); optionally add
   company details and other job metadata
3. Click **Analyze Job Posting**
4. Read the result: overall ML assessment, why it was flagged, the
   separate company verification panel, which model was used, and the
   disclaimer
5. Click **View Analysis History** to see all past analyses, or
   **Analyze another job** to check another posting

---

## Limitations

- The system provides a risk assessment and verification indicators.
  **It cannot guarantee that a job posting is genuinely fraudulent or
  genuine.**
- The ML model was trained on one historical, English-language
  dataset; it may not generalize equally well to postings very
  different in style, industry, or language from that dataset.
- Company verification performs only structural checks on
  user-supplied text — it does not contact the company, visit the
  website, or confirm real-world business registration.
- Domain comparison is an exact string match; a legitimate subdomain
  (e.g. `careers.example.com` vs. `example.com`) is currently reported
  as a mismatch.
- The risk score is a sigmoid-squashed transform of the model's
  decision function, not a calibrated probability of fraud.

## Future Enhancements

- Calibrated probability output (e.g. Platt scaling) instead of the
  current sigmoid-squashed decision-function score
- Root-domain-aware comparison (so legitimate subdomains aren't
  flagged as mismatches)
- Optional company registry/API verification, clearly separated from
  the current offline structural checks
- User accounts and authentication for a multi-user deployment
- Pagination and filtering on the history page as records grow

---

*This project was built and tested stage by stage, with every
statistic in this document coming from an actual run of the
corresponding script against the real dataset — nothing here is
estimated or fabricated.*
