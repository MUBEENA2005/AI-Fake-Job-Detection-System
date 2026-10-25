# Dataset Information

## Dataset Name
`fake_job_postings.csv`

Location: `data/raw/fake_job_postings.csv`

## Purpose
This dataset is used to train and evaluate a machine learning model that
distinguishes **genuine job postings** from **fraudulent (fake) job
postings**, as part of the *AI-Based Fake Job Detection and Verification
System* final-year project.

## Dataset Size
- **Number of records (rows):** 17,880
- **Number of features (columns):** 18

## Column Names and Data Types

| Column | Data Type | Notes |
|---|---|---|
| job_id | int64 | Unique identifier per posting |
| title | text | Job title |
| location | text/categorical | Country, state, city (high cardinality: 3,105 unique values) |
| department | text/categorical | 64.58% missing |
| salary_range | text/categorical | 83.96% missing |
| company_profile | text | NLP candidate; 18.50% missing |
| description | text | NLP candidate; 0.01% missing (nearly complete) |
| requirements | text | NLP candidate; 15.08% missing |
| benefits | text | NLP candidate; 40.34% missing |
| telecommuting | int64 (binary) | 0 = No, 1 = Yes |
| has_company_logo | int64 (binary) | 0 = No, 1 = Yes |
| has_questions | int64 (binary) | 0 = No, 1 = Yes |
| employment_type | categorical | Full-time / Part-time / Contract / Temporary / Other; 19.41% missing |
| required_experience | categorical | e.g. Entry level, Mid-Senior level; 39.43% missing |
| required_education | categorical | e.g. Bachelor's Degree; 45.33% missing |
| industry | categorical | 27.42% missing |
| function | categorical | 36.10% missing |
| **fraudulent** | int64 (binary) | **Target column.** 0 = Genuine, 1 = Fraudulent |

## Target Variable
- **Column name:** `fraudulent`
- **Meaning of values:**
  - `0` → Genuine job posting
  - `1` → Fraudulent (fake) job posting

## Class Distribution (Target Balance)
| Class | Count | Percentage |
|---|---|---|
| 0 (Genuine) | 17,014 | 95.16% |
| 1 (Fraudulent) | 866 | 4.84% |

**This dataset is highly imbalanced.** Genuine postings vastly outnumber
fraudulent ones, which reflects real-world conditions (fraud is rare) but
means that standard accuracy is NOT a reliable evaluation metric for this
project. Precision, recall, F1-score, and/or techniques like class
weighting or resampling (e.g. SMOTE) should be considered during model
development.

## Important Text Fields (for NLP feature engineering)
- `title`
- `company_profile`
- `description`
- `requirements`
- `benefits`

These are free-form natural-language fields and are strong candidates for
NLP techniques such as TF-IDF or word embeddings.

## Important Metadata Fields
- `telecommuting`, `has_company_logo`, `has_questions` — binary flags,
  directly usable as model features without encoding.
- `employment_type`, `required_experience`, `required_education`,
  `industry`, `function` — categorical fields that will likely need
  encoding (e.g. one-hot encoding) in the preprocessing stage.
- `location` — high-cardinality text field; may need to be parsed into
  country/state/city components before use.

## Fraud-Risk Candidate Columns (for future investigation, not yet tested)
`has_company_logo`, `has_questions`, `telecommuting`, `company_profile`,
`salary_range`, `employment_type`, `required_education`,
`required_experience`

These are flagged based on domain intuition about common scam-posting
patterns (e.g. missing company profile, no logo). Their actual predictive
value has **not** been statistically tested yet — that analysis belongs to
Day 2 (preprocessing/feature engineering), not this Day 1 exploration.

## Missing-Data Observations
Several columns have substantial missing data:

| Column | Missing % |
|---|---|
| salary_range | 83.96% |
| department | 64.58% |
| required_education | 45.33% |
| benefits | 40.34% |
| required_experience | 39.43% |
| function | 36.10% |
| industry | 27.42% |
| employment_type | 19.41% |
| company_profile | 18.50% |
| requirements | 15.08% |
| location | 1.94% |
| description | 0.01% |

All other columns (`job_id`, `title`, `telecommuting`, `has_company_logo`,
`has_questions`, `fraudulent`) have **0% missing values**.

## Duplicate-Data Observations
- **Fully duplicated rows found:** 0

No exact duplicate records exist in the dataset.

## Limitations and Imbalance Issues
1. **Severe class imbalance** (95.16% vs 4.84%) — must be accounted for
   during model evaluation and possibly during training.
2. **High missingness** in several potentially useful columns
   (`salary_range`, `department`) — these may require dropping,
   imputing, or converting "missingness" itself into a binary feature.
3. **High-cardinality categorical fields** (`location`, `department`) —
   will need careful encoding to avoid an explosion of one-hot columns.
4. No statistical relationship between any feature and the target has
   been tested yet — all statistics above are purely descriptive
   (Day 1 scope only). No correlation, feature importance, or modeling
   work has been performed.

---
*All statistics in this document were calculated directly from the
actual `fake_job_postings.csv` file using `src/data_exploration.py`.
No values were fabricated or assumed.*
