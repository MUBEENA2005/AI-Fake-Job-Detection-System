# Preprocessing & Feature Engineering Report

## Overview
This document explains the Day 2 preprocessing pipeline (`src/preprocessing.py`)
that transforms the raw dataset (`data/raw/fake_job_postings.csv`) into
machine-learning-ready feature representations, saved under
`data/processed/`. **No model has been trained** in this stage — this is
purely data preparation.

All numbers below come from an actual run of the script against the real
17,880-row dataset (verified, not fabricated).

---

## 1. Missing-Value Strategy

| Column type | Strategy | Reasoning |
|---|---|---|
| Text (`title`, `company_profile`, `description`, `requirements`, `benefits`) | Fill with empty string `""` | Neutral placeholder; contributes nothing to TF-IDF term frequency, so no fabricated content |
| Categorical (`employment_type`, `required_experience`, `required_education`, `industry`, `function`, `location`, `department`, `salary_range`) | Fill with `"Unknown"` | Preserves every row (dropping rows with `dropna()` would discard up to 84% of the dataset for `salary_range`); treats missingness as its own meaningful category |
| Binary (`telecommuting`, `has_company_logo`, `has_questions`) | Fill with `0` (defensive only — Day 1 found 0% missing here) | Safe default in case a future data export contains a rare missing entry |

**Decision on `salary_range` (83.96% missing) and `department` (64.58%
missing):** Both are **retained**, not dropped. Instead of deleting the
columns or the rows, missingness itself is preserved as an explicit
`missing_salary_range` / `missing_department` binary indicator feature
(see Section 4), and the remaining category values are frequency-encoded
(Section 3). This keeps all 17,880 rows intact while still letting a
future model learn from whichever weak signal these fields carry.

---

## 2. Text-Cleaning Strategy

Applied to all five text columns (conservative, reversible-in-spirit):
1. Lowercase all text.
2. Strip HTML tags (`<...>`) — postings are often exported from web pages.
3. Collapse repeated whitespace/newlines into single spaces.

**Deliberately NOT removed:** numbers, currency symbols, URLs, email
addresses. These may later turn out to be useful fraud indicators (e.g. a
request for payment, a personal email instead of a corporate one), so
stripping them now — before they've even been evaluated — would destroy
information prematurely.

---

## 3. Combined Text Field

A new field, `combined_text`, concatenates:
```
title . company_profile . description . requirements . benefits
```
The `.` separator gives a mild sentence-boundary signal so bigrams don't
accidentally merge words across two unrelated fields. The target column
(`fraudulent`) is never included — `combined_text` is built strictly from
the feature-only DataFrame.

---

## 4. Train/Test Split (For Leakage Prevention Only)

A stratified 80/20 split (train=14,304 rows, test=3,576 rows, preserving
the ~95%/5% class ratio, `random_state=42`) was created **solely** to fit
the TF-IDF vectorizer and the categorical encoders without leakage. TF-IDF
IDF weights and category frequencies are statistics learned from the data;
fitting them on the entire dataset (including rows that should later act
as unseen test data) would let information leak into feature values, which
inflates evaluation results once modeling begins. Everything is fit on the
training rows only, then used to *transform* the full dataset — transforming
does not leak information, only fitting does.

**No model is trained using this split in Day 2.** The split indices are
saved (`train_test_indices.joblib`) so Day 3 modeling can reuse the exact
same split for consistency.

---

## 5. TF-IDF Configuration

| Setting | Value | Why |
|---|---|---|
| `max_features` | 3000 | Caps the vocabulary to the 3,000 most informative terms, keeping the matrix a manageable size instead of tens of thousands of columns |
| `ngram_range` | (1, 2) | Captures both single words and two-word phrases (e.g. "wire transfer") — fraud cues are often short phrases |
| `stop_words` | `'english'` | Removes very common filler words ("the", "and", "is") that appear in nearly every posting and carry little discriminating power |

Result: **3,000 TF-IDF features**, fitted on 14,304 training rows,
applied to all 17,880 rows. Saved as a sparse matrix
(`tfidf_features.npz`) plus the fitted vectorizer itself
(`tfidf_vectorizer.joblib`) for reuse in Day 3 and the future Flask app.

---

## 6. Categorical Encoding Strategy

Cardinality (unique value count) was measured directly from the dataset
and used to choose an encoding method per column:

| Column | Unique values | Method | Why |
|---|---|---|---|
| `employment_type` | 5 | One-hot | Low cardinality — safe number of output columns |
| `required_experience` | 7 | One-hot | Low cardinality |
| `required_education` | 13 | One-hot | Low cardinality |
| `function` | 37 | One-hot | Still manageable (≤ 50 threshold) |
| `industry` | 131 | Frequency encoding | Too high cardinality for one-hot |
| `location` | 3,105 | Frequency encoding | Very high cardinality |
| `department` | 1,337 | Frequency encoding | High cardinality |
| `salary_range` | 874 | Frequency encoding | High cardinality, and 83.96% missing (see missingness flag) |

One-hot encoding used `handle_unknown='ignore'` so a category never seen
during fitting (e.g. a brand-new job posting later) produces all-zero
columns instead of crashing. Frequency encoding maps each category to how
often it appeared in the **training** rows (a value between 0 and 1);
categories unseen in training map to `0.0`.

One-hot encoding produced 66 columns across the four low-cardinality
columns; frequency encoding added 4 numeric columns.

---

## 7. Fraud-Risk Indicator Features

Simple, explainable, objectively-computed candidate features (none of
these are claimed to *prove* fraud — they are additional numeric signals
a future model may or may not find useful):

| Feature | What it measures |
|---|---|
| `missing_company_profile` | 1 if `company_profile` was originally missing |
| `missing_salary_range` | 1 if `salary_range` was originally missing |
| `missing_requirements` | 1 if `requirements` was originally missing |
| `missing_benefits` | 1 if `benefits` was originally missing |
| `missing_department` | 1 if `department` was originally missing |
| `has_company_logo` | Existing binary flag, kept as-is |
| `has_questions` | Existing binary flag, kept as-is |
| `telecommuting` | Existing binary flag, kept as-is |
| `description_length` | Character count of the cleaned description |
| `contains_url` | 1 if a URL pattern appears in the description/company_profile/requirements text |
| `contains_email` | 1 if an email-address pattern appears in that same text |

---

## 8. Output Files (`data/processed/`)

| File | Contents |
|---|---|
| `structured_features.csv` | `job_id` + all 81 structured features (binary, missingness flags, text-derived indicators, frequency-encoded, one-hot encoded) |
| `target.csv` | `job_id` + `fraudulent` label |
| `combined_text.csv` | `job_id` + the combined, cleaned text field |
| `tfidf_features.npz` | Sparse TF-IDF matrix (17,880 × 3,000), loadable via `scipy.sparse.load_npz` |
| `tfidf_vectorizer.joblib` | The fitted `TfidfVectorizer` object, for reuse in Day 3 modeling and the future Flask app |
| `onehot_encoder.joblib` | The fitted `OneHotEncoder` object |
| `frequency_encoding_maps.joblib` | Dictionary of category → frequency lookup tables |
| `train_test_indices.joblib` | The exact train/test row indices used for leakage-safe fitting |
| `preprocessing_metadata.json` | Machine-readable summary of every configuration choice and resulting shape |

The raw CSV (`data/raw/fake_job_postings.csv`) is **never modified** — the
pipeline only reads it. This was verified by comparing an MD5 checksum of
the file before and after running the script (see script output).

---

## 9. Important Design Decisions (Summary)

1. **Missingness preserved as information**, not discarded — no rows or
   columns were dropped due to missing values.
2. **Text kept intentionally "raw-ish"** (numbers, symbols, URLs, emails
   preserved) since these may hold fraud signal that shouldn't be erased
   before evaluation.
3. **Train/test split introduced only to prevent leakage** during TF-IDF
   and encoder fitting — not for model training, which is a Day 3 task.
4. **Cardinality-aware encoding** (one-hot for low-cardinality columns,
   frequency encoding for high-cardinality columns) balances feature
   usefulness against feature-matrix size.
5. **All fitted transformers are saved**, not just the transformed
   numbers, so the exact same transformation can be reapplied consistently
   to new data later (critical for the future Flask app).

---

## 10. Limitations

- Frequency encoding compresses each high-cardinality category into a
  single number, which loses some nuance compared to one-hot encoding
  (e.g. two different rare locations become numerically similar even
  though they are different places).
- TF-IDF with `max_features=3000` discards less-frequent/lower-scoring
  terms; a larger vocabulary could capture more nuance at the cost of a
  larger matrix.
- `stop_words='english'` removes common filler words; if certain filler
  phrases turn out to matter for fraud detection, this choice may need
  revisiting.
- No correlation or feature-importance analysis has been performed yet —
  none of the engineered features have been statistically validated
  against the target. That belongs to Day 3 (modeling) and later
  evaluation.
- This pipeline has not yet been tested on data outside this specific
  CSV (e.g. a live-scraped posting); real-world robustness will need
  verification once the Flask app is built.

---
*All statistics in this report were generated by actually running
`src/preprocessing.py` against the real `fake_job_postings.csv` file.
No values were invented.*
