"""
==================================================================
Day 2: Data Preprocessing & NLP Feature Engineering Pipeline
Project: AI-Based Fake Job Detection and Verification System
==================================================================

WHAT THIS SCRIPT DOES:
This script turns the RAW dataset (data/raw/fake_job_postings.csv)
into machine-learning-ready feature representations, and saves them
to data/processed/. It does NOT train any model and does NOT touch
the Flask app. The raw CSV is never modified — we only read it.

WHY A SEPARATE PREPROCESSING SCRIPT (AND NOT JUST NOTEBOOK CODE)?
Keeping preprocessing in a reusable, function-based script means:
    - It is reproducible: running it twice gives the same result.
    - It can be imported later without copy-pasting code.
    - It documents exactly what transformations were applied to the
      data, which is essential to explain in your viva.

HIGH-LEVEL PIPELINE (matches the Day 2 stages):
    1. Load raw dataset + verify structure
    2. Separate target column from features
    3. Handle missing values (explicit strategies)
    4. Clean text columns (conservative normalization)
    5. Build a single combined_text field
    6. Split into train/test indices (ONLY to prevent leakage while
       fitting TF-IDF and categorical encoders — no model is trained)
    7. Fit TF-IDF on the combined_text of the TRAINING rows only
    8. Encode structured (binary + categorical) features
    9. Engineer simple, explainable fraud-risk indicator features
    10. Save all processed artifacts to data/processed/
    11. Print a verification / sanity-check summary
==================================================================
"""

import os
import re
import json
import hashlib

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib


# ------------------------------------------------------------------
# PROJECT PATHS (relative, so this works on any machine)
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "fake_job_postings.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# ------------------------------------------------------------------
# COLUMN DEFINITIONS
# These were established during Day 1 exploration and are hard-coded
# here deliberately (not "invented") — they were verified against
# the real CSV headers.
# ------------------------------------------------------------------
TARGET_COL = "fraudulent"

EXPECTED_COLUMNS = [
    "job_id", "title", "location", "department", "salary_range",
    "company_profile", "description", "requirements", "benefits",
    "telecommuting", "has_company_logo", "has_questions",
    "employment_type", "required_experience", "required_education",
    "industry", "function", "fraudulent",
]

TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits"]

BINARY_COLUMNS = ["telecommuting", "has_company_logo", "has_questions"]

# All categorical/metadata columns that need some form of encoding.
CATEGORICAL_COLUMNS = [
    "employment_type", "required_experience", "required_education",
    "industry", "function", "location", "department", "salary_range",
]

# Design decision (explained fully in PREPROCESSING_INFO.md):
# Low-cardinality categorical columns (few unique values) are safe to
# one-hot encode without creating too many columns. High-cardinality
# columns (location, department, salary_range, industry) would create
# thousands of one-hot columns, so we use frequency encoding instead.
# Cardinalities were measured directly from the dataset during Day 1/2:
#   employment_type=5, required_experience=7, required_education=13,
#   function=37   -> safe to one-hot encode (<= 50 unique values)
#   industry=131, location=3105, department=1337, salary_range=874
#                 -> too high cardinality, use frequency encoding
ONE_HOT_COLUMNS = ["employment_type", "required_experience", "required_education", "function"]
FREQUENCY_ENCODE_COLUMNS = ["industry", "location", "department", "salary_range"]

# TF-IDF configuration (explained in detail in the report / comments below)
TFIDF_MAX_FEATURES = 3000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_STOP_WORDS = "english"

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ==================================================================
# STAGE 1 — LOAD RAW DATASET
# ==================================================================
def load_raw_dataset(path: str) -> pd.DataFrame:
    """
    WHAT: Loads the raw CSV and verifies that all expected columns
          are present before we do anything else.
    WHY:  Failing fast with a clear error message is much better than
          crashing later with a confusing KeyError deep inside the
          pipeline. This is basic defensive programming.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw dataset not found at: {path}\n"
            f"Make sure 'fake_job_postings.csv' exists inside data/raw/"
        )

    df = pd.read_csv(path)

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The dataset is missing expected columns: {missing_cols}\n"
            f"Found columns: {list(df.columns)}"
        )

    print(f"[Stage 1] Raw dataset loaded successfully. Shape: {df.shape}")
    return df


def file_md5(path: str) -> str:
    """
    WHAT: Computes an MD5 checksum of a file.
    WHY:  We use this BEFORE and AFTER running the pipeline to prove
          (not just claim) that the raw CSV was never modified. If the
          two checksums match, the file's bytes are 100% identical.
    """
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ==================================================================
# STAGE 2 — SEPARATE TARGET AND FEATURES
# ==================================================================
def separate_target(df: pd.DataFrame):
    """
    WHAT: Splits the DataFrame into `features_df` (everything except
          the target) and `target` (the fraudulent column, untouched).
    WHY:  Keeping the target physically separate from the feature
          pipeline is the single most important habit for avoiding
          "target leakage" — accidentally letting the answer sneak
          into the inputs the model will later see.
    """
    target = df[TARGET_COL].copy()          # untouched, original labels
    features_df = df.drop(columns=[TARGET_COL]).copy()
    print(f"[Stage 2] Target separated. Target shape: {target.shape}, "
          f"Feature columns: {features_df.shape[1]}")
    return features_df, target


# ==================================================================
# STAGE 8 (computed early) — MISSING-VALUE INDICATOR FLAGS
# ==================================================================
def create_missingness_flags(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    WHAT: Before we fill in any missing values, we record WHICH rows
          were originally missing important information, as their own
          0/1 columns (e.g. missing_company_profile).
    WHY:  The fact that a field is missing can itself be a signal —
          for example, scam postings often skip the company profile
          or salary range. Once we fill missing text with "" or
          "Unknown" in the next stage, that information would
          otherwise be lost. This does NOT claim missingness proves
          fraud — it's just an explainable candidate feature for the
          model to learn from (or not) later.
    """
    flags = pd.DataFrame(index=features_df.index)
    flags["missing_company_profile"] = features_df["company_profile"].isna().astype(int)
    flags["missing_salary_range"] = features_df["salary_range"].isna().astype(int)
    flags["missing_requirements"] = features_df["requirements"].isna().astype(int)
    flags["missing_benefits"] = features_df["benefits"].isna().astype(int)
    flags["missing_department"] = features_df["department"].isna().astype(int)

    print(f"[Stage 8] Missingness indicator flags created: {list(flags.columns)}")
    return flags


# ==================================================================
# STAGE 3 — HANDLE MISSING VALUES
# ==================================================================
def handle_missing_values(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    WHAT: Fills missing values using a strategy appropriate to each
          column TYPE, rather than one blanket rule.

    STRATEGY AND WHY (also explained in PREPROCESSING_INFO.md):
      - Text columns (title, company_profile, description,
        requirements, benefits): filled with an empty string "".
        An empty string is a neutral, safe placeholder for text —
        it contributes nothing to TF-IDF term frequencies, so it
        does not fabricate content that was never written.

      - Categorical columns (employment_type, required_experience,
        required_education, industry, function, location,
        department, salary_range): filled with the explicit label
        "Unknown". We deliberately do NOT drop rows or columns here,
        even for salary_range (83.96% missing) and department
        (64.58% missing) — dropping them would throw away 60-84%
        of the dataset's rows if done via dropna(), which is far
        worse than losing one weak feature. Treating "missing" as
        its own category preserves every row and lets the model
        learn whether "Unknown" itself correlates with fraud.

      - Binary columns (telecommuting, has_company_logo,
        has_questions): Day 1 exploration found 0% missing values
        in these columns, but we still defensively fill any future
        missing value with 0 (the more common / "safe default"
        value) so the pipeline never crashes if a future data
        export has a rare missing entry here.
    """
    df = features_df.copy()

    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("")

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna("Unknown")

    for col in BINARY_COLUMNS:
        df[col] = df[col].fillna(0).astype(int)

    print("[Stage 3] Missing values handled: text -> '', "
          "categorical -> 'Unknown', binary -> 0 (defensive).")
    return df


# ==================================================================
# STAGE 4 — TEXT CLEANING (conservative normalization)
# ==================================================================
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_single_text(text: str) -> str:
    """
    WHAT: Applies conservative, reversible-in-spirit cleaning to one
          piece of text:
            1. Lowercase everything (so "Company" and "company" are
               treated as the same word by TF-IDF).
            2. Strip HTML tags (job postings are often exported from
               web pages and may contain leftover tags like <p>, <br>).
            3. Collapse repeated whitespace/newlines into single spaces.

    WHY CONSERVATIVE: We deliberately do NOT remove numbers, currency
          symbols ($, ₹, etc.), URLs, or email addresses. In job-scam
          detection, these are exactly the kind of details (e.g. "pay
          $500 to start", a suspicious external link, a personal email
          instead of a company one) that could later become useful
          fraud signals. Aggressively stripping them now would destroy
          information before we've even had a chance to evaluate it.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def clean_text_columns(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    WHAT: Applies `clean_single_text` to every text column.
    WHY:  Centralizing this in one function keeps the cleaning rule
          identical and reproducible across all five text columns.
    """
    df = features_df.copy()
    for col in TEXT_COLUMNS:
        df[col] = df[col].apply(clean_single_text)
    print(f"[Stage 4] Text columns cleaned (lowercased, HTML stripped, "
          f"whitespace collapsed): {TEXT_COLUMNS}")
    return df


# ==================================================================
# STAGE 5 — COMBINED TEXT FIELD
# ==================================================================
def create_combined_text(features_df: pd.DataFrame) -> pd.Series:
    """
    WHAT: Concatenates the five text columns into one field called
          `combined_text`, using ' . ' as a separator between fields.
    WHY:  TF-IDF (Stage 6) needs a single text input per row. Using a
          period-based separator (rather than just a plain space)
          gives a mild "sentence boundary" signal to the tokenizer,
          so a 2-word phrase (bigram) is less likely to accidentally
          merge the last word of one field with the first word of the
          next (e.g. "...urgent" + "hiring now..." should not become
          the bigram "urgent hiring" if they came from different
          fields).
    NOTE: The target column is NEVER included here — combined_text is
          built purely from features_df, which had the target removed
          back in Stage 2.
    """
    combined = (
        features_df["title"] + " . " +
        features_df["company_profile"] + " . " +
        features_df["description"] + " . " +
        features_df["requirements"] + " . " +
        features_df["benefits"]
    )
    print("[Stage 5] combined_text field created from title, company_profile, "
          "description, requirements, benefits.")
    return combined


# ==================================================================
# STAGE 6 — TRAIN/TEST SPLIT (ONLY to prevent leakage during fitting)
# ==================================================================
def create_leakage_safe_split(target: pd.Series):
    """
    WHAT: Creates train/test ROW INDICES (stratified by the target,
          so both splits keep roughly the same 95%/5% class balance).

    WHY WE NEED THIS IN DAY 2 (even though we are NOT training a
    model yet): TF-IDF and the categorical encoders below "learn"
    statistics from the data — TF-IDF learns vocabulary + how rare/
    common each word is (its IDF weight), and frequency encoding
    learns how often each category appears. If we fit those on the
    ENTIRE dataset, information from rows that should later be
    "unseen test data" would leak into the vectorizer/encoder, and
    any model trained on the resulting features would get an
    unfairly optimistic evaluation score. Fitting only on the
    training rows (and simply transforming the test rows with what
    was learned) is the standard, correct way to avoid this.

    IMPORTANT: We only compute and save these indices for reuse. No
    model is trained here — this satisfies the Day 2 rule that
    splitting is allowed ONLY when required to prevent preprocessing
    leakage.
    """
    all_indices = np.arange(len(target))
    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    print(f"[Stage 6] Leakage-safe split created (for fitting only): "
          f"train={len(train_idx)} rows, test={len(test_idx)} rows "
          f"(stratified on '{TARGET_COL}').")
    return train_idx, test_idx


# ==================================================================
# STAGE 7 — TF-IDF FEATURE EXTRACTION
# ==================================================================
def fit_tfidf(combined_text: pd.Series, train_idx: np.ndarray) -> TfidfVectorizer:
    """
    WHAT: Fits a TfidfVectorizer using ONLY the training rows of
          combined_text, then this same fitted vectorizer will be
          used to transform the FULL dataset (Stage 7b below).

    CONFIGURATION CHOICES (why each one):
      - max_features=3000: Without a cap, TF-IDF on ~18k free-text
        job postings could generate tens of thousands of columns
        (one per unique word/phrase). Capping at the 3000 most
        informative terms (by TF-IDF score) keeps the feature matrix
        a manageable, trainable size while retaining the strongest
        signals.
      - ngram_range=(1, 2): Includes both single words (unigrams,
        e.g. "urgent") and two-word phrases (bigrams, e.g. "wire
        transfer", "no interview"). Many fraud cues in job postings
        are short phrases, not single words, so bigrams add real
        value here.
      - stop_words='english': Removes very common English filler
        words (e.g. "the", "and", "is") that appear in almost every
        posting and therefore carry little discriminating power for
        telling genuine vs fraudulent apart. This is a standard,
        justified default for general English text classification.
        (Trade-off, noted for the viva: a small number of stopword-
        adjacent phrases could theoretically be lost — this can be
        revisited later if the model under-performs.)
    """
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words=TFIDF_STOP_WORDS,
    )
    vectorizer.fit(combined_text.iloc[train_idx])
    print(f"[Stage 7] TF-IDF vectorizer fitted on {len(train_idx)} training rows. "
          f"Vocabulary size (capped): {len(vectorizer.vocabulary_)}")
    return vectorizer


# ==================================================================
# STAGE 7 (structured features) — CATEGORICAL ENCODING
# ==================================================================
def fit_one_hot_encoder(features_df: pd.DataFrame, train_idx: np.ndarray) -> OneHotEncoder:
    """
    WHAT: Fits a OneHotEncoder on the low-cardinality categorical
          columns, using training rows only (same leakage-avoidance
          reasoning as TF-IDF above).
    WHY handle_unknown='ignore': If a category appears later (e.g. a
          new "required_education" value not seen during fitting),
          the encoder will safely output all-zero columns for it
          instead of crashing. This matters because this project will
          later need to handle real, previously-unseen job postings.
    """
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    encoder.fit(features_df.loc[train_idx, ONE_HOT_COLUMNS])
    n_out = sum(len(cats) for cats in encoder.categories_)
    print(f"[Stage 7] OneHotEncoder fitted on {ONE_HOT_COLUMNS} "
          f"(train rows only) -> {n_out} output columns.")
    return encoder


def fit_frequency_maps(features_df: pd.DataFrame, train_idx: np.ndarray) -> dict:
    """
    WHAT: For each high-cardinality categorical column (location,
          department, salary_range, industry), computes how often
          each category value appears (as a proportion) using ONLY
          the training rows, and stores that as a lookup dictionary.
    WHY frequency encoding instead of one-hot: One-hot encoding
          `location` (3,105 unique values) or `department` (1,337
          unique values) would create thousands of mostly-empty
          columns, ballooning the feature matrix size for very
          little benefit. Frequency encoding instead represents each
          category with a single number (how common it is), which
          is compact, always numeric, and still captures useful
          signal (e.g. a very rare/unique location name might be more
          suspicious than a extremely common one like "Unknown").
    Unseen categories (in rows outside the training set) are mapped
    to 0.0, since we have no training-time information about them.
    """
    freq_maps = {}
    for col in FREQUENCY_ENCODE_COLUMNS:
        train_values = features_df.loc[train_idx, col]
        freq_maps[col] = train_values.value_counts(normalize=True).to_dict()
    print(f"[Stage 7] Frequency-encoding maps fitted (train rows only) "
          f"for: {FREQUENCY_ENCODE_COLUMNS}")
    return freq_maps


def apply_frequency_encoding(features_df: pd.DataFrame, freq_maps: dict) -> pd.DataFrame:
    """Transforms each high-cardinality categorical column into its
    training-set frequency (a float between 0 and 1). Unseen
    categories become 0.0."""
    encoded = pd.DataFrame(index=features_df.index)
    for col in FREQUENCY_ENCODE_COLUMNS:
        encoded[f"{col}_freq"] = features_df[col].map(freq_maps[col]).fillna(0.0)
    return encoded


# ==================================================================
# STAGE 8 — ADDITIONAL SIMPLE TEXT-BASED FRAUD-RISK INDICATORS
# ==================================================================
URL_PATTERN = re.compile(r"https?://|www\.")
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")


def create_text_derived_indicators(cleaned_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    WHAT: Creates a few simple, fully explainable indicators computed
          directly from the cleaned text:
            - description_length: character count of the description.
            - contains_url: 1 if a URL pattern appears anywhere in the
              combined text fields, else 0.
            - contains_email: 1 if an email pattern appears, else 0.
    WHY:  These are objective, easily-verifiable measurements — not
          opinions. IMPORTANT: none of these are claimed to PROVE a
          posting is fraudulent; they are simply additional numeric
          signals the model may (or may not) find useful later,
          exactly like the missingness flags in Stage 3.
    """
    indicators = pd.DataFrame(index=cleaned_features_df.index)
    indicators["description_length"] = cleaned_features_df["description"].str.len()

    text_for_scan = (
        cleaned_features_df["description"] + " " +
        cleaned_features_df["company_profile"] + " " +
        cleaned_features_df["requirements"]
    )
    indicators["contains_url"] = text_for_scan.str.contains(URL_PATTERN).astype(int)
    indicators["contains_email"] = text_for_scan.str.contains(EMAIL_PATTERN).astype(int)

    print("[Stage 8] Text-derived indicator features created: "
          f"{list(indicators.columns)}")
    return indicators


# ==================================================================
# STAGE 9 — SAVE PROCESSED OUTPUTS
# ==================================================================
def save_outputs(
    job_ids: pd.Series,
    target: pd.Series,
    structured_features: pd.DataFrame,
    combined_text: pd.Series,
    tfidf_matrix: sparse.csr_matrix,
    tfidf_vectorizer: TfidfVectorizer,
    one_hot_encoder: OneHotEncoder,
    freq_maps: dict,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    metadata: dict,
):
    """
    WHAT: Writes every processed artifact to data/processed/.
    WHY:  Saving the fitted vectorizer/encoders (not just the
          transformed numbers) is essential — Day 3 model training
          and the future Flask app must use these EXACT same fitted
          objects to transform any new job posting consistently.
          Re-fitting them later on different data would silently
          break the whole pipeline.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Structured (non-text) features + target, aligned by job_id
    structured_out = structured_features.copy()
    structured_out.insert(0, "job_id", job_ids.values)
    structured_out.to_csv(os.path.join(PROCESSED_DIR, "structured_features.csv"), index=False)

    target_out = pd.DataFrame({"job_id": job_ids.values, TARGET_COL: target.values})
    target_out.to_csv(os.path.join(PROCESSED_DIR, "target.csv"), index=False)

    combined_text_out = pd.DataFrame({"job_id": job_ids.values, "combined_text": combined_text.values})
    combined_text_out.to_csv(os.path.join(PROCESSED_DIR, "combined_text.csv"), index=False)

    # Sparse TF-IDF matrix (saved in scipy's efficient sparse format)
    sparse.save_npz(os.path.join(PROCESSED_DIR, "tfidf_features.npz"), tfidf_matrix)

    # Fitted objects, needed again for Day 3 modeling and the future Flask app
    joblib.dump(tfidf_vectorizer, os.path.join(PROCESSED_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(one_hot_encoder, os.path.join(PROCESSED_DIR, "onehot_encoder.joblib"))
    joblib.dump(freq_maps, os.path.join(PROCESSED_DIR, "frequency_encoding_maps.joblib"))
    joblib.dump(
        {"train_idx": train_idx, "test_idx": test_idx},
        os.path.join(PROCESSED_DIR, "train_test_indices.joblib"),
    )

    with open(os.path.join(PROCESSED_DIR, "preprocessing_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Stage 9] All processed artifacts saved to: {PROCESSED_DIR}")


# ==================================================================
# MAIN PIPELINE
# ==================================================================
def main():
    raw_md5_before = file_md5(RAW_PATH)

    # Stage 1
    raw_df = load_raw_dataset(RAW_PATH)
    original_shape = raw_df.shape

    # Stage 2
    features_df, target = separate_target(raw_df)
    job_ids = features_df["job_id"]

    # Stage 8 (part 1) — capture missingness BEFORE filling anything
    missingness_flags = create_missingness_flags(features_df)

    # Stage 3
    features_filled = handle_missing_values(features_df)

    # Stage 4
    features_clean = clean_text_columns(features_filled)

    # Stage 5
    combined_text = create_combined_text(features_clean)

    # Sanity check: the target must never leak into the text/feature side
    assert TARGET_COL not in features_clean.columns, "Target column leaked into features!"

    # Stage 6 — split ONLY for leakage-safe fitting (no model training)
    train_idx, test_idx = create_leakage_safe_split(target)

    # Stage 7 — TF-IDF (fit on train rows only, transform the full dataset)
    tfidf_vectorizer = fit_tfidf(combined_text, train_idx)
    tfidf_matrix = tfidf_vectorizer.transform(combined_text)  # full dataset, safe: fit used train only

    # Stage 7 — structured feature encoding (fit on train rows only)
    one_hot_encoder = fit_one_hot_encoder(features_clean, train_idx)
    one_hot_matrix_full = one_hot_encoder.transform(features_clean[ONE_HOT_COLUMNS])
    one_hot_cols = one_hot_encoder.get_feature_names_out(ONE_HOT_COLUMNS)
    one_hot_df = pd.DataFrame.sparse.from_spmatrix(one_hot_matrix_full, columns=one_hot_cols, index=features_clean.index)

    freq_maps = fit_frequency_maps(features_clean, train_idx)
    freq_encoded_df = apply_frequency_encoding(features_clean, freq_maps)

    # Stage 8 (part 2) — text-derived indicators
    text_indicators = create_text_derived_indicators(features_clean)

    # Binary columns, unchanged, plus everything engineered above
    binary_df = features_clean[BINARY_COLUMNS].astype(int)

    structured_features = pd.concat(
        [binary_df, missingness_flags, text_indicators, freq_encoded_df, one_hot_df],
        axis=1,
    )

    # Final leakage sanity check on the structured feature table
    assert TARGET_COL not in structured_features.columns, "Target column leaked into structured features!"

    # Stage 9 — save everything
    metadata = {
        "original_shape": list(original_shape),
        "n_rows": int(original_shape[0]),
        "target_column": TARGET_COL,
        "target_classes": sorted(target.unique().tolist()),
        "n_target_classes": int(target.nunique()),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "tfidf_max_features": TFIDF_MAX_FEATURES,
        "tfidf_ngram_range": list(TFIDF_NGRAM_RANGE),
        "tfidf_stop_words": TFIDF_STOP_WORDS,
        "tfidf_actual_vocabulary_size": len(tfidf_vectorizer.vocabulary_),
        "one_hot_columns": ONE_HOT_COLUMNS,
        "frequency_encoded_columns": FREQUENCY_ENCODE_COLUMNS,
        "n_structured_features": int(structured_features.shape[1]),
        "structured_feature_columns": list(structured_features.columns),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "raw_csv_md5_before_pipeline": raw_md5_before,
    }

    save_outputs(
        job_ids=job_ids,
        target=target,
        structured_features=structured_features,
        combined_text=combined_text,
        tfidf_matrix=tfidf_matrix,
        tfidf_vectorizer=tfidf_vectorizer,
        one_hot_encoder=one_hot_encoder,
        freq_maps=freq_maps,
        train_idx=train_idx,
        test_idx=test_idx,
        metadata=metadata,
    )

    # ==============================================================
    # STAGE 11 — VERIFICATION SUMMARY
    # ==============================================================
    raw_md5_after = file_md5(RAW_PATH)

    print("\n" + "=" * 70)
    print("STAGE 11 — VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"Original raw dataset shape:        {original_shape}")
    print(f"Structured feature matrix shape:   {structured_features.shape}")
    print(f"TF-IDF feature matrix shape:       {tfidf_matrix.shape}")
    print(f"Number of target classes:          {target.nunique()} {sorted(target.unique().tolist())}")
    print(f"Number of TF-IDF features:         {len(tfidf_vectorizer.vocabulary_)}")
    print(f"Number of structured features:     {structured_features.shape[1]}")
    print(f"Target column NOT in text data:    {TARGET_COL not in features_clean.columns}")
    print(f"Target column NOT in structured features: {TARGET_COL not in structured_features.columns}")
    print(f"Raw CSV MD5 before pipeline:        {raw_md5_before}")
    print(f"Raw CSV MD5 after pipeline:         {raw_md5_after}")
    print(f"Raw CSV unmodified (checksums match): {raw_md5_before == raw_md5_after}")
    print("=" * 70)
    print("DAY 2 PREPROCESSING COMPLETE — no model has been trained.")
    print("=" * 70)


if __name__ == "__main__":
    main()
