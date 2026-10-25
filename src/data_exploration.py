"""
==================================================================
Day 1: Dataset Exploration
Project: AI-Based Fake Job Detection and Verification System
==================================================================

WHAT THIS SCRIPT DOES:
This script performs Exploratory Data Analysis (EDA) on the raw
job postings dataset. It does NOT clean the data, does NOT train
any model, and does NOT build the Flask app. Its only job is to
help us UNDERSTAND the dataset before we touch it.

WHY THIS MATTERS FOR YOUR PROJECT / VIVA:
Every serious ML project starts with EDA. Before you can clean
data or engineer features, you must know:
    - How big is the dataset?
    - What does each column contain?
    - Where are the gaps (missing values)?
    - Is the target column balanced or imbalanced?
    - Which columns are text (for NLP) vs numeric vs binary?

Skipping this step is one of the most common mistakes final-year
students make — they jump straight to modeling and later discover
data problems that force them to redo everything.
==================================================================
"""

import os
import pandas as pd


# ------------------------------------------------------------------
# STEP 1: Locate the dataset using a RELATIVE path
# ------------------------------------------------------------------
# WHAT: We build the path to the CSV file relative to this script's
#       own location, instead of hardcoding an absolute path like
#       "C:/Users/YourName/project/data/raw/fake_job_postings.csv".
#
# WHY:  Absolute paths break the moment the project is moved to
#       another folder, another PC, or submitted to your guide/
#       evaluator. Relative paths make the project portable —
#       it will run correctly on ANY machine as long as the folder
#       structure is preserved.
#
# VIVA TIP: If asked "why did you use os.path instead of typing the
#       path directly?", say: "To make the project portable and
#       independent of the machine it runs on."
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)          # src/ -> project root
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "fake_job_postings.csv")


def load_dataset(path: str) -> pd.DataFrame:
    """
    WHAT: Loads the CSV file into a Pandas DataFrame.
    WHY:  A DataFrame is Pandas' table-like structure (rows + columns)
          that makes it easy to inspect, filter, and analyze data.
    VIVA: Be ready to explain — "Pandas DataFrame is similar to an
          Excel sheet or SQL table, held in memory as a Python object."
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            f"Make sure 'fake_job_postings.csv' is placed inside data/raw/"
        )
    df = pd.read_csv(path)
    return df


def show_basic_info(df: pd.DataFrame) -> None:
    """
    WHAT: Prints dataset shape, first 5 rows, last 5 rows, column
          names, and dtypes.
    WHY:  This is the very first thing any data scientist checks —
          it tells you the scale of the data (rows/columns) and gives
          a first visual impression of what the data looks like.
          Checking the LAST rows too (not just the first) confirms
          the file was loaded completely and wasn't truncated or
          corrupted partway through.
    VIVA: 'Shape' = (rows, columns). 'dtype' tells Pandas how to treat
          each column internally (numeric vs text/object).
    """
    print("=" * 70)
    print("1. DATASET SHAPE")
    print("=" * 70)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n" + "=" * 70)
    print("2. FIRST 5 RECORDS")
    print("=" * 70)
    print(df.head())

    print("\n" + "=" * 70)
    print("2b. LAST 5 RECORDS")
    print("=" * 70)
    print(df.tail())

    print("\n" + "=" * 70)
    print("3. COLUMN NAMES")
    print("=" * 70)
    for i, col in enumerate(df.columns, start=1):
        print(f"{i}. {col}")

    print("\n" + "=" * 70)
    print("4. DATA TYPES")
    print("=" * 70)
    print(df.dtypes)


def show_missing_values(df: pd.DataFrame) -> None:
    """
    WHAT: Counts missing (NaN) values per column, and converts that
          count into a percentage of the whole dataset.
    WHY:  Missing data affects which columns are usable. A column
          missing 80%+ of its values (e.g. salary_range) may need to
          be dropped or handled specially — but that decision belongs
          to Day 2 (preprocessing), NOT today. Today we only OBSERVE.
    VIVA: Explain the difference between MCAR/MAR/MNAR is optional,
          but you should at least say: "High missing percentage
          columns need special handling — imputation or removal —
          decided in the preprocessing stage."
    """
    print("\n" + "=" * 70)
    print("5. MISSING VALUES PER COLUMN")
    print("=" * 70)
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)

    missing_summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percent": missing_pct
    }).sort_values(by="missing_percent", ascending=False)

    print(missing_summary)


def show_duplicates(df: pd.DataFrame) -> None:
    """
    WHAT: Counts fully duplicated rows in the dataset.
    WHY:  Duplicate rows can bias a model — the same example gets
          seen (and weighted) multiple times during training, which
          can artificially inflate accuracy or cause overfitting to
          repeated patterns.
    VIVA: "We checked for exact duplicate rows across all columns to
          ensure no record is unfairly repeated during model training."
    """
    print("\n" + "=" * 70)
    print("6. DUPLICATE RECORDS")
    print("=" * 70)
    dup_count = df.duplicated().sum()
    print(f"Fully duplicated rows: {dup_count}")


def show_target_distribution(df: pd.DataFrame, target_col: str = "fraudulent") -> bool:
    """
    WHAT: Shows the count and percentage split of the target column
          (genuine=0 vs fraudulent=1).
    WHY:  This is THE most important check for a classification
          project. If one class massively outnumbers the other
          (class imbalance), standard accuracy becomes a misleading
          metric, and you'll need techniques like class weighting,
          SMOTE, or choosing precision/recall/F1 instead of accuracy.
    SAFETY: We first check that the target column actually exists in
          the dataset before analyzing it. This prevents a crash if
          the CSV structure ever changes (e.g. a different export of
          the dataset uses a different column name), and it is good
          defensive-programming practice to show in your viva.
    VIVA: Be ready to say the exact ratio, e.g. "our dataset is
          imbalanced, with genuine postings far outnumbering
          fraudulent ones, which is realistic since fraud is rare
          in real life — but it means we must evaluate our model
          using precision, recall, and F1-score, not just accuracy."
    Returns True if the target column was found and analyzed.
    """
    print("\n" + "=" * 70)
    print(f"7 & 8. TARGET COLUMN ('{target_col}') DISTRIBUTION")
    print("=" * 70)

    if target_col not in df.columns:
        print(f"WARNING: Target column '{target_col}' was not found in this "
              f"dataset. Available columns are: {list(df.columns)}")
        return False

    counts = df[target_col].value_counts()
    percentages = (df[target_col].value_counts(normalize=True) * 100).round(2)

    print("Counts:")
    print(counts)
    print("\nPercentages:")
    print(percentages)
    return True


def identify_column_types(df: pd.DataFrame) -> dict:
    """
    WHAT: Splits columns into three buckets:
            - text_columns: long free-text fields suitable for NLP
            - binary_columns: columns holding only 0/1 flags
            - numeric_columns: numeric but not binary (e.g. IDs)
    WHY:  Different column types need different treatment later:
            - text columns -> TF-IDF / embeddings for NLP features
            - binary columns -> used directly as model features
            - numeric columns -> may need scaling
          Getting this classification right NOW saves confusion
          during the feature engineering stage (Day 2+).
    VIVA: Explain WHY a column is text vs categorical vs binary —
          e.g. 'description' has long free-form sentences (NLP),
          while 'telecommuting' only ever holds 0 or 1 (binary flag).
    """
    print("\n" + "=" * 70)
    print("10, 11, 12. COLUMN TYPE IDENTIFICATION")
    print("=" * 70)

    text_columns = ["title", "company_profile", "description",
                     "requirements", "benefits"]
    text_columns = [c for c in text_columns if c in df.columns]

    binary_columns = [c for c in ["telecommuting", "has_company_logo",
                                   "has_questions", "fraudulent"]
                       if c in df.columns and df[c].dropna().isin([0, 1]).all()]

    numeric_columns = [c for c in df.select_dtypes(include="number").columns
                        if c not in binary_columns]

    categorical_columns = [c for c in ["location", "department", "salary_range",
                                        "employment_type", "required_experience",
                                        "required_education", "industry", "function"]
                            if c in df.columns]

    print(f"Text columns (NLP candidates): {text_columns}")
    print(f"Binary columns (0/1 flags):    {binary_columns}")
    print(f"Numeric columns (non-binary):  {numeric_columns}")
    print(f"Categorical/metadata columns:  {categorical_columns}")

    # ------------------------------------------------------------
    # 17. Columns that may be useful for FRAUD-RISK analysis.
    # WHAT: A shortlist of metadata columns that (based on domain
    #       knowledge of job-scam patterns) are plausible fraud
    #       signals — e.g. postings with no company logo, no
    #       screening questions, or missing company profile text
    #       are commonly associated with scam listings.
    # WHY:  These are only CANDIDATES at this stage. We are not
    #       testing or confirming their predictive power today —
    #       that correlation analysis belongs to Day 2. Today we
    #       only flag which columns are worth investigating further.
    # ------------------------------------------------------------
    fraud_risk_candidates = [c for c in [
        "has_company_logo", "has_questions", "telecommuting",
        "company_profile", "salary_range", "employment_type",
        "required_education", "required_experience"
    ] if c in df.columns]

    print(f"Fraud-risk candidate columns (for future analysis): {fraud_risk_candidates}")

    return {
        "text_columns": text_columns,
        "binary_columns": binary_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "fraud_risk_candidates": fraud_risk_candidates,
    }


def show_categorical_uniques(df: pd.DataFrame, categorical_columns: list) -> None:
    """
    WHAT: Prints unique value counts for each categorical/binary column.
    WHY:  Helps us understand the 'vocabulary' of each categorical
          field — e.g. how many employment types exist, how many
          education levels exist. This matters for later one-hot
          encoding or label encoding decisions.
    VIVA: "We inspected unique values to understand the cardinality
          of each categorical column before deciding how to encode it."
    """
    print("\n" + "=" * 70)
    print("7. UNIQUE VALUES IN CATEGORICAL / BINARY COLUMNS")
    print("=" * 70)
    for col in categorical_columns:
        n_unique = df[col].nunique(dropna=True)
        print(f"\n--- {col} (unique values: {n_unique}) ---")
        print(df[col].value_counts(dropna=False).head(10))


def data_quality_summary(df: pd.DataFrame, col_types: dict, target_col: str = "fraudulent") -> None:
    """
    WHAT: Prints a short, plain-English summary of everything found.
    WHY:  A good data scientist doesn't just dump numbers — they
          interpret them. This summary is exactly what you'll say
          out loud in your viva when asked "what did you find in
          your EDA step?"
    """
    print("\n" + "=" * 70)
    print("13. DATA QUALITY SUMMARY")
    print("=" * 70)

    n_rows, n_cols = df.shape
    dup_count = df.duplicated().sum()

    high_missing = df.isnull().mean().sort_values(ascending=False)
    high_missing = high_missing[high_missing > 0.3] * 100

    print(f"- Dataset contains {n_rows} job postings across {n_cols} columns.")
    print(f"- Duplicate rows found: {dup_count}.")

    if target_col in df.columns:
        fraud_pct = (df[target_col].value_counts(normalize=True) * 100).round(2)
        print(f"- Target column '{target_col}' distribution: "
              f"{fraud_pct.get(0, 0)}% genuine vs {fraud_pct.get(1, 0)}% fraudulent "
              f"-> this is an IMBALANCED classification problem.")
    else:
        print(f"- Target column '{target_col}' was not found; class-balance "
              f"could not be evaluated.")
    if len(high_missing) > 0:
        print("- Columns with more than 30% missing values (need special handling later):")
        for col, pct in high_missing.items():
            print(f"    * {col}: {pct:.2f}% missing")
    else:
        print("- No columns exceed 30% missing values.")
    print(f"- Identified {len(col_types['text_columns'])} text columns suitable for NLP: "
          f"{col_types['text_columns']}")
    print(f"- Identified {len(col_types['binary_columns'])} binary (0/1) columns: "
          f"{col_types['binary_columns']}")
    print("- No preprocessing or model training has been performed in this step, "
          "as per Day 1 scope.")


def main():
    print("Dataset filename:", os.path.basename(DATA_PATH))
    print("Loading dataset from:", DATA_PATH)
    df = load_dataset(DATA_PATH)

    show_basic_info(df)
    show_missing_values(df)
    show_duplicates(df)

    target_col = "fraudulent"
    target_found = show_target_distribution(df, target_col=target_col)

    col_types = identify_column_types(df)
    show_categorical_uniques(df, col_types["categorical_columns"] + col_types["binary_columns"])

    if target_found:
        data_quality_summary(df, col_types, target_col=target_col)
    else:
        print("\nSkipping target-based summary because the target column "
              "was not found in this dataset.")

    print("\n" + "=" * 70)
    print("DAY 1 EXPLORATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
