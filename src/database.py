
"""
==================================================================
Day 6: SQLite Database Module
Project: AI-Based Fake Job Detection and Verification System
==================================================================

Handles:

1. User registration
2. User login verification
3. User information
4. Job analysis history

Passwords are stored as secure password hashes.

Existing analysis_history data is preserved.
==================================================================
"""

import os
import sqlite3
from datetime import datetime, timezone

from werkzeug.security import (
    generate_password_hash,
    check_password_hash,
)


# ==================================================================
# DATABASE LOCATION
# ==================================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    SCRIPT_DIR
)

DATABASE_DIR = os.path.join(
    PROJECT_ROOT,
    "database",
)

DATABASE_PATH = os.path.join(
    DATABASE_DIR,
    "fake_job_detection.db",
)


# ==================================================================
# CONNECTION
# ==================================================================

def get_connection():
    """
    Opens the SQLite database.

    Creates the database directory if necessary.
    """

    os.makedirs(
        DATABASE_DIR,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# ==================================================================
# INITIALIZE DATABASE
# ==================================================================

def initialize_database():
    """
    Creates required tables if they do not already exist.

    Existing records are preserved.
    """

    create_users_table_sql = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """

    create_analysis_table_sql = """
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analyzed_at TEXT NOT NULL,
            job_title TEXT NOT NULL,
            company_name TEXT,
            company_website TEXT,
            application_email TEXT,
            predicted_label TEXT NOT NULL,
            ml_risk_score REAL NOT NULL,
            ml_risk_level TEXT NOT NULL,
            verification_score REAL,
            verification_level TEXT,
            domain_comparison TEXT
        )
    """

    connection = get_connection()

    try:

        connection.execute(
            create_users_table_sql
        )

        connection.execute(
            create_analysis_table_sql
        )

        connection.commit()

    finally:

        connection.close()


# ==================================================================
# REGISTER USER
# ==================================================================

def register_user(
    username,
    email,
    password,
):
    """
    Creates a new user.

    Returns:
        user ID  -> successful
        None     -> username/email already exists
    """

    if username is None:
        return None

    if email is None:
        return None

    if password is None:
        return None

    username = username.strip()
    email = email.strip().lower()

    if not username or not email or not password:
        return None

    password_hash = generate_password_hash(
        password
    )

    created_at = datetime.now(
        timezone.utc
    ).isoformat(
        timespec="seconds"
    )

    sql = """
        INSERT INTO users (
            username,
            email,
            password_hash,
            created_at
        )
        VALUES (?, ?, ?, ?)
    """

    connection = get_connection()

    try:

        cursor = connection.execute(
            sql,
            (
                username,
                email,
                password_hash,
                created_at,
            ),
        )

        connection.commit()

        return cursor.lastrowid

    except sqlite3.IntegrityError:

        return None

    finally:

        connection.close()


# ==================================================================
# GET USER BY USERNAME OR EMAIL
# ==================================================================

def get_user_by_login(login_value):
    """
    Finds a user using either username or email.
    """

    if login_value is None:
        return None

    login_value = login_value.strip()

    if not login_value:
        return None

    sql = """
        SELECT
            id,
            username,
            email,
            password_hash,
            created_at
        FROM users
        WHERE username = ?
           OR email = ?
        LIMIT 1
    """

    connection = get_connection()

    try:

        row = connection.execute(
            sql,
            (
                login_value,
                login_value.lower(),
            ),
        ).fetchone()

    finally:

        connection.close()

    if row is None:
        return None

    return dict(row)


# ==================================================================
# VERIFY USER
# ==================================================================

def verify_user(
    login_value,
    password,
):
    """
    Verifies username/email and password.

    Returns user dictionary on success.
    Returns None on failure.
    """

    if not login_value or not password:
        return None

    user = get_user_by_login(
        login_value
    )

    if user is None:
        return None

    try:

        password_valid = check_password_hash(
            user["password_hash"],
            password,
        )

    except Exception:

        return None

    if not password_valid:
        return None

    return user


# ==================================================================
# GET USER BY ID
# ==================================================================

def get_user_by_id(user_id):
    """
    Retrieves a user using their ID.

    Password hash is deliberately not returned.
    """

    sql = """
        SELECT
            id,
            username,
            email,
            created_at
        FROM users
        WHERE id = ?
        LIMIT 1
    """

    connection = get_connection()

    try:

        row = connection.execute(
            sql,
            (user_id,),
        ).fetchone()

    finally:

        connection.close()

    if row is None:
        return None

    return dict(row)


# ==================================================================
# SAVE ANALYSIS
# ==================================================================

def save_analysis(
    job_title,
    predicted_label,
    ml_risk_score,
    ml_risk_level,
    company_name=None,
    company_website=None,
    application_email=None,
    verification_score=None,
    verification_level=None,
    domain_comparison=None,
):
    """
    Saves one completed job analysis.
    """

    analyzed_at = datetime.now(
        timezone.utc
    ).isoformat(
        timespec="seconds"
    )

    sql = """
        INSERT INTO analysis_history (
            analyzed_at,
            job_title,
            company_name,
            company_website,
            application_email,
            predicted_label,
            ml_risk_score,
            ml_risk_level,
            verification_score,
            verification_level,
            domain_comparison
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    values = (
        analyzed_at,
        job_title,
        company_name,
        company_website,
        application_email,
        predicted_label,
        ml_risk_score,
        ml_risk_level,
        verification_score,
        verification_level,
        domain_comparison,
    )

    connection = get_connection()

    try:

        cursor = connection.execute(
            sql,
            values,
        )

        connection.commit()

        return cursor.lastrowid

    finally:

        connection.close()


# ==================================================================
# GET ANALYSIS HISTORY
# ==================================================================

def get_analysis_history():
    """
    Returns all analysis records.

    Newest records appear first.
    """

    sql = """
        SELECT *
        FROM analysis_history
        ORDER BY analyzed_at DESC, id DESC
    """

    connection = get_connection()

    try:

        rows = connection.execute(
            sql
        ).fetchall()

    finally:

        connection.close()

    return [
        dict(row)
        for row in rows
    ]


# ==================================================================
# GET ONE ANALYSIS
# ==================================================================

def get_analysis_by_id(
    analysis_id
):
    """
    Retrieves one analysis record.
    """

    sql = """
        SELECT *
        FROM analysis_history
        WHERE id = ?
        LIMIT 1
    """

    connection = get_connection()

    try:

        row = connection.execute(
            sql,
            (analysis_id,),
        ).fetchone()

    finally:

        connection.close()

    if row is None:
        return None

    return dict(row)


# ==================================================================
# STANDALONE TEST
# ==================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("DATABASE MODULE TEST")
    print("=" * 70)

    print(
        "Database file path:",
        DATABASE_PATH,
    )

    initialize_database()

    print(
        "Database initialized successfully."
    )

    connection = get_connection()

    try:

        tables = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            ORDER BY name
            """
        ).fetchall()

    finally:

        connection.close()

    print("\nAvailable tables:")

    for table in tables:

        print(
            "-",
            table["name"],
        )

    existing_count = len(
        get_analysis_history()
    )

    print(
        "\nExisting analysis records:",
        existing_count,
    )

    print(
        "\nNo test records were inserted."
    )

    print("=" * 70)
    print("Database module is ready.")
    print("=" * 70)

