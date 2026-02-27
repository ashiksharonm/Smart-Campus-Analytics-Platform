"""
Preprocessing: Clean raw student data.
- Impute missing values
- Cap outliers via IQR
- Type enforcement
- Schema validation
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_COLS = [
    "attendance_percentage",
    "avg_assignment_score",
    "lms_login_frequency",
    "library_visits_per_month",
    "disciplinary_actions",
]
BINARY_COLS = ["hostel_resident", "internet_access", "dropout"]
INT_COLS = ["lms_login_frequency", "library_visits_per_month", "disciplinary_actions", "semester"]

REQUIRED_COLS = [
    "student_id",
    "semester",
    "hostel_resident",
    "internet_access",
    "attendance_percentage",
    "avg_assignment_score",
    "lms_login_frequency",
    "library_visits_per_month",
    "disciplinary_actions",
    "dropout",
]


def validate_schema(df: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Median imputation for numeric columns."""
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    return df


def cap_outliers_iqr(df: pd.DataFrame, factor: float = 3.0) -> pd.DataFrame:
    """Cap outliers using IQR with a wider factor to preserve signal."""
    df = df.copy()
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - factor * iqr
        hi = q3 + factor * iqr
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def enforce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in INT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)
    df["attendance_percentage"] = df["attendance_percentage"].clip(0, 100)
    df["avg_assignment_score"] = df["avg_assignment_score"].clip(0, 100)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    validate_schema(df)
    df = impute_missing(df)
    df = cap_outliers_iqr(df)
    df = enforce_types(df)
    return df


def main():
    import sys
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.ingestion.loader import load_raw

    df = load_raw()
    df_clean = clean(df)
    out = PROCESSED_DIR / "students_clean.csv"
    df_clean.to_csv(out, index=False)
    print(f"✅ Cleaned data saved → {out}")
    print(f"   Shape: {df_clean.shape}  |  Missing: {df_clean.isnull().sum().sum()}")


if __name__ == "__main__":
    main()
