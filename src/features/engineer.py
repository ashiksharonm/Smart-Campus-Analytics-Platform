"""
Feature Engineering
Domain-driven features for the Smart Campus dropout risk model.

Features created:
  - engagement_index          : Weighted combination of attendance, LMS, library
  - academic_risk_score       : Inverse of academic performance
  - attendance_volatility     : Proxy using deviation buckets
  - resource_utilization_score: Usage of campus resources
  - composite_dropout_risk    : Weighted composite proxy
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def add_engagement_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engagement Index = 0.4*attendance + 0.35*lms_norm + 0.25*library_norm
    All inputs normalised to 0-100.
    """
    df = df.copy()
    lms_norm = (df["lms_login_frequency"] / df["lms_login_frequency"].max()) * 100
    lib_norm = (df["library_visits_per_month"] / df["library_visits_per_month"].max()) * 100
    df["engagement_index"] = (
        0.40 * df["attendance_percentage"]
        + 0.35 * lms_norm
        + 0.25 * lib_norm
    ).round(4)
    return df


def add_academic_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Academic Risk Score: higher score = more at risk.
    = 100 - avg_assignment_score + 5*disciplinary_actions
    capped at 100.
    """
    df = df.copy()
    df["academic_risk_score"] = (
        (100 - df["avg_assignment_score"]) + 5 * df["disciplinary_actions"]
    ).clip(0, 100).round(4)
    return df


def add_attendance_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attendance Volatility: Deviation from the mean (population-level proxy).
    Absolute deviation from median, normalised.
    High volatility = unusual attendance pattern.
    """
    df = df.copy()
    median_att = df["attendance_percentage"].median()
    df["attendance_volatility"] = (
        (df["attendance_percentage"] - median_att).abs() / (median_att + 1e-9)
    ).round(4)
    return df


def add_resource_utilization_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resource Utilization Score = 0.5*library_norm + 0.3*internet + 0.2*hostel
    """
    df = df.copy()
    lib_norm = (df["library_visits_per_month"] / (df["library_visits_per_month"].max() + 1e-9))
    df["resource_utilization_score"] = (
        0.5 * lib_norm
        + 0.3 * df["internet_access"]
        + 0.2 * df["hostel_resident"]
    ).round(4)
    return df


def add_composite_dropout_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite Dropout Risk Proxy (0-1 scale).
    Combines academic_risk_score, engagement_index (inverted), and resource utilization.
    Useful as an unsupervised / pre-model risk flag.
    """
    df = df.copy()
    risk = (
        0.40 * (df["academic_risk_score"] / 100)
        + 0.35 * (1 - df["engagement_index"] / 100)
        + 0.25 * (1 - df["resource_utilization_score"])
    )
    df["composite_dropout_risk"] = risk.round(4)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in sequence."""
    df = add_engagement_index(df)
    df = add_academic_risk_score(df)
    df = add_attendance_volatility(df)
    df = add_resource_utilization_score(df)
    df = add_composite_dropout_risk(df)
    return df


def main():
    import sys
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    clean_path = PROCESSED_DIR / "students_clean.csv"
    if not clean_path.exists():
        raise FileNotFoundError(
            "Run `python src/preprocessing/clean.py` first."
        )
    df = pd.read_csv(clean_path)
    df = engineer_features(df)
    out = PROCESSED_DIR / "students_processed.csv"
    df.to_csv(out, index=False)
    print(f"✅ Feature-engineered data saved → {out}")
    new_cols = ["engagement_index", "academic_risk_score",
                "attendance_volatility", "resource_utilization_score",
                "composite_dropout_risk"]
    print(df[new_cols].describe().T.to_string())


if __name__ == "__main__":
    main()
