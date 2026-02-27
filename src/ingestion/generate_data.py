"""
Data Generation Script
Generates 10,000 realistic synthetic student records with correlations,
missing values, and outliers for the Smart Campus Analytics Platform.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)
N = 10_000

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def generate(n: int = N) -> pd.DataFrame:
    # ── Structural / Demographic ─────────────────────────────────────────────
    semester = rng.integers(1, 9, size=n)                        # 1-8
    hostel_resident = rng.binomial(1, 0.45, size=n)              # 45% hostellers
    internet_access = rng.binomial(1, 0.78, size=n)              # 78% have internet

    # ── Core Behavioural features (inter-correlated) ─────────────────────────
    # Attendance: lower for students with poor engagement
    base_attendance = rng.normal(72, 15, size=n)
    attendance_percentage = _clip(base_attendance, 0, 100)

    # LMS logins correlated positively with attendance
    lms_base = 0.6 * (attendance_percentage - 50) / 50 * 30 + rng.normal(15, 8, size=n)
    lms_login_frequency = _clip(lms_base, 0, 60).astype(int)

    # Library visits: hostellers visit more; internet access reduces need slightly
    library_base = (
        rng.poisson(3, size=n)
        + hostel_resident * rng.integers(0, 4, size=n)
        - internet_access * rng.integers(0, 2, size=n)
        + rng.normal(0, 1, size=n)
    )
    library_visits_per_month = _clip(library_base, 0, 20).astype(int)

    # Assignment scores correlated with attendance and LMS
    score_base = (
        0.4 * attendance_percentage
        + 0.3 * (lms_login_frequency / 60 * 100)
        + rng.normal(0, 10, size=n)
    )
    avg_assignment_score = _clip(score_base, 0, 100)

    # Disciplinary actions: negatively correlated with attendance
    disc_prob = _clip(0.3 - 0.003 * attendance_percentage, 0.01, 0.4)
    disciplinary_actions = rng.binomial(5, disc_prob, size=n)

    # ── Dropout label (non-linear decision boundary) ─────────────────────────
    # Logistic model with realistic ~17% base dropout rate
    log_odds = (
        1.5                               # intercept tuned for ~17% base rate
        + (-0.04 * attendance_percentage)
        + (-0.03 * avg_assignment_score)
        + (-0.03 * lms_login_frequency)
        + (-0.06 * library_visits_per_month)
        + (0.45 * disciplinary_actions)
        + (-0.40 * internet_access)
        + (-0.25 * hostel_resident)
        + (0.08 * semester)
        # Non-linear interaction: very low attendance + poor scores amplifies risk
        + np.where((attendance_percentage < 50) & (avg_assignment_score < 40), 1.5, 0.0)
        + rng.normal(0, 0.5, size=n)
    )
    dropout_prob = 1 / (1 + np.exp(-log_odds))
    dropout = rng.binomial(1, dropout_prob, size=n)

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "student_id": [f"S{str(i).zfill(5)}" for i in range(1, n + 1)],
            "semester": semester,
            "hostel_resident": hostel_resident,
            "internet_access": internet_access,
            "attendance_percentage": attendance_percentage.round(2),
            "avg_assignment_score": avg_assignment_score.round(2),
            "lms_login_frequency": lms_login_frequency,
            "library_visits_per_month": library_visits_per_month,
            "disciplinary_actions": disciplinary_actions,
            "dropout": dropout,
        }
    )

    # ── Introduce realistic missing values (MCAR / MAR) ──────────────────────
    missing_rules = {
        "attendance_percentage": 0.04,
        "avg_assignment_score": 0.03,
        "lms_login_frequency": 0.05,
        "library_visits_per_month": 0.02,
    }
    for col, rate in missing_rules.items():
        mask = rng.random(n) < rate
        df.loc[mask, col] = np.nan

    # ── Inject outliers (5 % of numeric rows) ────────────────────────────────
    outlier_idx = rng.choice(n, size=int(0.05 * n), replace=False)
    df.loc[outlier_idx, "avg_assignment_score"] = rng.uniform(95, 100, size=len(outlier_idx))
    df.loc[outlier_idx[:50], "lms_login_frequency"] = rng.integers(55, 60, size=50)
    df.loc[outlier_idx[50:100], "attendance_percentage"] = rng.uniform(1, 5, size=50)

    return df


def main():
    df = generate()
    out = OUTPUT_DIR / "students.csv"
    df.to_csv(out, index=False)
    print(f"✅ Generated {len(df):,} rows → {out}")
    print(df.describe().T.to_string())
    print(f"\nDropout rate: {df['dropout'].mean():.2%}")


if __name__ == "__main__":
    main()
