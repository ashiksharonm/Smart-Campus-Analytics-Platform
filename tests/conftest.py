"""
Shared pytest fixtures for Smart Campus Analytics tests.
"""

import pandas as pd
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_student_dict():
    return {
        "student_id": "S00001",
        "semester": 3,
        "hostel_resident": 1,
        "internet_access": 1,
        "attendance_percentage": 75.0,
        "avg_assignment_score": 68.0,
        "lms_login_frequency": 18,
        "library_visits_per_month": 4,
        "disciplinary_actions": 0,
        "dropout": 0,
    }


@pytest.fixture
def sample_dataframe(sample_student_dict):
    rows = []
    for i in range(20):
        row = sample_student_dict.copy()
        row["student_id"] = f"S{str(i).zfill(5)}"
        row["attendance_percentage"] = max(10, 75 - i * 2)
        row["avg_assignment_score"] = max(10, 68 - i)
        row["dropout"] = 1 if i > 15 else 0
        rows.append(row)
    return pd.DataFrame(rows)
