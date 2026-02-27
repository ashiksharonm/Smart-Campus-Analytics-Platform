"""Tests for preprocessing/clean.py"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.clean import (
    cap_outliers_iqr,
    clean,
    enforce_types,
    impute_missing,
    validate_schema,
)


def test_validate_schema_passes(sample_dataframe):
    validate_schema(sample_dataframe)  # should not raise


def test_validate_schema_fails_missing_col(sample_dataframe):
    bad_df = sample_dataframe.drop(columns=["dropout"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(bad_df)


def test_impute_missing_fills_nulls(sample_dataframe):
    df = sample_dataframe.copy()
    df.loc[0, "attendance_percentage"] = np.nan
    df.loc[1, "avg_assignment_score"] = np.nan
    result = impute_missing(df)
    assert result["attendance_percentage"].isnull().sum() == 0
    assert result["avg_assignment_score"].isnull().sum() == 0


def test_impute_uses_median(sample_dataframe):
    df = sample_dataframe.copy()
    median_val = df["lms_login_frequency"].median()
    df.loc[0, "lms_login_frequency"] = np.nan
    result = impute_missing(df)
    assert result.loc[0, "lms_login_frequency"] == median_val


def test_cap_outliers_does_not_increase_range(sample_dataframe):
    df = sample_dataframe.copy()
    df.loc[0, "avg_assignment_score"] = 999  # extreme outlier
    result = cap_outliers_iqr(df)
    assert result["avg_assignment_score"].max() <= 100 + 1e-6


def test_enforce_types_integers(sample_dataframe):
    result = enforce_types(sample_dataframe)
    assert result["lms_login_frequency"].dtype in [int, "int64"]
    assert result["hostel_resident"].dtype in [int, "int64"]


def test_clean_pipeline_no_nulls(sample_dataframe):
    df = sample_dataframe.copy()
    df.loc[0, "attendance_percentage"] = None
    result = clean(df)
    assert result.isnull().sum().sum() == 0


def test_attendance_clipped_0_100(sample_dataframe):
    df = sample_dataframe.copy()
    result = clean(df)
    assert result["attendance_percentage"].between(0, 100).all()
