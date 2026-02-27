"""Tests for src/features/engineer.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.engineer import (  # noqa: E402
    add_academic_risk_score,
    add_attendance_volatility,
    add_composite_dropout_risk,
    add_engagement_index,
    add_resource_utilization_score,
    engineer_features,
)


def test_engagement_index_range(sample_dataframe):
    result = add_engagement_index(sample_dataframe)
    assert "engagement_index" in result.columns
    assert result["engagement_index"].between(0, 100).all()


def test_academic_risk_score_range(sample_dataframe):
    result = add_academic_risk_score(sample_dataframe)
    assert "academic_risk_score" in result.columns
    assert result["academic_risk_score"].between(0, 100).all()


def test_attendance_volatility_non_negative(sample_dataframe):
    result = add_attendance_volatility(sample_dataframe)
    assert "attendance_volatility" in result.columns
    assert (result["attendance_volatility"] >= 0).all()


def test_resource_utilization_range(sample_dataframe):
    result = add_resource_utilization_score(sample_dataframe)
    assert "resource_utilization_score" in result.columns
    assert result["resource_utilization_score"].between(0, 1).all()


def test_composite_risk_range(sample_dataframe):
    df = add_engagement_index(sample_dataframe)
    df = add_academic_risk_score(df)
    df = add_resource_utilization_score(df)
    result = add_composite_dropout_risk(df)
    assert "composite_dropout_risk" in result.columns
    assert result["composite_dropout_risk"].between(0, 1).all()


def test_engineer_features_adds_all_cols(sample_dataframe):
    result = engineer_features(sample_dataframe)
    expected = [
        "engagement_index", "academic_risk_score",
        "attendance_volatility", "resource_utilization_score",
        "composite_dropout_risk",
    ]
    for col in expected:
        assert col in result.columns, f"Missing column: {col}"


def test_high_attendance_low_risk(sample_dataframe):
    df = sample_dataframe.copy()
    df["attendance_percentage"] = 95
    df["lms_login_frequency"] = 55
    df["avg_assignment_score"] = 95
    df["disciplinary_actions"] = 0
    result = engineer_features(df)
    assert result["composite_dropout_risk"].mean() < 0.5


def test_low_attendance_high_risk(sample_dataframe):
    df = sample_dataframe.copy()
    df["attendance_percentage"] = 10
    df["lms_login_frequency"] = 1
    df["avg_assignment_score"] = 20
    df["disciplinary_actions"] = 5
    result = engineer_features(df)
    assert result["composite_dropout_risk"].mean() > 0.4
