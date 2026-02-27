"""
Pydantic schemas for request/response validation.
"""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class StudentInput(BaseModel):
    attendance_percentage: float = Field(..., ge=0, le=100, description="Attendance %")
    avg_assignment_score: float = Field(..., ge=0, le=100, description="Average assignment score")
    lms_login_frequency: int = Field(..., ge=0, le=200, description="LMS logins per month")
    library_visits_per_month: int = Field(..., ge=0, le=50, description="Library visits per month")
    hostel_resident: int = Field(..., ge=0, le=1, description="1 = hostel resident")
    internet_access: int = Field(..., ge=0, le=1, description="1 = has internet access")
    disciplinary_actions: int = Field(..., ge=0, le=20, description="Disciplinary incidents count")
    semester: int = Field(..., ge=1, le=8, description="Current semester (1-8)")


class PredictRequest(StudentInput):
    model_name: str = Field(
        "gradient_boosting",
        description="Model to use: logistic_regression | random_forest | gradient_boosting",
    )


class PredictResponse(BaseModel):
    dropout_prob: float
    dropout_pred: int
    risk_level: str  # Low | Medium | High


class BatchPredictRequest(BaseModel):
    students: list[StudentInput]
    model_name: str = "gradient_boosting"


class BatchPrediction(BaseModel):
    index: int
    dropout_prob: float
    dropout_pred: int
    risk_level: str


class BatchPredictResponse(BaseModel):
    predictions: list[BatchPrediction]
    count: int


class HealthResponse(BaseModel):
    status: str
    timestamp: float


class EDASummaryResponse(BaseModel):
    total_records: int
    dropout_rate: float
    numeric_summary: dict[str, Any]


class ModelMetricsResponse(BaseModel):
    metrics: dict[str, Any]
