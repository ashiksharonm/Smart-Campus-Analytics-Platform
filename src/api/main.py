"""
FastAPI Application
Smart Campus Resource Utilization & Dropout Risk Analytics Platform
"""

from __future__ import annotations
import json
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.logger import get_logger
from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    EDASummaryResponse,
    HealthResponse,
    ModelMetricsResponse,
    PredictRequest,
    PredictResponse,
)
from src.models.predict import predict_batch, predict_one

logger = get_logger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

app = FastAPI(
    title="Smart Campus Analytics API",
    description=(
        "REST API for student dropout risk prediction, EDA summaries, "
        "and model performance metrics."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness probe."""
    return HealthResponse(status="ok", timestamp=time.time())


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Predict dropout risk for a single student.
    Returns probability, binary prediction, and risk level (Low/Medium/High).
    """
    try:
        student_dict = request.model_dump()
        student_dict = _add_engineered_features(student_dict)
        result = predict_one(student_dict, model_name=request.model_name)
        logger.info(
            "prediction",
            extra={"dropout_prob": result["dropout_prob"], "risk_level": result["risk_level"]},
        )
        return PredictResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("prediction_error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictResponse, tags=["Prediction"])
def batch_predict(request: BatchPredictRequest):
    """Batch dropout prediction for multiple students."""
    try:
        students = [s.model_dump() for s in request.students]
        students = [_add_engineered_features(s) for s in students]
        results = predict_batch(students, model_name=request.model_name)
        return BatchPredictResponse(predictions=results, count=len(results))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("batch_prediction_error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/eda-summary", response_model=EDASummaryResponse, tags=["Analytics"])
def eda_summary():
    """Return precomputed EDA statistics from the processed dataset."""
    path = DATA_DIR / "students_processed.csv"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="Processed data not found. Run the data pipeline first.",
        )
    try:
        df = pd.read_csv(path)
        numeric_cols = [
            "attendance_percentage", "avg_assignment_score", "lms_login_frequency",
            "library_visits_per_month", "disciplinary_actions",
            "engagement_index", "academic_risk_score", "composite_dropout_risk",
        ]
        summary_stats = {}
        for col in numeric_cols:
            if col in df.columns:
                summary_stats[col] = {
                    "mean": round(float(df[col].mean()), 4),
                    "std": round(float(df[col].std()), 4),
                    "min": round(float(df[col].min()), 4),
                    "max": round(float(df[col].max()), 4),
                    "median": round(float(df[col].median()), 4),
                }
        return EDASummaryResponse(
            total_records=len(df),
            dropout_rate=round(float(df["dropout"].mean()), 4),
            numeric_summary=summary_stats,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-metrics", response_model=ModelMetricsResponse, tags=["Analytics"])
def model_metrics():
    """Return last training metrics for all models."""
    metrics_path = MODELS_DIR / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Model metrics not found. Run `python src/models/train.py` first.",
        )
    with open(metrics_path) as f:
        data = json.load(f)
    return ModelMetricsResponse(metrics=data)


# ── Helper ───────────────────────────────────────────────────────────────────

def _add_engineered_features(s: dict) -> dict:
    """Compute engineered features inline for API inference (no CSV dependency)."""
    att = s.get("attendance_percentage", 70)
    score = s.get("avg_assignment_score", 60)
    lms = s.get("lms_login_frequency", 15)
    lib = s.get("library_visits_per_month", 3)
    disc = s.get("disciplinary_actions", 0)
    internet = s.get("internet_access", 1)
    hostel = s.get("hostel_resident", 0)

    lms_norm = (lms / 60) * 100
    lib_norm = (lib / 20) * 100
    engagement_index = round(0.40 * att + 0.35 * lms_norm + 0.25 * lib_norm, 4)
    academic_risk_score = round(min((100 - score) + 5 * disc, 100), 4)
    attendance_volatility = round(abs(att - 72) / 73, 4)
    resource_utilization_score = round(
        0.5 * (lib / 20) + 0.3 * internet + 0.2 * hostel, 4
    )
    composite_dropout_risk = round(
        0.40 * (academic_risk_score / 100)
        + 0.35 * (1 - engagement_index / 100)
        + 0.25 * (1 - resource_utilization_score),
        4,
    )
    s.update(
        {
            "engagement_index": engagement_index,
            "academic_risk_score": academic_risk_score,
            "attendance_volatility": attendance_volatility,
            "resource_utilization_score": resource_utilization_score,
            "composite_dropout_risk": composite_dropout_risk,
        }
    )
    return s
