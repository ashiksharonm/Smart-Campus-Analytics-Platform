"""
Inference module: load model and predict on dicts or DataFrames.
"""

from __future__ import annotations
import json
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_MODEL = "gradient_boosting"


def load_model(name: str = DEFAULT_MODEL):
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model '{name}' not found at {path}. Run `python src/models/train.py` first."
        )
    return joblib.load(path)


def load_feature_cols() -> list[str]:
    path = MODELS_DIR / "feature_cols.json"
    if not path.exists():
        raise FileNotFoundError("feature_cols.json not found. Run training first.")
    with open(path) as f:
        return json.load(f)


def predict_one(
    student: dict,
    model_name: str = DEFAULT_MODEL,
) -> dict:
    """
    Predict dropout probability for a single student dict.
    Returns: {dropout_prob, dropout_pred, risk_level}
    """
    pipe = load_model(model_name)
    features = load_feature_cols()
    row = pd.DataFrame([student])[features]
    prob = float(pipe.predict_proba(row)[0, 1])
    pred = int(prob >= 0.5)
    risk_level = "High" if prob >= 0.65 else ("Medium" if prob >= 0.35 else "Low")
    return {"dropout_prob": round(prob, 4), "dropout_pred": pred, "risk_level": risk_level}


def predict_batch(
    students: list[dict],
    model_name: str = DEFAULT_MODEL,
) -> list[dict]:
    """Predict for a list of student dicts."""
    pipe = load_model(model_name)
    features = load_feature_cols()
    df = pd.DataFrame(students)[features]
    probs = pipe.predict_proba(df)[:, 1]
    results = []
    for i, prob in enumerate(probs):
        prob = float(prob)
        pred = int(prob >= 0.5)
        risk_level = "High" if prob >= 0.65 else ("Medium" if prob >= 0.35 else "Low")
        results.append(
            {
                "index": i,
                "dropout_prob": round(prob, 4),
                "dropout_pred": pred,
                "risk_level": risk_level,
            }
        )
    return results
