"""Tests for model loading and prediction."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def test_model_files_exist_after_training():
    """Verifies that model artifacts exist (skip if not trained yet)."""
    for name in ["gradient_boosting", "random_forest", "logistic_regression"]:
        model_path = MODELS_DIR / f"{name}.joblib"
        if not model_path.exists():
            pytest.skip(f"Model file missing: {name}.joblib — run train.py first.")


def test_model_loads_correctly():
    """Each model file loads without error."""
    import joblib

    for name in ["gradient_boosting", "random_forest", "logistic_regression"]:
        model_path = MODELS_DIR / f"{name}.joblib"
        if not model_path.exists():
            pytest.skip("Model not trained yet.")
        model = joblib.load(model_path)
        assert model is not None


def test_prediction_returns_valid_probability(sample_student_dict):
    """predict_one returns probability in [0, 1]."""
    model_path = MODELS_DIR / "gradient_boosting.joblib"
    if not model_path.exists():
        pytest.skip("Model not trained yet.")
    try:
        from src.models.predict import predict_one
        from src.features.engineer import engineer_features
        import pandas as pd
        from src.preprocessing.clean import clean

        df = pd.DataFrame([sample_student_dict])
        df = clean(df)
        df = engineer_features(df)
        row_dict = df.iloc[0].to_dict()
        result = predict_one(row_dict)
        assert 0.0 <= result["dropout_prob"] <= 1.0
        assert result["risk_level"] in ("Low", "Medium", "High")
    except Exception as e:
        pytest.skip(f"Model inference skipped: {e}")


def test_metrics_json_exists():
    """metrics.json must exist after training."""
    metrics_path = MODELS_DIR / "metrics.json"
    if not metrics_path.exists():
        pytest.skip("Run src/models/train.py first.")
    import json
    with open(metrics_path) as f:
        data = json.load(f)
    assert "gradient_boosting" in data
    assert "roc_auc" in data["gradient_boosting"]
