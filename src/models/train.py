"""
Model Training Script
Trains Logistic Regression, Random Forest, and Gradient Boosting classifiers.
Saves models and metrics to disk.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure project root on path when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import evaluate  # noqa: E402

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "attendance_percentage",
    "avg_assignment_score",
    "lms_login_frequency",
    "library_visits_per_month",
    "disciplinary_actions",
    "hostel_resident",
    "internet_access",
    "semester",
    "engagement_index",
    "academic_risk_score",
    "attendance_volatility",
    "resource_utilization_score",
    "composite_dropout_risk",
]
TARGET = "dropout"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    path = PROCESSED_DIR / "students_processed.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Processed data not found. Run the ingestion + preprocessing + feature pipelines first."
        )
    df = pd.read_csv(path)
    X = df[FEATURE_COLS]
    y = df[TARGET]
    return X, y


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=12,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=5,
                        subsample=0.8,
                        random_state=42,
                    ),
                )
            ]
        ),
    }


def train():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}  |  Dropout rate: {y.mean():.2%}")

    models = build_models()
    all_metrics: dict[str, dict] = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, pipe in models.items():
        print(f"\n── Training: {name} ──")
        pipe.fit(X_train, y_train)

        # CV AUC
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        print(f"   CV AUC (5-fold): {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

        # Detailed test metrics
        metrics = evaluate(pipe, X_test, y_test, feature_names=FEATURE_COLS)
        metrics["cv_auc_mean"] = round(float(cv_auc.mean()), 4)
        metrics["cv_auc_std"] = round(float(cv_auc.std()), 4)
        all_metrics[name] = metrics

        print(
            f"   Test AUC: {metrics['roc_auc']:.4f}  "
            f"F1: {metrics['f1']:.4f}  "
            f"Precision: {metrics['precision']:.4f}  "
            f"Recall: {metrics['recall']:.4f}"
        )

        # Save model
        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(pipe, model_path)
        print(f"   Saved → {model_path}")

    # Persist metrics
    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n✅ All metrics saved → {metrics_path}")

    # Save feature list for inference
    feat_path = MODELS_DIR / "feature_cols.json"
    with open(feat_path, "w") as f:
        json.dump(FEATURE_COLS, f)

    return all_metrics


if __name__ == "__main__":
    train()
