"""
Evaluation Metrics
Computes classification metrics and feature importance for trained pipelines.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: Optional[list[str]] = None,
) -> dict:
    """
    Return a structured evaluation dict with all standard metrics
    plus feature importance (if the final estimator supports it).
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred).tolist()
    result = {
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": cm,
    }

    # Feature importance
    clf = pipeline.named_steps.get("clf") or pipeline.steps[-1][1]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_.tolist()
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0]).tolist()
    else:
        importances = []

    if importances and feature_names:
        fi = sorted(
            zip(feature_names, importances), key=lambda x: x[1], reverse=True
        )
        result["feature_importance"] = [
            {"feature": f, "importance": round(v, 6)} for f, v in fi
        ]
    else:
        result["feature_importance"] = []

    return result
