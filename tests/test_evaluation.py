import numpy as np
from acare_ml.evaluation.metrics import calculate_metrics, calculate_clinical_metrics


def test_calculate_metrics():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    metrics = calculate_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_calculate_clinical_metrics():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    metrics = calculate_clinical_metrics(y_true, y_pred)
    assert "sensitivity" in metrics
    assert "specificity" in metrics
    assert 0 <= metrics["sensitivity"] <= 1
