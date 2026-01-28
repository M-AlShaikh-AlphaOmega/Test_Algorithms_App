import json
from pathlib import Path


def save_evaluation_report(metrics: dict, output_path: str | Path):
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


def generate_classification_report(y_true, y_pred, output_path: str | Path):
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, output_dict=True)
    save_evaluation_report(report, output_path)
