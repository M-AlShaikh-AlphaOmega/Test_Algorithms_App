import pandas as pd
import numpy as np


def aggregate_predictions_by_subject(predictions: pd.DataFrame, subject_ids, method: str = "majority") -> dict:
    df = pd.DataFrame({"subject_id": subject_ids, "prediction": predictions})

    if method == "majority":
        subject_preds = df.groupby("subject_id")["prediction"].apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else 0)
    elif method == "mean":
        subject_preds = df.groupby("subject_id")["prediction"].mean().round()
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    return subject_preds.to_dict()
