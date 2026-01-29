import numpy as np
import pandas as pd


def extract_statistical_features(window: pd.DataFrame) -> dict:
    features = {}
    for col in window.select_dtypes(include=[np.number]).columns:
        features[f"{col}_mean"] = window[col].mean()
        features[f"{col}_std"] = window[col].std()
    return features
