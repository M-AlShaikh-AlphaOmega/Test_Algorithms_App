import pandas as pd
import numpy as np


def check_data_quality(df: pd.DataFrame) -> dict:
    quality_report = {
        "total_rows": len(df),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": df.duplicated().sum(),
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    quality_report["outliers"] = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
        quality_report["outliers"][col] = int(outliers)

    return quality_report


def check_signal_quality(df: pd.DataFrame, sampling_rate: int = 100) -> dict:
    time_diffs = df["timestamp"].diff()
    expected_interval = 1.0 / sampling_rate
    irregular_samples = (time_diffs.abs() > expected_interval * 1.5).sum()

    return {
        "irregular_sampling": int(irregular_samples),
        "sampling_rate_hz": sampling_rate,
    }
