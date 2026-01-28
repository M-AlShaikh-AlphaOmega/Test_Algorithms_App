import pandas as pd
from pathlib import Path


def load_subject_metadata(metadata_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(metadata_path)


def get_subject_demographics(subject_id: str, metadata_df: pd.DataFrame) -> dict:
    subject_row = metadata_df[metadata_df["subject_id"] == subject_id]
    if subject_row.empty:
        return {}
    return subject_row.iloc[0].to_dict()
