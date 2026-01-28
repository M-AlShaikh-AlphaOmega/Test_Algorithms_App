from pathlib import Path
import pandas as pd


def read_raw_imu(file_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(file_path)
