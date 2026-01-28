import pandas as pd


def validate_dataframe_schema(df: pd.DataFrame, required_columns: list[str]) -> bool:
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def validate_imu_schema(df: pd.DataFrame) -> bool:
    required = ["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    return validate_dataframe_schema(df, required)
