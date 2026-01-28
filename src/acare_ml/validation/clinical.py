import pandas as pd


def validate_clinical_ranges(df: pd.DataFrame) -> dict:
    violations = {}

    if "acc_x" in df.columns:
        acc_range = 20
        acc_violations = (df[["acc_x", "acc_y", "acc_z"]].abs() > acc_range).any(axis=1).sum()
        violations["acceleration_out_of_range"] = int(acc_violations)

    if "gyro_x" in df.columns:
        gyro_range = 2000
        gyro_violations = (df[["gyro_x", "gyro_y", "gyro_z"]].abs() > gyro_range).any(axis=1).sum()
        violations["gyroscope_out_of_range"] = int(gyro_violations)

    return violations
