import pandas as pd
import pytest
from acare_ml.validation.schema import validate_dataframe_schema, validate_imu_schema
from acare_ml.validation.quality import check_data_quality


def test_validate_schema_success():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert validate_dataframe_schema(df, ["a", "b"]) is True


def test_validate_schema_missing_columns():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe_schema(df, ["a", "b"])


def test_check_data_quality():
    df = pd.DataFrame({"a": [1, 2, None, 4], "b": [5, 6, 7, 8]})
    report = check_data_quality(df)
    assert report["total_rows"] == 4
    assert report["missing_values"]["a"] == 1
