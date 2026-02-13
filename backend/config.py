"""
Configuration settings for the Parkinson's State Detection API.

This module centralizes all configuration parameters, making it easy to
adjust thresholds and settings without modifying core logic.
"""

from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """Application settings with sensible defaults for Parkinson's detection."""
    
    # API Metadata
    app_name: str = "aCare Parkinson's State Detection API"
    app_version: str = "0.1.0"
    debug: bool = True
    
    # Sensor Configuration
    expected_sample_rate_hz: int = 25  # Expected sampling rate
    min_samples: int = 125  # Minimum samples (~5 seconds at 25Hz)
    max_samples: int = 320  # Maximum samples (~10 seconds at 25Hz)
    
    # Accelerometer Column Names (flexible naming)
    acc_x_columns: list[str] = ["acc_x", "accel_x", "accelerometer_x", "ax", "x"]
    acc_y_columns: list[str] = ["acc_y", "accel_y", "accelerometer_y", "ay", "y"]
    acc_z_columns: list[str] = ["acc_z", "accel_z", "accelerometer_z", "az", "z"]

    # Gyroscope Column Names (flexible naming)
    gyro_x_columns: list[str] = ["gyro_x", "gyr_x", "gyroscope_x", "gx"]
    gyro_y_columns: list[str] = ["gyro_y", "gyr_y", "gyroscope_y", "gy"]
    gyro_z_columns: list[str] = ["gyro_z", "gyr_z", "gyroscope_z", "gz"]
    
    # Placeholder Thresholds (will be replaced by ML model)
    # These are illustrative values for the rule-based placeholder
    gravity_magnitude: float = 9.81  # m/s²
    gravity_tolerance: float = 2.0   # Acceptable deviation from gravity
    
    # Movement intensity thresholds
    low_activity_threshold: float = 0.3   # Below this = very low movement
    high_activity_threshold: float = 2.0  # Above this = high activity
    
    # Tremor detection (simplified)
    tremor_frequency_min: float = 4.0   # Hz - typical PD tremor range
    tremor_frequency_max: float = 7.0   # Hz
    
    # ML model path (relative to backend directory)
    # model_path: str = "parkinsons_model_acc_gyro.pkl"
    model_path: str = "rf_model.pkl"

    # Confidence thresholds
    confidence_threshold: float = 0.6  # Below this = UNKNOWN state
    
    class Config:
        env_prefix = "ACARE_"


settings = Settings()
