"""
Utility functions for data processing and validation.

This module handles CSV parsing, sensor data validation, and
preprocessing steps required before prediction.
"""

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from io import StringIO
from typing import Tuple, Optional, List, Dict
from config import settings


class DataValidationError(Exception):
    """Raised when sensor data fails validation checks."""
    pass


def find_accelerometer_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Identify accelerometer columns in the DataFrame.
    
    Supports flexible column naming conventions.
    
    Args:
        df: Input DataFrame with sensor data
        
    Returns:
        Tuple of (x_col, y_col, z_col) column names
        
    Raises:
        DataValidationError: If required columns are not found
    """
    columns_lower = {col.lower(): col for col in df.columns}
    
    x_col = y_col = z_col = None
    
    # Find X column
    for candidate in settings.acc_x_columns:
        if candidate.lower() in columns_lower:
            x_col = columns_lower[candidate.lower()]
            break
    
    # Find Y column
    for candidate in settings.acc_y_columns:
        if candidate.lower() in columns_lower:
            y_col = columns_lower[candidate.lower()]
            break
    
    # Find Z column
    for candidate in settings.acc_z_columns:
        if candidate.lower() in columns_lower:
            z_col = columns_lower[candidate.lower()]
            break
    
    if not all([x_col, y_col, z_col]):
        found = [c for c in [x_col, y_col, z_col] if c]
        raise DataValidationError(
            f"Missing accelerometer columns. Found: {found}. "
            f"Expected columns matching patterns for X, Y, Z axes. "
            f"Available columns: {list(df.columns)}"
        )
    
    return x_col, y_col, z_col


def find_gyroscope_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Identify gyroscope columns in the DataFrame.

    Returns:
        Tuple of (gx_col, gy_col, gz_col) column names

    Raises:
        DataValidationError: If required columns are not found
    """
    columns_lower = {col.lower(): col for col in df.columns}

    gx_col = gy_col = gz_col = None

    for candidate in settings.gyro_x_columns:
        if candidate.lower() in columns_lower:
            gx_col = columns_lower[candidate.lower()]
            break

    for candidate in settings.gyro_y_columns:
        if candidate.lower() in columns_lower:
            gy_col = columns_lower[candidate.lower()]
            break

    for candidate in settings.gyro_z_columns:
        if candidate.lower() in columns_lower:
            gz_col = columns_lower[candidate.lower()]
            break

    if not all([gx_col, gy_col, gz_col]):
        found = [c for c in [gx_col, gy_col, gz_col] if c]
        raise DataValidationError(
            f"Missing gyroscope columns. Found: {found}. "
            f"Expected columns matching patterns for gX, gY, gZ axes. "
            f"Available columns: {list(df.columns)}"
        )

    return gx_col, gy_col, gz_col


def parse_csv_content(content: bytes, filename: str) -> pd.DataFrame:
    """
    Parse CSV content into a pandas DataFrame.
    
    Args:
        content: Raw bytes of the CSV file
        filename: Original filename for error messages
        
    Returns:
        Parsed DataFrame
        
    Raises:
        DataValidationError: If CSV parsing fails
    """
    try:
        # Try UTF-8 first, fallback to latin-1
        try:
            text_content = content.decode('utf-8')
        except UnicodeDecodeError:
            text_content = content.decode('latin-1')
        
        df = pd.read_csv(StringIO(text_content))
        
        if df.empty:
            raise DataValidationError(f"CSV file '{filename}' is empty")
        
        return df
        
    except pd.errors.EmptyDataError:
        raise DataValidationError(f"CSV file '{filename}' contains no data")
    except pd.errors.ParserError as e:
        raise DataValidationError(f"Failed to parse CSV '{filename}': {str(e)}")


def validate_sensor_data(df: pd.DataFrame, filename: str):
    """
    Validate and prepare sensor data for prediction.

    Looks for accelerometer columns (required) and gyroscope columns (optional).

    Returns:
        Tuple of (validated_df, ax_col, ay_col, az_col, gx_col, gy_col, gz_col).
        Gyro columns are None when not present in the CSV.
    """
    # Find accelerometer columns (required)
    ax_col, ay_col, az_col = find_accelerometer_columns(df)

    # Find gyroscope columns (optional)
    try:
        gx_col, gy_col, gz_col = find_gyroscope_columns(df)
    except DataValidationError:
        gx_col = gy_col = gz_col = None

    # Build list of columns to validate
    all_cols = [ax_col, ay_col, az_col]
    if gx_col:
        all_cols.extend([gx_col, gy_col, gz_col])

    sensor_df = df[all_cols].copy()

    # Check for numeric data
    for col in all_cols:
        if not pd.api.types.is_numeric_dtype(sensor_df[col]):
            try:
                sensor_df[col] = pd.to_numeric(sensor_df[col], errors='coerce')
            except Exception:
                raise DataValidationError(
                    f"Column '{col}' contains non-numeric data that cannot be converted"
                )

    # Check for missing values
    total_cells = len(sensor_df) * len(all_cols)
    missing_pct = sensor_df.isna().sum().sum() / total_cells * 100
    if missing_pct > 10:
        raise DataValidationError(
            f"Excessive missing values: {missing_pct:.1f}% of data is missing"
        )

    sensor_df = sensor_df.dropna()

    # Validate sample count
    sample_count = len(sensor_df)
    if sample_count < settings.min_samples:
        raise DataValidationError(
            f"Insufficient data: {sample_count} samples. "
            f"Minimum required: {settings.min_samples} (~5 seconds at {settings.expected_sample_rate_hz}Hz)"
        )

    if sample_count > settings.max_samples:
        start_idx = (sample_count - settings.max_samples) // 2
        sensor_df = sensor_df.iloc[start_idx:start_idx + settings.max_samples]

    return sensor_df, ax_col, ay_col, az_col, gx_col, gy_col, gz_col


def compute_sensor_statistics(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    z_col: str
) -> dict:
    """
    Compute statistical features from accelerometer data.
    
    These statistics are used for both prediction and reporting.
    
    Args:
        df: Validated DataFrame with accelerometer data
        x_col, y_col, z_col: Column names for each axis
        
    Returns:
        Dictionary of computed statistics
    """
    # Extract arrays
    x = df[x_col].values
    y = df[y_col].values
    z = df[z_col].values
    
    # Compute magnitude (total acceleration)
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    
    # Basic statistics
    stats = {
        'sample_count': len(df),
        'duration_seconds': len(df) / settings.expected_sample_rate_hz,
        'mean_magnitude': float(np.mean(magnitude)),
        'std_magnitude': float(np.std(magnitude)),
        'min_magnitude': float(np.min(magnitude)),
        'max_magnitude': float(np.max(magnitude)),
    }
    
    # Gravity-normalized activity (remove gravity component)
    # Activity = deviation from expected gravity magnitude
    activity = np.abs(magnitude - settings.gravity_magnitude)
    stats['mean_activity'] = float(np.mean(activity))
    stats['std_activity'] = float(np.std(activity))
    
    # Movement intensity (normalized 0-1 scale)
    # Using RMS of activity as intensity measure
    rms_activity = np.sqrt(np.mean(activity**2))
    # Normalize to 0-1 range (cap at high_activity_threshold)
    stats['movement_intensity'] = float(
        min(rms_activity / settings.high_activity_threshold, 1.0)
    )
    
    # Per-axis statistics (useful for tremor detection)
    for axis_name, axis_data in [('x', x), ('y', y), ('z', z)]:
        # Detrend by removing mean
        detrended = axis_data - np.mean(axis_data)
        stats[f'{axis_name}_std'] = float(np.std(detrended))
        stats[f'{axis_name}_range'] = float(np.max(detrended) - np.min(detrended))
    
    # Simple frequency analysis (zero-crossing rate as proxy for tremor)
    # Higher zero-crossing rate suggests more oscillatory motion (tremor-like)
    for axis_name, axis_data in [('x', x), ('y', y), ('z', z)]:
        detrended = axis_data - np.mean(axis_data)
        zero_crossings = np.where(np.diff(np.signbit(detrended)))[0]
        zcr = len(zero_crossings) / (len(axis_data) / settings.expected_sample_rate_hz)
        stats[f'{axis_name}_zcr'] = float(zcr)  # Zero-crossings per second
    
    # Average zero-crossing rate (tremor indicator)
    stats['avg_zcr'] = float(np.mean([
        stats['x_zcr'], stats['y_zcr'], stats['z_zcr']
    ]))

    return stats


def _compute_axis_features(data: np.ndarray, sampling_rate: int, tremor_low: float, tremor_high: float, voluntary_high: float) -> Dict[str, float]:
    """Compute the 15 features for a single accelerometer axis."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=0)
    var = np.var(data, ddof=0)
    rms = np.sqrt(np.mean(data ** 2))
    peak_to_peak = np.max(data) - np.min(data)
    energy = np.sum(data ** 2)
    sma = np.mean(np.abs(data))
    skewness = float(scipy_stats.skew(data, bias=True))
    kurtosis = float(scipy_stats.kurtosis(data, bias=True, fisher=True))
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25

    # Zero-crossing rate (fraction of samples where sign changes)
    detrended = data - mean
    sign_changes = np.where(np.diff(np.signbit(detrended)))[0]
    zero_crossing = len(sign_changes) / n if n > 0 else 0.0

    # FFT-based features
    fft_vals = np.fft.rfft(data)
    fft_mag = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)

    # Dominant frequency
    if len(fft_mag) > 1:
        dominant_freq = freqs[1 + np.argmax(fft_mag[1:])]
    else:
        dominant_freq = 0.0

    # Power in frequency bands
    total_power = np.sum(fft_mag[1:]) if len(fft_mag) > 1 else 1e-10
    if total_power == 0:
        total_power = 1e-10

    tremor_mask = (freqs >= tremor_low) & (freqs <= tremor_high)
    tremor_power = np.sum(fft_mag[tremor_mask]) / total_power

    voluntary_mask = freqs <= voluntary_high
    voluntary_power = np.sum(fft_mag[voluntary_mask]) / total_power

    # Spectral entropy
    psd = fft_mag[1:] / total_power if len(fft_mag) > 1 else np.array([1.0])
    psd = psd[psd > 0]
    spectral_entropy = -np.sum(psd * np.log2(psd)) / np.log2(len(psd)) if len(psd) > 1 else 0.0

    return {
        'mean': mean, 'std': std, 'var': var, 'rms': rms,
        'peak_to_peak': peak_to_peak, 'energy': energy, 'sma': sma,
        'skewness': skewness, 'kurtosis': kurtosis, 'iqr': iqr,
        'zero_crossing': zero_crossing, 'dominant_freq': dominant_freq,
        'tremor_power': tremor_power, 'voluntary_power': voluntary_power,
        'spectral_entropy': spectral_entropy,
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def extract_window_features(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
    gx: np.ndarray, gy: np.ndarray, gz: np.ndarray,
    sampling_rate: int = 32,
    tremor_low: float = 4.0, tremor_high: float = 6.0,
    voluntary_high: float = 3.0,
) -> List[float]:
    """
    Extract the 103-feature vector from a single window of acc + gyro data.

    Feature order (matches model feature_names):
      acc_ax (15), acc_ay (15), acc_az (15),
      gyro_gx (15), gyro_gy (15), gyro_gz (15),
      acc_magnitude_mean/std/rms (3), gyro_magnitude_mean/std/rms (3),
      corr_ax_ay, corr_ax_az, corr_ay_az (3),
      corr_gx_gy, corr_gx_gz, corr_gy_gz (3),
      svm (1)
    Total = 90 + 6 + 6 + 1 = 103
    """
    features = []

    # Per-axis features: 6 axes x 15 features = 90
    for axis_data in [ax, ay, az, gx, gy, gz]:
        axis_feats = _compute_axis_features(axis_data, sampling_rate, tremor_low, tremor_high, voluntary_high)
        features.extend(axis_feats.values())

    # Accelerometer magnitude (3)
    acc_mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    features.append(float(np.mean(acc_mag)))
    features.append(float(np.std(acc_mag, ddof=0)))
    features.append(float(np.sqrt(np.mean(acc_mag ** 2))))

    # Gyroscope magnitude (3)
    gyro_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    features.append(float(np.mean(gyro_mag)))
    features.append(float(np.std(gyro_mag, ddof=0)))
    features.append(float(np.sqrt(np.mean(gyro_mag ** 2))))

    # Accelerometer correlations (3)
    features.append(_safe_corr(ax, ay))
    features.append(_safe_corr(ax, az))
    features.append(_safe_corr(ay, az))

    # Gyroscope correlations (3)
    features.append(_safe_corr(gx, gy))
    features.append(_safe_corr(gx, gz))
    features.append(_safe_corr(gy, gz))

    # SVM = acc magnitude mean (1)
    features.append(float(np.mean(acc_mag)))

    return features
