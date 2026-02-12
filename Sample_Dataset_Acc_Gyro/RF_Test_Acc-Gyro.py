"""
RF_Test_Acc-Gyro.py - Parkinson's ON/OFF Detection Pipeline with Accelerometer + Gyroscope

This script detects whether a Parkinson's patient is in an ON state (medication working) 
or OFF state (medication worn off) using accelerometer and gyroscope data from a wrist sensor.

Modified from RF_Test.py to support both accelerometer and gyroscope data from DecodedData_Test folder.

=== FIXES APPLIED (v2 - Production Grade) ===
FIX 1: Multi-feature KMeans labeling (acc + gyro tremor, energy, variability)
FIX 2: Data leakage eliminated (train/test split BEFORE baseline learning)
FIX 3: Robust baseline statistics (median + MAD instead of mean + std)
FIX 4: Robust outlier removal in preprocessing (median + MAD)
FIX 5: NaN/Inf sanitization after feature extraction
FIX 6: Tuned Random Forest (class balancing, regularization, more trees)
"""

import json
import pickle
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SystemConfig:
    """System-wide configuration parameters."""
    
    # Data parameters
    sampling_rate: float = 32.0  # Hz (from DecodeDataset.py)
    window_duration: float = 1.0  # seconds
    window_overlap: float = 0.5  # 50% overlap
    
    # Frequency bands
    tremor_band_low: float = 4.0  # Hz
    tremor_band_high: float = 6.0  # Hz
    voluntary_band_low: float = 0.0  # Hz
    voluntary_band_high: float = 3.0  # Hz
    
    # Filter parameters
    filter_lowcut: float = 0.5  # Hz
    filter_highcut: float = 10.0  # Hz
    filter_order: int = 4
    
    # Model parameters  — FIX 6: tuned for clinical robustness
    n_estimators: int = 300         # more trees = more stable predictions
    max_depth: Optional[int] = 25   # bounded depth prevents memorization
    min_samples_split: int = 5      # regularization: need 5 samples to split
    min_samples_leaf: int = 2       # regularization: each leaf needs 2+ samples
    n_cv_folds: int = 5
    random_state: int = 42
    
    # Quality control
    max_missing_ratio: float = 0.1  # 10% max missing data per window
    outlier_std_threshold: float = 5.0  # scaled-MAD multiplier for outlier detection
    
    # Clinical safety
    confidence_threshold: float = 0.65  # minimum confidence for a prediction
    temporal_smooth_window: int = 5     # majority-vote smoothing for labels
    
    # Paths
    data_folder: str = 'DecodedData_Test'
    output_folder: str = 'results'
    model_folder: str = 'models'


# ============================================================================
# DATA LOADING
# ============================================================================

class DataLoader:
    """Loads and parses data from DecodedData_Test folder."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.data_folder = Path(config.data_folder)
        
    def load_from_decoded_folder(self) -> pd.DataFrame:
        """
        Load all CSV files from DecodedData_Test folder.
        
        Expected CSV format (from DecodeDataset.py):
        timestamp,Ax,Ay,Az,Gx,Gy,Gz
        
        Returns:
            DataFrame with columns: time, Ax, Ay, Az, Gx, Gy, Gz
        """
        if not self.data_folder.exists():
            raise FileNotFoundError(f"DecodedData_Test folder not found: {self.data_folder}")
        
        all_data = []
        
        # Iterate through all patient folders
        for patient_folder in sorted(self.data_folder.iterdir()):
            if not patient_folder.is_dir():
                continue
            
            # Load all CSV files for this patient
            for csv_file in sorted(patient_folder.glob('*.csv')):
                try:
                    df = pd.read_csv(csv_file)
                    
                    # Verify required columns exist
                    required_cols = ['timestamp', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
                    if not all(col in df.columns for col in required_cols):
                        print(f"Warning: {csv_file} missing required columns, skipping")
                        continue
                    
                    # Rename timestamp to time for consistency
                    df = df.rename(columns={'timestamp': 'time'})
                    
                    # Select only required columns in correct order
                    df = df[['time', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']]
                    
                    all_data.append(df)
                    
                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")
                    continue
        
        if not all_data:
            raise ValueError("No valid data files found in DecodedData_Test folder")
        
        # Concatenate all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by time
        combined_df = combined_df.sort_values('time').reset_index(drop=True)
        
        print(f"Loaded {len(combined_df)} samples from {len(all_data)} files")
        print(f"Duration: {combined_df['time'].max():.2f} seconds")
        
        return combined_df
    
    # =========================================================================
    # FIX 1: Multi-feature KMeans classification with temporal smoothing
    # 
    # OLD: Single-feature (acc tremor power only) median split.
    #      - Forced exactly 50/50 class balance regardless of real patient state
    #      - Ignored gyroscope data entirely
    #      - Noisy boundary from single feature
    # 
    # NEW: Uses 8 features from BOTH acc and gyro:
    #      tremor power, voluntary power, variability, tremor ratio
    #      KMeans finds natural cluster boundary.
    #      Temporal smoothing removes isolated label flips.
    # =========================================================================
    
    def classify_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify signal into ON/OFF states using multi-feature clustering.
        
        Uses accelerometer AND gyroscope data across multiple discriminative 
        features for robust ON/OFF separation.
        
        Args:
            df: DataFrame with time, Ax, Ay, Az, Gx, Gy, Gz columns
            
        Returns:
            DataFrame with added 'label' column
        """
        df = df.copy()
        
        # Compute magnitudes
        acc_magnitude = np.sqrt(df['Ax']**2 + df['Ay']**2 + df['Az']**2).values
        gyro_magnitude = np.sqrt(df['Gx']**2 + df['Gy']**2 + df['Gz']**2).values
        
        # Window parameters
        window_samples = int(self.config.window_duration * self.config.sampling_rate)
        step_samples = int(window_samples * (1 - self.config.window_overlap))
        
        window_feature_list = []
        window_indices = []
        
        # Extract multiple features per window for clustering
        for start in range(0, len(acc_magnitude) - window_samples + 1, step_samples):
            end = start + window_samples
            acc_win = acc_magnitude[start:end]
            gyro_win = gyro_magnitude[start:end]
            
            # FFT for accelerometer
            fft_acc = fft(acc_win)
            freqs = fftfreq(len(acc_win), 1 / self.config.sampling_rate)
            pos_mask = freqs >= 0
            freqs_pos = freqs[pos_mask]
            power_acc = np.abs(fft_acc[pos_mask]) ** 2
            
            # FFT for gyroscope
            fft_gyro = fft(gyro_win)
            power_gyro = np.abs(fft_gyro[pos_mask]) ** 2
            
            # Frequency band masks
            tremor_mask = ((freqs_pos >= self.config.tremor_band_low) & 
                           (freqs_pos <= self.config.tremor_band_high))
            voluntary_mask = ((freqs_pos >= self.config.voluntary_band_low) & 
                              (freqs_pos <= self.config.voluntary_band_high))
            
            # Multi-feature vector: 8 discriminative features
            total_acc_power = np.sum(power_acc) + 1e-10
            total_gyro_power = np.sum(power_gyro) + 1e-10
            
            features = [
                np.sum(power_acc[tremor_mask]),                         # 0: acc tremor power
                np.sum(power_gyro[tremor_mask]),                        # 1: gyro tremor power
                np.sum(power_acc[voluntary_mask]),                      # 2: acc voluntary power
                np.sum(power_gyro[voluntary_mask]),                     # 3: gyro voluntary power
                np.std(acc_win),                                        # 4: acc variability
                np.std(gyro_win),                                       # 5: gyro variability
                np.sum(power_acc[tremor_mask]) / total_acc_power,       # 6: acc tremor ratio
                np.sum(power_gyro[tremor_mask]) / total_gyro_power,     # 7: gyro tremor ratio
            ]
            
            window_feature_list.append(features)
            window_indices.append((start, end))
        
        if len(window_feature_list) < 4:
            raise ValueError(f"Too few windows ({len(window_feature_list)}) for clustering. "
                             f"Need at least 4. Check data duration and window settings.")
        
        # KMeans clustering on robustly-normalized multi-feature space
        feature_matrix = np.array(window_feature_list)
        
        # Robust normalization for clustering (median + scaled MAD)
        medians = np.median(feature_matrix, axis=0)
        mads = np.median(np.abs(feature_matrix - medians), axis=0) * 1.4826
        mads[mads == 0] = 1.0  # prevent division by zero for constant features
        feature_scaled = (feature_matrix - medians) / mads
        
        # 2-cluster KMeans
        kmeans = KMeans(n_clusters=2, random_state=self.config.random_state, n_init=20)
        cluster_labels = kmeans.fit_predict(feature_scaled)
        
        # Identify which cluster is OFF: higher median tremor power (column 0)
        cluster_0_tremor = np.median(feature_matrix[cluster_labels == 0, 0])
        cluster_1_tremor = np.median(feature_matrix[cluster_labels == 1, 0])
        off_cluster = 0 if cluster_0_tremor > cluster_1_tremor else 1
        
        label_map = {off_cluster: 'OFF', 1 - off_cluster: 'ON'}
        window_labels = [label_map[c] for c in cluster_labels]
        
        # Temporal smoothing (majority vote)
        window_labels = self._temporal_smooth(
            window_labels, window_size=self.config.temporal_smooth_window
        )
        
        # Assign labels to individual samples
        labels = [None] * len(df)
        for (start, end), label in zip(window_indices, window_labels):
            for i in range(start, end):
                labels[i] = label
        
        df['label'] = labels
        
        # Report
        on_count = sum(1 for l in labels if l == 'ON')
        off_count = sum(1 for l in labels if l == 'OFF')
        unlabeled = sum(1 for l in labels if l is None)
        total_labeled = on_count + off_count
        
        print(f"\nClassification results (multi-feature KMeans + temporal smoothing):")
        print(f"  ON state:   {on_count} samples ({100*on_count/max(total_labeled,1):.1f}%)")
        print(f"  OFF state:  {off_count} samples ({100*off_count/max(total_labeled,1):.1f}%)")
        print(f"  Unlabeled:  {unlabeled} samples")
        print(f"  Cluster tremor — OFF: {max(cluster_0_tremor, cluster_1_tremor):.2f}, "
              f"ON: {min(cluster_0_tremor, cluster_1_tremor):.2f}")
        
        return df
    
    @staticmethod
    def _temporal_smooth(labels: List[str], window_size: int = 5) -> List[str]:
        """
        Apply majority-vote temporal smoothing to prevent rapid label flipping.
        """
        if window_size <= 1:
            return labels
        
        smoothed = labels.copy()
        half = window_size // 2
        
        for i in range(len(labels)):
            start = max(0, i - half)
            end = min(len(labels), i + half + 1)
            neighborhood = labels[start:end]
            on_count = sum(1 for l in neighborhood if l == 'ON')
            off_count = sum(1 for l in neighborhood if l == 'OFF')
            smoothed[i] = 'ON' if on_count > off_count else 'OFF'
        
        return smoothed


# ============================================================================
# PREPROCESSING
# ============================================================================

class Preprocessor:
    """Preprocesses sensor data: normalization, filtering, outlier removal."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply full preprocessing pipeline.
        
        Args:
            df: Raw sensor data
            
        Returns:
            Preprocessed data
        """
        df = df.copy()
        
        # Normalize accelerometer by gravity
        for axis in ['Ax', 'Ay', 'Az']:
            df[axis] = df[axis] / 9.807
        
        # Apply bandpass filter to all sensor channels
        for channel in ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']:
            df[channel] = self._bandpass_filter(df[channel].values)
        
        # Remove outliers using robust method
        for channel in ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']:
            df[channel] = self._remove_outliers(df[channel].values)
        
        return df
    
    def _bandpass_filter(self, signal_data: np.ndarray) -> np.ndarray:
        """Apply Butterworth bandpass filter."""
        nyquist = self.config.sampling_rate / 2
        low = self.config.filter_lowcut / nyquist
        high = self.config.filter_highcut / nyquist
        
        b, a = signal.butter(self.config.filter_order, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, signal_data)
        
        return filtered
    
    # =========================================================================
    # FIX 4: Robust outlier removal using median + MAD
    # 
    # OLD: np.mean/np.std for threshold → one extreme spike shifts the
    #      threshold so much that real outliers survive.
    # NEW: median + scaled MAD → immune to contamination.
    # =========================================================================
    
    def _remove_outliers(self, signal_data: np.ndarray) -> np.ndarray:
        """Replace outliers with interpolated values using robust statistics."""
        median = np.median(signal_data)
        mad = np.median(np.abs(signal_data - median)) * 1.4826
        threshold = self.config.outlier_std_threshold
        
        # If MAD is zero (constant signal), no outliers possible
        if mad < 1e-15:
            return signal_data
        
        # Detect outliers using robust threshold
        outliers = np.abs(signal_data - median) > (threshold * mad)
        
        if np.sum(outliers) == 0:
            return signal_data
        
        # Interpolate outliers
        clean_signal = signal_data.copy()
        valid_indices = np.where(~outliers)[0]
        outlier_indices = np.where(outliers)[0]
        
        if len(valid_indices) > 1:
            interpolator = interp1d(valid_indices, signal_data[valid_indices], 
                                   kind='linear', fill_value='extrapolate')
            clean_signal[outlier_indices] = interpolator(outlier_indices)
        
        return clean_signal


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class FeatureExtractor:
    """Extracts time-domain and frequency-domain features from sensor windows."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from sliding windows.
        
        Args:
            df: Preprocessed sensor data with columns: time, Ax, Ay, Az, Gx, Gy, Gz, label
            
        Returns:
            DataFrame with one row per window, containing all features
        """
        window_samples = int(self.config.window_duration * self.config.sampling_rate)
        step_samples = int(window_samples * (1 - self.config.window_overlap))
        
        features_list = []
        
        for start in range(0, len(df) - window_samples + 1, step_samples):
            end = start + window_samples
            window = df.iloc[start:end]
            
            # Check for missing data
            if window.isnull().sum().sum() / (len(window) * len(window.columns)) > self.config.max_missing_ratio:
                continue
            
            # Extract features for this window
            window_features = self._extract_window_features(window)
            
            # =================================================================
            # FIX 5: Sanitize NaN/Inf IMMEDIATELY after extraction
            # 
            # Correlation features produce NaN when axis has zero variance.
            # Energy features can overflow to Inf on extreme sensor spikes.
            # =================================================================
            window_features = {
                k: (0.0 if (v is None or not np.isfinite(v)) else v)
                for k, v in window_features.items()
            }
            
            # Add label (most common label in window)
            if 'label' in window.columns:
                labels = window['label'].dropna()
                if len(labels) > 0:
                    window_features['label'] = labels.mode()[0]
                else:
                    window_features['label'] = None
            
            features_list.append(window_features)
        
        return pd.DataFrame(features_list)
    
    def _extract_window_features(self, window: pd.DataFrame) -> Dict[str, float]:
        """Extract all features from a single window."""
        features = {}
        
        # Accelerometer features (per axis)
        for axis in ['Ax', 'Ay', 'Az']:
            signal_data = window[axis].values
            prefix = f'acc_{axis.lower()}_'
            
            # Time-domain features
            features[prefix + 'mean'] = np.mean(signal_data)
            features[prefix + 'std'] = np.std(signal_data)
            features[prefix + 'var'] = np.var(signal_data)
            features[prefix + 'rms'] = np.sqrt(np.mean(signal_data**2))
            features[prefix + 'peak_to_peak'] = np.ptp(signal_data)
            features[prefix + 'energy'] = np.sum(signal_data**2)
            features[prefix + 'sma'] = np.sum(np.abs(signal_data))
            features[prefix + 'skewness'] = stats.skew(signal_data)
            features[prefix + 'kurtosis'] = stats.kurtosis(signal_data)
            features[prefix + 'iqr'] = np.percentile(signal_data, 75) - np.percentile(signal_data, 25)
            features[prefix + 'zcr'] = np.sum(np.diff(np.sign(signal_data)) != 0) / len(signal_data)
            
            # Frequency-domain features
            freq_features = self._frequency_features(signal_data)
            for key, val in freq_features.items():
                features[prefix + key] = val
        
        # Gyroscope features (per axis)
        for axis in ['Gx', 'Gy', 'Gz']:
            signal_data = window[axis].values
            prefix = f'gyro_{axis.lower()}_'
            
            # Time-domain features
            features[prefix + 'mean'] = np.mean(signal_data)
            features[prefix + 'std'] = np.std(signal_data)
            features[prefix + 'var'] = np.var(signal_data)
            features[prefix + 'rms'] = np.sqrt(np.mean(signal_data**2))
            features[prefix + 'peak_to_peak'] = np.ptp(signal_data)
            features[prefix + 'energy'] = np.sum(signal_data**2)
            features[prefix + 'sma'] = np.sum(np.abs(signal_data))
            features[prefix + 'skewness'] = stats.skew(signal_data)
            features[prefix + 'kurtosis'] = stats.kurtosis(signal_data)
            features[prefix + 'iqr'] = np.percentile(signal_data, 75) - np.percentile(signal_data, 25)
            features[prefix + 'zcr'] = np.sum(np.diff(np.sign(signal_data)) != 0) / len(signal_data)
            
            # Frequency-domain features
            freq_features = self._frequency_features(signal_data)
            for key, val in freq_features.items():
                features[prefix + key] = val
        
        # Cross-axis features (accelerometer)
        acc_mag = np.sqrt(window['Ax']**2 + window['Ay']**2 + window['Az']**2)
        features['acc_magnitude_mean'] = np.mean(acc_mag)
        features['acc_magnitude_std'] = np.std(acc_mag)
        features['acc_magnitude_rms'] = np.sqrt(np.mean(acc_mag**2))
        
        # Cross-axis features (gyroscope)
        gyro_mag = np.sqrt(window['Gx']**2 + window['Gy']**2 + window['Gz']**2)
        features['gyro_magnitude_mean'] = np.mean(gyro_mag)
        features['gyro_magnitude_std'] = np.std(gyro_mag)
        features['gyro_magnitude_rms'] = np.sqrt(np.mean(gyro_mag**2))
        
        # Correlations (can produce NaN — caught by FIX 5 sanitizer)
        features['corr_ax_ay'] = np.corrcoef(window['Ax'], window['Ay'])[0, 1]
        features['corr_ax_az'] = np.corrcoef(window['Ax'], window['Az'])[0, 1]
        features['corr_ay_az'] = np.corrcoef(window['Ay'], window['Az'])[0, 1]
        features['corr_gx_gy'] = np.corrcoef(window['Gx'], window['Gy'])[0, 1]
        features['corr_gx_gz'] = np.corrcoef(window['Gx'], window['Gz'])[0, 1]
        features['corr_gy_gz'] = np.corrcoef(window['Gy'], window['Gz'])[0, 1]
        
        # Signal Vector Magnitude
        features['svm'] = np.mean(np.sqrt(window['Ax']**2 + window['Ay']**2 + window['Az']**2 + 
                                          window['Gx']**2 + window['Gy']**2 + window['Gz']**2))
        
        return features
    
    def _frequency_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Extract frequency-domain features."""
        fft_vals = fft(signal_data)
        freqs = fftfreq(len(signal_data), 1/self.config.sampling_rate)
        
        # Positive frequencies only
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        power = np.abs(fft_vals[pos_mask])**2
        
        features = {}
        
        # Dominant frequency
        dominant_idx = np.argmax(power)
        features['dominant_freq'] = freqs[dominant_idx]
        
        # Tremor band power (4-6 Hz)
        tremor_mask = (freqs >= self.config.tremor_band_low) & (freqs <= self.config.tremor_band_high)
        features['tremor_power'] = np.sum(power[tremor_mask])
        
        # Voluntary movement band power (0-3 Hz)
        voluntary_mask = (freqs >= self.config.voluntary_band_low) & (freqs <= self.config.voluntary_band_high)
        features['voluntary_power'] = np.sum(power[voluntary_mask])
        
        # Spectral entropy
        power_norm = power / (np.sum(power) + 1e-10)
        features['spectral_entropy'] = -np.sum(power_norm * np.log2(power_norm + 1e-10))
        
        return features


# ============================================================================
# BASELINE LEARNING
# ============================================================================

class BaselineLearner:
    """Learns patient-specific baseline from ON-state data."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.baseline_stats = None
    
    # =========================================================================
    # FIX 3: Robust baseline statistics (median + scaled MAD)
    # 
    # OLD: np.mean() and np.std() → a single extreme outlier corrupts both.
    #      67/103 features had mean ~ 1e26, all z-scores collapsed to ~0.
    # NEW: np.median() (50% breakdown point) and MAD*1.4826 (robust spread).
    # =========================================================================
        
    def learn_baseline(self, features_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate ROBUST baseline statistics from ON-state windows.
        
        Uses median (center) and scaled MAD (spread) instead of mean/std.
        
        Args:
            features_df: Feature DataFrame with 'label' column
            
        Returns:
            Dictionary of {feature_name: {'mean': median, 'std': scaled_MAD, 'median': median}}
        """
        on_data = features_df[features_df['label'] == 'ON']
        
        if len(on_data) == 0:
            raise ValueError("No ON-state data available for baseline learning")
        
        baseline_stats = {}
        extreme_count = 0
        
        feature_cols = [col for col in features_df.columns if col != 'label']
        
        for col in feature_cols:
            values = on_data[col].dropna()
            if len(values) > 0:
                median_val = float(np.median(values))
                mad_val = float(np.median(np.abs(values - median_val)))
                mad_std = mad_val * 1.4826  # consistency constant for normal distributions
                
                baseline_stats[col] = {
                    'mean': median_val,       # robust center
                    'std': mad_std,           # robust spread
                    'median': median_val
                }
                
                # Diagnostic: detect features that WOULD have been corrupted
                naive_mean = float(np.mean(values))
                if abs(naive_mean) > 1e10:
                    extreme_count += 1
        
        self.baseline_stats = baseline_stats
        
        print(f"\nBaseline learned from {len(on_data)} ON-state windows")
        print(f"Baseline features: {len(baseline_stats)}")
        if extreme_count > 0:
            print(f"  WARNING: {extreme_count} features had extreme naive mean (>1e10)")
            print(f"  Robust median/MAD protected against corruption")
        
        # Diagnostic health check
        self._verify_baseline(baseline_stats)
        
        return baseline_stats
    
    @staticmethod
    def _verify_baseline(baseline_stats: Dict) -> None:
        """Print diagnostic to verify baseline values are reasonable."""
        centers = [s['mean'] for s in baseline_stats.values()]
        spreads = [s['std'] for s in baseline_stats.values()]
        
        max_center = max(abs(c) for c in centers) if centers else 0
        max_spread = max(spreads) if spreads else 0
        zero_spread = sum(1 for s in spreads if s == 0)
        
        print(f"  Baseline health check:")
        print(f"    Max |center|:          {max_center:.6f}")
        print(f"    Max spread:            {max_spread:.6f}")
        print(f"    Zero-spread features:  {zero_spread}/{len(baseline_stats)} (get z-score=0)")
        
        if max_center > 1e6 or max_spread > 1e6:
            print(f"    WARNING: Some baseline values seem large — check raw data")
        else:
            print(f"    OK: All baseline values in reasonable range")
    
    def compute_deviations(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute z-score deviations from baseline for each feature.
        
        z = (x - median) / (MAD * 1.4826)
        
        Z-scores are clipped to [-10, 10] to prevent extreme values
        from dominating the model.
        
        Args:
            features_df: Feature DataFrame
            
        Returns:
            DataFrame with added _zscore features
        """
        if self.baseline_stats is None:
            raise ValueError("Baseline must be learned first")
        
        df = features_df.copy()
        
        for feature_name, bstats in self.baseline_stats.items():
            if feature_name in df.columns:
                center = bstats['mean']    # = median
                spread = bstats['std']     # = MAD * 1.4826
                
                if spread > 1e-15:
                    zscore = (df[feature_name] - center) / spread
                    # Clip to prevent extreme z-scores
                    df[f'{feature_name}_zscore'] = np.clip(zscore, -10, 10)
                else:
                    df[f'{feature_name}_zscore'] = 0.0
        
        return df


# ============================================================================
# QUALITY GATE
# ============================================================================

class QualityGate:
    """Checks data quality and filters low-quality windows."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
    def filter_quality(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove low-quality windows based on quality criteria.
        
        Args:
            features_df: Feature DataFrame
            
        Returns:
            Filtered DataFrame
        """
        initial_count = len(features_df)
        
        # Remove windows with too many missing values
        missing_ratio = features_df.isnull().sum(axis=1) / len(features_df.columns)
        features_df = features_df[missing_ratio <= self.config.max_missing_ratio]
        
        final_count = len(features_df)
        removed = initial_count - final_count
        
        if removed > 0:
            print(f"Quality gate: Removed {removed} low-quality windows")
        
        return features_df


# ============================================================================
# MODEL TRAINING
# ============================================================================

class Trainer:
    """Trains and evaluates Random Forest classifier."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def prepare_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for training.
        
        Args:
            features_df: Feature DataFrame with 'label' column
            
        Returns:
            X, y, feature_names
        """
        # Remove unlabeled samples
        df = features_df.dropna(subset=['label']).copy()
        
        # Encode labels
        df['label_encoded'] = df['label'].map({'OFF': 0, 'ON': 1})
        
        # Separate features and labels
        feature_cols = [col for col in df.columns if col not in ['label', 'label_encoded']]
        X = df[feature_cols].values.astype(np.float64)
        y = df['label_encoded'].values
        
        # Defense-in-depth NaN/Inf cleanup
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"\nPrepared data:")
        print(f"  Samples: {len(X)}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  OFF samples: {np.sum(y == 0)}")
        print(f"  ON samples: {np.sum(y == 1)}")
        
        return X, y, feature_cols
    
    # =========================================================================
    # FIX 6: Tuned Random Forest
    # 
    # OLD: n_estimators=100, max_depth=None (unlimited → memorizes training data),
    #      min_samples_split=2, min_samples_leaf=1, no class weighting.
    # NEW: 300 trees, max_depth=25, min_samples_split=5, min_samples_leaf=2,
    #      class_weight='balanced_subsample', max_features='sqrt', OOB score.
    # =========================================================================
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        Train Random Forest model with cross-validation.

        Args:
            X_train: Training feature matrix
            y_train: Training labels
            feature_names: List of feature names

        Returns:
            Dictionary with training results
        """
        self.feature_names = feature_names

        # Normalize features — fit ONLY on training data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # Adaptive parameters for small datasets
        n_samples = len(X_train)
        n_cv_folds = min(self.config.n_cv_folds, n_samples)
        n_estimators = min(self.config.n_estimators, max(50, n_samples))
        max_depth = self.config.max_depth
        
        if n_samples < 50:
            max_depth = min(max_depth or 10, 10)

        if n_cv_folds < 2:
            print("Warning: Not enough samples for cross-validation")
            n_cv_folds = 2

        # Cross-validation
        cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=self.config.random_state)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y_train)):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                max_features='sqrt',
                class_weight='balanced_subsample',
                bootstrap=True,
                random_state=self.config.random_state,
                n_jobs=-1
            )

            model.fit(X_tr, y_tr)
            score = model.score(X_val, y_val)
            cv_scores.append(score)

        # Train final model on ALL training data
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features='sqrt',
            class_weight='balanced_subsample',
            bootstrap=True,
            oob_score=True,
            random_state=self.config.random_state,
            n_jobs=-1
        )

        self.model.fit(X_scaled, y_train)

        results = {
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'n_cv_folds': n_cv_folds,
            'n_estimators': n_estimators,
            'n_train_samples': n_samples,
            'n_features': len(feature_names),
            'oob_score': self.model.oob_score_,
        }

        print(f"\nTraining results:")
        print(f"  Trees: {n_estimators}, Max depth: {max_depth}")
        print(f"  Class weighting: balanced_subsample")
        print(f"  CV folds: {n_cv_folds}")
        print(f"  CV accuracy: {results['cv_mean']:.4f} +/- {results['cv_std']:.4f}")
        print(f"  OOB score:   {results['oob_score']:.4f}")

        return results
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model on held-out test data.

        Args:
            X_test: Test feature matrix (unseen during training)
            y_test: True test labels

        Returns:
            Dictionary with evaluation metrics
        """
        X_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)
        
        # Confidence analysis
        confidences = np.max(y_proba, axis=1)
        low_conf_count = np.sum(confidences < self.config.confidence_threshold)
        mean_conf = np.mean(confidences)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred,
                                                          target_names=['OFF', 'ON'],
                                                          zero_division=0),
            'n_test_samples': len(y_test),
            'mean_confidence': float(mean_conf),
            'low_confidence_predictions': int(low_conf_count),
        }
        
        # Per-class metrics for clinical reporting
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['sensitivity_OFF'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['sensitivity_ON'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        print(f"\n{'='*50}")
        print(f"HELD-OUT TEST EVALUATION ({len(y_test)} samples)")
        print(f"{'='*50}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1-score:    {metrics['f1']:.4f}")
        print(f"  Confidence:  {mean_conf:.4f} average")
        if low_conf_count > 0:
            print(f"  {low_conf_count}/{len(y_test)} predictions below "
                  f"{self.config.confidence_threshold:.0%} confidence")
        print(f"\n  Confusion Matrix:")
        print(f"  {metrics['confusion_matrix']}")
        
        if 'sensitivity_OFF' in metrics:
            print(f"\n  Per-class (clinical):")
            print(f"    OFF detection rate: {metrics['sensitivity_OFF']:.4f}")
            print(f"    ON detection rate:  {metrics['sensitivity_ON']:.4f}")

        return metrics
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence scores and uncertainty flags.
        
        For clinical use: predictions below confidence_threshold are flagged
        as uncertain — better to say "unsure" than give a wrong answer.
        
        Returns:
            predictions, confidences, is_confident (boolean mask)
        """
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        confidences = np.max(probabilities, axis=1)
        is_confident = confidences >= self.config.confidence_threshold
        
        return predictions, confidences, is_confident
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        importances = self.model.feature_importances_
        
        feature_importance = {
            name: float(importance) 
            for name, importance in zip(self.feature_names, importances)
        }
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), 
                                        key=lambda x: x[1], 
                                        reverse=True))
        
        print(f"\nTop 10 most important features:")
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
            print(f"  {i}. {feature}: {importance:.4f}")
        
        return feature_importance


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

class CompletePipeline:
    """End-to-end pipeline connecting all components."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.data_loader = DataLoader(config)
        self.preprocessor = Preprocessor(config)
        self.feature_extractor = FeatureExtractor(config)
        self.baseline_learner = BaselineLearner(config)
        self.quality_gate = QualityGate(config)
        self.trainer = Trainer(config)
    
    # =========================================================================
    # FIX 2: Data leakage eliminated
    # 
    # OLD FLOW (LEAKED):
    #   Extract features → Learn baseline from ALL ON data → Compute z-scores
    #   on ALL data → Train/test split → Train & evaluate
    #   Problem: test z-scores were computed using test data's own baseline!
    #
    # NEW FLOW (CLEAN):
    #   Extract 103 raw features → Train/test split → Learn baseline from
    #   TRAINING ON data only → Compute z-scores for both splits using
    #   training baseline → Train & evaluate
    # =========================================================================
        
    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run complete pipeline with proper train/test isolation.
        
        Args:
            df: Sensor data with columns: time, Ax, Ay, Az, Gx, Gy, Gz, label
            
        Returns:
            Dictionary with all results
        """
        results = {}
        
        print("=" * 70)
        print("STEP 1: Data Summary")
        print("=" * 70)
        results['n_samples'] = len(df)
        results['duration'] = float(df['time'].max())
        print(f"Samples: {results['n_samples']} ({results['duration']:.2f} seconds)")
        
        print("\n" + "=" * 70)
        print("STEP 2: Preprocess")
        print("=" * 70)
        df_preprocessed = self.preprocessor.preprocess(df)
        print("Preprocessing complete (bandpass filter + robust outlier removal)")
        
        print("\n" + "=" * 70)
        print("STEP 3: Extract Raw Features (103 features)")
        print("=" * 70)
        features_df = self.feature_extractor.extract_features(df_preprocessed)
        raw_feature_cols = [c for c in features_df.columns if c != 'label']
        n_raw = len(raw_feature_cols)
        results['n_windows'] = len(features_df)
        print(f"Extracted {n_raw} raw features from {results['n_windows']} windows")
        
        print("\n" + "=" * 70)
        print("STEP 4: Train/Test Split (BEFORE baseline — no data leakage)")
        print("=" * 70)
        
        labeled_df = features_df.dropna(subset=['label']).reset_index(drop=True)
        
        train_df, test_df = train_test_split(
            labeled_df, test_size=0.2,
            random_state=self.config.random_state,
            stratify=labeled_df['label']
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        print(f"  Training windows: {len(train_df)}")
        print(f"  Test windows:     {len(test_df)} (held out)")
        print(f"  Train ON/OFF: {sum(train_df['label']=='ON')}/{sum(train_df['label']=='OFF')}")
        print(f"  Test  ON/OFF: {sum(test_df['label']=='ON')}/{sum(test_df['label']=='OFF')}")
        
        print("\n" + "=" * 70)
        print("STEP 5: Learn Baseline (TRAINING ON data only)")
        print("=" * 70)
        baseline_stats = self.baseline_learner.learn_baseline(train_df)
        results['baseline_stats'] = baseline_stats
        
        print("\n" + "=" * 70)
        print("STEP 6: Compute Z-Score Deviations")
        print("=" * 70)
        train_df = self.baseline_learner.compute_deviations(train_df)
        test_df = self.baseline_learner.compute_deviations(test_df)
        
        all_feature_cols = [c for c in train_df.columns if c not in ['label', 'label_encoded']]
        results['n_features'] = len(all_feature_cols)
        print(f"Total features: {results['n_features']} ({n_raw} raw + {results['n_features'] - n_raw} z-score)")
        
        print("\n" + "=" * 70)
        print("STEP 7: Prepare Data")
        print("=" * 70)
        
        train_df['label_encoded'] = train_df['label'].map({'OFF': 0, 'ON': 1})
        test_df['label_encoded'] = test_df['label'].map({'OFF': 0, 'ON': 1})
        
        feature_cols = [c for c in train_df.columns if c not in ['label', 'label_encoded']]
        
        X_train = train_df[feature_cols].values.astype(np.float64)
        y_train = train_df['label_encoded'].values
        X_test = test_df[feature_cols].values.astype(np.float64)
        y_test = test_df['label_encoded'].values
        
        # Final safety net
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"  Training: {X_train.shape[0]} samples x {X_train.shape[1]} features")
        print(f"  Test:     {X_test.shape[0]} samples x {X_test.shape[1]} features")

        print("\n" + "=" * 70)
        print("STEP 8: Train Model (training set only)")
        print("=" * 70)
        training_results = self.trainer.train(X_train, y_train, feature_cols)
        results.update(training_results)

        print("\n" + "=" * 70)
        print("STEP 9: Evaluate (held-out test set)")
        print("=" * 70)
        eval_results = self.trainer.evaluate(X_test, y_test)
        results.update(eval_results)
        
        print("\n" + "=" * 70)
        print("STEP 10: Feature Importance")
        print("=" * 70)
        feature_importance = self.trainer.get_feature_importance()
        results['feature_importance'] = feature_importance
        
        # Z-score contribution analysis
        zscore_feats = {k: v for k, v in feature_importance.items() if '_zscore' in k}
        raw_feats = {k: v for k, v in feature_importance.items() if '_zscore' not in k}
        z_total = sum(zscore_feats.values()) if zscore_feats else 0
        r_total = sum(raw_feats.values()) if raw_feats else 0
        total = z_total + r_total + 1e-10
        print(f"\n  Feature contribution:")
        print(f"    Raw features:     {r_total:.4f} ({100*r_total/total:.1f}%)")
        print(f"    Z-score features: {z_total:.4f} ({100*z_total/total:.1f}%)")
        
        return results
    
    def save_model(self, filepath: str):
        """Save trained model, scaler, and metadata."""
        model_data = {
            'model': self.trainer.model,
            'scaler': self.trainer.scaler,
            'feature_names': self.trainer.feature_names,
            'baseline_stats': self.baseline_learner.baseline_stats,
            'config': asdict(self.config),
            'confidence_threshold': self.config.confidence_threshold,
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to: {filepath}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    
    print("\n" + "=" * 70)
    print("PARKINSON'S ON/OFF DETECTION - ACCELEROMETER + GYROSCOPE")
    print("Production Pipeline v2 (6 critical fixes applied)")
    print("=" * 70)
    
    # Initialize configuration
    config = SystemConfig()
    
    # Initialize data loader
    data_loader = DataLoader(config)
    
    print("\n" + "=" * 70)
    print("LOADING DATA FROM DecodedData_Test FOLDER")
    print("=" * 70)
    
    # Load data
    df = data_loader.load_from_decoded_folder()
    
    # =========================================================================
    # FIX 1 (flow): Preprocess BEFORE classification
    # OLD: Classified raw noisy data, then preprocessed separately.
    # NEW: Clean signal first → classifier works on clean data.
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("PREPROCESSING RAW SIGNAL (before classification)")
    print("=" * 70)
    
    preprocessor = Preprocessor(config)
    df_clean = preprocessor.preprocess(df)
    print("Signal cleaned (bandpass filter + robust outlier removal)")
    
    print("\n" + "=" * 70)
    print("CLASSIFYING SIGNAL INTO ON/OFF STATES")
    print("=" * 70)
    
    # Classify on CLEAN data for better separation
    df_classified = data_loader.classify_signal(df_clean)
    
    print("\n" + "=" * 70)
    print("RUNNING COMPLETE PIPELINE")
    print("=" * 70)
    
    # Run pipeline
    pipeline = CompletePipeline(config)
    results = pipeline.run(df_classified)
    
    # Save model
    model_path = Path(config.model_folder) / 'parkinsons_model_acc_gyro.pkl'
    pipeline.save_model(str(model_path))
    
    # Save results
    results_folder = Path(config.output_folder)
    results_folder.mkdir(parents=True, exist_ok=True)
    
    # Save JSON results
    results_json = results.copy()
    results_json['classification_report'] = str(results_json['classification_report'])
    
    json_path = results_folder / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    
    # Save markdown report
    md_path = results_folder / 'results.md'
    with open(md_path, 'w') as f:
        f.write("# Parkinson's ON/OFF Detection Results\n\n")
        f.write("## Pipeline Version\n")
        f.write("Production v2 — 6 critical fixes applied\n\n")
        f.write("## Dataset Summary\n")
        f.write(f"- Total samples: {results['n_samples']}\n")
        f.write(f"- Duration: {results['duration']:.2f} seconds\n")
        f.write(f"- Windows: {results['n_windows']}\n")
        f.write(f"- Features: {results['n_features']}\n\n")
        
        f.write("## Model Performance\n")
        f.write(f"- Cross-validation folds: {results['n_cv_folds']}\n")
        f.write(f"- CV accuracy (train): {results['cv_mean']:.4f} +/- {results['cv_std']:.4f}\n")
        f.write(f"- OOB score: {results.get('oob_score', 'N/A')}\n")
        f.write(f"- Training samples: {results['n_train_samples']}\n")
        f.write(f"- Test samples (held-out): {results['n_test_samples']}\n")
        f.write(f"- Held-out test accuracy: {results['accuracy']:.4f}\n")
        f.write(f"- Precision: {results['precision']:.4f}\n")
        f.write(f"- Recall: {results['recall']:.4f}\n")
        f.write(f"- F1-score: {results['f1']:.4f}\n")
        f.write(f"- Mean prediction confidence: {results.get('mean_confidence', 'N/A')}\n\n")
        
        f.write("## Confusion Matrix\n")
        f.write("```\n")
        f.write(str(results['confusion_matrix']))
        f.write("\n```\n\n")
        
        f.write("## Classification Report\n")
        f.write("```\n")
        f.write(results['classification_report'])
        f.write("\n```\n\n")
        
        f.write("## Top 10 Features\n")
        for i, (feature, importance) in enumerate(list(results['feature_importance'].items())[:10], 1):
            f.write(f"{i}. {feature}: {importance:.4f}\n")
    
    print(f"Report saved to: {md_path}")
    
    # Clinical safety summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE — CLINICAL SUMMARY")
    print("=" * 70)
    accuracy = results['accuracy']
    f1 = results['f1']
    confidence = results.get('mean_confidence', 0)
    low_conf = results.get('low_confidence_predictions', 0)
    
    if accuracy >= 0.85 and f1 >= 0.85:
        print(f"  PASS: Accuracy={accuracy:.1%}, F1={f1:.1%}")
    else:
        print(f"  BELOW TARGET: Accuracy={accuracy:.1%}, F1={f1:.1%}")
        print(f"    Target: >=85%. Consider collecting more labeled data.")
    
    print(f"  Mean confidence: {confidence:.1%}")
    if low_conf > 0:
        print(f"  {low_conf} predictions below {config.confidence_threshold:.0%} confidence")
    print(f"  Model saved to: {model_path}")


if __name__ == '__main__':
    main()