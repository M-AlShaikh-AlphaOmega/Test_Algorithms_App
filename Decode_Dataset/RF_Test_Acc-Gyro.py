"""
RF_Test_Acc-Gyro.py - Parkinson's ON/OFF Detection Pipeline with Accelerometer + Gyroscope

This script detects whether a Parkinson's patient is in an ON state (medication working) 
or OFF state (medication worn off) using accelerometer and gyroscope data from a wrist sensor.

Modified from RF_Test.py to support both accelerometer and gyroscope data from DecodedData_Test folder.
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
    
    # Model parameters
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    n_cv_folds: int = 5
    random_state: int = 42
    
    # Quality control
    max_missing_ratio: float = 0.1  # 10% max missing data per window
    outlier_std_threshold: float = 5.0  # Standard deviations for outlier detection
    
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
    
    def classify_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify signal into ON/OFF states based on tremor-band power analysis.
        
        Uses accelerometer magnitude in 4-6 Hz band.
        Windows with above-median tremor power are labeled OFF.
        Windows with below-median tremor power are labeled ON.
        
        Args:
            df: DataFrame with time, Ax, Ay, Az, Gx, Gy, Gz columns
            
        Returns:
            DataFrame with added 'label' column
        """
        # Calculate accelerometer magnitude
        acc_magnitude = np.sqrt(df['Ax']**2 + df['Ay']**2 + df['Az']**2)
        
        # Window parameters
        window_samples = int(self.config.window_duration * self.config.sampling_rate)
        step_samples = int(window_samples * (1 - self.config.window_overlap))
        
        tremor_powers = []
        window_indices = []
        
        # Analyze each window
        for start in range(0, len(acc_magnitude) - window_samples + 1, step_samples):
            end = start + window_samples
            window_signal = acc_magnitude[start:end].values
            
            # Compute FFT
            fft_vals = fft(window_signal)
            freqs = fftfreq(len(window_signal), 1/self.config.sampling_rate)
            
            # Get positive frequencies only
            pos_mask = freqs >= 0
            freqs = freqs[pos_mask]
            power = np.abs(fft_vals[pos_mask])**2
            
            # Calculate tremor band power (4-6 Hz)
            tremor_mask = (freqs >= self.config.tremor_band_low) & (freqs <= self.config.tremor_band_high)
            tremor_power = np.sum(power[tremor_mask])
            
            tremor_powers.append(tremor_power)
            window_indices.append((start, end))
        
        # Classify based on median tremor power
        median_power = np.median(tremor_powers)
        
        # Initialize all labels as None
        labels = [None] * len(df)
        
        # Assign labels to each sample based on its window
        for (start, end), tremor_power in zip(window_indices, tremor_powers):
            label = 'OFF' if tremor_power > median_power else 'ON'
            for i in range(start, end):
                labels[i] = label
        
        df['label'] = labels
        
        # Count labeled samples
        on_count = sum(1 for l in labels if l == 'ON')
        off_count = sum(1 for l in labels if l == 'OFF')
        unlabeled_count = sum(1 for l in labels if l is None)
        
        print(f"\nClassification results:")
        print(f"  ON state: {on_count} samples")
        print(f"  OFF state: {off_count} samples")
        print(f"  Unlabeled: {unlabeled_count} samples")
        
        return df


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
        
        # Remove outliers
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
    
    def _remove_outliers(self, signal_data: np.ndarray) -> np.ndarray:
        """Replace outliers with interpolated values."""
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        threshold = self.config.outlier_std_threshold
        
        # Detect outliers
        outliers = np.abs(signal_data - mean) > (threshold * std)
        
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
        
        # Correlations
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
        
    def learn_baseline(self, features_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate baseline statistics from ON-state windows.
        
        Args:
            features_df: Feature DataFrame with 'label' column
            
        Returns:
            Dictionary of {feature_name: {'mean': x, 'std': y, 'median': z}}
        """
        on_data = features_df[features_df['label'] == 'ON']
        
        if len(on_data) == 0:
            raise ValueError("No ON-state data available for baseline learning")
        
        baseline_stats = {}
        
        # Calculate statistics for each feature
        feature_cols = [col for col in features_df.columns if col != 'label']
        
        for col in feature_cols:
            values = on_data[col].dropna()
            if len(values) > 0:
                baseline_stats[col] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values))
                }
        
        self.baseline_stats = baseline_stats
        
        print(f"\nBaseline learned from {len(on_data)} ON-state windows")
        print(f"Baseline features: {len(baseline_stats)}")
        
        return baseline_stats
    
    def compute_deviations(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute z-score deviations from baseline for each feature.
        
        Args:
            features_df: Feature DataFrame
            
        Returns:
            DataFrame with added deviation features
        """
        if self.baseline_stats is None:
            raise ValueError("Baseline must be learned first")
        
        df = features_df.copy()
        
        for feature_name, stats in self.baseline_stats.items():
            if feature_name in df.columns:
                mean = stats['mean']
                std = stats['std']
                
                # Calculate z-score
                if std > 0:
                    df[f'{feature_name}_zscore'] = (df[feature_name] - mean) / std
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
        X = df[feature_cols].values
        y = df['label_encoded'].values
        
        print(f"\nPrepared data for training:")
        print(f"  Samples: {len(X)}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  OFF samples: {np.sum(y == 0)}")
        print(f"  ON samples: {np.sum(y == 1)}")
        
        return X, y, feature_cols
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        Train Random Forest model with cross-validation on training data only.

        Args:
            X_train: Training feature matrix
            y_train: Training labels
            feature_names: List of feature names

        Returns:
            Dictionary with training results
        """
        self.feature_names = feature_names

        # Normalize features - fit ONLY on training data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # Adjust CV folds and model parameters for small datasets
        n_samples = len(X_train)
        n_cv_folds = min(self.config.n_cv_folds, n_samples)
        n_estimators = min(self.config.n_estimators, max(10, n_samples // 2))

        if n_cv_folds < 2:
            print("Warning: Not enough samples for cross-validation")
            n_cv_folds = 2

        # Cross-validation on training data
        cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=self.config.random_state)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y_train)):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                random_state=self.config.random_state
            )

            model.fit(X_tr, y_tr)
            score = model.score(X_val, y_val)
            cv_scores.append(score)

        # Train final model on training data only
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state
        )

        self.model.fit(X_scaled, y_train)

        results = {
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'n_cv_folds': n_cv_folds,
            'n_estimators': n_estimators,
            'n_train_samples': n_samples,
            'n_features': len(feature_names)
        }

        print(f"\nCross-validation results (on training set):")
        print(f"  Folds: {n_cv_folds}")
        print(f"  Mean accuracy: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")

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

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred,
                                                          target_names=['OFF', 'ON'],
                                                          zero_division=0),
            'n_test_samples': len(y_test)
        }

        print(f"\nEvaluation on HELD-OUT test set ({len(y_test)} samples):")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  {metrics['confusion_matrix']}")

        return metrics
    
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
        
    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run complete pipeline.
        
        Args:
            df: Raw sensor data with columns: time, Ax, Ay, Az, Gx, Gy, Gz, label
            
        Returns:
            Dictionary with all results
        """
        results = {}
        
        print("="*70)
        print("STEP 1: Load Data")
        print("="*70)
        results['n_samples'] = len(df)
        results['duration'] = float(df['time'].max())
        print(f"Loaded {results['n_samples']} samples ({results['duration']:.2f} seconds)")
        
        print("\n" + "="*70)
        print("STEP 2: Preprocess")
        print("="*70)
        df_preprocessed = self.preprocessor.preprocess(df)
        print("Preprocessing complete")
        
        print("\n" + "="*70)
        print("STEP 3: Extract Features")
        print("="*70)
        features_df = self.feature_extractor.extract_features(df_preprocessed)
        results['n_windows'] = len(features_df)
        print(f"Extracted features from {results['n_windows']} windows")
        
        print("\n" + "="*70)
        print("STEP 4: Learn Baseline")
        print("="*70)
        baseline_stats = self.baseline_learner.learn_baseline(features_df)
        results['baseline_stats'] = baseline_stats
        
        print("\n" + "="*70)
        print("STEP 5: Compute Deviations")
        print("="*70)
        features_df = self.baseline_learner.compute_deviations(features_df)
        results['n_features'] = len([col for col in features_df.columns if col != 'label'])
        print(f"Total features (including deviations): {results['n_features']}")
        
        print("\n" + "="*70)
        print("STEP 6: Prepare Data")
        print("="*70)
        X, y, feature_names = self.trainer.prepare_data(features_df)

        print("\n" + "="*70)
        print("STEP 6b: Train/Test Split")
        print("="*70)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.config.random_state, stratify=y
        )
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples:     {len(X_test)} (held out, never seen during training)")

        print("\n" + "="*70)
        print("STEP 7: Train Model (on training set only)")
        print("="*70)
        training_results = self.trainer.train(X_train, y_train, feature_names)
        results.update(training_results)

        print("\n" + "="*70)
        print("STEP 8: Evaluate (on held-out test set)")
        print("="*70)
        eval_results = self.trainer.evaluate(X_test, y_test)
        results.update(eval_results)
        
        print("\n" + "="*70)
        print("STEP 9: Feature Importance")
        print("="*70)
        feature_importance = self.trainer.get_feature_importance()
        results['feature_importance'] = feature_importance
        
        return results
    
    def save_model(self, filepath: str):
        """Save trained model, scaler, and metadata."""
        model_data = {
            'model': self.trainer.model,
            'scaler': self.trainer.scaler,
            'feature_names': self.trainer.feature_names,
            'baseline_stats': self.baseline_learner.baseline_stats,
            'config': asdict(self.config)
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
    
    print("\n" + "="*70)
    print("PARKINSON'S ON/OFF DETECTION - ACCELEROMETER + GYROSCOPE")
    print("="*70)
    
    # Initialize configuration
    config = SystemConfig()
    
    # Initialize data loader
    data_loader = DataLoader(config)
    
    print("\n" + "="*70)
    print("LOADING DATA FROM DecodedData_Test_Test FOLDER")
    print("="*70)
    
    # Load data from DecodedData_Test_Test folder
    df = data_loader.load_from_decoded_folder()
    
    print("\n" + "="*70)
    print("CLASSIFYING SIGNAL INTO ON/OFF STATES")
    print("="*70)
    
    # Classify signal
    df = data_loader.classify_signal(df)
    
    print("\n" + "="*70)
    print("RUNNING COMPLETE PIPELINE")
    print("="*70)
    
    # Run pipeline
    pipeline = CompletePipeline(config)
    results = pipeline.run(df)
    
    # Save model
    model_path = Path(config.model_folder) / 'parkinsons_model.pkl'
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
        f.write("## Dataset Summary\n")
        f.write(f"- Total samples: {results['n_samples']}\n")
        f.write(f"- Duration: {results['duration']:.2f} seconds\n")
        f.write(f"- Windows: {results['n_windows']}\n")
        f.write(f"- Features: {results['n_features']}\n\n")
        
        f.write("## Model Performance\n")
        f.write(f"- Cross-validation folds: {results['n_cv_folds']}\n")
        f.write(f"- CV accuracy (train): {results['cv_mean']:.4f} ± {results['cv_std']:.4f}\n")
        f.write(f"- Training samples: {results['n_train_samples']}\n")
        f.write(f"- Test samples (held-out): {results['n_test_samples']}\n")
        f.write(f"- Held-out test accuracy: {results['accuracy']:.4f}\n")
        f.write(f"- Precision: {results['precision']:.4f}\n")
        f.write(f"- Recall: {results['recall']:.4f}\n")
        f.write(f"- F1-score: {results['f1']:.4f}\n\n")
        
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
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()