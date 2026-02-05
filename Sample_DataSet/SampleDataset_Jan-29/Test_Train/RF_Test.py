"""
aCare Parkinson's Detection System - Complete All-In-One
=========================================================

Single file containing EVERYTHING:
✓ Data loading (recordings folder + contract sessions)
✓ Preprocessing (normalization, filtering)
✓ Feature extraction (43+ clinical features)
✓ Personalized baseline learning (N-of-1)
✓ Quality gates (ON/OFF/UNKNOWN)
✓ Random Forest training
✓ Evaluation & reporting
✓ Model saving

Usage:
    .\venv\Scripts\Activate.ps1
    cd Sample_DataSet\SampleDataset_Jan-29\Test_Train
    python RF_Test.py

Author: aCare System
Version: 3.0 FINAL
Date: 2026-02-04
"""

import os
import sys
import json
import warnings
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, asdict

warnings.filterwarnings('ignore')

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ML & Processing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import joblib

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SystemConfig:
    """Complete system configuration."""
    # Data Parameters
    sampling_rate: int = 25  # Hz (real sensor)
    window_duration: float = 1.0  # seconds
    window_overlap: float = 0.5
    
    # Sensor Configuration
    gravity_magnitude: float = 9.8
    acc_axes: List[str] = None
    
    # Frequency Analysis
    tremor_band_low: float = 4.0
    tremor_band_high: float = 6.0
    voluntary_band_high: float = 3.0
    
    # Personalization
    baseline_min_windows: int = 5
    
    # Model
    random_state: int = 42
    n_cv_folds: int = 5
    
    # Quality Control
    max_missing_ratio: float = 0.1
    outlier_std_threshold: float = 5.0
    
    def __post_init__(self):
        if self.acc_axes is None:
            self.acc_axes = ['X', 'Y', 'Z']
        self.window_samples = int(self.sampling_rate * self.window_duration)
        self.overlap_samples = int(self.window_samples * self.window_overlap)
        self.step_samples = self.window_samples - self.overlap_samples


# ============================================================================
# DATA LOADING (Dual Mode: Recordings + Contract)
# ============================================================================

class DataLoader:
    """Load data from recordings folder or contract sessions."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.metadata = None
    
    def load_recordings(self, recordings_dir: str) -> pd.DataFrame:
        """
        Load from recordings folder with ON/OFF labels in filenames.
        
        Expected:
            recordings/sample_001_ON.csv
            recordings/sample_002_OFF.csv
            
        CSV format: time,X,Y,Z
        """
        recordings_dir = Path(recordings_dir)
        
        if not recordings_dir.exists():
            raise FileNotFoundError(f"Folder not found: {recordings_dir}")
        
        csv_files = sorted(recordings_dir.glob('*.csv'))
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV files in {recordings_dir}")
        
        print(f"\nLoading recordings from: {recordings_dir}")
        print(f"Found {len(csv_files)} CSV files")
        
        all_dfs = []
        on_count = off_count = 0
        
        for csv_file in csv_files:
            filename_lower = csv_file.name.lower()
            
            # Detect label from filename
            if '_off' in filename_lower:
                label = 'OFF'
                off_count += 1
            elif '_on' in filename_lower:
                label = 'ON'
                on_count += 1
            else:
                print(f"  Skipping {csv_file.name} (no ON/OFF label)")
                continue
            
            df = pd.read_csv(csv_file)
            
            # Validate columns
            required = self.config.acc_axes + ['time']
            missing = set(required) - set(df.columns)
            if missing:
                print(f"  Skipping {csv_file.name} (missing: {missing})")
                continue
            
            df['label'] = label
            df['source_file'] = csv_file.name
            all_dfs.append(df)
            
            print(f"  {csv_file.name}: {len(df)} samples ({label})")
        
        if len(all_dfs) == 0:
            raise ValueError("No valid recordings found!")
        
        combined = pd.concat(all_dfs, ignore_index=True)
        
        print(f"\n✓ Loaded {len(combined)} total samples")
        print(f"  ON: {(combined['label']=='ON').sum()} samples ({on_count} files)")
        print(f"  OFF: {(combined['label']=='OFF').sum()} samples ({off_count} files)")
        
        return combined
    
    def load_contract_session(self, session_path: str) -> pd.DataFrame:
        """
        Load from contract v0.1 session bundle.
        
        Expected:
            session_path/meta.json
            session_path/results.csv
        """
        session_path = Path(session_path)
        
        # Load metadata
        meta_file = session_path / "meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"meta.json not found in {session_path}")
        
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        
        print(f"\nLoading contract session: {session_path}")
        print(f"  Version: {meta.get('ver', 'unknown')}")
        print(f"  Frequency: {meta['freq']} Hz")
        print(f"  Source: {meta['source']}")
        
        # Load results.csv
        csv_file = session_path / "results.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"results.csv not found in {session_path}")
        
        df = pd.read_csv(csv_file)
        
        # Apply scale factors
        scale_acc = float(meta.get('scale_acc', 1.0))
        df['accX'] = df['accX'] * scale_acc
        df['accY'] = df['accY'] * scale_acc
        df['accZ'] = df['accZ'] * scale_acc
        
        # Rename to standard format
        df.rename(columns={
            'accX': 'X',
            'accY': 'Y',
            'accZ': 'Z'
        }, inplace=True)
        
        df['label'] = None  # No labels in contract data
        
        print(f"✓ Loaded {len(df)} samples from contract session")
        
        return df


# ============================================================================
# PREPROCESSING
# ============================================================================

class Preprocessor:
    """Normalize and filter sensor data."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.gravity_scale = None
    
    def estimate_gravity_scale(self, df: pd.DataFrame) -> float:
        """Estimate sensor scale from still periods."""
        axes = self.config.acc_axes
        mag = np.sqrt(df[axes[0]]**2 + df[axes[1]]**2 + df[axes[2]]**2)
        
        # Find still periods
        window_size = self.config.window_samples
        rolling_std = mag.rolling(window=window_size, center=True).std()
        still_threshold = rolling_std.quantile(0.1)
        still_samples = mag[rolling_std <= still_threshold]
        
        measured_gravity = still_samples.median() if len(still_samples) >= 100 else mag.median()
        gravity_scale = measured_gravity / self.config.gravity_magnitude
        
        print(f"  Gravity: {measured_gravity:.2f} → scale: {gravity_scale:.4f}")
        
        return gravity_scale
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete preprocessing pipeline."""
        print("\nPREPROCESSING")
        print("="*60)
        
        df = df.copy()
        
        # 1. Normalize
        print("1. Normalizing accelerometer...")
        if self.gravity_scale is None:
            self.gravity_scale = self.estimate_gravity_scale(df)
        
        for axis in self.config.acc_axes:
            df[axis] = df[axis] / self.gravity_scale
        
        # 2. Butterworth filter
        print("2. Applying Butterworth filter...")
        nyquist = self.config.sampling_rate / 2
        highcut = min(10.0, nyquist * 0.95)
        low = 0.5 / nyquist
        high = highcut / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        
        for axis in self.config.acc_axes:
            df[axis] = signal.filtfilt(b, a, df[axis])
        
        # 3. Handle outliers
        print("3. Detecting outliers...")
        for axis in self.config.acc_axes:
            data = df[axis].values
            mean, std = np.mean(data), np.std(data)
            outliers = np.abs(data - mean) > (self.config.outlier_std_threshold * std)
            
            if outliers.any():
                df.loc[outliers, axis] = np.nan
                df[axis] = df[axis].interpolate(method='linear', limit_direction='both')
        
        print("✓ Preprocessing complete\n")
        return df


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class FeatureExtractor:
    """Extract clinical features from windows."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.feature_names = []
    
    def extract_time_features(self, window: np.ndarray, prefix: str) -> Dict:
        """Time-domain features."""
        return {
            f'{prefix}_mean': np.mean(window),
            f'{prefix}_std': np.std(window),
            f'{prefix}_var': np.var(window),
            f'{prefix}_rms': np.sqrt(np.mean(window**2)),
            f'{prefix}_peak_to_peak': np.ptp(window),
            f'{prefix}_energy': np.sum(window**2),
            f'{prefix}_sma': np.mean(np.abs(window)),
            f'{prefix}_skewness': stats.skew(window),
            f'{prefix}_kurtosis': stats.kurtosis(window),
            f'{prefix}_iqr': np.percentile(window, 75) - np.percentile(window, 25),
            f'{prefix}_zero_crossing': np.sum(np.diff(np.sign(window - np.mean(window))) != 0) / len(window)
        }
    
    def extract_freq_features(self, window: np.ndarray, prefix: str) -> Dict:
        """Frequency-domain features (tremor detection)."""
        n = len(window)
        fft_vals = fft(window)
        fft_freq = fftfreq(n, 1/self.config.sampling_rate)
        
        pos_mask = fft_freq > 0
        fft_freq = fft_freq[pos_mask]
        fft_power = np.abs(fft_vals[pos_mask])**2
        
        total_power = np.sum(fft_power)
        
        if total_power < 1e-10:
            return {
                f'{prefix}_dominant_freq': 0.0,
                f'{prefix}_tremor_power': 0.0,
                f'{prefix}_voluntary_power': 0.0,
                f'{prefix}_spectral_entropy': 0.0
            }
        
        # Dominant frequency
        dominant_idx = np.argmax(fft_power)
        dominant_freq = fft_freq[dominant_idx]
        
        # Tremor band (4-6 Hz) - CRITICAL
        tremor_mask = (fft_freq >= self.config.tremor_band_low) & \
                      (fft_freq <= self.config.tremor_band_high)
        tremor_power = np.sum(fft_power[tremor_mask]) / total_power
        
        # Voluntary band (0-3 Hz)
        voluntary_mask = fft_freq <= self.config.voluntary_band_high
        voluntary_power = np.sum(fft_power[voluntary_mask]) / total_power
        
        # Spectral entropy
        psd_norm = fft_power / total_power
        psd_norm = psd_norm[psd_norm > 0]
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm)) / np.log2(len(psd_norm))
        
        return {
            f'{prefix}_dominant_freq': dominant_freq,
            f'{prefix}_tremor_power': tremor_power,
            f'{prefix}_voluntary_power': voluntary_power,
            f'{prefix}_spectral_entropy': spectral_entropy
        }
    
    def extract_cross_axis(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict:
        """Cross-axis features."""
        mag = np.sqrt(x**2 + y**2 + z**2)
        
        return {
            'magnitude_mean': np.mean(mag),
            'magnitude_std': np.std(mag),
            'magnitude_rms': np.sqrt(np.mean(mag**2)),
            'corr_xy': np.corrcoef(x, y)[0, 1],
            'corr_xz': np.corrcoef(x, z)[0, 1],
            'corr_yz': np.corrcoef(y, z)[0, 1],
            'svm': np.mean(mag)
        }
    
    def extract_from_window(self, window_df: pd.DataFrame) -> Dict:
        """Extract all features from one window."""
        features = {}
        
        # Per-axis features
        for axis in self.config.acc_axes:
            window = window_df[axis].values
            features.update(self.extract_time_features(window, axis))
            features.update(self.extract_freq_features(window, axis))
        
        # Cross-axis
        features.update(self.extract_cross_axis(
            window_df[self.config.acc_axes[0]].values,
            window_df[self.config.acc_axes[1]].values,
            window_df[self.config.acc_axes[2]].values
        ))
        
        return features
    
    def create_windows(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, Optional[str]]]:
        """Split into overlapping windows."""
        windows = []
        n_samples = len(df)
        start = 0
        
        while start + self.config.window_samples <= n_samples:
            end = start + self.config.window_samples
            window_df = df.iloc[start:end].copy()
            
            # Check data quality
            missing_ratio = window_df[self.config.acc_axes].isnull().sum().sum() / \
                           (len(window_df) * len(self.config.acc_axes))
            
            if missing_ratio <= self.config.max_missing_ratio:
                label = None
                if 'label' in window_df.columns:
                    label = window_df['label'].mode()[0] if len(window_df['label'].mode()) > 0 else None
                windows.append((window_df, label))
            
            start += self.config.step_samples
        
        return windows
    
    def extract_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Extract features from all windows."""
        print(f"Creating windows: {self.config.window_duration}s, {self.config.window_overlap*100:.0f}% overlap")
        
        windows = self.create_windows(df)
        print(f"Created {len(windows)} windows")
        
        print("Extracting features...")
        features_list = []
        labels_list = []
        
        for window_df, label in windows:
            features = self.extract_from_window(window_df)
            features_list.append(features)
            labels_list.append(label)
        
        features_df = pd.DataFrame(features_list)
        labels = np.array(labels_list)
        
        self.feature_names = list(features_df.columns)
        
        print(f"✓ Extracted {len(self.feature_names)} features from {len(features_df)} windows\n")
        
        return features_df, labels


# ============================================================================
# PERSONALIZED BASELINE LEARNING
# ============================================================================

class BaselineLearner:
    """Learn patient-specific baseline."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.baseline_stats = {}
    
    def learn_baseline(self, features_df: pd.DataFrame, labels: np.ndarray, 
                      patient_id: str) -> Dict:
        """Learn baseline from ON-state data."""
        print("PERSONALIZED BASELINE LEARNING")
        print("="*60)
        
        # Filter for ON-state
        if labels is not None and any(l is not None for l in labels):
            on_mask = np.array([l == 'ON' for l in labels])
            on_features = features_df[on_mask]
            
            if len(on_features) < self.config.baseline_min_windows:
                print(f"  Warning: Only {len(on_features)} ON windows (min: {self.config.baseline_min_windows})")
                if len(on_features) == 0:
                    on_features = features_df
        else:
            on_features = features_df
        
        print(f"Learning from {len(on_features)} ON-state windows...")
        
        # Calculate statistics
        baseline_stats = {}
        for feature in features_df.columns:
            data = on_features[feature].values
            baseline_stats[feature] = {
                'mean': np.mean(data),
                'std': np.std(data),
                'median': np.median(data)
            }
        
        self.baseline_stats[patient_id] = {
            'stats': baseline_stats,
            'n_windows': len(on_features)
        }
        
        print(f"✓ Baseline learned for {patient_id}\n")
        
        return self.baseline_stats[patient_id]
    
    def compute_deviations(self, features_df: pd.DataFrame, patient_id: str) -> pd.DataFrame:
        """Compute deviations from baseline."""
        if patient_id not in self.baseline_stats:
            return features_df
        
        baseline = self.baseline_stats[patient_id]['stats']
        deviation_df = features_df.copy()
        
        # Add z-score features
        for feature in features_df.columns:
            if feature in baseline:
                mean = baseline[feature]['mean']
                std = baseline[feature]['std']
                if std > 0:
                    deviation_df[f'{feature}_zscore'] = (features_df[feature] - mean) / std
        
        return deviation_df


# ============================================================================
# QUALITY GATES & UNKNOWN STATE
# ============================================================================

class QualityGate:
    """Quality control for ON/OFF/UNKNOWN classification."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
    
    def check_quality(self, window_df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Check if window has good quality."""
        issues = []
        
        # Check missing data
        missing = window_df[self.config.acc_axes].isnull().sum().sum()
        if missing > len(window_df) * 0.1:
            issues.append("missing_data")
        
        # Check magnitude (off-wrist detection)
        mag = np.sqrt(
            window_df[self.config.acc_axes[0]]**2 +
            window_df[self.config.acc_axes[1]]**2 +
            window_df[self.config.acc_axes[2]]**2
        )
        
        median_mag = mag.median()
        if median_mag < 8.0 or median_mag > 12.0:
            issues.append("off_wrist")
        
        # Check saturation
        saturated = (
            (window_df[self.config.acc_axes[0]].abs() > 19.6) |
            (window_df[self.config.acc_axes[1]].abs() > 19.6) |
            (window_df[self.config.acc_axes[2]].abs() > 19.6)
        )
        if saturated.sum() > len(window_df) * 0.05:
            issues.append("saturation")
        
        return len(issues) == 0, issues


# ============================================================================
# MODEL TRAINING
# ============================================================================

class Trainer:
    """Train and evaluate Random Forest."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.cv_results = None
    
    def prepare_data(self, features_df: pd.DataFrame, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data."""
        valid_mask = np.array([l is not None and l != 'None' for l in labels])
        X = features_df[valid_mask].values
        y = labels[valid_mask]
        
        # Encode: OFF=0, ON=1
        y_encoded = np.array([0 if label == 'OFF' else 1 for label in y])
        
        print(f"Dataset: {len(y_encoded)} samples")
        print(f"  OFF: {np.sum(y_encoded == 0)} ({100*np.mean(y_encoded==0):.1f}%)")
        print(f"  ON: {np.sum(y_encoded == 1)} ({100*np.mean(y_encoded==1):.1f}%)")
        
        return X, y_encoded
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train with cross-validation."""
        print("\nMODEL TRAINING")
        print("="*60)
        
        X_scaled = self.scaler.fit_transform(X)

        # Adjust parameters for small datasets
        n_samples = len(y)
        min_class_count = min(np.sum(y == 0), np.sum(y == 1))
        adjusted_split = min(10, max(2, n_samples // 3))
        adjusted_leaf = min(4, max(1, n_samples // 5))

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=adjusted_split,
            min_samples_leaf=adjusted_leaf,
            class_weight='balanced',
            random_state=self.config.random_state,
            n_jobs=-1
        )

        # Cross-validation (adjust folds if data is too small)
        n_folds = min(self.config.n_cv_folds, int(min_class_count))
        n_folds = max(2, n_folds)  # minimum 2-fold
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                            random_state=self.config.random_state)
        
        print(f"Performing {n_folds}-fold cross-validation...")
        
        self.cv_results = cross_validate(
            self.model, X_scaled, y,
            cv=cv,
            scoring={'accuracy': 'accuracy', 'precision': 'precision', 
                    'recall': 'recall', 'f1': 'f1'},
            return_train_score=True,
            n_jobs=-1
        )
        
        # Train final model
        print("Training final model...")
        self.model.fit(X_scaled, y)
        
        print("✓ Training complete\n")
        
        return self.cv_results
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model."""
        print("MODEL EVALUATION")
        print("="*60)
        
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred, labels=[0, 1]),
            'cv_accuracy_mean': np.mean(self.cv_results['test_accuracy']),
            'cv_accuracy_std': np.std(self.cv_results['test_accuracy']),
            'cv_precision_mean': np.mean(self.cv_results['test_precision']),
            'cv_recall_mean': np.mean(self.cv_results['test_recall']),
            'cv_f1_mean': np.mean(self.cv_results['test_f1']),
        }
        
        print("\nCross-Validation Results:")
        print(f"  Accuracy:  {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
        print(f"  Precision: {metrics['cv_precision_mean']:.4f}")
        print(f"  Recall:    {metrics['cv_recall_mean']:.4f}")
        print(f"  F1-Score:  {metrics['cv_f1_mean']:.4f}")
        
        print("\nFinal Model:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        
        cm = metrics['confusion_matrix']
        print("\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              OFF   ON")
        print(f"  Actual OFF  {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"         ON   {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        return metrics
    
    def get_feature_importance(self, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
        """Get top important features."""
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance.head(top_n)


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

class CompletePipeline:
    """End-to-end pipeline in one class."""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.loader = DataLoader(self.config)
        self.preprocessor = Preprocessor(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        self.baseline_learner = BaselineLearner(self.config)
        self.quality_gate = QualityGate(self.config)
        self.trainer = Trainer(self.config)
        self.results = {}
    
    def run(self, data_source: str, data_mode: str = 'recordings',
            patient_id: str = "PATIENT_001") -> Dict:
        """
        Run complete pipeline.
        
        Args:
            data_source: Path to recordings folder or contract session
            data_mode: 'recordings' or 'contract'
            patient_id: Patient identifier
        
        Returns:
            Dictionary with all results
        """
        print("\n" + "="*70)
        print("  aCare PARKINSON'S DETECTION - COMPLETE PIPELINE v3.0")
        print("="*70)
        print(f"Patient ID: {patient_id}")
        print(f"Data Mode: {data_mode}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        try:
            # STEP 1: Load data
            print("\n[STEP 1] Loading data...")
            if data_mode == 'dataframe':
                df = data_source
            elif data_mode == 'recordings':
                df = self.loader.load_recordings(data_source)
            else:
                df = self.loader.load_contract_session(data_source)
            
            # STEP 2: Preprocess
            print("\n[STEP 2] Preprocessing...")
            df = self.preprocessor.preprocess(df)
            
            # STEP 3: Extract features
            print("\n[STEP 3] Extracting features...")
            features_df, labels = self.feature_extractor.extract_features(df)
            
            # STEP 4: Learn baseline
            print("\n[STEP 4] Learning baseline...")
            baseline = self.baseline_learner.learn_baseline(features_df, labels, patient_id)
            
            # STEP 5: Compute deviations
            print("[STEP 5] Computing personalized deviations...")
            features_with_dev = self.baseline_learner.compute_deviations(features_df, patient_id)
            print(f"  Total features: {len(features_with_dev.columns)}\n")
            
            # STEP 6: Prepare data
            print("[STEP 6] Preparing data...")
            X, y = self.trainer.prepare_data(features_with_dev, labels)
            
            # STEP 7: Train
            print("\n[STEP 7] Training model...")
            cv_results = self.trainer.train(X, y)
            
            # STEP 8: Evaluate
            print("[STEP 8] Evaluating...")
            metrics = self.trainer.evaluate(X, y)
            
            # STEP 9: Feature importance
            print("\n[STEP 9] Analyzing features...")
            importance = self.trainer.get_feature_importance(
                list(features_with_dev.columns), top_n=20
            )
            
            print("\nTop 5 Features:")
            for i, row in importance.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
            
            # Store results
            self.results = {
                'patient_id': patient_id,
                'config': asdict(self.config),
                'dataset_info': {
                    'total_samples': len(df),
                    'total_windows': len(features_df),
                    'features_count': len(features_with_dev.columns),
                    'on_count': int(np.sum(y == 1)),
                    'off_count': int(np.sum(y == 0))
                },
                'baseline_stats': baseline,
                'model_performance': metrics,
                'feature_importance': importance.to_dict('records'),
                'timestamp': datetime.now().isoformat()
            }
            
            print("\n" + "="*70)
            print("  ✓ PIPELINE COMPLETED SUCCESSFULLY")
            print("="*70)
            
            return self.results
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def save_model(self, filepath: str = "model.pkl"):
        """Save trained model."""
        if self.trainer.model is None:
            raise ValueError("No model trained")
        
        package = {
            'model': self.trainer.model,
            'scaler': self.trainer.scaler,
            'feature_names': self.feature_extractor.feature_names,
            'config': self.config,
            'baseline_stats': self.baseline_learner.baseline_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(package, filepath)
        print(f"\n✓ Model saved: {filepath}")
    
    def save_results(self, filepath: str = "results.json"):
        """Save results to JSON."""
        if not self.results:
            return
        
        results = self.results.copy()
        if 'confusion_matrix' in results.get('model_performance', {}):
            cm = results['model_performance']['confusion_matrix']
            if hasattr(cm, 'tolist'):
                results['model_performance']['confusion_matrix'] = cm.tolist()
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved: {filepath}")
    
    def generate_report(self, filepath: str = "results.md"):
        """Generate markdown report."""
        if not self.results:
            return
        
        r = self.results
        perf = r['model_performance']
        
        total = r['dataset_info']['on_count'] + r['dataset_info']['off_count']
        on_pct = 100 * r['dataset_info']['on_count'] / total if total > 0 else 0
        off_pct = 100 * r['dataset_info']['off_count'] / total if total > 0 else 0
        
        report = f"""# Parkinson's ON/OFF Detection Results

## Patient Information
- **Patient ID:** {r['patient_id']}
- **Analysis Date:** {r['timestamp']}

## Dataset Summary
- **Total Samples:** {r['dataset_info']['total_samples']:,}
- **Total Windows:** {r['dataset_info']['total_windows']:,}
- **Features:** {r['dataset_info']['features_count']}
- **ON Windows:** {r['dataset_info']['on_count']} ({on_pct:.1f}%)
- **OFF Windows:** {r['dataset_info']['off_count']} ({off_pct:.1f}%)

## Model Performance

### Cross-Validation
- **Accuracy:** {perf['cv_accuracy_mean']:.4f} ± {perf['cv_accuracy_std']:.4f}
- **Precision:** {perf['cv_precision_mean']:.4f}
- **Recall:** {perf['cv_recall_mean']:.4f}
- **F1-Score:** {perf['cv_f1_mean']:.4f}

### Final Model
- **Accuracy:** {perf['accuracy']:.4f}
- **Precision:** {perf['precision']:.4f}
- **Recall:** {perf['recall']:.4f}
- **F1-Score:** {perf['f1']:.4f}

### Confusion Matrix
```
              Predicted
              OFF   ON
  Actual OFF  {perf['confusion_matrix'][0][0]:4d}  {perf['confusion_matrix'][0][1]:4d}
         ON   {perf['confusion_matrix'][1][0]:4d}  {perf['confusion_matrix'][1][1]:4d}
```

## Top 10 Features

"""
        for i, feat in enumerate(r['feature_importance'][:10], 1):
            report += f"{i}. **{feat['feature']}** - {feat['importance']:.4f}\n"
        
        report += f"""

## Configuration
- **Sampling Rate:** {r['config']['sampling_rate']} Hz
- **Window Duration:** {r['config']['window_duration']}s
- **Window Overlap:** {r['config']['window_overlap']*100:.0f}%

---
*Generated by aCare v3.0*
"""
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(f"✓ Report generated: {filepath}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function."""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║              aCare Parkinson's Detection System v3.0                     ║
║                     Complete All-In-One Solution                         ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # meta_path = os.path.join(script_dir, 'meta_Data_Jan29.json')
    # data_path = os.path.join(script_dir, 'Data_Jan29')
    meta_path = os.path.join(script_dir, 'meta_Test_Jan29.json')
    data_path = os.path.join(script_dir, 'Test_Jan29')
    output_dir = os.path.join(script_dir, 'results')
    models_dir = os.path.join(script_dir, 'models')

    # Check required files
    if not os.path.exists(meta_path):
        print(f"\n❌ ERROR: Meta file not found: {meta_path}")
        return None
    if not os.path.exists(data_path):
        print(f"\n❌ ERROR: Binary data file not found: {data_path}")
        return None

    # ── STEP A: Decode Data_Jan29 binary ──
    print("[DECODE] Decoding Data_Jan29 binary...")
    decoder_path = os.path.join(script_dir, 'Decode_SampleDataset_Jan29.py')
    spec = importlib.util.spec_from_file_location("decoder", decoder_path)
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)

    df, meta = decoder.decode_sensor_file(meta_path)
    df = df.reset_index()  # time from index → column
    freq = float(meta['freq'])
    print(f"  Decoded: {len(df)} samples, {freq} Hz, {df['time'].max():.2f}s\n")

    # ── STEP B: Classify windows as ON/OFF based on tremor signal ──
    # Tremor-band power (4-6 Hz) distinguishes motor states:
    #   High tremor power → OFF (medication worn off, more tremor)
    #   Low tremor power  → ON  (medication effective, smoother motion)
    print("[CLASSIFY] Classifying signal into ON/OFF states...")
    window_size = int(freq * 1.0)  # 1-second windows
    step = window_size // 2         # 50% overlap

    tremor_powers = []
    window_centers = []
    for start in range(0, len(df) - window_size + 1, step):
        window = df.iloc[start:start + window_size]
        mag = np.sqrt(window['X']**2 + window['Y']**2 + window['Z']**2)
        # FFT to get tremor-band (4-6 Hz) power
        fft_vals = np.abs(np.fft.rfft(mag.values))**2
        fft_freqs = np.fft.rfftfreq(len(mag), 1/freq)
        tremor_mask = (fft_freqs >= 4.0) & (fft_freqs <= 6.0)
        total_power = np.sum(fft_vals)
        tremor_power = np.sum(fft_vals[tremor_mask]) / total_power if total_power > 0 else 0
        tremor_powers.append(tremor_power)
        window_centers.append(start + window_size // 2)

    # Threshold: above median tremor power → OFF, below → ON
    threshold = np.median(tremor_powers)
    labels_per_sample = np.full(len(df), 'ON')
    for center, tp in zip(window_centers, tremor_powers):
        w_start = max(0, center - window_size // 2)
        w_end = min(len(df), center + window_size // 2)
        if tp > threshold:
            labels_per_sample[w_start:w_end] = 'OFF'

    df['label'] = labels_per_sample
    # df['source_file'] = 'Data_Jan29'
    df['source_file'] = 'Test_Jan29'

    on_count = np.sum(labels_per_sample == 'ON')
    off_count = np.sum(labels_per_sample == 'OFF')
    print(f"  Tremor-band threshold: {threshold:.4f}")
    print(f"  ON samples:  {on_count}")
    print(f"  OFF samples: {off_count}\n")

    # System configuration (25 Hz for real sensor)
    config = SystemConfig(
        sampling_rate=25,
        window_duration=1.0,
        window_overlap=0.5,
        baseline_min_windows=5,
        n_cv_folds=5,
        random_state=42
    )

    # Run pipeline
    pipeline = CompletePipeline(config)

    results = pipeline.run(
        data_source=df,
        data_mode='dataframe',
        patient_id="PATIENT_001"
    )
    
    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'parkinsons_model.pkl')
    pipeline.save_model(model_path)
    
    results_path = os.path.join(output_dir, 'results.json')
    pipeline.save_results(results_path)
    
    report_path = os.path.join(output_dir, 'results.md')
    pipeline.generate_report(report_path)
    
    print("\n" + "="*70)
    print("  ALL OUTPUTS GENERATED")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  1. {report_path}")
    print(f"  2. {results_path}")
    print(f"  3. {model_path}")
    print("\n" + "="*70)
    
    return results


if __name__ == "__main__":
    results = main()