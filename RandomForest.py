"""
Parkinson's Detection Pipeline - Random Forest Classifier
Uses decoded IMU data from decoded_csv folder.

Design Patterns:
- Strategy Pattern: Interchangeable feature extractors
- Composition: Pipeline coordinates specialized components
- Config Pattern: Centralized configuration via dataclasses

SOLID Principles:
- SRP: Each class has single responsibility
- OCP: New extractors can be added without modifying existing code
- LSP: All extractors implement common interface
- DIP: Pipeline depends on abstractions, not concrete implementations
"""

import os
import logging
import argparse
import joblib
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Protocol
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import kurtosis, skew, iqr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Logging Setup (SRP)
# =============================================================================
def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Configure and return the project logger."""
    logger = logging.getLogger("RF_Pipeline")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Console handler
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console)

        # File handler (optional)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)

    return logger


log = setup_logging()


# =============================================================================
# Configuration (Config Pattern)
# =============================================================================
@dataclass
class FilterConfig:
    """Configuration for signal filtering."""
    cutoff: float = 15.0
    fs: int = 32
    order: int = 4


@dataclass
class WindowConfig:
    """Configuration for data windowing."""
    size: int = 320  # samples per window
    overlap: float = 0.5  # 50% overlap


@dataclass
class ModelConfig:
    """Configuration for Random Forest model."""
    n_estimators: int = 100
    max_depth: int = 7
    min_samples_leaf: int = 2
    max_features: str = 'sqrt'
    random_state: int = 42


@dataclass
class PipelineConfig:
    """Global configuration for the pipeline."""
    data_dir: Path = field(default_factory=lambda: Path("decoded_csv"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    labels_file: Optional[Path] = None
    fs: int = 32
    test_size: float = 0.3
    top_k_features: int = 15
    fast_mode: bool = False
    filter_config: FilterConfig = field(default_factory=FilterConfig)
    window_config: WindowConfig = field(default_factory=WindowConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)


# =============================================================================
# Signal Processing (Strategy Pattern + SRP)
# =============================================================================
class SignalFilter(Protocol):
    """Protocol defining interface for signal filters."""
    def apply(self, signal: np.ndarray) -> np.ndarray: ...


class ButterworthFilter:
    """Butterworth low-pass filter implementation."""

    def __init__(self, config: FilterConfig):
        self.config = config
        self._coeffs: Optional[Tuple] = None

    def _get_coeffs(self) -> Tuple:
        """Lazy computation of filter coefficients."""
        if self._coeffs is None:
            nyquist = 0.5 * self.config.fs
            normalized_cutoff = self.config.cutoff / nyquist
            self._coeffs = butter(self.config.order, normalized_cutoff, btype="low")
        return self._coeffs

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Apply low-pass filter to signal."""
        if len(signal) < 13:  # Minimum length for filtfilt
            return signal
        b, a = self._get_coeffs()
        return filtfilt(b, a, signal)


class Preprocessor:
    """Coordinates signal preprocessing (Composition)."""

    def __init__(self, signal_filter: SignalFilter):
        self.filter = signal_filter

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply filter to all sensor axes."""
        df_filtered = df.copy()
        axes = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']

        for axis in axes:
            if axis in df_filtered.columns:
                df_filtered[axis] = self.filter.apply(df_filtered[axis].values)

        return df_filtered


# =============================================================================
# Data Windowing (SRP)
# =============================================================================
class WindowSegmenter:
    """Segments time-series data into fixed-size windows."""

    def __init__(self, config: WindowConfig):
        self.size = config.size
        self.step = int(config.size * (1 - config.overlap))

    def segment(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Split DataFrame into overlapping windows."""
        windows = []
        for start in range(0, len(df) - self.size + 1, self.step):
            window = df.iloc[start:start + self.size].copy()
            windows.append(window)
        return windows


# =============================================================================
# Feature Extraction (Strategy Pattern + OCP + LSP)
# =============================================================================
class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    @abstractmethod
    def extract(self, acc_data: pd.DataFrame, gyro_data: pd.DataFrame) -> Dict[str, float]:
        """Extract features from sensor data."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return extractor name for logging."""
        pass


class StatisticalExtractor(FeatureExtractor):
    """Extracts time-domain statistical features."""

    @property
    def name(self) -> str:
        return "Statistical"

    def _compute_stats(self, signal: np.ndarray, prefix: str) -> Dict[str, float]:
        """Compute statistical features for a single signal."""
        return {
            f"{prefix}_mean": np.mean(signal),
            f"{prefix}_std": np.std(signal),
            f"{prefix}_min": np.min(signal),
            f"{prefix}_max": np.max(signal),
            f"{prefix}_range": np.ptp(signal),
            f"{prefix}_rms": np.sqrt(np.mean(signal ** 2)),
            f"{prefix}_skew": skew(signal),
            f"{prefix}_kurtosis": kurtosis(signal),
            f"{prefix}_iqr": iqr(signal),
            f"{prefix}_median": np.median(signal),
        }

    def extract(self, acc_data: pd.DataFrame, gyro_data: pd.DataFrame) -> Dict[str, float]:
        """Extract statistical features from accelerometer and gyroscope data."""
        features = {}

        # Accelerometer features
        for axis in ['Ax', 'Ay', 'Az']:
            if axis in acc_data.columns:
                features.update(self._compute_stats(acc_data[axis].values, f"acc_{axis}"))

        # Gyroscope features
        for axis in ['Gx', 'Gy', 'Gz']:
            if axis in gyro_data.columns:
                features.update(self._compute_stats(gyro_data[axis].values, f"gyro_{axis}"))

        # Magnitude features
        if all(ax in acc_data.columns for ax in ['Ax', 'Ay', 'Az']):
            acc_mag = np.sqrt(acc_data['Ax']**2 + acc_data['Ay']**2 + acc_data['Az']**2)
            features.update(self._compute_stats(acc_mag.values, "acc_mag"))

        if all(ax in gyro_data.columns for ax in ['Gx', 'Gy', 'Gz']):
            gyro_mag = np.sqrt(gyro_data['Gx']**2 + gyro_data['Gy']**2 + gyro_data['Gz']**2)
            features.update(self._compute_stats(gyro_mag.values, "gyro_mag"))

        return features


class GaitExtractor(FeatureExtractor):
    """Extracts gait-specific features from movement data."""

    def __init__(self, fs: int = 32):
        self.fs = fs

    @property
    def name(self) -> str:
        return "Gait"

    def extract(self, acc_data: pd.DataFrame, gyro_data: pd.DataFrame) -> Dict[str, float]:
        """Extract gait features like step count, cadence, arm swing."""
        features = {}

        # Step detection using vertical acceleration (Ay typically)
        if 'Ay' in acc_data.columns:
            signal = acc_data['Ay'].values
            peaks, properties = find_peaks(signal, prominence=0.5, distance=self.fs // 2)

            features['steps'] = len(peaks)

            if len(peaks) >= 2:
                step_times = np.diff(peaks) / self.fs
                features['step_time_mean'] = np.mean(step_times)
                features['step_time_std'] = np.std(step_times)
                features['cadence'] = len(peaks) / (len(acc_data) / self.fs / 60)  # steps per minute
                features['step_regularity'] = 1.0 / (np.std(step_times) + 1e-6)
            else:
                features['step_time_mean'] = 0.0
                features['step_time_std'] = 0.0
                features['cadence'] = 0.0
                features['step_regularity'] = 0.0

        # Arm swing magnitude from gyroscope
        if all(ax in gyro_data.columns for ax in ['Gx', 'Gy', 'Gz']):
            gyro_mag = np.sqrt(gyro_data['Gx']**2 + gyro_data['Gy']**2 + gyro_data['Gz']**2)
            features['arm_swing'] = np.ptp(gyro_mag)
            features['arm_swing_mean'] = np.mean(gyro_mag)

        return features


class FrequencyExtractor(FeatureExtractor):
    """Extracts frequency-domain features using FFT."""

    def __init__(self, fs: int = 32):
        self.fs = fs

    @property
    def name(self) -> str:
        return "Frequency"

    def _compute_freq_features(self, signal: np.ndarray, prefix: str) -> Dict[str, float]:
        """Compute frequency domain features."""
        n = len(signal)
        fft_vals = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(n, 1 / self.fs)

        # Dominant frequency
        dominant_idx = np.argmax(fft_vals[1:]) + 1  # Skip DC component
        dominant_freq = freqs[dominant_idx]

        # Spectral energy
        spectral_energy = np.sum(fft_vals ** 2)

        # Spectral centroid
        spectral_centroid = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-6)

        return {
            f"{prefix}_dominant_freq": dominant_freq,
            f"{prefix}_spectral_energy": spectral_energy,
            f"{prefix}_spectral_centroid": spectral_centroid,
        }

    def extract(self, acc_data: pd.DataFrame, gyro_data: pd.DataFrame) -> Dict[str, float]:
        """Extract frequency features from sensor data."""
        features = {}

        # Accelerometer frequency features
        if all(ax in acc_data.columns for ax in ['Ax', 'Ay', 'Az']):
            acc_mag = np.sqrt(acc_data['Ax']**2 + acc_data['Ay']**2 + acc_data['Az']**2)
            features.update(self._compute_freq_features(acc_mag.values, "acc"))

        # Gyroscope frequency features
        if all(ax in gyro_data.columns for ax in ['Gx', 'Gy', 'Gz']):
            gyro_mag = np.sqrt(gyro_data['Gx']**2 + gyro_data['Gy']**2 + gyro_data['Gz']**2)
            features.update(self._compute_freq_features(gyro_mag.values, "gyro"))

        return features


# =============================================================================
# Feature Extraction Coordinator (Composition + DIP)
# =============================================================================
class FeatureExtractorCoordinator:
    """Coordinates multiple feature extractors."""

    def __init__(self, extractors: List[FeatureExtractor]):
        self.extractors = extractors

    def extract_all(self, acc_data: pd.DataFrame, gyro_data: pd.DataFrame) -> Dict[str, float]:
        """Run all extractors and combine features."""
        all_features = {}
        for extractor in self.extractors:
            features = extractor.extract(acc_data, gyro_data)
            all_features.update(features)
        return all_features


# =============================================================================
# Data Loader (SRP)
# =============================================================================
class DataLoader:
    """Loads and manages decoded CSV data."""

    def __init__(self, data_dir: Path, labels_file: Optional[Path] = None):
        self.data_dir = data_dir
        self.labels = self._load_labels(labels_file) if labels_file else {}

    def _load_labels(self, labels_file: Path) -> Dict[str, int]:
        """Load user labels from CSV file."""
        if not labels_file.exists():
            return {}
        df = pd.read_csv(labels_file)
        return dict(zip(df['user_id'], df['label']))

    def get_user_dirs(self) -> List[Path]:
        """Get all user directories."""
        return [d for d in self.data_dir.iterdir() if d.is_dir()]

    def load_user_data(self, user_dir: Path, max_files: Optional[int] = None) -> List[pd.DataFrame]:
        """Load CSV files for a user (with optional limit for large datasets)."""
        csv_files = sorted(user_dir.glob("*.csv"))
        if max_files:
            csv_files = csv_files[:max_files]

        dataframes = []
        total = len(csv_files)

        for i, csv_file in enumerate(csv_files):
            if total > 100 and (i + 1) % 500 == 0:
                log.info(f"  Loading file {i + 1}/{total}...")
            try:
                df = pd.read_csv(csv_file)
                if not df.empty:
                    dataframes.append(df)
            except Exception as e:
                log.warning(f"Failed to load {csv_file}: {e}")

        return dataframes

    def get_label(self, user_id: str) -> Optional[int]:
        """Get label for a user."""
        return self.labels.get(user_id)


# =============================================================================
# Model Manager (SRP)
# =============================================================================
class ModelManager:
    """Manages ML model lifecycle."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            max_features=config.max_features,
            random_state=config.random_state,
            n_jobs=-1
        )
        self.selected_features: List[str] = []

    def select_features(self, X: pd.DataFrame, y: np.ndarray, k: int) -> pd.DataFrame:
        """Select top k features using feature importance."""
        log.info(f"Selecting top {k} features...")

        # Train preliminary model for feature importance
        selector = RandomForestClassifier(
            n_estimators=100,
            random_state=self.config.random_state
        )
        selector.fit(X, y)

        # Get top k features
        importances = selector.feature_importances_
        indices = np.argsort(importances)[::-1][:k]
        self.selected_features = list(X.columns[indices])

        log.info(f"Selected features: {self.selected_features}")
        return X[self.selected_features]

    def train(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Train the model."""
        log.info("Training Random Forest model...")
        self.model.fit(X, y)

    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict:
        """Evaluate model performance."""
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)

        results = {
            'accuracy': accuracy_score(y_test, predictions),
            'predictions': predictions,
            'probabilities': probabilities,
            'y_true': y_test,
            'report': classification_report(y_test, predictions, target_names=['ON', 'OFF']),
            'confusion_matrix': confusion_matrix(y_test, predictions),
        }

        # ROC AUC for binary classification
        if len(np.unique(y_test)) == 2:
            results['roc_auc'] = roc_auc_score(y_test, probabilities[:, 1])
            results['fpr'], results['tpr'], _ = roc_curve(y_test, probabilities[:, 1])

        return results

    def cross_validate(self, X: pd.DataFrame, y: np.ndarray, n_splits: int = 5) -> np.ndarray:
        """Perform cross-validation."""
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_state)
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        return scores

    def save_model(self, output_dir: Path) -> Path:
        """Save the trained model and selected features."""
        model_path = output_dir / "trained_model.joblib"
        # Save config as dict to avoid pickle issues with dataclasses
        config_dict = {
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'min_samples_leaf': self.config.min_samples_leaf,
            'max_features': self.config.max_features,
            'random_state': self.config.random_state,
        }
        model_data = {
            'model': self.model,
            'selected_features': self.selected_features,
            'config': config_dict,
        }
        joblib.dump(model_data, model_path)
        log.info(f"Model saved to {model_path}")
        return model_path

    @staticmethod
    def load_model(model_path: Path) -> 'ModelManager':
        """Load a trained model from file."""
        model_data = joblib.load(model_path)
        config_dict = model_data['config']
        config = ModelConfig(
            n_estimators=config_dict['n_estimators'],
            max_depth=config_dict['max_depth'],
            min_samples_leaf=config_dict['min_samples_leaf'],
            max_features=config_dict['max_features'],
            random_state=config_dict['random_state'],
        )
        manager = ModelManager(config)
        manager.model = model_data['model']
        manager.selected_features = model_data['selected_features']
        log.info(f"Model loaded from {model_path}")
        return manager


# =============================================================================
# Report Generator (SRP)
# =============================================================================
class ReportGenerator:
    """Generates evaluation reports and visualizations."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def save_text_report(self, results: Dict, cv_scores: np.ndarray, selected_features: List[str]) -> None:
        """Save text evaluation report."""
        report_path = self.output_dir / "evaluation_report.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("PARKINSON'S DETECTION - EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write("MODEL PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write(f"Test Accuracy: {results['accuracy']:.4f}\n")
            if 'roc_auc' in results:
                f.write(f"ROC AUC Score: {results['roc_auc']:.4f}\n")
            f.write(f"\nCross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n")

            f.write("\n\nCLASSIFICATION REPORT\n")
            f.write("-" * 40 + "\n")
            f.write(results['report'])

            f.write("\n\nSELECTED FEATURES\n")
            f.write("-" * 40 + "\n")
            for i, feat in enumerate(selected_features, 1):
                f.write(f"{i:2}. {feat}\n")

        log.info(f"Report saved to {report_path}")

    def save_feature_importance_plot(self, model: RandomForestClassifier, features: List[str]) -> None:
        """Save feature importance bar chart."""
        plt.figure(figsize=(10, 8))
        importances = model.feature_importances_

        # Sort by importance
        indices = np.argsort(importances)

        plt.barh(range(len(indices)), importances[indices], color='steelblue')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Random Forest - Feature Importances')
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_importance.png", dpi=150)
        plt.close()

        log.info("Feature importance plot saved")

    def save_confusion_matrix_plot(self, cm: np.ndarray) -> None:
        """Save confusion matrix heatmap."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['ON', 'OFF'], yticklabels=['ON', 'OFF'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=150)
        plt.close()

        log.info("Confusion matrix plot saved")

    def save_roc_curve_plot(self, fpr: np.ndarray, tpr: np.ndarray, auc: float) -> None:
        """Save ROC curve plot."""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(self.output_dir / "roc_curve.png", dpi=150)
        plt.close()

        log.info("ROC curve plot saved")

    def save_features_csv(self, df: pd.DataFrame) -> None:
        """Save extracted features to CSV."""
        csv_path = self.output_dir / "extracted_features.csv"
        df.to_csv(csv_path, index=False)
        log.info(f"Features saved to {csv_path}")


# =============================================================================
# Main Pipeline (Composition + DIP)
# =============================================================================
class Pipeline:
    """Orchestrates the complete ML pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Initialize components (Dependency Injection)
        self.preprocessor = Preprocessor(ButterworthFilter(config.filter_config))
        self.segmenter = WindowSegmenter(config.window_config)
        self.extractor_coordinator = FeatureExtractorCoordinator([
            StatisticalExtractor(),
            GaitExtractor(config.fs),
            FrequencyExtractor(config.fs),
        ])
        self.data_loader = DataLoader(config.data_dir, config.labels_file)
        self.model_manager = ModelManager(config.model_config)
        self.report_generator = ReportGenerator(config.output_dir)

    def _extract_features_from_windows(self, windows: List[pd.DataFrame]) -> List[Dict]:
        """Extract features from all windows."""
        features_list = []

        for window in windows:
            # Separate accelerometer and gyroscope data
            acc_cols = ['Ax', 'Ay', 'Az']
            gyro_cols = ['Gx', 'Gy', 'Gz']

            acc_data = window[acc_cols] if all(c in window.columns for c in acc_cols) else pd.DataFrame()
            gyro_data = window[gyro_cols] if all(c in window.columns for c in gyro_cols) else pd.DataFrame()

            features = self.extractor_coordinator.extract_all(acc_data, gyro_data)
            features_list.append(features)

        return features_list

    def run(self) -> None:
        """Execute the complete pipeline."""
        log.info("=" * 60)
        log.info("Starting Parkinson's Detection Pipeline")
        log.info("=" * 60)

        # Step 1: Load and process data
        log.info("\n[1/5] Loading and processing data...")
        all_features = []
        all_labels = []

        user_dirs = self.data_loader.get_user_dirs()
        log.info(f"Found {len(user_dirs)} user directories")

        for idx, user_dir in enumerate(user_dirs):
            user_id = user_dir.name
            label = self.data_loader.get_label(user_id)

            if label is None:
                log.warning(f"No label found for user {user_id}, skipping...")
                continue

            log.info(f"Processing user {idx + 1}/{len(user_dirs)}: {user_id} (label={label})")

            # Load user data (limit files per user for large datasets)
            user_dataframes = self.data_loader.load_user_data(user_dir, max_files=1000)
            log.info(f"  Loaded {len(user_dataframes)} files")

            for df in user_dataframes:
                # Preprocess
                df_processed = self.preprocessor.process(df)

                # Segment into windows
                windows = self.segmenter.segment(df_processed)

                # Extract features
                features_list = self._extract_features_from_windows(windows)

                for features in features_list:
                    features['user_id'] = user_id
                    all_features.append(features)
                    all_labels.append(label)

        if not all_features:
            log.error("No data found! Make sure labels_file is provided with user_id,label columns.")
            log.error("Example labels.csv:")
            log.error("  user_id,label")
            log.error("  oA9zb5Jvsia7Tl9fTnqhYzLhkXUo,1")
            log.error("  akwn2joina1,0")
            return

        # Create DataFrame
        features_df = pd.DataFrame(all_features)
        labels_array = np.array(all_labels)

        log.info(f"Extracted {len(features_df)} feature windows")
        log.info(f"Class distribution: ON={sum(labels_array == 0)}, OFF={sum(labels_array == 1)}")

        # Save features
        features_df['label'] = labels_array
        self.report_generator.save_features_csv(features_df)

        # Step 2: Prepare features
        log.info("\n[2/5] Preparing features...")
        X = features_df.drop(['user_id', 'label'], axis=1, errors='ignore')
        X = X.select_dtypes(include=[np.number])
        # Handle NaN and Infinity values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        # Clip extreme values to prevent overflow
        X = X.clip(-1e10, 1e10)
        y = labels_array

        # Step 3: Feature selection
        log.info("\n[3/5] Selecting features...")
        X_selected = self.model_manager.select_features(X, y, self.config.top_k_features)

        # Step 4: Train and evaluate
        log.info("\n[4/5] Training and evaluating model...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y,
            test_size=self.config.test_size,
            random_state=self.config.model_config.random_state,
            stratify=y
        )

        self.model_manager.train(X_train, y_train)

        # Evaluate
        results = self.model_manager.evaluate(X_test, y_test)
        cv_scores = self.model_manager.cross_validate(X_selected, y)

        log.info(f"\nTest Accuracy: {results['accuracy']:.4f}")
        log.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Save trained model
        self.model_manager.save_model(self.config.output_dir)

        # Step 5: Generate reports
        log.info("\n[5/5] Generating reports...")
        self.report_generator.save_text_report(results, cv_scores, self.model_manager.selected_features)

        if not self.config.fast_mode:
            self.report_generator.save_feature_importance_plot(
                self.model_manager.model,
                self.model_manager.selected_features
            )
            self.report_generator.save_confusion_matrix_plot(results['confusion_matrix'])

            if 'roc_auc' in results:
                self.report_generator.save_roc_curve_plot(
                    results['fpr'], results['tpr'], results['roc_auc']
                )

        log.info("\n" + "=" * 60)
        log.info("Pipeline completed successfully!")
        log.info(f"Results saved to: {self.config.output_dir}")
        log.info("=" * 60)


# =============================================================================
# CLI Entry Point
# =============================================================================
def main():
    """Command-line interface for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Parkinson's Detection Pipeline using Random Forest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python RandomForest.py --labels labels.csv
  python RandomForest.py --data decoded_csv --labels labels.csv --fast

Labels file format (CSV):
  user_id,label
  oA9zb5Jvsia7Tl9fTnqhYzLhkXUo,1
  akwn2joina1,0

  (0 = ON medication, 1 = OFF medication)
        """
    )

    parser.add_argument('--data', type=str, default='decoded_csv',
                        help='Path to decoded CSV data directory (default: decoded_csv)')
    parser.add_argument('--output', type=str, default='outputs',
                        help='Path to output directory (default: outputs)')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to labels CSV file (required)')
    parser.add_argument('--fs', type=int, default=32,
                        help='Sampling frequency in Hz (default: 32)')
    parser.add_argument('--window', type=int, default=320,
                        help='Window size in samples (default: 320)')
    parser.add_argument('--features', type=int, default=15,
                        help='Number of top features to select (default: 15)')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode - skip plot generation')

    args = parser.parse_args()

    # Build configuration
    config = PipelineConfig(
        data_dir=Path(args.data),
        output_dir=Path(args.output),
        labels_file=Path(args.labels) if args.labels else None,
        fs=args.fs,
        top_k_features=args.features,
        fast_mode=args.fast,
        window_config=WindowConfig(size=args.window),
    )

    # Run pipeline
    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()
