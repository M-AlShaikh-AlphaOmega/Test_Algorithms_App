"""
Random Forest Training Pipeline for Parkinson's OFF State Detection
aCare Project - Machine Learning Component

This script trains a Random Forest classifier to detect OFF states in 
Parkinson's disease patients using decoded IMU sensor data.

Usage:
    python RandomForest_Train.py --data_on path/to/on_data --data_off path/to/off_data

Requirements:
    - numpy, pandas, scipy, scikit-learn, matplotlib, seaborn
    - Decoded CSV files in specified directories

Author: aCare Project Team
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Protocol
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import kurtosis, skew, iqr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Log Management ---
def setup_logging():
    """Sets up the standardized project logger."""
    logger = logging.getLogger("RF_Pipeline")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(h)
    return logger

log = setup_logging()

# --- Configuration ---
@dataclass
class FilterConfig:
    """Configuration for signal filtering."""
    cutoff: float = 15.0  # Low-pass cutoff frequency (Hz)
    fs: int = 32  # Sampling frequency (Hz)
    order: int = 4  # Filter order

@dataclass
class PipelineConfig:
    """Global configuration for the pipeline execution."""
    on_data_dir: Path = Path("Decode_Dataset/DecodeDataset_Train_Test/ON")
    off_data_dir: Path = Path("Decode_Dataset/DecodeDataset_Train_Test/OFF")
    out_dir: Path = Path("Decode_Dataset/DecodeDataset_Train_Test/outputs")
    fs: int = 32  # Sampling frequency (Hz) - matches decoded data
    fast_mode: bool = False
    test_size: float = 0.3  # 30% for testing, 70% for training
    seed: int = 42  # For reproducibility
    top_k_features: int = 15  # Select top 15 most important features
    window_size_sec: int = 10  # 10-second windows for feature extraction

# --- Signal Processing ---
class SignalFilter(Protocol):
    """Protocol defining the interface for signal filtering strategies."""
    def apply(self, signal: np.ndarray) -> np.ndarray: ...

class ButterworthFilter:
    """
    Implements a Butterworth low-pass filter.
    Purpose: Remove high-frequency noise while preserving movement signals.
    """
    def __init__(self, config: FilterConfig):
        self.config = config
        self._coeffs = None

    def _get_coeffs(self):
        """Calculate filter coefficients (cached for efficiency)."""
        if self._coeffs is None:
            nyq = 0.5 * self.config.fs  # Nyquist frequency
            low = self.config.cutoff / nyq  # Normalized cutoff
            self._coeffs = butter(self.config.order, low, btype="low")
        return self._coeffs

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Applies the low-pass filter to the signal."""
        b, a = self._get_coeffs()
        return filtfilt(b, a, signal)

class Preprocessor:
    """Coordinates signal preprocessing across all axes."""
    def __init__(self, filter_obj: SignalFilter):
        self.filter = filter_obj

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the input DataFrame by applying filters.
        Works with decoded data format: Ax, Ay, Az, Gx, Gy, Gz
        """
        df_f = df.copy()
        for axis in ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]:
            if axis in df_f.columns:
                df_f[axis] = self.filter.apply(df_f[axis].values)
        return df_f

# --- Feature Extraction ---
class BaseExtractor(Protocol):
    """Protocol defining the interface for feature extraction strategies."""
    def extract(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]: ...

class StatExtractor:
    """
    Extracts time-domain statistical features.
    These features capture movement patterns and intensity.
    """
    def extract(self, data: pd.DataFrame, prefix: str = "") -> Dict[str, float]:
        """
        Calculates statistical features for all axes.
        Features: mean, std, min, max, range, rms, skewness, kurtosis, IQR
        """
        feats = {}
        axes = ["Ax", "Ay", "Az"] if "Ax" in data.columns else ["Gx", "Gy", "Gz"]
        
        for ax in axes:
            if ax not in data.columns: 
                continue
            s = data[ax].values
            pfx = f"{prefix}{ax}_"
            feats.update({
                f"{pfx}mean": np.mean(s),  # Average movement
                f"{pfx}std": np.std(s),  # Movement variability
                f"{pfx}min": np.min(s),
                f"{pfx}max": np.max(s),
                f"{pfx}range": np.ptp(s),  # Peak-to-peak range
                f"{pfx}rms": np.sqrt(np.mean(s**2)),  # Root mean square
                f"{pfx}skew": skew(s),  # Distribution asymmetry
                f"{pfx}kurt": kurtosis(s),  # Distribution peakedness
                f"{pfx}iqr": iqr(s)  # Interquartile range
            })
        return feats

class GaitExtractor:
    """
    Extracts physical gait features from movement data.
    Purpose: Detect gait-specific patterns affected by Parkinson's.
    """
    def __init__(self, fs: int): 
        self.fs = fs

    def extract(self, acc_data: pd.DataFrame, gyro_data: pd.DataFrame) -> Dict[str, float]:
        """
        Detects steps and calculates gait metrics.
        Uses accelerometer Y-axis (vertical movement) for step detection.
        """
        if "Ay" not in acc_data.columns or "Gx" not in gyro_data.columns:
            return {}
        
        # Step detection using peak finding
        # prominence=0.5: minimum height difference for peak detection
        # distance=16: minimum 0.5 seconds between steps (16 samples at 32Hz)
        peaks, _ = find_peaks(acc_data["Ay"].values, prominence=0.5, distance=16)
        
        if len(peaks) < 2:
            return {
                "steps": 0, 
                "step_time": 0.0,
                "cadence": 0.0, 
                "arm_swing": 0.0
            }
        
        # Calculate step timing
        step_times = np.diff(peaks) / self.fs  # Convert samples to seconds
        
        # Calculate arm swing magnitude using gyroscope
        gyro_mag = np.sqrt(
            gyro_data["Gx"].values**2 + 
            gyro_data["Gy"].values**2 + 
            gyro_data["Gz"].values**2
        )
        
        return {
            "steps": len(peaks),  # Total steps detected
            "step_time": np.mean(step_times),  # Average time between steps
            "cadence": len(peaks) / (len(acc_data)/self.fs/60),  # Steps per minute
            "arm_swing": np.ptp(gyro_mag)  # Range of arm rotation
        }

# --- Model Management ---
class ModelManager:
    """
    Manages the Machine Learning model lifecycle.
    Uses Random Forest: robust, interpretable, handles non-linear patterns.
    """
    def __init__(self, seed: int = 42):
        # Optimized parameters to prevent overfitting
        self.model = RandomForestClassifier(
            n_estimators=100,  # 100 trees for stable predictions
            max_depth=7,  # Limit depth to prevent overfitting
            min_samples_leaf=2,  # Minimum samples per leaf node
            max_features='sqrt',  # Use sqrt(features) for each split
            random_state=seed,  # Reproducibility
            n_jobs=-1  # Use all CPU cores
        )
        self.selected_features = []

    def select_features(self, X: pd.DataFrame, y: np.ndarray, k: int) -> pd.DataFrame:
        """
        Selects the top k most important features.
        Purpose: Reduce noise, improve generalization, faster inference.
        """
        log.info(f"Selecting top {k} features from {len(X.columns)} total features...")
        selector = RandomForestClassifier(n_estimators=100, random_state=42)
        selector.fit(X, y)
        importances = selector.feature_importances_
        indices = np.argsort(importances)[::-1][:k]
        self.selected_features = list(X.columns[indices])
        log.info(f"Selected features: {self.selected_features}")
        return X[self.selected_features]

    def train_and_evaluate(self, X: pd.DataFrame, y: np.ndarray, config: PipelineConfig) -> Dict:
        """
        Trains the model and performs comprehensive evaluation.
        Returns metrics, predictions, and classification report.
        """
        # Split data: 70% training, 30% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.seed, stratify=y
        )
        
        log.info(f"Training set: {len(X_train)} samples")
        log.info(f"Test set: {len(X_test)} samples")
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate on training and test sets
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        
        # Cross-validation: 5-fold stratified
        # Purpose: Assess model stability across different data splits
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.seed)
        cv_scores = cross_val_score(self.model, X, y, cv=cv)
        
        log.info(f"Training Accuracy: {train_acc:.4f}")
        log.info(f"Test Accuracy:     {test_acc:.4f}")
        log.info(f"CV Mean Accuracy:  {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate AUC score
        auc = roc_auc_score(y_test, y_proba)
        log.info(f"AUC Score: {auc:.4f}")
        
        return {
            "test_y": y_test,
            "preds": y_pred,
            "probs": y_proba,
            "report": classification_report(y_test, y_pred, target_names=["ON", "OFF"]),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "train_acc": train_acc,
            "test_acc": test_acc,
            "auc": auc
        }

# --- Core Pipeline Orchestrator ---
class Pipeline:
    """
    Orchestrates the end-to-end data processing and classification flow.
    """
    def __init__(self, cfg: PipelineConfig, extractors: List[BaseExtractor]):
        self.cfg = cfg
        self.prep = Preprocessor(ButterworthFilter(FilterConfig(fs=cfg.fs)))
        self.extractors = extractors
        self.model_manager = ModelManager(cfg.seed)

    def _segment(self, df: pd.DataFrame, window_size: int) -> List[pd.DataFrame]:
        """
        Segments the DataFrame into fixed-size windows.
        Window size = fs * window_size_sec (e.g., 32 Hz * 10 sec = 320 samples)
        """
        segments = []
        for i in range(0, len(df) - window_size + 1, window_size):
            segments.append(df.iloc[i : i + window_size].copy())
        return segments

    def load_and_process_files(self, directory: Path, label: int) -> pd.DataFrame:
        """
        Load all CSV files from a directory and extract features.
        
        Args:
            directory: Path to ON or OFF data folder
            label: 0 for ON state, 1 for OFF state
            
        Returns:
            DataFrame with extracted features and labels
        """
        all_features = []
        window_size = self.cfg.fs * self.cfg.window_size_sec  # 320 samples
        
        csv_files = list(directory.glob("*.csv"))
        log.info(f"Processing {len(csv_files)} files from {directory.name}...")
        
        for csv_file in csv_files:
            try:
                # Load decoded data
                data = pd.read_csv(csv_file)
                
                # Validate columns
                required_cols = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]
                if not all(col in data.columns for col in required_cols):
                    log.warning(f"Skipping {csv_file.name}: missing required columns")
                    continue
                
                # Drop rows with NaN
                data = data.dropna(subset=required_cols)
                
                if len(data) < window_size:
                    log.warning(f"Skipping {csv_file.name}: too short ({len(data)} samples)")
                    continue
                
                # Preprocessing: Apply Butterworth filter
                data_filtered = self.prep.process(data)
                
                # Segment into windows
                windows = self._segment(data_filtered, window_size)
                
                # Extract features from each window
                for win_idx, window in enumerate(windows):
                    features = {
                        "file": csv_file.name,
                        "window": win_idx,
                        "label": label
                    }
                    
                    # Separate accelerometer and gyroscope data
                    acc_data = window[["Ax", "Ay", "Az"]]
                    gyro_data = window[["Gx", "Gy", "Gz"]]
                    
                    # Extract statistical features
                    features.update(self.extractors[0].extract(acc_data, prefix="acc_"))
                    features.update(self.extractors[0].extract(gyro_data, prefix="gyro_"))
                    
                    # Extract gait features
                    features.update(self.extractors[1].extract(acc_data, gyro_data))
                    
                    all_features.append(features)
                    
            except Exception as e:
                log.error(f"Error processing {csv_file.name}: {e}")
                continue
        
        return pd.DataFrame(all_features)

    def run_data_extraction(self) -> pd.DataFrame:
        """
        Load and extract features from both ON and OFF directories.
        """
        log.info("Starting data extraction...")
        
        # Check directories exist
        if not self.cfg.on_data_dir.exists():
            raise FileNotFoundError(f"ON data directory not found: {self.cfg.on_data_dir}")
        if not self.cfg.off_data_dir.exists():
            raise FileNotFoundError(f"OFF data directory not found: {self.cfg.off_data_dir}")
        
        # Load ON state data (label=0)
        on_data = self.load_and_process_files(self.cfg.on_data_dir, label=0)
        log.info(f"Extracted {len(on_data)} windows from ON state data")
        
        # Load OFF state data (label=1)
        off_data = self.load_and_process_files(self.cfg.off_data_dir, label=1)
        log.info(f"Extracted {len(off_data)} windows from OFF state data")
        
        # Combine datasets
        all_data = pd.concat([on_data, off_data], ignore_index=True)
        
        log.info(f"Total windows extracted: {len(all_data)}")
        log.info(f"ON state windows: {len(on_data)}")
        log.info(f"OFF state windows: {len(off_data)}")
        
        return all_data

    def execute(self):
        """
        Standard execution flow for the entire pipeline.
        """
        log.info("="*60)
        log.info("Starting Random Forest Training Pipeline")
        log.info("="*60)
        
        # Extract features from data
        data = self.run_data_extraction()
        
        if data.empty:
            log.error("No valid data found. Please check your data directories.")
            return
        
        # Create output directory
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save extracted features
        data.to_csv(self.cfg.out_dir / "extracted_features.csv", index=False)
        log.info(f"Saved extracted features to {self.cfg.out_dir / 'extracted_features.csv'}")
        
        # Prepare data for training
        y = data["label"].values
        X = data.drop(["label", "file", "window"], axis=1).select_dtypes(include=[np.number])
        
        log.info(f"Total features: {len(X.columns)}")
        
        # Feature selection
        X_selected = self.model_manager.select_features(X, y, self.cfg.top_k_features)
        
        # Training & Evaluation
        results = self.model_manager.train_and_evaluate(X_selected, y, self.cfg)
        
        # Save evaluation report
        with open(self.cfg.out_dir / "evaluation_report.txt", "w") as f:
            f.write("="*60 + "\n")
            f.write("Random Forest - Parkinson's OFF State Detection\n")
            f.write("="*60 + "\n\n")
            f.write(f"Training Accuracy: {results['train_acc']:.4f}\n")
            f.write(f"Test Accuracy:     {results['test_acc']:.4f}\n")
            f.write(f"CV Mean Accuracy:  {results['cv_mean']:.4f} (+/- {results['cv_std']*2:.4f})\n")
            f.write(f"AUC Score:         {results['auc']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write("-"*60 + "\n")
            f.write(results["report"])
        
        log.info(f"Saved evaluation report to {self.cfg.out_dir / 'evaluation_report.txt'}")
        
        # Generate visualizations
        if not self.cfg.fast_mode:
            self._save_plots(results)
        
        log.info("="*60)
        log.info(f"Pipeline complete! Results saved in {self.cfg.out_dir}")
        log.info("="*60)

    def _save_plots(self, results):
        """Generates and saves performance visualization plots."""
        log.info("Generating visualization plots...")
        
        # 1. Feature Importance Plot
        plt.figure(figsize=(10, 8))
        importances = self.model_manager.model.feature_importances_
        features = self.model_manager.selected_features
        
        sns.barplot(
            x=importances, 
            y=features,
            hue=features,
            palette="viridis",
            legend=False
        )
        plt.title("Top Feature Importances", fontsize=16, fontweight='bold')
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()
        plt.savefig(self.cfg.out_dir / "feature_importance.png", dpi=300)
        plt.close()
        
        # 2. Confusion Matrix
        cm = confusion_matrix(results["test_y"], results["preds"])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=["ON", "OFF"], 
                    yticklabels=["ON", "OFF"])
        plt.title("Confusion Matrix", fontsize=16, fontweight='bold')
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.tight_layout()
        plt.savefig(self.cfg.out_dir / "confusion_matrix.png", dpi=300)
        plt.close()
        
        # 3. ROC Curve
        if results["probs"] is not None:
            fpr, tpr, _ = roc_curve(results["test_y"], results["probs"])
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {results["auc"]:.4f}')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
            plt.xlabel("False Positive Rate", fontsize=12)
            plt.ylabel("True Positive Rate", fontsize=12)
            plt.title("ROC Curve", fontsize=16, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.cfg.out_dir / "roc_curve.png", dpi=300)
            plt.close()
        
        log.info(f"Saved plots to {self.cfg.out_dir}")

# --- Entry Point ---
def main():
    """CLI entry point for the Parkinson's Detection Pipeline."""
    parser = argparse.ArgumentParser(
        description="Train Random Forest classifier for Parkinson's OFF state detection"
    )
    parser.add_argument(
        "--data_on", 
        default="Decode_Dataset/DecodeDataset_Train_Test/ON",
        help="Path to ON state data directory"
    )
    parser.add_argument(
        "--data_off", 
        default="data/Decode_Dataset/DecodeDataset_Train_Test/OFF",
        help="Path to OFF state data directory"
    )
    parser.add_argument(
        "--out", 
        default="Decode_Dataset/DecodeDataset_Train_Test/outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--fs", 
        type=int, 
        default=32,
        help="Sampling frequency (Hz)"
    )
    parser.add_argument(
        "--fast", 
        action="store_true",
        help="Skip visualization plots for faster execution"
    )
    parser.add_argument(
        "--window_sec", 
        type=int, 
        default=10,
        help="Window size in seconds for feature extraction"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=15,
        help="Number of top features to select"
    )
    
    args = parser.parse_args()

    # Create configuration
    cfg = PipelineConfig(
        on_data_dir=Path(args.data_on),
        off_data_dir=Path(args.data_off),
        out_dir=Path(args.out),
        fs=args.fs,
        fast_mode=args.fast,
        window_size_sec=args.window_sec,
        top_k_features=args.top_k
    )
    
    # Inject dependencies
    extractors = [StatExtractor(), GaitExtractor(cfg.fs)]
    pipeline = Pipeline(cfg, extractors)
    
    # Run pipeline
    try:
        pipeline.execute()
    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()