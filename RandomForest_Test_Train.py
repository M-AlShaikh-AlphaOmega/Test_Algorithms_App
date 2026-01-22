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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# --- Log Management (SRP) ---
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

# --- Configuration (Clean Code / Data Classes) ---
@dataclass
class FilterConfig:
    """Configuration for signal filtering."""
    cutoff: float = 15.0
    fs: int = 32
    order: int = 4

@dataclass
class PipelineConfig:
    """Global configuration for the pipeline execution."""
    data_dir: Path = Path("Data")
    out_dir: Path = Path("outputs")
    task: str = "walk_hold_left"
    fs: int = 32
    fast_mode: bool = False
    test_size: float = 0.3
    seed: int = 42
    top_k_features: int = 15

# --- Signal Processing (SRP / Strategy) ---
class SignalFilter(Protocol):
    """Protocol defining the interface for signal filtering strategies."""
    def apply(self, signal: np.ndarray) -> np.ndarray: ...

class ButterworthFilter:
    """Implements a Butterworth low-pass filter strategy."""
    def __init__(self, config: FilterConfig):
        self.config = config
        self._coeffs = None

    def _get_coeffs(self):
        if self._coeffs is None:
            nyq = 0.5 * self.config.fs
            low = self.config.cutoff / nyq
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
        """Processes the input DataFrame by applying filters to each axis."""
        df_f = df.copy()
        for ax in ["X", "Y", "Z"]:
            if ax in df_f.columns:
                df_f[ax] = self.filter.apply(df_f[ax].values)
        return df_f

# --- Feature Extraction (OCP / LSP / Strategy) ---
class BaseExtractor(Protocol):
    """Protocol defining the interface for feature extraction strategies."""
    def extract(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]: ...

class StatExtractor:
    """Extracts time-domain statistical features from sensor data."""
    def extract(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Calculates mean, std, rms, skew, kurtosis, etc., for all axes."""
        feats = {}
        for ax in ["X", "Y", "Z"]:
            if ax not in data.columns: continue
            s = data[ax].values
            pfx = f"{ax}_"
            feats.update({
                f"{pfx}mean": np.mean(s), f"{pfx}std": np.std(s), f"{pfx}min": np.min(s),
                f"{pfx}max": np.max(s), f"{pfx}range": np.ptp(s), f"{pfx}rms": np.sqrt(np.mean(s**2)),
                f"{pfx}skew": skew(s), f"{pfx}kurt": kurtosis(s), f"{pfx}iqr": iqr(s)
            })
        return feats

class GaitExtractor:
    """Extracts physical gait features from movement data."""
    def __init__(self, fs: int): self.fs = fs

    def extract(self, data: pd.DataFrame, gyro: pd.DataFrame = None, **kwargs) -> Dict[str, float]:
        """Detects steps, calculates cadence, and measures arm swing magnitude."""
        if gyro is None: return {}
        peaks, _ = find_peaks(data["Y"].values, prominence=0.5, distance=16)
        if len(peaks) < 2:
            return {f:0.0 for f in ["steps", "step_time", "cadence", "arm_swing"]}
        
        t = np.diff(peaks) / self.fs
        return {
            "steps": len(peaks), "step_time": np.mean(t),
            "cadence": len(peaks) / (len(data)/self.fs/60),
            "arm_swing": np.ptp(np.sqrt(gyro["X"]**2 + gyro["Y"]**2 + gyro["Z"]**2))
        }

# --- Model Management (SRP / DIP) ---
class ModelManager:
    """Manages the Machine Learning model lifecycle: Selection, Training, and Evaluation."""
    def __init__(self, seed: int = 42):
        # Best parameters found during optimization to prevent overfitting
        self.model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=7, 
            min_samples_leaf=2, 
            max_features='sqrt',
            random_state=seed, 
            n_jobs=-1
        )
        self.selected_features = []

    def select_features(self, X: pd.DataFrame, y: np.ndarray, k: int) -> pd.DataFrame:
        """Selects the top k most important features using a preliminary forest."""
        log.info(f"Selecting top {k} features...")
        selector = RandomForestClassifier(n_estimators=100, random_state=42)
        selector.fit(X, y)
        importances = selector.feature_importances_
        indices = np.argsort(importances)[::-1][:k]
        self.selected_features = list(X.columns[indices])
        log.info(f"Selected: {self.selected_features}")
        return X[self.selected_features]

    def train_and_evaluate(self, X: pd.DataFrame, y: np.ndarray, config: PipelineConfig) -> Dict:
        """Trains the model and performs Cross-Validation and Test Set evaluation."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.seed, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        
        # Diagnostics
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.seed)
        cv_scores = cross_val_score(self.model, X, y, cv=cv)
        
        log.info(f"Training Accuracy: {train_acc:.4f}")
        log.info(f"Test Accuracy:     {test_acc:.4f}")
        log.info(f"CV Mean Accuracy:  {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return {
            "test_y": y_test,
            "preds": self.model.predict(X_test),
            "probs": self.model.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None,
            "report": classification_report(y_test, self.model.predict(X_test), target_names=["ON", "OFF"]),
            "cv_mean": cv_scores.mean()
        }

# --- Core Pipeline Orchestrator (Composition) ---
class Pipeline:
    """Orchestrates the end-to-end data processing and classification flow."""
    def __init__(self, cfg: PipelineConfig, extractors: List[BaseExtractor]):
        self.cfg = cfg
        self.prep = Preprocessor(ButterworthFilter(FilterConfig(fs=cfg.fs)))
        self.extractors = extractors
        self.model_manager = ModelManager(cfg.seed)

    def _segment(self, df: pd.DataFrame, size: int = 320) -> List[pd.DataFrame]:
        """Segments the DataFrame into fixed-size windows."""
        return [df.iloc[i : i + size].copy() for i in range(0, len(df) - size + 1, size)]

    def run_data_extraction(self) -> pd.DataFrame:
        """Discovers data files and extracts features across all patients."""
        all_frames = []
        for case_dir in self.cfg.data_dir.iterdir():
            if not case_dir.is_dir(): continue
            
            acc_files = list(case_dir.glob(f"{self.cfg.task}_acc_*.csv"))
            gyr_files = list(case_dir.glob(f"{self.cfg.task}_gyro_*.csv"))
            if not acc_files or not gyr_files: continue

            # Loading & Preprocessing
            acc = self.prep.process(pd.read_csv(acc_files[0]).dropna(subset=["X", "Y", "Z"]))
            gyr = self.prep.process(pd.read_csv(gyr_files[0]).dropna(subset=["X", "Y", "Z"]))
            
            # Windowing
            a_windows = self._segment(acc)
            g_windows = self._segment(gyr)
            
            case_rows = []
            for i in range(min(len(a_windows), len(g_windows))):
                features = {"win_id": i}
                # Apply extraction strategy
                features.update(self.extractors[0].extract(a_windows[i])) # Acc Stats
                g_feats = self.extractors[0].extract(g_windows[i]) # Gyro Stats
                features.update({f"g_{k}": v for k, v in g_feats.items()})
                features.update(self.extractors[1].extract(a_windows[i], gyro=g_windows[i])) # Gait
                case_rows.append(features)
            
            df_case = pd.DataFrame(case_rows)
            # Label Inference from folder naming convention
            sfx = case_dir.name.split(' ')[-1].lower()
            if len(sfx) == 4 and sfx[1] in 'ny':
                df_case["label"] = 1 if sfx[1] == 'n' else 0
                all_frames.append(df_case)

        return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

    def execute(self):
        """Standard execution flow for the entire pipeline."""
        log.info(f"Starting pipeline for task: {self.cfg.task}")
        data = self.run_data_extraction()
        if data.empty: return log.error("No valid data found.")
        
        self.cfg.out_dir.mkdir(exist_ok=True)
        data.to_csv(self.cfg.out_dir / "extracted_features.csv", index=False)
        
        y = data["label"].values
        X = data.drop(["label", "win_id"], axis=1).select_dtypes(include=[np.number])
        
        # Optimization: Feature Selection
        X_selected = self.model_manager.select_features(X, y, self.cfg.top_k_features)
        
        # Training & Evaluation
        results = self.model_manager.train_and_evaluate(X_selected, y, self.cfg)
        
        # Save Report
        with open(self.cfg.out_dir / "evaluation_report.txt", "w") as f:
            f.write(f"Task: {self.cfg.task}\nCV Mean Accuracy: {results['cv_mean']:.4f}\n\n")
            f.write(results["report"])
        
        # Visuals
        if not self.cfg.fast_mode:
            self._save_plots(results)
        
        log.info(f"Pipeline complete. Results in {self.cfg.out_dir}")

    def _save_plots(self, results):
        """Generates and saves performance visualization plots."""
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=self.model_manager.model.feature_importances_, 
            y=self.model_manager.selected_features, 
            hue=self.model_manager.selected_features,
            palette="viridis",
            legend=False
        )
        plt.title("Top Feature Importances")
        plt.tight_layout()
        plt.savefig(self.cfg.out_dir / "feature_importance.png")
        
        cm = confusion_matrix(results["test_y"], results["preds"])
        plt.figure(); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues'); plt.savefig(self.cfg.out_dir / "confusion_matrix.png")
        
        if results["probs"] is not None:
            fpr, tpr, _ = roc_curve(results["test_y"], results["probs"])
            plt.figure(); plt.plot(fpr, tpr); plt.title("ROC Curve"); plt.savefig(self.cfg.out_dir / "roc_curve.png")

# --- Entry Point ---
def main():
    """CLI entry point for the Parkinson's Detection Pipeline."""
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="Data")
    p.add_argument("--out", default="outputs")
    p.add_argument("--task", default="walk_hold_left")
    p.add_argument("--fs", type=int, default=32)
    p.add_argument("--fast", action="store_true")
    args = p.parse_args()

    cfg = PipelineConfig(Path(args.data), Path(args.out), args.task, args.fs, args.fast)
    
    # Inject dependencies (DIP)
    extractors = [StatExtractor(), GaitExtractor(cfg.fs)]
    pipeline = Pipeline(cfg, extractors)
    
    pipeline.execute()

if __name__ == "__main__":
    main()
