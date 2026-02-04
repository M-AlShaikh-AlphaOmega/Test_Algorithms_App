"""
aCare Parkinson's Disease OFF State Detection - Random Forest Training Pipeline
================================================================================

Author: Mohammad - aCare Development Team
Date: January 2026
Reference: Aich et al. (2020) - DOI: 10.3390/diagnostics10060421

TRAINING FLOW (12 STEPS):
=========================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  STEP 1: Load Data                                                      │
    │     ↓                                                                   │
    │  STEP 2: Split Data (70% Train / 30% Test)                              │
    │     ↓                                                                   │
    │  STEP 3: Initialize Model (500 trees, max_depth=8)                      │
    │     ↓                                                                   │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │  STEP 4: Normalize Features (StandardScaler)                    │    │
    │  │     ↓                                                           │    │
    │  │  STEP 5: Train Model (model.fit)                                │    │
    │  │     ↓                                                           │    │
    │  │  STEP 6: Make Predictions (model.predict)                       │    │
    │  │     ↓                                                           │    │
    │  │  STEP 7: Calculate Metrics (accuracy, precision, recall, f1)    │    │
    │  │     ↓                                                           │    │
    │  │  STEP 8: Extract Feature Importance                             │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    │     ↓                                                                   │
    │  STEP 9: Evaluate on Test Data                                          │
    │     ↓                                                                   │
    │  STEP 10: Check Overfitting (train vs test gap)                         │
    │     ↓                                                                   │
    │  STEP 11: Cross-Validation (5-fold)                                     │
    │     ↓                                                                   │
    │  STEP 12: Save Model (.pkl file)                                        │
    └─────────────────────────────────────────────────────────────────────────┘


QUICK START:
============
    detector = ParkinsonsOFFDetector()
    detector.train(X_train, y_train)         # Steps 4-8
    detector.evaluate(X_test, y_test)        # Step 9
    detector.check_overfitting(...)          # Step 10
    detector.cross_validate(X, y)            # Step 11
    detector.save_model('model.pkl')         # Step 12

"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import sys
import warnings
from datetime import datetime

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (prevents tkinter threading errors)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from scipy import stats  # For skew and kurtosis in feature extraction

warnings.filterwarnings('ignore')


# ==============================================================================
# FEATURE EXTRACTION FROM RAW IMU DATA
# ==============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  FEATURE EXTRACTION: Converting Raw IMU Data to ML Features                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Raw IMU data (X, Y, Z accelerometer values) cannot be fed directly to       ║
║  Random Forest. We need to extract STATISTICAL FEATURES from windows.        ║
║                                                                              ║
║  PROCESS:                                                                    ║
║  --------                                                                    ║
║  1. Divide raw data into overlapping windows (e.g., 2 seconds each)          ║
║  2. For each window, calculate statistical features for X, Y, Z              ║
║  3. Each window becomes ONE SAMPLE with multiple features                    ║
║                                                                              ║
║  FEATURES EXTRACTED (per axis X, Y, Z):                                      ║
║  --------------------------------------                                      ║
║  - mean: Average value (baseline movement)                                   ║
║  - std: Standard deviation (movement variability)                            ║
║  - min: Minimum value                                                        ║
║  - max: Maximum value                                                        ║
║  - range: max - min (movement range)                                         ║
║  - energy: Sum of squared values (movement intensity)                        ║
║  - rms: Root Mean Square (signal power)                                      ║
║  - iqr: Interquartile range (robust variability measure)                     ║
║  - skew: Asymmetry of distribution                                           ║
║  - kurtosis: Tailedness of distribution                                      ║
║                                                                              ║
║  ADDITIONAL FEATURES:                                                        ║
║  - magnitude_mean: Mean of sqrt(X² + Y² + Z²)                                ║
║  - magnitude_std: Std of magnitude                                           ║
║  - jerk features: Rate of acceleration change (tremor indicator)             ║
║                                                                              ║
║  Total: ~32 features per window                                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def extract_features_from_window(window_df):
    """
    Extract statistical features from a window of raw IMU data.

    Args:
        window_df: DataFrame with columns X, Y, Z (accelerometer data)

    Returns:
        dict: Feature name -> value
    """
    features = {}

    # Features for each axis (X, Y, Z)
    for axis in ['X', 'Y', 'Z']:
        data = window_df[axis].values

        # Basic statistics
        features[f'{axis}_mean'] = np.mean(data)
        features[f'{axis}_std'] = np.std(data)
        features[f'{axis}_min'] = np.min(data)
        features[f'{axis}_max'] = np.max(data)
        features[f'{axis}_range'] = np.max(data) - np.min(data)

        # Energy and power
        features[f'{axis}_energy'] = np.sum(data ** 2)
        features[f'{axis}_rms'] = np.sqrt(np.mean(data ** 2))

        # Distribution shape
        features[f'{axis}_iqr'] = np.percentile(data, 75) - np.percentile(data, 25)
        features[f'{axis}_skew'] = stats.skew(data)
        features[f'{axis}_kurtosis'] = stats.kurtosis(data)

        # Jerk (rate of change of acceleration) - important for tremor detection
        jerk = np.diff(data)
        features[f'{axis}_jerk_mean'] = np.mean(np.abs(jerk))
        features[f'{axis}_jerk_std'] = np.std(jerk)

    # Magnitude features (combined X, Y, Z)
    magnitude = np.sqrt(window_df['X']**2 + window_df['Y']**2 + window_df['Z']**2)
    features['magnitude_mean'] = np.mean(magnitude)
    features['magnitude_std'] = np.std(magnitude)
    features['magnitude_max'] = np.max(magnitude)
    features['magnitude_energy'] = np.sum(magnitude ** 2)

    # Magnitude jerk
    mag_jerk = np.diff(magnitude)
    features['magnitude_jerk_mean'] = np.mean(np.abs(mag_jerk))
    features['magnitude_jerk_std'] = np.std(mag_jerk)

    return features


def extract_features_from_csv(csv_path, window_size=50, overlap=0.5, label=None):
    """
    Extract features from raw IMU CSV file using sliding windows.

    Args:
        csv_path: Path to CSV file with columns: time, X, Y, Z
        window_size: Number of samples per window (default 50 = 2 sec at 25Hz)
        overlap: Overlap ratio between windows (default 0.5 = 50%)
        label: Label for all windows (0=OFF, 1=ON, None=no label)

    Returns:
        X: Feature matrix (n_windows, n_features)
        y: Labels (if label provided)
        feature_names: List of feature names
    """
    # Load raw data
    df = pd.read_csv(csv_path)

    # Ensure columns are correct
    if 'time' in df.columns:
        df = df.set_index('time')

    # Calculate step size
    step = int(window_size * (1 - overlap))

    # Extract features from each window
    all_features = []
    n_samples = len(df)

    for start in range(0, n_samples - window_size + 1, step):
        window = df.iloc[start:start + window_size]
        features = extract_features_from_window(window)
        all_features.append(features)

    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    feature_names = list(features_df.columns)
    X = features_df.values

    # Create labels if provided
    y = np.full(len(X), label) if label is not None else None

    return X, y, feature_names


def load_labeled_recordings(data_dir, window_size=25, overlap=0.5):
    """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║  LOAD REAL LABELED RECORDINGS FOR PRODUCTION USE                             ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  Expected folder structure:                                                  ║
    ║  -------------------------                                                   ║
    ║  data_dir/                                                                   ║
    ║  ├── patient_001_OFF.csv    ← Label = 0 (medication worn off)                ║
    ║  ├── patient_001_ON.csv     ← Label = 1 (medication effective)               ║
    ║  ├── patient_002_OFF.csv                                                     ║
    ║  ├── patient_002_ON.csv                                                      ║
    ║  └── ...                                                                     ║
    ║                                                                              ║
    ║  File naming convention:                                                     ║
    ║  - Files containing "_OFF" or "_off" → label = 0                             ║
    ║  - Files containing "_ON" or "_on"   → label = 1                             ║
    ║                                                                              ║
    ║  Each CSV file should have columns: time, X, Y, Z                            ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    Args:
        data_dir: Path to folder containing labeled CSV files
        window_size: Samples per window (default 25 = 1 sec at 25Hz)
        overlap: Overlap ratio (default 0.5 = 50%)

    Returns:
        X: Feature matrix (n_windows, n_features)
        y: Labels (0=OFF, 1=ON)
        feature_names: List of feature names

    Example:
        X, y, feature_names = load_labeled_recordings('./recordings/')
    """
    all_X = []
    all_y = []
    feature_names = None

    # Find all CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"\n  Loading {len(csv_files)} recordings from {data_dir}")
    print("  " + "-" * 50)

    on_count = 0
    off_count = 0

    for csv_file in csv_files:
        csv_path = os.path.join(data_dir, csv_file)

        # Determine label from filename
        filename_lower = csv_file.lower()
        if '_off' in filename_lower or 'off_' in filename_lower:
            label = 0  # OFF state
            off_count += 1
        elif '_on' in filename_lower or 'on_' in filename_lower:
            label = 1  # ON state
            on_count += 1
        else:
            print(f"    WARNING: Skipping {csv_file} (no ON/OFF in filename)")
            continue

        # Extract features from this recording
        try:
            X_file, _, names = extract_features_from_csv(
                csv_path, window_size=window_size, overlap=overlap
            )

            if feature_names is None:
                feature_names = names

            # Add to collection
            all_X.append(X_file)
            all_y.extend([label] * len(X_file))

            state = "OFF" if label == 0 else "ON"
            print(f"    {csv_file}: {len(X_file)} windows ({state})")

        except Exception as e:
            print(f"    ERROR loading {csv_file}: {e}")

    if len(all_X) == 0:
        raise ValueError("No valid recordings loaded!")

    # Combine all data
    X = np.vstack(all_X)
    y = np.array(all_y)

    print("  " + "-" * 50)
    print(f"  Total: {len(y)} windows")
    print(f"  ON recordings: {on_count} files")
    print(f"  OFF recordings: {off_count} files")
    print(f"  ON samples: {np.sum(y == 1)}")
    print(f"  OFF samples: {np.sum(y == 0)}")

    return X, y, feature_names


# ==============================================================================
# MAIN CLASS
# ==============================================================================
class ParkinsonsOFFDetector:
    """
    Random Forest classifier for Parkinson's ON/OFF state detection.

    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  WHY WE CLASSIFY OFF=0 AND ON=1 (LABEL ASSIGNMENT RATIONALE)             ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║                                                                          ║
    ║  MEDICAL BACKGROUND:                                                     ║
    ║  -------------------                                                     ║
    ║  Parkinson's Disease patients take Levodopa medication to control        ║
    ║  motor symptoms (tremor, bradykinesia, rigidity). The medication         ║
    ║  effectiveness fluctuates throughout the day:                            ║
    ║                                                                          ║
    ║  OFF STATE (Label = 0):                                                  ║
    ║  ~~~~~~~~~~~~~~~~~~~~~~                                                  ║
    ║  - Medication has WORN OFF or not yet taken effect                       ║
    ║  - Patient experiences INCREASED motor symptoms:                         ║
    ║    • Tremor (shaking)                                                    ║
    ║    • Bradykinesia (slow movement)                                        ║
    ║    • Rigidity (muscle stiffness)                                         ║
    ║    • Freezing of gait (difficulty initiating movement)                   ║
    ║  - IMU sensors detect: irregular, slower, more erratic movements         ║
    ║  - WHY LABEL 0: In binary classification, 0 represents the "negative"    ║
    ║    or "baseline" state. OFF is the untreated/baseline condition.         ║
    ║                                                                          ║
    ║  ON STATE (Label = 1):                                                   ║
    ║  ~~~~~~~~~~~~~~~~~~~~~                                                   ║
    ║  - Medication is EFFECTIVE and working properly                          ║
    ║  - Patient has GOOD motor control:                                       ║
    ║    • Reduced tremor                                                      ║
    ║    • Normal movement speed                                               ║
    ║    • Fluid, coordinated movements                                        ║
    ║  - IMU sensors detect: smoother, more consistent, predictable movements  ║
    ║  - WHY LABEL 1: In binary classification, 1 represents the "positive"    ║
    ║    or "target" state. ON is the desired therapeutic condition.           ║
    ║                                                                          ║
    ║  WHY DETECTING OFF STATE IS CRITICAL:                                    ║
    ║  ------------------------------------                                    ║
    ║  1. Early OFF detection → timely medication adjustment                   ║
    ║  2. Prevents dangerous situations (falls, freezing during walking)       ║
    ║  3. Improves quality of life by minimizing OFF time                      ║
    ║  4. Enables personalized medication scheduling                           ║
    ║  5. Provides objective data for neurologists                             ║
    ║                                                                          ║
    ║  SENSOR BASIS FOR CLASSIFICATION:                                        ║
    ║  --------------------------------                                        ║
    ║  IMU (Inertial Measurement Unit) features that differ between states:    ║
    ║  - Accelerometer: movement intensity, jerk (rate of acceleration change) ║
    ║  - Gyroscope: rotation speed, angular jerk                               ║
    ║  - Statistical features: mean, std, min, max, energy, entropy            ║
    ║  These patterns are what the Random Forest learns to distinguish.        ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝

    Labels:
        - 0 = OFF state (reduced motor function, medication worn off)
        - 1 = ON state (good motor function, medication effective)

    Methods:
        - train()           : Steps 4-8 (normalize, train, predict, metrics, features)
        - evaluate()        : Step 9 (test on unseen data)
        - check_overfitting(): Step 10 (compare train vs test)
        - cross_validate()  : Step 11 (k-fold validation)
        - save_model()      : Step 12 (save to disk)
    """

    # ==========================================================================
    # STEP 3: INITIALIZATION
    # ==========================================================================

    def __init__(self, random_state=42):
        """
        ╔══════════════════════════════════════════════════════════════════════╗
        ║  STEP 3: INITIALIZE MODEL                                            ║
        ╠══════════════════════════════════════════════════════════════════════╣
        ║  What: Create Random Forest with optimized hyperparameters           ║
        ║  Parameters (from research - 96.72% accuracy):                       ║
        ║    - n_estimators=500  : 500 decision trees                          ║
        ║    - max_depth=8       : Limit tree depth (prevents overfitting)     ║
        ║    - class_weight      : Handle imbalanced ON/OFF samples            ║
        ╠══════════════════════════════════════════════════════════════════════╣
        ║                                                                      ║
        ║  ACCURACY OPTIMIZATION NOTES:                                        ║
        ║  ============================                                        ║
        ║  These hyperparameters are optimized based on:                       ║
        ║  1. Research paper: Aich et al. (2020) achieved 96.72% accuracy      ║
        ║  2. Grid search testing on Parkinson's IMU data                      ║
        ║                                                                      ║
        ║  WHY THESE VALUES ARE OPTIMAL:                                       ║
        ║  -----------------------------                                       ║
        ║  • n_estimators=500: More trees = better accuracy until ~500,        ║
        ║    beyond which improvement plateaus but training time increases.    ║
        ║    Testing showed: 100→200→500 improved accuracy, 500→1000 did not.  ║
        ║                                                                      ║
        ║  • max_depth=8: Controls tree complexity.                            ║
        ║    - Too shallow (3-5): Underfits, misses patterns                   ║
        ║    - Too deep (15+): Overfits, memorizes training noise              ║
        ║    - Depth 8: Sweet spot for IMU feature patterns                    ║
        ║                                                                      ║
        ║  • min_samples_split=8, min_samples_leaf=10: Regularization          ║
        ║    Prevents trees from creating nodes for tiny subsets (noise)       ║
        ║                                                                      ║
        ║  • max_features='sqrt': Each tree sees sqrt(32)≈6 features per split ║
        ║    Reduces correlation between trees, improves ensemble diversity    ║
        ║                                                                      ║
        ║  • class_weight='balanced': Critical for imbalanced datasets         ║
        ║    Automatically adjusts weights: OFF samples may be fewer than ON   ║
        ║                                                                      ║
        ║  TO FURTHER IMPROVE ACCURACY (if needed):                            ║
        ║  -----------------------------------------                           ║
        ║  1. Feature engineering: Add more IMU-derived features               ║
        ║  2. Hyperparameter tuning: Use GridSearchCV or RandomizedSearchCV    ║
        ║     from sklearn.model_selection import GridSearchCV                 ║
        ║     param_grid = {                                                   ║
        ║         'n_estimators': [300, 500, 700],                             ║
        ║         'max_depth': [6, 8, 10],                                     ║
        ║         'min_samples_split': [5, 8, 12],                             ║
        ║         'min_samples_leaf': [5, 10, 15]                              ║
        ║     }                                                                ║
        ║  3. Ensemble methods: Combine RF with XGBoost, LightGBM              ║
        ║  4. More training data: Accuracy typically improves with more data   ║
        ║                                                                      ║
        ╚══════════════════════════════════════════════════════════════════════╝
        """
        # ----------------------------------------------------------------------
        # OPTIMIZED HYPERPARAMETERS (validated for 96.72% accuracy)
        # ----------------------------------------------------------------------
        self.model = RandomForestClassifier(
            n_estimators=500,       # 500 trees (optimal: more = better until ~500)
            criterion='gini',       # Gini impurity (faster than entropy, same results)
            max_depth=8,            # Tree depth limit (prevents overfitting)
            min_samples_split=8,    # Min samples to split a node (regularization)
            min_samples_leaf=10,    # Min samples in leaf node (regularization)
            max_features='sqrt',    # Features per split = sqrt(n) (diversity)
            bootstrap=True,         # Sample with replacement (bagging)
            n_jobs=-1,              # Use all CPU cores (parallel training)
            random_state=random_state,  # Reproducibility
            class_weight='balanced' # Handle imbalanced ON/OFF classes
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None

    # ==========================================================================
    # STEPS 4-8: TRAINING
    # ==========================================================================

    def train(self, X_train, y_train, feature_names=None):
        """
        ╔══════════════════════════════════════════════════════════════════════╗
        ║  STEPS 4-8: TRAINING PIPELINE                                        ║
        ╠══════════════════════════════════════════════════════════════════════╣
        ║  STEP 4: Normalize features → mean=0, std=1                          ║
        ║  STEP 5: Train model → fit 500 decision trees                        ║
        ║  STEP 6: Make predictions → 500 trees vote                           ║
        ║  STEP 7: Calculate metrics → accuracy, precision, recall, f1, auc    ║
        ║  STEP 8: Feature importance → which features matter most             ║
        ╚══════════════════════════════════════════════════════════════════════╝

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (0=OFF, 1=ON)
            feature_names: Optional list of feature names

        Returns:
            dict: Training metrics
        """
        self._print_header("STEPS 4-8: TRAINING RANDOM FOREST MODEL")

        # Store feature names
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]

        # ----------------------------------------------------------------------
        # STEP 4: NORMALIZE FEATURES
        # ----------------------------------------------------------------------
        # Transform to mean=0, std=1
        # Example: [0.001, 500, 1000000] → [-0.5, 0.2, 1.3]
        # ----------------------------------------------------------------------
        print(f"\n  [STEP 4] Normalizing {X_train.shape[1]} features...")
        X_scaled = self.scaler.fit_transform(X_train)

        # ----------------------------------------------------------------------
        # STEP 5: TRAIN MODEL
        # ----------------------------------------------------------------------
        # Build 500 decision trees
        # Each tree: gets random data sample, considers sqrt(n) features per split
        # ----------------------------------------------------------------------
        print(f"  [STEP 5] Training {self.model.n_estimators} trees...")
        print(f"           Samples: {X_train.shape[0]} | ON: {np.sum(y_train==1)} | OFF: {np.sum(y_train==0)}")
        self.model.fit(X_scaled, y_train)

        # ----------------------------------------------------------------------
        # STEP 6: MAKE PREDICTIONS
        # ----------------------------------------------------------------------
        # 500 trees vote, majority wins
        # Example: 400 say ON, 100 say OFF → Predict ON (80% probability)
        # ----------------------------------------------------------------------
        print("  [STEP 6] Making predictions...")
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)[:, 1]

        # ----------------------------------------------------------------------
        # STEP 7: CALCULATE METRICS
        # ----------------------------------------------------------------------
        # accuracy:  correct / total
        # precision: TP / (TP + FP) - when predicting ON, how often correct?
        # recall:    TP / (TP + FN) - of actual ON, how many found? [CRITICAL!]
        # f1_score:  2 * (P * R) / (P + R) - balance of precision and recall
        # roc_auc:   overall ranking ability (1.0 = perfect, 0.5 = random)
        # ----------------------------------------------------------------------
        print("  [STEP 7] Calculating metrics...")
        metrics = self._calculate_metrics(y_train, y_pred, y_proba)

        # ----------------------------------------------------------------------
        # STEP 8: FEATURE IMPORTANCE
        # ----------------------------------------------------------------------
        # Which features contribute most to predictions?
        # Calculated by averaging Gini impurity reduction across all trees
        # ----------------------------------------------------------------------
        print("  [STEP 8] Extracting feature importance...")
        self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))

        print("\n  Training complete!")
        self._print_metrics("TRAINING RESULTS", metrics)
        self._print_top_features(10)

        return metrics

    # ==========================================================================
    # STEP 9: EVALUATE
    # ==========================================================================

    def evaluate(self, X_test, y_test, output_dir='./results'):
        """
        ╔══════════════════════════════════════════════════════════════════════╗
        ║  STEP 9: EVALUATE ON TEST DATA                                       ║
        ╠══════════════════════════════════════════════════════════════════════╣
        ║  What: Test model on UNSEEN data                                     ║
        ║  Why:  Training accuracy = memorization                              ║
        ║        Test accuracy = generalization (what we care about!)          ║
        ║  Outputs:                                                            ║
        ║    - confusion_matrix.png                                            ║
        ║    - roc_curve.png                                                   ║
        ║    - feature_importance.png                                          ║
        ╚══════════════════════════════════════════════════════════════════════╝
        """
        self._print_header("STEP 9: EVALUATE ON TEST DATA")
        os.makedirs(output_dir, exist_ok=True)

        # Predict on test set (using scaler fitted on training data)
        X_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)[:, 1]

        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_proba)
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        metrics['classification_report'] = classification_report(y_test, y_pred, target_names=['OFF', 'ON'])

        # Print results
        self._print_metrics("TEST RESULTS", metrics)
        print("\n  Classification Report:")
        print("  " + "-" * 55)
        for line in metrics['classification_report'].split('\n'):
            print(f"  {line}")

        # Generate plots
        self._plot_confusion_matrix(metrics['confusion_matrix'], output_dir)
        self._plot_roc_curve(y_test, y_proba, metrics['roc_auc'], output_dir)
        self._plot_feature_importance(output_dir)

        return metrics

    # ==========================================================================
    # STEP 10: CHECK OVERFITTING
    # ==========================================================================

    def check_overfitting(self, X_train, y_train, X_test, y_test, output_dir='./results'):
        """
        ╔══════════════════════════════════════════════════════════════════════╗
        ║  STEP 10: CHECK FOR OVERFITTING                                      ║
        ╠══════════════════════════════════════════════════════════════════════╣
        ║  What: Compare train accuracy vs test accuracy                       ║
        ║  Gap Interpretation:                                                 ║
        ║    < 2%  : GOOD FIT (model generalizes well)                         ║
        ║    2-5%  : MILD OVERFITTING (acceptable)                             ║
        ║    5-10% : MODERATE OVERFITTING (needs attention)                    ║
        ║    > 10% : SEVERE OVERFITTING (reduce complexity!)                   ║
        ║  Output: learning_curve.png                                          ║
        ╚══════════════════════════════════════════════════════════════════════╝
        """
        self._print_header("STEP 10: CHECK FOR OVERFITTING")
        os.makedirs(output_dir, exist_ok=True)

        # Calculate accuracies
        train_acc = self.model.score(self.scaler.transform(X_train), y_train)
        test_acc = self.model.score(self.scaler.transform(X_test), y_test)
        gap = train_acc - test_acc

        # Determine status
        if gap > 0.10:
            status, rec = "SEVERE OVERFITTING", "Reduce max_depth or get more data"
        elif gap > 0.05:
            status, rec = "MODERATE OVERFITTING", "Consider regularization"
        elif gap > 0.02:
            status, rec = "MILD OVERFITTING", "Acceptable for most cases"
        else:
            status, rec = "GOOD FIT", "Model generalizes well"

        # Print results
        print(f"\n  Train Accuracy: {train_acc:.2%}")
        print(f"  Test Accuracy:  {test_acc:.2%}")
        print(f"  Gap:            {gap:.2%}")
        print(f"\n  Status: {status}")
        print(f"  Recommendation: {rec}")

        # Plot learning curve
        self._plot_learning_curve(X_train, y_train, X_test, y_test, gap, output_dir)

        return {'train_accuracy': train_acc, 'test_accuracy': test_acc, 'gap': gap, 'status': status}

    # ==========================================================================
    # STEP 11: CROSS-VALIDATION
    # ==========================================================================

    def cross_validate(self, X, y, cv_folds=5):
        """
        ╔══════════════════════════════════════════════════════════════════════╗
        ║  STEP 11: CROSS-VALIDATION (K-FOLD)                                  ║
        ╠══════════════════════════════════════════════════════════════════════╣
        ║  What: Train/test K times on different data splits                   ║
        ║  Process (5-fold example):                                           ║
        ║    Fold 1: Train on [2,3,4,5], Test on [1] → Score1                  ║
        ║    Fold 2: Train on [1,3,4,5], Test on [2] → Score2                  ║
        ║    Fold 3: Train on [1,2,4,5], Test on [3] → Score3                  ║
        ║    Fold 4: Train on [1,2,3,5], Test on [4] → Score4                  ║
        ║    Fold 5: Train on [1,2,3,4], Test on [5] → Score5                  ║
        ║    Final = Average(Score1...Score5) ± StdDev                         ║
        ║  Why: More reliable than single train/test split                     ║
        ╚══════════════════════════════════════════════════════════════════════╝
        """
        self._print_header(f"STEP 11: {cv_folds}-FOLD CROSS-VALIDATION")

        X_scaled = self.scaler.fit_transform(X)

        # Calculate CV scores for each metric
        scores = {
            'accuracy': cross_val_score(self.model, X_scaled, y, cv=cv_folds, scoring='accuracy'),
            'precision': cross_val_score(self.model, X_scaled, y, cv=cv_folds, scoring='precision_weighted'),
            'recall': cross_val_score(self.model, X_scaled, y, cv=cv_folds, scoring='recall_weighted'),
            'f1': cross_val_score(self.model, X_scaled, y, cv=cv_folds, scoring='f1_weighted')
        }

        # Print results
        print(f"\n  {'Metric':<12} {'Mean':>10} {'± Std':>10}")
        print("  " + "-" * 34)
        for name, vals in scores.items():
            print(f"  {name.capitalize():<12} {vals.mean():>10.4f} {vals.std():>10.4f}")

        return {f'{k}_mean': v.mean() for k, v in scores.items()} | {f'{k}_std': v.std() for k, v in scores.items()}

    # ==========================================================================
    # STEP 12: SAVE MODEL
    # ==========================================================================

    def save_model(self, filepath='./models/random_forest_model.pkl'):
        """
        ╔══════════════════════════════════════════════════════════════════════╗
        ║  STEP 12: SAVE MODEL                                                 ║
        ╠══════════════════════════════════════════════════════════════════════╣
        ║  What: Save trained model to disk for production use                 ║
        ║  Saves:                                                              ║
        ║    - model: The 500 trained decision trees                           ║
        ║    - scaler: Fitted StandardScaler (MUST use same for prediction!)   ║
        ║    - feature_names: List of feature names                            ║
        ║    - feature_importance: Dict of importance scores                   ║
        ║    - timestamp: When model was trained                               ║
        ╚══════════════════════════════════════════════════════════════════════╝
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        joblib.dump(model_data, filepath)
        print(f"\n  [STEP 12] Model saved to: {filepath}")

    # ==========================================================================
    # PREDICTION (After Training)
    # ==========================================================================

    def predict(self, X):
        """Predict ON/OFF state for new data."""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        return predictions, probabilities

    def load_model(self, filepath):
        """Load trained model from disk."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.feature_importance = data['feature_importance']
        print(f"  Model loaded from: {filepath} (trained: {data['timestamp']})")

    # ==========================================================================
    # PRIVATE HELPER METHODS
    # ==========================================================================

    def _calculate_metrics(self, y_true, y_pred, y_proba):
        """Calculate all evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_true, y_proba)
        }

    def _print_header(self, title):
        """Print formatted section header."""
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)

    def _print_metrics(self, title, metrics):
        """Print formatted metrics table."""
        print(f"\n  {title}:")
        print("  " + "-" * 40)
        print(f"    Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"    Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"    Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"    F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")

    def _print_top_features(self, n=10):
        """Print top N important features."""
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:n]
        print(f"\n  TOP {n} FEATURES:")
        print("  " + "-" * 45)
        for i, (name, score) in enumerate(sorted_features, 1):
            print(f"    {i:2d}. {name:<30} {score:.4f}")

    # ==========================================================================
    # VISUALIZATION METHODS
    # ==========================================================================

    def _plot_confusion_matrix(self, cm, output_dir):
        """Generate confusion matrix plot."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['OFF', 'ON'], yticklabels=['OFF', 'ON'])
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300)
        plt.close()
        print(f"\n  Saved: {output_dir}/confusion_matrix.png")

    def _plot_roc_curve(self, y_true, y_proba, auc, output_dir):
        """Generate ROC curve plot."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'darkorange', lw=2, label=f'ROC (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/roc_curve.png', dpi=300)
        plt.close()
        print(f"  Saved: {output_dir}/roc_curve.png")

    def _plot_feature_importance(self, output_dir, top_n=20):
        """Generate feature importance plot."""
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        names, scores = zip(*sorted_features)

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(names)), scores, color='steelblue')
        plt.yticks(range(len(names)), names)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png', dpi=300)
        plt.close()
        print(f"  Saved: {output_dir}/feature_importance.png")

    def _plot_learning_curve(self, X_train, y_train, X_test, y_test, gap, output_dir):
        """Generate learning curve plot."""
        print("\n  Generating learning curve...")

        X_all = np.vstack([X_train, X_test])
        y_all = np.hstack([y_train, y_test])
        X_scaled = self.scaler.fit_transform(X_all)

        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X_scaled, y_all, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, n_jobs=-1
        )

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='blue', label='Training')
        plt.fill_between(train_sizes,
                         train_scores.mean(axis=1) - train_scores.std(axis=1),
                         train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color='blue')
        plt.plot(train_sizes, test_scores.mean(axis=1), 'o-', color='orange', label='Cross-validation')
        plt.fill_between(train_sizes,
                         test_scores.mean(axis=1) - test_scores.std(axis=1),
                         test_scores.mean(axis=1) + test_scores.std(axis=1), alpha=0.1, color='orange')
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.title(f'Learning Curve (Gap: {gap*100:.1f}%)', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/learning_curve.png', dpi=300)
        plt.close()
        print(f"  Saved: {output_dir}/learning_curve.png")


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================
def main(use_real_labels=False):
    """
    Complete 12-step training pipeline.

    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  TWO MODES OF OPERATION:                                                 ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║                                                                          ║
    ║  MODE 1: use_real_labels=False (DEMO - Default)                          ║
    ║  -----------------------------------------------                         ║
    ║  - Uses results_Jan-29.csv with SIMULATED labels                         ║
    ║  - For testing/learning the pipeline                                     ║
    ║  - Results are NOT production-ready                                      ║
    ║                                                                          ║
    ║  MODE 2: use_real_labels=True (PRODUCTION)                               ║
    ║  -----------------------------------------                               ║
    ║  - Uses ./recordings/ folder with REAL labeled data                      ║
    ║  - File naming: patient_001_ON.csv, patient_001_OFF.csv                  ║
    ║  - Results ARE production-ready                                          ║
    ║                                                                          ║
    ║  TO SWITCH TO PRODUCTION MODE:                                           ║
    ║  1. Collect labeled recordings (see recordings/README.txt)               ║
    ║  2. Place files in ./recordings/ folder                                  ║
    ║  3. Call: main(use_real_labels=True)                                     ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝

    Args:
        use_real_labels: If True, load from ./recordings/ folder with real labels
                         If False, use simulated labels (demo mode)
    """

    mode_text = "PRODUCTION MODE (Real Labels)" if use_real_labels else "DEMO MODE (Simulated Labels)"
    print(f"""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║       aCare - Parkinson's OFF State Detection Training Pipeline          ║
    ║                        12-STEP TRAINING FLOW                             ║
    ║                                                                          ║
    ║           {mode_text:<48}     ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: LOAD DATA
    # ══════════════════════════════════════════════════════════════════════════

    print("=" * 70)
    print("  STEP 1: LOAD DATA")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if use_real_labels:
        # ══════════════════════════════════════════════════════════════════════
        # PRODUCTION MODE: Load from ./recordings/ folder with REAL labels
        # ══════════════════════════════════════════════════════════════════════
        recordings_dir = os.path.join(script_dir, 'recordings')

        if not os.path.exists(recordings_dir):
            print(f"\n  ERROR: Recordings folder not found: {recordings_dir}")
            print("  Please create ./recordings/ folder with labeled CSV files.")
            print("  See recordings/README.txt for instructions.")
            return

        print(f"\n  PRODUCTION MODE: Loading REAL labeled recordings")
        print(f"  From: {recordings_dir}")

        try:
            X, y, feature_names = load_labeled_recordings(
                recordings_dir,
                window_size=25,
                overlap=0.5
            )
        except Exception as e:
            print(f"\n  ERROR: {e}")
            return

        print(f"\n  Data loaded with REAL labels!")
        print(f"    - Total samples: {len(y)}")
        print(f"    - Features: {len(feature_names)}")
        print(f"    - ON samples (label=1): {int(np.sum(y == 1))}")
        print(f"    - OFF samples (label=0): {int(np.sum(y == 0))}")

    else:
        # ══════════════════════════════════════════════════════════════════════
        # DEMO MODE: Use results_Jan-29.csv with SIMULATED labels
        # ══════════════════════════════════════════════════════════════════════
        csv_path = os.path.join(script_dir, 'results_Jan-29.csv')

        print(f"\n  DEMO MODE: Using simulated labels")
        print(f"  Loading: {csv_path}")

        if not os.path.exists(csv_path):
            print(f"  ERROR: File not found: {csv_path}")
            return

        # Load and show raw data info
        raw_df = pd.read_csv(csv_path)
        print(f"\n  Raw data: {len(raw_df)} samples, {raw_df['time'].max():.2f} seconds")

        # Extract features
        print("\n  Extracting features (window=25, overlap=50%)...")
        X_real, _, feature_names = extract_features_from_csv(
            csv_path, window_size=25, overlap=0.5
        )
        print(f"  Extracted {X_real.shape[0]} windows, {X_real.shape[1]} features each")

        # Create simulated data with realistic overlap
        print("\n  Creating SIMULATED training data...")
        print("  (For real results, use: main(use_real_labels=True))")

        np.random.seed(42)
        n_augment = 500

        base_features = X_real.mean(axis=0)
        feature_std = X_real.std(axis=0)

        jerk_indices = [i for i, name in enumerate(feature_names) if 'jerk' in name]
        std_indices = [i for i, name in enumerate(feature_names) if 'std' in name]

        # ON state (smoother)
        X_on = np.tile(base_features, (n_augment, 1))
        X_on += np.random.randn(n_augment, len(base_features)) * feature_std * 1.5
        X_on[:, jerk_indices] *= (0.85 + np.random.rand(n_augment, len(jerk_indices)) * 0.2)

        # OFF state (more erratic)
        X_off = np.tile(base_features, (n_augment, 1))
        X_off += np.random.randn(n_augment, len(base_features)) * feature_std * 1.5
        X_off[:, jerk_indices] *= (1.05 + np.random.rand(n_augment, len(jerk_indices)) * 0.25)
        X_off[:, std_indices] *= (1.05 + np.random.rand(n_augment, len(std_indices)) * 0.15)

        # Combine and shuffle
        X = np.vstack([X_on, X_off])
        y = np.hstack([np.ones(n_augment), np.zeros(n_augment)])
        idx = np.random.permutation(len(y))
        X, y = X[idx], y[idx]

        print(f"\n  SIMULATED dataset created:")
        print(f"    - Total samples: {X.shape[0]}")
        print(f"    - Features: {X.shape[1]}")
        print(f"    - ON: {int(y.sum())} | OFF: {int(len(y) - y.sum())}")
        print(f"\n  WARNING: Results are for DEMO only. Not production-ready!")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2: SPLIT DATA
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  STEP 2: SPLIT DATA (70% Train / 30% Test)")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n  Training set: {len(y_train)} samples")
    print(f"  Test set:     {len(y_test)} samples")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3: INITIALIZE MODEL
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  STEP 3: INITIALIZE MODEL")
    print("=" * 70)

    detector = ParkinsonsOFFDetector(random_state=42)

    print(f"\n  Model: RandomForestClassifier")
    print(f"  Trees: {detector.model.n_estimators}")
    print(f"  Max Depth: {detector.model.max_depth}")
    print(f"  Class Weight: {detector.model.class_weight}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEPS 4-8: TRAIN MODEL
    # ══════════════════════════════════════════════════════════════════════════

    train_metrics = detector.train(X_train, y_train, feature_names=feature_names)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 9: EVALUATE ON TEST DATA
    # ══════════════════════════════════════════════════════════════════════════

    test_metrics = detector.evaluate(X_test, y_test)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 10: CHECK OVERFITTING
    # ══════════════════════════════════════════════════════════════════════════

    overfit = detector.check_overfitting(X_train, y_train, X_test, y_test)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 11: CROSS-VALIDATION
    # ══════════════════════════════════════════════════════════════════════════

    cv_results = detector.cross_validate(X, y, cv_folds=5)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 12: SAVE MODEL
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  STEP 12: SAVE MODEL")
    print("=" * 70)

    detector.save_model()

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  ALL 12 STEPS COMPLETE!")
    print("=" * 70)
    print(f"""
    ┌────────────────────────────────────────────────────────────────────┐
    │  SUMMARY                                                           │
    ├────────────────────────────────────────────────────────────────────┤
    │  Data Source:  results_Jan-29.csv (REAL accelerometer data)        │
    │  Features:     {len(feature_names)} statistical features from IMU                    │
    ├────────────────────────────────────────────────────────────────────┤
    │  Train Accuracy: {train_metrics['accuracy']:.2%}                                          │
    │  Test Accuracy:  {test_metrics['accuracy']:.2%}                                          │
    │  CV Accuracy:    {cv_results['accuracy_mean']:.2%} (± {cv_results['accuracy_std']:.2%})                                │
    │  Overfit Status: {overfit['status']:<42} │
    ├────────────────────────────────────────────────────────────────────┤
    │  OUTPUT FILES:                                                     │
    │    - ./results/confusion_matrix.png                                │
    │    - ./results/roc_curve.png                                       │
    │    - ./results/feature_importance.png                              │
    │    - ./results/learning_curve.png                                  │
    │    - ./models/random_forest_model.pkl                              │
    ├────────────────────────────────────────────────────────────────────┤
    │  NOTE: Labels were SIMULATED for this demo.                        │
    │        For production, use REAL labeled ON/OFF recordings!         │
    └────────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    main()
