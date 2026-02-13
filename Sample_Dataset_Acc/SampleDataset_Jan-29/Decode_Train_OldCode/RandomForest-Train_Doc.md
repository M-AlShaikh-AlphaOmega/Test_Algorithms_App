# Random Forest Training Pipeline (Old Code)

Documentation for the accelerometer-only Random Forest training pipeline in `RandomForest-Train.py`.

**Note**: This is the older acc-only pipeline. The current pipeline is in `Sample_Dataset_Acc_Gyro/DecodeDataset_Train_Test/`.

## Overview

Trains a Random Forest classifier to detect ON/OFF medication states in Parkinson's patients using accelerometer data.

- **ON (label=1)**: Medication effective, good motor function
- **OFF (label=0)**: Medication worn off, reduced motor function

Based on Aich et al. (2020), DOI: 10.3390/diagnostics10060421

## Class: ParkinsonsOFFDetector

### Key Methods

| Method | Description |
|--------|-------------|
| `train(X_train, y_train)` | Fit model and scaler on training data |
| `evaluate(X_test, y_test)` | Evaluate on test data, generate plots |
| `cross_validate(X, y, cv_folds=5)` | K-fold cross-validation |
| `predict(X)` | Return (predictions, probabilities) |
| `save_model(filepath)` | Save model, scaler, feature names to .pkl |
| `load_model(filepath)` | Load saved model |
| `check_overfitting(...)` | Compare train vs test accuracy, plot learning curve |

### Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `n_estimators` | 500 | Number of trees |
| `max_depth` | 8 | Maximum tree depth |
| `min_samples_split` | 8 | Minimum samples to split |
| `min_samples_leaf` | 10 | Minimum samples per leaf |
| `max_features` | `sqrt` | Features considered per split |
| `class_weight` | `balanced` | Auto-adjust for class imbalance |

## Usage

### Training

```python
from RandomForest_Train import ParkinsonsOFFDetector
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('extracted_features.csv')
X = df.drop('label', axis=1).values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

detector = ParkinsonsOFFDetector(random_state=42)
detector.train(X_train, y_train, feature_names=df.drop('label', axis=1).columns.tolist())
detector.evaluate(X_test, y_test, output_dir='../results')
detector.save_model('../models/parkinsons_detector.pkl')
```

### Prediction

```python
detector = ParkinsonsOFFDetector()
detector.load_model('../models/parkinsons_detector.pkl')

predictions, probabilities = detector.predict(new_features)
for pred, prob in zip(predictions, probabilities):
    print(f"State: {'ON' if pred == 1 else 'OFF'}, Confidence: {prob:.2%}")
```

## Output Files

| File | Description |
|------|-------------|
| `random_forest_model.pkl` | Trained model + scaler + feature names |
| `confusion_matrix.png` | Prediction accuracy breakdown |
| `roc_curve.png` | ROC curve with AUC score |
| `feature_importance.png` | Top 20 features by importance |
| `learning_curve.png` | Train vs validation accuracy |

## Model Bundle (.pkl)

```python
{
    'model': RandomForestClassifier,
    'scaler': StandardScaler,
    'feature_names': list,
    'feature_importance': dict,
    'timestamp': str
}
```

## Overfitting Guide

| Gap (Train - Test) | Status | Action |
|--------------------|--------|--------|
| < 2% | Good fit | None needed |
| 2-5% | Mild overfitting | Acceptable |
| 5-10% | Moderate overfitting | Regularize or add data |
| > 10% | Severe overfitting | Reduce complexity |

## References

1. Aich, S., et al. (2020). "A Supervised Machine Learning Approach Using Different Feature Selection Techniques on Voice Datasets for Prediction of Parkinson's Disease." Diagnostics, 10(6), 421.
