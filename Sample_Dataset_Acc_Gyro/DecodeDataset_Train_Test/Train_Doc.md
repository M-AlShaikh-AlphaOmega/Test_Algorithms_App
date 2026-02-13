# Random Forest Training Pipeline (Acc + Gyro)

Documentation for the accelerometer + gyroscope Random Forest training pipeline.

## Overview

Trains a Random Forest classifier to detect ON/OFF medication states using 6-axis IMU data (accelerometer + gyroscope) from wrist-worn sensors.

- **ON State (label=0)**: Medication working, good motor function
- **OFF State (label=1)**: Medication worn off, movement difficulties

## Data Organization

```
DecodeDataset_Train_Test/
├── ON/                    # CSV files recorded during ON state
│   ├── patient1_morning.csv
│   └── ...
├── OFF/                   # CSV files recorded during OFF state
│   ├── patient1_evening.csv
│   └── ...
└── outputs/               # Results (auto-created)
    ├── extracted_features.csv
    ├── evaluation_report.txt
    ├── feature_importance.png
    ├── confusion_matrix.png
    └── roc_curve.png
```

### Labeling Data

Copy decoded CSV files into the appropriate folder:
- **ON/**: Recordings when medication is effective (30-90 min after dose, MDS-UPDRS motor < 25)
- **OFF/**: Recordings when medication has worn off (3-5 hours after dose, MDS-UPDRS motor > 30)
- Exclude transitional states (MDS-UPDRS 25-30) from training

Balance the dataset: aim for 40-60% of each class.

## CSV Format

Each CSV must have these columns:

```csv
Sample,Ax,Ay,Az,Gx,Gy,Gz
0,0.245,-0.123,9.807,0.015,-0.008,0.003
...
```

Minimum length: 320 samples (10 seconds at 32 Hz).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fs` | 32 Hz | Sampling frequency |
| `window_size_sec` | 10s | Analysis window length |
| `cutoff` | 15 Hz | Low-pass filter cutoff |
| `order` | 4 | Butterworth filter order |
| `n_estimators` | 100 | Number of trees |
| `max_depth` | 7 | Maximum tree depth |
| `min_samples_leaf` | 2 | Minimum samples per leaf |
| `max_features` | sqrt | Features per split |
| `test_size` | 0.3 | Test split ratio |
| `n_splits` | 5 | Cross-validation folds |
| `top_k_features` | 15 | Features to select |

## Feature Extraction

### Per-axis Features (9 x 6 axes = 54)

For each axis (Ax, Ay, Az, Gx, Gy, Gz): mean, std, min, max, range, RMS, skewness, kurtosis, IQR.

### Gait Features (4)

| Feature | Description |
|---------|-------------|
| `steps` | Step count from peak detection on Ay |
| `step_time` | Average time between steps |
| `cadence` | Steps per minute |
| `arm_swing` | Gyroscope magnitude range |

**Total**: 58 features before selection, reduced to top 15.

## How to Run

```bash
# Default paths
cd DecodeDataset_Train_Test
python ../../RandomForest_Train.py

# Custom paths
python RandomForest_Train.py \
    --data_on path/to/ON \
    --data_off path/to/OFF \
    --out path/to/outputs

# Fast mode (skip plots)
python RandomForest_Train.py --fast
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_on` | `...Train_Test/ON` | ON state CSV folder |
| `--data_off` | `...Train_Test/OFF` | OFF state CSV folder |
| `--out` | `...Train_Test/outputs` | Output directory |
| `--fs` | 32 | Sampling frequency |
| `--window_sec` | 10 | Window size (seconds) |
| `--top_k` | 15 | Number of features |
| `--fast` | False | Skip plot generation |

## Output Files

| File | Description |
|------|-------------|
| `extracted_features.csv` | All features per window with labels |
| `evaluation_report.txt` | Accuracy, precision, recall, F1, AUC |
| `feature_importance.png` | Top features ranked by importance |
| `confusion_matrix.png` | Prediction breakdown |
| `roc_curve.png` | ROC curve with AUC |

## Expected Performance

With sufficient labeled data:

| Metric | Target |
|--------|--------|
| Accuracy | > 93% |
| Precision | > 90% |
| Recall | > 90% |
| AUC | > 0.95 |

## Troubleshooting

| Issue | Likely Cause | Fix |
|-------|-------------|-----|
| "No valid data found" | Empty ON/OFF folders or wrong CSV format | Check paths and column names |
| Low accuracy (<85%) | Mislabeled data or class imbalance | Review labels, balance dataset |
| High train, low test accuracy | Overfitting | Reduce max_depth, add more data |
| Large CV variance between folds | Small dataset or patient-specific patterns | More patients, split by patient |
